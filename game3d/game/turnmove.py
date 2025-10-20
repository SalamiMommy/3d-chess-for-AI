# turnmove.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, TYPE_CHECKING

from game3d.board.board import Board
from game3d.movement.movepiece import Move, convert_legacy_move_args
from game3d.movement.generator import generate_legal_moves
from game3d.movement.pseudo_legal import generate_pseudo_legal_moves
from game3d.common.enums import Color, PieceType
from game3d.pieces.piece import Piece

from game3d.attacks.check import _any_priest_alive
from .performance import track_operation_time
from .zobrist import compute_zobrist
from game3d.movement.movepiece import MOVE_FLAGS
from game3d.common.move_utils import validate_moves
from game3d.common.debug_utils import UndoSnapshot

from .move_utils import (
    apply_hole_effects,
    apply_bomb_effects,
    apply_trailblaze_effect,
    apply_geomancy_effect,
    reconstruct_trailblazer_path,
    extract_enemy_slid_path,
)

if TYPE_CHECKING:
    from .gamestate import GameState

# ------------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------------
def validate_legal_moves(game_state: 'GameState', moves: List[Move], color: Color) -> List[Move]:
    """Paranoid validation of legal moves - remove any from empty squares."""
    return validate_moves(moves, game_state)

# ------------------------------------------------------------------
# MOVE GENERATION WITH ADVANCED CACHING
# ------------------------------------------------------------------

def legal_moves(game_state: 'GameState') -> List[Move]:
    """Fixed legal move generation with paranoid validation and cache check."""
    with track_operation_time(game_state._metrics, 'total_legal_moves_time'):
        game_state = game_state._with_metrics(
            legal_moves_calls=game_state._metrics.legal_moves_calls + 1
        )

        current_key = game_state.zkey
        if (game_state._legal_moves_cache is not None and
            game_state._legal_moves_cache_key == current_key):
            # CRITICAL: Validate cached moves before returning
            validated = validate_legal_moves(game_state, game_state._legal_moves_cache, game_state.color)
            # DEFENSIVE: Filter out None values
            return [m for m in validated if m is not None]

        # CORRECTED: Check move cache through manager
        if (game_state.cache.move._legal_by_color[game_state.color] and
            not game_state.cache.move._needs_rebuild):
            moves = game_state.cache.move._legal_by_color[game_state.color].copy()
        else:
            moves = generate_legal_moves(game_state)
            # Cache in move cache through manager
            game_state.cache.move._legal_by_color[game_state.color] = moves
            game_state.cache.move._rebuild_color_lists()

        # CRITICAL: Validate moves immediately after generation
        moves = validate_legal_moves(game_state, moves, game_state.color)

        # DEFENSIVE: Filter out any None values that slipped through
        moves = [m for m in moves if m is not None]

        # Cache result
        game_state._legal_moves_cache = moves
        game_state._legal_moves_cache_key = current_key

        return moves

def pseudo_legal_moves(game_state: 'GameState') -> List[Move]:
    return generate_pseudo_legal_moves(game_state)


# ------------------------------------------------------------------
# OPTIMIZED MOVE MAKING WITH INCREMENTAL UPDATES
# ------------------------------------------------------------------
def make_move(game_state: 'GameState', mv: Move) -> 'GameState':
    from .gamestate import GameState

    # CORRECTED: Access through manager
    if game_state.cache.occupancy.get(mv.from_coord) is None:
        raise ValueError(f"make_move: no piece at {mv.from_coord}")

    with track_operation_time(game_state._metrics, 'total_make_move_time'):
        game_state._metrics.make_move_calls += 1

        # --- sanity checks --------------------------------------------------
        # CORRECTED: Access through manager
        moving_piece = game_state.cache.occupancy.get(mv.from_coord)
        if moving_piece is None or moving_piece.color != game_state.color:
            raise RuntimeError(f"Cache/board desync or wrong side to move: {mv}")

        # --- clone board ----------------------------------------------------
        new_board = Board(game_state.board.tensor().clone())
        new_board.cache_manager = game_state.cache  # shared cache
        game_state.cache.board = new_board  # Update cache board reference

        # CORRECTED: Access through manager
        captured_piece = game_state.cache.occupancy.get(mv.to_coord)
        undo_info = _compute_undo_info(game_state, mv, moving_piece, captured_piece)

        # --- apply raw move -------------------------------------------------
        if not new_board.apply_move(mv):
            raise ValueError(f"Board refused move: {mv}")

        # --- build next state ----------------------------------------------
        new_state = GameState(
            board=new_board,
            color=game_state.color.opposite(),
            cache=game_state.cache,
            history=game_state.history + (mv,),
            halfmove_clock=game_state.halfmove_clock + 1,
            game_mode=game_state.game_mode,
            turn_number=game_state.turn_number + 1,
        )
        new_state._undo_info = undo_info

        # --- update halfmove clock -----------------------------------------
        is_pawn = moving_piece.ptype == PieceType.PAWN
        is_capture = mv.is_capture or captured_piece is not None
        if is_pawn or is_capture:
            new_state.halfmove_clock = 0

        # --- incremental occupancy sync ------------------------------------
        new_state.cache.occupancy.set_position(mv.from_coord, None)
        if mv.is_capture:
            new_state.cache.occupancy.set_position(mv.to_coord, None)
        new_state.cache.occupancy.set_position(mv.to_coord, moving_piece)

        # --- side effects --------------------------------------------------
        removed_pieces: List[Tuple[Tuple[int, int, int], Piece]] = []
        moved_pieces: List[Tuple[Tuple[int, int, int], Tuple[int, int, int], Piece]] = []

        # Fix: Pass new_state.board first, then new_state.cache
        is_self_detonate = apply_bomb_effects(new_state.board, new_state.cache, mv, moving_piece, captured_piece, removed_pieces, mv.flags & MOVE_FLAGS['SELF_DETONATE'])

        if moving_piece.ptype == PieceType.TRAILBLAZER:
            slid_squares = reconstruct_trailblazer_path(mv.from_coord, mv.to_coord, include_start=False, include_end=False)
            new_state.cache.mark_trail(mv.from_coord, slid_squares)

        # Fix: Pass new_state.board first, then new_state.cache
        apply_trailblaze_effect(new_state.board, new_state.cache, mv, game_state.color, removed_pieces)

        if moving_piece.ptype == PieceType.GEOMANCER:
            # Fix: Insert new_state.board as the first argument
            apply_geomancy_effect(new_state.board, new_state.cache, mv.to_coord, new_state.halfmove_clock)

        apply_hole_effects(
            new_state.board,                       # 1st → board (already correct)
            new_state.cache,                       # 2nd → cache (already correct)
            game_state.color.opposite(),           # 3rd → color
            moved_pieces,                          # 4th → moved_pieces
            is_self_detonate                       # 5th → _is_self_detonate
        )

        # --- sync occupancy for side effects -------------------------------
        for sq, _ in removed_pieces:
            new_state.cache.occupancy.set_position(sq, None)
        for from_sq, to_sq, piece in moved_pieces:
            new_state.cache.occupancy.set_position(from_sq, None)
            new_state.cache.occupancy.set_position(to_sq, piece)

        # --- apply hole effects --------------------------------------------
        new_state.cache.apply_blackhole_pulls(new_state.color.opposite())
        new_state.cache.apply_whitehole_pushes(new_state.color.opposite())

        # --- apply freeze effects ------------------------------------------
        new_state.cache.apply_freeze_effects(new_state.color.opposite())

        # --- incremental cache updates -------------------------------------
        affected_caches = new_state.cache.get_affected_caches(
            mv, game_state.color, moving_piece, new_state.cache.occupancy.get(mv.to_coord), captured_piece
        )
        new_state.cache.update_effect_caches(mv, game_state.color, affected_caches, new_state.halfmove_clock)

        # --- invalidate attacks cache --------------------------------------
        new_state.cache.attacks_cache.invalidate(game_state.color)
        new_state.cache.attacks_cache.invalidate(game_state.color.opposite())
        new_state.cache.move._lazy_revalidate()  # FIXED: Add this to match original

        # Sync generation if needed (to prevent rebuild triggers)
        if hasattr(new_board, 'generation') and hasattr(new_state.cache.occupancy, '_gen'):
            new_state.cache.occupancy._gen = new_board.generation
        if hasattr(new_board, 'generation') and hasattr(new_state.cache.move, '_gen'):
            new_state.cache.move._gen = new_board.generation

        # Sync Zobrist
        new_zkey = compute_zobrist(new_board, new_state.color)
        new_state._zkey = new_zkey
        new_state.cache.sync_zobrist(new_zkey)

        return new_state
# ------------------------------------------------------------------
# UNDO MOVE IMPLEMENTATION
# ------------------------------------------------------------------
def undo_move(game_state: 'GameState') -> 'GameState':
    if not game_state.history:
        raise ValueError("Cannot undo – no move history")

    with track_operation_time(game_state._metrics, 'total_undo_move_time'):
        game_state._metrics.undo_move_calls += 1

        # we always store undo_info since the last patch
        return _fast_undo(game_state)


def _fast_undo(game_state: 'GameState') -> 'GameState':
    """Revert last move using cached tensors + cache-manager undo."""
    from .gamestate import GameState

    undo_info = game_state._undo_info  # UndoSnapshot
    last_mv   = game_state.history[-1]

    # --- restore board tensor ------------------------------------------
    new_board = Board(undo_info.original_board_tensor.clone())

    # --- ask the cache manager to undo *its* part -----------------------
    game_state.cache.board = new_board
    game_state.cache.undo_move(last_mv, game_state.color.opposite(), 0, game_state._undo_info)
    game_state.cache.occupancy.rebuild(new_board)

    # --- rebuild previous state ----------------------------------------
    prev_state = GameState(
        board=new_board,
        color=game_state.color.opposite(),
        cache=game_state.cache,
        history=game_state.history[:-1],
        halfmove_clock=undo_info.original_halfmove_clock,
        game_mode=game_state.game_mode,
        turn_number=undo_info.original_turn_number,
    )
    prev_state._zkey = undo_info.original_zkey
    prev_state._clear_caches()
    return prev_state

def _full_undo(game_state: 'GameState') -> 'GameState':
    """Full undo by replaying moves from initial state."""
    raise NotImplementedError("Full undo requires storing initial state")

def _compute_undo_info(game_state: 'GameState',
                       mv: Move,
                       moving_piece: Piece,
                       captured_piece: Optional[Piece]) -> UndoSnapshot:
    """Snapshot everything needed for a one-step undo."""
    return UndoSnapshot(
        original_board_tensor=game_state.board.tensor().clone().cpu(),
        original_halfmove_clock=game_state.halfmove_clock,
        original_turn_number=game_state.turn_number,
        original_zkey=game_state._zkey,
        moving_piece=moving_piece,
        captured_piece=captured_piece,
    )

def _create_enriched_move(
    game_state: 'GameState',
    mv: Move,
    removed_pieces: List,
    moved_pieces: List,
    is_self_detonate: bool,
    undo_info: Dict[str, Any],
    captured_piece: Optional['Piece'] = None,
    is_pawn: bool = False,
    is_capture: bool = False
) -> 'EnrichedMove':
    """Create enriched move with all side effects and undo information."""
    is_capture_flag = mv.is_capture or (captured_piece is not None)
    core_move = convert_legacy_move_args(
        from_coord=mv.from_coord,
        to_coord=mv.to_coord,
        is_capture=is_capture_flag,
        captured_piece=captured_piece,
        is_promotion=mv.is_promotion,
        promotion_type=None,
        is_en_passant=False,
        is_castle=False,
    )
    return EnrichedMove(
        core_move=core_move,
        removed_pieces=removed_pieces,
        moved_pieces=moved_pieces,
        is_self_detonate=is_self_detonate,
        undo_info=undo_info,
        is_pawn_move=is_pawn,
        is_capture=is_capture
    )

@dataclass(slots=True)
class EnrichedMove:
    core_move: Move
    removed_pieces: List[Tuple[Tuple[int, int, int], Piece]]
    moved_pieces: List[Tuple[Tuple[int, int, int], Tuple[int, int, int], Piece]]
    is_self_detonate: bool
    undo_info: Dict[str, Any]
    is_pawn_move: bool = False
    is_capture: bool = False

    def __repr__(self):
        return (f"EnrichedMove(core_move={self.core_move}, "
                f"removed_pieces={len(self.removed_pieces)}, "
                f"moved_pieces={len(self.moved_pieces)}, "
                f"is_pawn_move={self.is_pawn_move}, "
                f"is_capture={self.is_capture})")
