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
from game3d.common.common import validate_moves, UndoSnapshot

from .move_utils import (
    apply_hole_effects,
    apply_bomb_effects,
    apply_trailblaze_effect,
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
            return validate_legal_moves(game_state, game_state._legal_moves_cache, game_state.color)

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
    if game_state.cache.piece_cache.get(mv.from_coord) is None:
        raise ValueError(f"make_move: no piece at {mv.from_coord}")

    with track_operation_time(game_state._metrics, 'total_make_move_time'):
        game_state._metrics.make_move_calls += 1

        # --- sanity checks --------------------------------------------------
        # CORRECTED: Access through manager
        moving_piece = game_state.cache.piece_cache.get(mv.from_coord)
        if moving_piece is None or moving_piece.color != game_state.color:
            raise RuntimeError(f"Cache/board desync or wrong side to move: {mv}")

        # --- clone board ----------------------------------------------------
        new_board = Board(game_state.board.tensor().clone())
        new_board.cache_manager = game_state.cache  # shared cache

        # CORRECTED: Access through manager
        captured_piece = game_state.cache.piece_cache.get(mv.to_coord)
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
            halfmove_clock=0 if mv.is_capture else game_state.halfmove_clock + 1,
            game_mode=game_state.game_mode,
            turn_number=game_state.turn_number + 1,
        )

        # --- side-effects ---------------------------------------------------
        # 1. Apply freeze effects - ANY move triggers freeze re-emission from all friendly freezers
        # CORRECTED: Access effects through manager
        new_state.cache.effects.apply_freeze_effects(game_state.color, new_board)

        # 2. Black hole pulls - CORRECTED: Use manager methods
        new_state.cache.effects.apply_blackhole_pulls(game_state.color, new_board)

        # 3. White hole pushes - CORRECTED: Use manager methods
        new_state.cache.effects.apply_whitehole_pushes(game_state.color, new_board)

        # 4. Other effects (bombs, trailblaze, etc.)
        removed_pieces: list = []
        moved_pieces: list = []

        apply_bomb_effects(
            board=new_state.board,
            cache=new_state.cache,  # pass manager
            mv=mv,
            moving_piece=moving_piece,
            captured_piece=captured_piece,
            removed_pieces=removed_pieces,
            is_self_detonate=getattr(mv, 'is_self_detonate', False)
        )

        apply_trailblaze_effect(
            board=new_state.board,
            cache=new_state.cache,  # pass manager
            mv=mv,
            color=game_state.color,
            removed_pieces=removed_pieces
        )

        apply_hole_effects(
            board=new_state.board,
            cache=new_state.cache,  # pass manager
            color=game_state.color,
            moved_pieces=moved_pieces
        )

        # --- track affected squares ----------------------------------------
        affected_squares = {mv.from_coord, mv.to_coord}
        if captured_piece:
            affected_squares.add(mv.to_coord)
        affected_squares.update(reconstruct_trailblazer_path(mv.from_coord, mv.to_coord))
        affected_squares.update(extract_enemy_slid_path(mv))
        for from_sq, to_sq, _ in moved_pieces:
            affected_squares.add(from_sq)
            affected_squares.add(to_sq)
        for sq, _ in removed_pieces:
            affected_squares.add(sq)

        # Incremental update for move cache through manager
        new_state.cache.move.invalidate_squares(affected_squares)
        new_state.cache.move.invalidate_attacked_squares(new_state.color)
        new_state.cache.move.invalidate_attacked_squares(new_state.color.opposite())

        # --- store minimal undo footprint -----------------------------------
        undo_info['removed_pieces'] = removed_pieces
        undo_info['moved_pieces'] = moved_pieces
        new_state._undo_info = undo_info

        # --- final zobrist --------------------------------------------------
        new_state._zkey = compute_zobrist(new_board, new_state.color)
        new_state.cache.sync_zobrist(new_state._zkey)

        # --- NEW: Collect and apply batch occupancy updates ----------------
        occupancy_updates = [
            (mv.from_coord, None),
            (mv.to_coord, moving_piece)
        ]

        # Captured piece already handled in raw move
        for sq, piece in removed_pieces:
            occupancy_updates.append((sq, None))

        # Moved pieces from hole effects
        for from_sq, to_sq, piece in moved_pieces:
            occupancy_updates.append((from_sq, None))
            occupancy_updates.append((to_sq, piece))

        # Apply batch update to occupancy cache through manager
        new_state.cache.piece_cache.batch_set_positions(occupancy_updates)

        # CORRECTION: Invalidate move cache for affected squares
        affected = set(removed_pieces.keys()) if isinstance(removed_pieces, dict) else {sq for sq, _ in removed_pieces}
        for from_sq, to_sq, _ in moved_pieces:
            affected.update([from_sq, to_sq])
        new_state.cache.move.invalidate_squares(affected)
        new_state.cache.move.invalidate_attacked_squares(game_state.color)
        new_state.cache.move.invalidate_attacked_squares(game_state.color.opposite())

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
