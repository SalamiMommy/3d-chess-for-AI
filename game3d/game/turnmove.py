# game3d/game/turnmove.py - CLEANED
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
from game3d.cache.caches.zobrist import compute_zobrist
from game3d.movement.movepiece import MOVE_FLAGS

# Common imports
from game3d.common.move_utils import validate_moves, apply_special_effects, create_enriched_move
from game3d.common.cache_utils import validate_cache_integrity
from game3d.common.debug_utils import UndoSnapshot
from game3d.common.move_validation import validate_move_basic, validate_move_destination

if TYPE_CHECKING:
    from .gamestate import GameState

# ------------------------------------------------------------------
# VALIDATION (use common modules)
# ------------------------------------------------------------------
def validate_legal_moves(game_state: 'GameState', moves: List[Move], color: Color) -> List[Move]:
    """Paranoid validation of legal moves - remove any from empty squares."""
    return validate_moves(moves, game_state)

# ------------------------------------------------------------------
# MOVE GENERATION WITH ADVANCED CACHING
# ------------------------------------------------------------------
def legal_moves(game_state: 'GameState') -> List[Move]:
    """Optimized legal move generation with smart validation."""
    with track_operation_time(game_state._metrics, 'total_legal_moves_time'):
        game_state = game_state._with_metrics(
            legal_moves_calls=game_state._metrics.legal_moves_calls + 1
        )

        current_key = game_state.zkey

        # Fast path: return cached moves if available and generation matches
        if (game_state._legal_moves_cache is not None and
            game_state._legal_moves_cache_key == current_key and
            hasattr(game_state.cache_manager.move, '_gen') and
            game_state.cache_manager.move._gen == game_state.board.generation):

            return game_state._legal_moves_cache

        # CORRECTED: Check move cache through manager
        move_cache = game_state.cache_manager.move
        if (move_cache._legal_by_color[game_state.color] and
            not move_cache._needs_rebuild and
            move_cache._gen == game_state.board.generation):

            moves = move_cache._legal_by_color[game_state.color].copy()
        else:
            moves = generate_legal_moves(game_state)
            # Cache in move cache through manager
            move_cache._legal_by_color[game_state.color] = moves
            move_cache._rebuild_color_lists()
            move_cache._gen = game_state.board.generation  # Sync generation

        # ONLY validate on cache miss or generation mismatch
        if (game_state._legal_moves_cache is None or
            game_state._legal_moves_cache_key != current_key):

            moves = validate_legal_moves(game_state, moves, game_state.color)
            moves = [m for m in moves if m is not None]  # Filter None values

        # Cache result with generation info
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

    # Use common validation
    if not validate_move_basic(game_state, mv, game_state.color):
        raise ValueError(f"make_move: invalid move {mv}")

    with track_operation_time(game_state._metrics, 'total_make_move_time'):
        game_state._metrics.make_move_calls += 1

        # --- clone board but REUSE cache manager ----------------------------
        new_board = Board(game_state.board.tensor().clone())

        # KEY FIX: Update existing cache manager to use new board
        existing_cache_manager = game_state.cache_manager
        existing_cache_manager.board = new_board
        new_board.cache_manager = existing_cache_manager

        moving_piece = existing_cache_manager.occupancy.get(mv.from_coord)
        captured_piece = existing_cache_manager.occupancy.get(mv.to_coord)
        undo_info = _compute_undo_info(game_state, mv, moving_piece, captured_piece)

        # --- apply raw move using cache manager's incremental update -------
        if not existing_cache_manager.apply_move(mv, game_state.color):
            raise ValueError(f"Board refused move: {mv}")

        # --- build next state REUSING the same cache manager ----------------
        new_state = GameState(
            board=new_board,
            color=game_state.color.opposite(),
            cache_manager=existing_cache_manager,  # REUSE same manager
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

        if not validate_cache_integrity(new_state):
            print("WARNING: Cache desync detected after move")
            # Optionally force rebuild
            new_state.cache_manager.rebuild(new_state.board, new_state.color)

        return new_state

# ------------------------------------------------------------------
# UNDO MOVE IMPLEMENTATION
# ------------------------------------------------------------------
def undo_move(game_state: 'GameState') -> 'GameState':
    if not game_state.history:
        raise ValueError("Cannot undo – no move history")

    with track_operation_time(game_state._metrics, 'total_undo_move_time'):
        game_state._metrics.undo_move_calls += 1

        return _fast_undo(game_state)

def _fast_undo(game_state: 'GameState') -> 'GameState':
    """Revert last move using cached tensors + cache-manager undo."""
    from .gamestate import GameState

    undo_info = game_state._undo_info
    last_mv   = game_state.history[-1]

    # Restore board tensor
    new_board = Board(undo_info.original_board_tensor.clone())

    # Update cache manager with restored board
    game_state.cache_manager.board = new_board

    # Call undo_move with proper parameters
    success = game_state.cache_manager.undo_move(last_mv, game_state.color.opposite())
    if not success:
        # Fallback: rebuild cache
        game_state.cache_manager.rebuild(new_board, game_state.color.opposite())

    # Rebuild previous state
    prev_state = GameState(
        board=new_board,
        color=game_state.color.opposite(),
        cache_manager=game_state.cache_manager,  # Reuse same manager
        history=game_state.history[:-1],
        halfmove_clock=undo_info.original_halfmove_clock,
        game_mode=game_state.game_mode,
        turn_number=undo_info.original_turn_number,
    )
    prev_state._zkey = undo_info.original_zkey
    prev_state._clear_caches()
    return prev_state

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
        # ADD these for cache manager undo
        original_aura_state=getattr(game_state.cache_manager.aura_cache, '_state', None),
        original_trailblaze_state=getattr(game_state.cache_manager.trailblaze_cache, '_state', None),
        original_geomancy_state=getattr(game_state.cache_manager.geomancy_cache, '_state', None),
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
