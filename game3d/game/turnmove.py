# game3d/game/turnmove.py - OPTIMIZED FOR NUMPY
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, TYPE_CHECKING
import numpy as np
from numba import jit, njit
import time

from game3d.board.board import Board
from game3d.movement.movepiece import Move
from game3d.movement.generator import generate_legal_moves, generate_legal_moves_for_piece
from game3d.common.enums import Color, PieceType
from game3d.pieces.piece import Piece

from game3d.attacks.check import _any_priest_alive
from .performance import track_operation_time
from game3d.cache.caches.zobrist import compute_zobrist
from game3d.movement.movepiece import MOVE_FLAGS

# Common imports
from game3d.common.move_utils import apply_special_effects, create_enriched_move
from game3d.common.validation import validate_moves, validate_cache_integrity, validate_move_basic, validate_move_destination
from game3d.common.debug_utils import UndoSnapshot

if TYPE_CHECKING:
    from .gamestate import GameState

# Constants for 9x9x9 board with 600 pieces
BOARD_SIZE = (9, 9, 9)
MAX_PIECES = 462
PIECE_TYPES = 40

# Optimized numpy operations using numba for speed
@njit
def create_board_tensor_copy(board_array: np.ndarray) -> np.ndarray:
    """Create a fast copy of board tensor using numpy."""
    return board_array.copy()

@njit
def validate_coordinates_numba(coords: np.ndarray) -> bool:
    """Fast coordinate validation using numba."""
    x, y, z = coords
    return (0 <= x < 9 and 0 <= y < 9 and 0 <= z < 9)

@njit
def get_piece_at_numba(board_array: np.ndarray, coord: Tuple[int, int, int]) -> int:
    """Fast piece lookup using numpy indexing."""
    x, y, z = coord
    return board_array[x, y, z]

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
    """Optimized legal move generation with smart validation using numpy."""
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

def legal_moves_for_piece(game_state: 'GameState', coord: Tuple[int, int, int]) -> List[Move]:
    """Get legal moves for a specific piece."""
    return generate_legal_moves_for_piece(game_state, coord)

# ------------------------------------------------------------------
# OPTIMIZED MOVE MAKING WITH NUMPY
# ------------------------------------------------------------------
def make_move(game_state: 'GameState', mv: Move) -> 'GameState':
    from .gamestate import GameState

    if not validate_move_basic(game_state, mv, game_state.color):
        raise ValueError(f"make_move: invalid move {mv}")

    with track_operation_time(game_state._metrics, 'total_make_move_time'):
        game_state._metrics.make_move_calls += 1

        # Use numpy array instead of torch tensor
        board_array = game_state.board.array()  # Assuming Board now has array() method
        new_board_array = board_array.copy()  # Fast numpy copy

        # Create new board with numpy array
        new_board = Board(new_board_array)

        # CRITICAL: Reuse existing cache manager with new board
        cache_manager = game_state.cache_manager
        cache_manager.board = new_board
        new_board.cache_manager = cache_manager

        moving_piece = cache_manager.occupancy_cache.get(mv.from_coord)
        captured_piece = cache_manager.occupancy_cache.get(mv.to_coord)
        undo_info = _compute_undo_info(game_state, mv, moving_piece, captured_piece)

        # Apply move using cache manager's incremental update
        if not cache_manager.apply_move(mv, game_state.color):
            raise ValueError(f"Board refused move: {mv}")

        # Build next state REUSING the same cache manager
        new_state = GameState(
            board=new_board,
            color=game_state.color.opposite(),
            cache_manager=cache_manager,  # REUSE
            history=game_state.history + (mv,),
            halfmove_clock=game_state.halfmove_clock + 1,
            game_mode=game_state.game_mode,
            turn_number=game_state.turn_number + 1,
        )
        new_state._undo_info = undo_info

        # Update halfmove clock
        is_pawn = moving_piece.ptype == PieceType.PAWN
        is_capture = mv.is_capture or captured_piece is not None
        if is_pawn or is_capture:
            new_state.halfmove_clock = 0

        return new_state

# ------------------------------------------------------------------
# UNDO MOVE IMPLEMENTATION WITH NUMPY
# ------------------------------------------------------------------
def undo_move(game_state: 'GameState') -> 'GameState':
    if not game_state.history:
        raise ValueError("Cannot undo – no move history")

    with track_operation_time(game_state._metrics, 'total_undo_move_time'):
        game_state._metrics.undo_move_calls += 1

        return _fast_undo(game_state)

def _fast_undo(game_state: 'GameState') -> 'GameState':
    """Revert last move using cached numpy arrays + cache-manager undo."""
    from .gamestate import GameState

    undo_info = game_state._undo_info
    last_mv   = game_state.history[-1]

    # Restore board array using numpy copy
    new_board_array = undo_info.original_board_array.copy()
    new_board = Board(new_board_array)

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
    prev_state._clear_caches()
    return prev_state

def _compute_undo_info(game_state: 'GameState',
                       mv: Move,
                       moving_piece: Piece,
                       captured_piece: Optional[Piece]) -> UndoSnapshot:
    """Snapshot everything needed for a one-step undo using numpy."""
    # Use numpy array instead of torch tensor
    board_array = game_state.board.array()

    return UndoSnapshot(
        original_board_array=board_array.copy(),  # numpy copy instead of torch clone
        original_halfmove_clock=game_state.halfmove_clock,
        original_turn_number=game_state.turn_number,
        original_zkey=game_state.zkey,
        moving_piece=moving_piece,
        captured_piece=captured_piece,
        # ADD these for cache manager undo
        original_aura_state=getattr(game_state.cache_manager.aura_cache, '_state', None),
        original_trailblaze_state=getattr(game_state.cache_manager.trailblaze_cache, '_state', None),
        original_geomancy_state=getattr(game_state.cache_manager.geomancy_cache, '_state', None),
    )

# Optimized helper functions for 9x9x9 board
@njit
def is_valid_position(x: int, y: int, z: int) -> bool:
    """Fast position validation for 9x9x9 board."""
    return (0 <= x < 9) and (0 <= y < 9) and (0 <= z < 9)

@njit
def get_adjacent_positions(x: int, y: int, z: int) -> List[Tuple[int, int, int]]:
    """Get all valid adjacent positions in 3D space."""
    positions = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nx, ny, nz = x + dx, y + dy, z + dz
                if is_valid_position(nx, ny, nz):
                    positions.append((nx, ny, nz))
    return positions

def batch_validate_moves(moves: List[Move], board_array: np.ndarray) -> List[Move]:
    """Batch validate moves using numpy operations."""
    valid_moves = []

    for move in moves:
        from_coord = move.from_coord
        to_coord = move.to_coord

        # Fast coordinate validation
        if not (is_valid_position(*from_coord) and is_valid_position(*to_coord)):
            continue

        # Check if from position has a piece
        if board_array[from_coord] == 0:  # 0 represents empty
            continue

        valid_moves.append(move)

    return valid_moves

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

# Performance optimization: Precompute common coordinate patterns
def precompute_3d_patterns():
    """Precompute common 3D movement patterns for 9x9x9 board."""
    patterns = {}

    # Straight lines in 3D
    for axis in range(3):
        for direction in (-1, 1):
            pattern = []
            for i in range(1, 9):  # Maximum 8 steps in any direction
                offset = [0, 0, 0]
                offset[axis] = i * direction
                pattern.append(tuple(offset))
            patterns[f'straight_{axis}_{direction}'] = pattern

    # Diagonals in 3D
    for dx in (-1, 1):
        for dy in (-1, 1):
            for dz in (-1, 1):
                pattern = []
                for i in range(1, 9):
                    pattern.append((i * dx, i * dy, i * dz))
                patterns[f'diagonal_{dx}_{dy}_{dz}'] = pattern

    return patterns

# Precompute patterns at module load
_3D_PATTERNS = precompute_3d_patterns()
