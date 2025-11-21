"""Fully optimized turn-based move operations using centralized common modules."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, TYPE_CHECKING
import numpy as np
from numba import njit, prange

# Import consolidated modules (single source of truth)
from game3d.board.board import Board
from game3d.movement.movepiece import Move
from game3d.movement.generator import generate_legal_moves, generate_legal_moves_for_piece
from game3d.common.validation import validate_move
from game3d.common.shared_types import (
    Color, PieceType, Result,  # ✅ Added PieceType import
    COORD_DTYPE, INDEX_DTYPE, BOOL_DTYPE, COLOR_DTYPE, PIECE_TYPE_DTYPE,
    SIZE, MOVE_STEPS_MIN, MOVE_STEPS_MAX,
    get_empty_coord_batch, get_empty_bool_array,
    MOVE_DTYPE as MOVE_DTYPE
)
from game3d.common.validation import (
    validate_coord, validate_coords_batch, validate_coords_bounds, ValidationError
)
from game3d.common.coord_utils import in_bounds_vectorized, ensure_coords, get_neighbors_vectorized
from game3d.common.move_utils import (
    create_move_array_from_objects_vectorized,
    extract_from_coords_vectorized,
    extract_to_coords_vectorized
)
from game3d.common.performance_utils import _safe_increment_counter
from game3d.common.debug_utils import UndoSnapshot
from game3d.cache.manager import OptimizedCacheManager
from game3d.movement.movepiece import MOVE_FLAGS
# from game3d.game.moveeffects import apply_passive_mechanics

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

import logging
logger = logging.getLogger(__name__)

# =============================================================================
# MOVE PATTERN GENERATION (Uses shared_types constants)
# =============================================================================
def _get_adjacent_squares(coord: np.ndarray) -> np.ndarray:
    """Get adjacent squares to a coordinate for capture effect updates."""
    # Ensure coordinate is in batch format (1, 3) for vectorized operations
    coord_batch = coord.reshape(1, 3)
    return get_neighbors_vectorized(coord_batch)
# =============================================================================
# OPTIMIZED MOVE GENERATION (Uses centralized validation)
# =============================================================================

@njit(cache=True, fastmath=True, parallel=True)
def _generate_moves_vectorized(from_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate all possible moves from positions using vectorized operations."""
    if from_coords.size == 0:
        return get_empty_coord_batch(), get_empty_bool_array()

    # Ensure proper shape
    from_coords = np.asarray(from_coords, dtype=COORD_DTYPE)
    if from_coords.ndim == 1:
        from_coords = from_coords.reshape(1, 3) if from_coords.size == 3 else from_coords.reshape(-1, 3)

    # Handle edge cases
    if from_coords.ndim != 2 or from_coords.shape[1] != 3:
        return get_empty_coord_batch(), get_empty_bool_array()

    # Broadcast patterns to all from_coords
    destinations = from_coords[:, np.newaxis, :] + _MOVE_PATTERNS[np.newaxis, :, :]

    # Valid destinations within bounds
    valid_mask = validate_coords_bounds(destinations.reshape(-1, 3))
    valid_destinations = destinations.reshape(-1, 3)[valid_mask]

    # Capture flags (simple - all moves are potential captures)
    is_capture = get_empty_bool_array(valid_mask.sum())

    return valid_destinations, is_capture

# =============================================================================
# MAIN MOVE FUNCTIONS (Optimized with centralized utilities)
# =============================================================================
def legal_moves(game_state: 'GameState') -> np.ndarray:
    """Return legal moves as structured NumPy array using centralized cache."""
    logger = logging.getLogger(__name__)

    if not hasattr(game_state, '_metrics'):
        game_state._metrics = PerformanceMetrics()

    _safe_increment_counter(game_state._metrics, 'legal_moves_calls')

    cache_key = game_state.cache_manager.get_move_cache_key(game_state.color)
    # logger.debug(f"Legal moves called for color {game_state.color}, cache_key={cache_key}")

    # Check occupancy cache state
    occ_cache = game_state.cache_manager.occupancy_cache
    coords, types, colors = occ_cache.get_all_occupied_vectorized()
    # logger.info(f"Occupancy cache: {len(coords)} pieces found")

    # Fast cache path
    cache_manager = game_state.cache_manager
    if (game_state._legal_moves_cache is not None and
        game_state._legal_moves_cache_key == cache_key and
        hasattr(cache_manager.move_cache, '_board_generation') and
        cache_manager.move_cache._board_generation == getattr(game_state.board, 'generation', 0)):
        # logger.debug(f"Cache hit: returning {len(game_state._legal_moves_cache)} cached moves")
        return game_state._legal_moves_cache

    # logger.info("Cache miss - regenerating moves")

    # Generate moves using centralized generator
    raw = generate_legal_moves(game_state)
    # logger.info(f"Move generator returned {type(raw)} with size {getattr(raw, 'size', 'N/A')}")

    if isinstance(raw, np.ndarray):
        structured_moves = raw
    else:
        logger.error(f"🚨 CONTRACT VIOLATION: Generator returned {type(raw)} instead of ndarray")
        structured_moves = np.empty(0, dtype=MOVE_DTYPE)

    # logger.info(f"Final structured_moves size: {structured_moves.size}")

    if structured_moves.size > 0:
        game_state._legal_moves_cache = structured_moves
        game_state._legal_moves_cache_key = cache_key
        cache_manager.move_cache._board_generation = getattr(game_state.board, 'generation', 0)
        # logger.info(f"Cached {structured_moves.size} moves")
    else:
        logger.warning("⚠️ No legal moves generated - game loop will exit")

    return structured_moves

def legal_moves_for_piece(game_state: 'GameState', coord: np.ndarray) -> np.ndarray:
    """Get legal moves for piece at coordinate using centralized validation."""
    coord_arr = ensure_coords(coord)
    if coord_arr.size == 0:
        return np.empty(0, dtype=MOVE_DTYPE)

    raw = generate_legal_moves_for_piece(game_state, coord_arr[0])

    if isinstance(raw, np.ndarray):
        return raw

    return np.empty(0, dtype=MOVE_DTYPE)

# =============================================================================
# MOVE EXECUTION (Optimized with centralized utilities)
# =============================================================================
def make_move(game_state: 'GameState', mv: np.ndarray) -> 'GameState':
    """Execute move with proper incremental update pipeline."""
    from game3d.game.gamestate import GameState

    # 1. VALIDATE
    mv_obj = Move(mv[:3], mv[3:])
    if not validate_move(game_state, mv_obj):
        raise ValueError(f"Invalid move: {mv}")

    _safe_increment_counter(game_state._metrics, 'make_move_calls')

    cache_manager = game_state.cache_manager
    board = game_state.board

    # 2. EXTRACT PIECE DATA (BEFORE modification)
    from_piece = cache_manager.occupancy_cache.get(mv[:3])
    captured_piece = cache_manager.occupancy_cache.get(mv[3:])

    if from_piece is None:
        raise ValueError(f"No piece at source coordinate {mv[:3]}")

    # 🔥 CRITICAL FIX: Determine if this move should reset the 50-move clock
    is_capture = captured_piece is not None
    is_pawn_move = (from_piece["piece_type"] == PieceType.PAWN.value)  # ✅ PAWN check
    # Reset clock on captures or pawn moves, else increment
    new_halfmove_clock = 0 if (is_capture or is_pawn_move) else game_state.halfmove_clock + 1

    # 3. UPDATE BOARD IN-PLACE (O(1))
    changed_coords = np.array([mv[:3], mv[3:]], dtype=COORD_DTYPE)
    piece_types = np.array([0, from_piece["piece_type"]], dtype=np.int32)
    colors = np.array([0, from_piece["color"]], dtype=COLOR_DTYPE)
    
    # Create pieces_data for cache updates (shape: N x 2, columns: [piece_type, color])
    pieces_data = np.column_stack([piece_types, colors])

    board.batch_set_pieces_at(changed_coords, piece_types, colors)

    affected_squares = changed_coords.copy()
    if captured_piece:
        # Add adjacent squares for capture effects
        affected_squares = np.vstack([affected_squares, _get_adjacent_squares(mv[3:])])

    # 5. PIPELINE EXECUTION (in correct order)

    # 5a. OCCUPANCY CACHE FIRST (must be updated before other caches)
    cache_manager.occupancy_cache.batch_set_positions(changed_coords, pieces_data)

    # 5b. PARALLEL CACHE UPDATES (using manager's parallel executor)
    executor = cache_manager.get_parallel_executor()

    # Submit parallel tasks
    zobrist_future = executor.submit(
        cache_manager.update_zobrist_after_move,
        game_state._zkey, mv, from_piece, captured_piece
    )

    effects_future = executor.submit(
        cache_manager._notify_all_effect_caches,
        changed_coords, pieces_data
    )

    # Wait for completion
    new_zkey = zobrist_future.result()
    effects_future.result()

    # 5c. MARK AFFECTED PIECES FOR MOVE REGENERATION (BOTH COLORS)
    cache_manager.dependency_graph.notify_update('move_applied')
    
    # ✅ CRITICAL FIX: Invalidate affected piece moves for BOTH colors
    # The current player's moves need updating (they moved a piece)
    cache_manager._invalidate_affected_piece_moves(affected_squares, game_state.color, game_state)
    
    # The opponent's moves also need updating (board state changed, may affect their legal moves)
    opponent_color = Color.WHITE if game_state.color == Color.BLACK else Color.BLACK
    cache_manager._invalidate_affected_piece_moves(affected_squares, opponent_color, game_state)

    # 6. CREATE NEW GAME STATE (before passive mechanics)
    move_record = np.array([(
        mv[0], mv[1], mv[2], mv[3], mv[4], mv[5],
        captured_piece is not None,
        0  # flags
    )], dtype=MOVE_DTYPE)

    new_state = GameState(
        board=board,
        color=game_state.color.opposite(),
        cache_manager=cache_manager,
        history=np.append(game_state.history, move_record),
        halfmove_clock=new_halfmove_clock,  # ✅ Use corrected clock
        turn_number=game_state.turn_number + 1,
    )
    new_state._zkey = new_zkey
    
    # 8. APPLY PASSIVE MECHANICS (Freeze, Blackhole, Whitehole, Trailblazer)
    # new_state = apply_passive_mechanics(new_state, mv)
    
    # 9. INCREMENTAL MOVE CACHE UPDATE FOR BOTH COLORS (AFTER passive mechanics)
    # CRITICAL: This must happen AFTER passive mechanics to ensure the cache generation
    # matches the final board generation (after blackhole/whitehole may have modified it)
    from game3d.movement.generator import generate_legal_moves
    
    # Update current player's move cache (this is now the opponent since we switched turns)
    current_color = new_state.color
    current_player_moves = generate_legal_moves(new_state)
    cache_manager.move_cache.store_moves(current_color, current_player_moves)
    
    # Update opponent's move cache (the one who just moved)  
    opponent_color = Color.WHITE if current_color == Color.BLACK else Color.BLACK
    # Temporarily switch state color to generate opponent's moves
    original_color = new_state.color
    new_state.color = opponent_color
    opponent_moves = generate_legal_moves(new_state)
    cache_manager.move_cache.store_moves(opponent_color, opponent_moves)
    # Restore original color
    new_state.color = original_color

    # Logging every 100 moves
    move_number = game_state.history.size + 1
    if move_number % 100 == 0:
        try:
            piece_name = PieceType(from_piece["piece_type"]).name
            color_name = Color(from_piece["color"]).name
            is_capture_str = "Capture" if captured_piece is not None else "Move"
            print(f"Move {move_number}: {color_name} {piece_name} from {mv[:3]} to {mv[3:]} ({is_capture_str})")
        except Exception as e:
            logger.warning(f"Failed to log move {move_number}: {e}")

    return new_state

def undo_move(game_state: 'GameState') -> 'GameState':
    """Undo last move using centralized utilities."""
    if game_state.history.size == 0:
        raise ValueError("No move history to undo")

    _safe_increment_counter(game_state._metrics, 'undo_move_calls')
    return _undo_move_fast(game_state)

def _undo_move_fast(game_state: 'GameState') -> 'GameState':
    """Fast undo implementation using centralized utilities."""
    from game3d.game.gamestate import GameState

    last_mv = game_state.history[-1].view(np.ndarray).flatten()  # Extract as array

    # Restore board
    new_board_array = game_state.board.array().copy(order='C')
    new_board = Board(new_board_array)

    # Handle cache manager
    cache_manager = game_state.cache_manager
    if cache_manager is not None:
        cache_manager.board = new_board

        # Undo move
        if hasattr(cache_manager, 'undo_move'):
            cache_manager.undo_move(last_mv, game_state.color.opposite())
        elif hasattr(cache_manager, 'rebuild'):
            cache_manager.rebuild(new_board, game_state.color.opposite())

        # Invalidate move cache after undo
        cache_manager._invalidate_affected_piece_moves(affected_squares, game_state.color)

    # Create previous state
    prev_state = GameState(
        board=new_board,
        color=game_state.color.opposite(),
        cache_manager=game_state.cache_manager,
        history=game_state.history[:-1],
        halfmove_clock=game_state.halfmove_clock - 1,
        turn_number=game_state.turn_number - 1,
    )
    prev_state._clear_caches()

    # Update position counts
    prev_state._update_position_counts(game_state.zkey, -1)

    return prev_state

def _compute_undo_info(game_state: 'GameState', mv: np.ndarray, moving_piece: Any, captured_piece: Optional[Any]) -> UndoSnapshot:
    """Create undo snapshot using centralized debug utilities."""
    board_array = game_state.board.array()
    return UndoSnapshot(
        original_board_array=board_array.copy(order='C'),
        original_halfmove_clock=game_state.halfmove_clock,
        original_turn_number=game_state.turn_number,
        original_zkey=game_state.zkey,
        moving_piece=moving_piece,
        captured_piece=captured_piece,
        original_aura_state=None,
        original_trailblaze_state=None,
        original_geomancy_state=None,
    )

# =============================================================================
# PERFORMANCE METRICS (Uses centralized performance utilities)
# =============================================================================

@dataclass(slots=True)
class PerformanceMetrics:
    """Performance tracking for game state operations."""
    legal_moves_calls: int = 0
    make_move_calls: int = 0
    undo_move_calls: int = 0

    def reset(self):
        self.legal_moves_calls = 0
        self.make_move_calls = 0
        self.undo_move_calls = 0

    def get_stats(self) -> Dict[str, int]:
        return {
            'legal_moves_calls': self.legal_moves_calls,
            'make_move_calls': self.make_move_calls,
            'undo_move_calls': self.undo_move_calls
        }

# =============================================================================
# CACHE VALIDATION (Uses diagnostics patterns)
# =============================================================================

def validate_cache_integrity(game_state: 'GameState') -> None:
    """Validate cache manager consistency using centralized patterns."""
    if not hasattr(game_state, 'cache_manager'):
        raise RuntimeError("GameState missing cache_manager")

    if game_state.cache_manager is None:
        raise RuntimeError("Cache manager is None")

    if not hasattr(game_state.cache_manager, 'board'):
        raise RuntimeError("Cache manager missing board reference")

    # Additional validation can be added here

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    'legal_moves',
    'legal_moves_for_piece',
    'make_move',
    'undo_move',
    'PerformanceMetrics',
    'validate_cache_integrity'
]
