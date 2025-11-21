# cache_consistency.py
"""Enhanced dependency graph for hybrid move generation pipeline."""

import numpy as np
import logging
from typing import TYPE_CHECKING, Optional, Tuple
from numba import njit, prange

from game3d.common.shared_types import (
    BOOL_DTYPE, INDEX_DTYPE, COORD_DTYPE, SIZE, MIN_COORD_VALUE, SIZE_MINUS_1,
    PieceType, N_PIECE_TYPES
)
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
    from game3d.cache.manager import OptimizedCacheManager

logger = logging.getLogger(__name__)

# Precompute slider lookup table
IS_SLIDER_LUT = np.zeros(N_PIECE_TYPES + 1, dtype=BOOL_DTYPE)
for pt in [PieceType.ROOK, PieceType.BISHOP, PieceType.QUEEN, 
           PieceType.EDGEROOK, PieceType.XYQUEEN, PieceType.XZQUEEN, PieceType.YZQUEEN,
           PieceType.VECTORSLIDER, PieceType.CONESLIDER, PieceType.REFLECTOR]:
    if pt.value <= N_PIECE_TYPES:
        IS_SLIDER_LUT[pt.value] = True

class DependencyGraph:
    """
    Tracks move generation dependencies for hybrid move generation pipeline.

    Responsibilities:
    1. Track which squares are affected by last move
    2. Compute which pieces need move regeneration based on affected squares
    3. Provide O(1) lookup for piece-level cache invalidation
    """

    def __init__(self):
        # Affected squares from last move (to be invalidated)
        self._affected_squares = np.empty((0, 3), dtype=COORD_DTYPE)

        # Precomputed piece reach for fast dependency calculation
        # Maps piece coordinate -> set of squares in its movement range
        self._piece_reach_cache = {}
        
        # OPTIMIZATION: LRU tracking for cache eviction
        self._reach_cache_max_size = 1000
        self._reach_cache_access_order = []  # Track access order for LRU

        # Cache generation for piece reach validity
        self._reach_cache_generation = -1

        # Which pieces are affected by current move
        self._affected_pieces = set()

    def mark_affected(self, coords: np.ndarray) -> None:
        """Mark squares as affected by last move."""
        if coords.size == 0:
            return

        # Merge with existing affected squares
        self._affected_squares = np.unique(
            np.vstack([self._affected_squares, coords.astype(COORD_DTYPE)]),
            axis=0
        )

        # Log affected squares for debugging
        logger.debug(f"Marked {len(coords)} squares as affected: {coords}")

    def get_affected_pieces(self, state: 'GameState', color: int) -> np.ndarray:
        """
        Get coordinates of pieces whose move sets are affected by current move.

        This is the CORE of the hybrid pipeline: determines which pieces need
        regeneration vs. which can reuse cached moves.
        """
        if self._affected_squares.size == 0:
            return np.empty((0, 3), dtype=COORD_DTYPE)

        # Check if board generation changed - if so, invalidate all reach caches
        current_gen = getattr(state.board, 'generation', 0)
        if current_gen != self._reach_cache_generation:
            self._piece_reach_cache.clear()
            self._reach_cache_access_order.clear()
            self._reach_cache_generation = current_gen
        
        # OPTIMIZATION: Evict old entries if cache is too large
        if len(self._piece_reach_cache) > self._reach_cache_max_size:
            # Keep most recent 50% of entries
            keep_count = self._reach_cache_max_size // 2
            keys_to_keep = self._reach_cache_access_order[-keep_count:]
            self._piece_reach_cache = {k: self._piece_reach_cache[k] for k in keys_to_keep if k in self._piece_reach_cache}
            self._reach_cache_access_order = keys_to_keep

        # Get all pieces of the color AND their types
        # We need types to know if they are sliders
        occ_cache = state.cache_manager.occupancy_cache
        
        # Optimized fetch: get coords and types in one go if possible
        # But get_positions only returns coords.
        # Let's use get_all_occupied_vectorized and filter by color
        all_coords, all_types, all_colors = occ_cache.get_all_occupied_vectorized()
        
        color_mask = (all_colors == color)
        piece_coords = all_coords[color_mask]
        piece_types = all_types[color_mask]

        if piece_coords.size == 0:
            return np.empty((0, 3), dtype=COORD_DTYPE)

        # Determine which pieces are sliders
        is_slider = IS_SLIDER_LUT[piece_types]

        # Determine which pieces are affected
        affected_mask = self._compute_affected_mask(
            piece_coords,
            self._affected_squares,
            is_slider
        )

        affected_pieces = piece_coords[affected_mask]

        # Log hybrid performance metrics
        total_pieces = len(piece_coords)
        affected_count = len(affected_pieces)
        # logger.debug(
        #     f"Hybrid: {affected_count}/{total_pieces} pieces need regeneration "
        #     f"({100*(1-affected_count/total_pieces):.1f}% cache reuse)"
        # )

        return affected_pieces

    def clear(self) -> None:
        """Clear affected squares after move is fully processed."""
        self._affected_squares = np.empty((0, 3), dtype=COORD_DTYPE)
        self._affected_pieces.clear()

    def notify_update(self, update_type: str) -> None:
        """Notify of board update - used for monitoring."""
        # logger.debug(f"Dependency graph notified: {update_type}")

    @staticmethod
    @njit(cache=True, fastmath=True, parallel=True)
    def _compute_affected_mask(
        piece_coords: np.ndarray,
        affected_squares: np.ndarray,
        is_slider: np.ndarray
    ) -> np.ndarray:
        """
        Vectorized computation: which pieces are affected by changed squares?

        A piece is affected if:
        1. It's on an affected square (was moved/captured)
        2. It's within range of an affected square
           - Sliders: Infinite range (check alignment) or large distance
           - Leapers: Small distance (<= 2)
        """
        n_pieces = piece_coords.shape[0]
        n_affected = affected_squares.shape[0]
        mask = np.zeros(n_pieces, dtype=BOOL_DTYPE)

        for i in prange(n_pieces):
            piece = piece_coords[i]
            slider = is_slider[i]
            
            # Threshold: 2 for leapers, SIZE for sliders
            threshold = SIZE if slider else 2

            # Check if this piece is on an affected square
            for j in range(n_affected):
                if np.array_equal(piece, affected_squares[j]):
                    mask[i] = True
                    break

            # If not directly affected, check proximity
            if not mask[i]:
                for j in range(n_affected):
                    # Chebyshev distance
                    dist = np.max(np.abs(piece - affected_squares[j]))
                    
                    if dist <= threshold:
                        # For sliders, we could be more precise by checking alignment
                        # But distance check is safe and fast
                        mask[i] = True
                        break

        return mask

def get_resource_monitor():
    """Simple stub for resource monitoring."""
    return {
        'active_resources': set(),
        'resource_counts': {}
    }


def validate_cache_consistency(game_state: 'GameState') -> bool:
    """Validate all cache references are synchronized."""
    cache_manager = game_state.cache_manager

    if cache_manager is None:
        logger.error("GameState has no cache manager")
        return False

    # Check board-cache_manager link
    if (hasattr(game_state.board, 'cache_manager') and
        game_state.board.cache_manager is not cache_manager):
        logger.error("Board cache manager mismatch")
        return False

    # Verify occupancy cache has pieces (critical for Zobrist)
    occ_coords, _, _ = cache_manager.occupancy_cache.get_all_occupied_vectorized()
    if len(occ_coords) == 0:
        logger.critical("Occupancy cache is EMPTY - board not initialized!")
        return False

    # Check Zobrist consistency
    try:
        current_hash = cache_manager._zkey
        computed_hash = cache_manager.zobrist_cache.compute_from_scratch(
            game_state.board, game_state.color
        )

        if current_hash != computed_hash:
            logger.error(f"Zobrist desync - current:{current_hash}, computed:{computed_hash}")
            return False
    except Exception as e:
        logger.error(f"Zobrist validation failed: {e}")
        return False

    logger.info("✅ Cache consistency validated")
    return True


def validate_hybrid_pipeline(state: 'GameState', move: np.ndarray) -> Tuple[bool, str]:
    """
    Validate that the hybrid pipeline works correctly.

    Tests:
    1. Partial regeneration produces same result as full regeneration
    2. Affected pieces are correctly identified
    3. Unaffected pieces reuse cache correctly
    """
    # Generate moves using hybrid (partial) method
    hybrid_moves = state.legal_moves

    # Force full regeneration for comparison
    state._legal_moves_cache = None
    state.cache_manager.move_cache.invalidate()
    full_moves = state.legal_moves

    # Compare results
    if len(hybrid_moves) != len(full_moves):
        return False, f"Move count mismatch: hybrid={len(hybrid_moves)}, full={len(full_moves)}"

    # Sort and compare move arrays
    hybrid_sorted = np.sort(hybrid_moves.view(f'U{hybrid_moves.dtype.itemsize}'))
    full_sorted = np.sort(full_moves.view(f'U{full_moves.dtype.itemsize}'))

    if not np.array_equal(hybrid_sorted, full_sorted):
        return False, "Move sets are not identical between hybrid and full regeneration"

    return True, "Hybrid pipeline validated successfully"


def compute_piece_reach(state: 'GameState', piece_coord: np.ndarray) -> np.ndarray:
    """
    Precompute all squares a piece can potentially move to/from.
    Used for dependency tracking in hybrid pipeline.

    Returns: (N, 3) array of reachable squares
    """
    # Get piece type and movement patterns
    piece_info = state.cache_manager.occupancy_cache.get(piece_coord)
    if piece_info is None:
        return np.empty((0, 3), dtype=COORD_DTYPE)

    ptype = piece_info["piece_type"]
    color = piece_info["color"]

    # Get movement directions and range
    from game3d.movement.generator import get_piece_directions
    dirs, max_dist, is_slider = get_piece_directions(ptype)

    if dirs.size == 0:
        return np.empty((0, 3), dtype=COORD_DTYPE)

    # Generate all potential destinations (without validation)
    reach_coords = []

    if is_slider:
        for direction in dirs:
            for step in range(1, max_dist + 1):
                dest = piece_coord + direction * step
                if np.all((dest >= MIN_COORD_VALUE) & (dest <= SIZE_MINUS_1)):
                    reach_coords.append(dest)
                else:
                    break
    else:
        for direction in dirs:
            dest = piece_coord + direction
            if np.all((dest >= MIN_COORD_VALUE) & (dest <= SIZE_MINUS_1)):
                reach_coords.append(dest)

    # Include the piece's own square
    reach_coords.append(piece_coord)

    return np.unique(np.array(reach_coords, dtype=COORD_DTYPE), axis=0)
