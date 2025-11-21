# auracache_optimized.py – FULLY NUMPY-NATIVE CONSOLIDATED EFFECT CACHE
from __future__ import annotations
import numpy as np
from numba import njit, prange
from typing import Optional, Dict, Any, Tuple
import threading

from game3d.common.shared_types import (
    COORD_DTYPE, BOOL_DTYPE, INDEX_DTYPE, COLOR_DTYPE, PIECE_TYPE_DTYPE,
    SIZE, VOLUME, MAX_COORD_VALUE, PieceType, Color,
    FREEZER, SPEEDER, SLOWER, BLACKHOLE, WHITEHOLE,
    RADIUS_2_OFFSETS
)
from game3d.common.coord_utils import ensure_coords, in_bounds_vectorized, coord_to_idx
from game3d.cache.cache_protocols import CacheListener

# =============================================================================
# EFFECT TYPE DEFINITIONS - NUMPY CONSTANTS
# =============================================================================

# Effect types as uint8 constants for numpy operations
EFFECT_FREEZE = np.uint8(1)
EFFECT_BUFF = np.uint8(2)
EFFECT_DEBUFF = np.uint8(3)
EFFECT_PULL = np.uint8(4)
EFFECT_PUSH = np.uint8(5)

# Number of effect types
N_EFFECT_TYPES = 5

# =============================================================================
# PIECE TYPE TO EFFECT MAPPING - SIMPLIFIED FOR 5 EFFECTS
# =============================================================================

# Create boolean mask: piece_type -> effect_type
# Only include the 5 pieces we're using
max_piece_val = max(FREEZER.value, SPEEDER.value, SLOWER.value, BLACKHOLE.value, WHITEHOLE.value)
EFFECT_PIECE_MASK = np.zeros((max_piece_val + 1, N_EFFECT_TYPES + 1), dtype=BOOL_DTYPE)

# Populate mask: piece_type -> which effects it applies
EFFECT_PIECE_MASK[FREEZER.value, EFFECT_FREEZE] = True
EFFECT_PIECE_MASK[SPEEDER.value, EFFECT_BUFF] = True
EFFECT_PIECE_MASK[SLOWER.value, EFFECT_DEBUFF] = True
EFFECT_PIECE_MASK[BLACKHOLE.value, EFFECT_PULL] = True
EFFECT_PIECE_MASK[WHITEHOLE.value, EFFECT_PUSH] = True

# =============================================================================
# NUMBA-OPTIMIZED CORE OPERATIONS
# =============================================================================

@njit(cache=True, fastmath=True, parallel=True)
def _apply_aura_to_targets_numba(
    target_coords: np.ndarray,
    effect_type: np.uint8,
    effects_grid: np.ndarray,
    target_flags_grid: np.ndarray
) -> None:
    """Vectorized effect application to target coordinates."""
    n = target_coords.shape[0]
    for i in prange(n):
        x, y, z = target_coords[i, 0], target_coords[i, 1], target_coords[i, 2]
        effects_grid[x, y, z] = effect_type
        target_flags_grid[x, y, z] = True

@njit(cache=True, fastmath=True, parallel=True)
def _clear_effects_numba(
    coords: np.ndarray,
    effects_grid: np.ndarray,
    flags_grid: np.ndarray,
    pull_push_grid: np.ndarray
) -> None:
    """Vectorized clearing of effects at coordinates."""
    n = coords.shape[0]
    for i in prange(n):
        x, y, z = coords[i, 0], coords[i, 1], coords[i, 2]
        effects_grid[x, y, z] = 0
        flags_grid[x, y, z] = False
        pull_push_grid[x, y, z] = 0

# =============================================================================
# MAIN EFFECT CACHE CLASS
# =============================================================================

class ConsolidatedAuraCache(CacheListener):
    """Fully numpy-native consolidated effect cache for 5 effects."""

    __slots__ = (
        "board", "cm", "config",
        "_effects", "_effect_sources", "_effect_ages",
        "_frozen_squares", "_debuffed_squares", "_buffed_squares",
        "_pull_map", "_push_map",
        "_memory_pool", "_lock"
    )

    def __init__(self, board, cache_manager, config=None):
        self.board = board
        self.cm = cache_manager
        self.config = config or self._default_config()

        # Memory pool for efficient allocations
        self._memory_pool = getattr(cache_manager, '_memory_pool', None)

        # Thread safety for complex updates
        self._lock = threading.RLock()

        # Initialize all effect arrays
        self._init_effect_arrays()

    def _default_config(self):
        """Configuration as numpy-compatible dict."""
        return {
            'freeze_radius': 2,
            'buff_radius': 2,
            'debuff_radius': 2,
            'directional_radius': 1
        }

    def _init_effect_arrays(self):
        """Pre-allocate all effect arrays."""
        # Main effect types per square
        self._effects = np.zeros((SIZE, SIZE, SIZE), dtype=np.uint8)

        # Source piece tracking
        self._effect_sources = np.full((SIZE, SIZE, SIZE), -1, dtype=np.int32)

        # Effect ages for decay
        self._effect_ages = np.zeros((SIZE, SIZE, SIZE), dtype=INDEX_DTYPE)

        # Specific effect flags (dense arrays for fast lookup)
        self._frozen_squares = np.zeros((SIZE, SIZE, SIZE), dtype=BOOL_DTYPE)
        self._debuffed_squares = np.zeros((SIZE, SIZE, SIZE), dtype=BOOL_DTYPE)
        self._buffed_squares = np.zeros((SIZE, SIZE, SIZE), dtype=BOOL_DTYPE)

        # Directional effect maps (pull/push)
        self._pull_map = np.zeros((SIZE, SIZE, SIZE, 3), dtype=COORD_DTYPE)
        self._push_map = np.zeros((SIZE, SIZE, SIZE, 3), dtype=COORD_DTYPE)

    # ========================================================================
    # CACHE LISTENER INTERFACE
    # ========================================================================

    def get_priority(self) -> int:
        """Return priority for update order (lower = higher priority)."""
        return 2  # High priority for aura effects

    def on_occupancy_changed(self, changed_coords: np.ndarray, pieces: np.ndarray) -> None:
        """Handle single coordinate change - wraps batch handler."""
        if changed_coords.size == 0:
            return

        changed_coords = self._ensure_coords(changed_coords)
        pieces_array = np.asarray(pieces, dtype=PIECE_TYPE_DTYPE)
        pieces_array = pieces_array.reshape(-1, 2)  # Ensure (N, 2) shape

        self.on_batch_occupancy_changed(changed_coords, pieces_array)

    def on_batch_occupancy_changed(self, coords: np.ndarray, pieces: np.ndarray) -> None:
        """Batch occupancy change - vectorized update."""
        if coords.size == 0:
            return

        with self._lock:
            coords = self._ensure_coords(coords)

            # Get old piece types before change
            old_pieces = self.cm.occupancy_cache.batch_get_attributes(coords)[1]

            # Clear old effects
            self._clear_piece_effects_batch(coords, old_pieces)

            # Apply new effects for effect pieces
            new_piece_types = pieces[:, 0] if pieces.ndim == 2 else pieces
            effect_mask = self._is_effect_piece_batch(new_piece_types)

            if np.any(effect_mask):
                effect_coords = coords[effect_mask]
                effect_types = new_piece_types[effect_mask]
                self._apply_piece_effects_batch(effect_coords, effect_types)

    # ========================================================================
    # VECTORIZED EFFECT APPLICATION
    # ========================================================================

    def _apply_piece_effects_batch(self, coords: np.ndarray, piece_types: np.ndarray) -> None:
        """Batch apply effects from multiple pieces."""
        if coords.size == 0:
            return

        # Filter valid piece types first
        valid_types = (piece_types >= 0) & (piece_types < len(EFFECT_PIECE_MASK))
        if not np.any(valid_types):
            return
            
        valid_coords = coords[valid_types]
        valid_piece_types = piece_types[valid_types]

        # Process each effect type separately for optimal vectorization
        for effect_type in (EFFECT_FREEZE, EFFECT_BUFF, EFFECT_DEBUFF):
            mask = EFFECT_PIECE_MASK[valid_piece_types, effect_type]
            if np.any(mask):
                self._apply_aura_effect_vectorized(valid_coords[mask], effect_type)

        # Process directional effects
        for effect_type in (EFFECT_PULL, EFFECT_PUSH):
            mask = EFFECT_PIECE_MASK[valid_piece_types, effect_type]
            if np.any(mask):
                self._apply_directional_effect_vectorized(valid_coords[mask], effect_type)

    def _apply_aura_effect_vectorized(self, source_coords: np.ndarray, effect_type: np.uint8) -> None:
        """Fully vectorized aura effect application for multiple sources."""
        if source_coords.size == 0:
            return

        # Get source piece colors
        n_sources = source_coords.shape[0]
        source_colors = np.empty(n_sources, dtype=COLOR_DTYPE)
        for i in range(n_sources):
            piece = self.cm.occupancy_cache.get(source_coords[i])
            source_colors[i] = piece['color'] if piece else Color.EMPTY

        # Get all potential target coordinates
        offsets = RADIUS_2_OFFSETS
        n_offsets = offsets.shape[0]

        # Vectorized: (n_sources, n_offsets, 3)
        source_expanded = source_coords[:, np.newaxis, :]
        offset_expanded = offsets[np.newaxis, :, :]
        target_coords = source_expanded + offset_expanded
        target_coords = target_coords.reshape(-1, 3)

        # Remove duplicates and check bounds
        target_coords = np.unique(target_coords, axis=0)
        valid_mask = in_bounds_vectorized(target_coords)
        valid_targets = target_coords[valid_mask]

        if valid_targets.size == 0:
            return

        # Determine which targets should be affected based on color relationships
        affected_coords = self._filter_targets_by_color(valid_targets, source_coords, source_colors, effect_type)

        if affected_coords.size == 0:
            return

        # Vectorized application to flag grids
        x, y, z = affected_coords[:, 0], affected_coords[:, 1], affected_coords[:, 2]
        self._effects[x, y, z] = effect_type

        if effect_type == EFFECT_FREEZE:
            self._frozen_squares[x, y, z] = True
        elif effect_type == EFFECT_BUFF:
            self._buffed_squares[x, y, z] = True
        elif effect_type == EFFECT_DEBUFF:
            self._debuffed_squares[x, y, z] = True

    def _filter_targets_by_color(self, targets: np.ndarray, sources: np.ndarray, source_colors: np.ndarray, effect_type: np.uint8) -> np.ndarray:
        """Filter target coordinates based on color relationships with sources."""
        affected_list = []

        for target in targets:
            target_piece = self.cm.occupancy_cache.get(target)
            if target_piece is None:
                continue

            target_color = target_piece['color']

            # Check each source for color-based effect application
            for i in range(sources.shape[0]):
                if effect_type == EFFECT_FREEZE and target_color != source_colors[i]:
                    affected_list.append(target)
                    break
                elif effect_type == EFFECT_BUFF and target_color == source_colors[i] and not np.array_equal(target, sources[i]):
                    affected_list.append(target)
                    break
                elif effect_type == EFFECT_DEBUFF and target_color != source_colors[i]:
                    affected_list.append(target)
                    break

        if not affected_list:
            return np.empty((0, 3), dtype=COORD_DTYPE)

        return np.unique(np.array(affected_list, dtype=COORD_DTYPE), axis=0)

    def _apply_directional_effect_vectorized(self, source_coords: np.ndarray, effect_type: np.uint8) -> None:
        """Vectorized directional effect (PULL/PUSH) for multiple sources."""
        if source_coords.size == 0:
            return

        # Get controller color from first source
        source_piece = self.cm.occupancy_cache.get(source_coords[0])
        if source_piece is None:
            return

        controller_color = source_piece['color']
        enemy_color = Color.WHITE if controller_color == Color.BLACK else Color.BLACK

        # Get all enemy coordinates
        enemy_coords = self.cm.occupancy_cache.get_positions(enemy_color)
        if enemy_coords.size == 0:
            return

        # Process each source
        for i in range(source_coords.shape[0]):
            source_coord = source_coords[i]
            self._apply_directional_for_single_source(source_coord, enemy_coords, effect_type)

    def _apply_directional_for_single_source(self, source_coord: np.ndarray, enemy_coords: np.ndarray, effect_type: np.uint8) -> None:
        """Apply directional effect for a single source."""
        # Calculate Chebyshev distances to all enemies
        distances = np.max(np.abs(enemy_coords - source_coord), axis=1)
        in_range = distances <= 1

        if not np.any(in_range):
            return

        # Find closest enemy
        valid_distances = np.where(in_range, distances, np.iinfo(INDEX_DTYPE).max)
        closest_idx = np.argmin(valid_distances)

        if not in_range[closest_idx]:
            return

        enemy_coord = enemy_coords[closest_idx]

        # Calculate movement direction
        if effect_type == EFFECT_PULL:
            direction = np.sign(enemy_coord - source_coord).astype(COORD_DTYPE)
        else:  # PUSH
            direction = np.sign(source_coord - enemy_coord).astype(COORD_DTYPE)

        # Validate target position
        target_coord = enemy_coord + direction

        if not in_bounds_vectorized(target_coord):
            return

        if self.cm.occupancy_cache.get(target_coord) is not None:
            return

        # Apply effect
        x, y, z = enemy_coord
        self._effects[x, y, z] = effect_type

        if effect_type == EFFECT_PULL:
            self._pull_map[x, y, z] = direction
        else:
            self._push_map[x, y, z] = direction

    # ========================================================================
    # VECTORIZED EFFECT CLEARING
    # ========================================================================

    def _clear_piece_effects_batch(self, coords: np.ndarray, piece_types: np.ndarray) -> None:
        """Batch clear effects from multiple pieces."""
        if coords.size == 0:
            return

        # Collect all coordinates that need clearing (source + affected area)
        clear_coords = [coords]  # Start with direct coordinates

        # Add radius-2 area for aura pieces
        aura_mask = np.isin(piece_types, [FREEZER.value, SPEEDER.value, SLOWER.value])
        if np.any(aura_mask):
            aura_coords = coords[aura_mask]
            for coord in aura_coords:
                offsets = RADIUS_2_OFFSETS
                area_coords = coord + offsets
                valid_mask = in_bounds_vectorized(area_coords)
                clear_coords.append(area_coords[valid_mask])

        # Combine and deduplicate
        all_clear_coords = np.concatenate(clear_coords)
        unique_coords = np.unique(all_clear_coords, axis=0)

        # Vectorized clearing of all arrays
        x, y, z = unique_coords[:, 0], unique_coords[:, 1], unique_coords[:, 2]

        self._effects[x, y, z] = 0
        self._effect_sources[x, y, z] = -1
        self._effect_ages[x, y, z] = 0
        self._frozen_squares[x, y, z] = False
        self._debuffed_squares[x, y, z] = False
        self._buffed_squares[x, y, z] = False
        self._pull_map[x, y, z] = 0
        self._push_map[x, y, z] = 0

    # ========================================================================
    # QUERY OPERATIONS - VECTORIZED
    # ========================================================================

    def batch_is_frozen(self, coords: np.ndarray, color: int) -> np.ndarray:
        """Batch check frozen status."""
        coords = self._ensure_coords(coords)
        if coords.size == 0:
            return np.empty(0, dtype=BOOL_DTYPE)

        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        return self._frozen_squares[x, y, z]

    def batch_is_debuffed(self, coords: np.ndarray, color: int) -> np.ndarray:
        """Batch check debuffed status."""
        coords = self._ensure_coords(coords)
        if coords.size == 0:
            return np.empty(0, dtype=BOOL_DTYPE)

        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        return self._debuffed_squares[x, y, z]

    def batch_is_buffed(self, coords: np.ndarray, color: int) -> np.ndarray:
        """Batch check buffed status."""
        coords = self._ensure_coords(coords)
        if coords.size == 0:
            return np.empty(0, dtype=BOOL_DTYPE)

        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        return self._buffed_squares[x, y, z]

    def get_frozen_squares(self, color: int) -> np.ndarray:
        """Get all frozen squares as coordinate array."""
        frozen_indices = np.argwhere(self._frozen_squares)
        if frozen_indices.size == 0:
            return np.empty((0, 3), dtype=COORD_DTYPE)
        # Convert from (z, y, x) to (x, y, z)
        return frozen_indices[:, [2, 1, 0]].astype(COORD_DTYPE)

    def get_buffed_squares(self, color: int) -> np.ndarray:
        """Get all buffed squares as coordinate array."""
        buffed_indices = np.argwhere(self._buffed_squares)
        if buffed_indices.size == 0:
            return np.empty((0, 3), dtype=COORD_DTYPE)
        return buffed_indices[:, [2, 1, 0]].astype(COORD_DTYPE)

    def get_debuffed_squares(self, color: int) -> np.ndarray:
        """Get all debuffed squares as coordinate array."""
        debuffed_indices = np.argwhere(self._debuffed_squares)
        if debuffed_indices.size == 0:
            return np.empty((0, 3), dtype=COORD_DTYPE)
        return debuffed_indices[:, [2, 1, 0]].astype(COORD_DTYPE)

    def pull_map(self, controller: int) -> np.ndarray:
        """Get pull map as array of (source, target) pairs."""
        pull_sources = np.argwhere(self._effects == EFFECT_PULL)
        if pull_sources.size == 0:
            return np.empty((0, 2, 3), dtype=COORD_DTYPE)

        result = np.empty((pull_sources.shape[0], 2, 3), dtype=COORD_DTYPE)
        for i, (x, y, z) in enumerate(pull_sources):
            source = np.array([z, y, x], dtype=COORD_DTYPE)
            target = self._pull_map[x, y, z]
            result[i] = [source, target]

        return result

    def push_map(self, controller: int) -> np.ndarray:
        """Get push map as array of (source, target) pairs."""
        push_sources = np.argwhere(self._effects == EFFECT_PUSH)
        if push_sources.size == 0:
            return np.empty((0, 2, 3), dtype=COORD_DTYPE)

        result = np.empty((push_sources.shape[0], 2, 3), dtype=COORD_DTYPE)
        for i, (x, y, z) in enumerate(push_sources):
            source = np.array([z, y, x], dtype=COORD_DTYPE)
            target = self._push_map[x, y, z]
            result[i] = [source, target]

        return result

    # ========================================================================
    # INFRASTRUCTURE
    # ========================================================================

    def invalidate_all(self) -> None:
        """Clear all effect data."""
        with self._lock:
            self._init_effect_arrays()

    def _is_effect_piece_batch(self, piece_types: np.ndarray) -> np.ndarray:
        """Batch check which piece types have effects."""
        # Bounds checking: only check types within the mask range
        valid_types = (piece_types >= 0) & (piece_types < len(EFFECT_PIECE_MASK))
        result = np.zeros(piece_types.shape[0], dtype=BOOL_DTYPE)
        
        if np.any(valid_types):
            valid_indices = piece_types[valid_types]
            result[valid_types] = EFFECT_PIECE_MASK[valid_indices].any(axis=1)
        
        return result

    def _ensure_coords(self, coords: np.ndarray) -> np.ndarray:
        """Ensure coordinates are (N, 3) numpy arrays."""
        if coords is None or coords.size == 0:
            return np.empty((0, 3), dtype=COORD_DTYPE)

        coords = np.asarray(coords, dtype=COORD_DTYPE)

        if coords.ndim == 0:
            return coords.reshape(1, 3)
        elif coords.ndim == 1 and coords.shape[0] == 3:
            return coords.reshape(1, 3)
        elif coords.ndim == 2 and coords.shape[1] == 3:
            return coords
        else:
            raise ValueError(f"Invalid coordinate shape: {coords.shape}")

    def get_memory_usage(self) -> Dict[str, int]:
        """Get detailed memory usage."""
        with self._lock:
            return {
                'effects_array': self._effects.nbytes,
                'sources_array': self._effect_sources.nbytes,
                'ages_array': self._effect_ages.nbytes,
                'frozen_flags': self._frozen_squares.nbytes,
                'debuff_flags': self._debuffed_squares.nbytes,
                'buff_flags': self._buffed_squares.nbytes,
                'pull_map': self._pull_map.nbytes,
                'push_map': self._push_map.nbytes,
                'total_bytes': (
                    self._effects.nbytes + self._effect_sources.nbytes +
                    self._effect_ages.nbytes + self._frozen_squares.nbytes +
                    self._debuffed_squares.nbytes + self._buffed_squares.nbytes +
                    self._pull_map.nbytes + self._push_map.nbytes
                )
            }

# Module exports
__all__ = ['ConsolidatedAuraCache', 'EFFECT_FREEZE', 'EFFECT_BUFF', 'EFFECT_DEBUFF',
           'EFFECT_PULL', 'EFFECT_PUSH']
