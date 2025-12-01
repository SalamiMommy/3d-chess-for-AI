# manager.py - FULLY NUMPY-NATIVE OPTIMIZED CACHE MANAGER
from __future__ import annotations
import numpy as np
from numba import njit, prange
import threading
import os
from typing import Optional, Dict, Any
import logging
import weakref

logger = logging.getLogger(__name__)

from game3d.common.shared_types import (
    COORD_DTYPE, BOOL_DTYPE, FLOAT_DTYPE, SIZE, VOLUME, PIECE_TYPE_DTYPE,
    INDEX_DTYPE, COLOR_DTYPE, HASH_DTYPE, Color, N_PIECE_TYPES, MOVE_DTYPE,
    MAX_COORD_VALUE, MIN_COORD_VALUE, VECTORIZATION_THRESHOLD, TRAILBLAZER,
    PIECE_SLICE
)
from game3d.common.coord_utils import coord_to_idx, idx_to_coord, ensure_coords, in_bounds_vectorized, get_neighbors_vectorized
from game3d.cache.managerconfig import ManagerConfig
from game3d.cache.caches.occupancycache import OccupancyCache
from game3d.cache.caches.zobrist import ZobristHash
from game3d.cache.caches.movecache import create_move_cache
from game3d.cache.effectscache.trailblazecache import TrailblazeCache
from game3d.cache.effectscache.geomancycache import GeomancyCache
from game3d.cache.effectscache.auracache import ConsolidatedAuraCache
from game3d.board.symmetry import SymmetryManager
from game3d.cache.caches.symmetry_tt import SymmetryAwareTranspositionTable


# =============================================================================
# GLOBAL REGISTRY - NUMPY-NATIVE IMPLEMENTATION
# =============================================================================

_MAX_MANAGERS = 128
_REGISTRY_DTYPE = np.dtype([
    ('board_id', np.int64),
    ('color', COLOR_DTYPE),
    ('manager_weakref', np.int64),  # id() of manager for weak reference simulation
    ('active', BOOL_DTYPE)
])

_ACTIVE_CACHE_MANAGERS = np.zeros(_MAX_MANAGERS, dtype=_REGISTRY_DTYPE)
_REGISTRY_LOCK = threading.Lock()
_REGISTRY_COUNT = np.array([0], dtype=INDEX_DTYPE)

@njit(cache=True, nogil=True)
def _find_manager_idx(registry: np.ndarray, board_id: int, color: int) -> int:
    """Numba-optimized registry lookup."""
    for i in range(len(registry)):
        if (registry[i]['active'] and
            registry[i]['board_id'] == board_id and
            registry[i]['color'] == color):
            return i
    return -1

def get_cache_manager(board, color: int = 0, initial_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None) -> "OptimizedCacheManager":
    """Thread-safe manager retrieval/caching using numpy registry."""
    global _ACTIVE_CACHE_MANAGERS, _REGISTRY_COUNT

    board_id = id(board)

    with _REGISTRY_LOCK:
        idx = _find_manager_idx(_ACTIVE_CACHE_MANAGERS, board_id, color)

        if idx >= 0:
            # Manager exists - would use weakref in production
            return OptimizedCacheManager(board, color, initial_data)

        # Create new manager
        if _REGISTRY_COUNT[0] >= _MAX_MANAGERS:
            logger.error(f"Registry full ({_MAX_MANAGERS} managers)")
            raise RuntimeError("Cache manager registry overflow")

        manager = OptimizedCacheManager(board, color, initial_data)

        _ACTIVE_CACHE_MANAGERS[_REGISTRY_COUNT[0]] = (
            board_id, color, id(manager), True
        )
        _REGISTRY_COUNT[0] += 1

        logger.debug(f"Created cache manager #{_REGISTRY_COUNT[0]} for board {board_id}")
        return manager

def _get_adjacent_squares(coord: np.ndarray) -> np.ndarray:
    """Return adjacent coordinates for capture effect processing."""
    coord_batch = coord.reshape(1, 3)
    return get_neighbors_vectorized(coord_batch)
# =============================================================================
# NUMPY-NATIVE STATISTICS TRACKER
# =============================================================================

class NumpyStatsTracker:
    """Lock-protected statistics using numpy structured arrays."""

    __slots__ = ("_stats", "_lock")

    def __init__(self):
        self._stats = np.zeros(1, dtype=[
            ('hits', INDEX_DTYPE),
            ('misses', INDEX_DTYPE),
            ('operations', INDEX_DTYPE),
            ('time', FLOAT_DTYPE)
        ])
        self._lock = threading.Lock()

    def record_hit(self) -> None:
        with self._lock:
            self._stats['hits'] += 1
            self._stats['operations'] += 1

    def record_miss(self) -> None:
        with self._lock:
            self._stats['misses'] += 1
            self._stats['operations'] += 1

    def add_timing(self, duration: float) -> None:
        with self._lock:
            self._stats['time'] += duration

    def get_stats(self) -> Dict[str, float]:
        with self._lock:
            s = self._stats[0]
            total = s['hits'] + s['misses']
            return {
                'hits': int(s['hits']),
                'misses': int(s['misses']),
                'hit_rate': float(s['hits'] / max(total, 1)),
                'avg_time': float(s['time'] / max(s['operations'], 1))
            }

# =============================================================================
# NUMBA-OPTIMIZED CORE OPERATIONS
# =============================================================================



@njit(cache=True, nogil=True)
def _coords_to_keys(coords: np.ndarray) -> np.ndarray:
    """Convert coordinates to cache keys using tobytes-equivalent."""
    n = coords.shape[0]
    # Use flat indices as keys (much faster than bytes)
    keys = np.empty(n, dtype=INDEX_DTYPE)
    for i in range(n):
        x, y, z = coords[i]
        # Match generator.py bit packing: x | (y << 9) | (z << 18)
        keys[i] = x | (y << 9) | (z << 18)
    return keys

# =============================================================================
# OPTIMIZED CACHE MANAGER
# =============================================================================

class OptimizedCacheManager:
    """Pure numpy-native cache manager - zero Python loops in hot paths."""

    __slots__ = (
        "_board", "_board_generation", "_current", "_move_counter",
        "config", "_stats_tracker", "_move_stats",
        "occupancy_cache", "zobrist_cache", "_zkey",
        "transposition_table", "symmetry_manager", "move_cache",
        "_effect_caches", "_effect_cache_instances", "_batch_effect_cache",
        "dependency_graph", "_memory_pool", "_parallel_executor",
        "_start_tensor_cache", "_lock", "__weakref__"
    )

    def __init__(self, board, color: int = 0, initial_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None):
        self._board = board
        self._board_generation = getattr(board, 'generation', 0)
        self._current = color
        self._move_counter = 0

        # Config and stats
        self.config = ManagerConfig()
        self._stats_tracker = NumpyStatsTracker()
        self._move_stats = NumpyStatsTracker()
        self._lock = threading.RLock()

        # Initialize caches
        self._initialize_caches()

        # CRITICAL: Initialize occupancy from provided data or board setup
        if initial_data is not None:
            self._initialize_from_data(initial_data)
        else:
            self._initialize_from_setup(board.get_initial_setup())

        # Compute initial Zobrist
        self._zkey = self._compute_initial_zobrist(color)

        # Link board (bidirectional)
        self._board._cache_manager = self

        # Sync generation
        self.move_cache._board_generation = self._board_generation

        # Diagnostics
        from game3d.common.diagnostics import record_cache_creation
        record_cache_creation(self, board)

    @property
    def board(self):
        """Get the board (read-only access preferred)."""
        return self._board


    @board.setter
    def board(self, new_board):
        """
        Set board - ONLY for clone/reset operations.

        During normal gameplay, DON'T use this setter.
        """
        new_generation = getattr(new_board, 'generation', 0)
        self._board = new_board
        self._board_generation = new_generation

    def _initialize_caches(self) -> None:
        """Initialize all cache subsystems."""
        # Core caches
        self.occupancy_cache = OccupancyCache()
        self.zobrist_cache = ZobristHash()
        
        # Symmetry and Transposition
        self.symmetry_manager = SymmetryManager(self)
        self.transposition_table = SymmetryAwareTranspositionTable(self.symmetry_manager)
        
        # Move Cache (requires self for callback/access)
        self.move_cache = create_move_cache(self)
        
        # Dependency Graph
        self.dependency_graph = NumpyDependencyGraph(self)
        
        # Effect Caches
        self._effect_caches = {}
        
        aura_cache = ConsolidatedAuraCache(self._board, self)
        geomancy_cache = GeomancyCache(self)
        trailblaze_cache = TrailblazeCache(self)
        
        self._effect_cache_instances = [
            aura_cache,
            geomancy_cache,
            trailblaze_cache
        ]

    def _initialize_from_setup(self, setup: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """Initialize occupancy from board setup (coords, types, colors)."""
        coords, types, colors = setup
        self.occupancy_cache.rebuild(coords, types, colors)
        logger.info(f"Initialized occupancy cache from board setup: {len(coords)} pieces")

    def _initialize_from_data(self, data: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """Initialize occupancy from existing data (cloning)."""
        coords, types, colors = data
        self.occupancy_cache.import_state(coords, types, colors)
        logger.info(f"Initialized occupancy cache from data: {len(coords)} pieces")


    def _compute_initial_zobrist(self, color: int) -> HASH_DTYPE:
        """Vectorized Zobrist computation."""
        coords, piece_types, colors = self.occupancy_cache.get_all_occupied_vectorized()

        if coords.shape[0] == 0:
            return int(self.zobrist_cache._side_key) if color == Color.BLACK else int(HASH_DTYPE(0))

        # Convert to indices
        piece_indices = (piece_types - 1).astype(PIECE_TYPE_DTYPE)
        color_indices = (colors - Color.WHITE).astype(COLOR_DTYPE)

        # Extract keys
        keys = self.zobrist_cache._piece_keys[
            piece_indices,
            color_indices,
            coords[:, 0], coords[:, 1], coords[:, 2]
        ]

        # XOR reduction
        zkey = np.bitwise_xor.reduce(keys)
        if color == Color.BLACK:
            zkey ^= self.zobrist_cache._side_key

        # Convert to Python int to ensure hashability
        return int(zkey)

    def apply_move(self, mv_obj: np.ndarray, color: int) -> bool:
        """Apply move - fully vectorized invalidation."""
        with self._lock:
            self._move_counter += 1

            # Invalidate opponent's move cache
            opp_color = Color.WHITE if color == Color.BLACK else Color.BLACK
            self.move_cache.invalidate_color(opp_color)

            # Update Zobrist hash
            from_coord = mv_obj[:3]
            to_coord = mv_obj[3:]
            from_piece = self.occupancy_cache.get(from_coord)
            captured_piece = self.occupancy_cache.get(to_coord)

            if from_piece is None:
                raise ValueError(f"No piece at source {from_coord}")

            # Ensure _zkey is always a Python int, not numpy scalar
            self._zkey = int(self.zobrist_cache.update_hash_move(
                self._zkey, mv_obj, from_piece, captured_piece
            ))

            # Notify dependency graph
            self.dependency_graph.notify_update('move_applied')

            # Batch notify effect caches
            # mv_obj is expected to be (6,) array: [fx, fy, fz, tx, ty, tz]
            affected_coords = np.array([
                mv_obj[:3], mv_obj[3:]
            ], dtype=COORD_DTYPE)
            self._notify_all_effect_caches(affected_coords, np.array([[0,0],[0,0]], dtype=PIECE_TYPE_DTYPE))

            return True

    def undo_move(self, last_mv: np.ndarray, color: int) -> None:
        """
        Undo move - requires full rebuild.

        This is acceptable because undo is rare (only for search/debug).
        """
        with self._lock:
            self._move_counter -= 1

            # Recompute Zobrist from scratch
            self._zkey = self._compute_initial_zobrist(color)

            # Invalidate all caches
            self.dependency_graph.notify_update('move_undone')
            self.move_cache.invalidate()

            logger.debug(f"Undo completed: Rebuilt caches")

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Aggregate statistics from all caches - numpy native."""
        stats = self.move_cache.get_statistics()
        stats.update(self.occupancy_cache.get_memory_usage())

        # Add per-effect cache stats - only sum numeric values
        total_effect_memory = np.int64(0)
        for cache in self._effect_cache_instances:
            if hasattr(cache, 'get_memory_usage'):
                usage = cache.get_memory_usage()
                # Use numpy-native sum of only numeric values
                for value in usage.values():
                    if isinstance(value, (int, np.integer)):
                        total_effect_memory += np.int64(value)

        stats['effect_cache_memory'] = int(total_effect_memory)
        stats.update(self._stats_tracker.get_stats())
        stats.update(self._move_stats.get_stats())

        return stats

    def request_legal_moves(self, state: 'GameState') -> np.ndarray:
        """Enforce move generation through generator."""
        from game3d.movement.generator import generate_legal_moves
        return generate_legal_moves(state)

    def get_move_cache_key(self, color: int) -> int:
        """Generate cache key from Zobrist."""
        # Board generation is no longer used as Board is stateless
        # Zobrist key is sufficient for position uniqueness
        return ((int(self._zkey) & 0xFFFF) << 1) | (int(color) & 1)

    def update_zobrist_after_move(self, current_hash: HASH_DTYPE, move: np.ndarray,
                                 from_piece: object, captured_piece: object | None) -> HASH_DTYPE:
        """Facade for Zobrist updates."""
        # move is already np.ndarray
        return self.zobrist_cache.update_hash_move(current_hash, move, from_piece, captured_piece)

    def update_position_history(self, zkey: HASH_DTYPE) -> None:
        """Placeholder for position tracking."""
        pass



    def get_parallel_executor(self) -> 'ParallelManager':
        """Lazy parallel executor creation."""
        if not hasattr(self, '_parallel_executor'):
            from game3d.cache.parallelmanager import ParallelManager

            config = ManagerConfig()
            n_cpus = os.cpu_count() or 6
            config.max_workers = max(1, n_cpus - 1)
            config.enable_parallel = True

            self._parallel_executor = ParallelManager(config)

        return self._parallel_executor

    @property
    def consolidated_aura_cache(self) -> 'ConsolidatedAuraCache':
        """Access the consolidated aura cache."""
        return self._effect_cache_instances[0]

    @property
    def geomancy_cache(self) -> 'GeomancyCache':
        """Access the geomancy cache."""
        return self._effect_cache_instances[1]
    
    @property
    def trailblaze_cache(self) -> 'TrailblazeCache':
        """Access the trailblaze cache."""
        return self._effect_cache_instances[2]



    def _notify_all_effect_caches(self, changed_coords: np.ndarray, pieces: np.ndarray) -> None:
        """Batch notification to all effect caches."""
        # Iterate over pre-cached instances (avoid getattr in hot path)
        for cache in self._effect_cache_instances:
            cache.on_batch_occupancy_changed(changed_coords, pieces)

    def _invalidate_affected_piece_moves(self, affected_coords: np.ndarray, color: int, game_state) -> None:
        """
        Vectorized invalidation of moved pieces and dependencies.

        This marks pieces for regeneration but does NOT regenerate immediately.
        Regeneration happens lazily when moves are requested.
        """
        if affected_coords.size == 0:
            return

        with self._lock:
            opp_color = Color.WHITE if color == Color.BLACK else Color.BLACK
            c_idx = 0 if color == Color.WHITE else 1
            opp_c_idx = 0 if opp_color == Color.WHITE else 1

            # --- STEP 1: Mark direct pieces (at affected squares) ---
            if affected_coords.size > 0:
                keys = _coords_to_keys(affected_coords)
                
                # Invalidate for both colors using batch operations
                n = keys.size
                c_indices = np.full(n, c_idx, dtype=np.int8)
                opp_c_indices = np.full(n, opp_c_idx, dtype=np.int8)
                
                self.move_cache.mark_pieces_invalid_batch(c_indices, keys)
                self.move_cache.mark_pieces_invalid_batch(opp_c_indices, keys)

                # --- STEP 2: Mark pieces targeting these squares (Reverse Map) ---
                self.move_cache.invalidate_targeting_pieces(keys)

            # --- STEP 3: Get dependency-affected pieces ---
            dep_coords = self.dependency_graph.get_affected_pieces_vectorized(game_state, color)

            if dep_coords.size > 0:
                dep_keys = _coords_to_keys(dep_coords)
                
                # Batch invalidate
                c_indices = np.full(dep_keys.size, c_idx, dtype=np.int8)
                self.move_cache.mark_pieces_invalid_batch(c_indices, dep_keys)




    def apply_move_incremental(self, mv: np.ndarray, from_piece: dict,
                            captured_piece: Optional[dict], game_state: 'GameState') -> None:
        """
        Incrementally update ALL caches after a move.

        CRITICAL PRECONDITION: Board has already been updated by turnmove.make_move()
        We're syncing the caches to match the board state.

        UPDATE ORDER:
        1. Occupancy cache (from board changes)
        2. Zobrist hash (from occupancy changes)
        3. Effect caches (from occupancy changes)
        4. Move cache invalidation (from effect changes)
        """
        with self._lock:
            # --- STEP 1: SYNC OCCUPANCY CACHE (board already updated) ---
            # This batch_set_positions will update king cache and priest counts
            changed_coords = np.array([mv[:3], mv[3:]], dtype=COORD_DTYPE)
            pieces_data = np.array([
                [0, 0],  # Source square empty
                [from_piece["piece_type"], from_piece["color"]]  # Dest square occupied
            ], dtype=PIECE_TYPE_DTYPE)

            self.occupancy_cache.batch_set_positions(changed_coords, pieces_data)

            # --- STEP 2: UPDATE ZOBRIST HASH ---
            self._zkey = int(self.zobrist_cache.update_hash_move(
                self._zkey, mv, from_piece, captured_piece
            ))

            # --- STEP 3: SYMMETRY TT (age out old entries) ---
            self.transposition_table.age_counter += 1

            # --- STEP 4: NOTIFY EFFECT CACHES ---
            self._notify_all_effect_caches(changed_coords, pieces_data)

            # --- STEP 5: FREEZE EFFECT (temporal) ---
            self.consolidated_aura_cache.trigger_freeze(game_state.color, game_state.turn_number)

            # --- STEP 6: INVALIDATE MOVE CACHE ---
            self.dependency_graph.notify_update('move_applied')

            # Calculate affected squares
            affected_squares = changed_coords.copy()
            if captured_piece:
                affected_squares = np.vstack([
                    affected_squares,
                    _get_adjacent_squares(mv[3:])
                ])

            # Mark affected pieces for BOTH colors
            opp_color = Color.WHITE if game_state.color == Color.BLACK else Color.BLACK
            self._invalidate_affected_piece_moves(affected_squares, game_state.color, game_state)
            self._invalidate_affected_piece_moves(affected_squares, opp_color, game_state)
# =============================================================================
# NUMPY-NATIVE DEPENDENCY GRAPH
# =============================================================================

class NumpyDependencyGraph:
    """Matrix-based dependency tracking - fully vectorized."""

    __slots__ = ("_dependencies", "_update_timestamps", "_manager", "_last_update")

    def __init__(self, manager: OptimizedCacheManager):
        self._manager = manager

        # Dependency matrix: which piece types affect which others
        self._dependencies = np.zeros((N_PIECE_TYPES + 1, N_PIECE_TYPES + 1), dtype=BOOL_DTYPE)

        # Last update timestamp per piece type
        self._update_timestamps = np.zeros(N_PIECE_TYPES + 1, dtype=INDEX_DTYPE)

        # Global last update counter
        self._last_update = 0

    def notify_update(self, event_type: str) -> None:
        """Vectorized dependency invalidation."""
        self._last_update = self._manager._move_counter

        # Mark affected piece types based on event
        if event_type == 'move_applied':
            # Move affects: straight sliders, diagonals, kings, aura pieces
            affected = np.array([4, 5, 6, 7, 22, 23, 39], dtype=PIECE_TYPE_DTYPE)
            self._update_timestamps[affected] = self._last_update

    def get_affected_pieces_vectorized(self, game_state, color: int) -> np.ndarray:
        """Get coordinates of pieces with outdated dependencies."""
        # Get all pieces of color
        all_coords = self._manager.occupancy_cache.get_positions(color)

        if all_coords.size == 0:
            return np.empty((0, 3), dtype=COORD_DTYPE)

        # Get their types
        colors, piece_types = self._manager.occupancy_cache.batch_get_attributes(all_coords)

        # Check if any dependencies are outdated
        outdated_mask = self._update_timestamps[piece_types] < self._last_update

        return all_coords[outdated_mask]

__all__ = ['OptimizedCacheManager', 'get_cache_manager']
