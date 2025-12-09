# gamestate.py
"""Fully optimized numpy-native game state with complete vectorization - refactored to use common modules.

This module provides complete numpy/numba operations using centralized utilities from game3d.common.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple
from numba import njit, prange
from collections import defaultdict, deque
import logging
logger = logging.getLogger(__name__)

from game3d.common.shared_types import (
    COORD_DTYPE, INDEX_DTYPE, BOOL_DTYPE, COLOR_DTYPE, PIECE_TYPE_DTYPE, FLOAT_DTYPE,
    BATCH_COORD_DTYPE, SIZE, VOLUME, N_TOTAL_PLANES, N_PIECE_TYPES, HASH_DTYPE,
    COLOR_WHITE, COLOR_BLACK, COLOR_EMPTY,
    MOVE_DTYPE as MOVE_DTYPE, MAX_HISTORY_SIZE, Color
)
from game3d.common.coord_utils import CoordinateUtils, in_bounds_vectorized
from game3d.common.state_utils import create_new_state
from game3d.cache.manager import OptimizedCacheManager

# Precompute deterministic Zobrist table - unified via cache_manager, but fallback here
np.random.seed(42)  # For reproducibility
ZOBRIST_TABLE = np.random.randint(0, 2**64, (N_PIECE_TYPES * 2, SIZE, SIZE, SIZE), dtype=np.uint64)

# =============================================================================
# OPTIMIZED GAME STATE
# =============================================================================

class GameState:
    """Fully numpy-native game state with complete vectorization."""
    __slots__ = (
        'board', 'color', 'cache_manager', 'history', 'halfmove_clock',
        'turn_number', '_zkey', '_position_keys', '_position_counts', '_legal_moves_cache',
        '_legal_moves_cache_key', '_metrics', '_cache_key_multipliers', '_undo_info',
        '_pending_hive_moves', '_moved_hive_positions', '_terminal_logged'
    )

    def __init__(self, board, color: int = COLOR_BLACK, cache_manager: Optional[OptimizedCacheManager] = None,
                history: Union[deque, np.ndarray, list] = None, halfmove_clock: int = 0, turn_number: int = 1):
        """Initialize with proper cache handling."""
        if board is None:
            raise ValueError("Board cannot be None")

        self.board = board
        self.color = color

        # --- CRITICAL: Set up cache manager WITHOUT triggering rebuild ---
        if cache_manager is None:
            # Create new manager (will rebuild during __init__)
            self.cache_manager = OptimizedCacheManager(board, color)
        else:
            # Reuse existing manager - DO NOT modify it
            # The cache should already be in sync with the board
            self.cache_manager = cache_manager

        # Ensure bidirectional link
        if self.board._cache_manager is not self.cache_manager:
            logger.warning("Board cache_manager mismatch - syncing")
            self.board._cache_manager = self.cache_manager

        # Retrieve pre-computed Zobrist (no recomputation)
        self._zkey = self.cache_manager._zkey

        # Initialize game state arrays
        if history is None:
            self.history = deque(maxlen=MAX_HISTORY_SIZE)
        elif isinstance(history, deque):
            self.history = history
        else:
            self.history = deque(history, maxlen=MAX_HISTORY_SIZE)

        self.halfmove_clock = halfmove_clock
        self.turn_number = turn_number

        # Move cache
        self._legal_moves_cache = None
        self._legal_moves_cache_key = 0

        # Position repetition tracking
        self._position_keys = np.array([self._zkey], dtype=HASH_DTYPE)
        self._position_counts = np.array([1], dtype=INDEX_DTYPE)

        # Performance metrics
        self._metrics = PerformanceMetrics()

        # Cache key generation
        PRIME1, PRIME2, PRIME3 = 1000003, 1000033, 1000037
        self._cache_key_multipliers = np.array([PRIME1, PRIME2, PRIME3], dtype=np.uint64)

        # Undo stack
        self._undo_info = []

        # Multi-hive move tracking
        self._pending_hive_moves = []
        self._moved_hive_positions = set()
        
        # Terminal state logging flag
        self._terminal_logged = False

    @classmethod
    def from_startpos(cls) -> 'GameState':
        """Factory method to create game state from standard starting position."""
        from game3d.board.board import Board
        board = Board.startpos()
        color = COLOR_WHITE
        cache_manager = OptimizedCacheManager(board, color)
        return cls(board=board, color=color, cache_manager=cache_manager)

    def gen_moves(self) -> np.ndarray:
        """Generate legal moves for current state (compatibility alias)."""
        from game3d.movement.generator import generate_legal_moves
        return generate_legal_moves(self)

    def reset(self, start_state: Optional['GameState'] = None) -> None:
        """Reset game state to initial or specified state."""
        if start_state:
            # Copy from provided state
            self.board = start_state.board.copy()
            self.color = start_state.color
            self.cache_manager = start_state.cache_manager
            self.history = deque(list(start_state.history), maxlen=MAX_HISTORY_SIZE)
            self.halfmove_clock = start_state.halfmove_clock
            self.turn_number = start_state.turn_number
            self._zkey = start_state._zkey
            self._position_keys = start_state._position_keys.copy()
            self._position_counts = start_state._position_counts.copy()
            self._undo_info = start_state._undo_info.copy()
            self._pending_hive_moves = start_state._pending_hive_moves.copy()
            self._moved_hive_positions = start_state._moved_hive_positions.copy()
            self._terminal_logged = getattr(start_state, '_terminal_logged', False)
        else:
            # Reset to start position
            from game3d.board.board import Board
            board = Board.startpos()
            self.board = board
            self.color = COLOR_WHITE
            self.cache_manager = OptimizedCacheManager(board, COLOR_WHITE)
            self.history = deque(maxlen=MAX_HISTORY_SIZE)
            self.halfmove_clock = 0
            self.turn_number = 1
            self._zkey = self.cache_manager._zkey
            self._position_keys = np.array([self._zkey], dtype=HASH_DTYPE)
            self._position_counts = np.array([1], dtype=INDEX_DTYPE)
            self._undo_info = []
            self._pending_hive_moves = []
            self._moved_hive_positions = set()
            self._terminal_logged = False

        self._clear_caches()

    @property
    def zkey(self) -> int:
        """Get Zobrist key."""
        return self._zkey

    @property
    def history_array(self) -> np.ndarray:
        """Get history as numpy array (for backward compatibility)."""
        if not self.history:
            return np.empty(0, dtype=MOVE_DTYPE)
        return np.array(self.history, dtype=MOVE_DTYPE)

    def _get_cache_key(self, prefix_id: int, *params) -> int:
        """O(1) cache key generation"""
        key = prefix_id
        for i, p in enumerate(params):
            key ^= (hash(p) * self._cache_key_multipliers[i % 3])
        return key

    def _update_position_counts(self, zkey: int, increment: int = 1) -> None:
        """Update position counts using array-based method."""
        idx = np.searchsorted(self._position_keys, zkey)
        if idx < self._position_keys.size and self._position_keys[idx] == zkey:
            self._position_counts[idx] += increment
            if self._position_counts[idx] <= 0:
                self._position_keys = np.delete(self._position_keys, idx)
                self._position_counts = np.delete(self._position_counts, idx)
        else:
            # Insert sorted
            self._position_keys = np.insert(self._position_keys, idx, zkey)
            self._position_counts = np.insert(self._position_counts, idx, increment)

    def _clear_caches(self) -> None:
        """Clear local state caches."""
        self._legal_moves_cache = None
        self._legal_moves_cache_key = 0

    # =============================================================================
    # MULTI-HIVE MOVE TRACKING
    # =============================================================================
    @property
    def is_in_hive_multi_move_state(self) -> bool:
        """Check if currently in multi-hive move mode (first hive moved but turn not ended)."""
        return len(self._pending_hive_moves) > 0

    def has_hive_moved(self, pos: np.ndarray) -> bool:
        """Check if a hive at the given position has already moved this turn."""
        pos_tuple = tuple(pos.flatten().tolist())
        return pos_tuple in self._moved_hive_positions

    def get_unmoved_hives(self) -> List[np.ndarray]:
        """Get list of hive positions that haven't moved yet this turn."""
        from game3d.common.shared_types import PieceType

        # ✅ OPTIMIZED: Vectorized Hive movement tracking
        # 1. Get all hive positions
        hive_coords = self.cache_manager.occupancy_cache.get_positions(self.color, PieceType.HIVE)
        if hive_coords.size == 0:
            return np.empty((0, 3), dtype=COORD_DTYPE)
            
        # 2. Filter out moved hives
        if not self._moved_hive_positions:
            return hive_coords
            
        # Vectorized filter using set lookup via list comprehension (fastest for small N)
        is_moved = np.array([tuple(c) in self._moved_hive_positions for c in hive_coords], dtype=bool)
        
        return hive_coords[~is_moved]

    def clear_hive_move_tracking(self) -> None:
        """Clear hive move tracking (called when turn ends)."""
        self._pending_hive_moves.clear()
        self._moved_hive_positions.clear()

    # =============================================================================
    # VECTORIZED MOVE OPERATIONS - REFACTORED TO USE GENERATOR
    # =============================================================================
    @property
    def legal_moves(self) -> np.ndarray:
        """
        Get all legal moves for current position.
        DELEGATES to turnmove.legal_moves() - single source of truth.
        """
        from game3d.game import turnmove
        return turnmove.legal_moves(self)

    def make_move_vectorized(self, move: np.ndarray) -> Optional['GameState']:
        """Make a single move - DELEGATES to turnmove.make_move()."""
        from game3d.game import turnmove

        # Ensure move is in correct format (flatten if needed)
        if move.ndim == 2:
            move = move.flatten()

        return turnmove.make_move(self, move)

    def make_multiple_moves_vectorized(self, moves: np.ndarray) -> Optional['GameState']:
        """Make multiple moves in sequence - DELEGATES to turnmove.make_move()."""
        if moves.shape[0] == 0:
            return self

        from game3d.game import turnmove

        # Chain moves through turnmove.make_move()
        current_state = self
        for move in moves:
            current_state = turnmove.make_move(current_state, move)
            if current_state is None:
                raise RuntimeError(f"Move execution failed in sequence")

        return current_state


    def _switch_turn(self) -> 'GameState':
        """
        Switch turn between players.

        Board is already updated, just need to flip color and increment turn.
        """
        from game3d.common.state_utils import create_new_state

        return create_new_state(
            original_state=self,
            new_board=self.board,  # Board already updated
            new_color=Color(self.color).opposite(),
            move=None,
            increment_turn=True,
            reuse_cache=True  # Reuse cache, don't rebuild
        )

    # =============================================================================
    # GAME ANALYSIS OPERATIONS - REFACTORED TO USE COMMON MODULES
    # =============================================================================
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        moves = self.legal_moves  # Use property

        return {
            'legal_moves_count': moves.shape[0],
            'move_generation_time': getattr(self._metrics, 'move_gen_time', 0.0),
            'cache_hit_rate': getattr(self._metrics, 'cache_hit_rate', 0.0),
            'vectorization_efficiency': min(1.0, moves.shape[0] / 100.0),
            'memory_usage': self.board.array().nbytes / (1024 * 1024)  # MB
        }

    def is_game_over(self) -> bool:
        """Check if game is over (delegated to terminal module)."""
        from game3d.game.terminal import is_game_over as terminal_is_game_over
        return terminal_is_game_over(self)

    def result(self) -> Optional[int]:
        """Get game result (delegated to terminal module)."""
        from game3d.game.terminal import result as terminal_result
        return terminal_result(self)

    # =============================================================================
    # CLONE AND SERIALIZATION
    # =============================================================================
    def clone(self, deep_cache: bool = False) -> 'GameState':
        """
        Create a deep copy with properly synchronized cache.

        CRITICAL: Rebuild cache from current occupancy data to ensure consistency.
        """
        # 1. Create a fresh board configuration (stateless)
        new_board = self.board.copy()

        # 2. Export current state from OccupancyCache (Single Source of Truth)
        current_state_data = self.cache_manager.occupancy_cache.export_state()

        # 3. Create fresh cache manager initialized with current state data
        new_cache = OptimizedCacheManager(
            board=new_board, 
            color=self.color,
            initial_data=current_state_data
        )

        # 4. Create new game state
        new_state = GameState(
            board=new_board,
            color=self.color,
            cache_manager=new_cache,
            history=list(self.history),  # Copy history
            halfmove_clock=self.halfmove_clock,
            turn_number=self.turn_number
        )

        # 5. Copy move history
        # new_state._move_history was legacy code, removed.

        # 6. CRITICAL: Zobrist was recomputed during cache rebuild
        new_state._zkey = new_cache._zkey

        # 7. Copy hive tracking state
        new_state._pending_hive_moves = self._pending_hive_moves.copy()
        new_state._moved_hive_positions = self._moved_hive_positions.copy()
        new_state._terminal_logged = self._terminal_logged

        return new_state

    def get_state_vector(self) -> np.ndarray:
        """
        Get flattened state vector for neural networks.
        
        ✅ OPTIMIZED: Direct sparse-to-dense construction.
        Avoids allocating intermediate (P, S, S, S) array and flattening it.
        """
        # 1. Initialize flat vector
        # Size = P * S^3
        vector_size = N_TOTAL_PLANES * VOLUME
        flat_vector = np.zeros(vector_size, dtype=FLOAT_DTYPE)
        
        # 2. Get sparse occupied positions
        # coords: (N, 3), types: (N,), colors: (N,)
        coords, types, colors = self.cache_manager.occupancy_cache.get_all_occupied_vectorized()
        
        if coords.shape[0] > 0:
            # 3. Calculate plane indices (P)
            # plane = (piece_type - 1) + (is_black * N_PIECE_TYPES)
            # Assumption: Color.WHITE is min value, and colors are sequential or we subtract base
            # In shared_types, Color.WHITE is usually 1, Color.BLACK is 2
            # Offset: White=0, Black=1 -> (color - Color.WHITE)
            color_offsets = (colors - int(Color.WHITE)).astype(INDEX_DTYPE)
            plane_indices = (types - 1) + (color_offsets * N_PIECE_TYPES)
            
            # 4. Calculate spatial indices (S*S*S)
            # flat_spatial = x * S*S + y * S + z
            # Note: coords are already in (x, y, z) order from occupancy cache
            x = coords[:, 0].astype(INDEX_DTYPE)
            y = coords[:, 1].astype(INDEX_DTYPE)
            z = coords[:, 2].astype(INDEX_DTYPE)
            flat_spatial = x * (SIZE * SIZE) + y * SIZE + z
            
            # 5. Calculate final indices into flat_vector
            # index = p * VOLUME + flat_spatial
            final_indices = plane_indices * VOLUME + flat_spatial
            
            # 6. Set values
            # Using flat indexing is much faster than tuple indexing on ndarray
            flat_vector[final_indices] = 1.0
            
        # 7. Add state info
        state_info = np.array([
            int(self.color), self.halfmove_clock, self.turn_number,
            self._zkey % 1000  # Normalize zkey
        ], dtype=FLOAT_DTYPE)
        
        return np.concatenate([flat_vector, state_info])

    def __str__(self) -> str:
        """String representation."""
        return (f"GameState(color={self.color}, turn={self.turn_number}, "
                f"moves={len(self.history)}, zkey={self._zkey})")

# =============================================================================
# PERFORMANCE METRICS CLASS
# =============================================================================

class PerformanceMetrics:
    """Performance tracking for game state operations."""

    __slots__ = ('legal_moves_calls', 'make_move_calls', 'move_gen_time', 'cache_hit_rate')

    def __init__(self):
        self.legal_moves_calls = 0
        self.make_move_calls = 0
        self.move_gen_time = 0.0
        self.cache_hit_rate = 0.0

    def __repr__(self):
        return f"PerformanceMetrics(calls={self.legal_moves_calls + self.make_move_calls})"

# Module exports
__all__ = ['GameState', 'PerformanceMetrics']
