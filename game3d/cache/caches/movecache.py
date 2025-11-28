"""Move Cache - MULTI-LEVEL CACHING SYSTEM.

This module caches moves at THREE levels:
1. Piece-level RAW moves (pseudolegal) - individual piece moves before validation
2. Color-level RAW moves (pseudolegal) - all pieces' moves before filtering  
3. Color-level LEGAL moves - final moves after all filtering (frozen, hive, king capture, safe)

Caching Strategy:
- Raw moves cached per piece for incremental updates
- Raw moves also cached per color for fast regeneration
- Legal moves cached per color (most commonly accessed)
"""

import numpy as np
from numba import njit
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
import threading
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

from game3d.common.shared_types import (
    COORD_DTYPE, INDEX_DTYPE, BOOL_DTYPE, VOLUME, MOVE_DTYPE as MOVE_DTYPE,
    Color, SIZE
)

@njit(cache=True)
def _find_move_by_key(keys: np.ndarray, target_key: int) -> int:
    mask = keys == target_key
    matches = np.where(mask)[0]
    return matches[0] if matches.size > 0 else -1

@dataclass
class MoveCacheConfig:
    """Configuration for move cache."""
    max_cache_size: int = 10000
    enable_transposition_table: bool = True

class MoveCache:
    """
    Multi-level caching layer for moves.

    THREE cache levels:
    1. Piece-level RAW (pseudolegal) moves - stored in _piece_moves_cache
    2. Color-level RAW (pseudolegal) moves - stored in _raw_moves_cache
    3. Color-level LEGAL moves - stored in _legal_moves_cache

    Responsibilities:
    1. Cache moves at different stages of generation pipeline
    2. Invalidate cache when board state changes
    3. Track cache statistics for each level

    Does NOT:
    - Generate moves (that's pseudolegal.py and generator.py)
    - Validate moves (that's generator.py)
    - Filter moves (that's turnmove.py)
    - Apply moves (that's gamestate.py)
    """
    def __init__(self, cache_manager, config=None):
        self.cache_manager = cache_manager
        self.config = config or MoveCacheConfig()
        self._lock = threading.RLock()

        # ✅ FIXED: Initialize to current board generation, not -1
        current_gen = getattr(cache_manager.board, 'generation', 0)
        
        # THREE separate caches for different move types:
        # 1. LEGAL MOVES (final, after all filtering)
        self._legal_moves_cache = [None, None]  # [White, Black]
        
        # 2. RAW MOVES (pseudolegal, before filtering)
        self._raw_moves_cache = [None, None]  # [White, Black]
        
        self._cache_generation = 0
        self._board_generation = current_gen  # Use current generation

        stats_dtype = np.dtype([
            ('legal_cache_hits', INDEX_DTYPE),
            ('legal_cache_misses', INDEX_DTYPE),
            ('raw_cache_hits', INDEX_DTYPE),
            ('raw_cache_misses', INDEX_DTYPE),
            ('piece_cache_hits', INDEX_DTYPE),
            ('piece_cache_misses', INDEX_DTYPE),
            ('total_moves_cached', INDEX_DTYPE)
        ])
        self._stats = np.zeros(1, dtype=stats_dtype)[0]
        self._affected_pieces = set()
        self._piece_board_generations = {}
        self._board_generation_per_color = [current_gen, current_gen]

        # Initialize missing attributes
        self._affected_coord_keys_list = []  # ✅ Use list for O(1) append
        self._affected_color_idx_list = []
        self._affected_coord_keys = np.empty(0, dtype=np.int64)
        self._affected_color_idx = np.empty(0, dtype=np.int8)
        
        # ✅ OPTIMIZED: Use OrderedDict for LRU cache (piece-level raw moves)
        self._piece_moves_cache = OrderedDict()

        # Reverse Move Map: Square Key -> Set of Piece Keys
        # Used for incremental updates to find pieces attacking a square
        self._reverse_map: Dict[int, set] = {}
        # Track which squares a piece targets to allow efficient removal
        self._piece_targets: Dict[tuple, set] = {}

        # ✅ NEW: LRU tracking and size limits to prevent memory explosion
        self._max_piece_entries = 1000
        self._prune_triggered = 0  # Statistics

    # =========================================================================
    # LEGAL MOVES CACHE (Final moves after ALL filtering)
    # =========================================================================
    
    def get_legal_moves(self, color: int) -> Optional[np.ndarray]:
        """Retrieve cached LEGAL moves (after filtering)."""
        with self._lock:
            color_idx = 0 if color == Color.WHITE else 1

            if self._legal_moves_cache[color_idx] is None:
                self._stats['legal_cache_misses'] += 1
                return None

            # Check if affected pieces need regeneration
            affected = self.get_affected_pieces(color)

            if affected.size > 0:
                self._stats['legal_cache_misses'] += 1
                return None

            # Cache hit
            self._stats['legal_cache_hits'] += 1
            return self._legal_moves_cache[color_idx]
    
    def store_legal_moves(self, color: int, moves: np.ndarray) -> None:
        """Store LEGAL moves (after filtering)."""
        with self._lock:
            color_idx = 0 if color == Color.WHITE else 1

            # Validate non-empty unless truly stalemate
            if moves.size == 0:
                piece_count = len(self.cache_manager.occupancy_cache.get_positions(color))
                if piece_count > 0:
                    logger.warning(
                        f"Storing empty legal moves for {piece_count} pieces - possible stalemate"
                    )

            self._legal_moves_cache[color_idx] = moves.copy() if moves.size > 0 else moves
            current_gen = getattr(self.cache_manager.board, 'generation', 0)
            self._board_generation_per_color[color_idx] = current_gen
            self._cache_generation += 1
            self._stats['total_moves_cached'] += len(moves)
    
    def invalidate_legal_moves(self, color: Optional[int] = None) -> None:
        """Invalidate legal moves cache for one or both colors."""
        with self._lock:
            if color is None:
                self._legal_moves_cache[0] = None
                self._legal_moves_cache[1] = None
            else:
                color_idx = 0 if color == Color.WHITE else 1
                self._legal_moves_cache[color_idx] = None
            self._cache_generation += 1
    
    # =========================================================================
    # RAW MOVES CACHE (Pseudolegal moves before filtering)
    # =========================================================================
    
    def get_raw_moves(self, color: int) -> Optional[np.ndarray]:
        """Retrieve cached RAW (pseudolegal) moves before filtering."""
        with self._lock:
            color_idx = 0 if color == Color.WHITE else 1

            if self._raw_moves_cache[color_idx] is None:
                self._stats['raw_cache_misses'] += 1
                return None

            # Check if affected pieces need regeneration
            affected = self.get_affected_pieces(color)

            if affected.size > 0:
                self._stats['raw_cache_misses'] += 1
                return None

            # Cache hit
            self._stats['raw_cache_hits'] += 1
            return self._raw_moves_cache[color_idx]
    
    def store_raw_moves(self, color: int, moves: np.ndarray) -> None:
        """Store RAW (pseudolegal) moves before filtering."""
        with self._lock:
            color_idx = 0 if color == Color.WHITE else 1
            self._raw_moves_cache[color_idx] = moves.copy() if moves.size > 0 else moves
    
    def invalidate_raw_moves(self, color: Optional[int] = None) -> None:
        """Invalidate raw moves cache for one or both colors."""
        with self._lock:
            if color is None:
                self._raw_moves_cache[0] = None
                self._raw_moves_cache[1] = None
            else:
                color_idx = 0 if color == Color.WHITE else 1
                self._raw_moves_cache[color_idx] = None
    
    # =========================================================================
    # LEGACY get_cached_moves / store_moves (backwards compatibility)
    # These now map to LEGAL moves cache
    # =========================================================================

    def get_cached_moves(self, color: int) -> Optional[np.ndarray]:
        """LEGACY: Retrieve cached moves (maps to legal moves).
        
        DEPRECATED: Use get_legal_moves() or get_raw_moves() instead.
        """
        return self.get_legal_moves(color)

    def store_moves(self, color: int, moves: np.ndarray) -> None:
        """LEGACY: Store moves (maps to legal moves).
        
        DEPRECATED: Use store_legal_moves() or store_raw_moves() instead.
        """
        return self.store_legal_moves(color, moves)

    def invalidate(self) -> None:
        """Invalidate ALL caches (legal, raw, and piece-level)."""
        with self._lock:
            # Invalidate legal moves
            self._legal_moves_cache[0] = None
            self._legal_moves_cache[1] = None
            # Invalidate raw moves  
            self._raw_moves_cache[0] = None
            self._raw_moves_cache[1] = None
            # Increment generation
            self._cache_generation += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics for all cache levels."""
        with self._lock:
            # Legal cache stats
            legal_lookups = self._stats['legal_cache_hits'] + self._stats['legal_cache_misses']
            legal_hit_rate = self._stats['legal_cache_hits'] / max(legal_lookups, 1)
            
            # Raw cache stats
            raw_lookups = self._stats['raw_cache_hits'] + self._stats['raw_cache_misses']
            raw_hit_rate = self._stats['raw_cache_hits'] / max(raw_lookups, 1)
            
            # Piece cache stats
            piece_lookups = self._stats['piece_cache_hits'] + self._stats['piece_cache_misses']
            piece_hit_rate = self._stats['piece_cache_hits'] / max(piece_lookups, 1)

            # Count cached moves
            white_legal = 0 if self._legal_moves_cache[0] is None else len(self._legal_moves_cache[0])
            black_legal = 0 if self._legal_moves_cache[1] is None else len(self._legal_moves_cache[1])
            white_raw = 0 if self._raw_moves_cache[0] is None else len(self._raw_moves_cache[0])
            black_raw = 0 if self._raw_moves_cache[1] is None else len(self._raw_moves_cache[1])

            return {
                # Legal moves cache
                'legal_cache_hits': self._stats['legal_cache_hits'],
                'legal_cache_misses': self._stats['legal_cache_misses'],
                'legal_hit_rate': legal_hit_rate,
                'white_legal_moves': white_legal,
                'black_legal_moves': black_legal,
                # Raw moves cache
                'raw_cache_hits': self._stats['raw_cache_hits'],
                'raw_cache_misses': self._stats['raw_cache_misses'],
                'raw_hit_rate': raw_hit_rate,
                'white_raw_moves': white_raw,
                'black_raw_moves': black_raw,
                # Piece cache
                'piece_cache_hits': self._stats['piece_cache_hits'],
                'piece_cache_misses': self._stats['piece_cache_misses'],
                'piece_hit_rate': piece_hit_rate,
                'piece_moves_cache_size': len(self._piece_moves_cache),
                # Other
                'total_moves_cached': self._stats['total_moves_cached'],
                'reverse_map_size': len(self._reverse_map),
                'prune_operations': self._prune_triggered,
                # BACKWARDS COMPATIBILITY: Legacy keys for old code
                'cache_hits': self._stats['legal_cache_hits'],  # Map to legal cache
                'cache_misses': self._stats['legal_cache_misses'],
                'hit_rate': legal_hit_rate,
                'white_moves_cached': white_legal,
                'black_moves_cached': black_legal,
            }

    def clear(self) -> None:
        """Clear all cached data at all levels."""
        with self._lock:
            self.invalidate()
            # Clear all stats
            self._stats['legal_cache_hits'] = 0
            self._stats['legal_cache_misses'] = 0
            self._stats['raw_cache_hits'] = 0
            self._stats['raw_cache_misses'] = 0
            self._stats['piece_cache_hits'] = 0
            self._stats['piece_cache_misses'] = 0
            self._stats['total_moves_cached'] = 0
            # Clear piece caches
            self._piece_moves_cache.clear()
            self._reverse_map.clear()
            self._piece_targets.clear()
            self._affected_coord_keys = np.empty(0, dtype=np.int64)
            self._affected_color_idx = np.empty(0, dtype=np.int8)
            self._affected_coord_keys_list = []
            self._affected_color_idx_list = []


    # =========================================================================
    # PIECE-LEVEL CACHE (Raw moves per piece for incremental updates)
    # =========================================================================

    def mark_piece_invalid(self, color: int, coord_key: Union[int, bytes]) -> None:
        """Mark piece for regeneration - USE INTEGER KEYS."""
        color_idx = 0 if color == Color.WHITE else 1

        if isinstance(coord_key, (int, np.integer)):
            int_key = int(coord_key)
        else:
            int_key = int.from_bytes(coord_key, 'little') if coord_key else 0

        with self._lock:
            # ✅ OPTIMIZED: Use list append instead of np.append
            self._affected_coord_keys_list.append(int_key)
            self._affected_color_idx_list.append(color_idx)

    def has_piece_moves(self, color: int, coord_key: Union[int, bytes]) -> bool:
        """Check if piece moves are cached."""
        color_idx = 0 if color == Color.WHITE else 1

        if isinstance(coord_key, (int, np.integer)):
            int_key = int(coord_key)
        else:
            int_key = int.from_bytes(coord_key, 'little') if coord_key else 0

        with self._lock:
            piece_id = (color_idx, int_key)
            if piece_id in self._piece_moves_cache:
                # Move to end to mark as recently used
                self._piece_moves_cache.move_to_end(piece_id)
                self._stats['piece_cache_hits'] += 1
                return True
            self._stats['piece_cache_misses'] += 1
            return False

    def get_piece_moves(self, color: int, coord_key: Union[int, bytes]) -> np.ndarray:
        """Retrieve cached moves for a piece."""
        color_idx = 0 if color == Color.WHITE else 1

        if isinstance(coord_key, (int, np.integer)):
            int_key = int(coord_key)
        else:
            int_key = int.from_bytes(coord_key, 'little') if coord_key else 0
        
        piece_id = (color_idx, int_key)
        
        with self._lock:
            if piece_id in self._piece_moves_cache:
                # Move to end to mark as recently used
                self._piece_moves_cache.move_to_end(piece_id)
                return self._piece_moves_cache[piece_id]
            
            return np.empty((0, 6), dtype=MOVE_DTYPE)

    def store_piece_moves(self, color: int, coord_key: Union[int, bytes], moves: np.ndarray) -> None:
        """Cache moves for a specific piece - USE INTEGER KEYS."""
        color_idx = 0 if color == Color.WHITE else 1

        # Handle both int and bytes keys
        if isinstance(coord_key, (int, np.integer)):
            int_key = int(coord_key)
        else:
            int_key = int.from_bytes(coord_key, 'little') if coord_key else 0

        piece_id = (color_idx, int_key)

        with self._lock:
            # ✅ NEW: Prune cache if exceeding limit
            if len(self._piece_moves_cache) >= self._max_piece_entries:
                self._prune_piece_cache()

            # Update Reverse Map
            self._update_reverse_map(piece_id, moves)

            # Update dictionary cache (primary storage)
            # OrderedDict handles insertion order automatically
            self._piece_moves_cache[piece_id] = moves
            # Ensure it's marked as most recently used (if it was already there)
            self._piece_moves_cache.move_to_end(piece_id)

    def _update_reverse_map(self, piece_id: tuple, moves: np.ndarray) -> None:
        """Update the reverse map for a piece."""
        # 1. Clear old entries
        if piece_id in self._piece_targets:
            old_targets = self._piece_targets[piece_id]
            for target_key in old_targets:
                if target_key in self._reverse_map:
                    self._reverse_map[target_key].discard(piece_id)
                    if not self._reverse_map[target_key]:
                        del self._reverse_map[target_key]

        # 2. Add new entries
        if moves.size > 0:
            # Vectorized key generation for targets (x, y, z) -> columns 3, 4, 5
            to_x = moves[:, 3].astype(np.int64)
            to_y = moves[:, 4].astype(np.int64)
            to_z = moves[:, 5].astype(np.int64)
            # Simple packing: x | (y << 9) | (z << 18)
            target_keys = to_x | (to_y << 9) | (to_z << 18)

            unique_targets = np.unique(target_keys)

            new_targets = set()
            for t_key in unique_targets:
                t_key_int = int(t_key)
                if t_key_int not in self._reverse_map:
                    self._reverse_map[t_key_int] = set()
                self._reverse_map[t_key_int].add(piece_id)
                new_targets.add(t_key_int)

            self._piece_targets[piece_id] = new_targets
        else:
            self._piece_targets[piece_id] = set()

    def get_pieces_targeting(self, coord_keys: np.ndarray) -> list:
        """Get all pieces targeting the given coordinates. Returns list of (color_idx, piece_key)."""
        affected_pieces = set()
        with self._lock:
            for key in coord_keys:
                key_int = int(key)
                if key_int in self._reverse_map:
                    affected_pieces.update(self._reverse_map[key_int])
        return list(affected_pieces)

    def get_affected_pieces(self, color: int) -> np.ndarray:
        """Get affected pieces as numpy array."""
        color_idx = 0 if color == Color.WHITE else 1

        # ✅ OPTIMIZED: Convert lists to arrays only when reading
        if self._affected_coord_keys_list:
            # Consolidate lists into arrays once
            self._affected_coord_keys = np.array(self._affected_coord_keys_list, dtype=np.int64)
            self._affected_color_idx = np.array(self._affected_color_idx_list, dtype=np.int8)
            # Don't clear lists yet - they may be used again soon

        mask = self._affected_color_idx == color_idx
        return self._affected_coord_keys[mask]

    def clear_affected_pieces(self, color: int) -> None:
        """Clear affected pieces after regeneration."""
        color_idx = 0 if color == Color.WHITE else 1
        self._affected_pieces = {(c_idx, key) for (c_idx, key) in self._affected_pieces
                                if c_idx != color_idx}

        # ✅ OPTIMIZED: Clear both arrays and lists
        mask = self._affected_color_idx != color_idx
        self._affected_coord_keys = self._affected_coord_keys[mask]
        self._affected_color_idx = self._affected_color_idx[mask]
        
        # Clear lists for the specified color
        indices_to_keep = [i for i, c_idx in enumerate(self._affected_color_idx_list) if c_idx != color_idx]
        self._affected_coord_keys_list = [self._affected_coord_keys_list[i] for i in indices_to_keep]
        self._affected_color_idx_list = [self._affected_color_idx_list[i] for i in indices_to_keep]

    # ✅ NEW: LRU PRUNING METHOD
    def _prune_piece_cache(self) -> None:
        """Prune oldest 20% of entries when cache exceeds limit."""
        if len(self._piece_moves_cache) <= self._max_piece_entries:
            return

        prune_start = len(self._piece_moves_cache)
        prune_target = int(self._max_piece_entries * 0.8)  # Keep 80%
        num_to_remove = prune_start - prune_target

        for _ in range(num_to_remove):
            # Remove oldest entry (first in OrderedDict)
            try:
                piece_id, _ = self._piece_moves_cache.popitem(last=False)
                
                # Clean up reverse map
                if piece_id in self._piece_targets:
                    for target_key in self._piece_targets[piece_id]:
                        if target_key in self._reverse_map:
                            self._reverse_map[target_key].discard(piece_id)
                            if not self._reverse_map[target_key]:
                                del self._reverse_map[target_key]
                    del self._piece_targets[piece_id]
            except KeyError:
                break

        prune_end = len(self._piece_moves_cache)
        self._prune_triggered += 1
        logger.debug(f"Pruned piece cache: {prune_start} -> {prune_end} entries")

def create_move_cache(cache_manager, config: Optional[MoveCacheConfig] = None) -> MoveCache:
    """Factory function to create move cache."""
    return MoveCache(cache_manager, config)

__all__ = ['MoveCache', 'MoveCacheConfig', 'create_move_cache']
