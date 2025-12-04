"""Move Cache - MULTI-LEVEL CACHING SYSTEM.

This module caches moves at FOUR levels:
1. Piece-level RAW moves (geometric, ignore occupancy) - individual piece moves
2. Piece-level PSEUDOLEGAL moves (geometric + occupancy) - individual piece moves
3. Color-level RAW moves (geometric, ignore occupancy) - all pieces' moves
4. Color-level PSEUDOLEGAL moves (geometric + occupancy) - all pieces' moves
5. Color-level LEGAL moves - final moves after all filtering (frozen, hive, king capture, safe)

Caching Strategy:
- Raw moves cached per piece for incremental updates
- Pseudolegal moves cached per piece for incremental updates
- Raw/Pseudolegal moves also cached per color for fast regeneration
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

    Cache levels:
    1. Piece-level moves - stored in _piece_moves_cache
    2. Color-level RAW moves (ignore occupancy) - stored in _raw_moves_cache
    3. Color-level PSEUDOLEGAL moves (respect occupancy) - stored in _pseudolegal_moves_cache
    4. Color-level LEGAL moves - stored in _legal_moves_cache

    Responsibilities:
    1. Cache moves at different stages of generation pipeline
    2. Invalidate cache when board state changes
    3. Track cache statistics for each level
    """
    def __init__(self, cache_manager, config=None):
        self.cache_manager = cache_manager
        self.config = config or MoveCacheConfig()
        self._lock = threading.RLock()

        # Initialize to current board generation
        current_gen = getattr(cache_manager.board, 'generation', 0)
        
        # 1. LEGAL MOVES (final, after all filtering)
        self._legal_moves_cache = [None, None]  # [White, Black]
        
        # 2. PSEUDOLEGAL MOVES (respect occupancy, before filtering)
        self._pseudolegal_moves_cache = [None, None]  # [White, Black]

        # 3. RAW MOVES (ignore occupancy)
        self._raw_moves_cache = [None, None]  # [White, Black]
        
        self._cache_generation = 0
        self._board_generation = current_gen

        stats_dtype = np.dtype([
            ('legal_cache_hits', INDEX_DTYPE),
            ('legal_cache_misses', INDEX_DTYPE),
            ('pseudolegal_cache_hits', INDEX_DTYPE),
            ('pseudolegal_cache_misses', INDEX_DTYPE),
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
        self._affected_coord_keys_list = []
        self._affected_color_idx_list = []
        self._affected_coord_keys = np.empty(0, dtype=np.int64)
        self._affected_color_idx = np.empty(0, dtype=np.int8)
        
        # Piece-level cache (stores pseudolegal moves by default)
        self._piece_moves_cache = OrderedDict()
        # Piece-level RAW cache (stores raw moves for incremental updates)
        self._piece_raw_moves_cache = OrderedDict()

        # Reverse Move Map: Square Key -> Set of Piece Keys
        self._reverse_map: Dict[int, set] = {}
        # Track which squares a piece targets
        self._piece_targets: Dict[tuple, set] = {}

        self._max_piece_entries = 1000
        self._prune_triggered = 0

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

            if moves.size == 0:
                piece_count = len(self.cache_manager.occupancy_cache.get_positions(color))
                if piece_count > 0:
                    priest_count = self.cache_manager.occupancy_cache.get_priest_count(color)
                    
                    # Check if it's actually Checkmate
                    from game3d.attacks.check import king_in_check
                    is_check = king_in_check(self.cache_manager.board, Color(color).opposite(), color, self.cache_manager)
                    
                    if is_check:
                        logger.info(f"Checkmate detected for {Color(color).name} (Priests: {priest_count})")
                    else:
                        logger.warning(
                            f"Stalemate detected for {Color(color).name} - {piece_count} pieces (Priests: {priest_count}) - 0 legal moves"
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
    # PSEUDOLEGAL MOVES CACHE (Respect occupancy, before filtering)
    # =========================================================================
    
    def get_pseudolegal_moves(self, color: int) -> Optional[np.ndarray]:
        """Retrieve cached PSEUDOLEGAL moves (respect occupancy)."""
        with self._lock:
            color_idx = 0 if color == Color.WHITE else 1

            if self._pseudolegal_moves_cache[color_idx] is None:
                self._stats['pseudolegal_cache_misses'] += 1
                return None

            affected = self.get_affected_pieces(color)
            if affected.size > 0:
                self._stats['pseudolegal_cache_misses'] += 1
                return None

            self._stats['pseudolegal_cache_hits'] += 1
            return self._pseudolegal_moves_cache[color_idx]
    
    def store_pseudolegal_moves(self, color: int, moves: np.ndarray) -> None:
        """Store PSEUDOLEGAL moves (respect occupancy)."""
        with self._lock:
            color_idx = 0 if color == Color.WHITE else 1
            self._pseudolegal_moves_cache[color_idx] = moves.copy() if moves.size > 0 else moves
    
    def invalidate_pseudolegal_moves(self, color: Optional[int] = None) -> None:
        """Invalidate pseudolegal moves cache."""
        with self._lock:
            if color is None:
                self._pseudolegal_moves_cache[0] = None
                self._pseudolegal_moves_cache[1] = None
            else:
                color_idx = 0 if color == Color.WHITE else 1
                self._pseudolegal_moves_cache[color_idx] = None

    # =========================================================================
    # RAW MOVES CACHE (Ignore occupancy)
    # =========================================================================

    def get_raw_moves(self, color: int) -> Optional[np.ndarray]:
        """Retrieve cached RAW moves (ignore occupancy)."""
        with self._lock:
            color_idx = 0 if color == Color.WHITE else 1

            if self._raw_moves_cache[color_idx] is None:
                self._stats['raw_cache_misses'] += 1
                return None

            affected = self.get_affected_pieces(color)
            if affected.size > 0:
                self._stats['raw_cache_misses'] += 1
                return None

            self._stats['raw_cache_hits'] += 1
            return self._raw_moves_cache[color_idx]

    def store_raw_moves(self, color: int, moves: np.ndarray) -> None:
        """Store RAW moves (ignore occupancy)."""
        with self._lock:
            color_idx = 0 if color == Color.WHITE else 1
            self._raw_moves_cache[color_idx] = moves.copy() if moves.size > 0 else moves

    def invalidate_raw_moves(self, color: Optional[int] = None) -> None:
        """Invalidate raw moves cache."""
        with self._lock:
            if color is None:
                self._raw_moves_cache[0] = None
                self._raw_moves_cache[1] = None
            else:
                color_idx = 0 if color == Color.WHITE else 1
                self._raw_moves_cache[color_idx] = None

    # =========================================================================
    # LEGACY / COMPATIBILITY
    # =========================================================================

    def get_cached_moves(self, color: int) -> Optional[np.ndarray]:
        """LEGACY: Retrieve cached moves (maps to legal moves)."""
        return self.get_legal_moves(color)

    def store_moves(self, color: int, moves: np.ndarray) -> None:
        """LEGACY: Store moves (maps to legal moves)."""
        return self.store_legal_moves(color, moves)

    def invalidate(self) -> None:
        """Invalidate ALL caches."""
        with self._lock:
            print(f"DEBUG: MoveCache.invalidate called. ID: {id(self)}")
            self.invalidate_legal_moves()
            self.invalidate_pseudolegal_moves()
            self.invalidate_raw_moves()
            self._cache_generation += 1
            self._piece_moves_cache.clear()
            self._piece_raw_moves_cache.clear()
            self._reverse_map.clear()
            self._piece_targets.clear()
            print(f"DEBUG: MoveCache cleared. Piece cache size: {len(self._piece_moves_cache)}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics for all cache levels."""
        with self._lock:
            # Legal cache stats
            legal_lookups = self._stats['legal_cache_hits'] + self._stats['legal_cache_misses']
            legal_hit_rate = self._stats['legal_cache_hits'] / max(legal_lookups, 1)
            
            # Pseudolegal cache stats
            pseudo_lookups = self._stats['pseudolegal_cache_hits'] + self._stats['pseudolegal_cache_misses']
            pseudo_hit_rate = self._stats['pseudolegal_cache_hits'] / max(pseudo_lookups, 1)

            # Raw cache stats
            raw_lookups = self._stats['raw_cache_hits'] + self._stats['raw_cache_misses']
            raw_hit_rate = self._stats['raw_cache_hits'] / max(raw_lookups, 1)
            
            # Piece cache stats
            piece_lookups = self._stats['piece_cache_hits'] + self._stats['piece_cache_misses']
            piece_hit_rate = self._stats['piece_cache_hits'] / max(piece_lookups, 1)

            return {
                'legal_hit_rate': legal_hit_rate,
                'pseudolegal_hit_rate': pseudo_hit_rate,
                'raw_hit_rate': raw_hit_rate,
                'piece_hit_rate': piece_hit_rate,
                'total_moves_cached': self._stats['total_moves_cached'],
                'piece_moves_cache_size': len(self._piece_moves_cache),
                'prune_operations': self._prune_triggered,
                # Legacy keys
                'cache_hits': self._stats['legal_cache_hits'],
                'cache_misses': self._stats['legal_cache_misses'],
            }

    def clear(self) -> None:
        """Clear all cached data at all levels."""
        with self._lock:
            self.invalidate()
            self._stats.fill(0)
            self._piece_moves_cache.clear()
            self._piece_raw_moves_cache.clear()
            self._reverse_map.clear()
            self._piece_targets.clear()
            self._affected_coord_keys = np.empty(0, dtype=np.int64)
            self._affected_color_idx = np.empty(0, dtype=np.int8)
            self._affected_coord_keys_list = []
            self._affected_color_idx_list = []

    # =========================================================================
    # PIECE-LEVEL CACHE (Pseudolegal moves per piece for incremental updates)
    # =========================================================================

    def mark_piece_invalid(self, color: int, coord_key: Union[int, bytes]) -> None:
        """Mark piece for regeneration - USE INTEGER KEYS."""
        color_idx = 0 if color == Color.WHITE else 1

        if isinstance(coord_key, (int, np.integer)):
            int_key = int(coord_key)
        else:
            int_key = int.from_bytes(coord_key, 'little') if coord_key else 0

        with self._lock:
            self._affected_coord_keys_list.append(int_key)
            self._affected_color_idx_list.append(color_idx)

    def mark_pieces_invalid_batch(self, color_indices: np.ndarray, keys: np.ndarray) -> None:
        """Batch mark pieces for regeneration."""
        if keys.size == 0:
            return
            
        with self._lock:
            # Convert to list and extend
            # Assuming keys are already integers (from _coords_to_keys)
            self._affected_coord_keys_list.extend(keys.tolist())
            self._affected_color_idx_list.extend(color_indices.tolist())

    def invalidate_targeting_pieces(self, coord_keys: np.ndarray) -> None:
        """Invalidate all pieces targeting the given coordinates."""
        if coord_keys.size == 0:
            return

        with self._lock:
            # We must iterate because _reverse_map is a dict
            # But we avoid returning a list and looping again in manager
            for key in coord_keys:
                key_int = int(key)
                if key_int in self._reverse_map:
                    for (c_idx, p_key) in self._reverse_map[key_int]:
                        self._affected_color_idx_list.append(c_idx)
                        self._affected_coord_keys_list.append(p_key)

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
                self._piece_moves_cache.move_to_end(piece_id)
                return self._piece_moves_cache[piece_id]
            
            return np.empty((0, 6), dtype=MOVE_DTYPE)

    def store_piece_moves(self, color: int, coord_key: Union[int, bytes], moves: np.ndarray) -> None:
        """Cache moves for a specific piece - USE INTEGER KEYS."""
        color_idx = 0 if color == Color.WHITE else 1

        if isinstance(coord_key, (int, np.integer)):
            int_key = int(coord_key)
        else:
            int_key = int.from_bytes(coord_key, 'little') if coord_key else 0

        piece_id = (color_idx, int_key)

        with self._lock:
            if len(self._piece_moves_cache) >= self._max_piece_entries:
                self._prune_piece_cache()

            self._update_reverse_map(piece_id, moves)
            self._piece_moves_cache[piece_id] = moves
            self._piece_moves_cache.move_to_end(piece_id)

    def has_piece_raw_moves(self, color: int, coord_key: Union[int, bytes]) -> bool:
        """Check if piece RAW moves are cached."""
        color_idx = 0 if color == Color.WHITE else 1
        if isinstance(coord_key, (int, np.integer)):
            int_key = int(coord_key)
        else:
            int_key = int.from_bytes(coord_key, 'little') if coord_key else 0
        
        piece_id = (color_idx, int_key)
        with self._lock:
            return piece_id in self._piece_raw_moves_cache

    def get_piece_raw_moves(self, color: int, coord_key: Union[int, bytes]) -> np.ndarray:
        """Retrieve cached RAW moves for a piece."""
        color_idx = 0 if color == Color.WHITE else 1
        if isinstance(coord_key, (int, np.integer)):
            int_key = int(coord_key)
        else:
            int_key = int.from_bytes(coord_key, 'little') if coord_key else 0
        
        piece_id = (color_idx, int_key)
        with self._lock:
            if piece_id in self._piece_raw_moves_cache:
                self._piece_raw_moves_cache.move_to_end(piece_id)
                return self._piece_raw_moves_cache[piece_id]
            return np.empty((0, 6), dtype=MOVE_DTYPE)

    def store_piece_raw_moves(self, color: int, coord_key: Union[int, bytes], moves: np.ndarray) -> None:
        """Cache RAW moves for a specific piece."""
        color_idx = 0 if color == Color.WHITE else 1
        if isinstance(coord_key, (int, np.integer)):
            int_key = int(coord_key)
        else:
            int_key = int.from_bytes(coord_key, 'little') if coord_key else 0

        piece_id = (color_idx, int_key)
        with self._lock:
            # Pruning logic shared or separate? Shared limit for simplicity?
            # For now, just add. Pruning might need to handle both.
            self._piece_raw_moves_cache[piece_id] = moves
            self._piece_raw_moves_cache.move_to_end(piece_id)

    def _update_reverse_map(self, piece_id: tuple, moves: np.ndarray) -> None:
        """Update the reverse map for a piece."""
        if piece_id in self._piece_targets:
            old_targets = self._piece_targets[piece_id]
            for target_key in old_targets:
                if target_key in self._reverse_map:
                    self._reverse_map[target_key].discard(piece_id)
                    if not self._reverse_map[target_key]:
                        del self._reverse_map[target_key]

        if moves.size > 0:
            to_x = moves[:, 3].astype(np.int64)
            to_y = moves[:, 4].astype(np.int64)
            to_z = moves[:, 5].astype(np.int64)
            target_keys = to_x | (to_y << 9) | (to_z << 18)

            unique_targets = np.unique(target_keys)

            new_targets = set()
            # Optimize: Convert to list to avoid numpy scalar overhead in loop
            for t_key in unique_targets.tolist():
                if t_key not in self._reverse_map:
                    self._reverse_map[t_key] = set()
                self._reverse_map[t_key].add(piece_id)
                new_targets.add(t_key)

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

        if self._affected_coord_keys_list:
            self._affected_coord_keys = np.array(self._affected_coord_keys_list, dtype=np.int64)
            self._affected_color_idx = np.array(self._affected_color_idx_list, dtype=np.int8)

        mask = self._affected_color_idx == color_idx
        return self._affected_coord_keys[mask]

    def clear_affected_pieces(self, color: int) -> None:
        """Clear affected pieces after regeneration."""
        color_idx = 0 if color == Color.WHITE else 1
        self._affected_pieces = {(c_idx, key) for (c_idx, key) in self._affected_pieces
                                if c_idx != color_idx}

        mask = self._affected_color_idx != color_idx
        self._affected_coord_keys = self._affected_coord_keys[mask]
        self._affected_color_idx = self._affected_color_idx[mask]
        
        indices_to_keep = [i for i, c_idx in enumerate(self._affected_color_idx_list) if c_idx != color_idx]
        self._affected_coord_keys_list = [self._affected_coord_keys_list[i] for i in indices_to_keep]
        self._affected_color_idx_list = [self._affected_color_idx_list[i] for i in indices_to_keep]

    # =========================================================================
    # INCREMENTAL DELTA UPDATES (for optimized check detection)
    # =========================================================================

    @staticmethod
    def coord_key_to_coord(coord_key: int) -> np.ndarray:
        """Convert coordinate key back to (x, y, z) coordinate.
        
        Reverses the bit-packing done in pseudolegal.coord_to_key:
        - x in bits 0-8
        - y in bits 9-17
        - z in bits 18-26
        """
        x = coord_key & 0x1FF  # Extract bits 0-8
        y = (coord_key >> 9) & 0x1FF  # Extract bits 9-17
        z = (coord_key >> 18) & 0x1FF  # Extract bits 18-26
        return np.array([x, y, z], dtype=COORD_DTYPE)

    def get_pieces_affected_by_move(
        self,
        from_coord: np.ndarray,
        to_coord: np.ndarray,
        color: int
    ) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
        """
        Identify pieces whose moves are affected by a simulated move.
        
        When a piece moves from A → B, the following pieces may be affected:
        1. Pieces attacking square A (may gain new moves - A is now empty)
        2. Pieces attacking square B (may lose moves - B is now occupied)
        3. The moved piece itself (needs regeneration from new position)
        
        Args:
            from_coord: Source coordinate of the move (3,)
            to_coord: Destination coordinate of the move (3,)
            color: Color of pieces to check (opponent color typically)
            
        Returns:
            Tuple of:
            - List of (color_idx, coord_key) for affected pieces
            - Coordinates array (N, 3) of affected pieces
            - Coord keys array (N,) for affected pieces
        """
        from game3d.movement.pseudolegal import coord_to_key
        
        color_idx = 0 if color == Color.WHITE else 1
        
        # Convert coordinates to keys for reverse map lookup
        from_key = coord_to_key(from_coord.reshape(1, 3))[0]
        to_key = coord_to_key(to_coord.reshape(1, 3))[0]
        
        affected_piece_ids = set()
        
        with self._lock:
            # 1. Find pieces targeting the source square (A)
            if int(from_key) in self._reverse_map:
                for piece_id in self._reverse_map[int(from_key)]:
                    # Only include pieces of the specified color
                    if piece_id[0] == color_idx:
                        affected_piece_ids.add(piece_id)
            
            # 2. Find pieces targeting the destination square (B)
            if int(to_key) in self._reverse_map:
                for piece_id in self._reverse_map[int(to_key)]:
                    # Only include pieces of the specified color
                    if piece_id[0] == color_idx:
                        affected_piece_ids.add(piece_id)
        
        # Convert to lists for return
        affected_list = list(affected_piece_ids)
        
        if not affected_list:
            return ([], np.empty((0, 3), dtype=COORD_DTYPE), np.empty(0, dtype=np.int64))
        
        # Extract coordinates and keys
        affected_coords = []
        affected_keys = []
        
        for piece_id in affected_list:
            _, coord_key = piece_id
            coord = self.coord_key_to_coord(coord_key)
            affected_coords.append(coord)
            affected_keys.append(coord_key)
        
        coords_array = np.array(affected_coords, dtype=COORD_DTYPE)
        keys_array = np.array(affected_keys, dtype=np.int64)
        
        return (affected_list, coords_array, keys_array)

    def extract_moves_for_pieces(
        self,
        all_moves: np.ndarray,
        piece_coords: np.ndarray
    ) -> np.ndarray:
        """
        Extract moves that originate from specific piece coordinates.
        
        ✅ OPTIMIZATION: Uses O(1) piece cache lookups instead of O(N*M) linear search.
        
        Args:
            all_moves: (N, 6) array of all moves [fx, fy, fz, tx, ty, tz] - IGNORED in optimized version
            piece_coords: (M, 3) array of piece coordinates to extract
            
        Returns:
            (K, 6) array of moves from the specified pieces
        """
        if piece_coords.size == 0:
            return np.empty((0, 6), dtype=MOVE_DTYPE)
        
        # ✅ OPTIMIZATION: Use piece cache for O(1) lookup instead of linear search
        # Convert coordinates to keys
        from game3d.movement.pseudolegal import coord_to_key
        coord_keys = coord_to_key(piece_coords)
        
        # Collect moves from piece cache (assume color from first piece in all_moves)
        # Since we're in a check detection context, we know the color from context
        moves_list = []
        
        with self._lock:
            # Try both colors to find the pieces (we don't know which color in this context)
            for color_idx in [0, 1]:
                for coord_key in coord_keys:
                    piece_id = (color_idx, int(coord_key))
                    if piece_id in self._piece_moves_cache:
                        cached_moves = self._piece_moves_cache[piece_id]
                        if cached_moves.size > 0:
                            moves_list.append(cached_moves)
        
        if not moves_list:
            return np.empty((0, 6), dtype=MOVE_DTYPE)
        
        return np.concatenate(moves_list, axis=0)

    def remove_moves_from_mask(
        self,
        attack_mask: np.ndarray,
        moves_to_remove: np.ndarray
    ) -> None:
        """
        Remove moves from attack mask (modifies in-place).
        
        Args:
            attack_mask: (SIZE, SIZE, SIZE) boolean array
            moves_to_remove: (N, 6) array of moves to remove from mask
        """
        if moves_to_remove.size == 0:
            return
        
        # Vectorized implementation
        tx = moves_to_remove[:, 3].astype(np.int64)
        ty = moves_to_remove[:, 4].astype(np.int64)
        tz = moves_to_remove[:, 5].astype(np.int64)
        
        # Bounds check (vectorized)
        valid_mask = (tx >= 0) & (tx < SIZE) & (ty >= 0) & (ty < SIZE) & (tz >= 0) & (tz < SIZE)
        
        if np.any(valid_mask):
            attack_mask[tx[valid_mask], ty[valid_mask], tz[valid_mask]] = False

    def add_moves_to_mask(
        self,
        attack_mask: np.ndarray,
        moves_to_add: np.ndarray
    ) -> None:
        """
        Add moves to attack mask (modifies in-place).
        
        Args:
            attack_mask: (SIZE, SIZE, SIZE) boolean array
            moves_to_add: (N, 6) array of moves to add to mask
        """
        if moves_to_add.size == 0:
            return
        
        # Vectorized implementation
        tx = moves_to_add[:, 3].astype(np.int64)
        ty = moves_to_add[:, 4].astype(np.int64)
        tz = moves_to_add[:, 5].astype(np.int64)
        
        # Bounds check (vectorized)
        valid_mask = (tx >= 0) & (tx < SIZE) & (ty >= 0) & (ty < SIZE) & (tz >= 0) & (tz < SIZE)
        
        if np.any(valid_mask):
            attack_mask[tx[valid_mask], ty[valid_mask], tz[valid_mask]] = True

    def _prune_piece_cache(self) -> None:
        """Prune oldest 20% of entries when cache exceeds limit."""
        if len(self._piece_moves_cache) <= self._max_piece_entries:
            return

        prune_start = len(self._piece_moves_cache)
        prune_target = int(self._max_piece_entries * 0.8)
        num_to_remove = prune_start - prune_target

        for _ in range(num_to_remove):
            try:
                piece_id, _ = self._piece_moves_cache.popitem(last=False)
                
                if piece_id in self._piece_targets:
                    for target_key in self._piece_targets[piece_id]:
                        if target_key in self._reverse_map:
                            self._reverse_map[target_key].discard(piece_id)
                            if not self._reverse_map[target_key]:
                                del self._reverse_map[target_key]
                    del self._piece_targets[piece_id]
            except KeyError:
                break

        self._prune_triggered += 1

def create_move_cache(cache_manager, config: Optional[MoveCacheConfig] = None) -> MoveCache:
    """Factory function to create move cache."""
    return MoveCache(cache_manager, config)

__all__ = ['MoveCache', 'MoveCacheConfig', 'create_move_cache']
