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
from numba import njit, prange
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
import threading
import logging
from collections import OrderedDict, defaultdict

logger = logging.getLogger(__name__)

from game3d.common.shared_types import (
    COORD_DTYPE, INDEX_DTYPE, BOOL_DTYPE, VOLUME, MOVE_DTYPE as MOVE_DTYPE,
    Color, SIZE
)

from game3d.common.bitboard import (
    create_empty_bitboard, set_bit, get_bit, inplace_or, 
    inplace_clear, get_set_bits, BITBOARD_SIZE
)
from game3d.cache.unified_memory_pool import get_memory_pool

# ✅ OPTIMIZATION: Singleton empty arrays to avoid repeated allocations
_EMPTY_INT64 = np.empty(0, dtype=np.int64)
_EMPTY_INT64.flags.writeable = False  # Make immutable for safety

# ✅ OPTIMIZATION P2: Pre-allocated buffer size for affected pieces
_AFFECTED_BUFFER_SIZE = 512


@njit(cache=True)
def _find_move_by_key(keys: np.ndarray, target_key: int) -> int:
    mask = keys == target_key
    matches = np.where(mask)[0]
    return matches[0] if matches.size > 0 else -1

@njit(cache=True)
def _unpack_keys(keys_arr: np.ndarray) -> np.ndarray:
    """
    ✅ OPTIMIZED: Numba-accelerated key unpacking.
    Converts packed coordinate keys back to (x, y, z) coordinates.
    
    Key format: x | (y << 9) | (z << 18)
    """
    n = len(keys_arr)
    coords = np.empty((n, 3), dtype=COORD_DTYPE)
    for i in range(n):
        key = keys_arr[i]
        coords[i, 0] = key & 0x1FF        # bits 0-8
        coords[i, 1] = (key >> 9) & 0x1FF  # bits 9-17
        coords[i, 2] = (key >> 18) & 0x1FF # bits 18-26
    return coords


@dataclass
class MoveCacheConfig:
    """Configuration for move cache."""
    max_cache_size: int = 10000
    enable_transposition_table: bool = True

from numba import njit
@njit
def _extract_bits_indices(blocks):
    """Extract set bit indices from blocks array."""
    indices = []
    # 12 blocks
    for i in range(12):
        block = blocks[i]
        if block == 0: continue
        
        base = i * 64
        # simple iter
        for j in range(64):
            if (block >> np.uint64(j)) & np.uint64(1):
                indices.append(base + j)
    return indices


# ✅ OPTIMIZATION P1: Numba-accelerated batch bit operations
# Pre-compute bit modifications outside lock, apply atomically inside

@njit(cache=True)
def _compute_bit_ops_clear(old_target_keys: np.ndarray, piece_key: int, size: int):
    """Pre-compute clear operations for old targets (outside lock).
    
    Returns: (target_flats, block_indices, clear_masks) arrays
    """
    n = len(old_target_keys)
    target_flats = np.empty(n, dtype=np.int32)
    block_indices = np.empty(n, dtype=np.int32)
    clear_masks = np.empty(n, dtype=np.uint64)
    
    # Pre-compute piece flat index
    px = piece_key & 0x1FF
    py = (piece_key >> 9) & 0x1FF
    pz = (piece_key >> 18) & 0x1FF
    piece_flat = px + py * size + pz * size * size
    block_idx = piece_flat // 64
    bit_idx = piece_flat % 64
    clear_mask = ~(np.uint64(1) << np.uint64(bit_idx))
    
    for i in range(n):
        target_key = old_target_keys[i]
        tx = target_key & 0x1FF
        ty = (target_key >> 9) & 0x1FF
        tz = (target_key >> 18) & 0x1FF
        target_flats[i] = tx + ty * size + tz * size * size
        block_indices[i] = block_idx
        clear_masks[i] = clear_mask
    
    return target_flats, block_indices, clear_masks


@njit(cache=True)
def _compute_bit_ops_set(new_target_keys: np.ndarray, piece_key: int, size: int):
    """Pre-compute set operations for new targets (outside lock).
    
    Returns: (target_flats, block_indices, set_masks) arrays
    """
    n = len(new_target_keys)
    target_flats = np.empty(n, dtype=np.int32)
    block_indices = np.empty(n, dtype=np.int32)
    set_masks = np.empty(n, dtype=np.uint64)
    
    # Pre-compute piece flat index
    px = piece_key & 0x1FF
    py = (piece_key >> 9) & 0x1FF
    pz = (piece_key >> 18) & 0x1FF
    piece_flat = px + py * size + pz * size * size
    block_idx = piece_flat // 64
    bit_idx = piece_flat % 64
    set_mask = np.uint64(1) << np.uint64(bit_idx)
    
    for i in range(n):
        target_key = new_target_keys[i]
        tx = target_key & 0x1FF
        ty = (target_key >> 9) & 0x1FF
        tz = (target_key >> 18) & 0x1FF
        target_flats[i] = tx + ty * size + tz * size * size
        block_indices[i] = block_idx
        set_masks[i] = set_mask
    
    return target_flats, block_indices, set_masks


@njit(cache=True)
def _apply_bit_clears(matrix_slice, target_flats, block_indices, clear_masks):
    """Apply pre-computed clear operations to matrix slice."""
    for i in range(len(target_flats)):
        matrix_slice[target_flats[i], block_indices[i]] &= clear_masks[i]


@njit(cache=True)
def _apply_bit_sets(matrix_slice, target_flats, block_indices, set_masks):
    """Apply pre-computed set operations to matrix slice."""
    for i in range(len(target_flats)):
        matrix_slice[target_flats[i], block_indices[i]] |= set_masks[i]


# ✅ OPTIMIZATION #7: SIMD-style vectorized bit operations for bulk updates
# Operates on entire arrays at once for better CPU vectorization
@njit(cache=True, parallel=True)
def _apply_bit_clears_vectorized(matrix_slice, flat_indices, block_idx, clear_mask):
    """Apply same clear operation to multiple target squares (SIMD-friendly).
    
    All operations use the SAME block_idx and clear_mask (from same source piece).
    This enables better CPU vectorization than the generic version.
    """
    for i in prange(len(flat_indices)):
        matrix_slice[flat_indices[i], block_idx] &= clear_mask


@njit(cache=True, parallel=True)
def _apply_bit_sets_vectorized(matrix_slice, flat_indices, block_idx, set_mask):
    """Apply same set operation to multiple target squares (SIMD-friendly).
    
    All operations use the SAME block_idx and set_mask (from same source piece).
    This enables better CPU vectorization than the generic version.
    """
    for i in prange(len(flat_indices)):
        matrix_slice[flat_indices[i], block_idx] |= set_mask


@njit(cache=True)
def _build_excluded_mask(excluded_keys: np.ndarray) -> np.ndarray:
    """Build exclusion mask for has_other_attackers (Numba-accelerated)."""
    excluded_blocks = np.zeros(12, dtype=np.uint64)
    for i in range(len(excluded_keys)):
        key = excluded_keys[i]
        px = key & 0x1FF
        py = (key >> 9) & 0x1FF
        pz = (key >> 18) & 0x1FF
        piece_flat = px + py * 9 + pz * 81  # SIZE=9
        block_idx = piece_flat // 64
        bit_idx = piece_flat % 64
        excluded_blocks[block_idx] |= np.uint64(1) << np.uint64(bit_idx)
    return excluded_blocks


# ✅ OPTIMIZATION P3: Direct key extraction without type conversion
@njit(cache=True)
def _extract_target_keys_direct(moves: np.ndarray) -> np.ndarray:
    """Extract target keys from moves WITHOUT astype conversion.
    
    Moves columns 3,4,5 are already COORD_DTYPE (int16).
    We just need to pack them into int64 keys.
    """
    n = moves.shape[0]
    keys = np.empty(n, dtype=np.int64)
    for i in range(n):
        # Direct int conversion from int16 - no array allocation
        keys[i] = int(moves[i, 3]) | (int(moves[i, 4]) << 9) | (int(moves[i, 5]) << 18)
    return keys


# ✅ OPTIMIZATION P1: Binary search partitioning for get_incremental_state
@njit(cache=True)
def _binary_search_contains(sorted_arr: np.ndarray, target: int) -> bool:
    """O(log N) check if target exists in sorted array."""
    lo = 0
    hi = len(sorted_arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if sorted_arr[mid] == target:
            return True
        elif sorted_arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return False


@njit(cache=True)
def _partition_keys_by_validity(
    all_piece_keys: np.ndarray,
    invalid_keys_sorted: np.ndarray,
    cached_piece_ids_sorted: np.ndarray,
    color_shift: int
) -> tuple:
    """Partition piece keys into clean and dirty based on cache state.
    
    Uses O(N log M) binary search instead of O(N * M) set lookup.
    
    Args:
        all_piece_keys: Keys of all pieces to check
        invalid_keys_sorted: SORTED array of invalid coord keys
        cached_piece_ids_sorted: SORTED array of valid piece_ids in cache
        color_shift: Bit shift for color (0 or 1 << 30)
        
    Returns:
        (dirty_indices, dirty_count) - indices needing regeneration
    """
    n = len(all_piece_keys)
    dirty_indices = np.empty(n, dtype=np.int32)
    dirty_count = 0
    
    has_invalid = len(invalid_keys_sorted) > 0
    has_cached = len(cached_piece_ids_sorted) > 0
    
    for i in range(n):
        key = all_piece_keys[i]
        piece_id = key | color_shift
        
        # Check 1: Is key in invalid set? (binary search)
        is_invalid = False
        if has_invalid:
            is_invalid = _binary_search_contains(invalid_keys_sorted, key)
        
        # Check 2: Is piece_id NOT in cache? (binary search)
        not_cached = True
        if has_cached:
            not_cached = not _binary_search_contains(cached_piece_ids_sorted, piece_id)
        
        if is_invalid or not_cached:
            dirty_indices[dirty_count] = i
            dirty_count += 1
    
    return dirty_indices[:dirty_count], dirty_count


# ✅ OPTIMIZATION #2: Numba-accelerated unique (replaces np.unique)
@njit(cache=True)
def _unique_sorted_int64(arr: np.ndarray) -> np.ndarray:
    """Numba-accelerated unique for int64 arrays.
    
    O(N log N) complexity but fully Numba-compiled, avoiding Python/NumPy overhead.
    Returns sorted unique values.
    """
    if len(arr) == 0:
        return arr
    if len(arr) == 1:
        return arr.copy()
    
    # Sort in-place (copy first)
    sorted_arr = np.sort(arr)
    
    # First pass: count unique elements
    count = 1
    for i in range(1, len(sorted_arr)):
        if sorted_arr[i] != sorted_arr[i-1]:
            count += 1
    
    # Second pass: extract unique values
    result = np.empty(count, dtype=np.int64)
    result[0] = sorted_arr[0]
    idx = 1
    for i in range(1, len(sorted_arr)):
        if sorted_arr[i] != sorted_arr[i-1]:
            result[idx] = sorted_arr[i]
            idx += 1
    
    return result


# ✅ OPTIMIZATION P1: Extract affected piece keys directly to NumPy arrays
# Eliminates Python list allocation in hot paths

@njit(cache=True)
def _count_bits_in_blocks(blocks):
    """Count total set bits in blocks array."""
    count = 0
    for i in range(12):
        block = blocks[i]
        if block == 0:
            continue
        # Population count
        for j in range(64):
            if (block >> np.uint64(j)) & np.uint64(1):
                count += 1
    return count


@njit(cache=True)
def _extract_piece_keys_from_blocks(blocks, size):
    """Extract piece keys from blocks directly to array.
    
    Returns packed coordinate keys (x | y<<9 | z<<18) for each set bit.
    """
    # First count how many keys we need
    count = _count_bits_in_blocks(blocks)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    
    keys = np.empty(count, dtype=np.int64)
    idx = 0
    
    for i in range(12):
        block = blocks[i]
        if block == 0:
            continue
        
        base = i * 64
        for j in range(64):
            if (block >> np.uint64(j)) & np.uint64(1):
                flat = base + j
                # Convert flat to packed key
                pz = flat // (size * size)
                rem = flat % (size * size)
                py = rem // size
                px = rem % size
                keys[idx] = px | (py << 9) | (pz << 18)
                idx += 1
    
    return keys


@njit(cache=True)
def _extract_targeting_pieces_for_squares(
    attack_matrix_color,  # (729, 12) slice for one color
    target_flats,         # Array of target flat indices
    size
):
    """Extract all piece keys targeting the given target squares.
    
    Returns array of packed coordinate keys.
    """
    # First pass: count total
    total = 0
    for i in range(len(target_flats)):
        tf = target_flats[i]
        blocks = attack_matrix_color[tf]
        total += _count_bits_in_blocks(blocks)
    
    if total == 0:
        return np.empty(0, dtype=np.int64)
    
    # Second pass: extract
    keys = np.empty(total, dtype=np.int64)
    idx = 0
    
    for i in range(len(target_flats)):
        tf = target_flats[i]
        blocks = attack_matrix_color[tf]
        
        for b in range(12):
            block = blocks[b]
            if block == 0:
                continue
            
            base = b * 64
            for j in range(64):
                if (block >> np.uint64(j)) & np.uint64(1):
                    flat = base + j
                    pz = flat // (size * size)
                    rem = flat % (size * size)
                    py = rem // size
                    px = rem % size
                    keys[idx] = px | (py << 9) | (pz << 18)
                    idx += 1
    
    return keys





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
        
        # ✅ OPTIMIZED: Per-color locks reduce contention for concurrent access
        self._lock = threading.RLock()  # Global lock for cross-color operations
        self._color_locks = [threading.RLock(), threading.RLock()]  # Per-color locks
        self._lock_shards = [threading.RLock() for _ in range(4)] # For bitboard access

        # Attack Mask Cache (One per color)
        # Using Bitboards for O(1) check
        # Initialized as None, created on demand
        self._attack_bitboards = [None, None]  # [White, Black]
        self._bitboard_dirty = [True, True]
        
        # Parallel Execution
        self._memory_pool = get_memory_pool()
        
        # Initialize to current board generation
        current_gen = getattr(cache_manager.board, 'generation', 0)
        
        # 1. LEGAL MOVES (final, after all filtering)
        self._legal_moves_cache = [None, None]  # [White, Black]
        
        # 2. PSEUDOLEGAL MOVES (respect occupancy, before filtering)
        self._pseudolegal_moves_cache = [None, None]  # [White, Black]

        self._cache_generation = 0
        self._board_generation = current_gen

        stats_dtype = np.dtype([
            ('legal_cache_hits', INDEX_DTYPE),
            ('legal_cache_misses', INDEX_DTYPE),
            ('pseudolegal_cache_hits', INDEX_DTYPE),
            ('pseudolegal_cache_misses', INDEX_DTYPE),
            ('piece_cache_hits', INDEX_DTYPE),
            ('piece_cache_misses', INDEX_DTYPE),
            ('total_moves_cached', INDEX_DTYPE)
        ])
        self._stats = np.zeros(1, dtype=stats_dtype)[0]
        self._affected_pieces = set()
        self._piece_board_generations = {}
        self._board_generation_per_color = [current_gen, current_gen]

        # ✅ OPTIMIZED: Per-color affected tracking
        # - keys list for fast appends
        # - array cache for fast reads
        # - dirty flag for lazy updates
        self._affected_keys_per_color = [[], []]
        self._affected_arrays_per_color = [_EMPTY_INT64, _EMPTY_INT64]
        self._affected_dirty_per_color = [False, False]
        
        # ✅ OPTIMIZATION P1: Sorted cache key index for O(log N) lookups
        # Updated when pieces are added/removed from cache
        self._cached_piece_ids_sorted = [_EMPTY_INT64, _EMPTY_INT64]  # [White, Black]
        self._cache_index_dirty = [True, True]  # Needs rebuild when dirty

        # ✅ OPTIMIZATION #4: Cached Boolean Attack Mask (Lazy)
        self._cached_boolean_mask = [None, None]  # [White, Black]
        
        # Piece-level cache (stores pseudolegal moves by default)
        self._piece_moves_cache = OrderedDict()


        # ✅ OPTIMIZED: Dense Attack Interaction Matrix (Bitmatrix)
        # Dimensions: (2 colors, 729 target squares, 12 uint64 blocks for 768 source pieces)
        # Replaces defaultdict(set) for O(1) bitwise operations
        # White=0, Black=1
        # ✅ OPT #4: Fortran order for better cache locality during target-square iteration
        self._attack_matrix = np.zeros((2, SIZE**3, 12), dtype=np.uint64, order='F')
        
        # ✅ OPT 1.1: Source-transposed view for faster source-focused operations
        # When updating _update_reverse_map, we iterate by SOURCE piece
        # This transposed view allows cache-friendly access: (2, 12 blocks, 729 targets)
        self._attack_matrix_source_view = self._attack_matrix.reshape(2, SIZE**3, 12).transpose(0, 2, 1)
        
        # ✅ OPT 2.2: Last attacker tracking for check detection fast path
        # Stores packed coord keys of pieces that last attacked each king
        # Updated when pseudolegal moves are stored
        self._last_attackers_of_king = [np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)]  # [White King attackers, Black King attackers]
        self._last_attacker_generation = [0, 0]  # Track when last updated
        
        # Track which squares a piece targets
        self._piece_targets: Dict[tuple, set] = {}

        self._max_piece_entries = 1000
        self._prune_triggered = 0
        
        # ✅ OPTIMIZED: Batch operation tracking for reduced lock contention
        self._in_batch_mode = False

    # =========================================================================
    # BATCH OPERATIONS CONTEXT MANAGER
    # =========================================================================
    
    class _BatchContext:
        """Context manager for batch cache operations with single lock acquisition."""
        def __init__(self, cache, color: int):
            self.cache = cache
            self.color_idx = 0 if color == Color.WHITE else 1
            self.lock = cache._color_locks[self.color_idx]
            
        def __enter__(self):
            self.lock.acquire()
            self.cache._in_batch_mode = True
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.cache._in_batch_mode = False
            self.lock.release()
            return False
    
    def batch_operations(self, color: int):
        """
        ✅ OPTIMIZED: Context manager for batch cache operations.
        Acquires lock once for multiple operations, reducing contention.
        
        Usage:
            with cache.batch_operations(color) as batch:
                cache.store_piece_moves(color, key1, moves1)
                cache.store_piece_moves(color, key2, moves2)
        """
        return self._BatchContext(self, color)
    
    def get_legal_moves(self, color: int) -> Optional[np.ndarray]:
        """Retrieve cached LEGAL moves (after filtering)."""
        color_idx = 0 if color == Color.WHITE else 1
        
        # ✅ OPTIMIZED: Use per-color lock for reduced contention
        with self._color_locks[color_idx]:
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
    
    def store_attack(self, color: int, attacked_squares: np.ndarray) -> None:
        """
        Update attack bitboard for a color.
        
        Args:
            color: Color.WHITE or Color.BLACK
            attacked_squares: Array of (N, 3) coordinates
        """
        color_idx = 0 if color == Color.WHITE else 1
        
        with self._lock_shards[color_idx % 4]: # Simple lock
            # Initialize if needed
            if self._attack_bitboards[color_idx] is None:
                self._attack_bitboards[color_idx] = create_empty_bitboard()
            
            # If we are doing a FULL rebuild, we might clear first?
            # But store_attack is usually called incrementally or in batch?
            # Actually, generator calls this. 
            # If it's a full generation, we should probably clear first.
            # But currently we accumulate.
            # Let's assume this adds to the mask. Clear is implicit via invalidate?
            
            # Wait, currently `store_attack` isn't fully utilized in previous logic. 
            # It was `_attack_mask_cache`.
            
            if len(self._attack_bitboards) <= color_idx:
                # Expand bitboards if needed (shouldn't happen with standard 2-player)
                while len(self._attack_bitboards) <= color_idx:
                    self._attack_bitboards.append(create_empty_bitboard())

            bb = self._attack_bitboards[color_idx]
            
            # Optimized batch update
            self._batch_set_bits(bb, attacked_squares)

            # 2. Update Dense Bitmatrix for Reverse Lookup (O(1))
            # This part is for tracking which pieces attack which squares,
            # but `store_attack` only receives `attacked_squares` without source piece info.
            # So, this part is conceptually for a different function or needs source info.
            # For now, we only update the bitboard.
            
            self._bitboard_dirty[color_idx] = False
            self._cached_boolean_mask[color_idx] = None  # Invalidate boolean mask (needs rebuild)

    @staticmethod
    @njit(cache=True)
    def _batch_set_bits(bb, coords):
        n = coords.shape[0]
        for i in range(n):
            idx = coords[i, 0] + coords[i, 1] * 9 + coords[i, 2] * 81
            set_bit(bb, idx)

    def is_under_attack(self, coord: np.ndarray, defender_color: int) -> bool:
        """
        Check if a square is under attack by the OPPONENT of defender_color.
        Uses cached bitboard.
        """
        attacker_color = Color.BLACK if defender_color == Color.WHITE else Color.WHITE # 2=Black, 1=White
        attacker_idx = 0 if attacker_color == Color.WHITE else 1
        
        with self._lock_shards[attacker_idx % 4]:
            if self._bitboard_dirty[attacker_idx] or self._attack_bitboards[attacker_idx] is None:
                # If dirty/missing, we can't answer from cache!
                # Must return False or raise? 
                # Ideally we should generate it. But that risks recursion.
                # For now, return False and log warning? 
                # Or assume caller handles regeneration.
                # Implementation plan says we optimize check.py to use this.
                # check.py usually generates attacks if needed.
                return False # Cannot determine if dirty/missing
            
            bb = self._attack_bitboards[attacker_idx]
            if bb is None:
                return False
                
            flat_idx = coord[0] + coord[1] * 9 + coord[2] * 81
            return get_bit(bb, flat_idx)
    
    def store_legal_moves(self, color: int, moves: np.ndarray) -> None:
        """Store LEGAL moves (after filtering)."""
        color_idx = 0 if color == Color.WHITE else 1
        
        # ✅ OPTIMIZED: Use per-color lock for reduced contention
        with self._color_locks[color_idx]:
            if moves.size == 0:
                piece_count = len(self.cache_manager.occupancy_cache.get_positions(color))
                if piece_count > 0:
                    priest_count = self.cache_manager.occupancy_cache.get_priest_count(color)
                    
                    logger.debug(f"No legal moves for {Color(color).name} - {piece_count} pieces (Priests: {priest_count})")
                    
                    # Check cache consistency
                    is_valid, msg = self.cache_manager.occupancy_cache.validate_consistency()
                    if not is_valid:
                        logger.error(f"CACHE DESYNC DETECTED: {msg}")

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
        color_idx = 0 if color == Color.WHITE else 1
        
        # ✅ OPTIMIZED: Use per-color lock for reduced contention
        with self._color_locks[color_idx]:
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
        color_idx = 0 if color == Color.WHITE else 1
        
        # ✅ OPTIMIZED: Use per-color lock for reduced contention
        with self._color_locks[color_idx]:
            self._pseudolegal_moves_cache[color_idx] = moves.copy() if moves.size > 0 else moves
            
            # ✅ OPTIMIZATION: Build and cache attack mask
            # Extract destination coordinates for attack bitboard
            if moves.size > 0:
                # ✅ OPTIMIZED: Direct slice, no type conversion needed
                attacked_coords = moves[:, 3:6]
                
                # ✅ FIX: Must clear bitboard before rebuilding it from full move list
                # Otherwise old attacks persist (Ghost Checks) causing "Attackers: Unknown"
                with self._lock_shards[color_idx % 4]:
                    if self._attack_bitboards[color_idx] is not None:
                        inplace_clear(self._attack_bitboards[color_idx])
                        
                self.store_attack(color, attacked_coords)
            else:
                # If no moves, clear the attack bitboard for this color
                with self._lock_shards[color_idx % 4]:
                    if self._attack_bitboards[color_idx] is not None:
                        inplace_clear(self._attack_bitboards[color_idx])
                    self._bitboard_dirty[color_idx] = False # It's now explicitly clear
                    self._cached_boolean_mask[color_idx] = None # Clear cached boolean mask

    def invalidate_pseudolegal_moves(self, color: Optional[int] = None) -> None:
        """Invalidate pseudolegal moves cache."""
        with self._lock:
            if color is None:
                self._pseudolegal_moves_cache[0] = None
                self._pseudolegal_moves_cache[1] = None
                # ✅ OPTIMIZATION: Also invalidate attack masks
                with self._lock_shards[0 % 4]:
                    self._bitboard_dirty[0] = True
                    self._cached_boolean_mask[0] = None
                with self._lock_shards[1 % 4]:
                    self._bitboard_dirty[1] = True
                    self._cached_boolean_mask[1] = None
            else:
                color_idx = 0 if color == Color.WHITE else 1
                self._pseudolegal_moves_cache[color_idx] = None
                # ✅ OPTIMIZATION: Also invalidate attack mask
                with self._lock_shards[color_idx % 4]:
                    self._bitboard_dirty[color_idx] = True
                    self._cached_boolean_mask[color_idx] = None  # Invalidate cached boolean mask

    # =========================================================================
    # ATTACK MASK CACHE (for O(1) check detection)
    # =========================================================================
    
    # =========================================================================
    # ATTACK MASK CACHE (for O(1) check detection)
    # =========================================================================
    
    def get_attack_mask(self, color: int) -> Optional[np.ndarray]:
        """
        Get the full attack mask as a boolean array.
        
        Compatibility method for check.py.
        Reconstructs the boolean mask from the cached bitboard.
        """
        color_idx = 0 if color == Color.WHITE else 1
        
        with self._lock_shards[color_idx % 4]:
            if self._bitboard_dirty[color_idx] or self._attack_bitboards[color_idx] is None:
                return None
            
            # Check if we have a valid cached boolean mask
            if self._cached_boolean_mask[color_idx] is not None:
                return self._cached_boolean_mask[color_idx]

            # Construct boolean mask from bitboard
            mask = np.zeros((SIZE, SIZE, SIZE), dtype=BOOL_DTYPE)
            
            # Efficiently populate mask from set bits
            indices = get_set_bits(self._attack_bitboards[color_idx])
            if indices.size > 0:
                # Convert flat indices to coordinates
                # Flat index = x + y*9 + z*81
                z = indices // 81
                rem = indices % 81
                y = rem // 9
                x = rem % 9
                
                mask[x, y, z] = True
                
            self._cached_boolean_mask[color_idx] = mask
            return mask

    def get_cached_moves(self, color: int) -> Optional[np.ndarray]:
        """LEGACY: Retrieve cached moves (maps to legal moves)."""
        return self.get_legal_moves(color)

    def store_moves(self, color: int, moves: np.ndarray) -> None:
        """LEGACY: Store moves (maps to legal moves)."""
        return self.store_legal_moves(color, moves)

    def invalidate(self) -> None:
        """Invalidate ALL caches."""
        with self._lock:
            self.invalidate_legal_moves()
            self.invalidate_pseudolegal_moves() # This will also invalidate bitboards
            self._cache_generation += 1
            self._piece_moves_cache.clear()
            # ✅ OPTIMIZED: Reset matrix
            self._attack_matrix.fill(0)

            self._piece_targets.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics for all cache levels."""
        with self._lock:
            # Legal cache stats
            legal_lookups = self._stats['legal_cache_hits'] + self._stats['legal_cache_misses']
            legal_hit_rate = self._stats['legal_cache_hits'] / max(legal_lookups, 1)
            
            # Pseudolegal cache stats
            pseudo_lookups = self._stats['pseudolegal_cache_hits'] + self._stats['pseudolegal_cache_misses']
            pseudo_hit_rate = self._stats['pseudolegal_cache_hits'] / max(pseudo_lookups, 1)


            
            # Piece cache stats
            piece_lookups = self._stats['piece_cache_hits'] + self._stats['piece_cache_misses']
            piece_hit_rate = self._stats['piece_cache_hits'] / max(piece_lookups, 1)

            return {
                'legal_hit_rate': legal_hit_rate,
                'pseudolegal_hit_rate': pseudo_hit_rate,
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
            # ✅ OPTIMIZED: Reset matrix
            self._attack_matrix.fill(0)

            self._piece_targets.clear()
            # ✅ OPTIMIZED: Reset per-color affected tracking
            self._affected_keys_per_color = [[], []]
            self._affected_arrays_per_color = [_EMPTY_INT64, _EMPTY_INT64]
            self._affected_dirty_per_color = [False, False]
            # Clear bitboards
            with self._lock_shards[0 % 4]:
                if self._attack_bitboards[0] is not None:
                    inplace_clear(self._attack_bitboards[0])
                self._bitboard_dirty[0] = True
                self._cached_boolean_mask[0] = None
            with self._lock_shards[1 % 4]:
                if self._attack_bitboards[1] is not None:
                    inplace_clear(self._attack_bitboards[1])
                self._bitboard_dirty[1] = True
                self._cached_boolean_mask[1] = None

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
            self._affected_keys_per_color[color_idx].append(int_key)
            self._affected_dirty_per_color[color_idx] = True

    def mark_pieces_invalid_batch(self, color_indices: np.ndarray, keys: np.ndarray) -> None:
        """Batch mark pieces for regeneration."""
        if keys.size == 0:
            return
            
        with self._lock:
            # ✅ OPTIMIZED: Direct per-color append avoids later filtering
            for i in range(len(keys)):
                color_idx = int(color_indices[i])
                self._affected_keys_per_color[color_idx].append(int(keys[i]))
                self._affected_dirty_per_color[color_idx] = True

    def invalidate_targeting_pieces(self, coord_keys: np.ndarray) -> None:
        """Invalidate all pieces targeting the given coordinates."""
        if coord_keys.size == 0:
            return

        with self._lock:
            # Iterate over target keys
            for key in coord_keys:
                key_int = int(key)
                
                # Unpack target key
                tx = key_int & 0x1FF
                ty = (key_int >> 9) & 0x1FF
                tz = (key_int >> 18) & 0x1FF
                target_flat = tx + ty * SIZE + tz * SIZE * SIZE
                
                # Check WHITE pieces targeting this square
                blocks_w = self._attack_matrix[0, target_flat]
                if np.any(blocks_w):
                    flat_indices = _extract_bits_indices(blocks_w)
                    for f in flat_indices:
                        # Convert flat to packed key
                        pz = f // (SIZE * SIZE)
                        rem = f % (SIZE * SIZE)
                        py = rem // SIZE
                        px = rem % SIZE
                        
                        p_key = px | (py << 9) | (pz << 18)
                        self._affected_keys_per_color[0].append(p_key)  # White

                # Check BLACK pieces targeting this square
                blocks_b = self._attack_matrix[1, target_flat]
                if np.any(blocks_b):
                    flat_indices = _extract_bits_indices(blocks_b)
                    for f in flat_indices:
                        # Convert flat to packed key
                        pz = f // (SIZE * SIZE)
                        rem = f % (SIZE * SIZE)
                        py = rem // SIZE
                        px = rem % SIZE
                        
                        p_key = px | (py << 9) | (pz << 18)
                        self._affected_keys_per_color[1].append(p_key)  # Black
            
            self._affected_dirty_per_color[0] = True
            self._affected_dirty_per_color[1] = True

    def has_piece_moves(self, color: int, coord_key: Union[int, bytes]) -> bool:
        """Check if piece moves are cached."""
        color_idx = 0 if color == Color.WHITE else 1

        if isinstance(coord_key, (int, np.integer)):
            int_key = int(coord_key)
        else:
            int_key = int.from_bytes(coord_key, 'little') if coord_key else 0

        # ✅ OPTIMIZATION: Pack key into int (bit 30 = color)
        piece_id = int_key | (color_idx << 30)
        
        with self._lock:
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
        
        # ✅ OPTIMIZATION: Pack key into int (bit 30 = color)
        piece_id = int_key | (color_idx << 30)
        
        with self._lock:
            if piece_id in self._piece_moves_cache:
                self._piece_moves_cache.move_to_end(piece_id)
                self._stats['piece_cache_hits'] += 1
                return self._piece_moves_cache[piece_id]
            
            self._stats['piece_cache_misses'] += 1
            return np.empty((0, 6), dtype=COORD_DTYPE)


    def has_other_attackers(self, target_coord: np.ndarray, color: int, excluded_keys: np.ndarray) -> bool:
        """
        Check if target square is attacked by pieces OTHER than those in excluded_keys.
        Used for optimized incremental check detection (Hard Case).
        """
        # Calculate target key
        tx, ty, tz = int(target_coord[0]), int(target_coord[1]), int(target_coord[2])
        target_flat = tx + ty * SIZE + tz * SIZE * SIZE
        color_idx = 0 if color == Color.WHITE else 1
        
        # ✅ OPTIMIZATION: Build mask outside lock (read-only Numba operation)
        # significantly reduces lock contention during heavy check detection
        excluded_blocks = _build_excluded_mask(excluded_keys)
        
        with self._lock:
            # Atomic read of the 12 blocks of attackers
            attackers_blocks = self._attack_matrix[color_idx, target_flat]
            
            # Numpy bitwise operations on uint64 array
            # We want: bitwise_and(attackers, bitwise_not(excluded))
            # If any block is non-zero, return True
            result_blocks = attackers_blocks & (~excluded_blocks)
            return np.any(result_blocks)

    # ✅ OPT 2.2: Check Detection Fast Path
    def is_king_attacked_fast_path(self, king_coord: np.ndarray, defender_color: int) -> Optional[bool]:
        """
        Fast path check for king attack status using cached attacker info.
        
        Returns:
            True if king is definitely attacked
            False if king is definitely NOT attacked (cache valid + no attackers)
            None if we can't determine (cache miss, must use slow path)
            
        Use case: Most moves don't change check status when not already in check.
        This provides O(1) lookup vs O(N) regeneration for common cases.
        """
        # Attacker is opponent color
        attacker_color_idx = 1 if defender_color == Color.WHITE else 0
        defender_color_idx = 0 if defender_color == Color.WHITE else 1
        
        # Check if attacker's pseudolegal cache is valid
        if self._pseudolegal_moves_cache[attacker_color_idx] is None:
            return None  # Cache miss - must use slow path
        
        # Check if there are affected pieces that might change the attack status
        affected = self.get_affected_pieces(Color.WHITE if attacker_color_idx == 0 else Color.BLACK)
        if affected.size > 0:
            return None  # Dirty pieces - can't trust cached attackers
        
        # Calculate king's flat index for bitboard lookup
        kx, ky, kz = int(king_coord[0]), int(king_coord[1]), int(king_coord[2])
        king_flat = kx + ky * SIZE + kz * SIZE * SIZE
        
        with self._lock:
            # Check if any attacker piece targets king's square
            attackers_blocks = self._attack_matrix[attacker_color_idx, king_flat]
            is_attacked = np.any(attackers_blocks)
            
            return bool(is_attacked)


    def store_piece_moves(self, color: int, coord_key: Union[int, bytes], moves: np.ndarray) -> None:
        """Cache moves for a specific piece - USE INTEGER KEYS."""
        color_idx = 0 if color == Color.WHITE else 1

        if isinstance(coord_key, (int, np.integer)):
            int_key = int(coord_key)
        else:
            int_key = int.from_bytes(coord_key, 'little') if coord_key else 0

        # ✅ OPTIMIZATION: Pack key into int (bit 30 = color)
        piece_id = int_key | (color_idx << 30)

        with self._lock:
            if len(self._piece_moves_cache) >= self._max_piece_entries:
                self._prune_piece_cache()

            self._update_reverse_map(piece_id, moves)
            self._piece_moves_cache[piece_id] = moves
            self._piece_moves_cache.move_to_end(piece_id)






    def _update_reverse_map(self, piece_id: int, moves: np.ndarray) -> None:
        """
        Update the reverse map for a piece using DENSE BITMATRIX.
        piece_id = int_key | (color_idx << 30)
        
        ✅ OPTIMIZATION P1: Single lock acquisition (reduced from 2).
        ✅ OPTIMIZATION P2: Numba-accelerated unique (replaces np.unique).
        Lock-free read for old_targets (dict.get is thread-safe for read).
        """
        color_idx = (piece_id >> 30) & 1
        piece_key = piece_id & 0x3FFFFFFF
        
        # ===== PRE-COMPUTE OUTSIDE LOCK =====
        # ✅ OPT #1: Lock-free read (dict.get is thread-safe for read in Python)
        old_targets = self._piece_targets.get(piece_id, np.empty(0, dtype=np.int64))
        
        # Compute clear operations (Numba - no lock needed)
        if len(old_targets) > 0:
            clear_flats, clear_blocks, clear_masks = _compute_bit_ops_clear(
                old_targets, piece_key, SIZE
            )
        else:
            clear_flats = np.empty(0, dtype=np.int32)
        
        # Compute new targets and set operations
        if moves.size > 0:
            to_x = moves[:, 3].astype(np.int64)
            to_y = moves[:, 4].astype(np.int64)
            to_z = moves[:, 5].astype(np.int64)
            target_keys = to_x | (to_y << 9) | (to_z << 18)
            # ✅ OPT #2: Numba-accelerated unique
            new_targets = _unique_sorted_int64(target_keys)
            
            set_flats, set_blocks, set_masks = _compute_bit_ops_set(
                new_targets, piece_key, SIZE
            )
        else:
            new_targets = np.empty(0, dtype=np.int64)
            set_flats = np.empty(0, dtype=np.int32)
        
        # ===== APPLY ATOMICALLY UNDER LOCK (Single acquisition) =====
        with self._lock:
            # Apply clears (Numba-accelerated - minimal time under lock)
            if len(clear_flats) > 0:
                _apply_bit_clears(self._attack_matrix[color_idx], clear_flats, clear_blocks, clear_masks)
            
            # Apply sets (Numba-accelerated)
            if len(set_flats) > 0:
                _apply_bit_sets(self._attack_matrix[color_idx], set_flats, set_blocks, set_masks)
            
            # Update targets tracking
            self._piece_targets[piece_id] = new_targets



    def _set_attacker_bit(self, color_idx: int, target_key: int, piece_key: int):
        """Set bit in attack matrix."""
        tx = target_key & 0x1FF
        ty = (target_key >> 9) & 0x1FF
        tz = (target_key >> 18) & 0x1FF
        target_flat = tx + ty * SIZE + tz * SIZE * SIZE
        
        px = piece_key & 0x1FF
        py = (piece_key >> 9) & 0x1FF
        pz = (piece_key >> 18) & 0x1FF
        piece_flat = px + py * SIZE + pz * SIZE * SIZE
        
        block_idx = piece_flat // 64
        bit_idx = piece_flat % 64
        
        self._attack_matrix[color_idx, target_flat, block_idx] |= (np.uint64(1) << np.uint64(bit_idx))

    def _clear_attacker_bit(self, color_idx: int, target_key: int, piece_key: int):
        """Clear bit in attack matrix."""
        tx = target_key & 0x1FF
        ty = (target_key >> 9) & 0x1FF
        tz = (target_key >> 18) & 0x1FF
        target_flat = tx + ty * SIZE + tz * SIZE * SIZE
        
        px = piece_key & 0x1FF
        py = (piece_key >> 9) & 0x1FF
        pz = (piece_key >> 18) & 0x1FF
        piece_flat = px + py * SIZE + pz * SIZE * SIZE
        
        block_idx = piece_flat // 64
        bit_idx = piece_flat % 64
        
        self._attack_matrix[color_idx, target_flat, block_idx] &= ~(np.uint64(1) << np.uint64(bit_idx))


    def get_pieces_targeting(self, coord_keys: np.ndarray) -> list:
        """Get all pieces targeting the given coordinates. Returns list of (color_idx, piece_key)."""
        affected_pieces = []
        with self._lock:
            for key in coord_keys:
                key_int = int(key)
                
                # key unpack
                tx = key_int & 0x1FF
                ty = (key_int >> 9) & 0x1FF
                tz = (key_int >> 18) & 0x1FF
                target_flat = tx + ty * SIZE + tz * SIZE * SIZE
                
                # Check White
                blocks_w = self._attack_matrix[0, target_flat]
                if np.any(blocks_w):
                    flat_indices = _extract_bits_indices(blocks_w)
                    for f in flat_indices:
                        # Convert flat to packed key
                        pz = f // (SIZE * SIZE)
                        rem = f % (SIZE * SIZE)
                        py = rem // SIZE
                        px = rem % SIZE
                        
                        p_key = px | (py << 9) | (pz << 18)
                        affected_pieces.append((0, p_key))

                # Check Black
                blocks_b = self._attack_matrix[1, target_flat]
                if np.any(blocks_b):
                    flat_indices = _extract_bits_indices(blocks_b)
                    for f in flat_indices:
                        # Convert flat to packed key
                        pz = f // (SIZE * SIZE)
                        rem = f % (SIZE * SIZE)
                        py = rem // SIZE
                        px = rem % SIZE
                        
                        p_key = px | (py << 9) | (pz << 18)
                        affected_pieces.append((1, p_key))
                        
        return affected_pieces


    def get_affected_pieces(self, color: int) -> np.ndarray:
        """Get affected pieces as numpy array.
        
        ✅ OPTIMIZED: Uses per-color lists, eliminating O(N) mask filtering.
        - Fast path: O(1) for empty list (most common)
        - Lazy conversion: Only converts list→array when dirty
        - No filtering: Data is already per-color
        """
        color_idx = 0 if color == Color.WHITE else 1
        
        # ✅ FAST PATH: Check empty list BEFORE any work
        if not self._affected_keys_per_color[color_idx]:
            return _EMPTY_INT64
        
        # Only rebuild array when dirty
        if self._affected_dirty_per_color[color_idx]:
            self._affected_arrays_per_color[color_idx] = np.array(
                self._affected_keys_per_color[color_idx], dtype=np.int64
            )
            self._affected_dirty_per_color[color_idx] = False
        
        return self._affected_arrays_per_color[color_idx]

    def clear_affected_pieces(self, color: int) -> None:
        """Clear affected pieces after regeneration.
        
        ✅ OPTIMIZED: O(1) clear instead of O(N) filtering.
        """
        color_idx = 0 if color == Color.WHITE else 1
        
        # Clear the list and array for this color
        self._affected_keys_per_color[color_idx] = []
        self._affected_arrays_per_color[color_idx] = _EMPTY_INT64
        self._affected_dirty_per_color[color_idx] = False

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
        
        ✅ OPTIMIZATION P1: Uses Numba kernels instead of Python set/list.
        
        When a piece moves from A -> B, the following pieces may be affected:
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
        # Compute target flat indices (outside lock)
        from_flat = int(from_coord[0]) + int(from_coord[1]) * SIZE + int(from_coord[2]) * SIZE * SIZE
        to_flat = int(to_coord[0]) + int(to_coord[1]) * SIZE + int(to_coord[2]) * SIZE * SIZE
        target_flats = np.array([from_flat, to_flat], dtype=np.int32)
        
        color_idx = 0 if color == Color.WHITE else 1
        
        # Extract piece keys using Numba kernel (minimal lock time)
        with self._lock:
            # Get the slice for the color we care about
            matrix_slice = self._attack_matrix[color_idx]
            keys_array = _extract_targeting_pieces_for_squares(matrix_slice, target_flats, SIZE)
        
        # ✅ FIX: Explicitly check for King adjacency
        # Kings (and some other pieces) might "defend" friendly squares (e.g. Pawn at (5,3,4) blocked Black King at (5,4,4))
        # but pseudolegal generator usually excludes friendly squares, so they are NOT in _attack_matrix.
        # If the friendly piece is captured, the King MUST be regenerated to see the new attack line.
        # We manually add the King to the keys list if it is within range (Chebyshev distance <= 1).
        
        occ_cache = getattr(self.cache_manager, 'occupancy_cache', None)
        if occ_cache is not None and hasattr(occ_cache, 'find_king'):
             king_pos = occ_cache.find_king(color)
             if king_pos is not None:
                 kx, ky, kz = int(king_pos[0]), int(king_pos[1]), int(king_pos[2])
                 fx, fy, fz = int(from_coord[0]), int(from_coord[1]), int(from_coord[2])
                 tx, ty, tz = int(to_coord[0]), int(to_coord[1]), int(to_coord[2])
                 
                 # Check adjacency (Chebyshev distance <= 1)
                 dist_f = max(abs(kx - fx), abs(ky - fy), abs(kz - fz))
                 dist_t = max(abs(kx - tx), abs(ky - ty), abs(kz - tz))
                 
                 if dist_f <= 1 or dist_t <= 1:
                     king_key = kx | (ky << 9) | (kz << 18)
                     # Append to keys_array
                     keys_array = np.append(keys_array, np.int64(king_key))
        
        # Deduplicate keys (outside lock) using Numba-accelerated unique
        if keys_array.size == 0:
            return ([], np.empty((0, 3), dtype=COORD_DTYPE), np.empty(0, dtype=np.int64))
        
        # ✅ OPT #2: Use Numba-accelerated unique
        keys_array = _unique_sorted_int64(keys_array)
        
        # Build legacy list format (required by current API, but cheap now)
        affected_list = [(color_idx, int(k)) for k in keys_array]
        
        # Convert keys to coordinates using Numba kernel
        coords = _unpack_keys(keys_array)
        
        return (affected_list, coords, keys_array)




    def extract_moves_for_pieces(
        self,
        all_moves: np.ndarray,
        piece_coords: np.ndarray
    ) -> np.ndarray:
        """
        Extract moves that originate from specific piece coordinates.
        
        OPTIMIZATION: Uses O(1) piece cache lookups instead of O(N*M) linear search.
        
        Args:
            all_moves: (N, 6) array of all moves [fx, fy, fz, tx, ty, tz] - IGNORED in optimized version
            piece_coords: (M, 3) array of piece coordinates to extract
            
        Returns:
            (K, 6) array of moves from the specified pieces
        """
        if piece_coords.size == 0:
            return np.empty((0, 6), dtype=COORD_DTYPE)
        
        # OPTIMIZATION: Use piece cache for O(1) lookup instead of linear search
        # Convert coordinates to keys
        coord_keys = coords_to_keys(piece_coords)
        
        # Collect moves from piece cache (assume color from first piece in all_moves)
        # Since we're in a check detection context, we know the color from context
        moves_list = []
        
        with self._lock:
            # Try both colors to find the pieces (we don't know which color in this context)
            for color_idx in [0, 1]:
                # Pre-compute color shift
                color_shift = color_idx << 30
                for coord_key in coord_keys:
                    piece_id = int(coord_key) | color_shift
                    if piece_id in self._piece_moves_cache:
                        cached_moves = self._piece_moves_cache[piece_id]
                        if cached_moves.size > 0:
                            moves_list.append(cached_moves)
        
        if not moves_list:
            return np.empty((0, 6), dtype=COORD_DTYPE)
        
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

    def get_incremental_state(
        self, 
        color: int, 
        all_piece_keys: np.ndarray
    ) -> tuple[list[np.ndarray], list[int]]:
        """
        Retrieve valid cached moves and identify pieces needing regeneration.
        
        Args:
            color: Color to check
            all_piece_keys: (N,) keys corresponding to pieces
            
        Returns:
            clean_moves_list: List of move arrays for valid pieces
            dirty_indices: List of INDICES in `all_piece_keys` that need regen
        """
        clean_moves = []
        dirty_indices = []
        
        # Get set of invalid keys
        # ✅ OPTIMIZED: Use set comprehension instead of tolist() for direct iteration
        invalid_keys_arr = self.get_affected_pieces(color)
        invalid_keys_set = set(int(k) for k in invalid_keys_arr) if invalid_keys_arr.size > 0 else set()
        
        color_idx = 0 if color == Color.WHITE else 1
        color_shift = color_idx << 30
        
        with self._lock:
            # Optimize loop: avoid attribute lookups
            cache = self._piece_moves_cache
            
            for i in range(len(all_piece_keys)):
                key = int(all_piece_keys[i])
                # ✅ OPTIMIZED: Use int key (no tuple alloc)
                piece_id = key | color_shift
                
                # Condition for DIRTY/REGENERATE:
                # 1. Piece marked as invalid (affected by recent move)
                # 2. Piece not in cache at all (new piece or pruned)
                
                if key in invalid_keys_set or piece_id not in cache:
                    dirty_indices.append(i)
                else:
                    # Valid cache entry - use it!
                    move_arr = cache[piece_id]
                    # Only append if moves exist (optimization)
                    if move_arr.size > 0:
                        clean_moves.append(move_arr)
                        
        return clean_moves, dirty_indices

    def _prune_piece_cache(self) -> None:
        """Prune oldest 20% of entries when cache exceeds limit."""
        if len(self._piece_moves_cache) <= self._max_piece_entries:
            return

        prune_start = len(self._piece_moves_cache)
        prune_target = int(self._max_piece_entries * 0.8)
        num_to_remove = prune_start - prune_target

        for _ in range(num_to_remove):
            try:
                key, _ = self._piece_moves_cache.popitem(last=False)
                
                # Cleanup reverse map
                if key in self._piece_targets:
                    # Clear bits in global matrix
                    color_idx = (key >> 30) & 1
                    piece_key = key & 0x3FFFFFFF
                    
                    old_targets = self._piece_targets[key]
                    if len(old_targets) > 0:
                        for t_key in old_targets:
                            self._clear_attacker_bit(color_idx, t_key, piece_key)
                        
                    del self._piece_targets[key]
            except KeyError:
                break

        self._prune_triggered += 1

def create_move_cache(cache_manager, config: Optional[MoveCacheConfig] = None) -> MoveCache:
    """Factory function to create move cache."""
    return MoveCache(cache_manager, config)

__all__ = ['MoveCache', 'MoveCacheConfig', 'create_move_cache']

@njit(cache=True, nogil=True)
def _build_excluded_mask(excluded_keys: np.ndarray) -> np.ndarray:
    """Build a 12-element uint64 mask from excluded coordinate keys."""
    mask = np.zeros(12, dtype=np.uint64)
    for key in excluded_keys:
        # Convert packed coord key -> flat index
        # key = x | y<<9 | z<<18
        # flat = x + y*SIZE + z*SIZE*SIZE
        
        # Optimize decoding avoiding full unpack
        flat = (key & 0x1FF) + ((key >> 9) & 0x1FF) * SIZE + ((key >> 18) & 0x1FF) * 81 # SIZE*SIZE (9*9=81)
        
        b_idx = flat // 64
        bit_idx = flat % 64
        
        # Set bit
        mask[b_idx] |= (np.uint64(1) << np.uint64(bit_idx))
    return mask
