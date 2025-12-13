"""
Numba JIT Warmup Script - Pre-compile critical functions on startup.

This script should be called during application initialization to ensure
all Numba-compiled functions are pre-warmed, avoiding first-call latency.
"""

import numpy as np
from game3d.common.shared_types import COORD_DTYPE, BOOL_DTYPE, SIZE

def warmup_jit_functions():
    """Pre-compile all critical Numba functions by invoking them once."""
    print("Warming up Numba JIT functions...")
    
    # --- 1. movecache.py functions ---
    from game3d.cache.caches.movecache import (
        _unpack_keys, _compute_bit_ops_clear, _compute_bit_ops_set,
        _apply_bit_clears, _apply_bit_sets, _build_excluded_mask,
        _extract_target_keys_direct, _binary_search_contains,
        _unique_sorted_int64, _extract_bits_indices
    )
    
    # Warmup with minimal dummy data
    dummy_keys = np.array([0, 1, 81], dtype=np.int64)
    _ = _unpack_keys(dummy_keys)
    
    dummy_target_keys = np.array([0, 1], dtype=np.int64)
    _ = _compute_bit_ops_clear(dummy_target_keys, 0, SIZE)
    _ = _compute_bit_ops_set(dummy_target_keys, 0, SIZE)
    
    dummy_matrix = np.zeros((SIZE**3, 12), dtype=np.uint64)
    dummy_flats = np.array([0, 1], dtype=np.int32)
    dummy_block_indices = np.array([0, 0], dtype=np.int32)
    dummy_masks = np.array([1, 1], dtype=np.uint64)
    _apply_bit_clears(dummy_matrix, dummy_flats, dummy_block_indices, dummy_masks)
    _apply_bit_sets(dummy_matrix, dummy_flats, dummy_block_indices, dummy_masks)
    
    _ = _build_excluded_mask(dummy_keys)
    
    dummy_moves = np.zeros((2, 6), dtype=COORD_DTYPE)
    _ = _extract_target_keys_direct(dummy_moves)
    
    _ = _binary_search_contains(dummy_keys, 1)
    _ = _unique_sorted_int64(dummy_keys)
    
    dummy_blocks = np.zeros(12, dtype=np.uint64)
    dummy_blocks[0] = 1
    _ = _extract_bits_indices(dummy_blocks)
    
    # --- 2. check.py functions ---
    from game3d.attacks.check import (
        _find_attackers_of_square, _batch_check_move_blocks_or_captures,
        _check_move_blocks_or_captures
    )
    
    dummy_pos = np.array([0, 0, 0], dtype=COORD_DTYPE)
    dummy_opponent_moves = np.zeros((2, 6), dtype=COORD_DTYPE)
    dummy_from_coords = np.zeros((2, 3), dtype=COORD_DTYPE)
    
    _ = _find_attackers_of_square(dummy_pos, dummy_opponent_moves, dummy_from_coords)
    
    dummy_indices = np.array([0, 1], dtype=np.int32)
    _ = _batch_check_move_blocks_or_captures(dummy_opponent_moves, dummy_pos, dummy_pos, dummy_indices)
    
    dummy_move = np.array([0, 0, 0, 1, 1, 1], dtype=COORD_DTYPE)
    _ = _check_move_blocks_or_captures(dummy_move, dummy_pos, dummy_pos, 1)
    
    # --- 3. coord_utils.py functions ---
    from game3d.common.coord_utils import (
        coords_to_keys, coord_to_key_scalar, pack_coords, unpack_coords
    )
    
    dummy_coords = np.array([[0, 0, 0], [1, 1, 1]], dtype=COORD_DTYPE)
    _ = coords_to_keys(dummy_coords)
    _ = coord_to_key_scalar(0, 0, 0)
    _ = pack_coords(dummy_coords)
    _ = unpack_coords(dummy_keys)
    
    # --- 4. occupancycache.py functions ---
    from game3d.cache.caches.occupancycache import (
        _vectorized_batch_occupied, _vectorized_batch_occupied_serial,
        _vectorized_batch_attributes_unsafe
    )
    
    dummy_occ = np.zeros((SIZE, SIZE, SIZE), dtype=np.int8)
    dummy_ptype = np.zeros((SIZE, SIZE, SIZE), dtype=np.int8)
    _ = _vectorized_batch_occupied(dummy_occ, dummy_coords)
    _ = _vectorized_batch_occupied_serial(dummy_occ, dummy_coords)
    _ = _vectorized_batch_attributes_unsafe(dummy_occ, dummy_ptype, dummy_coords)
    
    print("JIT warmup complete.")


if __name__ == "__main__":
    warmup_jit_functions()
