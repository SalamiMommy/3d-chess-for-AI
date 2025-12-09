
"""
Bitboard implementation for 9x9x9 3D Chess.
Total squares: 729.
Representation: Fixed-size Numpy array of 12 uint64 integers (768 bits).

optimized for Numba usage.
"""

import numpy as np
from numba import njit, uint64, int32, int64, void, boolean

# CONSTANTS
# 9*9*9 = 729 squares
# 12 * 64 = 768 bits
BITBOARD_SIZE = 12
DTYPE = np.uint64

@njit(cache=True)
def create_empty_bitboard() -> np.ndarray:
    """Create an empty bitboard (all zeros)."""
    return np.zeros(BITBOARD_SIZE, dtype=DTYPE)

@njit(cache=True)
def create_full_bitboard() -> np.ndarray:
    """Create a full bitboard (all ones, up to 729th bit)."""
    bb = np.full(BITBOARD_SIZE, 0xFFFFFFFFFFFFFFFF, dtype=DTYPE)
    # Mask out bits >= 729
    # 729 % 64 = 25.
    # The 11th index (12th element) should only have lower 25 bits set.
    # 11 * 64 = 704. 729 - 704 = 25 bits in the last uint64.
    # (1 << 25) - 1
    last_mask = (uint64(1) << uint64(25)) - uint64(1)
    bb[11] = last_mask
    return bb

@njit(cache=True)
def set_bit(bb: np.ndarray, one_d_idx: int) -> None:
    """Set a bit at the given 1D index."""
    if one_d_idx < 0 or one_d_idx >= 729:
        return
    array_idx = one_d_idx // 64
    bit_offset = one_d_idx % 64
    bb[array_idx] |= (uint64(1) << uint64(bit_offset))

@njit(cache=True)
def clear_bit(bb: np.ndarray, one_d_idx: int) -> None:
    """Clear a bit at the given 1D index."""
    if one_d_idx < 0 or one_d_idx >= 729:
        return
    array_idx = one_d_idx // 64
    bit_offset = one_d_idx % 64
    bb[array_idx] &= ~(uint64(1) << uint64(bit_offset))

@njit(cache=True)
def get_bit(bb: np.ndarray, one_d_idx: int) -> bool:
    """Check if a bit is set at the given 1D index."""
    if one_d_idx < 0 or one_d_idx >= 729:
        return False
    array_idx = one_d_idx // 64
    bit_offset = one_d_idx % 64
    return (bb[array_idx] >> uint64(bit_offset)) & uint64(1) != 0

@njit(cache=True)
def bitboard_and(bb1: np.ndarray, bb2: np.ndarray) -> np.ndarray:
    """Bitwise AND."""
    res = np.empty_like(bb1)
    for i in range(BITBOARD_SIZE):
        res[i] = bb1[i] & bb2[i]
    return res

@njit(cache=True)
def bitboard_or(bb1: np.ndarray, bb2: np.ndarray) -> np.ndarray:
    """Bitwise OR."""
    res = np.empty_like(bb1)
    for i in range(BITBOARD_SIZE):
        res[i] = bb1[i] | bb2[i]
    return res

@njit(cache=True)
def bitboard_xor(bb1: np.ndarray, bb2: np.ndarray) -> np.ndarray:
    """Bitwise XOR."""
    res = np.empty_like(bb1)
    for i in range(BITBOARD_SIZE):
        res[i] = bb1[i] ^ bb2[i]
    return res

@njit(cache=True)
def bitboard_not(bb: np.ndarray) -> np.ndarray:
    """Bitwise NOT (only for valid 729 bits)."""
    res = np.empty_like(bb)
    for i in range(BITBOARD_SIZE):
        res[i] = ~bb[i]
    
    # Mask last integer to avoid garbage in upper bits
    last_mask = (uint64(1) << uint64(25)) - uint64(1)
    res[11] &= last_mask
    return res

@njit(cache=True)
def is_empty(bb: np.ndarray) -> bool:
    """Check if bitboard is empty."""
    for i in range(BITBOARD_SIZE):
        if bb[i] != 0:
            return False
    return True

@njit(cache=True)
def popcount(bb: np.ndarray) -> int:
    """Count set bits."""
    count = 0
    # Numba doesn't strictly have a __builtin_popcount wrapper for array, 
    # but we can use a simple loop or a hack.
    # Actually, recent numba versions support bit_count() on integers or we can implement swar.
    # A simple loop over words with bit manipulation is fine, or just iterating bits? 
    # Iterating 768 bits is slow.
    # Let's use a simple SWAR or lookup if performance matters. 
    # For now, let's use a standard popcount algorithm for uint64.
    
    for i in range(BITBOARD_SIZE):
        n = bb[i]
        # SWAR algorithm for 64-bit popcount
        n = n - ((n >> uint64(1)) & uint64(0x5555555555555555))
        n = (n & uint64(0x3333333333333333)) + ((n >> uint64(2)) & uint64(0x3333333333333333))
        n = (n + (n >> uint64(4))) & uint64(0x0f0f0f0f0f0f0f0f)
        n = n + (n >> uint64(8))
        n = n + (n >> uint64(16))
        n = n + (n >> uint64(32))
        count += (n & uint64(0x7f))
        
    return int(count)

@njit(cache=True)
def get_set_bits(bb: np.ndarray) -> np.ndarray:
    """Get indices of all set bits. Returns int32 array."""
    # Two pass: count then fill
    count = popcount(bb)
    res = np.empty(count, dtype=np.int32)
    
    idx = 0
    for i in range(BITBOARD_SIZE):
        word = bb[i]
        if word == 0:
            continue
            
        base_idx = i * 64
        # Scan bits in word
        # We can optimize this with ctz... but keep it simple for now
        for bit in range(64):
            if (word >> uint64(bit)) & uint64(1):
                res[idx] = base_idx + bit
                idx += 1
                
    return res

@njit(cache=True)
def inplace_or(target: np.ndarray, source: np.ndarray) -> None:
    """In-place Bitwise OR: target |= source."""
    for i in range(BITBOARD_SIZE):
        target[i] |= source[i]
        
@njit(cache=True)
def inplace_and(target: np.ndarray, source: np.ndarray) -> None:
    """In-place Bitwise AND: target &= source."""
    for i in range(BITBOARD_SIZE):
        target[i] &= source[i]

@njit(cache=True)
def inplace_clear(target: np.ndarray) -> None:
    """Clear all bits in target."""
    for i in range(BITBOARD_SIZE):
        target[i] = 0
