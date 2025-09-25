"""Training-aware caching utilities."""

import functools
import torch
from .constants import VOLUME
from .geometry import coord_to_idx

# ---------- config ----------
TENSOR_CACHE_SIZE: int = 200_000     # fits ~200 k unique boards in RAM
MOVE_CACHE_SIZE:   int = 200_000

# ---------- decorators ----------
def tensor_cache(user_function):
    """LRU for functions that return a torch.Tensor."""
    return functools.lru_cache(maxsize=TENSOR_CACHE_SIZE)(user_function)

def move_cache(user_function):
    """LRU for functions that return tuple[Move, ...]."""
    return functools.lru_cache(maxsize=MOVE_CACHE_SIZE)(user_function)

# ---------- helpers ----------
def hash_board_tensor(t: torch.Tensor) -> int:
    """Fast non-cryptographic hash for (9,9,9,C) tensor."""
    # t is already contiguous after .permute()
    return hash(t.numpy().tobytes())

def hash_coord_list(coords) -> int:
    """Hash a list of coordinates."""
    return hash(tuple(coord_to_idx(c) for c in coords))
