"""Top-level common API – importable from *any* package."""

# constants
from .constants import (
    SIZE_X, SIZE_Y, SIZE_Z, SIZE, VOLUME,
    N_PIECE_TYPES, N_PLANES_PER_SIDE, N_COLOR_PLANES,
    N_AUX_PLANES, N_TOTAL_PLANES,
    X, Y, Z,
)

# geometry
from .geometry import (
    Coord, in_bounds, clamp, add, sub, scale, manhattan,
    ray_iter, apply_matrix, identity, coord_to_idx, idx_to_coord,
)

# cache
from .cache import (
    tensor_cache, move_cache, hash_board_tensor, hash_coord_list,
    TENSOR_CACHE_SIZE, MOVE_CACHE_SIZE,
)

# typing
from .typing import MoveTuple, PlaneIndex

__all__ = [
    # constants
    "SIZE_X", "SIZE_Y", "SIZE_Z", "SIZE", "VOLUME",
    "N_PIECE_TYPES", "N_PLANES_PER_SIDE", "N_COLOR_PLANES",
    "N_AUX_PLANES", "N_TOTAL_PLANES",
    "X", "Y", "Z",
    # geometry
    "Coord", "in_bounds", "clamp", "add", "sub", "scale", "manhattan",
    "ray_iter", "apply_matrix", "identity", "coord_to_idx", "idx_to_coord",
    # cache
    "tensor_cache", "move_cache", "hash_board_tensor", "hash_coord_list",
    "TENSOR_CACHE_SIZE", "MOVE_CACHE_SIZE",
    # typing
    "MoveTuple", "PlaneIndex",
]
