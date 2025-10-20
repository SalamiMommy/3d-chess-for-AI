from __future__ import annotations
"""
game3d/board/symmetry.py
Tensor-based ROTATIONAL symmetry operations for 9x9x9 3D chess board.
Compatible with Board class tensor format: (N_TOTAL_PLANES, Z, Y, X)
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Set, Callable, Any
from enum import Enum
from dataclasses import dataclass
from functools import lru_cache

from game3d.common.common import (
    SIZE_X, SIZE_Y, SIZE_Z, N_TOTAL_PLANES, N_COLOR_PLANES
)
from game3d.common.enums import Color, PieceType

class RotationType(Enum):
    IDENTITY = "identity"
    ROTATE_X_90 = "rotate_x_90"
    ROTATE_X_180 = "rotate_x_180"
    ROTATE_X_270 = "rotate_x_270"
    ROTATE_Y_90 = "rotate_y_90"
    ROTATE_Y_180 = "rotate_y_180"
    ROTATE_Y_270 = "rotate_y_270"
    ROTATE_Z_90 = "rotate_z_90"
    ROTATE_Z_180 = "rotate_z_180"
    ROTATE_Z_270 = "rotate_z_270"
    ROTATE_XYZ_120 = "rotate_xyz_120"
    ROTATE_XYZ_240 = "rotate_xyz_240"
    ROTATE_XYmZ_120 = "rotate_xymz_120"
    ROTATE_XYmZ_240 = "rotate_xymz_240"
    ROTATE_XmYZ_120 = "rotate_xmyz_120"
    ROTATE_XmYZ_240 = "rotate_xmyz_240"
    ROTATE_XmYmZ_120 = "rotate_xmymz_120"
    ROTATE_XmYmZ_240 = "rotate_xmymz_240"
    ROTATE_XY_EDGE = "rotate_xy_edge"
    ROTATE_XmY_EDGE = "rotate_xmy_edge"
    ROTATE_XZ_EDGE = "rotate_xz_edge"
    ROTATE_XmZ_EDGE = "rotate_xmz_edge"
    ROTATE_YZ_EDGE = "rotate_yz_edge"
    ROTATE_YmZ_EDGE = "rotate_ymz_edge"

@dataclass
class TensorTransformation:
    name: str
    transform_fn: Callable[[torch.Tensor], torch.Tensor]
    inverse_fn: Callable[[torch.Tensor], torch.Tensor]
    hash_multiplier: int

class SymmetryManager:
    rotation_matrices = {  # Precompute as class attr
        RotationType.IDENTITY: np.eye(3, dtype=np.int32),
        RotationType.ROTATE_X_90: np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=np.int32),
        RotationType.ROTATE_X_180: np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.int32),
        RotationType.ROTATE_X_270: np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=np.int32),
        RotationType.ROTATE_Y_90: np.array([[0,0,1],[0,1,0],[-1,0,0]], dtype=np.int32),
        RotationType.ROTATE_Y_180: np.array([[-1,0,0],[0,1,0],[0,0,-1]], dtype=np.int32),
        RotationType.ROTATE_Y_270: np.array([[0,0,-1],[0,1,0],[1,0,0]], dtype=np.int32),
        RotationType.ROTATE_Z_90: np.array([[0,-1,0],[1,0,0],[0,0,1]], dtype=np.int32),
        RotationType.ROTATE_Z_180: np.array([[-1,0,0],[0,-1,0],[0,0,1]], dtype=np.int32),
        RotationType.ROTATE_Z_270: np.array([[0,1,0],[-1,0,0],[0,0,1]], dtype=np.int32),
        RotationType.ROTATE_XYZ_120: np.array([[0,0,1],[1,0,0],[0,1,0]], dtype=np.int32),
        RotationType.ROTATE_XYZ_240: np.array([[0,1,0],[0,0,1],[1,0,0]], dtype=np.int32),
        RotationType.ROTATE_XYmZ_120: np.array([[0,0,-1],[1,0,0],[0,-1,0]], dtype=np.int32),
        RotationType.ROTATE_XYmZ_240: np.array([[0,-1,0],[0,0,-1],[1,0,0]], dtype=np.int32),
        RotationType.ROTATE_XmYZ_120: np.array([[0,0,1],[-1,0,0],[0,-1,0]], dtype=np.int32),
        RotationType.ROTATE_XmYZ_240: np.array([[0,-1,0],[0,0,1],[-1,0,0]], dtype=np.int32),
        RotationType.ROTATE_XmYmZ_120: np.array([[0,0,-1],[-1,0,0],[0,1,0]], dtype=np.int32),
        # Complete the rest as per original truncated
    }

    def __init__(self):
        self.size_x = SIZE_X
        self.size_y = SIZE_Y
        self.size_z = SIZE_Z
        self.n_planes = N_TOTAL_PLANES
        self.transformations = self._init_all_rotational_transformations()
        self.canonical_cache = {}
        self.symmetry_operations = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def _init_all_rotational_transformations(self) -> List[TensorTransformation]:
        # Implement initialization
        return []  # Placeholder

    @lru_cache(maxsize=1000)
    def get_canonical_form(self, board) -> Tuple['Board', str]:
        tensor = board.tensor()

        symmetric_variants = self.get_symmetric_boards(board)
        all_variants = [("identity", board)] + symmetric_variants

        variant_representations = []
        for name, sym_board in all_variants:
            representation = self._board_to_comparable(sym_board)
            variant_representations.append((representation, name, sym_board))

        variant_representations.sort(key=lambda x: x[0])

        canonical_rep, transformation_name, canonical_board = variant_representations[0]

        return canonical_board, transformation_name

    def _board_to_comparable(self, board) -> Tuple:
        tensor = board.tensor()
        flat_tensor = tensor.view(-1)
        first_elements = tuple(flat_tensor[:100].int().tolist())
        total_sum = int(tensor.sum().item())  # Avoid float precision
        non_zero_count = int((tensor != 0).sum().item())
        return (first_elements, total_sum, non_zero_count)

    def is_symmetric_position(self, board1, board2) -> bool:
        canonical1, _ = self.get_canonical_form(board1)
        canonical2, _ = self.get_canonical_form(board2)
        return torch.equal(canonical1.tensor(), canonical2.tensor())

    def get_rotation_count(self) -> int:
        return len(self.transformations)

    def clear_cache(self) -> None:
        self.canonical_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def get_performance_stats(self) -> Dict[str, int]:
        total_operations = self.symmetry_operations
        total_cache_accesses = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(1, total_cache_accesses)
        return {
            'symmetry_operations': total_operations,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': hit_rate,
            'cache_size': len(self.canonical_cache),
            'rotation_count': self.get_rotation_count()
        }

    def normalize_position(self, position: Tuple[int, int, int]) -> Tuple[int, int, int]:
        x, y, z = position
        center = (self.size_x - 1) // 2
        norm_x = min(x, self.size_x - 1 - x)
        norm_y = min(y, self.size_y - 1 - y)
        norm_z = min(z, self.size_z - 1 - z)
        return (norm_x, norm_y, norm_z)

    def create_movement_symmetry_key(self, piece_type: Any, position: Tuple[int, int, int],
                                    color: Any) -> Tuple:
        norm_pos = self.normalize_position(position)
        return (piece_type, norm_pos, color)

    def get_or_compute_with_symmetry(self, key: Any, compute_fn: Callable,
                                    use_symmetry: bool = True) -> Any:
        if use_symmetry and key in self.canonical_cache:
            self.cache_hits += 1
            return self.canonical_cache[key]

        self.cache_misses += 1
        result = compute_fn()

        if use_symmetry:
            self.canonical_cache[key] = result
            if len(self.canonical_cache) > 5000:
                oldest_key = next(iter(self.canonical_cache))
                del self.canonical_cache[oldest_key]

        return result

    def tensor(self) -> torch.Tensor:
        # Placeholder for board.tensor()
        return torch.zeros(1)
