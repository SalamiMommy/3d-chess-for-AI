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

from game3d.common.common import (
    SIZE_X, SIZE_Y, SIZE_Z, N_TOTAL_PLANES, N_COLOR_PLANES
)
from game3d.pieces.enums import Color, PieceType

class RotationType(Enum):
    """24 ROTATIONAL symmetry operations only (full octahedral group)."""
    IDENTITY = "identity"

    # Face rotations (9 total)
    ROTATE_X_90 = "rotate_x_90"
    ROTATE_X_180 = "rotate_x_180"
    ROTATE_X_270 = "rotate_x_270"
    ROTATE_Y_90 = "rotate_y_90"
    ROTATE_Y_180 = "rotate_y_180"
    ROTATE_Y_270 = "rotate_y_270"
    ROTATE_Z_90 = "rotate_z_90"
    ROTATE_Z_180 = "rotate_z_180"
    ROTATE_Z_270 = "rotate_z_270"

    # Vertex rotations (8 total) - 120° around space diagonals
    ROTATE_XYZ_120 = "rotate_xyz_120"
    ROTATE_XYZ_240 = "rotate_xyz_240"
    ROTATE_XYmZ_120 = "rotate_xymz_120"
    ROTATE_XYmZ_240 = "rotate_xymz_240"
    ROTATE_XmYZ_120 = "rotate_xmyz_120"
    ROTATE_XmYZ_240 = "rotate_xmyz_240"
    ROTATE_XmYmZ_120 = "rotate_xmymz_120"
    ROTATE_XmYmZ_240 = "rotate_xmymz_240"

    # Edge rotations (6 total) - 180° around edge midpoints
    ROTATE_XY_EDGE = "rotate_xy_edge"
    ROTATE_XmY_EDGE = "rotate_xmy_edge"
    ROTATE_XZ_EDGE = "rotate_xz_edge"
    ROTATE_XmZ_EDGE = "rotate_xmz_edge"
    ROTATE_YZ_EDGE = "rotate_yz_edge"
    ROTATE_YmZ_EDGE = "rotate_ymz_edge"

@dataclass
class TensorTransformation:
    """Represents a tensor ROTATIONAL transformation (no reflections)."""
    name: str
    transform_fn: Callable[[torch.Tensor], torch.Tensor]
    inverse_fn: Callable[[torch.Tensor], torch.Tensor]
    hash_multiplier: int

class SymmetryManager:
    _rotation_matrices = None
    _initialized = False

    def __init__(self):
        if not SymmetryManager._initialized:
            SymmetryManager._rotation_matrices = self._build_rotation_matrices()
            SymmetryManager._initialized = True
        self.rotation_matrices = SymmetryManager._rotation_matrices
        self.size_x = SIZE_X
        self.size_y = SIZE_Y
        self.size_z = SIZE_Z
        self.n_planes = N_TOTAL_PLANES

        # Pre-compute all 24 rotational transformations
        self.transformations = self._init_all_rotational_transformations()
        self.canonical_cache = {}
        self.rotation_matrices = self._build_rotation_matrices()

        # Performance tracking
        self.symmetry_operations = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def _build_rotation_matrices(self) -> Dict[RotationType, np.ndarray]:
        """Build matrices for all 24 rotational symmetries."""
        matrices = {}

        # Identity
        matrices[RotationType.IDENTITY] = np.eye(3, dtype=np.int32)

        # Face rotations
        matrices[RotationType.ROTATE_X_90] = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=np.int32)
        matrices[RotationType.ROTATE_X_180] = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.int32)
        matrices[RotationType.ROTATE_X_270] = np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=np.int32)

        matrices[RotationType.ROTATE_Y_90] = np.array([[0,0,1],[0,1,0],[-1,0,0]], dtype=np.int32)
        matrices[RotationType.ROTATE_Y_180] = np.array([[-1,0,0],[0,1,0],[0,0,-1]], dtype=np.int32)
        matrices[RotationType.ROTATE_Y_270] = np.array([[0,0,-1],[0,1,0],[1,0,0]], dtype=np.int32)

        matrices[RotationType.ROTATE_Z_90] = np.array([[0,-1,0],[1,0,0],[0,0,1]], dtype=np.int32)
        matrices[RotationType.ROTATE_Z_180] = np.array([[-1,0,0],[0,-1,0],[0,0,1]], dtype=np.int32)
        matrices[RotationType.ROTATE_Z_270] = np.array([[0,1,0],[-1,0,0],[0,0,1]], dtype=np.int32)

        # Vertex rotations (120° around diagonals)
        matrices[RotationType.ROTATE_XYZ_120] = np.array([[0,0,1],[1,0,0],[0,1,0]], dtype=np.int32)
        matrices[RotationType.ROTATE_XYZ_240] = np.array([[0,1,0],[0,0,1],[1,0,0]], dtype=np.int32)
        matrices[RotationType.ROTATE_XYmZ_120] = np.array([[0,0,-1],[1,0,0],[0,-1,0]], dtype=np.int32)
        matrices[RotationType.ROTATE_XYmZ_240] = np.array([[0,-1,0],[0,0,-1],[1,0,0]], dtype=np.int32)
        matrices[RotationType.ROTATE_XmYZ_120] = np.array([[0,0,1],[-1,0,0],[0,-1,0]], dtype=np.int32)
        matrices[RotationType.ROTATE_XmYZ_240] = np.array([[0,-1,0],[0,0,1],[-1,0,0]], dtype=np.int32)
        matrices[RotationType.ROTATE_XmYmZ_120] = np.array([[0,0,-1],[-1,0,0],[0,1,0]], dtype=np.int32)
        matrices[RotationType.ROTATE_XmYmZ_240] = np.array([[0,1,0],[0,0,-1],[-1,0,0]], dtype=np.int32)

        # Edge rotations (180° around edges)
        matrices[RotationType.ROTATE_XY_EDGE] = np.array([[0,1,0],[1,0,0],[0,0,-1]], dtype=np.int32)
        matrices[RotationType.ROTATE_XmY_EDGE] = np.array([[0,-1,0],[-1,0,0],[0,0,-1]], dtype=np.int32)
        matrices[RotationType.ROTATE_XZ_EDGE] = np.array([[0,0,1],[0,-1,0],[1,0,0]], dtype=np.int32)
        matrices[RotationType.ROTATE_XmZ_EDGE] = np.array([[0,0,-1],[0,-1,0],[-1,0,0]], dtype=np.int32)
        matrices[RotationType.ROTATE_YZ_EDGE] = np.array([[-1,0,0],[0,0,1],[0,1,0]], dtype=np.int32)
        matrices[RotationType.ROTATE_YmZ_EDGE] = np.array([[-1,0,0],[0,0,-1],[0,-1,0]], dtype=np.int32)

        return matrices

    def _init_all_rotational_transformations(self) -> List[TensorTransformation]:
        """Initialize all 24 rotational transformations."""
        transformations = []
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                 53, 59, 61, 67, 71, 73, 79, 83, 89]

        # Create transformations for each rotation type
        for i, rotation_type in enumerate(RotationType):
            transform_fn, inverse_fn = self._create_rotation_functions(rotation_type)

            transformations.append(TensorTransformation(
                name=rotation_type.value,
                transform_fn=transform_fn,
                inverse_fn=inverse_fn,
                hash_multiplier=primes[i]
            ))

        return transformations

    def _create_rotation_functions(self, rotation_type: RotationType) -> Tuple[Callable, Callable]:
        """Create transform and inverse functions for a specific rotation type."""
        if rotation_type == RotationType.IDENTITY:
            return lambda t: t, lambda t: t

        # For all rotations, use coordinate mapping that works with (C, Z, Y, X) format
        def transform_fn(tensor):
            return self._apply_rotation_by_type(tensor, rotation_type)

        def inverse_fn(tensor):
            inverse_type = self._get_inverse_rotation(rotation_type)
            return self._apply_rotation_by_type(tensor, inverse_type)

        return transform_fn, inverse_fn

    def _get_inverse_rotation(self, rotation_type: RotationType) -> RotationType:
        """Get the inverse rotation type."""
        inverse_map = {
            RotationType.ROTATE_X_90: RotationType.ROTATE_X_270,
            RotationType.ROTATE_X_270: RotationType.ROTATE_X_90,
            RotationType.ROTATE_Y_90: RotationType.ROTATE_Y_270,
            RotationType.ROTATE_Y_270: RotationType.ROTATE_Y_90,
            RotationType.ROTATE_Z_90: RotationType.ROTATE_Z_270,
            RotationType.ROTATE_Z_270: RotationType.ROTATE_Z_90,
            RotationType.ROTATE_XYZ_120: RotationType.ROTATE_XYZ_240,
            RotationType.ROTATE_XYZ_240: RotationType.ROTATE_XYZ_120,
            RotationType.ROTATE_XYmZ_120: RotationType.ROTATE_XYmZ_240,
            RotationType.ROTATE_XYmZ_240: RotationType.ROTATE_XYmZ_120,
            RotationType.ROTATE_XmYZ_120: RotationType.ROTATE_XmYZ_240,
            RotationType.ROTATE_XmYZ_240: RotationType.ROTATE_XmYZ_120,
            RotationType.ROTATE_XmYmZ_120: RotationType.ROTATE_XmYmZ_240,
            RotationType.ROTATE_XmYmZ_240: RotationType.ROTATE_XmYmZ_120,
        }
        # 180° rotations and edge rotations are their own inverses
        return inverse_map.get(rotation_type, rotation_type)

    def _apply_rotation_by_type(self, tensor: torch.Tensor, rotation_type: RotationType) -> torch.Tensor:
        """Apply a specific rotation type to the tensor using coordinate mapping."""
        if rotation_type == RotationType.IDENTITY:
            return tensor.clone()

        # Get rotation matrix for this rotation type
        R = self.rotation_matrices[rotation_type]
        return self._apply_rotation_matrix(tensor, R)

    def _apply_rotation_matrix(self, tensor: torch.Tensor, R: np.ndarray) -> torch.Tensor:
        """Apply a rotation matrix to the tensor using coordinate mapping."""
        R_tensor = torch.tensor(R, dtype=torch.float32, device=tensor.device)

        # Create coordinate grid for SPATIAL dimensions (Z, Y, X)
        coords = torch.stack(torch.meshgrid(
            torch.arange(self.size_z, device=tensor.device),  # Z dimension
            torch.arange(self.size_y, device=tensor.device),  # Y dimension
            torch.arange(self.size_x, device=tensor.device),  # X dimension
            indexing='ij'
        ), dim=-1).float()  # Shape: (Z, Y, X, 3)

        # Center coordinates
        center = torch.tensor([(self.size_z-1)/2, (self.size_y-1)/2, (self.size_x-1)/2],
                            device=tensor.device)
        centered_coords = coords - center

        # Apply rotation
        rotated_coords = torch.einsum('ij,zyxj->zyxi', R_tensor, centered_coords)
        new_coords = rotated_coords + center

        # Round to nearest integer and clamp to bounds
        new_coords = torch.round(new_coords).long()
        new_coords = torch.clamp(new_coords, 0, torch.tensor([self.size_z-1, self.size_y-1, self.size_x-1],
                                                        device=tensor.device))

        # Create new tensor by gathering from original
        channels = tensor.shape[0]
        result = torch.zeros_like(tensor)

        # For each channel, gather values from rotated coordinates
        for c in range(channels):
            # Use advanced indexing to remap values
            result[c] = tensor[c, new_coords[..., 0], new_coords[..., 1], new_coords[..., 2]]

        return result

    def apply_transformation(self, tensor: torch.Tensor,
                           transformation: TensorTransformation) -> torch.Tensor:
        """Apply rotational transformation to tensor."""
        self.symmetry_operations += 1
        return transformation.transform_fn(tensor)

    def get_symmetric_boards(self, board) -> List[Tuple[str, 'Board']]:
        """Generate all rotationally symmetric variants of a board."""
        from game3d.board.board import Board
        tensor = board.tensor()
        symmetric_boards = []

        for transformation in self.transformations:
            if transformation.name == "identity":
                continue

            cache_key = (hash(tensor.tobytes()), transformation.name)
            if cache_key in self.canonical_cache:
                sym_board = self.canonical_cache[cache_key]
                symmetric_boards.append((transformation.name, sym_board))
                self.cache_hits += 1
                continue

            # Apply transformation to tensor
            transformed_tensor = self.apply_transformation(tensor, transformation)
            sym_board = Board(transformed_tensor)

            symmetric_boards.append((transformation.name, sym_board))
            self.canonical_cache[cache_key] = sym_board
            self.cache_misses += 1

            if len(self.canonical_cache) > 5000:
                oldest_key = next(iter(self.canonical_cache))
                del self.canonical_cache[oldest_key]

        return symmetric_boards

    def get_canonical_form(self, board) -> Tuple['Board', str]:
        """Return canonical form using all 24 rotational symmetries."""
        from game3d.board.board import Board
        tensor = board.tensor()

        cache_key = tensor.data_ptr()
        if cache_key in self.canonical_cache:
            canonical_data = self.canonical_cache[cache_key]
            return canonical_data['board'], canonical_data['transformation']

        # Generate all symmetric variants
        symmetric_variants = self.get_symmetric_boards(board)
        all_variants = [("identity", board)] + symmetric_variants

        # Convert to comparable format
        variant_representations = []
        for name, sym_board in all_variants:
            representation = self._board_to_comparable(sym_board)
            variant_representations.append((representation, name, sym_board))

        # Sort to find "smallest" representation
        variant_representations.sort(key=lambda x: x[0])

        # Return canonical form and transformation used
        canonical_rep, transformation_name, canonical_board = variant_representations[0]

        # Cache the result
        self.canonical_cache[cache_key] = {
            'board': canonical_board,
            'transformation': transformation_name
        }

        return canonical_board, transformation_name

    def _board_to_comparable(self, board) -> Tuple:
        """Convert board to comparable representation."""
        tensor = board.tensor()
        flat_tensor = tensor.flatten()
        first_elements = tuple(int(val) for val in flat_tensor[:100].tolist())  # Use int for better comparison
        total_sum = float(tensor.sum())
        non_zero_count = int((tensor != 0).sum())
        return (first_elements, total_sum, non_zero_count)

    def is_symmetric_position(self, board1, board2) -> bool:
        """Check if two board positions are rotationally equivalent."""
        canonical1, _ = self.get_canonical_form(board1)
        canonical2, _ = self.get_canonical_form(board2)
        return torch.equal(canonical1.tensor(), canonical2.tensor())

    def get_rotation_count(self) -> int:
        """Return the number of rotational symmetries (24)."""
        return len(self.transformations)

    def clear_cache(self) -> None:
        """Clear symmetry cache."""
        self.canonical_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def get_performance_stats(self) -> Dict[str, int]:
        """Get performance statistics."""
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

    # Add these methods to SymmetryManager class:

    def normalize_position(self, position: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Normalize position using board symmetry."""
        x, y, z = position
        center = (self.size_x - 1) // 2

        # Map to first octant using symmetry
        norm_x = min(x, self.size_x - 1 - x) if x < center else max(x, self.size_x - 1 - x)
        norm_y = min(y, self.size_y - 1 - y) if y < center else max(y, self.size_y - 1 - y)
        norm_z = min(z, self.size_z - 1 - z) if z < center else max(z, self.size_z - 1 - z)

        return (norm_x, norm_y, norm_z)

    def create_movement_symmetry_key(self, piece_type: Any, position: Tuple[int, int, int],
                                    color: Any) -> Tuple:
        """Create symmetry-aware cache key for movement generation."""
        norm_pos = self.normalize_position(position)
        return (piece_type, norm_pos, color)

    def get_or_compute_with_symmetry(self, key: Any, compute_fn: Callable,
                                    use_symmetry: bool = True) -> Any:
        """Generic symmetry-aware caching with LRU eviction."""
        if use_symmetry and key in self.canonical_cache:
            self.cache_hits += 1
            return self.canonical_cache[key]

        self.cache_misses += 1
        result = compute_fn()

        if use_symmetry:
            self.canonical_cache[key] = result
            # LRU eviction if needed
            if len(self.canonical_cache) > 5000:
                oldest_key = next(iter(self.canonical_cache))
                del self.canonical_cache[oldest_key]

        return result
