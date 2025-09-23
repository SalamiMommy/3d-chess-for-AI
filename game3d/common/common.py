"""
game3d/common/common.py
Common utilities for 3D 9×9×9 chess game - used by piece movement generators."""

from __future__ import annotations
import torch
import numpy as np
from typing import Tuple, List, Iterable, Optional, Union
from enum import Enum

# ------------------------------------------------------------------
# Board dimensions and constants
# ------------------------------------------------------------------
SIZE_X = 9
SIZE_Y = 9
SIZE_Z = 9
SIZE = 9  # for square dimensions
VOLUME = SIZE_X * SIZE_Y * SIZE_Z  # 729 total squares

# Piece representation constants
N_PIECE_TYPES = 40  # Assuming 42 different piece types based on tensor layout
N_PLANES_PER_SIDE = N_PIECE_TYPES
N_COLOR_PLANES = N_PLANES_PER_SIDE * 2  # White + Black pieces
N_AUX_PLANES = 1  # Current player plane
N_TOTAL_PLANES = N_COLOR_PLANES + N_AUX_PLANES  # 85 total planes

# Axis indices for coordinates (x, y, z)
X, Y, Z = 0, 1, 2

# Coordinate type alias
Coord = Tuple[int, int, int]

# ------------------------------------------------------------------
# Coordinate utilities
# ------------------------------------------------------------------
def in_bounds(c: Coord) -> bool:
    """Check if coordinate is within board bounds (0-8 for each axis)."""
    x, y, z = c
    return 0 <= x < SIZE_X and 0 <= y < SIZE_Y and 0 <= z < SIZE_Z

def coord_to_idx(c: Coord) -> int:
    """Convert 3D coordinate to 1D index (row-major: z, then y, then x)."""
    x, y, z = c
    return z * (SIZE_Y * SIZE_X) + y * SIZE_X + x

def idx_to_coord(idx: int) -> Coord:
    """Convert 1D index back to 3D coordinate."""
    z = idx // (SIZE_Y * SIZE_X)
    remainder = idx % (SIZE_Y * SIZE_X)
    y = remainder // SIZE_X
    x = remainder % SIZE_X
    return (x, y, z)

def add_coords(c1: Coord, c2: Coord) -> Coord:
    """Add two coordinates component-wise."""
    return (c1[0] + c2[0], c1[1] + c2[1], c1[2] + c2[2])

def subtract_coords(c1: Coord, c2: Coord) -> Coord:
    """Subtract two coordinates component-wise."""
    return (c1[0] - c2[0], c1[1] - c2[1], c1[2] - c2[2])

def scale_coord(c: Coord, factor: int) -> Coord:
    """Multiply coordinate by scalar factor."""
    return (c[0] * factor, c[1] * factor, c[2] * factor)

def manhattan_distance(c1: Coord, c2: Coord) -> int:
    """Calculate Manhattan distance between two coordinates."""
    return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2])

def chebyshev_distance(c1: Coord, c2: Coord) -> int:
    """Calculate Chebyshev distance between two coordinates."""
    return max(abs(c1[0] - c2[0]), abs(c1[1] - c2[1]), abs(c1[2] - c2[2]))

def euclidean_distance_squared(c1: Coord, c2: Coord) -> int:
    """Calculate squared Euclidean distance between two coordinates."""
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    dz = c1[2] - c2[2]
    return dx*dx + dy*dy + dz*dz

# ------------------------------------------------------------------
# Movement generation utilities
# ------------------------------------------------------------------
def generate_ray(
    start: Coord,
    direction: Tuple[int, int, int],
    max_steps: Optional[int] = None
) -> List[Coord]:
    """
    Generate coordinates along a ray from start in given direction.
    Stops at board edge or after max_steps.
    """
    ray = []
    current = start
    steps = 0

    while True:
        current = add_coords(current, direction)
        if not in_bounds(current):
            break
        if max_steps is not None and steps >= max_steps:
            break

        ray.append(current)
        steps += 1

    return ray

def generate_sliding_moves(
    start: Coord,
    directions: List[Tuple[int, int, int]],
    board: Optional[object] = None,  # Accepts Board object
    can_capture: bool = True,
    can_move_to_empty: bool = True,
    max_distance: Optional[int] = None
) -> List[Coord]:
    """
    Generate sliding moves in multiple directions.
    If board is provided, stops at first occupied square (captures allowed based on can_capture).
    """
    moves = []

    for direction in directions:
        current = start
        distance = 0

        while True:
            current = add_coords(current, direction)
            if not in_bounds(current):
                break
            if max_distance is not None and distance >= max_distance:
                break

            # If board is provided, check occupancy
            if board is not None:
                piece = board.piece_at(current) if hasattr(board, 'piece_at') else None
                if piece is not None:
                    if can_capture:
                        moves.append(current)
                    break  # Can't move through pieces
                else:
                    if can_move_to_empty:
                        moves.append(current)
            else:
                # No board provided, just generate all possible moves in bounds
                moves.append(current)

            distance += 1

    return moves

def generate_step_moves(
    start: Coord,
    steps: List[Tuple[int, int, int]],
    board: Optional[object] = None,
    can_capture: bool = True,
    can_move_to_empty: bool = True
) -> List[Coord]:
    """
    Generate moves that are single steps (like knight moves).
    """
    moves = []

    for step in steps:
        target = add_coords(start, step)
        if not in_bounds(target):
            continue

        # Check board constraints if provided
        if board is not None:
            piece = board.piece_at(target) if hasattr(board, 'piece_at') else None
            if piece is not None:
                if can_capture:
                    moves.append(target)
            else:
                if can_move_to_empty:
                    moves.append(target)
        else:
            moves.append(target)

    return moves

def get_between_squares(start: Coord, end: Coord) -> List[Coord]:
    """
    Get all squares between start and end (inclusive of neither).
    Returns empty list if not on same rank, file, or diagonal.
    """
    between = []

    # Calculate direction vector
    diff = subtract_coords(end, start)

    # Check if movement is orthogonal or diagonal
    axes_with_movement = sum(1 for d in diff if d != 0)
    if axes_with_movement == 0:
        return between  # Same square

    # Check if it's a valid sliding path (all non-zero components have same absolute value for diagonals,
    # or only one non-zero component for orthogonals)
    non_zero_components = [d for d in diff if d != 0]
    abs_components = [abs(d) for d in non_zero_components]

    # For orthogonal moves: exactly one non-zero component
    # For diagonal moves: all non-zero components have same absolute value
    is_orthogonal = len(non_zero_components) == 1
    is_diagonal = len(set(abs_components)) == 1 if non_zero_components else False

    if not (is_orthogonal or is_diagonal):
        return between  # Not a straight line

    # Calculate step direction
    step = tuple(1 if d > 0 else -1 if d < 0 else 0 for d in diff)
    distance = max(abs_components) if non_zero_components else 0

    # Generate squares between start and end
    for i in range(1, distance):
        between_coord = add_coords(start, scale_coord(step, i))
        if in_bounds(between_coord):
            between.append(between_coord)
        else:
            break

    return between

# ------------------------------------------------------------------
# Board analysis utilities
# ------------------------------------------------------------------
def is_path_clear(
    board: object,  # Accepts Board object
    start: Coord,
    end: Coord
) -> bool:
    """
    Check if path between start and end is clear of pieces.
    Assumes start and end are on a valid sliding path.
    """
    between_squares = get_between_squares(start, end)
    for square in between_squares:
        if board.piece_at(square) is not None:
            return False
    return True

def get_attackers(
    board: object,  # Accepts Board object
    target: Coord,
    attacking_color: Optional[object] = None  # Accepts Color enum
) -> List[Tuple[Coord, object]]:  # Returns list of (coord, piece)
    """
    Get all pieces attacking the target square.
    If attacking_color is provided, only return attackers of that color.
    """
    attackers = []

    # This is a simplified version - in practice, you'd need to know
    # piece movement rules to accurately determine attackers
    # For now, we'll iterate through all pieces and check if they can attack

    if not hasattr(board, 'list_occupied'):
        return attackers

    for coord, piece in board.list_occupied():
        if attacking_color is not None and piece.color != attacking_color:
            continue

        # TODO: Implement actual attack detection based on piece type
        # This would require importing piece movement logic
        # For now, this is a placeholder
        pass

    return attackers

# ------------------------------------------------------------------
# Tensor utilities for zero-copy operations
# ------------------------------------------------------------------
_tensor_cache = {}  # Simple cache for tensor operations

def hash_board_tensor(tensor: torch.Tensor) -> int:
    """
    Fast hash for board tensor for caching purposes.
    Uses tensor data to generate consistent hash.
    """
    # Use tensor's data pointer and shape for fast hashing
    # This is not cryptographically secure but fast for caching
    data_ptr = tensor.data_ptr()
    shape_hash = hash(tensor.shape)
    return hash((data_ptr, shape_hash)) % (2**63 - 1)

# Cache for frequently used tensors
tensor_cache = _tensor_cache

# ------------------------------------------------------------------
# Layer utilities for board tensor
# ------------------------------------------------------------------
def get_piece_plane_indices(color: int, piece_type: int) -> int:
    """
    Get the plane index for a specific piece type and color.
    Color: 0 for white, 1 for black (or use enum values)
    """
    if color == 0:  # White
        return piece_type
    else:  # Black
        return N_PLANES_PER_SIDE + piece_type

def get_color_from_plane(plane_idx: int) -> int:
    """Get color (0=white, 1=black) from plane index."""
    return 0 if plane_idx < N_PLANES_PER_SIDE else 1

def get_piece_type_from_plane(plane_idx: int) -> int:
    """Get piece type from plane index."""
    if plane_idx < N_PLANES_PER_SIDE:
        return plane_idx
    else:
        return plane_idx - N_PLANES_PER_SIDE

# ------------------------------------------------------------------
# Geometric utilities
# ------------------------------------------------------------------
def get_layer_coords(layer_type: str, index: int) -> List[Coord]:
    """
    Get all coordinates in a specific layer.
    layer_type: 'x', 'y', 'z', 'xy_diag', 'xz_diag', 'yz_diag', 'space_diag'
    """
    coords = []

    if layer_type == 'x':
        # All coords with x = index
        for y in range(SIZE_Y):
            for z in range(SIZE_Z):
                coords.append((index, y, z))
    elif layer_type == 'y':
        # All coords with y = index
        for x in range(SIZE_X):
            for z in range(SIZE_Z):
                coords.append((x, index, z))
    elif layer_type == 'z':
        # All coords with z = index
        for x in range(SIZE_X):
            for y in range(SIZE_Y):
                coords.append((x, y, index))
    elif layer_type == 'xy_diag':
        # Diagonal in XY plane at height z=index
        for i in range(min(SIZE_X, SIZE_Y)):
            coords.append((i, i, index))
            if i != SIZE_X - 1 - i:  # Other diagonal
                coords.append((SIZE_X - 1 - i, i, index))
    elif layer_type == 'xz_diag':
        # Diagonal in XZ plane at y=index
        for i in range(min(SIZE_X, SIZE_Z)):
            coords.append((i, index, i))
            if i != SIZE_X - 1 - i:
                coords.append((SIZE_X - 1 - i, index, i))
    elif layer_type == 'yz_diag':
        # Diagonal in YZ plane at x=index
        for i in range(min(SIZE_Y, SIZE_Z)):
            coords.append((index, i, i))
            if i != SIZE_Y - 1 - i:
                coords.append((index, SIZE_Y - 1 - i, i))
    elif layer_type == 'space_diag':
        # 3D space diagonals passing through point (index, index, index) if valid
        if 0 <= index < SIZE:
            # Main space diagonal
            for i in range(SIZE):
                coords.append((i, i, i))
            # Other space diagonals
            for i in range(SIZE):
                coords.append((i, i, SIZE-1-i))
                coords.append((i, SIZE-1-i, i))
                coords.append((SIZE-1-i, i, i))

    return [c for c in coords if in_bounds(c)]

def get_distance_layers(start: Coord, max_distance: int) -> List[List[Coord]]:
    """
    Get layers of coordinates by Chebyshev distance from start.
    Returns list of lists, where index is distance.
    """
    layers = [[] for _ in range(max_distance + 1)]

    for x in range(SIZE_X):
        for y in range(SIZE_Y):
            for z in range(SIZE_Z):
                coord = (x, y, z)
                dist = chebyshev_distance(start, coord)
                if dist <= max_distance:
                    layers[dist].append(coord)

    return layers

# ------------------------------------------------------------------
# Validation and debugging utilities
# ------------------------------------------------------------------
def validate_coord(c: Coord) -> bool:
    """Validate coordinate format and bounds."""
    if not isinstance(c, tuple) or len(c) != 3:
        return False
    if not all(isinstance(i, int) for i in c):
        return False
    return in_bounds(c)

def coord_to_string(c: Coord) -> str:
    """Convert coordinate to human-readable string."""
    return f"({c[0]}, {c[1]}, {c[2]})"

def batch_coords_to_tensor(coords: List[Coord]) -> torch.Tensor:
    """Convert list of coordinates to tensor of shape (N, 3)."""
    if not coords:
        return torch.empty((0, 3), dtype=torch.long)
    return torch.tensor(coords, dtype=torch.long)

def tensor_to_batch_coords(tensor: torch.Tensor) -> List[Coord]:
    """Convert tensor of shape (N, 3) to list of coordinates."""
    return [tuple(coord.tolist()) for coord in tensor]

# ------------------------------------------------------------------
# Performance optimized utilities
# ------------------------------------------------------------------
def create_occupancy_mask_tensor(board_tensor: torch.Tensor) -> torch.Tensor:
    """Create boolean occupancy mask directly from board tensor."""
    white_pieces = board_tensor[:N_PLANES_PER_SIDE].sum(dim=0)  # Sum over piece types
    black_pieces = board_tensor[N_PLANES_PER_SIDE:N_COLOR_PLANES].sum(dim=0)
    return (white_pieces + black_pieces) > 0

def get_current_player(board_tensor: torch.Tensor) -> int:
    """Get current player from board tensor (1=white, 0=black)."""
    return int(board_tensor[N_COLOR_PLANES, 0, 0, 0] > 0.5)

def find_pieces_by_type(board_tensor: torch.Tensor, piece_type: int, color: int) -> List[Coord]:
    """
    Find all pieces of given type and color.
    Optimized using tensor operations.
    """
    if color == 0:  # White
        plane_idx = piece_type
    else:  # Black
        plane_idx = N_PLANES_PER_SIDE + piece_type

    # Get the specific plane
    plane = board_tensor[plane_idx]

    # Find all positions where value is 1.0
    occupied_positions = torch.nonzero(plane == 1.0, as_tuple=False)

    # Convert to list of coordinates (x, y, z)
    coords = []
    for pos in occupied_positions:
        z, y, x = pos.tolist()  # Note: tensor indices are (z, y, x)
        coords.append((x, y, z))

    return coords
