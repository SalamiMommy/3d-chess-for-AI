#game3d/common/common.py
#game3d/common/common.py
# ------------------------------------------------------------------
# Coordinate utilities – optimised drop-in replacements
# ------------------------------------------------------------------
from __future__ import annotations
import torch
from typing import Tuple, List, Optional
from game3d.pieces.enums import PieceType


Coord = Tuple[int, int, int]

# Board geometry
SIZE = SIZE_X = SIZE_Y = SIZE_Z = 9
VOLUME = SIZE ** 3

# ----------------------------------------------------------
# NEW colour-aware channel budget (41 planes total)

N_PIECE_TYPES  = 40          # unchanged
N_COLOR_PLANES = 1           # unchanged
N_TOTAL_PLANES = 82
N_PLANES_PER_SIDE = N_PIECE_TYPES + N_COLOR_PLANES
# ------------------------------------------

# slices the rest of the code expects
PIECE_SLICE       = slice(0, N_PIECE_TYPES)
COLOR_SLICE       = slice(N_PIECE_TYPES, N_PIECE_TYPES + N_COLOR_PLANES)
CURRENT_SLICE     = slice(N_PIECE_TYPES + N_COLOR_PLANES, N_PIECE_TYPES + N_COLOR_PLANES + 1)
EFFECT_SLICE      = slice(N_PIECE_TYPES + N_COLOR_PLANES + 1, N_PIECE_TYPES + N_COLOR_PLANES + 7)

# ------------------------------------------------------------------
# Fast bounds check – branch-free, no Python-level tuple unpacking
# ------------------------------------------------------------------
_UPPER = SIZE - 1  # 8
def in_bounds(c: Coord) -> bool:
    """Branch-free bounds check for 9×9×9 board."""
    # All comparisons are unsigned after the shift, so a single & gives the answer.
    return ((c[0] | c[1] | c[2]) & ~_UPPER) == 0

# ------------------------------------------------------------------
# Helpers – avoid tuple unpacking in the inner loop
# ------------------------------------------------------------------
def coord_to_idx(c: Coord) -> int:
    # 9×9×9  →  z*81 + y*9 + x
    return (c[2] * 81) + (c[1] * 9) + c[0]

def idx_to_coord(idx: int) -> Coord:
    z, r = divmod(idx, 81)
    y, x = divmod(r, 9)
    return x, y, z

def add_coords(a: Coord, b: Coord) -> Coord:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]

def subtract_coords(a: Coord, b: Coord) -> Coord:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]

def scale_coord(c: Coord, k: int) -> Coord:
    return c[0] * k, c[1] * k, c[2] * k

# ------------------------------------------------------------------
# Distance helpers – kept as one-liners, already fast
# ------------------------------------------------------------------
manhattan_distance      = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])
chebyshev_distance      = lambda a, b: max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))
euclidean_distance_squared = lambda a, b: (a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2

# ------------------------------------------------------------------
# Ray generation (keep — useful for some pieces)
# ------------------------------------------------------------------
def generate_ray(
    start: Coord,
    direction: Tuple[int, int, int],
    max_steps: Optional[int] = None
) -> List[Coord]:
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

# ------------------------------------------------------------------
# Path utilities (fixed)
# ------------------------------------------------------------------
def get_between_squares(start: Coord, end: Coord) -> List[Coord]:
    """Get squares strictly between start and end on a straight line."""
    if start == end:
        return []

    diff = subtract_coords(end, start)
    non_zero = [d for d in diff if d != 0]
    if not non_zero:
        return []

    # Check if straight line: orthogonal or diagonal
    abs_vals = [abs(d) for d in non_zero]
    is_orthogonal = len(non_zero) == 1
    is_diagonal = len(set(abs_vals)) == 1

    if not (is_orthogonal or is_diagonal):
        return []

    step = tuple(0 if d == 0 else (1 if d > 0 else -1) for d in diff)
    distance = max(abs_vals)

    between = []
    for i in range(1, distance):
        sq = add_coords(start, scale_coord(step, i))
        if in_bounds(sq):
            between.append(sq)
        else:
            break
    return between

def is_path_clear(board, start: Coord, end: Coord) -> bool:
    """Check if path between start and end is clear."""
    for sq in get_between_squares(start, end):
        piece = board.piece_at(sq)
        if piece is not None:
            return False
    return True

# ------------------------------------------------------------------
# Tensor utilities (updated for new structure)
# ------------------------------------------------------------------
def hash_board_tensor(tensor: torch.Tensor) -> int:
    """Content-based hash — slow but correct."""
    # For performance, you might want to use a rolling hash in Board class instead
    return hash(tensor.cpu().numpy().tobytes())

def create_occupancy_mask_tensor(board_tensor: torch.Tensor) -> torch.Tensor:
    """Create boolean occupancy mask from board tensor."""
    pieces = board_tensor[PIECE_SLICE].sum(dim=0)
    return pieces > 0

def get_current_player(board_tensor: torch.Tensor) -> int:
    """Get current player (1=white, 0=black)."""
    return int(board_tensor[N_PIECE_TYPES + N_COLOR_PLANES, 0, 0, 0].item() > 0.5)

def get_piece_color(board_tensor: torch.Tensor, coord: Coord) -> Optional[int]:
    """Get color of piece at coordinate (1=white, 0=black, None=empty)."""
    x, y, z = coord
    # Check if any piece exists at this coordinate
    piece_planes = board_tensor[PIECE_SLICE, z, y, x]
    if piece_planes.sum() == 0:
        return None

    # Get color from color mask
    color_value = board_tensor[N_PIECE_TYPES, z, y, x].item()
    return 1 if color_value > 0.5 else 0

def find_pieces_by_type(board_tensor: torch.Tensor, piece_type: int, color: int) -> List[Coord]:
    """Find all pieces of given type and color."""
    piece_plane = board_tensor[piece_type]
    color_plane = board_tensor[N_PIECE_TYPES]  # color mask

    # Find positions where piece exists
    piece_positions = piece_plane > 0.5

    # Filter by color
    if color == 1:  # white
        color_positions = color_plane > 0.5
    else:  # black
        color_positions = color_plane <= 0.5

    # Combine conditions
    target_positions = piece_positions & color_positions
    positions = torch.nonzero(target_positions, as_tuple=False)
    return [(int(x), int(y), int(z)) for z, y, x in positions.tolist()]

def find_all_pieces_of_color(board_tensor: torch.Tensor, color: int) -> List[Tuple[int, Coord]]:
    """Find all pieces of given color, returning (piece_type, coordinate) pairs."""
    color_plane = board_tensor[N_PIECE_TYPES]  # color mask

    # Filter by color
    if color == 1:  # white
        color_positions = color_plane > 0.5
    else:  # black
        color_positions = color_plane <= 0.5

    results = []
    for piece_type in range(N_PIECE_TYPES):
        piece_plane = board_tensor[piece_type]
        # Find positions where both piece exists and color matches
        target_positions = (piece_plane > 0.5) & color_positions
        positions = torch.nonzero(target_positions, as_tuple=False)
        for z, y, x in positions.tolist():
            results.append((piece_type, (int(x), int(y), int(z))))

    return results

# ------------------------------------------------------------------
# Geometric utilities (fixed space_diag)
# ------------------------------------------------------------------
def get_layer_coords(layer_type: str, index: int) -> List[Coord]:
    coords = []
    if layer_type == 'x':
        coords = [(index, y, z) for y in range(SIZE) for z in range(SIZE)]
    elif layer_type == 'y':
        coords = [(x, index, z) for x in range(SIZE) for z in range(SIZE)]
    elif layer_type == 'z':
        coords = [(x, y, index) for x in range(SIZE) for y in range(SIZE)]
    elif layer_type == 'xy_diag':
        coords = [(i, i, index) for i in range(SIZE)]
        coords += [(SIZE-1-i, i, index) for i in range(SIZE) if i != SIZE-1-i]
    elif layer_type == 'xz_diag':
        coords = [(i, index, i) for i in range(SIZE)]
        coords += [(SIZE-1-i, index, i) for i in range(SIZE) if i != SIZE-1-i]
    elif layer_type == 'yz_diag':
        coords = [(index, i, i) for i in range(SIZE)]
        coords += [(index, SIZE-1-i, i) for i in range(SIZE) if i != SIZE-1-i]
    elif layer_type == 'space_diag':
        # All 4 main space diagonals (ignore index)
        coords = [(i, i, i) for i in range(SIZE)]
        coords += [(i, i, SIZE-1-i) for i in range(SIZE)]
        coords += [(i, SIZE-1-i, i) for i in range(SIZE)]
        coords += [(SIZE-1-i, i, i) for i in range(SIZE)]
    return [c for c in coords if in_bounds(c)]

def get_distance_layers(start: Coord, max_distance: int) -> List[List[Coord]]:
    layers = [[] for _ in range(max_distance + 1)]
    for x in range(SIZE):
        for y in range(SIZE):
            for z in range(SIZE):
                dist = chebyshev_distance(start, (x, y, z))
                if dist <= max_distance:
                    layers[dist].append((x, y, z))
    return layers

N_CHANNELS = N_TOTAL_PLANES

# ------------------------------------------------------------------
# Validation and debugging
# ------------------------------------------------------------------
def validate_coord(c: Coord) -> bool:
    return isinstance(c, tuple) and len(c) == 3 and all(isinstance(i, int) for i in c) and in_bounds(c)

def coord_to_string(c: Coord) -> str:
    return f"({c[0]}, {c[1]}, {c[2]})"

def batch_coords_to_tensor(coords: List[Coord]) -> torch.Tensor:
    return torch.tensor(coords, dtype=torch.long) if coords else torch.empty((0, 3), dtype=torch.long)

def tensor_to_batch_coords(tensor: torch.Tensor) -> List[Coord]:
    return [tuple(row.tolist()) for row in tensor]
