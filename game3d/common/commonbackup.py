# game3d/common/common.py
# ------------------------------------------------------------------
# Coordinate utilities – optimised drop-in replacements
# ------------------------------------------------------------------
from __future__ import annotations
import torch
import math
from typing import Tuple, List, Optional, TYPE_CHECKING, Iterable, Dict, Any, Union
from dataclasses import dataclass, field
import time
from contextlib import contextmanager
from game3d.common.enums import Color, PieceType
from game3d.pieces.piece import Piece
import numpy as np
from numba import njit
from functools import wraps

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
    from game3d.movement.movepiece import Move
    from game3d.board.board import Board

Coord = Tuple[int, int, int]

# Board geometry
SIZE = SIZE_X = SIZE_Y = SIZE_Z = 9
VOLUME = SIZE ** 3

# ----------------------------------------------------------
# NEW colour-aware channel budget (41 planes total)

N_PIECE_TYPES  = 40          # unchanged
N_COLOR_PLANES = 2           # unchanged
N_TOTAL_PLANES = 82
N_PLANES_PER_SIDE = N_PIECE_TYPES + N_COLOR_PLANES
# ------------------------------------------

# slices the rest of the code expects
PIECE_SLICE = slice(0, 2 * N_PIECE_TYPES)
COLOR_SLICE = slice(2 * N_PIECE_TYPES, 2 * N_PIECE_TYPES + N_COLOR_PLANES)  # 80-82
CURRENT_SLICE = slice(2 * N_PIECE_TYPES + N_COLOR_PLANES, 2 * N_PIECE_TYPES + N_COLOR_PLANES + 1)  # 82-83
EFFECT_SLICE = slice(2 * N_PIECE_TYPES + N_COLOR_PLANES + 1, 2 * N_PIECE_TYPES + N_COLOR_PLANES + 1 + 6)  # 83-89

# ------------------------------------------------------------------
# Fast bounds check – branch-free, no Python-level tuple unpacking
# ------------------------------------------------------------------
_UPPER = SIZE - 1  # 8

@njit
def in_bounds(c: Coord) -> bool:
    return (0 <= c[0] < SIZE) and (0 <= c[1] < SIZE) and (0 <= c[2] < SIZE)

def in_bounds_vectorised(coords: np.ndarray) -> np.ndarray:
    """
    Return a 1-D bool array: True for every row whose x,y,z are all in 0-8.
    coords: int array shape (N, 3)  or  (3,)
    """
    coords = np.atleast_2d(coords)          # (3,)  →  (1,3)
    return np.all((coords >= 0) & (coords < SIZE), axis=1)

@njit(cache=True)
def in_bounds_scalar(x: int, y: int, z: int) -> bool:
    return 0 <= x < SIZE and 0 <= y < SIZE and 0 <= z < SIZE

def filter_valid_coords(coords: np.ndarray, log_oob: bool = True, clamp: bool = False) -> np.ndarray:
    # 1. optionally coerce edge values
    if clamp:
        coords = np.clip(coords, 0, SIZE - 1)

    # 2. build mask (True ⇒ keep)
    valid_mask = np.all((coords >= 0) & (coords < SIZE), axis=1)

    # 3. logging
    if log_oob and not np.all(valid_mask):
        bad = coords[~valid_mask]
        # print(f"[WARNING] Filtered {len(bad)} OOB coords. Sample: {bad[:3]}")

    # 4. return only valid rows
    return coords[valid_mask]

_COORD_TO_IDX = {
    (x, y, z): x + 9 * y + 81 * z
    for x in range(9) for y in range(9) for z in range(9)
}
_IDX_TO_COORD = {v: k for k, v in _COORD_TO_IDX.items()}

#  NEW – real functions that the rest of the code can call
def coord_to_idx(coord: Coord) -> int:
    """9×9×9 linear index:  x + 9*y + 81*z   (0 … 728)"""
    # we reuse the pre-built dict for speed
    return _COORD_TO_IDX[coord]

def idx_to_coord(idx: int) -> Coord:
    """Inverse of the above."""
    return _IDX_TO_COORD[idx]

def clip_coords(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clamp coordinates to [0, 8] to prevent index errors."""
    return np.clip(x, 0, 8), np.clip(y, 0, 8), np.clip(z, 0, 8)

def extract_directions_and_steps_vectorized(start: Coord, to_coords: np.ndarray) -> Tuple[np.ndarray, int]:
    if to_coords.size == 0:                      # empty request
        return np.empty((0, 3), dtype=np.int8), 0

    start_arr = np.asarray(start, dtype=np.int32)
    deltas = to_coords.astype(np.int32) - start_arr

    # Chebyshev norm along each row
    norms = np.max(np.abs(deltas), axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)       # avoid div-by-zero

    unit_dirs = (deltas // norms).astype(np.int8)
    uniq_dirs = np.unique(unit_dirs, axis=0)

    max_steps = int(norms.max())
    return uniq_dirs, max_steps

def extract_directions_and_steps(to_coords: np.ndarray, start: Coord) -> Tuple[np.ndarray, int]:
    return extract_directions_and_steps_vectorized(start, to_coords)

def rebuild_moves_from_directions(start: Coord, directions: np.ndarray, max_steps: int, capture_set: Set[Coord]) -> List[Move]:
    """Batch-rebuild moves from directions and steps."""
    from game3d.movement.movepiece import Move
    if len(directions) == 0 or max_steps <= 0:
        return []
    sx, sy, sz = start
    rebuilt = []
    for dx, dy, dz in directions:
        for step in range(1, max_steps + 1):
            to = (sx + step * dx, sy + step * dy, sz + step * dz)
            rebuilt.append(Move.create_simple(start, to, to in capture_set))
    return rebuilt


def find_king(state: "GameState", color: Color) -> Optional[Coord]:
    """Vectorised, lock-free search for the king of *color*."""
    for coord, piece in state.cache.piece_cache.iter_color(color):
        if piece.ptype == PieceType.KING:
            return coord
    return None

def infer_piece_from_cache(
    cache_manager: "OptimizedCacheManager",
    coord: Coord,
    fallback_type: PieceType = PieceType.PAWN
) -> Piece:
    """
    Infer piece from cache, fallback to given type.
    FIXED: Handles Piece objects correctly.
    """
    piece = cache_manager.occupancy.get(coord)
    if piece:
        return piece
    # Fallback - need to know the color
    # This is a limitation of the current API
    return Piece(Color.WHITE, fallback_type)

def reconstruct_path(start: Coord, end: Coord, include_start: bool = False, include_end: bool = True, as_set: bool = False) -> Union[List[Coord], Set[Coord]]:
    """Reconstruct path from start to end, optionally as set, including/excluding endpoints."""
    if start == end:
        return set() if as_set else []
    fx, fy, fz = start
    tx, ty, tz = end
    dx = tx - fx
    dy = ty - fy
    dz = tz - fz
    max_delta = max(abs(dx), abs(dy), abs(dz))
    if max_delta == 0:
        return [] if not as_set else set()
    step_x = dx // max_delta if dx != 0 else 0
    step_y = dy // max_delta if dy != 0 else 0
    step_z = dz // max_delta if dz != 0 else 0
    path = set() if as_set else []
    x, y, z = fx, fy, fz
    if include_start:
        if as_set:
            path.add((x, y, z))
        else:
            path.append((x, y, z))
    for _ in range(max_delta):
        x += step_x
        y += step_y
        z += step_z
        if _ < max_delta - 1 or include_end:
            if as_set:
                path.add((x, y, z))
            else:
                path.append((x, y, z))
    return path

# NEW: Batch version for multiple ends
def reconstruct_paths_batch(start: Coord, ends: np.ndarray, include_start: bool = False, include_end: bool = True, as_sets: bool = False) -> List[Union[List[Coord], Set[Coord]]]:
    paths = []
    for end in ends:
        paths.append(reconstruct_path(start, tuple(end), include_start, include_end, as_sets))
    return paths

def get_piece_locations_by_color(board_tensor: torch.Tensor, color: int) -> List[Tuple[int, Coord]]:
    color_plane = board_tensor[COLOR_SLICE.start + (0 if color == 1 else 1)]
    color_positions = color_plane > 0.5 if color == 1 else color_plane <= 0.5
    results = []
    for piece_type in range(N_PIECE_TYPES):
        piece_plane = board_tensor[piece_type]
        target_positions = (piece_plane > 0.5) & color_positions
        positions = torch.nonzero(target_positions, as_tuple=False)
        for z, y, x in positions.tolist():
            results.append((piece_type, (int(x), int(y), int(z))))
    return results

def fallback_mode(default_mode):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log error (replace print with logging)
                print(f"Error in mode: {e}, falling back to {default_mode}")
                import traceback
                traceback.print_exc()
                # Assume args[0] is state, kwargs.get('mode') or args[1] is mode
                mode_arg_idx = 1 if len(args) > 1 else None
                if mode_arg_idx:
                    args = list(args)
                    args[mode_arg_idx] = default_mode
                else:
                    kwargs['mode'] = default_mode
                return func(*args, **kwargs)
            return wrapper
    return decorator

def get_player_pieces(state: GameState, color: Color) -> List[Tuple[Coord, Piece]]:
    result = []
    # BETTER - use piece_cache property
    for coord, piece_data in state.cache.piece_cache.iter_color(color):
        if not isinstance(piece_data, Piece):
            continue
        result.append((coord, piece_data))
    return result

def iterate_occupied(board: "Board", color: Optional[Color] = None) -> Iterable[Tuple[Coord, Piece]]:
    """
    Iterate over occupied squares, optionally filtered by color. Uses cache if available.
    FIXED: Handles case where iter_color returns Piece objects (which it should).
    """
    if board.cache_manager:  # fast path
        if color is None:
            for c in [Color.WHITE, Color.BLACK]:
                for coord, piece_data in board.cache_manager.occupancy.iter_color(c):
                    # piece_data should already be a Piece object from iter_color
                    if not isinstance(piece_data, Piece):
                        # Defensive fallback
                        if isinstance(piece_data, PieceType):
                            piece = Piece(c, piece_data)
                        else:
                            print(f"[ERROR] Unexpected data type in iterate_occupied: {type(piece_data)}")
                            continue
                    else:
                        piece = piece_data
                    yield coord, piece
        else:
            for coord, piece_data in board.cache_manager.occupancy.iter_color(color):
                # piece_data should already be a Piece object
                if not isinstance(piece_data, Piece):
                    # Defensive fallback
                    if isinstance(piece_data, PieceType):
                        piece = Piece(color, piece_data)
                    else:
                        print(f"[ERROR] Unexpected data type in iterate_occupied: {type(piece_data)}")
                        continue
                else:
                    piece = piece_data

                if piece and (color is None or piece.color == color):
                    yield coord, piece
    else:  # slow path, tensor scan
        occ = board._tensor[PIECE_SLICE].sum(dim=0) > 0
        indices = torch.nonzero(occ, as_tuple=False)
        for z, y, x in indices.tolist():
            piece = board.piece_at((x, y, z))
            if piece and (color is None or piece.color == color):
                yield (x, y, z), piece

@njit(cache=True)
def color_to_code(color: "Color") -> int:
    """Return the occupancy-array code (1 or 2) for the given Color enum."""
    return 1 if color.value == 1 else 2

@njit(cache=True)
def add_coords(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Component-wise addition of two 3-D coordinates."""
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]

@njit(cache=True)
def subtract_coords(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Component-wise subtraction of two 3-D coordinates."""
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]

def extend_move_range(move: Move, start: Coord, max_steps: int = 1, debuffed: bool = False) -> List[Move]:
    """Extend move range for buffed/debuffed pieces."""
    direction = tuple((b - a) for a, b in zip(start, move.to_coord))
    norm = max(abs(d) for d in direction) if direction else 0
    if norm == 0:
        return [move]
    unit_dir = tuple(d // norm for d in direction)
    extended_moves = [move]
    for step in range(1, max_steps + 1):
        next_step = tuple(a + step * b for a, b in zip(move.to_coord, unit_dir))
        if all(0 <= c < SIZE for c in next_step):
            extended_moves.append(Move.create_simple(start, next_step, is_capture=move.is_capture, debuffed=debuffed))
        else:
            break
    return extended_moves
# ------------------------------------------------------------------
# Distance helpers – kept as one-liners, already fast
# ------------------------------------------------------------------
manhattan_distance = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])
chebyshev_distance = lambda a, b: max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))
euclidean_distance_squared = lambda a, b: (a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2

# Add the missing euclidean_distance function
def euclidean_distance(a: Coord, b: Coord) -> float:
    """Calculate the Euclidean distance between two coordinates."""
    return math.sqrt(euclidean_distance_squared(a, b))

# Add the missing manhattan function (alias for manhattan_distance)
def manhattan(a: Coord, b: Coord) -> int:
    """Calculate the Manhattan distance between two coordinates."""
    return manhattan_distance(a, b)

# Add the missing get_path_squares function
def get_path_squares(start: Coord, end: Coord) -> List[Coord]:
    """Get all squares on the path from start to end, including both endpoints."""
    if start == end:
        return [start]

    diff = subtract_coords(end, start)
    non_zero = [d for d in diff if d != 0]
    if not non_zero:
        return [start]

    # Check if straight line: orthogonal or diagonal
    abs_vals = [abs(d) for d in non_zero]
    is_orthogonal = len(non_zero) == 1
    is_diagonal = len(set(abs_vals)) == 1

    if not (is_orthogonal or is_diagonal):
        # Not a straight line, just return start and end
        return [start, end]

    step = tuple(0 if d == 0 else (1 if d > 0 else -1) for d in diff)
    distance = max(abs_vals)

    path = [start]
    for i in range(1, distance + 1):
        sq = add_coords(start, scale_coord(step, i))
        if in_bounds(sq):
            path.append(sq)
        else:
            break

    return path

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
        piece = board.cache_manager.occupancy.get(sq)
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

def _clip_coords(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clamp coordinates to [0, 8] to prevent index errors."""
    return np.clip(x, 0, 8), np.clip(y, 0, 8), np.clip(z, 0, 8)

N_CHANNELS = N_TOTAL_PLANES

def validate_directions(start: Coord, directions: np.ndarray, board_size: int = 9) -> np.ndarray:
    """Filter directions that would lead to out-of-bounds coordinates."""
    start_arr = np.array(start, dtype=np.int16)
    dest_coords = start_arr + directions
    valid_mask = np.all((dest_coords >= 0) & (dest_coords < board_size), axis=1)
    return directions[valid_mask]

def validate_moves(
    moves: List["Move"],
    state: "GameState",
    piece: Optional[Piece] = None
) -> List["Move"]:
    """
    Vectorized validation of moves: from consistency, bounds, not friendly dest.
    FIXED: Better error handling for piece validation.
    """
    if not moves:
        return []

    # DEFENSIVE: Filter out None moves immediately
    moves = filter_none_moves(moves)
    if not moves:
        return []

    from_coords = np.array([m.from_coord for m in moves])
    to_coords = np.array([m.to_coord for m in moves])

    # Validate piece parameter
    if piece is not None:
        if not isinstance(piece, Piece):
            print(f"[ERROR] Invalid piece parameter in validate_moves: {type(piece)}")
            return []
        expected_color = piece.color
    else:
        expected_color = state.color

    color_code = 1 if expected_color == Color.WHITE else 2
    cache = state.cache

    # Build occupancy array from iter_color instead of direct access
    occ = np.zeros((9, 9, 9), dtype=np.uint8)
    for color in [Color.WHITE, Color.BLACK]:
        code = 1 if color == Color.WHITE else 2
        for coord, piece_obj in cache.occupancy.iter_color(color):
            if not isinstance(piece_obj, Piece):
                print(f"[WARNING] Non-Piece object in validate_moves iter_color: {type(piece_obj)}")
                continue
            x, y, z = coord
            occ[z, y, x] = code

    # Validate from coordinates match expected color
    valid_from = np.ones(len(from_coords), dtype=bool)
    for i, (x, y, z) in enumerate(from_coords):
        if not (0 <= x < 9 and 0 <= y < 9 and 0 <= z < 9):
            valid_from[i] = False
            continue
        if occ[z, y, x] != color_code:
            valid_from[i] = False

    # Filter valid to_coords
    to_coords = filter_valid_coords(to_coords, log_oob=True)
    if len(to_coords) == 0:
        return []

    # Check destinations are not friendly pieces
    to_x, to_y, to_z = to_coords[:, 0], to_coords[:, 1], to_coords[:, 2]
    dest_codes = occ[to_z, to_y, to_x]
    valid_dest = dest_codes != color_code

    # Combine masks
    min_len = min(len(valid_from), len(valid_dest))
    valid_mask = valid_from[:min_len] & valid_dest[:min_len]

    validated = [moves[i] for i in np.flatnonzero(valid_mask)]

    # DEFENSIVE: Final None filter before returning
    return filter_none_moves(validated)

def prepare_batch_data(state: "GameState") -> Tuple[List[Coord], List[PieceType], List[int]]:
    """
    Prepare coords, types, debuffed indices for batch dispatch.
    FIXED: Properly handles Piece objects.
    """
    coords, types, debuffed = [], [], []
    for idx, (coord, piece) in enumerate(get_player_pieces(state, state.color)):
        if not isinstance(piece, Piece):
            print(f"[ERROR] Non-Piece in prepare_batch_data: {type(piece)}")
            continue

        coords.append(coord)
        types.append(piece.ptype)

        if coord in state.cache.move._debuffed_set and piece.ptype != PieceType.PAWN:
            debuffed.append(idx)

    return coords, types, debuffed
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

@contextmanager
def measure_time_ms():
    start = time.perf_counter()
    yield lambda: (time.perf_counter() - start) * 1000

@dataclass(slots=True)
class StatsTracker:
    total_calls: int = 0
    average_time_ms: float = 0.0
    # Add common fields like total_moves_generated, etc., as needed

    def update_average(self, elapsed_ms: float) -> None:
        self.average_time_ms = (
            (self.average_time_ms * (self.total_calls - 1) + elapsed_ms) / self.total_calls
            if self.total_calls > 0 else elapsed_ms
        )

    def get_stats(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def reset(self) -> None:
        self.total_calls = 0
        self.average_time_ms = 0.0

@dataclass(slots=True)
class MoveStatsTracker:
    total_calls: int = 0
    average_time_ms: float = 0.0
    total_moves_generated: int = 0
    total_moves_filtered: int = 0
    freeze_filtered: int = 0
    check_filtered: int = 0
    piece_breakdown: Dict[PieceType, int] = field(default_factory=lambda: {pt: 0 for pt in PieceType})

    def update_average(self, elapsed_ms: float) -> None:
        self.average_time_ms = (
            (self.average_time_ms * (self.total_calls - 1) + elapsed_ms) / self.total_calls
            if self.total_calls > 0 else elapsed_ms
        )

    def get_stats(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def reset(self) -> None:
        self.total_calls = 0
        self.average_time_ms = 0.0
        self.total_moves_generated = 0
        self.total_moves_filtered = 0
        self.freeze_filtered = 0
        self.check_filtered = 0
        self.piece_breakdown = {pt: 0 for pt in PieceType}

# NEW: Track time decorator/context
def track_time(tracker: MoveStatsTracker):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with measure_time_ms() as elapsed:
                result = func(*args, **kwargs)
            tracker.update_average(elapsed())
            return result
        return wrapper
    return decorator

# NEW: UndoSnapshot dataclass
@dataclass(slots=True)
class UndoSnapshot:
    original_board_tensor: torch.Tensor
    original_halfmove_clock: int
    original_turn_number: int
    original_zkey: int
    moving_piece: Piece
    captured_piece: Optional[Piece] = None

# NEW: Abstract GeneratorBase
class GeneratorBase:
    def __init__(self, mode_enum, default_mode, stats_tracker: MoveStatsTracker):
        self.mode_enum = mode_enum
        self.default_mode = default_mode
        self.stats = stats_tracker

    def generate(self, state: GameState) -> List[Move]:
        return self._impl(state, mode=self.default_mode.value)

    def _impl(self, state: GameState, mode: str) -> List[Move]:
        raise NotImplementedError

# NEW: For logging OOB
def log_oob(coords: np.ndarray):
    bad = coords[~in_bounds_vectorised(coords)]
    if len(bad) > 0:
        print(f"[OOB] {len(bad)} invalid coords: {bad[:3]}")

# Precomputed offsets for radius=2 aura (cube minus center)
RADIUS_2_OFFSETS: List[Tuple[int, int, int]] = [
    (dx, dy, dz) for dx in range(-2, 3) for dy in range(-2, 3) for dz in range(-2, 3)
    if not (dx == dy == dz == 0)
]  # Length: 124 (5^3 - 1)

RADIUS_3_OFFSETS: List[Tuple[int, int, int]] = [
    (dx, dy, dz)
    for dx in range(-3, 4)
    for dy in range(-3, 4)
    for dz in range(-3, 4)
    if max(abs(dx), abs(dy), abs(dz)) == 3
]

def get_aura_squares(center: Tuple[int, int, int], radius: int = 2) -> Set[Tuple[int, int, int]]:
    """Compute aura squares around center using precomputed offsets, with bounds check."""
    if radius != 2:
        raise ValueError("Only radius=2 supported for precompute")
    cx, cy, cz = center
    aura = set()
    for dx, dy, dz in RADIUS_2_OFFSETS:
        sq = (cx + dx, cy + dy, cz + dz)
        if in_bounds(sq):  # Use existing njit in_bounds
            aura.add(sq)
    return aura

def get_pieces_by_type(
    board: "Board",
    ptype: PieceType,
    color: Optional[Color] = None
) -> List[Tuple[Coord, Piece]]:
    """
    Return every (coord, piece) on *board* whose type == *ptype*
    (and optionally colour == *color*).
    Uses the occupancy cache if already available, otherwise falls back
    to a direct tensor scan.
    FIXED: Properly handles Piece objects from iter_color.
    """
    # Fast path – cache is ready
    if board.cache_manager is not None:
        result = []
        for sq, piece_data in board.cache_manager.occupancy.iter_color(color):
            # piece_data is a Piece object from iter_color
            if not isinstance(piece_data, Piece):
                # Defensive fallback - should not happen
                if isinstance(piece_data, PieceType):
                    if piece_data == ptype:
                        result.append((sq, Piece(color, ptype)))
                continue

            # Normal case: piece_data is a Piece
            if piece_data.ptype == ptype:
                result.append((sq, piece_data))
        return result

    # Slow path – cache not yet attached (e.g. during initial aura rebuild)
    tensor = board._tensor
    piece_planes = tensor[PIECE_SLICE]
    col_plane = tensor[N_PIECE_TYPES]

    wanted_type = ptype.value
    wanted_color = 1 if color is Color.WHITE else 0

    mask = (piece_planes[wanted_type] > 0.5)
    if color is not None:
        mask &= (col_plane > 0.5) if color is Color.WHITE else (col_plane <= 0.5)

    indices = torch.nonzero(mask, as_tuple=False)
    return [
        ((int(x), int(y), int(z)),
         Piece(color if color is not None else Color.WHITE, ptype))
        for z, y, x in indices.tolist()
    ]

def filter_none_moves(moves: List["Move"]) -> List["Move"]:
    """
    Defensive filter to remove None values from move lists.

    This is a safety measure to prevent None values from propagating through
    the move generation pipeline. If None moves are found, logs a warning.

    Args:
        moves: List of moves that may contain None values

    Returns:
        List of moves with all None values removed
    """
    if not moves:
        return []

    # Count None values for debugging
    none_count = sum(1 for m in moves if m is None)

    if none_count > 0:
        print(f"[WARNING] filter_none_moves: Filtered {none_count} None values from {len(moves)} moves")
        import traceback
        traceback.print_stack(limit=5)  # Show where the None came from

    # Filter out None values
    filtered = [m for m in moves if m is not None]

    # Extra validation: check that remaining moves have required attributes
    valid = []
    for m in filtered:
        if not hasattr(m, 'from_coord') or not hasattr(m, 'to_coord'):
            print(f"[WARNING] filter_none_moves: Move missing required attributes: {m}")
            continue
        valid.append(m)

    if len(valid) < len(filtered):
        print(f"[WARNING] filter_none_moves: Removed {len(filtered) - len(valid)} invalid move objects")

    return valid
