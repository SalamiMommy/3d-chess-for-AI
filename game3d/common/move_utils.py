# game3d/common/move_utils.py
# ------------------------------------------------------------------
# Move-related utilities
# ------------------------------------------------------------------
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Set, Optional, TYPE_CHECKING

from game3d.common.constants import SIZE, N_PIECE_TYPES
from game3d.common.coord_utils import in_bounds_vectorised, filter_valid_coords, add_coords
from game3d.common.piece_utils import get_player_pieces
from game3d.common.enums import Color, PieceType
from game3d.pieces.piece import Piece
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

Coord = Tuple[int, int, int]

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
