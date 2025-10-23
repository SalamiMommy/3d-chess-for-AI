# game3d/common/move_utils.py - OPTIMIZED VERSION
# ------------------------------------------------------------------
# Move-related utilities with consolidated validation
# ------------------------------------------------------------------
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Set, Optional, TYPE_CHECKING

from game3d.common.constants import SIZE, VOLUME
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

def validate_moves_batch(
    moves: List["Move"],
    state: "GameState",
    piece: Optional[Piece] = None
) -> List["Move"]:
    """
    Consolidated single-pass validation of moves.
    Combines bounds checking, occupancy validation, and effect filtering.
    """
    if not moves:
        return []

    # DEFENSIVE: Filter out None moves immediately
    moves = filter_none_moves(moves)
    if not moves:
        return []

    # Extract coordinates for vectorized processing
    from_coords = np.array([m.from_coord for m in moves])
    to_coords = np.array([m.to_coord for m in moves])

    # Single-pass bounds validation
    from_valid = in_bounds_vectorised(from_coords)
    to_valid = in_bounds_vectorised(to_coords)
    bounds_valid = from_valid & to_valid

    if not np.any(bounds_valid):
        return []

    # Get piece info for validation
    if piece is not None:
        expected_color = piece.color
    else:
        expected_color = state.color

    # Single-pass occupancy and effect validation
    valid_moves = []
    cache_manager = state.cache_manager

    for i, move in enumerate(moves):
        if not bounds_valid[i]:
            continue

        # Check piece exists and is correct color
        from_piece = cache_manager.occupancy.get(move.from_coord)
        if not from_piece or from_piece.color != expected_color:
            continue

        # Check frozen status
        if cache_manager.is_frozen(move.from_coord, expected_color):
            continue

        # Check destination is not friendly
        to_piece = cache_manager.occupancy.get(move.to_coord)
        if to_piece and to_piece.color == expected_color:
            continue

        valid_moves.append(move)

    return valid_moves

# Alias for backward compatibility
validate_moves = validate_moves_batch

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

        if state.cache_manager.is_movement_debuffed(coord, state.color) and piece.ptype != PieceType.PAWN:
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
