"""
game3d/movement/pathvalidation.py — Centralized movement path and blocking logic for 3D chess.
"""
import numpy as np
from typing import List, Tuple, Optional
from game3d.movement.movepiece import Move
from game3d.common.common import in_bounds, add_coords
from game3d.pieces.enums import PieceType

from game3d.pieces.piece import Piece

BOARD_SIZE = 9

def is_edge_square(x: int, y: int, z: int, board_size: int = BOARD_SIZE) -> bool:
    edge = board_size - 1
    return x in (0, edge) or y in (0, edge) or z in (0, edge)

def should_stop_at(
    board,
    color,
    target: Tuple[int, int, int],
    allow_capture: bool = True,
    allow_self_block: bool = False,
) -> Tuple[bool, bool]:
    """
    Determine if movement should stop at target square.
    """
    piece = cache.piece_cache.get(target)
    has_piece = piece is not None
    is_friendly = has_piece and (piece.color == color)
    is_enemy = has_piece and (piece.color != color)

    stop_friendly = is_friendly and (not allow_self_block)
    stop_enemy = is_enemy and allow_capture

    stop = has_piece and (stop_friendly or stop_enemy or (is_enemy and not allow_capture))
    can_land = (not has_piece) or (is_enemy and allow_capture) or (is_friendly and allow_self_block)

    return stop, can_land

def slide_along_directions(
    board,
    color,
    start: Tuple[int, int, int],
    directions: np.ndarray,
    allow_capture: bool = True,
    allow_self_block: bool = False,
    max_steps: Optional[int] = BOARD_SIZE,
) -> List[Move]:
    moves = []
    start_array = np.array(start)

    if max_steps is None:
        max_steps = BOARD_SIZE

    steps = np.arange(1, max_steps + 1)
    direction_steps = directions[:, np.newaxis, :] * steps[np.newaxis, :, np.newaxis]
    all_targets = start_array + direction_steps
    valid_targets = np.all((all_targets >= 0) & (all_targets < BOARD_SIZE), axis=2)

    for i, direction in enumerate(directions):
        for j in range(max_steps):
            if not valid_targets[i, j]:
                break

            target = tuple(all_targets[i, j])
            stop, can_land = should_stop_at(board, color, target, allow_capture, allow_self_block)

            if can_land:
                target_piece = cache.piece_cache.get(target)
                is_capture = target_piece is not None and target_piece.color != color
                moves.append(Move(from_coord=start, to_coord=target, is_capture=is_capture))

            if stop:
                break

    return moves

def jump_to_targets(
    board,
    color,
    start: Tuple[int, int, int],
    offsets: List[Tuple[int, int, int]],
    allow_capture: bool = True,
    allow_self_block: bool = False,
) -> List[Move]:
    moves = []
    for offset in offsets:
        target = add_coords(start, offset)
        if not in_bounds(target):
            continue

        _, can_land = should_stop_at(board, color, target, allow_capture, allow_self_block)
        if can_land:
            target_piece = cache.piece_cache.get(target)
            is_capture = target_piece is not None and target_piece.color != color
            moves.append(Move(from_coord=start, to_coord=target, is_capture=is_capture))

    return moves

def validate_piece_at(
    board,
    color,
    pos: Tuple[int, int, int],
    expected_type: Optional[PieceType] = None,
) -> bool:
    piece = cache.piece_cache.get(pos)
    if piece is None:
        return False
    if piece.color != color:
        return False
    if expected_type is not None and piece.ptype != expected_type:
        return False
    return True
