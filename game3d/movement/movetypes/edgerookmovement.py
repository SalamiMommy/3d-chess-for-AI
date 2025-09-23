# game3d/movement/movetypes/edgerookmovement.py

"""3D Edge-Rook move generation logic — moves along board edges, can turn freely."""

from typing import List, Set, Tuple
from pieces.enums import PieceType
from game3d.game.gamestate import GameState
from game3d.movement.movepiece import Move
from game3d.movement.pathvalidation import (
    is_edge_square,
    is_path_blocked,
    validate_piece_at
)
from common import add_coords, in_bounds

def generate_edgerook_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    start = (x, y, z)

    if not validate_piece_at(state, start, PieceType.EDGEROOK):
        return []

    if not is_edge_square(x, y, z, board_size=9):
        return []

    moves: List[Move] = []
    visited: Set[Tuple[int, int, int]] = set()
    queue: List[Tuple[Tuple[int, int, int], List[Tuple[int, int, int]]]] = [(start, [start])]
    directions = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    while queue:
        current, path = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        for dx, dy, dz in directions:
            step = 1
            while True:
                offset = (dx * step, dy * step, dz * step)
                target = add_coords(current, offset)
                if not in_bounds(target):
                    break
                if not is_edge_square(*target, board_size=9):
                    break
                target_piece = state.board.piece_at(target)
                if target_piece and target_piece.color == state.current:
                    break
                if target_piece is None or (target_piece and target_piece.color != state.current):
                    if target in path:
                        step += 1
                        continue
                    new_path = path + [target]
                    is_capture = target_piece is not None and target_piece.color != state.current
                    move = Move(
                        from_coord=start,
                        to_coord=target,
                        is_capture=is_capture
                    )
                    if target not in visited:
                        moves.append(move)
                        queue.append((target, new_path))
                if target_piece and target_piece.color != state.current:
                    break
                step += 1
    return moves
