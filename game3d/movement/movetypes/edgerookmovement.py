# game3d/movement/movetypes/edgerookmovement.py

"""3D Edge-Rook move generation logic — moves along board edges, can turn freely.
Pure movement logic — no registration, no dispatcher.
"""

from typing import List, Set, Tuple
from pieces.enums import PieceType
from game.state import GameState
from game.move import Move

# Import centralized pathing logic
from game3d.movement.pathvalidation import (
    is_edge_square,
    is_path_blocked,
    validate_piece_at
)
from common import add_coords, in_bounds


def generate_edge_rook_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """
    Generate all legal EDGE ROOK moves from (x, y, z).
    Moves only along edge squares (where at least one coord is 0 or 8).
    Can turn any number of times — as long as every square visited is on an edge.

    Uses BFS to explore all reachable edge squares with turning allowed.
    """
    start = (x, y, z)

    # Validate piece exists and is correct type/color
    if not validate_piece_at(state, start, PieceType.EDGE_ROOK):
        return []

    # Must start on edge
    if not is_edge_square(x, y, z, board_size=9):
        return []

    # Use BFS to explore all reachable edge squares with turning allowed
    moves: List[Move] = []
    visited: Set[Tuple[int, int, int]] = set()
    # Queue stores (current_position, path_taken) — path used to avoid cycles
    queue: List[Tuple[Tuple[int, int, int], List[Tuple[int, int, int]]]] = [(start, [start])]

    # Orthogonal directions
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

        # From current edge square, try sliding in all 6 directions — but only on edge
        for dx, dy, dz in directions:
            step = 1
            while True:
                offset = (dx * step, dy * step, dz * step)
                target = add_coords(current, offset)

                # Stop if out of bounds
                if not in_bounds(target):
                    break

                # Must stay on edge
                if not is_edge_square(*target, board_size=9):
                    break

                # Check if blocked at target — friendly piece blocks, enemy can be captured
                target_piece = state.board.piece_at(target)

                # If blocked by friendly → stop this ray
                if target_piece and target_piece.color == state.current:
                    break

                # If we can land here (empty or capturable enemy), consider it
                if target_piece is None or (target_piece and target_piece.color != state.current):
                    # Avoid cycles — don't revisit squares in this path
                    if target in path:
                        step += 1
                        continue

                    # Create new path
                    new_path = path + [target]

                    # Generate move from start to target (not full path — just endpoint)
                    is_capture = target_piece is not None and target_piece.color != state.current
                    move = Move(
                        from_coord=start,
                        to_coord=target,
                        is_capture=is_capture,
                        piece_type=PieceType.EDGE_ROOK  # optional, if Move requires it
                    )

                    # Avoid duplicate move targets (different paths to same square)
                    if target not in visited:
                        moves.append(move)
                        queue.append((target, new_path))

                # Even if we captured, we stop this ray — Edge-Rook doesn't jump or continue after capture
                if target_piece and target_piece.color != state.current:
                    break

                step += 1

    return moves
