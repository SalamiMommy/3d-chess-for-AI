"""Reflecting Bishop — diagonal rays that bounce off cube walls and pieces (≤3 bounces)."""

from typing import List, Set, Tuple
from game3d.pieces.enums import PieceType
from game3d.pieces.enums import PieceType, Color
from game3d.movement.pathvalidation import validate_piece_at

MAX_BOUNCES = 3
BOARD_SIZE = 9

def reflect_direction(direction: Tuple[int, int, int], axes_to_flip: Tuple[bool, bool, bool]) -> Tuple[int, int, int]:
    """Reflect direction on specified axes."""
    dx, dy, dz = direction
    if axes_to_flip[0]:
        dx = -dx
    if axes_to_flip[1]:
        dy = -dy
    if axes_to_flip[2]:
        dz = -dz
    return (dx, dy, dz)

def generate_reflecting_bishop_moves(board, color: Color, x: int, y: int, z: int) -> List['Move']:
    """Generate all legal reflecting-bishop moves (wall-bouncing diagonal rays)."""
    start_pos = (x, y, z)

    if not validate_piece_at(state, start_pos, pos, expected_type=PieceType.REFLECTOR):
        return []

    # 8 space diagonal directions
    initial_directions = [
        (dx, dy, dz)
        for dx in (-1, 1)
        for dy in (-1, 1)
        for dz in (-1, 1)
    ]

    visited_targets: Set[Tuple[int, int, int]] = set()
    moves: List['Move'] = []
    current_color = state.color

    for init_dir in initial_directions:
        pos = start_pos
        direction = init_dir
        bounces = 0

        while bounces <= MAX_BOUNCES:
            # Calculate next position
            next_pos = (pos[0] + direction[0], pos[1] + direction[1], pos[2] + direction[2])
            nx, ny, nz = next_pos

            # Check if out of bounds on any axis
            out_of_bounds = [False, False, False]
            if nx < 0 or nx >= BOARD_SIZE:
                out_of_bounds[0] = True
            if ny < 0 or ny >= BOARD_SIZE:
                out_of_bounds[1] = True
            if nz < 0 or nz >= BOARD_SIZE:
                out_of_bounds[2] = True

            if any(out_of_bounds):
                # Reflect on all out-of-bounds axes
                if bounces >= MAX_BOUNCES:
                    break  # Can't bounce anymore
                direction = reflect_direction(direction, tuple(out_of_bounds))
                bounces += 1
                continue  # Try again with new direction

            # In bounds - check occupancy
            target_piece = state.board.piece_at(next_pos)

            # Cannot land on friendly pieces
            if target_piece is not None and target_piece.color == current_color:
                break

            # Add unique target
            if next_pos not in visited_targets:
                visited_targets.add(next_pos)
                is_capture = target_piece is not None
                moves.append(Move(
                    from_coord=start_pos,
                    to_coord=next_pos,
                    is_capture=is_capture
                ))

            # Stop ray at any piece (friendly already handled above)
            if target_piece is not None:
                break

            # Move to next position
            pos = next_pos

    return moves
