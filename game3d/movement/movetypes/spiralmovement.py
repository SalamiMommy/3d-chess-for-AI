"""Counter-clockwise 3-D spiral slider, radius 2 per axis."""

from typing import List, Tuple
from pieces.enums import PieceType
from game.state import GameState
from game.move import Move
from game3d.movement.pathvalidation import validate_piece_at
from common import in_bounds


def _spiral_plane(
    state: GameState,
    start: Tuple[int, int, int],
    plane: str,          # 'XY' | 'XZ' | 'YZ'
    fixed: int,
    radius: int,
    current_color: int
) -> List[Move]:
    """Yield one CCW square-ring of given radius in the chosen plane."""
    moves: List[Move] = []
    x, y, z = start
    board = state.board

    # map plane to axis indices
    if plane == 'XY':
        idx_a, idx_b = 0, 1
        get_coord = lambda a, b: (a, b, fixed)
    elif plane == 'XZ':
        idx_a, idx_b = 0, 2
        get_coord = lambda a, b: (a, fixed, b)
    else:  # YZ
        idx_a, idx_b = 1, 2
        get_coord = lambda a, b: (fixed, a, b)

    coords = [x, y, z]
    cen_a, cen_b = coords[idx_a], coords[idx_b]

    # build CCW square ring: bottom→right→top→left
    ring = []
    # bottom edge (left → right)
    for da in range(-radius, radius + 1):
        ring.append((cen_a + da, cen_b - radius))
    # right edge (bottom+1 → top-1)
    for db in range(-radius + 1, radius):
        ring.append((cen_a + radius, cen_b + db))
    # top edge (right → left)
    for da in range(radius, -radius - 1, -1):
        ring.append((cen_a + da, cen_b + radius))
    # left edge (top-1 → bottom+1)
    for db in range(radius - 1, -radius, -1):
        ring.append((cen_a - radius, cen_b + db))

    # walk the ring, stop on first block
    for a, b in ring:
        target = get_coord(a, b)
        if not in_bounds(target) or target == start:
            continue
        occupant = board.piece_at(target)
        if occupant is not None:
            if occupant.color != current_color:
                moves.append(Move(start, target, is_capture=True))
            return moves  # blocked
        moves.append(Move(start, target, is_capture=False))
    return moves


def generate_spiral_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """Generate all CCW spiral moves (radius 2, all three planes)."""
    start = (x, y, z)
    if not validate_piece_at(state, start, expected_type=PieceType.SPIRAL):
        return []

    moves: List[Move] = []
    current_color = state.current

    for r in range(2, 0, -1):  # radius 2 → 1
        moves.extend(_spiral_plane(state, start, 'XY', z, r, current_color))
        moves.extend(_spiral_plane(state, start, 'XZ', y, r, current_color))
        moves.extend(_spiral_plane(state, start, 'YZ', x, r, current_color))

    return moves
