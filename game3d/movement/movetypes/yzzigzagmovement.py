"""YZ-Zig-Zag Slider — zig-zag rays along X-, Y-, Z-axis normals (no king)."""

from typing import List, Tuple
from pieces.enums import PieceType
from game.state import GameState
from game.move import Move
from game3d.movement.pathvalidation import validate_piece_at
from common import in_bounds


SEGMENT = 3          # steps before direction flip
DIRECTIONS = [(1, -1), (-1, 1)]   # (primary, flip) pairs


def _zigzag_ray(
    state: GameState,
    start: Tuple[int, int, int],
    plane: str,                # 'YZ' | 'XZ' | 'XY'
    primary: int,              # +1 or –1 for first 3-step leg
    secondary: int,            # +1 or –1 for second 3-step leg
    fixed: int                 # coordinate that stays constant
) -> List[Move]:
    """Cast one zig-zag ray in the chosen plane until blocked or edge."""
    moves: List[Move] = []
    x, y, z = start
    board = state.board
    current_color = state.current

    # axis mapping
    if plane == 'YZ':           # X fixed
        idx_a, idx_b = 1, 2
        get_coord = lambda a, b: (fixed, a, b)
    elif plane == 'XZ':         # Y fixed
        idx_a, idx_b = 0, 2
        get_coord = lambda a, b: (a, fixed, b)
    else:                       # 'XY'  Z fixed
        idx_a, idx_b = 0, 1
        get_coord = lambda a, b: (a, b, fixed)

    coords = [x, y, z]
    a, b = coords[idx_a], coords[idx_b]

    flip = False
    while True:
        step = secondary if flip else primary
        for _ in range(SEGMENT):
            a += step
            target = get_coord(a, b)
            if not in_bounds(target):
                return moves
            occupant = board.piece_at(target)
            if occupant is not None:
                if occupant.color != current_color:   # capture
                    moves.append(Move(start, target, is_capture=True))
                return moves
            moves.append(Move(start, target, is_capture=False))
        flip = not flip


def generate_yz_zigzag_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """Generate all YZ-zig-zag moves (no king)."""
    start = (x, y, z)
    if not validate_piece_at(state, start, PieceType.YZ_ZIGZAG_SLIDER):
        return []

    moves: List[Move] = []

    # X-normal faces  →  X fixed  →  YZ plane
    for pri, sec in DIRECTIONS:
        moves.extend(_zigzag_ray(state, start, 'YZ', pri, sec, x))

    # Y-normal faces  →  Y fixed  →  XZ plane
    for pri, sec in DIRECTIONS:
        moves.extend(_zigzag_ray(state, start, 'XZ', pri, sec, y))

    # Z-normal faces  →  Z fixed  →  XY plane
    for pri, sec in DIRECTIONS:
        moves.extend(_zigzag_ray(state, start, 'XY', pri, sec, z))

    return moves
