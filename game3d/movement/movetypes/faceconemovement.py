"""3D Face Cone Slider — projects sliding rays in conical volumes outward from each face."""

from typing import List, Set, Tuple
from math import gcd
from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move
from game3d.movement.pathvalidation import slide_along_directions, validate_piece_at

# Precompute all unique primitive directions for 6 cones
def _precompute_cone_directions() -> List[Tuple[int, int, int]]:
    directions: Set[Tuple[int, int, int]] = set()
    MAX_D = 8  # max displacement on 9x9x9 board

    # Define cone conditions without unused axis_idx
    cones = [
        lambda dx, dy, dz: dx > 0 and abs(dy) <= dx and abs(dz) <= dx,  # +X
        lambda dx, dy, dz: dx < 0 and abs(dy) <= -dx and abs(dz) <= -dx, # -X
        lambda dx, dy, dz: dy > 0 and abs(dx) <= dy and abs(dz) <= dy,  # +Y
        lambda dx, dy, dz: dy < 0 and abs(dx) <= -dy and abs(dz) <= -dy, # -Y
        lambda dx, dy, dz: dz > 0 and abs(dx) <= dz and abs(dy) <= dz,  # +Z
        lambda dx, dy, dz: dz < 0 and abs(dx) <= -dz and abs(dy) <= -dz, # -Z
    ]

    for cone in cones:
        for dx in range(-MAX_D, MAX_D + 1):
            for dy in range(-MAX_D, MAX_D + 1):
                for dz in range(-MAX_D, MAX_D + 1):
                    if dx == dy == dz == 0:
                        continue
                    if not cone(dx, dy, dz):
                        continue
                    # Reduce to primitive direction
                    g = gcd(gcd(abs(dx), abs(dy)), abs(dz))
                    if g > 0:
                        prim = (dx // g, dy // g, dz // g)
                        directions.add(prim)

    return list(directions)

# Compute once at import time
CONE_DIRECTIONS = _precompute_cone_directions()

def generate_face_cone_slider_moves(
    board,
    color: Color,
    x: int, y: int, z: int
) -> List['Move']:
    """
    Generate slider moves in 6 conical volumes, each projecting outward perpendicular to a face.
    Example: from +Z face: rays where dz > 0 and |dx| <= dz, |dy| <= dz.
    """
    start = (x, y, z)
    if not validate_piece_at(board, color, start, PieceType.CONESLIDER):
        return []

    return slide_along_directions(
        board,
        color,
        start=start,
        directions=CONE_DIRECTIONS,
        allow_capture=True,
        allow_self_block=False
    )
