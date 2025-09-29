"""Panel piece — jumps to any square on 6 orthogonal 3x3 walls centered 2 squares away along each axis.
Pure movement logic — no registration, no dispatcher.
"""

from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.movement.pathvalidation import jump_to_targets, validate_piece_at
from game3d.movement.movepiece import Move
from game3d.cache.manager import OptimizedCacheManager

# Precomputed: 54 offsets for six 3×3 walls, 2 units away along each axis
_PANEL_OFFSETS = []

# Directions and their perpendicular wall planes
_DIRECTIONS_AND_PLANES = [
    ((1, 0, 0), 'yz'),   # +X → YZ wall
    ((-1, 0, 0), 'yz'),  # -X → YZ wall
    ((0, 1, 0), 'xz'),   # +Y → XZ wall
    ((0, -1, 0), 'xz'),  # -Y → XZ wall
    ((0, 0, 1), 'xy'),   # +Z → XY wall
    ((0, 0, -1), 'xy'),  # -Z → XY wall
]

# Wall offsets per plane (3×3 centered at origin)
_WALL_OFFSETS = {
    'yz': [(0, dy, dz) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
    'xz': [(dx, 0, dz) for dx in (-1, 0, 1) for dz in (-1, 0, 1)],
    'xy': [(dx, dy, 0) for dx in (-1, 0, 1) for dy in (-1, 0, 1)],
}

# Generate all 6 × 9 = 54 offsets
for direction, plane_key in _DIRECTIONS_AND_PLANES:
    dx, dy, dz = direction
    anchor = (2*dx, 2*dy, 2*dz)
    for wx, wy, wz in _WALL_OFFSETS[plane_key]:
        _PANEL_OFFSETS.append((
            anchor[0] + wx,
            anchor[1] + wy,
            anchor[2] + wz
        ))

def generate_panel_moves(
    cache: OptimizedCacheManager,  # ← CHANGED: board → cache
    color: Color,
    x: int,
    y: int,
    z: int
) -> List['Move']:
    """Generate all legal Panel moves from (x, y, z)."""
    pos = (x, y, z)

    # Validate piece at starting position
    if not validate_piece_at(cache, color, pos, PieceType.PANEL):  # ← cache, not board
        return []

    # Use jump_to_targets with correct arguments
    return jump_to_targets(
        cache=cache,  # ← FIXED: cache, not board
        color=color,
        start=pos,
        offsets=_PANEL_OFFSETS,
        allow_capture=True,
        allow_self_block=False
    )

def get_panel_offsets():
    return _PANEL_OFFSETS.copy()
