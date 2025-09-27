# game3d/movement/movetypes/orbitalmovement.py
"""Orbital piece — jumps to any square with Manhattan distance 4.
Pure movement logic — no registration, no dispatcher.
"""

from __future__ import annotations
from typing import List
from game3d.pieces.enums import PieceType
from game3d.pieces.enums import PieceType, Color
from game3d.movement.pathvalidation import jump_to_targets, validate_piece_at

# Precomputed offsets: all (dx,dy,dz) where |dx|+|dy|+|dz| == 4
_ORBITAL_OFFSETS = [
    (dx, dy, dz)
    for dx in range(-4, 5)
    for dy in range(-4, 5)
    for dz in range(-4, 5)
    if abs(dx) + abs(dy) + abs(dz) == 4
]
# Total: 66 offsets (3D octahedron surface)

def generate_orbital_moves(board, color: Color, x: int, y: int, z: int) -> List['Move']:
    """Generate all legal Orbiter moves from (x, y, z)."""
    pos = (x, y, z)
    if not validate_piece_at(state.board, state.color, pos, PieceType.ORBITER):
        return []

    return jump_to_targets(
        state,
        start=pos,
        offsets=_ORBITAL_OFFSETS,
        allow_capture=True,
        allow_self_block=False
    )

# Optional helpers
def get_orbital_offsets():
    return _ORBITAL_OFFSETS.copy()

def count_valid_orbital_moves_from(state: GameState, x: int, y: int, z: int) -> int:
    return len(generate_orbital_moves(state, x, y, z))
