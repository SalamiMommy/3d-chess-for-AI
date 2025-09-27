# game3d/movement/movetypes/nebulamovement.py
"""Nebula piece — jumps to any square within or on sphere of radius 3 (Euclidean distance <= 3).
Pure movement logic — no registration, no dispatcher.
"""

from __future__ import annotations
from typing import List
from game3d.pieces.enums import PieceType
from game3d.pieces.enums import PieceType, Color
from game3d.movement.pathvalidation import jump_to_targets, validate_piece_at

# Precomputed offsets: all (dx,dy,dz) where dx²+dy²+dz² <= 9, excluding (0,0,0)
_NEBULA_OFFSETS = [
    (dx, dy, dz)
    for dx in range(-3, 4)
    for dy in range(-3, 4)
    for dz in range(-3, 4)
    if dx*dx + dy*dy + dz*dz <= 9 and not (dx == dy == dz == 0)
]
# Total: 122 offsets

def generate_nebula_moves(board, color: Color, x: int, y: int, z: int) -> List['Move']:
    """Generate all legal Nebula moves from (x, y, z)."""
    pos = (x, y, z)
    if not validate_piece_at(state.board, state.color, pos, PieceType.NEBULA):
        return []

    return jump_to_targets(
        state,
        start=pos,
        offsets=_NEBULA_OFFSETS,
        allow_capture=True,
        allow_self_block=False
    )

# Optional helpers
def get_nebula_offsets():
    return _NEBULA_OFFSETS.copy()

def count_valid_nebula_moves_from(state: GameState, x: int, y: int, z: int) -> int:
    return len(generate_nebula_moves(state, x, y, z))

def get_nebula_reach_volume() -> int:
    return len(_NEBULA_OFFSETS)  # 122
