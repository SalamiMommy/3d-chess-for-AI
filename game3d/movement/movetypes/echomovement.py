# game3d/movement/movetypes/echomovement.py
"""Echo piece — jumps to any square within radius 2 of any point offset by ±3 in each axis.
Pure movement logic — no registration, no dispatcher.
"""

from __future__ import annotations
from typing import List, Set
from game3d.pieces.enums import PieceType
from game3d.pieces.enums import PieceType, Color
from game3d.movement.pathvalidation import validate_piece_at
from game3d.common.common import in_bounds, add_coords, Coord

# Precomputed: 32 offsets within Euclidean distance <= 2 (excluding origin)
_BUBBLE_OFFSETS = [
    (dx, dy, dz)
    for dx in range(-2, 3)
    for dy in range(-2, 3)
    for dz in range(-2, 3)
    if dx*dx + dy*dy + dz*dz <= 4 and not (dx == dy == dz == 0)
]

# Precomputed: 8 anchor offsets at (±3, ±3, ±3)
_ANCHOR_OFFSETS = [
    (dx, dy, dz)
    for dx in (-3, 3)
    for dy in (-3, 3)
    for dz in (-3, 3)
]

def generate_echo_moves(board, color: Color, x: int, y: int, z: int) -> List['Move']:
    """Generate all legal Echo moves from (x, y, z)."""
    pos = (x, y, z)
    if not validate_piece_at(state.board, state.color, pos, PieceType.ECHO):
        return []

    current_color = state.color
    move_targets: Set[Coord] = set()

    # Generate all potential targets
    for ax, ay, az in _ANCHOR_OFFSETS:
        anchor = (x + ax, y + ay, z + az)
        for bx, by, bz in _BUBBLE_OFFSETS:
            target = (anchor[0] + bx, anchor[1] + by, anchor[2] + bz)
            if in_bounds(target):
                move_targets.add(target)

    # Build moves
    moves = []
    for target in move_targets:
        target_piece = state.board.piece_at(target)
        if target_piece is None:
            moves.append(Move(from_coord=pos, to_coord=target, is_capture=False))
        elif target_piece.color != current_color:
            moves.append(Move(from_coord=pos, to_coord=target, is_capture=True))
        # Skip if friendly piece

    return moves

# Optional helpers (keep if used elsewhere)
def get_bubble_offsets():
    return _BUBBLE_OFFSETS.copy()

def get_anchor_offsets():
    return _ANCHOR_OFFSETS.copy()

def count_valid_echo_moves_from(state: GameState, x: int, y: int, z: int) -> int:
    return len(generate_echo_moves(state, x, y, z))

def get_echo_theoretical_reach() -> int:
    return 8 * len(_BUBBLE_OFFSETS)  # 256
