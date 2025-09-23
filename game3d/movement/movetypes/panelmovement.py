# game3d/movement/movetypes/panelmovement.py

"""Panel piece — jumps to any square on 6 orthogonal 3x3 walls centered 2 squares away along each axis.
Pure movement logic — no registration, no dispatcher.
"""

from __future__ import annotations
from typing import List, Tuple
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.pathvalidation import jump_to_targets, validate_piece_at


# ==============================================================================
# Precompute ALL 54 jump offsets relative to (0,0,0)
# Each is: 2 steps in one axis + 3x3 wall in perpendicular plane
# ==============================================================================
_PANEL_OFFSETS: List[Tuple[int, int, int]] = []

# Define directions and their associated wall planes
_DIRECTIONS_AND_PLANES = [
    ((1, 0, 0), 'yz'),   # +X → YZ wall
    ((-1, 0, 0), 'yz'),  # -X → YZ wall
    ((0, 1, 0), 'xz'),   # +Y → XZ wall
    ((0, -1, 0), 'xz'),  # -Y → XZ wall
    ((0, 0, 1), 'xy'),   # +Z → XY wall
    ((0, 0, -1), 'xy'),  # -Z → XY wall
]

# Define wall offsets per plane
_WALL_OFFSETS = {
    'yz': [(0, dy, dz) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
    'xz': [(dx, 0, dz) for dx in (-1, 0, 1) for dz in (-1, 0, 1)],
    'xy': [(dx, dy, 0) for dx in (-1, 0, 1) for dy in (-1, 0, 1)],
}

# Generate all 6 × 9 = 54 offsets
for direction, plane_key in _DIRECTIONS_AND_PLANES:
    dx, dy, dz = direction
    anchor_offset = (2*dx, 2*dy, 2*dz)
    for wx, wy, wz in _WALL_OFFSETS[plane_key]:
        total_offset = (
            anchor_offset[0] + wx,
            anchor_offset[1] + wy,
            anchor_offset[2] + wz
        )
        _PANEL_OFFSETS.append(total_offset)

# ✅ Total: 54 unique offsets — forms six 3×3 "blocky walls" floating 2 units away
# No duplicates — each wall is in distinct space


def generate_panel_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """
    Generate all legal moves for the Panel piece from (x, y, z).

    Projects 6 anchor points, 2 squares away along each orthogonal axis.
    At each anchor, creates a 3x3 wall (orthogonal plane) of target squares.

    ✅ Pure jumper — ignores blocking pieces.
    ✅ Can capture enemy pieces.
    ✅ Cannot land on friendly pieces.
    ✅ Movement is "blocky" — aligned to cubic grid.

    Now fully uses shared `jump_to_targets` logic — consistent with Knight, Orbital, Nebula.
    """
    pos = (x, y, z)

    # Validate piece exists and is correct type/color
    if not validate_piece_at(state, pos, PieceType.PANEL):
        return []

    # Delegate to shared jumper logic — same as Knight!
    return jump_to_targets(
        state,
        start=pos,
        offsets=_PANEL_OFFSETS,
        allow_capture=True,      # Can capture enemies
        allow_self_block=False   # Cannot land on friendlies
    )


# ==============================================================================
# Optional: Helper functions (unchanged)
# ==============================================================================

def get_panel_offsets() -> List[Tuple[int, int, int]]:
    """Return all 54 precomputed panel jump offsets."""
    return _PANEL_OFFSETS.copy()


def count_valid_panel_moves_from(state: GameState, x: int, y: int, z: int) -> int:
    """Count how many panel moves are possible from given coord."""
    return len(generate_panel_moves(state, x, y, z))


def get_panel_theoretical_reach() -> int:
    """Return max theoretical targets (6 walls × 9 squares = 54)."""
    return len(_PANEL_OFFSETS)
