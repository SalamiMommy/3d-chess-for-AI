# game3d/movement/movetypes/orbitalmovement.py

"""Orbital piece — jumps to any square with Manhattan distance 4.
Pure movement logic — no registration, no dispatcher.
"""

from __future__ import annotations
from typing import List, Tuple
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.pathvalidation import jump_to_targets, validate_piece_at
from common import Coord


# ==============================================================================
# Precompute all 3D offsets where |dx| + |dy| + |dz| == 4
# Generated once when module loads — no runtime cost
# ==============================================================================
_ORBITAL_OFFSETS: List[Tuple[int, int, int]] = [
    (dx, dy, dz)
    for dx in range(-4, 5)
    for dy in range(-4, 5)
    for dz in range(-4, 5)
    if abs(dx) + abs(dy) + abs(dz) == 4
]

# ✅ Total: 202 offsets — forms a "blocky sphere" (3D octahedron) on cubic grid
# This is the discrete L1-ball of radius 4 — looks like a diamond in 3D space.
# Perfectly suited for cubic chess boards — no interpolation, no curves, just cubes.


def generate_orbital_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """
    Generate all legal moves for the Orbital piece from (x, y, z).

    The Orbital jumps to any square exactly Manhattan distance 4 away:
        |Δx| + |Δy| + |Δz| = 4

    ✅ Movement is "blocky" — aligned to cubic grid.
    ✅ Ignores blocking pieces — pure jump.
    ✅ Can land on empty squares or capture enemy pieces.

    Refactored to use shared jump logic from pathvalidation.py.
    """
    pos: Coord = (x, y, z)

    # Validate piece exists and is correct type/color
    if not validate_piece_at(state, pos, PieceType.ORBITAL):
        return []

    # Delegate to shared jump logic — consistent with Knight, etc.
    return jump_to_targets(
        state,
        start=pos,
        offsets=_ORBITAL_OFFSETS,
        allow_capture=True,      # Can capture enemies
        allow_self_block=False   # Cannot land on friendly pieces
    )


# ==============================================================================
# Optional: Helper functions (unchanged)
# ==============================================================================

def get_orbital_offsets() -> List[Tuple[int, int, int]]:
    """Return immutable copy of precomputed orbital offsets (for testing/debug)."""
    return _ORBITAL_OFFSETS.copy()


def count_valid_orbital_moves_from(state: GameState, x: int, y: int, z: int) -> int:
    """Utility: count how many orbital moves are possible from given coord."""
    return len(generate_orbital_moves(state, x, y, z))
