# game3d/movement/movetypes/nebulamovement.py

"""Nebula piece — jumps to any square within or on sphere of radius 3 (Euclidean distance <= 3).
Pure movement logic — no registration, no dispatcher.
"""

from __future__ import annotations
from typing import List, Tuple
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.pathvalidation import jump_to_targets, validate_piece_at
from common import Coord


# ==============================================================================
# Precompute all 3D offsets where dx² + dy² + dz² <= 9 and not (0,0,0)
# Generated once when module loads — forms a "blocky sphere" on cubic grid
# ==============================================================================
_NEBULA_OFFSETS: List[Tuple[int, int, int]] = [
    (dx, dy, dz)
    for dx in range(-3, 4)  # -3 to +3 inclusive
    for dy in range(-3, 4)
    for dz in range(-3, 4)
    if not (dx == 0 and dy == 0 and dz == 0)  # exclude self
    if dx*dx + dy*dy + dz*dz <= 9  # Euclidean² <= 9 → distance <= 3
]

# ✅ Total: 122 offsets — forms a discrete "blocky sphere" (Euclidean ball on cubic grid)
# Looks like a fuzzy ball made of cubes — denser near center, sparse near edges.
# Perfect for grid-based 3D chess — no smooth geometry, just integer coordinates.


def generate_nebula_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """
    Generate all legal moves for the Nebula piece from (x, y, z).

    The Nebula jumps to any square within or on the sphere of radius 3:
        √(Δx² + Δy² + Δz²) <= 3

    ✅ Movement is "blocky" — aligned to cubic grid.
    ✅ Ignores blocking pieces — pure jump.
    ✅ Can land on empty squares or capture enemy pieces.

    Refactored to use shared jump logic from pathvalidation.py.
    """
    pos: Coord = (x, y, z)

    # Validate piece exists and is correct type/color
    if not validate_piece_at(state, pos, PieceType.NEBULA):
        return []

    # Delegate to shared jump logic — consistent with Knight, Orbital, etc.
    return jump_to_targets(
        state,
        start=pos,
        offsets=_NEBULA_OFFSETS,
        allow_capture=True,      # Can capture enemies
        allow_self_block=False   # Cannot land on friendly pieces
    )


# ==============================================================================
# Optional: Helper functions (unchanged)
# ==============================================================================

def get_nebula_offsets() -> List[Tuple[int, int, int]]:
    """Return immutable copy of precomputed nebula offsets (for testing/debug)."""
    return _NEBULA_OFFSETS.copy()


def count_valid_nebula_moves_from(state: GameState, x: int, y: int, z: int) -> int:
    """Utility: count how many nebula moves are possible from given coord."""
    return len(generate_nebula_moves(state, x, y, z))


def get_nebula_reach_volume() -> int:
    """Return total number of offsets (theoretical max moves from center)."""
    return len(_NEBULA_OFFSETS)  # 122
