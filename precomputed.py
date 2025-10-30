# precompute.py
"""Module for precomputing jump targets and sliding rays for pieces."""
from __future__ import annotations
from typing import Dict, List, Tuple
from itertools import product, permutations
import os
import pickle
import numpy as np

from game3d.common.enums import PieceType

BOARD_SIZE = 9
Pos = Tuple[int, int, int]

# Directory for precomputed data
PRECOMPUTE_DIR = os.path.join(os.path.dirname(__file__), 'precomputed')
os.makedirs(PRECOMPUTE_DIR, exist_ok=True)

# Helper to generate leaper offsets (e.g., for knights)
def generate_leaper_offsets(major: int, minor: int = 0, trivial: int = 0) -> np.ndarray:
    """Generate all permutations and signs for leaper offsets like (major, minor, trivial)."""
    bases = set(permutations([major, minor, trivial]))
    offsets: List[List[int]] = []
    for base in bases:
        non_zero_indices = [i for i, v in enumerate(base) if v != 0]
        for signs in product([-1, 1], repeat=len(non_zero_indices)):
            off = list(base)
            for idx, sign in zip(non_zero_indices, signs):
                off[idx] *= sign
            offsets.append(off)
    return np.array(offsets, dtype=np.int32)

# Helper to generate univector directions (rook-like)
def generate_univector_directions() -> np.ndarray:
    """Generate axis-aligned directions (±1 in one axis, 0 elsewhere)."""
    offs: List[List[int]] = []
    for i in range(3):
        for s in [-1, 1]:
            off = [0, 0, 0]
            off[i] = s
            offs.append(off)
    return np.array(offs, dtype=np.int32)

# Helper to generate divector directions (bishop-like)
def generate_divector_directions() -> np.ndarray:
    """Generate plane-diagonal directions (±1 in two axes, 0 in one)."""
    offs: List[List[int]] = []
    for zero_axis in range(3):
        non_zero = [0, 1, 2]
        non_zero.remove(zero_axis)
        for s1, s2 in product([-1, 1], repeat=2):
            off = [0, 0, 0]
            off[non_zero[0]] = s1
            off[non_zero[1]] = s2
            offs.append(off)
    return np.array(offs, dtype=np.int32)

# Helper to generate trivector directions (3D diagonal)
def generate_trivector_directions() -> np.ndarray:
    """Generate full 3D diagonal directions (±1 in all axes)."""
    return np.array(list(product([-1, 1], repeat=3)), dtype=np.int32)

# New helper for king-like offsets (1 step in any direction)
def generate_king_offsets() -> np.ndarray:
    """Generate offsets for king-like movements (adjacent in 3D)."""
    return np.vstack((
        generate_univector_directions(),
        generate_divector_directions(),
        generate_trivector_directions()
    ))

# New helper for offsets within Chebyshev distance
def generate_within_chebyshev(dist: int) -> np.ndarray:
    """Generate all offsets within Chebyshev distance <= dist (excluding (0,0,0))."""
    offsets: List[List[int]] = []
    for dx, dy, dz in product(range(-dist, dist + 1), repeat=3):
        if (dx, dy, dz) != (0, 0, 0) and max(abs(dx), abs(dy), abs(dz)) <= dist:
            offsets.append([dx, dy, dz])
    return np.array(offsets, dtype=np.int32)

# New helper for pawn forward offsets (assuming +z forward; flip for black in usage)
def generate_pawn_forward_offsets() -> np.ndarray:
    """Generate offsets for pawn movements (forward and capture diagonals in +z)."""
    offsets: List[List[int]] = []
    # Forward non-capture
    offsets.append([0, 0, 1])
    # Capture diagonals (in xy plane)
    for dx, dy in product([-1, 0, 1], repeat=2):
        if dx == 0 and dy == 0:
            continue
        offsets.append([dx, dy, 1])
    return np.array(offsets, dtype=np.int32)

# Define jump offsets for jumping pieces
JUMP_OFFSETS: Dict[PieceType, np.ndarray] = {
    PieceType.PAWN: generate_pawn_forward_offsets(),
    PieceType.KNIGHT: generate_leaper_offsets(2, 1),
    PieceType.KING: generate_king_offsets(),
    PieceType.KNIGHT32: generate_leaper_offsets(3, 2),
    PieceType.KNIGHT31: generate_leaper_offsets(3, 1),
    PieceType.ARCHER: generate_within_chebyshev(2),
    PieceType.SPIRAL: generate_within_chebyshev(2),
    PieceType.BLACKHOLE: generate_king_offsets(),
    PieceType.WHITEHOLE: generate_king_offsets(),
    PieceType.FREEZER: generate_king_offsets(),
    # Additional jumping pieces can be added here if their offsets are known
}

# Define slider directions for sliding pieces
SLIDER_DIRECTIONS: Dict[PieceType, np.ndarray] = {
    PieceType.ROOK: generate_univector_directions(),
    PieceType.BISHOP: generate_divector_directions(),
    PieceType.QUEEN: np.vstack((
        generate_univector_directions(),
        generate_divector_directions()
    )),
    PieceType.PRIEST: generate_divector_directions(),
    PieceType.TRIGONALBISHOP: generate_trivector_directions(),
    PieceType.EDGEROOK: generate_univector_directions(),
    PieceType.XYQUEEN: np.array(
        [d for d in np.vstack((generate_univector_directions(), generate_divector_directions())) if d[2] == 0],
        dtype=np.int32
    ),
    PieceType.XZQUEEN: np.array(
        [d for d in np.vstack((generate_univector_directions(), generate_divector_directions())) if d[1] == 0],
        dtype=np.int32
    ),
    PieceType.YZQUEEN: np.array(
        [d for d in np.vstack((generate_univector_directions(), generate_divector_directions())) if d[0] == 0],
        dtype=np.int32
    ),
    PieceType.VECTORSLIDER: generate_univector_directions(),
    PieceType.CONESLIDER: generate_trivector_directions(),
    # Additional sliding pieces can be added with custom directions
}

# Load or precompute jumps per piece
PRECOMPUTED_JUMPS: Dict[PieceType, Dict[Pos, List[Pos]]] = {}
for ptype, offsets in JUMP_OFFSETS.items():
    jumps_file = os.path.join(PRECOMPUTE_DIR, f'{ptype.name.lower()}_jumps.pkl')
    if os.path.exists(jumps_file):
        with open(jumps_file, 'rb') as f:
            PRECOMPUTED_JUMPS[ptype] = pickle.load(f)
    else:
        jumps: Dict[Pos, List[Pos]] = {}
        for px, py, pz in product(range(BOARD_SIZE), repeat=3):
            pos = (px, py, pz)
            valid: List[Pos] = []
            for dx, dy, dz in offsets:
                nx, ny, nz = px + dx, py + dy, pz + dz
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and 0 <= nz < BOARD_SIZE:
                    valid.append((nx, ny, nz))
            jumps[pos] = valid
        PRECOMPUTED_JUMPS[ptype] = jumps
        with open(jumps_file, 'wb') as f:
            pickle.dump(jumps, f)

# Load or precompute rays per piece
PRECOMPUTED_RAYS: Dict[PieceType, Dict[Pos, List[List[Pos]]]] = {}
for ptype, dirs in SLIDER_DIRECTIONS.items():
    rays_file = os.path.join(PRECOMPUTE_DIR, f'{ptype.name.lower()}_rays.pkl')
    if os.path.exists(rays_file):
        with open(rays_file, 'rb') as f:
            PRECOMPUTED_RAYS[ptype] = pickle.load(f)
    else:
        rays: Dict[Pos, List[List[Pos]]] = {}
        for px, py, pz in product(range(BOARD_SIZE), repeat=3):
            pos = (px, py, pz)
            pos_rays: List[List[Pos]] = []
            for dx, dy, dz in dirs:
                ray: List[Pos] = []
                step = 1
                while True:
                    nx, ny, nz = px + step * dx, py + step * dy, pz + step * dz
                    if not (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and 0 <= nz < BOARD_SIZE):
                        break
                    ray.append((nx, ny, nz))
                    step += 1
                pos_rays.append(ray)
            rays[pos] = pos_rays
        PRECOMPUTED_RAYS[ptype] = rays
        with open(rays_file, 'wb') as f:
            pickle.dump(rays, f)

__all__ = [
    "PRECOMPUTED_JUMPS",
    "PRECOMPUTED_RAYS",
    "JUMP_OFFSETS",
    "SLIDER_DIRECTIONS",
    "generate_leaper_offsets",
    "generate_univector_directions",
    "generate_divector_directions",
    "generate_trivector_directions",
    "generate_king_offsets",
    "generate_within_chebyshev",
    "generate_pawn_forward_offsets",
]
