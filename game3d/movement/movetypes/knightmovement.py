# ------------------------------------------------------------------
#  New knight generator – share-square with *any* friendly
# ------------------------------------------------------------------
from typing import List
import numpy as np

from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move
from game3d.movement.movetypes.jumpmovement import (
    get_integrated_jump_movement_generator,
)

# 24 knight offsets (unchanged)
KNIGHT_OFFSETS = np.array([
    (1, 2, 0), (1, -2, 0), (-1, 2, 0), (-1, -2, 0),
    (2, 1, 0), (2, -1, 0), (-2, 1, 0), (-2, -1, 0),
    (1, 0, 2), (1, 0, -2), (-1, 0, 2), (-1, 0, -2),
    (2, 0, 1), (2, 0, -1), (-2, 0, 1), (-2, 0, -1),
    (0, 1, 2), (0, 1, -2), (0, -1, 2), (0, -1, -2),
    (0, 2, 1), (0, 2, -1), (0, -2, 1), (0, -2, -1),
], dtype=np.int8)

def generate_knight_moves(state: "GameState", x: int, y: int, z: int) -> List[Move]:
    """Generate all legal knight moves from (x, y, z) – share-square with *any* friendly."""
    pos = (x, y, z)

    # 2. delegate the heavy lifting to the integrated jump generator
    gen = get_integrated_jump_movement_generator(state.cache)
    raw_moves = gen.generate_jump_moves(
        color=state.color,
        pos=pos,
        directions=KNIGHT_OFFSETS,
                 # we *do* want captures
    )

    # 3. filter out friendly-occupied squares *only* when they are **not** knights
    #    (the kernel already removed illegal king captures when enemy has priests)
    moves: List[Move] = []
    for m in raw_moves:
        occupants = state.cache.pieces_at(m.to_coord)

        # empty or enemy → keep
        if not occupants or any(p.color != state.color for p in occupants):
            moves.append(m)
            continue

        # square contains only friendly pieces → always allowed (share)
        if all(p.color == state.color for p in occupants):
            # convert capture flag to False (it is not a capture)
            moves.append(
                Move(from_coord=m.from_coord, to_coord=m.to_coord, is_capture=False)
            )

    return moves
