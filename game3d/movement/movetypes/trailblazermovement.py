#game3d/movement/movetypes/trailblazermovement.py
"""Trailblazer — rook moves capped at 3 squares per ray; marks full path for aura."""

from typing import List, Set, Tuple
from game3d.pieces.enums import PieceType
from game3d.game.gamestate import GameState
from game3d.movement.movepiece import Move
from game3d.movement.pathvalidation import slide_along_directions, validate_piece_at

ROOK_DIRECTIONS_3D = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]


def generate_trailblazer_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """Generate all 3-step rook slides from (x, y, z) and mark the full slid path."""
    start = (x, y, z)

    if not validate_piece_at(state, start, expected_type=PieceType.TRAILBLAZER):
        return []

    # ---------- collect moves ----------
    moves = slide_along_directions(
        state=state,
        start=start,
        directions=ROOK_DIRECTIONS_3D,
        allow_capture=True,
        allow_self_block=False,
        max_steps=3,
        edge_only=False,
    )

    # ---------- mark full slid path in aura cache ----------
    cache = state.cache._effect["trailblaze"]  # ← USE STATE'S CACHE INSTEAD!
    for mv in moves:
        # rebuild the entire ray for this move
        dx, dy, dz = mv.to_coord[0] - start[0], mv.to_coord[1] - start[1], mv.to_coord[2] - start[2]
        step = max(abs(dx), abs(dy), abs(dz))  # Chebyshev length
        if step == 0:
            continue
        direction = (dx // step, dy // step, dz // step)

        path: Set[Tuple[int, int, int]] = set()
        for s in range(1, step + 1):
            path.add((start[0] + direction[0] * s,
                      start[1] + direction[1] * s,
                      start[2] + direction[2] * s))
        cache.mark_trail(start, path)  # FIFO history inside cache

    return moves
