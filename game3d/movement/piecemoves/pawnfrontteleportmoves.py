"""Exports Pawn-Front-Teleport + king moves (combined) and registers them."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.pawnfrontteleportmovement import generate_pawn_front_teleport_moves
from game3d.movement.movetypes.kingmovement import generate_king_moves


def generate_pawn_front_teleport_with_king_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """Combine pawn-front teleport moves + 1-step king moves, deduplicated."""
    teleport_moves = generate_pawn_front_teleport_moves(state, x, y, z)
    king_moves = generate_king_moves(state, x, y, z)

    seen = {m.to_coord for m in teleport_moves}
    combined = teleport_moves[:]
    for m in king_moves:
        if m.to_coord not in seen:
            combined.append(m)
            seen.add(m.to_coord)
    return combined


@register(PieceType.PAWN_FRONT_TELEPORTER)
def pawn_front_teleporter_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """Registered dispatcher for Pawn-Front-Teleporter moves (now with king moves)."""
    return generate_pawn_front_teleport_with_king_moves(state, x, y, z)


__all__ = ['generate_pawn_front_teleport_with_king_moves']
