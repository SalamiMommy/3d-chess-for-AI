"""Exports Mirror-Teleport + king moves (combined) and registers them."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.mirrorteleportmovement import generate_mirror_teleport_move
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movepiece import Move


def generate_mirror_teleport_with_king_moves(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    """
    Combines mirror teleport moves + king moves.
    Deduplicates by target coordinate.
    """
    # Both functions must accept (state, x, y, z) - but they don't!
    # Instead, call them with the correct parameters
    teleport_moves = generate_mirror_teleport_move(state.board, state.color, x, y, z)
    king_moves = generate_king_moves(state.board, state.color, x, y, z)

    seen_targets = {move.to_coord for move in teleport_moves}
    combined_moves = list(teleport_moves)

    for move in king_moves:
        if move.to_coord not in seen_targets:
            combined_moves.append(move)
            seen_targets.add(move.to_coord)

    return combined_moves


@register(PieceType.MIRROR)
def mirror_teleport_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_mirror_teleport_with_king_moves(state, x, y, z)


__all__ = ['generate_mirror_teleport_with_king_moves']
