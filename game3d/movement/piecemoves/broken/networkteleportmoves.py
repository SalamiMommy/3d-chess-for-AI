"""Exports Network-Teleport + king moves (combined) and registers them."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.game.gamestate import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.networkteleportmovement import generate_network_teleport_moves
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movepiece import Move

def generate_network_teleport_with_king_moves(board, color, *coord, cache=None) -> List[Move]:
def generate_network_teleport_with_king_moves    from game3d.game.gamestate import GameState
def generate_network_teleport_with_king_moves    state = GameState(board, color, cache=cache)
    Deduplicates by target coordinate.
    """
    teleport_moves = generate_network_teleport_moves(state, x, y, z)
    king_moves = generate_king_moves(state, x, y, z)

    seen_targets = set(move.to_coord for move in teleport_moves)
    combined_moves = teleport_moves[:]

    for move in king_moves:
        if move.to_coord not in seen_targets:
            combined_moves.append(move)
            seen_targets.add(move.to_coord)

    return combined_moves


@register(PieceType.FRIENDLYTELEPORTER)
def network_teleport_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
    """
    Registered dispatcher for Network-Teleporter moves.
    Delegates to combined teleport + king movement.
    """
    return generate_network_teleport_with_king_moves(state, x, y, z)


# Re-export for external use
__all__ = ['generate_network_teleport_with_king_moves']
