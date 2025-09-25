"""Exports Network-Teleport + king moves (combined) and registers them."""

from typing import List
from game3d.pieces.enums import PieceType
from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # ← run-time no-op
    from game3d.game.gamestate import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.networkteleportmovement import generate_network_teleport_moves
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movepiece import Move

def generate_network_teleport_with_king_moves(board, color, *coord, cache=None) -> List[Move]:
    """
    Combines network-teleport moves + 1-step king moves.
    Deduplicates by target coordinate.
    """
    from game3d.game.gamestate import GameState
    state = GameState(board, color, cache=cache)

    teleport_moves = generate_network_teleport_moves(state, *coord)
    king_moves = generate_king_moves(state, *coord)

    seen_targets = set(move.to_coord for move in teleport_moves)
    combined_moves = teleport_moves[:]

    for move in king_moves:
        if move.to_coord not in seen_targets:
            combined_moves.append(move)
            seen_targets.add(move.to_coord)

    return combined_moves


@register(PieceType.FRIENDLYTELEPORTER)
def network_teleport_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
    return generate_network_teleport_with_king_moves(board, color, *coord)


# Re-export for external use
__all__ = ['generate_network_teleport_with_king_moves']
