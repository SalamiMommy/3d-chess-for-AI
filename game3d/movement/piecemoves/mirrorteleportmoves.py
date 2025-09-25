"""Exports Mirror-Teleport + king moves (combined) and registers them."""
#game3d/movement/piecemoves/mirrorteleportmoves.py
from typing import List
from game3d.pieces.enums import PieceType
from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # ← run-time no-op
    from game3d.game.gamestate import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.mirrorteleportmovement import generate_mirror_teleport_move
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movepiece import Move

def generate_mirror_teleport_with_king_moves(board, color, *coord, cache=None) -> List[Move]:
    from game3d.game.gamestate import GameState
    state = GameState(board, color, cache=cache)

    teleport_moves = generate_mirror_teleport_move(state, *coord)
    king_moves = generate_king_moves(state, *coord)

    seen_targets = set(move.to_coord for move in teleport_moves)
    combined_moves = teleport_moves[:]

    for move in king_moves:
        if move.to_coord not in seen_targets:
            combined_moves.append(move)
            seen_targets.add(move.to_coord)

    return combined_moves


@register(PieceType.MIRROR)
def mirror_teleport_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
    from game3d.game.gamestate import GameState
    state = GameState(board, color, cache=cache)
    return generate_mirror_teleport_with_king_moves(state, *coord)


# Re-export for external use
__all__ = ['generate_mirror_teleport_with_king_moves']
