# game3d/movement/piecemoves/yzqueenmoves.py

"""Exports YZ queen move generator (queen + king moves in YZ plane) and registers it."""

from typing import List
from game3d.pieces.enums import PieceType
from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # ← run-time no-op
    from game3d.game.gamestate import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.yzqueenmovement import generate_yz_queen_moves
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movepiece import Move

def generate_yz_queen_with_king_moves(board, color, *coord, cache=None) -> List[Move]:
    """
    Combines YZ queen sliding moves + 1-step king moves within YZ plane (X fixed).
    Deduplicates by target coordinate.
    """
    queen_moves = generate_yz_queen_moves(state, *coord)
    king_moves = generate_king_moves(state, *coord)

    # Filter king moves to only those in YZ plane (dx = 0)
    in_plane_king_moves = [
        move for move in king_moves
        if move.to_coord[0] == x  # X unchanged → in YZ plane
    ]

    seen_targets = set(move.to_coord for move in queen_moves)
    combined_moves = queen_moves[:]

    for move in in_plane_king_moves:
        if move.to_coord not in seen_targets:
            combined_moves.append(move)
            seen_targets.add(move.to_coord)

    return combined_moves


@register(PieceType.YZQUEEN)
def yz_queen_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
    from game3d.game.gamestate import GameState
    state = GameState(board, color, cache=cache)
    return generate_yz_queen_with_king_moves(state, *coord)


# Re-export for external use
__all__ = ['generate_yz_queen_with_king_moves']
