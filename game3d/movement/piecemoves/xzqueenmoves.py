# game3d/movement/piecemoves/xzqueenmoves.py
"""Exports XZ queen move generator (queen + king moves in XZ plane) and registers it."""

from typing import List
from game3d.pieces.enums import PieceType
from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # ← run-time no-op
    from game3d.game.gamestate import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.xzqueenmovement import generate_xz_queen_moves
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movepiece import Move

def generate_xz_queen_with_king_moves(board, color, *coord, cache=None) -> List[Move]:
    """
    Combines XZ queen sliding moves + 1-step king moves within XZ plane (Y fixed).
    Deduplicates by target coordinate.
    """
    # Create the state object that was missing
    from game3d.game.gamestate import GameState
    state = GameState(board, color, cache=cache)

    queen_moves = generate_xz_queen_moves(state, *coord)
    king_moves = generate_king_moves(state, *coord)

    # Get the y coordinate from the input
    x, y, z = coord

    # Filter king moves to only those in XZ plane (dy = 0)
    in_plane_king_moves = [
        move for move in king_moves
        if move.to_coord[1] == y  # Y unchanged → in XZ plane
    ]

    seen_targets = set(move.to_coord for move in queen_moves)
    combined_moves = queen_moves[:]

    for move in in_plane_king_moves:
        if move.to_coord not in seen_targets:
            combined_moves.append(move)
            seen_targets.add(move.to_coord)

    return combined_moves


@register(PieceType.XZQUEEN)
def xz_queen_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
    return generate_xz_queen_with_king_moves(board, color, *coord)


# Re-export for external use
__all__ = ['generate_xz_queen_with_king_moves']
