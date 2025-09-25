# game3d/movement/piecemoves/xyqueenmoves.py

"""Exports XY queen move generator (queen + king moves in XY plane) and registers it."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.game.gamestate import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.xyqueenmovement import generate_xy_queen_moves
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movepiece import Move

def generate_xy_queen_with_king_moves(board, color, *coord, cache=None) -> List[Move]:
def generate_xy_queen_with_king_moves    from game3d.game.gamestate import GameState
def generate_xy_queen_with_king_moves    state = GameState(board, color, cache=cache)
    Deduplicates by target coordinate.
    """
    queen_moves = generate_xy_queen_moves(state, x, y, z)
    king_moves = generate_king_moves(state, x, y, z)

    # Filter king moves to only those in XY plane (dz = 0)
    in_plane_king_moves = [
        move for move in king_moves
        if move.to_coord[2] == z  # Z unchanged → in XY plane
    ]

    # Deduplicate by target square (prefer queen move if both exist — shouldn't happen, but safe)
    seen_targets = set(move.to_coord for move in queen_moves)
    combined_moves = queen_moves[:]

    for move in in_plane_king_moves:
        if move.to_coord not in seen_targets:
            combined_moves.append(move)
            seen_targets.add(move.to_coord)

    return combined_moves


@register(PieceType.XYQUEEN)
def xy_queen_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
def xy_queen_move_dispatcher    from game3d.game.gamestate import GameState
def xy_queen_move_dispatcher    state = GameState(board, color, cache=cache)
    Delegates to combined queen + king movement in XY plane.
    """
    return generate_xy_queen_with_king_moves(state, x, y, z)


# Re-export for external use
__all__ = ['generate_xy_queen_with_king_moves']
