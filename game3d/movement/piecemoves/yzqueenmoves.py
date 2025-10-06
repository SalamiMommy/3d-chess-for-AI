"""Exports YZ queen move generator (queen + king moves in YZ plane) and registers it."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.yzqueenmovement import generate_yz_queen_moves
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movepiece import Move


def generate_yz_queen_with_king_moves(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    queen_moves = generate_yz_queen_moves(state.cache, state.color, x, y, z)
    king_moves = generate_king_moves(state.cache, state.color, x, y, z)

    in_plane_king_moves = [
        move for move in king_moves
        if move.to_coord[0] == x  # X unchanged → in YZ plane
    ]

    seen_targets = {move.to_coord for move in queen_moves}
    combined_moves = list(queen_moves)

    for move in in_plane_king_moves:
        if move.to_coord not in seen_targets:
            combined_moves.append(move)
            seen_targets.add(move.to_coord)

    return combined_moves


@register(PieceType.YZQUEEN)
def yz_queen_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    # ✅ Pass 'state', not board/color
    return generate_yz_queen_with_king_moves(state, x, y, z)


__all__ = ['generate_yz_queen_with_king_moves']
