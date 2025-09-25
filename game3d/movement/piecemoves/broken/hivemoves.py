"""Hive dispatcher – emits **one move per Hive piece**; UI aggregates."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.game.gamestate import GameState
from game3d.movement.movepiece import Move
from game3d.movement.registry import register
from game3d.movement.movetypes.kingmovement import generate_king_moves  # Hive ≡ King

generate_hive_moves = generate_king_moves  # alias


@register(PieceType.HIVE)
def hive_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
    return generate_hive_moves(state, x, y, z)


__all__ = ["hive_dispatcher"]
