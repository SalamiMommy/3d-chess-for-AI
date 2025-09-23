"""Hive dispatcher – emits **one move per Hive piece**; UI aggregates."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game.move import Move
from game3d.movement.registry import register
from game3d.movement.movetypes.hivemovement import generate_hive_moves


@register(PieceType.HIVE)
def hive_dispatcher(state: GameState, x: int, y, z: int) -> List[Move]:
    return generate_hive_moves(state, x, y, z)

__all__ = ["hive_dispatcher"]
