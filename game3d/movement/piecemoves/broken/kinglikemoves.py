"""Bulk registration of pure king movers."""

from typing import List
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.registry import register
from game3d.pieces.enums import PieceType
from game3d.movement.movepiece import Move
from game3d.game.gamestate import GameState

# --- register every listed type as a pure king mover ---
pure_king_pieces = [
    PieceType.PRIEST,
    PieceType.FREEZER,
    PieceType.WALL,
    PieceType.ARCHER,
    PieceType.BOMB,
    PieceType.SPEEDER,
    PieceType.SLOWER,
    PieceType.GEOMANCER,
    PieceType.SWAPPER,
    PieceType.BLACKHOLE,
    PieceType.WHITEHOLE,
    PieceType.ARMOUR
]

for p in pure_king_pieces:
    register(p)(generate_king_moves)


@register(PieceType.KINGLIKE)
def kinglike_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
    """
    Registered dispatcher for king-like moves.
    Simply delegates to the core move generator.
    """
    return generate_king_moves(state, x, y, z)


# re-export for convenience
__all__ = []  # nothing to export; registry is already populated
