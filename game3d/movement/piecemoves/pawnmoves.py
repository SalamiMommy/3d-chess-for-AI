# game3d/movement/piecemoves/pawnmoves.py
"""Exports pawn move generator and registers it with the dispatcher."""
from game3d.pieces.enums import PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.pawnmovement import generate_pawn_moves

@register(PieceType.PAWN)
def pawn_move_dispatcher(state, x, y, z):
    return generate_pawn_moves(state.cache, state.color, x, y, z)
