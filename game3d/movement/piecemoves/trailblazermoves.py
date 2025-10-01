"""Exports Trailblazer moves (3-square rook slides) and registers them."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.rookmovement import generate_rook_moves
from game3d.movement.movepiece import Move

@register(PieceType.TRAILBLAZER)
def trailblazer_move_dispatcher(state, x: int, y: int, z: int) -> List[Move]:
    """Generate all legal Trailblazer moves – max 3 squares in each rook direction."""
    return generate_rook_moves(
        cache=state.cache,
        color=state.color,
        x=x,
        y=y,
        z=z,
        max_steps=3  # ✅ now supported!
    )

__all__ = ['trailblazer_move_dispatcher']
