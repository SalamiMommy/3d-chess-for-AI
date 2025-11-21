"""King piece - imports from kinglike module."""
# Re-export everything from kinglike module
from game3d.pieces.pieces.kinglike import (
    KING_MOVEMENT_VECTORS,
    generate_king_moves,
    priest_move_dispatcher,
    king_move_dispatcher
)

__all__ = ['KING_MOVEMENT_VECTORS', 'generate_king_moves', 'priest_move_dispatcher', 'king_move_dispatcher']