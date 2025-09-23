"""Bulk registration of pure king movers."""

from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.registry import register
from pieces.enums import PieceType

# --- register every listed type as a pure king mover ---
pure_king_pieces = [
    PieceType.PRIEST,
    PieceType.TWIN,
    PieceType.FREEZER,
    PieceType.WALL,
    PieceType.ARCHER,
    PieceType.BOMB,
    PieceType.SPEEDER,
    PieceType.SLOWER,
    PieceType.GEOMANCER,
    PieceType.SWAPPER,
    PieceType.BLACK_HOLE,
    PieceType.WHITE_HOLE,
]

for p in pure_king_pieces:
    register(p)(generate_king_moves)


# re-export for convenience
__all__ = []  # nothing to export; registry is already populated
