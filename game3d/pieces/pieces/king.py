"""Master definition for King – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.kingmoves import king_dispatcher

DISPATCHER = king_dispatcher
CACHES = []  # no auras
