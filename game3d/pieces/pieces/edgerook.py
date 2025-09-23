"""Master definition for Edge-Rook – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.edgerookmoves import edgerook_dispatcher

DISPATCHER = edgerook_dispatcher
CACHES = []
