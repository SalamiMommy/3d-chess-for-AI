"""Master definition for Edge-Rook – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.edgerookmoves import edgerook_dispatcher

DISPATCHER = edgerook_dispatcher
CACHES = []
