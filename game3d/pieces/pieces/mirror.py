"""Master definition for Mirror – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.mirrormoves import mirror_dispatcher

DISPATCHER = mirror_dispatcher
CACHES = []
