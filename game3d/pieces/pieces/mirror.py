"""Master definition for Mirror – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.mirrormoves import mirror_dispatcher

DISPATCHER = mirror_dispatcher
CACHES = []
