"""Master definition for Archer – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.archermoves import archer_dispatcher

DISPATCHER = archer_dispatcher
CACHES = ["archery"]
