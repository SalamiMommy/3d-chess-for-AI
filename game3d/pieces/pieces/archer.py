"""Master definition for Archer – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.archermoves import archer_dispatcher

DISPATCHER = archer_dispatcher
CACHES = ["archery"]
