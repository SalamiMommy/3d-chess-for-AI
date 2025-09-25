"""Master definition for Slower – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.slowermoves import slower_dispatcher

DISPATCHER = slower_dispatcher
CACHES = ["movement_debuff"]
