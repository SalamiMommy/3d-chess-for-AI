"""Master definition for Slower – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.slowermoves import slower_dispatcher

DISPATCHER = slower_dispatcher
CACHES = ["movement_debuff"]
