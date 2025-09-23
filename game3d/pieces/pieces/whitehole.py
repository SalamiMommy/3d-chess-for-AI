"""Master definition for White-Hole – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.whiteholemoves import whitehole_dispatcher

DISPATCHER = whitehole_dispatcher
CACHES = ["white_hole_push"]
