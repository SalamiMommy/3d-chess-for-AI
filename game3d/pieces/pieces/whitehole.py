"""Master definition for White-Hole – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.whiteholemoves import whitehole_dispatcher

DISPATCHER = whitehole_dispatcher
CACHES = ["white_hole_push"]
