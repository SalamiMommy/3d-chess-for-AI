"""Master definition for Cone-Slider – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.coneslidemoves import coneslide_dispatcher

DISPATCHER = coneslide_dispatcher
CACHES = []
