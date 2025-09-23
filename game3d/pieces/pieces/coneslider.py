"""Master definition for Cone-Slider – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.coneslidemoves import coneslide_dispatcher

DISPATCHER = coneslide_dispatcher
CACHES = []
