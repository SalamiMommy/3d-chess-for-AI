"""Master definition for Black-Hole – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.blackholemoves import blackhole_dispatcher

DISPATCHER = blackhole_dispatcher
CACHES = ["black_hole_suck"]
