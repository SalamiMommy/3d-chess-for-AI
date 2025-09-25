"""Master definition for Black-Hole – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.blackholemoves import blackhole_dispatcher

DISPATCHER = blackhole_dispatcher
CACHES = ["black_hole_suck"]
