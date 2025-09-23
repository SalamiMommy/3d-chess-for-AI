"""Master definition for Geomancer – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.geomancermoves import geomancer_dispatcher

DISPATCHER = geomancer_dispatcher
CACHES = ["geomancy"]
