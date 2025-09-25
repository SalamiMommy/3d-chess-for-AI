"""Master definition for Geomancer – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.geomancermoves import geomancer_dispatcher

DISPATCHER = geomancer_dispatcher
CACHES = ["geomancy"]
