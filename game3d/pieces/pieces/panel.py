"""Master definition for Panel – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.panelmoves import panel_dispatcher

DISPATCHER = panel_dispatcher
CACHES = []
