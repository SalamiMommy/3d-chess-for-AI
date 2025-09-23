"""Master definition for Panel – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.panelmoves import panel_dispatcher

DISPATCHER = panel_dispatcher
CACHES = []
