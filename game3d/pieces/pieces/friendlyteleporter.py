"""Master definition for Friendly-Teleporter – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.friendlyteleportmoves import friendlyteleporter_dispatcher

DISPATCHER = friendlyteleporter_dispatcher
CACHES = []
