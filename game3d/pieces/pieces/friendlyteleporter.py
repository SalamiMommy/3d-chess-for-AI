"""Master definition for Friendly-Teleporter – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.friendlyteleportmoves import friendlyteleporter_dispatcher

DISPATCHER = friendlyteleporter_dispatcher
CACHES = []
