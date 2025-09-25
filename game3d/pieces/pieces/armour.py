"""Master definition for Armour – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.armourmoves import armour_dispatcher

DISPATCHER = armour_dispatcher
CACHES = ["armoured"]  # pawn-capture immunity
