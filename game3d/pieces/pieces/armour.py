"""Master definition for Armour – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.armourmoves import armour_dispatcher

DISPATCHER = armour_dispatcher
CACHES = ["armoured"]  # pawn-capture immunity
