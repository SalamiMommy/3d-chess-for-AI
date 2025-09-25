"""Master definition for Vector-Slider – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.vectorslidermoves import vectorslider_dispatcher

DISPATCHER = vectorslider_dispatcher
CACHES = []
