"""Master definition for Vector-Slider – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.vectorslidermoves import vectorslider_dispatcher

DISPATCHER = vectorslider_dispatcher
CACHES = []
