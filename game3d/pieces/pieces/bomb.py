"""Master definition for Bomb – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.bombmoves import bomb_dispatcher

DISPATCHER = bomb_dispatcher
CACHES = []  # detonation is instantaneous, no persistent cache
