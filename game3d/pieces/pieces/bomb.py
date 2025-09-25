"""Master definition for Bomb – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.bombmoves import bomb_dispatcher

DISPATCHER = bomb_dispatcher
CACHES = []  # detonation is instantaneous, no persistent cache
