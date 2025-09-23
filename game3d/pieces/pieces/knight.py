"""Master definition for Knight – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.knightmoves import knight_dispatcher

DISPATCHER = knight_dispatcher
CACHES = ["share_square"]  # Share-Square multi-occupancy
