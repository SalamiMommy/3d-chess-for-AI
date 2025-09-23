"""Master definition for Hive – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.hivemoves import hive_dispatcher

DISPATCHER = hive_dispatcher
CACHES = []  # no auras (rule is enforced in submit_hive_turn)
