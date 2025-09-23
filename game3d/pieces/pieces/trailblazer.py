"""Master definition for Trailblazer – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.trailblazermoves import trailblazer_dispatcher

DISPATCHER = trailblazer_dispatcher
CACHES = ["trailblaze"]
