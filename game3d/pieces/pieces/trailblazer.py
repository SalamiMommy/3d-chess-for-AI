"""Master definition for Trailblazer – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.trailblazermoves import trailblazer_dispatcher

DISPATCHER = trailblazer_dispatcher
CACHES = ["trailblaze"]
