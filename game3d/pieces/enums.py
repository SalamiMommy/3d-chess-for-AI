from enum import IntEnum, unique

@unique
class Color(IntEnum):
    WHITE = 0
    BLACK = 1

from enum import IntEnum

class PieceType(IntEnum):
    PAWN = 0
    KNIGHT = 1
    BISHOP = 2
    ROOK = 3
    QUEEN = 4
    KING = 5
    ORBITAL = 6  # ← ADD THIS LINE
    # ... continue up to 39 for 40 total pieces
    # e.g., CUSTOM_1 = 7, ..., CUSTOM_34 = 39

# Convenience
N_PIECE_TYPES = 40
