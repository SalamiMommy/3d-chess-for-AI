from enum import IntEnum, unique

@unique
class Color(IntEnum):
    WHITE = 0
    BLACK = 1

@unique
class PieceType(IntEnum):
    # 42 unique types – placeholders until you define movement
    TYPE_00 = 0
    TYPE_01 = 1
    ...
    TYPE_41 = 41

# Convenience
N_PIECE_TYPES = 42
