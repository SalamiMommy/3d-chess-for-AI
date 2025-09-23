from enum import IntEnum, unique

@unique
class Color(IntEnum):
    WHITE = 0
    BLACK = 1


class PieceType(IntEnum):
    # Base orthodox set
    PAWN          = 0
    KNIGHT        = 1
    BISHOP        = 2
    ROOK          = 3
    QUEEN         = 4
    KING          = 5

    # New 3-D / special pieces (40 total)
    PRIEST              = 6
    _32KNIGHT           = 7   # 3,2 leaper
    _31KNIGHT           = 8   # 3,1 leaper
    TRIGONALBISHOP      = 9   # diagonal in XY, XZ, YZ planes
    HIVE                = 10  # move every friendly Hive once per turn
    ORBITER             = 11  # orbital movement (placeholder)
    NEBULA              = 12  # area cloud effect (placeholder)
    ECHO                = 13  # echo / copy (placeholder)
    PANEL               = 14  # panel / wall segment (placeholder)
    EDGEROOK            = 15  # rook bounded to board edges
    XYQUEEN             = 16  # queen in XY plane only
    XZQUEEN             = 17  # queen in XZ plane only
    YZQUEEN             = 18  # queen in YZ plane only
    VECTORSLIDER        = 19  # vector-slider (placeholder)
    CONESLIDER          = 20  # cone-slider (placeholder)
    MIRROR              = 21  # mirror teleport
    FREEZER             = 22  # freeze aura
    WALL                = 23  # wall piece
    ARCHER              = 24  # archery attack within 2-sphere
    BOMB                = 25  # detonates on capture / self-move
    FRIENDLYTELEPORTER  = 26  # teleports to friendly-adjacent empty
    ARMOUR              = 27  # immune to pawn captures
    SPEEDER             = 28  # +1 step buff
    SLOWER              = 29  # -1 step debuff
    GEOMANCER           = 30  # blocks squares for 5 plies
    SWAPPER             = 31  # swaps with any friendly
    XZZIGZAG            = 32  # zig-zag in XZ planes
    YZZIGZAG            = 33  # zig-zag in YZ planes
    REFLECTOR           = 34  # reflects off walls
    BLACKHOLE           = 35  # sucks enemies 1 step closer
    WHITEHOLE           = 36  # pushes enemies 1 step away
    INFILTRATOR         = 37  # infiltration (placeholder)
    TRAILBLAZER         = 38  # marks trail, counter removal
    SPIRAL              = 39  # counter-clockwise spiral (radius 2)

# Convenience constant
N_PIECE_TYPES = 40
