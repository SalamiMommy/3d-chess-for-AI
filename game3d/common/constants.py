"""Physical constants for 9×9×9 3-D chess."""

from typing import Final

SIZE_X: Final[int] = 9
SIZE_Y: Final[int] = 9
SIZE_Z: Final[int] = 9

SIZE: Final[tuple[int, int, int]] = (SIZE_X, SIZE_Y, SIZE_Z)
VOLUME: Final[int] = SIZE_X * SIZE_Y * SIZE_Z

N_PIECE_TYPES: Final[int] = 42
N_PLANES_PER_SIDE: Final[int] = N_PIECE_TYPES
N_COLOR_PLANES: Final[int] = N_PLANES_PER_SIDE * 2
N_AUX_PLANES: Final[int] = 1                      # current player
N_TOTAL_PLANES: Final[int] = N_COLOR_PLANES + N_AUX_PLANES

# Coordinate system conventions
X, Y, Z = 0, 1, 2          # index into tuples
