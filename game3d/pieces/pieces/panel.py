"""Panel movement generator - teleport on same x/y/z plane + king moves."""
import numpy as np
from typing import List, TYPE_CHECKING

from game3d.common.shared_types import COORD_DTYPE, SIZE, SIZE_MINUS_1
from game3d.common.shared_types import Color, PieceType
from game3d.common.registry import register
from game3d.movement.jump_engine import get_jump_movement_generator
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# Piece-specific movement vectors - teleport along same x/y/z plane + king moves
# Converted to numpy-native using meshgrid and arange for better performance

# X-axis lines (y,z constant) - teleport moves
x_range = np.arange(-SIZE_MINUS_1, SIZE, dtype=COORD_DTYPE)
x_range = x_range[x_range != 0]  # Remove 0
x_moves = np.column_stack([x_range, np.zeros_like(x_range), np.zeros_like(x_range)])

# Y-axis lines (x,z constant) - teleport moves
y_range = np.arange(-SIZE_MINUS_1, SIZE, dtype=COORD_DTYPE)
y_range = y_range[y_range != 0]  # Remove 0
y_moves = np.column_stack([np.zeros_like(y_range), y_range, np.zeros_like(y_range)])

# Z-axis lines (x,y constant) - teleport moves
z_range = np.arange(-SIZE_MINUS_1, SIZE, dtype=COORD_DTYPE)
z_range = z_range[z_range != 0]  # Remove 0
z_moves = np.column_stack([np.zeros_like(z_range), np.zeros_like(z_range), z_range])

# King moves (1-step) - converted to numpy-native using meshgrid
dx_vals, dy_vals, dz_vals = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing='ij')
all_coords = np.stack([dx_vals.ravel(), dy_vals.ravel(), dz_vals.ravel()], axis=1)
# Remove the (0, 0, 0) origin
origin_mask = np.all(all_coords != 0, axis=1)
king_moves = all_coords[origin_mask].astype(COORD_DTYPE)

# Combine all movement vectors
PANEL_MOVEMENT_VECTORS = np.vstack([x_moves, y_moves, z_moves, king_moves])

def generate_panel_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    pos_arr = pos.astype(COORD_DTYPE)

    # Validate position
    if not in_bounds_vectorized(pos_arr.reshape(1, 3))[0]:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Use jump generator with piece-specific vectors
    jump_engine = get_jump_movement_generator(cache_manager)
    return jump_engine.generate_jump_moves(
        color=color,
        pos=pos_arr,
        directions=PANEL_MOVEMENT_VECTORS,
        allow_capture=True,
        piece_type=PieceType.PANEL
    )

@register(PieceType.PANEL)
def panel_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_panel_moves(state.cache_manager, state.color, pos)

__all__ = ['PANEL_MOVEMENT_VECTORS', 'generate_panel_moves']
