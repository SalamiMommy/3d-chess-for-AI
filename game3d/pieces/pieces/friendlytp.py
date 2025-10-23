# game3d/movement/pieces/friendlytp.py
"""
Friendly-Teleporter – teleport to any empty neighbour of a friendly piece
PLUS normal 1-step King moves.
"""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING
from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movepiece import Move
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.movement.cache_utils import ensure_int_coords
from game3d.common.coord_utils import in_bounds_vectorised

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# --------------------------------------------------------------------------- #
#  King directions (1-step moves)                                             #
# --------------------------------------------------------------------------- #
_KING_DIRECTIONS = np.array([
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
], dtype=np.int8)

def generate_friendlytp_moves(
    cache: 'OptimizedCacheManager',
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate friendly teleporter moves: king walks + network teleports."""
    x, y, z = ensure_int_coords(x, y, z)
    start = np.array([x, y, z], dtype=np.int16)

    # Get teleport directions
    teleport_dirs = _build_network_directions(cache, color, x, y, z)

    # Combine all directions
    if len(teleport_dirs) > 0:
        all_dirs = np.unique(np.vstack((teleport_dirs, _KING_DIRECTIONS)), axis=0)
    else:
        all_dirs = _KING_DIRECTIONS

    # Generate all moves using jump movement
    jump_gen = get_integrated_jump_movement_generator(cache)
    moves = jump_gen.generate_jump_moves(
        color=color,
        pos=(x, y, z),
        directions=all_dirs,
        allow_capture=True,
    )

    # Mark teleport moves with metadata
    teleport_set = {tuple(d) for d in teleport_dirs}
    for move in moves:
        direction = tuple(np.array(move.to_coord) - start)
        if direction in teleport_set:
            move.metadata["is_teleport"] = True

    return moves

def _build_network_directions(
    cache: 'OptimizedCacheManager',
    color: Color,
    x: int, y: int, z: int
) -> np.ndarray:
    """Build directions to empty squares adjacent to friendly pieces."""
    start = np.array([x, y, z], dtype=np.int16)

    # Collect friendly pieces
    friendly_coords = []
    for coord, piece in cache.occupancy.iter_color(color):
        if coord != (x, y, z):
            friendly_coords.append(coord)

    if not friendly_coords:
        return np.empty((0, 3), dtype=np.int8)

    friendly_arr = np.array(friendly_coords, dtype=np.int16)

    # Get all neighbors of friendly pieces
    neighbours = (friendly_arr[:, np.newaxis, :] + _KING_DIRECTIONS[np.newaxis, :, :]).reshape(-1, 3)

    # Filter: in-bounds and empty
    valid_mask = in_bounds_vectorised(neighbours)
    neighbours = neighbours[valid_mask]

    if len(neighbours) == 0:
        return np.empty((0, 3), dtype=np.int8)

    # Check occupancy
    occ_mask = (cache.occupancy._occ != 0)
    x_coords = np.clip(neighbours[:, 0], 0, 8)
    y_coords = np.clip(neighbours[:, 1], 0, 8)
    z_coords = np.clip(neighbours[:, 2], 0, 8)
    empty_mask = ~occ_mask[z_coords, y_coords, x_coords]

    empty_neighbours = neighbours[empty_mask]

    # Remove duplicates and start position
    unique_targets = np.unique(empty_neighbours, axis=0)
    unique_targets = unique_targets[~np.all(unique_targets == start, axis=1)]

    # Convert to directions
    directions = (unique_targets - start).astype(np.int8)

    # Final validation
    dest_coords = start + directions
    valid_mask = np.all((dest_coords >= 0) & (dest_coords < 9), axis=1)
    return directions[valid_mask]

@register(PieceType.FRIENDLYTELEPORTER)
def friendlytp_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    return generate_friendlytp_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_friendlytp_moves"]
