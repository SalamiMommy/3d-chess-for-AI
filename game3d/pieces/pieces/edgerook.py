# edgerookmoves.py - FIXED
"""Edge-Rook – slides along the 9×9×9 edge graph, stops at first blocker, single jump-batch."""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING, Dict
from collections import deque

from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import in_bounds
from game3d.common.cache_utils import ensure_int_coords  # REMOVED: get_occupancy_safe

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
    from game3d.cache.manager import OptimizedCacheManager

# Build once: edge adjacency graph (6 axial directions)
_EDGE_GRAPH: Dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}

def _build_edge_graph(size: int = 9) -> None:
    global _EDGE_GRAPH
    if _EDGE_GRAPH:                       # already built
        return
    axial = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for x in range(size):
        for y in range(size):
            for z in range(size):
                if not (x == 0 or x == size - 1 or y == 0 or y == size - 1 or z == 0 or z == size - 1):
                    continue
                neighbours = []
                for dx, dy, dz in axial:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < size and 0 <= ny < size and 0 <= nz < size:
                        if (nx == 0 or nx == size - 1 or ny == 0 or ny == size - 1 or nz == 0 or nz == size - 1):
                            neighbours.append((nx, ny, nz))
                _EDGE_GRAPH[(x, y, z)] = neighbours

_build_edge_graph()

def generate_edgerook_moves(
    cache_manager: 'OptimizedCacheManager',
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate edge-rook moves using single cache manager."""
    x, y, z = ensure_int_coords(x, y, z)
    start = (x, y, z)
    if start not in _EDGE_GRAPH:
        return []

    visited: set[tuple[int, int, int]] = set()
    queue: deque[tuple[int, int, int]] = deque([start])
    visited.add(start)

    while queue:
        cur = queue.popleft()
        for nxt in _EDGE_GRAPH[cur]:
            if nxt in visited:
                continue
            nx, ny, nz = nxt
            if not in_bounds((nx, ny, nz)):
                continue
            visited.add(nxt)
            # FIXED: Use cache_manager.get_piece() instead of get_occupancy_safe
            if cache_manager.get_piece((nx, ny, nz)) is None:
                queue.append(nxt)

    targets = []
    for tx, ty, tz in visited:
        if not in_bounds((tx, ty, tz)):
            continue
        if (tx, ty, tz) == start:
            continue

        # FIXED: Use cache_manager.get_piece() instead of get_occupancy_safe
        victim = cache_manager.get_piece((tx, ty, tz))
        if victim is not None:
            if victim.color != color:
                targets.append((tx, ty, tz))
        else:
            targets.append((tx, ty, tz))

    if not targets:
        return []

    tarr = np.array(targets, dtype=np.int16)
    directions = tarr - np.array(start, dtype=np.int16)

    jump = get_integrated_jump_movement_generator(cache_manager)
    return jump.generate_jump_moves(
        color=color, pos=start,
        directions=directions.astype(np.int8),
        allow_capture=True,
    )

@register(PieceType.EDGEROOK)
def edgerook_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    return generate_edgerook_moves(state.cache_manager, state.color, x, y, z)

__all__ = ["generate_edgerook_moves"]
