# edgerookmoves.py
"""Edge-Rook – slides along the 9×9×9 edge graph, stops at first blocker,
armour-aware, single jump-batch."""

from __future__ import annotations

import numpy as np
from typing import List, TYPE_CHECKING, Dict
from collections import deque
from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.movement.movepiece import Move, MOVE_FLAGS
from game3d.common.coord_utils import in_bounds


# ----------------------------------------------------------
# 1.  Build once: edge adjacency graph (6 axial directions)
# ----------------------------------------------------------
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



# ----------------------------------------------------------
# 3.  Generator – BFS on edge graph, then single jump batch
# ----------------------------------------------------------
def generate_edgerook_moves(cache: OptimizedCacheManager, color: Color, x: int, y: int, z: int) -> List[Move]:
    start = (int(x), int(y), int(z))  # FIX: Force int to prevent np.int64 promotion issues
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
            # FIX: Safety check (redundant but prevents anomalous OOB from graph)
            nx, ny, nz = nxt
            if not (0 <= nx < 9 and 0 <= ny < 9 and 0 <= nz < 9):
                continue
            visited.add(nxt)
            # FIXED: Use cache_manager's occupancy with is_occupied
            if not cache.occupancy.is_occupied(nx, ny, nz):
                queue.append(nxt)

    targets = []
    for tx_, ty_, tz_ in visited:
        # FIX: Safety check (prevents OOB indexing if graph corrupted)
        if not (0 <= tx_ < 9 and 0 <= ty_ < 9 and 0 <= tz_ < 9):
            continue
        if (tx_, ty_, tz_) == start:
            continue
        # FIXED: Use is_occupied instead of mask
        if cache.occupancy.is_occupied(tx_, ty_, tz_):
            victim = cache.occupancy.get((tx_, ty_, tz_))
            if victim and victim.color != color:
                targets.append((tx_, ty_, tz_))
        else:
            targets.append((tx_, ty_, tz_))

    if not targets:
        return []

    tarr = np.array(targets, dtype=np.int16)
    directions = tarr - np.array(start, dtype=np.int16)

    # FIXED: Pass cache_manager
    jump = get_integrated_jump_movement_generator(cache)
    return jump.generate_jump_moves(
        color=color, pos=start,
        directions=directions.astype(np.int8),
        allow_capture=True,
    )
# ----------------------------------------------------------
# 4.  Dispatcher – in-file
# ----------------------------------------------------------
@register(PieceType.EDGEROOK)
def edgerook_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    # FIX: Force int to ensure consistent typing across callers (e.g., batch gen with np.int64)
    return generate_edgerook_moves(state.cache, state.color, int(x), int(y), int(z))

__all__ = ["generate_edgerook_moves"]
