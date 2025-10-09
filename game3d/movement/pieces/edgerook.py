# edgerookmoves.py
"""Edge-Rook – slides along the 9×9×9 edge graph, stops at first blocker,
armour-aware, single jump-batch."""

from __future__ import annotations

import numpy as np
from typing import List, TYPE_CHECKING, Dict
from collections import deque
from game3d.pieces.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.movement.movepiece import Move, MOVE_FLAGS
from game3d.common.common import in_bounds

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager

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
    start = (x, y, z)
    if start not in _EDGE_GRAPH:          # not on edge
        return []

    # 3a. BFS to discover every reachable edge square
    visited: set[tuple[int, int, int]] = set()
    queue: deque[tuple[int, int, int]] = deque([start])
    visited.add(start)

    while queue:
        cur = queue.popleft()
        for nxt in _EDGE_GRAPH[cur]:
            if nxt in visited:
                continue
            # stop traversal at first blocker, but still include as destination
            visited.add(nxt)
            if cache.piece_cache.get(nxt) is None:        # empty → continue
                queue.append(nxt)

    # 3b. Build direction array (empty OR enemy & not armoured)
    targets = []
    for tx, ty, tz in visited:
        if (tx, ty, tz) == start:
            continue
        if cache.occupancy.mask[tz, ty, tx]:              # occupied
            victim = cache.piece_cache.get((tx, ty, tz))
            if victim and victim.color != color:
                targets.append((tx, ty, tz))
        else:                                             # empty
            targets.append((tx, ty, tz))

    if not targets:
        return []

    # 3c. Single vectorised batch to jump engine
    tarr = np.array(targets, dtype=np.int16)
    directions = tarr - np.array(start, dtype=np.int16)

    jump = get_integrated_jump_movement_generator(cache)
    return jump.generate_jump_moves(
        color=color,
        pos=start,
        directions=directions.astype(np.int8),
        allow_capture=True,
    )

# ----------------------------------------------------------
# 4.  Dispatcher – in-file
# ----------------------------------------------------------
@register(PieceType.EDGEROOK)
def edgerook_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    return generate_edgerook_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_edgerook_moves"]
