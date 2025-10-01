"""3-D Edge-Rook (Edge-Walker) move generation — traverses the edge network
but uses the integrated *jump* engine for final square legality."""
from __future__ import annotations

from typing import List, Dict, Tuple
from collections import deque
import numpy as np

from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move
from game3d.cache.manager import OptimizedCacheManager
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator

# ------------------------------------------------------------------
#  Pre-computed edge adjacency graph  (unchanged)
# ------------------------------------------------------------------
_EDGE_GRAPH: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]] = {}


def _build_edge_graph(board_size: int = 9) -> None:
    if _EDGE_GRAPH:                       # already built
        return
    axial = [(1, 0, 0), (-1, 0, 0),
             (0, 1, 0), (0, -1, 0),
             (0, 0, 1), (0, 0, -1)]
    for x in range(board_size):
        for y in range(board_size):
            for z in range(board_size):
                if not is_edge_square(x, y, z, board_size):
                    continue
                neighbors = []
                for dx, dy, dz in axial:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if (0 <= nx < board_size and
                        0 <= ny < board_size and
                        0 <= nz < board_size and
                            is_edge_square(nx, ny, nz, board_size)):
                        neighbors.append((nx, ny, nz))
                _EDGE_GRAPH[(x, y, z)] = neighbors


def is_edge_square(x: int, y: int, z: int, size: int) -> bool:
    """True if the square lies on the outermost layer of the cube."""
    return x == 0 or x == size - 1 or y == 0 or y == size - 1 or z == 0 or z == size - 1


_build_edge_graph()


# ------------------------------------------------------------------
#  Move generator
# ------------------------------------------------------------------
def generate_edgerook_moves(
    cache: OptimizedCacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """
    Generate legal Edge-Rook moves.

    1.  BFS walks the edge graph until it hits an occupied square.
    2.  Every square that was *reached* (empty or enemy) is collected.
    3.  The integrated jump generator is asked: “may I land here?”
        – friendly piece  → discarded
        – enemy king w/ priests → discarded
        – wall            → discarded
    4.  We build the final Move list from the surviving destinations.
    """
    start = (x, y, z)

    if start not in _EDGE_GRAPH:          # not on edge
        return []

    # 1. Discover every edge square reachable along empty edge squares
    visited: set[Tuple[int, int, int]] = set()
    queue: deque[Tuple[int, int, int]] = deque([start])
    visited.add(start)

    while queue:
        cur = queue.popleft()
        for nxt in _EDGE_GRAPH[cur]:
            if nxt in visited:
                continue
            # Check if the next square is empty using PieceCache.get()
            if cache.piece_cache.get(nxt) is None:
                # Empty → can continue traversing
                visited.add(nxt)
                queue.append(nxt)
            else:
                # Occupied → stop traversal, but still include as destination
                visited.add(nxt)

    # 2. Ask the jump generator to filter the destinations
    jump_gen = get_integrated_jump_movement_generator(cache)

    # Build a *single* direction array: every reachable square as a
    # "jump vector" from start. (The kernel will test each one.)
    destinations = np.array(list(visited), dtype=np.int8)
    directions = destinations - np.array(start, dtype=np.int8)

    legal_moves = jump_gen.generate_jump_moves(
        color=color,
        position=start,
        directions=directions,
        allow_capture=True,
    )
    return legal_moves
