"""3D Edge-Rook (Edge-Walker) move generation — traverses entire edge network."""

from typing import List, Set
from game3d.pieces.enums import PieceType, Color
from game3d.movement.pathvalidation import is_edge_square, validate_piece_at
from game3d.movement.movepiece import Move
from collections import deque
from game3d.cache.manager import OptimizedCacheManager
# Precomputed edge adjacency graph: {coord: [neighbor_coords]}
_EDGE_GRAPH: dict[tuple, list[tuple]] = {}

def _build_edge_graph(board_size: int = 9) -> None:
    """Build the edge graph once at import time."""
    if _EDGE_GRAPH:
        return  # already built

    directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

    for x in range(board_size):
        for y in range(board_size):
            for z in range(board_size):
                if is_edge_square(x, y, z, board_size):
                    neighbors = []
                    for dx, dy, dz in directions:
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if (0 <= nx < board_size and
                            0 <= ny < board_size and
                            0 <= nz < board_size):
                            if is_edge_square(nx, ny, nz, board_size):
                                neighbors.append((nx, ny, nz))
                    _EDGE_GRAPH[(x, y, z)] = neighbors

# Build the graph immediately when module is imported
_build_edge_graph()

def generate_edgerook_moves(cache: OptimizedCacheManager, color: Color, x: int, y: int, z: int) -> List['Move']:
    """
    Generate legal Edge-Rook moves from (x, y, z) using precomputed edge graph.

    Rules:
    - Must start on an edge square.
    - Can move to any edge square reachable via unoccupied edge squares.
    - Can capture on destination (enemy), but cannot pass through occupied squares.
    - Cannot land on friendly pieces.
    """
    start = (x, y, z)

    # Validate piece at start position
    if not validate_piece_at(cache, color, start, PieceType.EDGEROOK):
        return []

    if start not in _EDGE_GRAPH:
        return []  # not on edge

    current_color = color  # ← use parameter, not state.color
    moves: List[Move] = []
    visited: set = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        current = queue.popleft()

        # Get precomputed neighbors (only edge squares)
        for neighbor in _EDGE_GRAPH.get(current, []):
            if neighbor in visited:
                continue

            piece = cache.piece_cache.get(neighbor)  # ← use board, not state.board

            if piece is None:
                # Empty square: can traverse and move here
                visited.add(neighbor)
                queue.append(neighbor)
                moves.append(Move(from_coord=start, to_coord=neighbor, is_capture=False))
            elif piece.color != current_color:
                # Enemy piece: capture allowed, but don't traverse further
                visited.add(neighbor)
                moves.append(Move(from_coord=start, to_coord=neighbor, is_capture=True))
                # Note: do NOT append to queue (can't move through enemy)
            else:
                # Friendly piece: block this square (can't land or traverse)
                visited.add(neighbor)
                # Do nothing else

    return moves
