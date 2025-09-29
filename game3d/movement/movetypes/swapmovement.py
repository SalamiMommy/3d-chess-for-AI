"""Swap-Move — instantly swap positions with any friendly piece on the board.
Pure movement logic — no GameState.
"""
from typing import List
from game3d.pieces.enums import Color
from game3d.movement.movepiece import Move
from game3d.cache.manager import OptimizedCacheManager

def generate_swapper_moves(
    cache: OptimizedCacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """OPTIMIZED: Pre-filter friendly pieces, avoid redundant checks."""
    self_pos = (x, y, z)
    self_piece = cache.piece_cache.get(self_pos)
    if self_piece is None:
        return []

    moves = []
    if color == Color.WHITE:
        friendly_pieces = cache.piece_cache._white_pieces
    else:
        friendly_pieces = cache.piece_cache._black_pieces

    for target_pos, ptype in friendly_pieces.items():
        if target_pos != self_pos:
            moves.append(Move(from_coord=self_pos, to_coord=target_pos, is_capture=False))

    return moves
