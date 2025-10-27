"""
YZ-Zig-Zag — 9-step zig-zag rays in YZ-plane + dispatcher (consolidated).
"""
from __future__ import annotations
from typing import List, TYPE_CHECKING
import numpy as np

from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.slidermovement import generate_moves
from game3d.movement.movepiece import Move, Move

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

def _build_yz_zigzag_vectors() -> np.ndarray:
    vecs = []
    for pri, sec in ((1, -1), (-1, 1)):
        seq = []
        curr = np.zeros(3, dtype=np.int8)
        move_primary = True
        for seg in range(3):
            step = np.zeros(3, dtype=np.int8)
            ax = 1 if move_primary else 2  # YZ-plane uses axes 1 (Y) and 2 (Z)
            step[ax] = pri if move_primary else sec
            for _ in range(3):
                curr += step
                seq.append(curr.copy())
            move_primary ^= 1
        vecs.extend(seq)
    return np.array(vecs, dtype=np.int8)

YZ_ZIGZAG_DIRECTIONS = _build_yz_zigzag_vectors()

def generate_yz_zigzag_moves(
    cache_manager: 'OptimizedCacheManager',
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate YZ-zig-zag moves using only the slider kernel."""
    return generate_moves(
        piece_type='yzzigzag',
        pos=(x, y, z),
        color=color,
        max_distance=16,
        directions=YZ_ZIGZAG_DIRECTIONS,
        cache_manager=cache_manager,
    )

@register(PieceType.YZZIGZAG)
def yz_zigzag_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_yz_zigzag_moves(state.cache_manager, state.color, x, y, z)

__all__ = ['generate_yz_zigzag_moves']
