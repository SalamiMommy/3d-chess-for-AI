"""
XZ-Zig-Zag — 9-step zig-zag rays in XZ-plane + dispatcher (consolidated).
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

def _build_xz_zigzag_vectors() -> np.ndarray:
    vecs = []
    for pri, sec in ((1, -1), (-1, 1)):
        seq = []
        curr = np.zeros(3, dtype=np.int8)
        move_primary = True
        for seg in range(3):
            step = np.zeros(3, dtype=np.int8)
            ax = 0 if move_primary else 2  # XZ-plane uses axes 0 (X) and 2 (Z)
            step[ax] = pri if move_primary else sec
            for _ in range(3):
                curr += step
                seq.append(curr.copy())
            move_primary ^= 1
        vecs.extend(seq)
    return np.array(vecs, dtype=np.int8)

XZ_ZIGZAG_DIRECTIONS = _build_xz_zigzag_vectors()

def generate_xz_zigzag_moves(
    cache_manager: 'OptimizedCacheManager',  # FIXED: Consistent parameter name
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate XZ-zig-zag moves using only the slider kernel."""
    return generate_moves(
        piece_type='xzzigzag',
        pos=(x, y, z),
        color=color,
        max_distance=16,
        directions=XZ_ZIGZAG_DIRECTIONS,
        cache_manager=cache_manager,  # FIXED: Use parameter
    )

@register(PieceType.XZZIGZAG)
def xz_zigzag_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_xz_zigzag_moves(state.cache_manager, state.color, x, y, z)  # FIXED: Use cache_manager

__all__ = ['generate_xz_zigzag_moves']
