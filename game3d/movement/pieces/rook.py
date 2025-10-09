"""
Rook + Trailblazer — orthogonal slider rays (consolidated).
Exports:
  generate_rook_moves(cache, color, x, y, z, max_steps=8) -> list[Move]
  (decorated) rook_dispatcher(state, x, y, z) -> list[Move]
  (decorated) trailblazer_dispatcher(state, x, y, z) -> list[Move]  # max 3
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING
import numpy as np

from game3d.pieces.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.slidermovement import get_slider_generator
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager as CacheManager

# 6 orthogonal directions (±X, ±Y, ±Z)
_ROOK_DIRS = np.array([
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
], dtype=np.int8)

def generate_rook_moves(cache: CacheManager,
                        color: Color,
                        x: int, y: int, z: int,
                        max_steps: int = 8) -> List[Move]:
    """Orthogonal slider; steps capped by max_steps."""
    return get_slider_generator().generate_moves(
        piece_type='rook',
        pos=(x, y, z),
        color=color.value,
        max_distance=max_steps,
        cache_manager=cache          # ← REQUIRED keyword-only argument
    )

# ----------------------------------------------------------
# Dispatchers (both in same file)
# ----------------------------------------------------------
@register(PieceType.ROOK)
def rook_move_dispatcher(state, x: int, y: int, z: int) -> List[Move]:
    return generate_rook_moves(state.cache, state.color, x, y, z)

@register(PieceType.TRAILBLAZER)
def trailblazer_move_dispatcher(state, x: int, y: int, z: int) -> List[Move]:
    return generate_rook_moves(state.cache, state.color, x, y, z, max_steps=3)

__all__ = ['generate_rook_moves']
