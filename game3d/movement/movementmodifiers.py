"""Central place to apply **all** movement buffs / debuffs to raw sliders."""

from typing import List, Tuple
from pieces.enums import Color, PieceType
from game3d.cache.manager import get_cache_manager
from common import add_coords

def apply_movement_effects(
    state,
    start: Tuple[int, int, int],
    raw_directions: List[Tuple[int, int, int]],
    max_steps: int,
) -> Tuple[List[Tuple[int, int, int]], int]:
    mgr = get_cache_manager()

    # 1. range buffs / debuffs
    if mgr.is_movement_buffed(start, state.current):
        max_steps += 1
    # Implement/Stub: is_movement_slowed
    if hasattr(mgr, "is_movement_slowed") and mgr.is_movement_slowed(start, state.current):
        max_steps = max(1, max_steps - 1)

    # 2. direction filters (example: geomancer “no diagonals”)
    if hasattr(mgr, "is_diagonal_blocked") and mgr.is_diagonal_blocked(start, state.current):
        raw_directions = [d for d in raw_directions if 0 in d]

    # 4. geomancy blocked squares – filter directions that land on blocked square
    directions = [
        d for d in raw_directions
        if not mgr.is_geomancy_blocked(
            add_coords(start, (d[0] * max_steps, d[1] * max_steps, d[2] * max_steps)),
            getattr(state, "halfmove_clock", 0),
        )
    ]
    # 5. special wall capture rules
    victim = state.board.piece_at(add_coords(start, (0, 0, 0)))
    if victim is not None and victim.ptype == PieceType.WALL:
        if hasattr(mgr, "can_capture_wall") and not mgr.can_capture_wall(start, add_coords(start, (0, 0, 0)), state.current):
            return [], max_steps
    return directions, max_steps
