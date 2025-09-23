"""Central place to apply **all** movement buffs / debuffs to raw sliders."""

from typing import List, Tuple
from pieces.enums import Color
from game3d.cache.manager import get_cache_manager


def apply_movement_effects(
    state,
    start: Tuple[int, int, int],
    raw_directions: List[Tuple[int, int, int]],
    max_steps: int,
) -> Tuple[List[Tuple[int, int, int]], int]:
    """
    Return (possibly altered directions, new max_steps) after all auras.
    Directions stay the same; only **distance** and **eligibility** change.
    """
    mgr = get_cache_manager()

    # 1. range buffs / debuffs
    if mgr.is_movement_buffed(start, state.current):
        max_steps += 1
    if mgr.is_movement_slowed(start, state.current):
        max_steps = max(1, max_steps - 1)

    # 2. direction filters (example: geomancer “no diagonals”)
    if mgr.is_diagonal_blocked(start, state.current):
        raw_directions = [d for d in raw_directions if 0 in d]  # keep axial only

    # 3. future: poison → max_steps = 1, wall → no slides, etc.

    return raw_directions, max_steps
