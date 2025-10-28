# registry.py - ENHANCED BATCH VERSION WITH CACHED LOOKUP
from __future__ import annotations
from typing import Callable, List, Dict, TYPE_CHECKING, Tuple
from game3d.common.enums import PieceType, Color
from game3d.movement.movepiece import Move
import numpy as np
from collections import defaultdict

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

_REGISTRY: Dict[PieceType, Callable[["GameState", int, int, int], List]] = {}

def register(pt: PieceType):
    def _decorator(fn):
        if pt in _REGISTRY:
            raise ValueError(f"Dispatcher for {pt} already registered.")
        _REGISTRY[pt] = fn
        return fn
    return _decorator

# OPTIMIZED: Cached lookup for faster repeated access (singleton-like)
_dispatcher_cache: Dict[PieceType, Callable] = {}  # Thread-local if needed

def get_dispatcher(pt: PieceType) -> Callable:
    if pt not in _dispatcher_cache:
        try:
            _dispatcher_cache[pt] = _REGISTRY[pt]
        except KeyError:
            raise ValueError(f"No dispatcher registered for {pt}.") from None
    return _dispatcher_cache[pt]

def dispatch_batch_enhanced(
    state: "GameState",
    piece_coords: List[Tuple[int, int, int]],
    piece_types: List[PieceType],
    color: Color,
) -> List[List[Move]]:
    """Enhanced batch dispatch with better memory management."""
    if not piece_coords:
        return []

    # Group by piece type for more efficient processing
    type_groups: Dict[PieceType, List[Tuple[int, Tuple[int, int, int]]]] = defaultdict(list)
    for idx, (coord, ptype) in enumerate(zip(piece_coords, piece_types)):
        type_groups[ptype].append((idx, coord))

    # Process each type group separately
    all_moves = [[] for _ in range(len(piece_coords))]

    for ptype, group in type_groups.items():
        # OPTIMIZED: Use cached dispatcher
        dispatcher = get_dispatcher(ptype)

        orig_indices = [i for i, _ in group]
        coords = [c for _, c in group]

        # Process coordinates of this type
        group_moves = [dispatcher(state, *c) for c in coords]
        for j, moves in enumerate(group_moves):
            all_moves[orig_indices[j]] = moves

    return all_moves

# Update the main function to use enhanced version
dispatch_batch = dispatch_batch_enhanced

__all__ = ["register", "get_dispatcher", "dispatch_batch"]
