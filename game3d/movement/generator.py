# generator.py - UPDATED
from __future__ import annotations
from typing import Callable, List, Dict, Any, Optional, Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict

import numpy as np

from game3d.common.enums import PieceType, Color
from game3d.common.piece_utils import get_player_pieces
from game3d.common.move_utils import prepare_batch_data, filter_none_moves
from game3d.common.debug_utils import fallback_mode, track_time, MoveStatsTracker, GeneratorBase
from game3d.common.validation import validate_moves

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import Coord, in_bounds, SIZE

_STATS = MoveStatsTracker()

class MoveGenMode(Enum):
    STANDARD = "standard"
    BATCH = "batch"
    PARALLEL = "parallel"

class LegalMoveGenerator(GeneratorBase):
    def __init__(self):
        super().__init__(MoveGenMode, MoveGenMode.STANDARD, _STATS)

    @track_time(_STATS)
    def _impl(self, state: GameState, mode: str) -> List[Move]:
        # Determine whether to autodetect or force a mode
        if mode:
            try:
                mode_enum = self.mode_enum[mode.upper()]
            except KeyError:
                mode_enum = MoveGenMode.STANDARD
            # Map mode to threshold:
            # - STANDARD: autodetect (None)
            # - BATCH/PARALLEL: force batch (threshold = 31)
            if mode_enum == MoveGenMode.STANDARD:
                threshold = None
            else:
                threshold = 31  # forces batch or higher
        else:
            threshold = None  # autodetect

        moves = _generate_legal_moves(state, threshold=threshold)
        moves = filter_none_moves(moves)
        _STATS.total_moves_filtered += len(moves)
        return moves

generate_legal_moves = LegalMoveGenerator().generate

def _generate_legal_moves(state: GameState, threshold: Optional[int] = None) -> List[Move]:
    """
    Unified legal move generator with autodetection of scalar vs batch mode.

    Parameters
    ----------
    state : GameState
        Current game state.
    threshold : int or None
        If None: autodetect based on piece count.
        If <=30: use scalar pseudo-legal + validate.
        If >30: use batch pseudo-legal + validate.
    """
    cache_manager = state.cache_manager
    color = state.color

    # Early exit if no batch support
    if not hasattr(cache_manager.occupancy, 'batch_get_all_pieces_data'):
        # Fallback to pseudo-legal + validate without batching
        from game3d.movement.pseudo_legal import generate_pseudo_legal_moves
        from game3d.common.validation import filter_legal_moves  # UPDATED: Use new validation module
        pseudo_moves = generate_pseudo_legal_moves(state)
        return filter_legal_moves(pseudo_moves, state)

    # Get all pieces
    all_coords, _ = cache_manager.occupancy.batch_get_all_pieces_data(color)
    piece_count = len(all_coords)

    if piece_count == 0:
        return []

    # Determine active (non-frozen) pieces
    frozen_mask = cache_manager.batch_get_frozen_status(all_coords, color)
    active_count = int(np.sum(~frozen_mask))

    # Decide mode
    if threshold is None:
        use_batch = active_count > 30
    else:
        use_batch = active_count >= threshold

    # Generate pseudo-legal moves (scalar or batch handled internally by pseudo-legal module)
    from game3d.movement.pseudo_legal import generate_pseudo_legal_moves
    from game3d.common.validation import filter_legal_moves  # UPDATED: Use new validation module

    if use_batch:
        # Force batch mode in pseudo-legal generator
        pseudo_moves = generate_pseudo_legal_moves(state, mode="batch")
    else:
        # Use scalar mode
        pseudo_moves = generate_pseudo_legal_moves(state, mode="standard")

    return filter_legal_moves(pseudo_moves, state)

def generate_legal_moves_for_piece(state: GameState, coord: Tuple[int, int, int]) -> List[Move]:
    """Use cache manager for piece-specific moves"""
    cache_manager = state.cache_manager
    piece = cache_manager.occupancy.get(coord)
    if not piece or piece.color != state.color:
        return []

    if cache_manager.is_frozen(coord, state.color):
        return []

    from game3d.movement.pseudo_legal import generate_pseudo_legal_moves_for_piece
    from game3d.common.validation import filter_legal_moves  # UPDATED: Use new validation module

    pseudo_moves = generate_pseudo_legal_moves_for_piece(state, coord)
    return filter_legal_moves(pseudo_moves, state)

def generate_legal_captures(state: GameState) -> List[Move]:
    """Reuse main generator to avoid duplication."""
    all_legal = generate_legal_moves(state)
    return [mv for mv in all_legal if mv.is_capture]

def generate_legal_non_captures(state: GameState) -> List[Move]:
    """Reuse main generator to avoid duplication."""
    all_legal = generate_legal_moves(state)
    return [mv for mv in all_legal if not mv.is_capture]

def get_move_generation_stats() -> Dict[str, Any]:
    return _STATS.get_stats()

def reset_move_gen_stats() -> None:
    _STATS.reset()
