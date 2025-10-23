# pseudo_legal.py - FIXED VERSION
"""Optimized pseudo-legal move generator with reduced redundancy."""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import numpy as np
from enum import Enum
import time

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

from game3d.common.enums import Color, PieceType
from game3d.movement.movepiece import Move
from game3d.common.move_utils import prepare_batch_data, validate_moves, filter_none_moves
from game3d.common.debug_utils import MoveStatsTracker, GeneratorBase, track_time
from game3d.common.piece_utils import get_player_pieces

_STATS = MoveStatsTracker()

class PseudoLegalMode(Enum):
    STANDARD    = "standard"
    BATCH       = "batch"

class PseudoLegalGenerator(GeneratorBase):
    def __init__(self):
        super().__init__(PseudoLegalMode, PseudoLegalMode.STANDARD, _STATS)

    @track_time(_STATS)
    def _impl(self, state: GameState, mode: str) -> List[Move]:
        mode_enum = self.mode_enum.get(mode.upper(), PseudoLegalMode.STANDARD)

        if mode_enum == PseudoLegalMode.BATCH:
            moves = _generate_pseudo_legal_batch(state)
        else:
            moves = _generate_pseudo_legal_standard(state)

        # SINGLE FILTER CALL at the end to reduce redundancy
        moves = filter_none_moves(moves)
        _STATS.total_moves_generated += len(moves)
        return moves

generate_pseudo_legal_moves = PseudoLegalGenerator().generate

def generate_pseudo_legal_moves_for_piece(
    state: GameState, coord: Tuple[int, int, int]
) -> List[Move]:
    """Single implementation without redundant filtering."""
    # STANDARDIZED: cache_manager - use occupancy_cache property
    piece = state.cache_manager.occupancy_cache.get(coord)
    if not piece or piece.color != state.color:
        return []

    # Consolidated frozen check
    if state.cache_manager.is_frozen(coord, piece.color):
        return []

    # De-buffed pieces move like pawns
    is_debuffed = state.cache_manager.is_movement_debuffed(coord, piece.color)
    pt = PieceType.PAWN if is_debuffed and piece.ptype != PieceType.PAWN else piece.ptype

    # BREAK CIRCULAR IMPORT: Import locally
    from game3d.movement.registry import get_dispatcher
    from game3d.movement.movementmodifiers import modify_raw_moves

    dispatcher = get_dispatcher(pt)
    if dispatcher is None:
        return []

    piece_moves = dispatcher(state, *coord)
    if not piece_moves:
        return []

    # Convert to arrays for batch processing
    to_coords = np.array([m.to_coord for m in piece_moves], dtype=np.int32)
    captures = np.array([m.is_capture for m in piece_moves], dtype=bool)

    # Apply modifiers in one call
    modified_moves = modify_raw_moves(
        from_coord=coord,
        to_coords=to_coords,
        captures=captures,
        color=piece.color,
        cache_manager=state.cache_manager,  # STANDARDIZED
        debuffed=is_debuffed,
        current_ply=state.ply,
    )

    # Single validation call
    return validate_moves(modified_moves, state, piece)

def _generate_pseudo_legal_batch(state: GameState) -> List[Move]:
    """Batch generation with consolidated processing."""
    # BREAK CIRCULAR IMPORT: Import locally
    from game3d.movement.registry import dispatch_batch
    from game3d.movement.movementmodifiers import modify_raw_moves

    coords, types, debuffed_indices = prepare_batch_data(state)
    raw_moves = dispatch_batch(state, coords, types, state.color)

    all_moves = []
    for i, (coord, piece_moves) in enumerate(zip(coords, raw_moves)):
        if not piece_moves:
            continue

        piece_type = types[i]
        is_debuffed = i in debuffed_indices

        to_coords = np.array([m.to_coord for m in piece_moves], dtype=np.int32)
        captures = np.array([m.is_capture for m in piece_moves], dtype=bool)

        modified_moves = modify_raw_moves(
            from_coord=coord,
            to_coords=to_coords,
            captures=captures,
            color=state.color,
            cache_manager=state.cache_manager,  # STANDARDIZED
            debuffed=is_debuffed,
            current_ply=state.ply,
        )

        validated = validate_moves(modified_moves, state)
        all_moves.extend(validated)
        _STATS.piece_breakdown[piece_type] += len(validated)

    return all_moves

def _generate_pseudo_legal_standard(state: GameState) -> List[Move]:
    """Standard generation reusing piece function to avoid duplication."""
    all_moves = []
    for coord, piece in get_player_pieces(state, state.color):
        moves = generate_pseudo_legal_moves_for_piece(state, coord)
        if moves:
            all_moves.extend(moves)
            _STATS.piece_breakdown[piece.ptype] += len(moves)
    return all_moves

def get_pseudo_legal_stats() -> Dict[str, Any]:
    return _STATS.get_stats()

def reset_pseudo_legal_stats() -> None:
    _STATS.reset()
