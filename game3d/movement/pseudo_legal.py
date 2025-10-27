# pseudo_legal.py - FIXED
"""Optimized pseudo-legal move generator using cache manager with autodetection of batch vs scalar mode."""
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
from game3d.common.move_utils import prepare_batch_data, filter_none_moves
from game3d.common.debug_utils import MoveStatsTracker, GeneratorBase, track_time
from game3d.common.piece_utils import get_player_pieces
from game3d.common.validation import validate_moves

_STATS = MoveStatsTracker()

class PseudoLegalMode(Enum):
    STANDARD    = "standard"
    BATCH       = "batch"
    MEGA_BATCH  = "mega_batch"

class PseudoLegalGenerator(GeneratorBase):
    def __init__(self):
        super().__init__(PseudoLegalMode, PseudoLegalMode.STANDARD, _STATS)

    @track_time(_STATS)
    def _impl(self, state: GameState, mode: str) -> List[Move]:
        # Respect explicit mode if provided
        if mode:
            # FIX: Use proper Enum lookup instead of .get()
            try:
                mode_enum = self.mode_enum[mode.upper()]
            except KeyError:
                mode_enum = PseudoLegalMode.STANDARD

            if mode_enum == PseudoLegalMode.MEGA_BATCH:
                threshold = 0  # force mega
            elif mode_enum == PseudoLegalMode.BATCH:
                threshold = 31  # force batch (skip scalar)
            else:
                threshold = float('inf')  # force scalar
        else:
            threshold = None  # autodetect

        moves = _generate_pseudo_legal(state, threshold=threshold)
        moves = filter_none_moves(moves)
        _STATS.total_moves_generated += len(moves)
        return moves

generate_pseudo_legal_moves = PseudoLegalGenerator().generate

def generate_pseudo_legal_moves_for_piece(
    state: GameState, coord: Tuple[int, int, int]
) -> List[Move]:
    """Use cache manager for all operations"""
    cache_manager = state.cache_manager
    piece = cache_manager.occupancy.get(coord)
    if not piece or piece.color != state.color:
        return []

    if cache_manager.is_frozen(coord, piece.color):
        return []

    is_debuffed = cache_manager.is_movement_debuffed(coord, piece.color)
    pt = PieceType.PAWN if is_debuffed and piece.ptype != PieceType.PAWN else piece.ptype

    from game3d.movement.registry import get_dispatcher
    from game3d.movement.movementmodifiers import modify_raw_moves

    dispatcher = get_dispatcher(pt)
    if dispatcher is None:
        return []

    piece_moves = dispatcher(state, *coord)
    if not piece_moves:
        return []

    to_coords = np.array([m.to_coord for m in piece_moves], dtype=np.int32)
    captures = np.array([m.is_capture for m in piece_moves], dtype=bool)

    modified_moves = modify_raw_moves(
        from_coord=coord,
        to_coords=to_coords,
        captures=captures,
        color=piece.color,
        cache_manager=cache_manager,
        debuffed=is_debuffed,
        current_ply=state.ply,
    )

    return validate_moves(modified_moves, state)

def _generate_pseudo_legal(state: GameState, threshold: Optional[int] = None) -> List[Move]:
    """
    Unified pseudo-legal move generator that autodetects scalar vs batch vs mega-batch mode.

    Parameters
    ----------
    state : GameState
        Current game state.
    threshold : int or None
        If None: autodetect based on piece count.
        If 0: force mega-batch.
        If <=30: force scalar.
        If >30 and <=100: force standard batch.
        If >100: force mega-batch.
    """
    cache_manager = state.cache_manager
    color = state.color

    # Get all player pieces via cache manager
    all_coords, all_types = cache_manager.occupancy.batch_get_all_pieces_data(color)
    piece_count = len(all_coords)

    if piece_count == 0:
        return []

    # Determine mode
    if threshold is None:
        if piece_count > 100:
            mode = 'mega'
        elif piece_count > 30:
            mode = 'batch'
        else:
            mode = 'scalar'
    else:
        if threshold == 0:
            mode = 'mega'
        elif threshold <= 30:
            mode = 'scalar'
        elif threshold <= 100:
            mode = 'batch'
        else:
            mode = 'mega'

    if mode == 'scalar':
        return _run_scalar_mode(state, cache_manager, all_coords, all_types)
    elif mode == 'batch':
        return _run_standard_batch_mode(state, cache_manager, np.array(all_coords, dtype=np.int32), np.array(all_types, dtype=np.uint8))
    else:  # mega
        return _run_mega_batch_mode(state, cache_manager, np.array(all_coords, dtype=np.int32), np.array(all_types, dtype=np.uint8))

def _run_scalar_mode(state: GameState, cache_manager, all_coords: List[Tuple], all_types: List[int]) -> List[Move]:
    all_moves = []
    for coord in all_coords:
        moves = generate_pseudo_legal_moves_for_piece(state, coord)
        if moves:
            all_moves.extend(moves)
            piece = cache_manager.occupancy.get(coord)
            if piece:
                _STATS.piece_breakdown[piece.ptype] = _STATS.piece_breakdown.get(piece.ptype, 0) + len(moves)
    return all_moves

def _run_standard_batch_mode(state: GameState, cache_manager, all_coords: np.ndarray, all_types: np.ndarray) -> List[Move]:
    color = state.color

    frozen_mask = cache_manager.batch_get_frozen_status(all_coords, color)
    active_mask = ~frozen_mask
    active_coords = all_coords[active_mask]
    active_types = all_types[active_mask]

    if len(active_coords) == 0:
        return []

    batch_size = 50
    all_moves = []

    for i in range(0, len(active_coords), batch_size):
        batch_coords = active_coords[i:i+batch_size]
        batch_types = active_types[i:i+batch_size]
        batch_moves = _process_coord_batch(state, batch_coords, batch_types, color, cache_manager)
        all_moves.extend(batch_moves)

    return all_moves

def _run_mega_batch_mode(state: GameState, cache_manager, all_coords: np.ndarray, all_types: np.ndarray) -> List[Move]:
    color = state.color

    frozen_mask = cache_manager.batch_get_frozen_status(all_coords, color)
    active_coords = all_coords[~frozen_mask]
    active_types = all_types[~frozen_mask]

    if len(active_coords) == 0:
        return []

    debuffed_mask = cache_manager.batch_get_debuffed_status(active_coords, color)
    pawn_mask = (active_types == PieceType.PAWN.value)
    debuffed_types = np.where(debuffed_mask & ~pawn_mask, PieceType.PAWN.value, active_types)

    batch_size = 25
    all_moves = []

    for i in range(0, len(active_coords), batch_size):
        batch_coords = active_coords[i:i+batch_size]
        batch_types = debuffed_types[i:i+batch_size]
        batch_debuffed = debuffed_mask[i:i+batch_size]
        batch_moves = _process_mega_batch(state, batch_coords, batch_types, batch_debuffed, color, cache_manager)
        all_moves.extend(batch_moves)

    return all_moves

def _process_coord_batch(state: GameState, coords: np.ndarray, types: np.ndarray,
                        color: Color, cache_manager) -> List[Move]:
    if len(coords) == 0:
        return []

    current_ply = getattr(state, 'ply', state.halfmove_clock)
    debuffed_mask = cache_manager.batch_get_debuffed_status(coords, color)

    pawn_mask = (types == PieceType.PAWN.value)
    debuffed_types = np.where(debuffed_mask & ~pawn_mask, PieceType.PAWN.value, types)

    from game3d.movement.registry import dispatch_batch
    piece_types_list = [PieceType(t) for t in debuffed_types]
    raw_moves_batch = dispatch_batch(state, coords.tolist(), piece_types_list, color)

    all_moves = []
    for i, (coord, piece_moves) in enumerate(zip(coords, raw_moves_batch)):
        if not piece_moves:
            continue

        coord_tuple = tuple(coord)
        is_debuffed = debuffed_mask[i]

        to_coords = np.array([m.to_coord for m in piece_moves], dtype=np.int32)
        captures = np.array([m.is_capture for m in piece_moves], dtype=bool)

        if len(to_coords) == 0:
            continue

        from game3d.movement.movementmodifiers import modify_raw_moves
        modified_moves = modify_raw_moves(
            from_coord=coord_tuple,
            to_coords=to_coords,
            captures=captures,
            color=color,
            cache_manager=cache_manager,
            debuffed=is_debuffed,
            current_ply=current_ply,
        )

        validated = validate_moves(modified_moves, state)
        all_moves.extend(validated)

        piece_type = PieceType(debuffed_types[i])
        _STATS.piece_breakdown[piece_type] = _STATS.piece_breakdown.get(piece_type, 0) + len(validated)

    return all_moves

def _process_mega_batch(state: GameState, coords: np.ndarray, types: np.ndarray,
                       debuffed_mask: np.ndarray, color: Color, cache_manager) -> List[Move]:
    if len(coords) == 0:
        return []

    current_ply = getattr(state, 'ply', state.halfmove_clock)

    from game3d.movement.registry import dispatch_batch
    piece_types_list = [PieceType(t) for t in types]
    raw_moves_batch = dispatch_batch(state, coords.tolist(), piece_types_list, color)

    to_coords_batch = [np.array([m.to_coord for m in piece_moves]) for piece_moves in raw_moves_batch]
    captures_batch = [np.array([m.is_capture for m in piece_moves]) for piece_moves in raw_moves_batch]

    from game3d.movement.movementmodifiers import modify_raw_moves_batch
    modified_moves_batch = modify_raw_moves_batch(
        from_coords=coords,
        to_coords_batch=to_coords_batch,
        captures_batch=captures_batch,
        colors=np.full(len(coords), color.value),
        cache_manager=cache_manager,
        debuffed_mask=debuffed_mask,
        current_ply=current_ply,
    )

    all_modified_moves = []
    for moves in modified_moves_batch:
        all_modified_moves.extend(moves)

    return validate_moves(all_modified_moves, state)

def generate_pseudo_legal_moves_batch(state: GameState, coords: np.ndarray) -> List[List[Move]]:
    """Batch generate moves per coordinate using scalar logic (for external use)."""
    if coords.size == 0:
        return []
    results = []
    for coord in coords:
        moves = generate_pseudo_legal_moves_for_piece(state, tuple(coord))
        results.append(moves)
    return results

def get_pseudo_legal_stats() -> Dict[str, Any]:
    return _STATS.get_stats()

def reset_pseudo_legal_stats() -> None:
    _STATS.reset()
