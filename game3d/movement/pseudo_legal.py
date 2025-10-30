# pseudo_legal.py - FIXED COORDINATE TYPE ISSUE
"""Optimized pseudo-legal move generator - RAW MOVES ONLY, no modifiers or validation."""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable, List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import numpy as np
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

from game3d.common.enums import Color, PieceType
from game3d.movement.movepiece import Move
from game3d.common.move_utils import prepare_batch_data, filter_none_moves
from game3d.common.debug_utils import MoveStatsTracker, GeneratorBase, track_time
from game3d.common.piece_utils import get_player_pieces

_STATS = MoveStatsTracker()

class PseudoLegalMode(Enum):
    STANDARD    = "standard"
    PARALLEL    = "parallel"
    MEGA_BATCH  = "mega_batch"

class PseudoLegalGenerator(GeneratorBase):
    def __init__(self):
        super().__init__(PseudoLegalMode, PseudoLegalMode.STANDARD, _STATS)
        self.parallel_executor = ThreadPoolExecutor(max_workers=4)

    @track_time(_STATS)
    def _impl(self, state: GameState, mode: str) -> List[Move]:
        # Respect explicit mode if provided
        if mode:
            try:
                mode_enum = self.mode_enum[mode.upper()]
            except KeyError:
                mode_enum = PseudoLegalMode.STANDARD
        else:
            mode_enum = PseudoLegalMode.STANDARD

        # Generate RAW pseudo-legal moves (no modifiers, no validation)
        raw_moves = _generate_pseudo_legal(state, mode=mode_enum.value)
        _STATS.total_moves_generated += len(raw_moves)
        return raw_moves

    def _generate_parallel(self, state: GameState, coords: List[Tuple[int, int, int]]) -> List[Move]:
        """Generate moves for multiple pieces in parallel"""
        if not coords:
            return []

        batch_size = min(25, len(coords) // 4 + 1)
        coord_batches = [coords[i:i + batch_size] for i in range(0, len(coords), batch_size)]

        all_moves = []
        futures = []

        for batch in coord_batches:
            future = self.parallel_executor.submit(self._process_coord_batch_sequential, state, batch)
            futures.append(future)

        for future in as_completed(futures):
            try:
                all_moves.extend(future.result())
            except Exception as e:
                print(f"Error in parallel pseudo-legal: {e}")
                # Fallback to sequential
                all_moves.extend(self._process_coord_batch_sequential(state, batch))

        return all_moves

    def _process_coord_batch_sequential(self, state: GameState, coords: List[Tuple[int, int, int]]) -> List[Move]:
        """Process a batch of coordinates sequentially"""
        moves = []
        for coord in coords:
            # FIX: Ensure coord is a tuple, not numpy array
            if isinstance(coord, np.ndarray):
                coord_tuple = tuple(coord.tolist())
            else:
                coord_tuple = coord
            piece_moves = generate_pseudo_legal_moves_for_piece(state, coord_tuple)
            moves.extend(piece_moves)
        return moves

    def __del__(self):
        if hasattr(self, 'parallel_executor'):
            self.parallel_executor.shutdown(wait=False)

generate_pseudo_legal_moves = PseudoLegalGenerator().generate

def generate_pseudo_legal_moves_for_piece(
    state: GameState, coord: Tuple[int, int, int]
) -> List[Move]:
    """Generate RAW pseudo-legal moves for a single piece - NO MODIFIERS, NO VALIDATION"""
    cache_manager = state.cache_manager

    # FIX: Ensure coord is always a tuple for dictionary lookups
    if isinstance(coord, np.ndarray):
        coord_tuple = tuple(coord.tolist())
    else:
        coord_tuple = coord

    piece = cache_manager.occupancy.get(coord_tuple)
    if not piece or piece.color != state.color:
        return []

    # RAW GENERATION: Don't check frozen status here - let modifiers handle it
    # RAW GENERATION: Don't apply debuffs here - let modifiers handle it

    from game3d.movement.registry import get_dispatcher
    dispatcher = get_dispatcher(piece.ptype)
    if dispatcher is None:
        return []

    # Get RAW moves from piece dispatcher
    raw_moves = dispatcher(state, *coord_tuple)
    if not raw_moves:
        return []

    # Convert to proper Move objects but don't apply any game effects
    to_coords = np.array([m.to_coord for m in raw_moves], dtype=np.int32)
    captures = np.array([m.is_capture for m in raw_moves], dtype=bool)

    # Create Move objects without any modification
    moves = Move.create_batch(coord_tuple, to_coords, captures, debuffed=False)

    # Update statistics
    piece_type = piece.ptype
    _STATS.piece_breakdown[piece_type] = _STATS.piece_breakdown.get(piece_type, 0) + len(moves)

    return moves

def _generate_pseudo_legal(state: GameState, mode: str = "standard") -> List[Move]:
    """
    Unified pseudo-legal move generator that returns RAW moves only.
    No movement modifiers applied, no validation performed.
    """
    cache_manager = state.cache_manager
    color = state.color

    # Get all player pieces
    all_coords, all_types = cache_manager.occupancy.batch_get_all_pieces_data(color)
    piece_count = len(all_coords)

    if piece_count == 0:
        return []

    # FIX: Convert numpy arrays to tuples for dictionary compatibility
    if isinstance(all_coords, np.ndarray):
        coord_tuples = [tuple(coord.tolist()) for coord in all_coords]
    else:
        coord_tuples = all_coords

    if mode == "parallel":
        return _run_parallel_mode(state, coord_tuples)
    elif mode == "mega_batch":
        return _run_mega_batch_mode(state, np.array(coord_tuples, dtype=np.int32), np.array(all_types, dtype=np.uint8))
    else:  # standard
        return _run_standard_mode(state, coord_tuples)

def _run_standard_mode(state: GameState, all_coords: List[Tuple[int, int, int]]) -> List[Move]:
    """Standard sequential processing"""
    all_moves = []
    for coord in all_coords:
        # FIX: coord is already a tuple here
        moves = generate_pseudo_legal_moves_for_piece(state, coord)
        all_moves.extend(moves)
    return all_moves

def _run_parallel_mode(state: GameState, all_coords: List[Tuple[int, int, int]]) -> List[Move]:
    """Parallel processing using thread pool"""
    generator = PseudoLegalGenerator()
    return generator._generate_parallel(state, all_coords)

def _run_mega_batch_mode(state: GameState, all_coords: np.ndarray, all_types: np.ndarray) -> List[Move]:
    """Mega-batch processing for very large piece counts"""
    color = state.color
    cache_manager = state.cache_manager

    # Get active pieces (not frozen)
    frozen_mask = cache_manager.batch_get_frozen_status(all_coords, color)
    active_coords = all_coords[~frozen_mask]
    active_types = all_types[~frozen_mask]

    if len(active_coords) == 0:
        return []

    # Process in large batches
    batch_size = 100
    all_moves = []

    for i in range(0, len(active_coords), batch_size):
        batch_coords = active_coords[i:i+batch_size]
        batch_types = active_types[i:i+batch_size]
        batch_moves = _process_mega_batch(state, batch_coords, batch_types, color, cache_manager)
        all_moves.extend(batch_moves)

    return all_moves

def _process_mega_batch(state: GameState, coords: np.ndarray, types: np.ndarray,
                       color: Color, cache_manager) -> List[Move]:
    """Process a mega-batch of pieces"""
    if len(coords) == 0:
        return []

    from game3d.movement.registry import dispatch_batch
    piece_types_list = [PieceType(t) for t in types]

    # FIX: Convert numpy coordinates to tuples for dispatcher
    coord_tuples = [tuple(coord.tolist()) for coord in coords]

    # Get RAW moves from batch dispatcher
    raw_moves_batch = dispatch_batch(state, coord_tuples, piece_types_list, color)

    all_moves = []
    for i, (coord, piece_moves) in enumerate(zip(coords, raw_moves_batch)):
        if not piece_moves:
            continue

        # FIX: Ensure coord is a tuple
        if isinstance(coord, np.ndarray):
            coord_tuple = tuple(coord.tolist())
        else:
            coord_tuple = coord

        to_coords = np.array([m.to_coord for m in piece_moves], dtype=np.int32)
        captures = np.array([m.is_capture for m in piece_moves], dtype=bool)

        if len(to_coords) == 0:
            continue

        # Create Move objects without modification
        moves = Move.create_batch(coord_tuple, to_coords, captures, debuffed=False)
        all_moves.extend(moves)

        piece_type = PieceType(types[i])
        _STATS.piece_breakdown[piece_type] = _STATS.piece_breakdown.get(piece_type, 0) + len(moves)

    return all_moves

def get_pseudo_legal_stats() -> Dict[str, Any]:
    return _STATS.get_stats()

def reset_pseudo_legal_stats() -> None:
    _STATS.reset()
