# movementmodifiers.py - PARALLELIZED PIPELINE VERSION
"""Centralized movement modifier application - now focused on Move objects."""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any, Set, Union
from dataclasses import dataclass
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from game3d.common.enums import Color, PieceType
from game3d.common.coord_utils import in_bounds_vectorised
from game3d.movement.movepiece import Move
from game3d.common.debug_utils import StatsTracker, measure_time_ms

@dataclass(slots=True)
class MovementEffectStats:
    total_calls: int = 0
    average_time_ms: float = 0.0
    moves_modified: int = 0
    moves_filtered: int = 0
    geomancy_blocks: int = 0
    wall_restrictions: int = 0
    debuffs_applied: int = 0

    def update_average(self, elapsed_ms: float) -> None:
        self.total_calls += 1
        self.average_time_ms = (
            (self.average_time_ms * (self.total_calls - 1) + elapsed_ms) / self.total_calls
            if self.total_calls > 0 else elapsed_ms
        )

    def get_stats(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def reset(self) -> None:
        self.total_calls = 0
        self.average_time_ms = 0.0
        self.moves_modified = 0
        self.moves_filtered = 0
        self.geomancy_blocks = 0
        self.wall_restrictions = 0
        self.debuffs_applied = 0

_STATS = MovementEffectStats()

class ParallelModifier:
    """Handles parallel application of movement modifiers"""
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_workers = max_workers

    def apply_modifiers_batch(self, state, moves_batch: List[Move]) -> List[Move]:
        """Apply modifiers to a batch of moves in parallel"""
        if len(moves_batch) == 0:
            return []

        # For small batches, use sequential processing
        if len(moves_batch) < 50:
            return [m for m in moves_batch if _is_move_modified_valid(state, m)]

        # Split into smaller chunks for parallel processing
        chunk_size = max(10, len(moves_batch) // self.max_workers)
        chunks = [moves_batch[i:i + chunk_size] for i in range(0, len(moves_batch), chunk_size)]

        valid_moves = []
        futures = []

        for chunk in chunks:
            future = self.executor.submit(self._process_chunk, state, chunk)
            futures.append(future)

        for future in as_completed(futures):
            try:
                valid_moves.extend(future.result())
            except Exception as e:
                print(f"Error in parallel modifier: {e}")
                # Fallback to sequential
                valid_moves.extend([m for m in chunk if _is_move_modified_valid(state, m)])

        return valid_moves

    def _process_chunk(self, state, chunk: List[Move]) -> List[Move]:
        """Process a chunk of moves sequentially"""
        return [m for m in chunk if _is_move_modified_valid(state, m)]

    def __del__(self):
        self.executor.shutdown(wait=False)

# Global parallel modifier instance
_parallel_modifier = ParallelModifier()

def apply_movement_effects_to_moves(state, moves: List[Move]) -> List[Move]:
    """
    MAIN ENTRY POINT: Apply all movement effects to a list of moves.
    This is the central function that sits between pseudo-legal generation and validation.
    """
    with measure_time_ms() as elapsed_ms:
        _STATS.total_calls += 1

        if not moves:
            return []

        cache_manager = state.cache_manager
        current_ply = getattr(state, 'ply', state.halfmove_clock)

        # Use parallel processing for large batches
        if len(moves) > 100:
            valid_moves = _parallel_modifier.apply_modifiers_batch(state, moves)
        else:
            valid_moves = [m for m in moves if _is_move_modified_valid(state, m)]

        _STATS.moves_filtered += (len(moves) - len(valid_moves))
        _STATS.moves_modified += len(valid_moves)
        _STATS.update_average(elapsed_ms())

        return valid_moves

def _is_move_modified_valid(state, move: Move) -> bool:
    """
    Core logic: Check if a move passes all movement modifiers.
    Returns True if the move should be kept, False if filtered out.
    """
    cache_manager = state.cache_manager
    current_ply = getattr(state, 'ply', state.halfmove_clock)
    color = state.color

    # 1. Check if piece is frozen
    if cache_manager.is_frozen(move.from_coord, color):
        _STATS.moves_filtered += 1
        return False

    # 2. Check geomancy blocking
    if cache_manager.is_geomancy_blocked(move.to_coord, current_ply):
        _STATS.geomancy_blocks += 1
        return False

    # 3. Check wall capture restrictions
    piece = cache_manager.occupancy.get(move.from_coord)
    if piece and piece.ptype == PieceType.WALL and move.is_capture:
        from game3d.pieces.pieces.wall import can_capture_wall
        if not can_capture_wall(move.to_coord, move.from_coord):
            _STATS.wall_restrictions += 1
            return False

    # 4. Apply debuff effects (range reduction)
    if cache_manager.is_movement_debuffed(move.from_coord, color):
        _STATS.debuffs_applied += 1
        # Note: Debuff application to Move objects would require Move modification
        # For now, we assume pseudo-legal generation already accounts for debuffs
        pass

    return True

def apply_movement_effects_to_moves_batch(state, moves_batch: List[List[Move]]) -> List[List[Move]]:
    """
    Batch version for processing multiple move lists in parallel.
    Used by mega-batch pseudo-legal generator.
    """
    if not moves_batch:
        return []

    # Process each piece's moves in parallel
    results = []
    futures = []

    for moves in moves_batch:
        future = _parallel_modifier.executor.submit(apply_movement_effects_to_moves, state, moves)
        futures.append(future)

    for future in as_completed(futures):
        try:
            results.append(future.result())
        except Exception as e:
            print(f"Error in batch modifier: {e}")
            results.append([])

    return results

# ================================
# BACKWARD COMPATIBILITY
# ================================

def modify_raw_moves(
    from_coord: Tuple[int, int, int],
    to_coords: np.ndarray,
    captures: np.ndarray,
    color: Color,
    cache_manager,
    debuffed: bool = False,
    *,
    current_ply: int,
) -> List[Move]:
    """Legacy function - converts to Move objects and uses new pipeline"""
    if len(to_coords) == 0:
        return []

    # Convert to Move objects
    moves = Move.create_batch(from_coord, to_coords, captures, debuffed=debuffed)

    # Create mock state for compatibility
    class MockState:
        def __init__(self):
            self.cache_manager = cache_manager
            self.color = color
            self.ply = current_ply

    mock_state = MockState()

    # Apply modifiers using new pipeline
    return apply_movement_effects_to_moves(mock_state, moves)

def modify_raw_moves_batch(
    from_coords: np.ndarray,
    to_coords_batch: List[np.ndarray],
    captures_batch: List[np.ndarray],
    colors: np.ndarray,
    cache_manager,
    debuffed_mask: np.ndarray,
    current_ply: int,
) -> List[List[Move]]:
    """Legacy batch function - converts to new pipeline"""
    all_moves = []

    for i in range(len(from_coords)):
        from_coord = tuple(from_coords[i])
        to_coords = to_coords_batch[i] if i < len(to_coords_batch) else np.empty((0, 3))
        captures = captures_batch[i] if i < len(captures_batch) else np.empty(0, dtype=bool)
        color = Color(colors[i]) if i < len(colors) else colors[0]
        debuffed = debuffed_mask[i] if i < len(debuffed_mask) else False

        if len(to_coords) > 0:
            piece_moves = modify_raw_moves(
                from_coord, to_coords, captures, color, cache_manager,
                debuffed, current_ply=current_ply
            )
            all_moves.append(piece_moves)
        else:
            all_moves.append([])

    return all_moves

def get_movement_modifier_stats() -> Dict[str, Any]:
    return _STATS.get_stats()

def reset_movement_modifier_stats() -> None:
    _STATS.reset()
