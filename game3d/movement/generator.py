# generator.py - PARALLELIZED PIPELINE VERSION
from __future__ import annotations
from typing import Callable, List, Dict, Any, Optional, Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    PARALLEL = "parallel"
    MEGA_BATCH = "mega_batch"

class LegalMoveGenerator(GeneratorBase):
    def __init__(self):
        super().__init__(MoveGenMode, MoveGenMode.STANDARD, _STATS)
        self.parallel_executor = ThreadPoolExecutor(max_workers=4)

    @track_time(_STATS)
    def _impl(self, state: GameState, mode: str) -> List[Move]:
        # Determine processing mode
        if mode:
            try:
                mode_enum = self.mode_enum[mode.upper()]
            except KeyError:
                mode_enum = MoveGenMode.STANDARD
        else:
            mode_enum = MoveGenMode.STANDARD

        # Generate pseudo-legal moves (raw moves without modifiers)
        from game3d.movement.pseudo_legal import generate_pseudo_legal_moves
        raw_moves = generate_pseudo_legal_moves(state, mode=mode_enum.value)

        # Apply movement modifiers in parallel for large batches
        if mode_enum == MoveGenMode.PARALLEL and len(raw_moves) > 50:
            modified_moves = self._apply_modifiers_parallel(state, raw_moves)
        else:
            modified_moves = self._apply_modifiers_sequential(state, raw_moves)

        # FINAL STEP: Validate modified moves (check, pins, etc.)
        from game3d.common.validation import filter_legal_moves
        legal_moves = filter_legal_moves(modified_moves, state)

        _STATS.total_moves_filtered += len(legal_moves)
        return legal_moves

    def _apply_modifiers_sequential(self, state: GameState, raw_moves: List[Move]) -> List[Move]:
        """Apply movement modifiers sequentially"""
        from game3d.movement.movementmodifiers import apply_movement_effects_to_moves
        return apply_movement_effects_to_moves(state, raw_moves)

    def _apply_modifiers_parallel(self, state: GameState, raw_moves: List[Move]) -> List[Move]:
        """Apply movement modifiers in parallel batches"""
        batch_size = min(100, len(raw_moves) // 4 + 1)
        batches = [raw_moves[i:i + batch_size] for i in range(0, len(raw_moves), batch_size)]

        modified_batches = []
        futures = []

        for batch in batches:
            future = self.parallel_executor.submit(self._apply_modifiers_sequential, state, batch)
            futures.append(future)

        for future in as_completed(futures):
            try:
                modified_batches.extend(future.result())
            except Exception as e:
                print(f"Error in parallel modifier application: {e}")
                # Fallback to sequential for failed batches
                modified_batches.extend(self._apply_modifiers_sequential(state, batch))

        return modified_batches

    def __del__(self):
        if hasattr(self, 'parallel_executor'):
            self.parallel_executor.shutdown(wait=False)

generate_legal_moves = LegalMoveGenerator().generate

def _generate_legal_moves(state: GameState, threshold: Optional[int] = None) -> List[Move]:
    """
    Unified legal move generator following the pipeline:
    PSEUDO-LEGAL → MOVEMENT MODIFIERS → VALIDATION
    """
    # Get all pieces for mode decision
    cache_manager = state.cache_manager
    color = state.color

    all_coords, _ = cache_manager.occupancy.batch_get_all_pieces_data(color)
    piece_count = len(all_coords)

    if piece_count == 0:
        return []

    # Determine processing mode
    if threshold is None:
        if piece_count > 100:
            mode = "mega_batch"
        elif piece_count > 30:
            mode = "parallel"
        else:
            mode = "standard"
    else:
        if threshold == 0:
            mode = "mega_batch"
        elif threshold <= 30:
            mode = "standard"
        else:
            mode = "parallel"

    return generate_legal_moves(state, mode=mode)

def generate_legal_moves_for_piece(state: GameState, coord: Tuple[int, int, int]) -> List[Move]:
    """Single-piece legal moves following the same pipeline"""
    # Pseudo-legal generation
    from game3d.movement.pseudo_legal import generate_pseudo_legal_moves_for_piece
    raw_moves = generate_pseudo_legal_moves_for_piece(state, coord)

    # Movement modifiers
    from game3d.movement.movementmodifiers import apply_movement_effects_to_moves
    modified_moves = apply_movement_effects_to_moves(state, raw_moves)

    # Final validation
    from game3d.common.validation import filter_legal_moves
    return filter_legal_moves(modified_moves, state)

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
