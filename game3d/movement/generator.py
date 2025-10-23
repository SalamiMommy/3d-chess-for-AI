# generator.py
from __future__ import annotations
from typing import Callable, List, Dict, Any, Optional, Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict

from game3d.common.enums import PieceType, Color
from game3d.common.piece_utils import get_player_pieces
from game3d.common.move_utils import prepare_batch_data, validate_moves, filter_none_moves
from game3d.common.debug_utils import fallback_mode, track_time, MoveStatsTracker, GeneratorBase

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import Coord, in_bounds, SIZE

# BREAK CIRCULAR IMPORT: Import locally within functions
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
        try:
            mode_enum = self.mode_enum[mode.upper()]
        except KeyError:
            mode_enum = MoveGenMode.STANDARD

        # BREAK CIRCULAR IMPORT: Import locally
        from game3d.movement.pseudo_legal import generate_pseudo_legal_moves
        from game3d.movement.validation import filter_legal_moves

        if mode_enum == MoveGenMode.BATCH:
            moves = _generate_legal_moves_batch(state)
        elif mode_enum == MoveGenMode.PARALLEL:
            # Simplified: Remove complex parallel logic that wasn't working
            moves = _generate_legal_moves_batch(state)
        else:
            moves = _generate_legal_moves_standard(state)

        moves = filter_none_moves(moves)
        _STATS.total_moves_filtered += len(moves)
        return moves

generate_legal_moves = LegalMoveGenerator().generate

def _generate_legal_moves_batch(state: GameState) -> List[Move]:
    """Consolidated batch generation."""
    # BREAK CIRCULAR IMPORT: Import locally
    from game3d.movement.pseudo_legal import generate_pseudo_legal_moves
    from game3d.movement.validation import filter_legal_moves

    pseudo_moves = generate_pseudo_legal_moves(state)
    return filter_legal_moves(pseudo_moves, state)

def _generate_legal_moves_standard(state: GameState) -> List[Move]:
    """Standard generation - reuse batch logic to reduce duplication."""
    return _generate_legal_moves_batch(state)

def generate_legal_moves_excluding_checks(state: GameState) -> List[Move]:
    """Reuse pseudo-legal generation."""
    from game3d.movement.pseudo_legal import generate_pseudo_legal_moves
    return generate_pseudo_legal_moves(state)

def generate_legal_moves_for_piece(state: GameState, coord: Tuple[int, int, int]) -> List[Move]:
    """Single implementation for piece-specific moves."""
    # STANDARDIZED: cache_manager
    piece = state.cache_manager.occupancy.get(coord)
    if not piece or piece.color != state.color:
        return []

    from game3d.movement.pseudo_legal import generate_pseudo_legal_moves_for_piece
    from game3d.movement.validation import filter_legal_moves

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
