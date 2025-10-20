# generator.py
from __future__ import annotations
from typing import Callable, List, Dict, Any, Optional, Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from game3d.common.enums import PieceType, Color
from game3d.common.piece_utils import get_player_pieces
from game3d.common.move_utils import prepare_batch_data, validate_moves, filter_none_moves
from game3d.common.debug_utils import fallback_mode, track_time, MoveStatsTracker, GeneratorBase

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import Coord, in_bounds, SIZE
from game3d.movement.registry import register, get_dispatcher, get_all_dispatchers
from game3d.movement.pseudo_legal import generate_pseudo_legal_moves, generate_pseudo_legal_moves_for_piece
from game3d.attacks.check import king_in_check, get_check_summary
from game3d.movement.validation import (
    leaves_king_in_check,
    resolves_check,
    filter_legal_moves
)
import sys
from functools import wraps

def recursion_limit(max_depth):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # no depth tracking – straight call
            return func(*args, **kwargs)
        return wrapper
    return decorator


def _generate_legal_moves_fallback(state: GameState) -> List[Move]:
    pseudo_moves = generate_pseudo_legal_moves(state)
    return filter_legal_moves(pseudo_moves, state)

BOARD_SIZE = SIZE

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

        if mode_enum == MoveGenMode.BATCH:
            moves = _generate_legal_moves_batch(state)
        elif mode_enum == MoveGenMode.PARALLEL:
            if len([p for p in state.board.list_occupied()]) > 400:
                moves = _generate_legal_moves_batch(state)
            else:
                moves = _generate_legal_moves_parallel(state)
        else:
            moves = _generate_legal_moves_standard(state)

        # DEFENSIVE: Filter out None moves before returning
        moves = filter_none_moves(moves)

        _STATS.total_moves_filtered += len(moves)

        return moves

generate_legal_moves = LegalMoveGenerator().generate

def _generate_legal_moves_batch(state: GameState) -> List[Move]:
    pseudo_moves = generate_pseudo_legal_moves(state)
    return filter_legal_moves(pseudo_moves, state)

def _generate_legal_moves_parallel(state: GameState) -> List[Move]:
    pseudo_legal_moves = generate_pseudo_legal_moves(state)

    if not pseudo_legal_moves:
        return []

    legal_moves = filter_legal_moves(pseudo_legal_moves, state)

    return legal_moves

def _generate_legal_moves_standard(state: GameState) -> List[Move]:
    pseudo_moves = generate_pseudo_legal_moves(state)
    return filter_legal_moves(pseudo_moves, state)

def generate_legal_moves_excluding_checks(state: GameState) -> List[Move]:
    return generate_pseudo_legal_moves(state)

def generate_legal_moves_for_piece(state: GameState, coord: Tuple[int, int, int]) -> List[Move]:
    piece = state.cache.piece_cache.get(coord)
    if not piece or piece.color != state.color:
        return []
    pseudo_moves = generate_pseudo_legal_moves_for_piece(state, coord)
    return filter_legal_moves(pseudo_moves, state)

def generate_legal_captures(state: GameState) -> List[Move]:
    all_legal = generate_legal_moves(state)
    return [mv for mv in all_legal if mv.is_capture]

def generate_legal_non_captures(state: GameState) -> List[Move]:
    all_legal = generate_legal_moves(state)
    return [mv for mv in all_legal if not mv.is_capture]

def get_move_generation_stats() -> Dict[str, Any]:
    return _STATS.get_stats()

def reset_move_gen_stats() -> None:
    _STATS.reset()
