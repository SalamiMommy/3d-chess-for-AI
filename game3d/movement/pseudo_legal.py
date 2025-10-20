# pseudo_legal.py
"""Optimized pseudo-legal move generator with enhanced caching and performance."""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import numpy as np
from enum import Enum
import time
from collections import defaultdict

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

from game3d.common.enums import Color, PieceType
from game3d.movement.movepiece import Move
from game3d.movement.registry import (
    register, get_dispatcher, get_all_dispatchers, dispatch_batch
)
from game3d.movement.movementmodifiers import modify_raw_moves
from game3d.common.common import (
    SIZE, get_player_pieces, extend_move_range, filter_valid_coords,
    color_to_code, extract_directions_and_steps_vectorized, MoveStatsTracker,
    track_time, fallback_mode, validate_moves, prepare_batch_data, GeneratorBase, filter_none_moves
)

BOARD_SIZE = SIZE

_STATS = MoveStatsTracker()


class PseudoLegalMode(Enum):
    STANDARD    = "standard"
    BATCH       = "batch"
    INCREMENTAL = "incremental"


class PseudoLegalGenerator(GeneratorBase):
    def __init__(self):
        super().__init__(PseudoLegalMode, PseudoLegalMode.STANDARD, _STATS)

    @track_time(_STATS)
    def _impl(self, state: GameState, mode: str) -> List[Move]:
        mode_enum = self.mode_enum[mode.upper()]

        if mode_enum == PseudoLegalMode.BATCH:
            moves = _generate_pseudo_legal_batch(state)
        elif mode_enum == PseudoLegalMode.INCREMENTAL:
            moves = _generate_pseudo_legal_incremental(state)
        else:
            moves = _generate_pseudo_legal_standard(state)

        # DEFENSIVE: Filter out None moves
        moves = filter_none_moves(moves)

        _STATS.total_moves_generated += len(moves)
        return moves

generate_pseudo_legal_moves = PseudoLegalGenerator().generate

def generate_pseudo_legal_moves_for_piece(
    state: GameState, coord: Tuple[int, int, int]
) -> List[Move]:
    """Generate every pseudo-legal move for the piece standing on *coord*."""
    piece = state.cache.piece_cache.get(coord)
    if not piece or piece.color != state.color:
        return []

    is_debuffed = state.cache.is_movement_debuffed(coord, piece.color)
    is_frozen   = state.cache.is_frozen(coord, piece.color)

    # Frozen pieces cannot move at all
    if is_frozen:
        return []

    # De-buffed pieces move like pawns (unless they already are pawns)
    pt = PieceType.PAWN if is_debuffed and piece.ptype != PieceType.PAWN else piece.ptype

    dispatcher = get_dispatcher(pt)
    if dispatcher is None:
        return []

    piece_moves = dispatcher(state, *coord)
    if not piece_moves:
        return []

    # DEFENSIVE: Filter out None moves early
    piece_moves = filter_none_moves(piece_moves)
    if not piece_moves:
        return []

    to_coords = np.array([m.to_coord for m in piece_moves], dtype=np.int32)
    captures  = np.array([m.is_capture for m in piece_moves], dtype=bool)

    modified_moves = modify_raw_moves(
        from_coord=coord,
        to_coords=to_coords,
        captures=captures,
        color=piece.color,
        cache_manager=state.cache,
        debuffed=is_debuffed,
        current_ply=state.ply,
    )

    # DEFENSIVE: Filter out None moves before validation
    modified_moves = filter_none_moves(modified_moves)

    validated = validate_moves(modified_moves, state, piece)

    # DEFENSIVE: Final None filter
    return filter_none_moves(validated)
# ------------------------------------------------------------------
#  Batch mode
# ------------------------------------------------------------------
def _generate_pseudo_legal_batch(state: GameState) -> List[Move]:
    """Batch mode: generate moves for all pieces at once."""
    coords, types, debuffed_indices = prepare_batch_data(state)

    raw_moves = dispatch_batch(state, coords, types, state.color)

    if debuffed_indices:
        debuffed_coords = [coords[i] for i in debuffed_indices]
        debuffed_types  = [PieceType.PAWN] * len(debuffed_indices)
        debuffed_raw    = dispatch_batch(state, debuffed_coords, debuffed_types, state.color)
        for idx, moves in zip(debuffed_indices, debuffed_raw):
            raw_moves[idx] = moves

    all_moves = []
    for coord, piece_moves, piece_type in zip(coords, raw_moves, types):
        if not piece_moves:
            continue

        # DEFENSIVE: Filter out None moves early
        piece_moves = filter_none_moves(piece_moves)
        if not piece_moves:
            continue

        is_debuffed = state.cache.is_movement_debuffed(coord, state.color)

        to_coords = np.array([m.to_coord for m in piece_moves], dtype=np.int32)
        captures  = np.array([m.is_capture for m in piece_moves], dtype=bool)

        modified_moves = modify_raw_moves(
            from_coord=coord,
            to_coords=to_coords,
            captures=captures,
            color=state.color,
            cache_manager=state.cache,
            debuffed=is_debuffed,
            current_ply=state.ply,
        )

        # DEFENSIVE: Filter None moves
        modified_moves = filter_none_moves(modified_moves)

        validated = validate_moves(modified_moves, state)

        # DEFENSIVE: Filter None moves again
        validated = filter_none_moves(validated)

        all_moves.extend(validated)
        _STATS.piece_breakdown[piece_type] += len(validated)

    # DEFENSIVE: Final None filter
    return filter_none_moves(all_moves)
# ------------------------------------------------------------------
#  Incremental mode
# ------------------------------------------------------------------
def _generate_pseudo_legal_incremental(state: GameState) -> List[Move]:
    """Incremental mode: generate moves using per-piece function."""
    all_moves = []

    for coord, piece in get_player_pieces(state, state.color):
        moves = generate_pseudo_legal_moves_for_piece(state, coord)

        # DEFENSIVE: Filter None moves (already done in generate_pseudo_legal_moves_for_piece, but be safe)
        moves = filter_none_moves(moves)

        validated = validate_moves(moves, state, piece)

        # DEFENSIVE: Filter None again
        validated = filter_none_moves(validated)

        all_moves.extend(validated)

    # DEFENSIVE: Final None filter
    return filter_none_moves(all_moves)
# ------------------------------------------------------------------
#  Standard mode
# ------------------------------------------------------------------
def _generate_pseudo_legal_standard(state: GameState) -> List[Move]:
    """Standard mode: generate pseudo-legal moves piece by piece."""
    all_moves = []

    for coord, piece in get_player_pieces(state, state.color):
        try:
            dispatcher = get_dispatcher(piece.ptype)
            if dispatcher is None:
                continue

            piece_moves = dispatcher(state, *coord)

            # DEFENSIVE: Filter out None moves early
            if piece_moves:
                piece_moves = filter_none_moves(piece_moves)

            is_debuffed = state.cache.is_movement_debuffed(coord, piece.color)
            is_frozen   = state.cache.is_frozen(coord, piece.color)

            if is_frozen:          # frozen pieces do not move
                continue

            modified_moves = []
            if piece_moves:
                to_coords = np.array([m.to_coord for m in piece_moves], dtype=np.int32)
                captures  = np.array([m.is_capture for m in piece_moves], dtype=bool)

                modified_moves = modify_raw_moves(
                    from_coord=coord,
                    to_coords=to_coords,
                    captures=captures,
                    color=piece.color,
                    cache_manager=state.cache,
                    debuffed=is_debuffed,
                    current_ply=state.ply,
                )

                # DEFENSIVE: Filter None moves from modified_moves
                modified_moves = filter_none_moves(modified_moves)

            validated_moves = validate_moves(modified_moves, state, piece)

            # DEFENSIVE: Filter None moves from validated_moves
            validated_moves = filter_none_moves(validated_moves)

            if validated_moves:
                all_moves.extend(validated_moves)
                _STATS.piece_breakdown[piece.ptype] += len(validated_moves)
                _STATS.total_moves_generated += len(validated_moves)

        except Exception as e:
            print(f"Error generating moves for {piece.ptype} at {coord}: {e}")
            continue

    # DEFENSIVE: Final None filter before returning
    return filter_none_moves(all_moves)
# ------------------------------------------------------------------
#  Statistics helpers
# ------------------------------------------------------------------
def get_pseudo_legal_stats() -> Dict[str, Any]:
    return _STATS.get_stats()


def reset_pseudo_legal_stats() -> None:
    _STATS.reset()
