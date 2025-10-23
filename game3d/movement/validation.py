# validation.py - FIXED VERSION
"""Centralized validation logic – now fully cache_manager-centric."""
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING

from game3d.movement.movepiece import Move
from game3d.common.enums import Color, PieceType
from game3d.common.piece_utils import get_player_pieces, color_to_code, find_king
from game3d.common.coord_utils import filter_valid_coords
from game3d.common.move_utils import filter_none_moves

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
    from game3d.cache.manager import OptimizedCacheManager

def _get_check_summary(state: GameState) -> Dict[str, Any]:
    # STANDARDIZED: cache_manager
    return state.cache_manager.get_check_summary()

def _attacked_by(state: GameState, attacker: Color) -> set[Tuple[int, int, int]]:
    # STANDARDIZED: cache_manager
    return state.cache_manager.get_attacked_squares(attacker) or set()

def validate_legal_moves(moves: List[Move], state: GameState) -> List[Move]:
    """Single-pass validation without redundant checks."""
    if not moves:
        return []

    valid_moves = []
    for move in moves:
        piece = state.cache_manager.occupancy.get(move.from_coord)
        if piece is None or piece.color != state.color:
            continue
        if state.cache_manager.is_frozen(move.from_coord, state.color):
            continue
        valid_moves.append(move)
    return valid_moves

def _blocked_by_own_color(move: Move, state: GameState) -> bool:
    # STANDARDIZED: cache_manager
    dest = state.cache_manager.occupancy.get(move.to_coord)
    if dest is None:
        return False
    return color_to_code(dest.color) == color_to_code(state.color)

def leaves_king_in_check(move: Move, state: GameState) -> bool:
    # STANDARDIZED: cache_manager
    if state.cache_manager.is_frozen(move.from_coord, state.color):
        return True

    tmp = state.clone()
    tmp.make_move(move)
    summary = _get_check_summary(tmp)
    del tmp
    return summary[f"{state.color.name.lower()}_check"]

def resolves_check(move: Move, state: GameState, check_summary: Dict[str, Any]) -> bool:
    """Consolidated check resolution logic."""
    # STANDARDIZED: cache_manager
    if state.cache_manager.is_frozen(move.from_coord, state.color):
        return False

    if _blocked_by_own_color(move, state):
        return False

    king_color = state.color
    king_pos = find_king(state, king_color)

    if not king_pos:
        return True

    checkers = check_summary.get(f'{king_color.name.lower()}_checkers', [])
    if not checkers:
        return True

    return not leaves_king_in_check(move, state)

def is_between(p: Tuple[int, int, int], start: Tuple[int, int, int], end: Tuple[int, int, int]) -> bool:
    """Vector math for between check - single implementation."""
    dx1 = p[0] - start[0]
    dy1 = p[1] - start[1]
    dz1 = p[2] - start[2]

    dx2 = end[0] - start[0]
    dy2 = end[1] - start[1]
    dz2 = end[2] - start[2]

    cross_x = dy1 * dz2 - dz1 * dy2
    cross_y = dz1 * dx2 - dx1 * dz2
    cross_z = dx1 * dy2 - dy1 * dx2

    if cross_x != 0 or cross_y != 0 or cross_z != 0:
        return False

    if dx2 != 0:
        t = dx1 / dx2
    elif dy2 != 0:
        t = dy1 / dy2
    elif dz2 != 0:
        t = dz1 / dz2
    else:
        return p == start

    return 0 <= t <= 1

def blocks_check(move: Move, king: Tuple[int, int, int], checker: Tuple[int, int, int]) -> bool:
    return is_between(move.to_coord, king, checker)

def validate_archery_attack(game_state: GameState, target_sq: Tuple[int, int, int]) -> Dict[str, Any]:
    """Single-pass archery validation."""
    # STANDARDIZED: cache_manager
    archer_pos = None
    for coord, piece in get_player_pieces(game_state, game_state.color):
        if piece.ptype == PieceType.ARCHER:
            archer_pos = coord
            break

    if archer_pos is None:
        return {'valid': False, 'message': "No archer controlled."}

    # STANDARDIZED: cache_manager
    if game_state.cache_manager.is_frozen(archer_pos, game_state.color):
        return {'valid': False, 'message': "Archer is frozen and cannot attack."}

    return {'valid': True, 'message': ""}

def validate_hive_moves(game_state: GameState, moves: List[Move]) -> Dict[str, Any]:
    """Consolidated hive move validation."""
    if not moves:
        return {'valid': False, 'message': "No moves submitted."}

    for coord, piece in get_player_pieces(game_state, game_state.color):
        # STANDARDIZED: cache_manager
        if piece.ptype == PieceType.HIVE and game_state.cache_manager.is_frozen(coord, game_state.color):
            return {'valid': False, 'message': "Hive is frozen and cannot move."}

    return {'valid': True, 'message': ""}

def filter_legal_moves(moves: List[Move], state: GameState) -> List[Move]:
    """Single-pass legal move filtering with consolidated logic."""
    if not moves:
        return []

    moves = filter_none_moves(moves)
    if not moves:
        return []

    # STANDARDIZED: cache_manager
    priest_flags = state.cache_manager.priest_status()
    friendly_priests_key = f"{state.color.name.lower()}_priests_alive"
    friendly_priests_alive = priest_flags.get(friendly_priests_key, False)

    # Fast path: if priests alive, only filter frozen pieces
    if friendly_priests_alive:
        filtered = [m for m in moves if not state.cache_manager.is_frozen(m.from_coord, state.color)]
        return filter_none_moves(filtered)

    # Full validation path
    summary = _get_check_summary(state)
    attacked = summary[f"attacked_squares_{state.color.opposite().name.lower()}"]
    king_pos = summary[f"{state.color.name.lower()}_king_position"]
    in_check = summary[f"{state.color.name.lower()}_check"]

    legal = []
    for m in moves:
        # STANDARDIZED: cache_manager
        if state.cache_manager.is_frozen(m.from_coord, state.color):
            continue

        # King safety check
        if king_pos and m.from_coord == king_pos and m.to_coord in attacked:
            continue

        # Check resolution
        if in_check and not resolves_check(m, state, summary):
            continue

        legal.append(m)

    return filter_none_moves(legal)
