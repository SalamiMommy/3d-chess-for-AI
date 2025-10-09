"""Centralized validation logic – now fully cache-centric."""
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING

from game3d.movement.movepiece import Move
from game3d.pieces.enums import Color, PieceType

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

# ------------------------------------------------------------------
# 1.  ONE-LINE CACHE HELPERS
# ------------------------------------------------------------------
def _get_check_summary(state: GameState) -> Dict[str, Any]:
    """Return the *single* check summary object stored in cache."""
    # lazily computed once per ply inside the cache manager
    return state.cache.get_check_summary()


def _attacked_by(state: GameState, attacker: Color) -> set[Tuple[int, int, int]]:
    """Attacked squares for a colour – straight from AttacksCache."""
    return state.cache.effects.attacks.get_for_color(attacker) or set()

# ==============================================================================
# BASIC MOVE VALIDATION (MOVED TO PSEUDO_LEGAL.PY)
# ==============================================================================

def validate_legal_moves(cache: OptimizedCacheManager, moves: List[Move], color: Color) -> List[Move]:
    """Paranoid validation of legal moves - remove any from empty squares."""
    valid_moves = []

    for move in moves:
        piece = cache.occupancy.get(move.from_coord)
        if piece is None:
            continue  # Skip instead of raise to avoid crash, but log
        # Additional validation
        if piece.color != color:
            continue
        valid_moves.append(move)

    return valid_moves

# ==============================================================================
# CHECK VALIDATION
# ==============================================================================
def _blocked_by_own_color(move: Move, state: GameState) -> bool:
    """True if destination is friendly (KNIGHT exception handled)."""
    dest = state.cache.occupancy.get(move.to_coord)
    if dest is None:
        return False
    if dest.color != state.color:
        return False
    # Knight may jump over friends
    mover = state.cache.occupancy.get(move.from_coord)
    return mover is None or mover.ptype is not PieceType.KNIGHT


def leaves_king_in_check(move: Move, state: GameState) -> bool:
    """Slower path that actually plays the move and re-uses cache."""
    # Check if piece is frozen before attempting move
    if state.cache.is_frozen(move.from_coord, state.color):
        return True  # Frozen piece can't move, so would leave king in check

    tmp = state.clone()
    tmp.make_move(move)
    summary = _get_check_summary(tmp)
    return summary[f"{state.color.name.lower()}_check"]


# Remove the optimized version since we don't have pinning in cache yet
# def leaves_king_in_check_optimized(...): ...


def resolves_check(move: Move, state: GameState, check_summary: Dict[str, Any]) -> bool:
    # First check if the move is even possible (frozen pieces can't move)
    if state.cache.is_frozen(move.from_coord, state.color):
        return False

    if _blocked_by_own_color(move, state):       # <-- NEW guard
        return False
    king_color = state.color
    king_pos = check_summary[f'{king_color.name.lower()}_king_position']

    if not king_pos:
        return True  # No king, so no check

    # Get checkers information
    checkers = check_summary.get(f'{king_color.name.lower()}_checkers', [])

    if not checkers:
        return True  # No check, any move is fine

    # Always use the full check validation for safety
    return not leaves_king_in_check(move, state)


# ==============================================================================
# GEOMETRIC VALIDATION
# ==============================================================================

def is_between(p: Tuple[int, int, int], start: Tuple[int, int, int], end: Tuple[int, int, int]) -> bool:
    """Check if point lies on the line segment between start and end."""
    # Check if point is collinear with start and end
    dx1 = p[0] - start[0]
    dy1 = p[1] - start[1]
    dz1 = p[2] - start[2]

    dx2 = end[0] - start[0]
    dy2 = end[1] - start[1]
    dz2 = end[2] - start[2]

    # Cross product should be zero for collinear points
    cross_x = dy1 * dz2 - dz1 * dy2
    cross_y = dz1 * dx2 - dx1 * dz2
    cross_z = dx1 * dy2 - dy1 * dx2

    if cross_x != 0 or cross_y != 0 or cross_z != 0:
        return False

    # Check if point is between start and end
    if dx2 != 0:
        t = dx1 / dx2
    elif dy2 != 0:
        t = dy1 / dy2
    elif dz2 != 0:
        t = dz1 / dz2
    else:
        return p == start  # start and end are the same

    return 0 <= t <= 1


def blocks_check(move: Move, king: Tuple[int, int, int], checker: Tuple[int, int, int]) -> bool:
    return is_between(move.to_coord, king, checker)

# ==============================================================================
# SPECIAL MOVE VALIDATION
# ==============================================================================

def validate_archery_attack(game_state: GameState, target_sq: Tuple[int, int, int]) -> Dict[str, Any]:
    """Validate archery attack for current player."""
    # First check if archer is frozen
    archer_pos = None
    for coord, piece in _get_current_player_pieces(game_state):
        if piece.ptype == PieceType.ARCHER:
            archer_pos = coord
            break

    if archer_pos is None:
        return {'valid': False, 'message': "No archer controlled."}

    if game_state.cache.is_frozen(archer_pos, game_state.color):
        return {'valid': False, 'message': "Archer is frozen and cannot attack."}

    if not game_state._is_valid_archery_target(target_sq):
        return {'valid': False, 'message': "Invalid archery target - must be on 2-radius sphere surface."}

    if not game_state._has_archery_line_of_sight(target_sq):
        return {'valid': False, 'message': "No clear line of sight to target."}

    return {'valid': True, 'message': ""}


def validate_hive_moves(game_state: GameState, moves: List[Move]) -> Dict[str, Any]:
    """Validate hive moves for current player."""
    if not moves:
        return {'valid': False, 'message': "No moves submitted."}

    # Check if any hive piece is frozen
    for coord, piece in _get_current_player_pieces(game_state):
        if piece.ptype == PieceType.HIVE and game_state.cache.is_frozen(coord, game_state.color):
            return {'valid': False, 'message': "Hive is frozen and cannot move."}

    return {'valid': True, 'message': ""}

# ------------------------------------------------------------------
# 4.  BATCH VALIDATION – ZERO BOARD SCANNING
# ------------------------------------------------------------------
def filter_legal_moves(moves: List[Move], state: GameState) -> List[Move]:
    """Return fully legal moves – all data already in cache."""
    if not moves:
        return []

    summary = _get_check_summary(state)
    attacked = summary[f"attacked_squares_{state.color.opposite().name.lower()}"]
    king_pos = summary[f"{state.color.name.lower()}_king_position"]
    in_check = summary[f"{state.color.name.lower()}_check"]

    legal = []
    for m in moves:
        # Skip moves from frozen pieces
        if state.cache.is_frozen(m.from_coord, state.color):
            continue

        # Fast king safety
        if king_pos and m.from_coord == king_pos and m.to_coord in attacked:
            continue
        # Must resolve check
        if in_check and not resolves_check(m, state, summary):
            continue
        legal.append(m)
    return legal

# Remove unused batch validation functions
# def batch_check_validation(...): ...
# def validate_move_batch(...): ...
