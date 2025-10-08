# validation.py
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


def leaves_king_in_check(move: Move, state: State) -> bool:
    """Slower path that actually plays the move and re-uses cache."""
    tmp = state.clone()
    tmp.make_move(move)
    summary = _get_check_summary(tmp)
    return summary[f"{state.color.name.lower()}_check"]


def leaves_king_in_check_optimized(
    move: Move,
    state: GameState,
    summary: Dict[str, Any]
) -> bool:
    """Fast path: every lookup is cached."""
    if _blocked_by_own_color(move, state):
        return True

    king_pos = summary[f"{state.color.name.lower()}_king_position"]
    if not king_pos:
        return False

    # King move -> verify destination not attacked
    if move.from_coord == king_pos:
        attacked = summary[f"attacked_squares_{state.color.opposite().name.lower()}"]
        return move.to_coord in attacked

    # Already in check → must resolve
    if summary[f"{state.color.name.lower()}_check"]:
        # fall back to full test (still cache-based)
        return leaves_king_in_check(move, state)

    # Discovered-check (pin) test
    if state.cache.is_pinned(move.from_coord):
        pin_dir = state.cache.get_pin_direction(move.from_coord)
        return not along_pin_line(move, pin_dir)

    return False


def resolves_check(move: Move, state: 'GameState', check_summary: Dict[str, Any]) -> bool:
    if _blocked_by_own_color(move, state):       # <-- NEW guard
        return False
    king_color = state.color
    king_pos = check_summary[f'{king_color.name.lower()}_king_position']

    if not king_pos:
        return False

    # Get checkers information
    checkers = check_summary.get(f'{king_color.name.lower()}_checkers', [])

    if not checkers:
        return True  # No check, any move is fine

    # If multiple checkers, only king moves can resolve
    if len(checkers) > 1:
        return move.from_coord == king_pos

    # Single checker - can block, capture, or move king
    checker_pos = checkers[0]

    # Moving the king
    if move.from_coord == king_pos:
        attacked_squares = check_summary[f'attacked_squares_{king_color.opposite().name.lower()}']
        return move.to_coord not in attacked_squares

    # Capturing the checker
    if move.to_coord == checker_pos:
        return True

    # Blocking the check ray
    if blocks_check(move, king_pos, checker_pos):
        return True

    return False


# ==============================================================================
# GEOMETRIC VALIDATION
# ==============================================================================

def is_between(p: Coord, start: Coord, end: Coord) -> bool:
    """Check if point lies on the line segment between start and end."""
    # Check if point is collinear with start and end
    dx1 = point[0] - start[0]
    dy1 = point[1] - start[1]
    dz1 = point[2] - start[2]

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
        return point == start  # start and end are the same

    return 0 <= t <= 1


def blocks_check(move: Move, king: Coord, checker: Coord) -> bool:
    return is_between(move.to_coord, king, checker)

def along_pin_line(move: Move, pin_dir: Tuple[int, int, int]) -> bool:
    """Check if move stays along pin line."""
    move_direction = (
        move.to_coord[0] - move.from_coord[0],
        move.to_coord[1] - move.from_coord[1],
        move.to_coord[2] - move.from_coord[2]
    )

    # Normalize directions for comparison
    def normalize_direction(dx, dy, dz):
        length = max(abs(dx), abs(dy), abs(dz))
        if length == 0:
            return (0, 0, 0)
        return (dx // length, dy // length, dz // length)

    return normalize_direction(*move_direction) == normalize_direction(*pin_direction)


# ==============================================================================
# BATCH VALIDATION
# ==============================================================================
def batch_check_validation(moves: List[Move], state: GameState) -> List[Move]:
    """Optimized batch validation (assumes basic-legal)."""
    if not moves:
        return []

    # Cache expensive lookups
    check_summary = get_check_summary(state.board, state.cache)
    in_check = check_summary[f'{state.color.name.lower()}_check']

    if not in_check:
        # Fast path - no check validation needed
        return moves

    # Full validation only when in check
    # Use list comprehension instead of loop for performance
    return [
        mv for mv in moves
        if not leaves_king_in_check(mv, state)
    ]


def validate_move_batch(moves: List[Move], state: GameState) -> List[Move]:
    """Validate a batch of moves in parallel (assumes basic-legal)."""
    legal_batch = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(leaves_king_in_check, move, state) for move in moves]
        for future, move in zip(as_completed(futures), moves):
            if not future.result():
                legal_batch.append(move)
    return legal_batch

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
        # Fast king safety
        if king_pos and m.from_coord == king_pos and m.to_coord in attacked:
            continue
        # Must resolve check
        if in_check and not resolves_check(m, state, summary):
            continue
        legal.append(m)
    return legal

# ==============================================================================
# SPECIAL MOVE VALIDATION
# ==============================================================================

def validate_archery_attack(game_state: GameState, target_sq: Tuple[int, int, int]) -> Dict[str, Any]:
    """Validate archery attack for current player."""
    if not game_state._current_player_has_archer():
        return {'valid': False, 'message': "No archer controlled."}

    if not game_state._is_valid_archery_target(target_sq):
        return {'valid': False, 'message': "Invalid archery target - must be on 2-radius sphere surface."}

    if not game_state._has_archery_line_of_sight(target_sq):
        return {'valid': False, 'message': "No clear line of sight to target."}

    return {'valid': True, 'message': ""}


def validate_hive_moves(game_state: GameState, moves: List[Move]) -> Dict[str, Any]:
    """Validate hive moves for current player."""
    if not moves:
        return {'valid': False, 'message': "No moves submitted."}

