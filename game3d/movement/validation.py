# validation.py
"""
game3d/movement/validation.py
Centralized validation logic for move generation and game rules.
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed

from game3d.movement.movepiece import Move
from game3d.pieces.enums import Color, PieceType

from game3d.common.common import Coord
from game3d.attacks.check import king_in_check, get_check_summary

# Use forward references to avoid circular imports
if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
    from game3d.cache.manager import OptimizedCacheManager
# ==============================================================================
# BASIC MOVE VALIDATION
# ==============================================================================

def validate_legal_moves(cache: OptimizedCacheManager, moves: List[Move], color: Color) -> List[Move]:
    """Paranoid validation of legal moves - remove any from empty squares."""
    valid_moves = []

    for move in moves:
        piece = cache.piece_cache.get(move.from_coord)
        if piece is None:
            continue  # Skip instead of raise to avoid crash, but log
        # Additional validation
        if piece.color != color:
            continue
        valid_moves.append(move)

    return valid_moves


def is_basic_legal(move: Move, state: 'GameState') -> bool:
    """Basic legality checks."""
    if not (0 <= move.to_coord[0] < 9 and
            0 <= move.to_coord[1] < 9 and
            0 <= move.to_coord[2] < 9):
        return False

    dest_piece = state.cache.piece_cache.get(move.to_coord)
    if dest_piece and dest_piece.color == state.color:
        return False

    return True


# ==============================================================================
# CHECK VALIDATION
# ==============================================================================

def leaves_king_in_check(move: Move, state: 'GameState') -> bool:
    """Check if move leaves king in check."""
    temp_state = state.clone()
    temp_state.make_move(move)
    return king_in_check(temp_state.board, state.color, state.color.opposite(), temp_state.cache)


def leaves_king_in_check_optimized(move: Move, state: 'GameState', check_summary: Dict[str, Any]) -> bool:
    """Optimized check validation using pre-computed position state."""
    king_color = state.color
    king_pos = check_summary[f'{king_color.name.lower()}_king_position']
    if not king_pos:
        return False

    # Case 1: Moving the king - check destination safety
    if move.from_coord == king_pos:
        attacked_squares = check_summary[f'attacked_squares_{king_color.opposite().name.lower()}']
        return move.to_coord in attacked_squares

    # Case 2: In check - must block or capture
    if check_summary[f'{king_color.name.lower()}_check']:
        return leaves_king_in_check(move, state)

    # Case 3: Check for discovered attacks (pinned pieces)
    if state.cache.is_pinned(move.from_coord):
        pin_direction = state.cache.get_pin_direction(move.from_coord)
        if not along_pin_line(move, pin_direction):
            return True

    # Fast case: no immediate check concerns
    return False


def resolves_check(move: Move, state: 'GameState', check_summary: Dict[str, Any]) -> bool:
    """Check if move resolves the current check situation."""
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

def is_between(point: Coord, start: Coord, end: Coord) -> bool:
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


def blocks_check(move: Move, king_pos: Coord, checker_pos: Coord) -> bool:
    """Check if move blocks the check ray."""
    return is_between(move.to_coord, king_pos, checker_pos)


def along_pin_line(move: Move, pin_direction: Tuple[int, int, int]) -> bool:
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
    """Optimized batch validation."""
    if not moves:
        return []

    # Cache expensive lookups
    check_summary = get_check_summary(state.board, state.cache)
    in_check = check_summary[f'{state.color.name.lower()}_check']

    if not in_check:
        # Fast path - only basic validation needed
        return [mv for mv in moves if is_basic_legal(mv, state)]

    # Full validation only when in check
    # Use list comprehension instead of loop for performance
    return [
        mv for mv in moves
        if is_basic_legal(mv, state) and not leaves_king_in_check(mv, state)
    ]


def validate_move_batch(moves: List[Move], state: GameState) -> List[Move]:
    """Validate a batch of moves in parallel."""
    legal_batch = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(leaves_king_in_check, move, state) for move in moves]
        for future, move in zip(as_completed(futures), moves):
            if not future.result():
                legal_batch.append(move)
    return legal_batch


# ==============================================================================
# LEGAL MOVE FILTERING
# ==============================================================================
def filter_legal_moves(moves: List[Move], state: GameState) -> List[Move]:
    """Optimized batch legal move filtering with incremental validation."""
    if not moves:
        return moves

    # Get position state ONCE for all moves
    check_summary = get_check_summary(state.board, state.cache)
    legal_moves = []

    # Pre-compute attacked squares for efficiency
    attacked_squares = check_summary[f'attacked_squares_{state.color.opposite().name.lower()}']
    king_pos = check_summary[f'{state.color.name.lower()}_king_position']
    in_check = check_summary[f'{state.color.name.lower()}_check']

    for move in moves:
        # Basic legality check
        if not is_basic_legal(move, state):
            continue

        # Fast check for king moves
        if move.from_coord == king_pos:
            if move.to_coord in attacked_squares:
                continue
        # If in check, only allow moves that resolve the check
        elif in_check:
            if not resolves_check(move, state, check_summary):
                continue

        legal_moves.append(move)

    return legal_moves


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

    # Check all pieces are hive pieces
    for move in moves:
        piece = game_state.cache.piece_cache.get(move.from_coord)
        if not piece or piece.ptype != PieceType.HIVE or piece.color != game_state.color:
            return {'valid': False, 'message': "Only Hive pieces may move."}

    # Check no non-hive alternatives exist
    all_legal = game_state.legal_moves()
    non_hive_moves = [m for m in all_legal if
                      game_state.cache.piece_cache.get(m.from_coord).ptype != PieceType.HIVE]
    if non_hive_moves:
        return {'valid': False, 'message': "Must move only Hive pieces this turn."}

    # Validate each move
    for move in moves:
        if move not in all_legal:
            return {'valid': False, 'message': f"Illegal move: {move}"}

    return {'valid': True, 'message': ""}


def validate_move_fast(move: Move, state: GameState, legal_moves: List[Move]) -> Dict[str, Any]:
    """Fast move validation pipeline."""
    # Check game over
    if state.is_game_over():
        return {'valid': False, 'message': "Game already finished."}

    # Check piece exists
    piece = state.cache.piece_cache.get(move.from_coord)
    if piece is None:
        return {'valid': False, 'message': f"No piece at {move.from_coord}"}

    # Check color
    if piece.color != state.color:
        return {'valid': False, 'message': "Not your turn."}

    # Check legality (cached)
    if move not in legal_moves:
        return {'valid': False, 'message': "Illegal move."}

    return {'valid': True, 'message': ""}
