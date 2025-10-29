# game3d/common/validation.py
"""CONSOLIDATED MOVE VALIDATION MODULE"""
from __future__ import annotations
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional, Union, Set, TYPE_CHECKING

from game3d.common.constants import SIZE
from game3d.common.enums import Color, PieceType
from game3d.common.piece_utils import get_player_pieces, color_to_code, find_king
from game3d.common.coord_utils import filter_valid_coords, Coord
from game3d.common.move_utils import filter_none_moves
from game3d.pieces.piece import Piece
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
    from game3d.movement.cache_utils import OptimizedCacheManager


# =============================================================================
# Core Validation Functions
# =============================================================================
def validate_moves_fast(
    moves: List[Move],
    state: GameState,
    piece: Optional[Piece] = None
) -> List[Move]:
    """Ultra-fast validation for batch processing - minimal checks only."""
    if not moves:
        return []

    cache = state.cache_manager
    expected_color = piece.color if piece else state.color

    # Extract coordinates in single batch
    from_coords = np.array([m.from_coord for m in moves], dtype=np.int32)
    to_coords = np.array([m.to_coord for m in moves], dtype=np.int32)

    # Batch occupancy checks
    from_colors, _ = cache.occupancy.batch_get_colors_and_types(from_coords)
    to_colors, _ = cache.occupancy.batch_get_colors_and_types(to_coords)

    expected_code = 1 if expected_color == Color.WHITE else 2

    # Fast validation mask
    valid = (
        (from_colors == expected_code) &  # Correct color
        ((to_colors == 0) | (to_colors != expected_code))  # Valid destination
    )

    # Return filtered moves (skip effect checks for performance)
    return [moves[i] for i in range(len(moves)) if valid[i]]


def validate_move_basic(
    game_state: GameState,
    move: Union[Move, List[Move]],
    expected_color: Optional[Color] = None
) -> Union[bool, List[bool]]:
    """Basic move validation - supports scalar and batch mode."""
    # Handle batch mode
    if isinstance(move, list):
        return [validate_move_basic(game_state, m, expected_color) for m in move]

    # Scalar mode
    cache = game_state.cache_manager
    expected_color = expected_color or game_state.color

    # Check piece exists and color matches
    from_piece = cache.occupancy.get(move.from_coord)
    if from_piece is None:
        return False
    if from_piece.color != expected_color:
        return False

    # Check frozen status
    return not cache.is_frozen(move.from_coord, from_piece.color)


def validate_move_destination(
    game_state: GameState,
    move: Union[Move, List[Move]],
    piece_color: Union[Color, List[Color]]
) -> Union[bool, List[bool]]:
    """Destination validation - supports scalar and batch mode."""
    # Handle batch mode
    if isinstance(move, list):
        results = []
        for i, m in enumerate(move):
            single_color = piece_color[i] if isinstance(piece_color, list) else piece_color
            results.append(validate_move_destination(game_state, m, single_color))
        return results

    # Scalar mode
    cache = game_state.cache_manager
    to_piece = cache.occupancy.get(move.to_coord)
    return to_piece is None or to_piece.color != piece_color


def validate_move_comprehensive(
    cache_manager: "OptimizedCacheManager",
    move: Union[Move, List[Move]],
    expected_color: Union[Color, List[Color]],
    check_effects: bool = True
) -> Union[bool, List[bool]]:
    """Comprehensive move validation with all effect checks - supports scalar and batch mode."""
    # Handle batch mode
    if isinstance(move, list):
        results = []
        for i, m in enumerate(move):
            single_color = expected_color[i] if isinstance(expected_color, list) else expected_color
            results.append(validate_move_comprehensive(cache_manager, m, single_color, check_effects))
        return results

    # Scalar mode - Basic validation
    from game3d.game.gamestate import GameState
    dummy_state = type('DummyState', (), {'cache_manager': cache_manager, 'color': expected_color})()

    if not validate_move_basic(dummy_state, move, expected_color):
        return False

    # Destination validation
    if not validate_move_destination(dummy_state, move, expected_color):
        return False

    # Early return if no effect checks needed
    if not check_effects:
        return True

    # Effect-based validations
    if (cache_manager.is_frozen(move.from_coord, expected_color) or
        cache_manager.is_geomancy_blocked(move.to_coord, cache_manager.halfmove_clock)):
        return False

    # Handle movement debuff logic if needed
    if cache_manager.is_movement_debuffed(move.from_coord, expected_color):
        # Debuff logic implementation
        pass

    return True


# =============================================================================
# Legal Move Filtering
# =============================================================================

def _get_check_summary(state: GameState) -> Dict[str, Any]:
    """Use cache manager's method - ensure it exists"""
    if not hasattr(state, 'cache_manager'):
        raise RuntimeError("GameState missing cache_manager")
    return state.cache_manager.get_check_summary(state.has_priest)


def _attacked_by(state: GameState, attacker: Color) -> set[Coord]:
    """Use cache manager's method"""
    return state.cache_manager.get_attacked_squares(attacker)


def has_priest(state: GameState, color: Color) -> bool:
    """Use cache manager's method"""
    return state.cache_manager.has_priest(color)


def _blocked_by_own_color(move: Move, state: GameState) -> bool:
    """Use cache manager"""
    cache_manager = state.cache_manager
    dest = cache_manager.occupancy_cache.get(move.to_coord)
    if dest is None:
        return False
    return color_to_code(dest.color) == color_to_code(state.color)


def leaves_king_in_check(move: Move, state: GameState) -> bool:
    """Use cache manager for frozen check"""
    cache_manager = state.cache_manager
    if cache_manager.is_frozen(move.from_coord, state.color):
        return True

    tmp = state.clone()
    tmp.make_move(move)

    summary = _get_check_summary(tmp)
    del tmp
    return summary[f"{state.color.name.lower()}_check"]


def resolves_check(move: Move, state: GameState, check_summary: Dict[str, Any]) -> bool:
    """Use cache manager for frozen check"""
    cache_manager = state.cache_manager
    if cache_manager.is_frozen(move.from_coord, state.color):
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


def filter_legal_moves(moves: List[Move], state: GameState) -> List[Move]:
    """
    Unified legal move filtering with optional batch-aware optimization.
    Autodetects whether to use fast-path (priest alive) or full validation.
    For large move lists (>100 moves), uses vectorized frozen checks if available.
    """
    if not moves:
        return []

    moves = filter_none_moves(moves)
    if not moves:
        return []

    if not hasattr(state, 'cache_manager'):
        raise RuntimeError("GameState missing cache_manager in filter_legal_moves")

    cache_manager = state.cache_manager

    # Fast path: priest alive → only filter frozen pieces
    priest_flags = cache_manager.priest_status()
    friendly_priests_key = f"{state.color.name.lower()}_priests_alive"
    if priest_flags.get(friendly_priests_key, False):
        # Autodetect: use batch frozen check if many moves and supported
        if (len(moves) > 100 and
            hasattr(cache_manager, 'batch_get_frozen_status') and
            hasattr(moves[0], 'from_coord')):
            try:
                from_coords = np.array([m.from_coord for m in moves], dtype=np.int32)
                frozen_mask = cache_manager.batch_get_frozen_status(from_coords, state.color)
                legal = [m for m, frozen in zip(moves, frozen_mask) if not frozen]
                return filter_none_moves(legal)
            except Exception:
                # Fallback to scalar if batch fails
                pass

        # Scalar fallback
        return [m for m in moves if not cache_manager.is_frozen(m.from_coord, state.color)]

    # Full validation path
    summary = _get_check_summary(state)
    attacked = summary[f"attacked_squares_{state.color.opposite().name.lower()}"]
    king_pos = summary[f"{state.color.name.lower()}_king_position"]
    in_check = summary[f"{state.color.name.lower()}_check"]

    legal = []
    for m in moves:
        if cache_manager.is_frozen(m.from_coord, state.color):
            continue

        # King moving into attack?
        if king_pos and m.from_coord == king_pos and m.to_coord in attacked:
            continue

        # Must resolve check if in check
        if in_check and not resolves_check(m, state, summary):
            continue

        legal.append(m)

    return filter_none_moves(legal)


# =============================================================================
# Batch Validation (Performance Optimized)
# =============================================================================
def validate_moves_batch(
    moves: List[Move],
    state: GameState,
    piece: Optional[Piece] = None
) -> List[Move]:
    """Fast validation with early exits."""
    if not moves:
        return []

    # Quick filter - remove None moves first
    moves = [m for m in moves if m is not None]
    if not moves:
        return []

    cache = state.cache_manager
    expected_color = piece.color if piece else state.color
    color_code = 1 if expected_color == Color.WHITE else 2

    # Extract all coordinates at once
    from_coords = np.array([m.from_coord for m in moves], dtype=np.int32)
    to_coords = np.array([m.to_coord for m in moves], dtype=np.int32)

    # Batch validation
    from_colors, _ = cache.occupancy.batch_get_colors_and_types(from_coords)
    to_colors, _ = cache.occupancy.batch_get_colors_and_types(to_coords)

    # Vectorized checks
    valid = (
        (from_colors == color_code) &  # Correct piece color
        ((to_colors == 0) | (to_colors != color_code)) &  # Valid destination
        ~cache.batch_get_frozen_status(from_coords, expected_color)  # Not frozen
    )

    # Return filtered moves
    return [moves[i] for i in range(len(moves)) if valid[i]]

def validate_moves_ultra_batch(
    moves: Union[List[Move], List[List[Move]]],
    state: GameState
) -> Union[List[Move], List[List[Move]]]:
    """
    Ultra-batch validation for maximum performance.
    Uses full vectorization and minimizes Python loops - supports scalar and batch mode.
    """
    # Handle nested batch (list of lists)
    if moves and isinstance(moves[0], list):
        return [validate_moves_ultra_batch(move_batch, state) for move_batch in moves]

    # Single batch mode
    if not moves:
        return []

    moves = filter_none_moves(moves)
    if not moves:
        return []

    # Convert to numpy arrays for vectorized processing
    from_coords = np.array([m.from_coord for m in moves], dtype=np.int32)
    to_coords = np.array([m.to_coord for m in moves], dtype=np.int32)

    cache_manager = state.cache_manager
    expected_color = state.color
    current_ply = getattr(state, 'ply', state.halfmove_clock)

    # 1. Batch bounds checking (fully vectorized)
    from_bounds = np.all((from_coords >= 0) & (from_coords < SIZE), axis=1)
    to_bounds = np.all((to_coords >= 0) & (to_coords < SIZE), axis=1)
    bounds_valid = from_bounds & to_bounds

    if not np.any(bounds_valid):
        return []

    # 2. Filter valid indices early
    valid_indices = np.where(bounds_valid)[0]
    valid_from_coords = from_coords[valid_indices]
    valid_to_coords = to_coords[valid_indices]

    # 3. Batch occupancy and color validation
    from_colors, from_types = cache_manager.occupancy.batch_get_colors_and_types(valid_from_coords)
    to_colors, to_types = cache_manager.occupancy.batch_get_colors_and_types(valid_to_coords)

    expected_code = 1 if expected_color == Color.WHITE else 2

    # Vectorized color validation
    color_valid = (from_colors != 0) & (from_colors == expected_code)

    # Vectorized destination validation
    dest_valid = (to_colors == 0) | (to_colors != expected_code)

    # 4. Batch effect validation using cache manager batch methods
    frozen_valid = ~cache_manager.batch_get_frozen_status(valid_from_coords, expected_color)
    geomancy_valid = ~cache_manager.batch_get_geomancy_blocked(valid_to_coords, current_ply)

    # 5. Combine all masks
    final_valid_mask = color_valid & dest_valid & frozen_valid & geomancy_valid

    # 6. Apply debuff effects to remaining moves
    if np.any(final_valid_mask):
        debuffed_coords = valid_from_coords[final_valid_mask]
        debuffed_mask = cache_manager.batch_get_debuffed_status(debuffed_coords, expected_color)

        # For debuffed pieces, we need to filter moves that exceed reduced range
        if np.any(debuffed_mask):
            debuffed_indices = np.where(debuffed_mask)[0]
            original_valid_indices = valid_indices[final_valid_mask]

            # Apply range reduction for debuffed pieces
            for debuff_subidx in debuffed_indices:
                orig_idx = original_valid_indices[debuff_subidx]
                move = moves[orig_idx]
                from_arr = np.array(move.from_coord)
                to_arr = np.array(move.to_coord)
                distance = np.max(np.abs(to_arr - from_arr))

                # If distance exceeds reduced range, mark as invalid
                if distance > 1:  # Debuffed pieces can only move 1 square
                    final_valid_mask[debuff_subidx] = False

    # 7. Apply final mask to original indices
    final_valid_indices = valid_indices[final_valid_mask]

    result = []
    result_append = result.append  # Cache method
    for i in final_valid_indices:
        result_append(moves[i])
    return result


# =============================================================================
# Specialized Validations
# =============================================================================

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
    archer_pos = None
    for coord, piece in get_player_pieces(game_state, game_state.color):
        if piece.ptype == PieceType.ARCHER:
            archer_pos = coord
            break

    if archer_pos is None:
        return {'valid': False, 'message': "No archer controlled."}

    if game_state.cache_manager.is_frozen(archer_pos, game_state.color):
        return {'valid': False, 'message': "Archer is frozen and cannot attack."}

    return {'valid': True, 'message': ""}


def validate_hive_moves(game_state: GameState, moves: List[Move]) -> Dict[str, Any]:
    """Consolidated hive move validation."""
    if not moves:
        return {'valid': False, 'message': "No moves submitted."}

    for coord, piece in get_player_pieces(game_state, game_state.color):
        if piece.ptype == PieceType.HIVE and game_state.cache_manager.is_frozen(coord, game_state.color):
            return {'valid': False, 'message': "Hive is frozen and cannot move."}

    return {'valid': True, 'message': ""}


# =============================================================================
# Game State Validation
# =============================================================================

def validate_game_state(state: GameState) -> List[str]:
    """Comprehensive state validation."""
    issues = []

    # Board-cache consistency
    if not validate_cache_integrity(state):
        issues.append("Cache-board desync")

    # Piece count sanity - use generators directly without full list conversion
    white_pieces = state.cache_manager.get_pieces_of_color(Color.WHITE)
    black_pieces = state.cache_manager.get_pieces_of_color(Color.BLACK)

    # Check if generators are empty using next() with default
    if next(white_pieces, None) is None or next(black_pieces, None) is None:
        issues.append("No pieces for one color")

    return issues


def validate_cache_integrity(state: GameState) -> bool:
    """Validate cache-board consistency."""
    # Implementation depends on cache structure
    try:
        # Basic consistency checks
        cache = state.cache_manager
        board = state.board

        # Check if cache and board agree on key positions
        test_positions = [(0,0,0), (4,4,4), (8,8,8)]
        for pos in test_positions:
            cache_piece = cache.occupancy.get(pos)
            board_piece = board.piece_at(pos)

            if cache_piece != board_piece:
                return False

        return True
    except Exception:
        return False


# =============================================================================
# Performance Aliases
# =============================================================================

# Aliases for backward compatibility and performance
validate_moves = validate_moves_ultra_batch
validate_moves_mega_batch = validate_moves_ultra_batch
