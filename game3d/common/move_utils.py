# game3d/common/move_utils.py - OPTIMIZED VERSION
# ------------------------------------------------------------------
# Move-related utilities with consolidated validation
# ------------------------------------------------------------------
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Set, Optional, TYPE_CHECKING

from game3d.common.constants import SIZE, VOLUME
from game3d.common.coord_utils import in_bounds_vectorised, filter_valid_coords, add_coords
from game3d.common.piece_utils import get_player_pieces
from game3d.common.enums import Color, PieceType
from game3d.pieces.piece import Piece
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.board.board import Board
    from game3d.pieces.piece import Piece
    from game3d.movement.movepiece import Move
    from game3d.common.enums import Color, PieceType
    from game3d.cache.manager import OptimizedCacheManager

from game3d.pieces.pieces.bomb import detonate
from game3d.attacks.check import _any_priest_alive
from game3d.common.coord_utils import reconstruct_path
if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

Coord = Tuple[int, int, int]

def extract_directions_and_steps_vectorized(start: Coord, to_coords: np.ndarray) -> Tuple[np.ndarray, int]:
    if to_coords.size == 0:                      # empty request
        return np.empty((0, 3), dtype=np.int8), 0

    start_arr = np.asarray(start, dtype=np.int32)
    deltas = to_coords.astype(np.int32) - start_arr

    # Chebyshev norm along each row
    norms = np.max(np.abs(deltas), axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)       # avoid div-by-zero

    unit_dirs = (deltas // norms).astype(np.int8)
    uniq_dirs = np.unique(unit_dirs, axis=0)

    max_steps = int(norms.max())
    return uniq_dirs, max_steps

def extract_directions_and_steps(to_coords: np.ndarray, start: Coord) -> Tuple[np.ndarray, int]:
    return extract_directions_and_steps_vectorized(start, to_coords)

def rebuild_moves_from_directions(start: Coord, directions: np.ndarray, max_steps: int, capture_set: Set[Coord]) -> List[Move]:
    """Batch-rebuild moves from directions and steps."""
    from game3d.movement.movepiece import Move
    if len(directions) == 0 or max_steps <= 0:
        return []
    sx, sy, sz = start
    rebuilt = []
    for dx, dy, dz in directions:
        for step in range(1, max_steps + 1):
            to = (sx + step * dx, sy + step * dy, sz + step * dz)
            rebuilt.append(Move.create_simple(start, to, to in capture_set))
    return rebuilt

def extend_move_range(move: Move, start: Coord, max_steps: int = 1, debuffed: bool = False) -> List[Move]:
    """Extend move range for buffed/debuffed pieces."""
    direction = tuple((b - a) for a, b in zip(start, move.to_coord))
    norm = max(abs(d) for d in direction) if direction else 0
    if norm == 0:
        return [move]
    unit_dir = tuple(d // norm for d in direction)
    extended_moves = [move]
    for step in range(1, max_steps + 1):
        next_step = tuple(a + step * b for a, b in zip(move.to_coord, unit_dir))
        if all(0 <= c < SIZE for c in next_step):
            extended_moves.append(Move.create_simple(start, next_step, is_capture=move.is_capture, debuffed=debuffed))
        else:
            break
    return extended_moves

def validate_moves_batch(
    moves: List["Move"],
    state: "GameState",
    piece: Optional[Piece] = None
) -> List["Move"]:
    """
    Consolidated single-pass validation of moves.
    Combines bounds checking, occupancy validation, and effect filtering.
    """
    if not moves:
        return []

    # DEFENSIVE: Filter out None moves immediately
    moves = filter_none_moves(moves)
    if not moves:
        return []

    # Extract coordinates for vectorized processing
    from_coords = np.array([m.from_coord for m in moves])
    to_coords = np.array([m.to_coord for m in moves])

    # Single-pass bounds validation
    from_valid = in_bounds_vectorised(from_coords)
    to_valid = in_bounds_vectorised(to_coords)
    bounds_valid = from_valid & to_valid

    if not np.any(bounds_valid):
        return []

    # Get piece info for validation
    if piece is not None:
        expected_color = piece.color
    else:
        expected_color = state.color

    # Single-pass occupancy and effect validation
    valid_moves = []
    cache_manager = state.cache_manager

    for i, move in enumerate(moves):
        if not bounds_valid[i]:
            continue

        # Check piece exists and is correct color
        from_piece = cache_manager.occupancy.get(move.from_coord)
        if not from_piece or from_piece.color != expected_color:
            continue

        # Check frozen status
        if cache_manager.is_frozen(move.from_coord, expected_color):
            continue

        # Check destination is not friendly
        to_piece = cache_manager.occupancy.get(move.to_coord)
        if to_piece and to_piece.color == expected_color:
            continue

        valid_moves.append(move)

    return valid_moves

# Alias for backward compatibility
validate_moves = validate_moves_batch

def prepare_batch_data(state: "GameState") -> Tuple[List[Coord], List[PieceType], List[int]]:
    """
    Prepare coords, types, debuffed indices for batch dispatch.
    FIXED: Properly handles Piece objects.
    """
    coords, types, debuffed = [], [], []
    for idx, (coord, piece) in enumerate(get_player_pieces(state, state.color)):
        if not isinstance(piece, Piece):
            print(f"[ERROR] Non-Piece in prepare_batch_data: {type(piece)}")
            continue

        coords.append(coord)
        types.append(piece.ptype)

        if state.cache_manager.is_movement_debuffed(coord, state.color) and piece.ptype != PieceType.PAWN:
            debuffed.append(idx)

    return coords, types, debuffed

def filter_none_moves(moves: List["Move"]) -> List["Move"]:
    """
    Defensive filter to remove None values from move lists.

    This is a safety measure to prevent None values from propagating through
    the move generation pipeline. If None moves are found, logs a warning.

    Args:
        moves: List of moves that may contain None values

    Returns:
        List of moves with all None values removed
    """
    if not moves:
        return []

    # Count None values for debugging
    none_count = sum(1 for m in moves if m is None)

    if none_count > 0:
        print(f"[WARNING] filter_none_moves: Filtered {none_count} None values from {len(moves)} moves")
        import traceback
        traceback.print_stack(limit=5)  # Show where the None came from

    # Filter out None values
    filtered = [m for m in moves if m is not None]

    # Extra validation: check that remaining moves have required attributes
    valid = []
    for m in filtered:
        if not hasattr(m, 'from_coord') or not hasattr(m, 'to_coord'):
            print(f"[WARNING] filter_none_moves: Move missing required attributes: {m}")
            continue
        valid.append(m)

    if len(valid) < len(filtered):
        print(f"[WARNING] filter_none_moves: Removed {len(filtered) - len(valid)} invalid move objects")

    return valid

def apply_special_effects(game_state, move, moving_piece, captured_piece):
    """Unified special effects application."""
    cache = get_cache_manager(game_state)
    removed_pieces = []
    moved_pieces = []

    # Apply all effects in sequence
    apply_hole_effects(game_state.board, cache, moving_piece.color, moved_pieces)
    is_self_detonate = apply_bomb_effects(
        game_state.board, cache, move, moving_piece,
        captured_piece, removed_pieces, False
    )
    apply_trailblaze_effect(game_state.board, cache, move, moving_piece.color, removed_pieces)

    return removed_pieces, moved_pieces, is_self_detonate

def create_enriched_move(game_state, move, removed_pieces, moved_pieces,
                        is_self_detonate, captured_piece=None):
    """Unified enriched move creation."""
    from game3d.movement.movepiece import convert_legacy_move_args

    is_capture_flag = move.is_capture or (captured_piece is not None)
    is_pawn = (captured_piece and captured_piece.ptype == PieceType.PAWN) if captured_piece else False

    core_move = convert_legacy_move_args(
        from_coord=move.from_coord,
        to_coord=move.to_coord,
        is_capture=is_capture_flag,
        captured_piece=captured_piece,
        is_promotion=getattr(move, 'is_promotion', False),
        promotion_type=getattr(move, 'promotion_type', None),
        is_en_passant=False,
        is_castle=False,
    )

    return EnrichedMove(
        core_move=core_move,
        removed_pieces=removed_pieces,
        moved_pieces=moved_pieces,
        is_self_detonate=is_self_detonate,
        is_pawn_move=is_pawn,
        is_capture=is_capture_flag
    )

def apply_hole_effects(
    board: Board,
    cache: 'OptimizedCacheManager',  # CORRECTED: Type hint shows it's the manager
    color: Color,
    moved_pieces: List[Tuple[Tuple[int, int, int], Tuple[int, int, int], Piece]],
    _is_self_detonate: bool = False,
) -> None:
    """Apply black-hole pulls & white-hole pushes."""
    enemy_color = color.opposite()

    # CORRECTED: Use manager methods
    pull_map = cache.black_hole_pull_map(color)
    push_map = cache.white_hole_push_map(color)

    combined_map = {**pull_map, **push_map}
    for from_sq, to_sq in combined_map.items():
        # CORRECTED: Access through manager
        piece = cache.occupancy.get(from_sq)
        if piece and piece.color == enemy_color:
            moved_pieces.append((from_sq, to_sq, piece))
            board.set_piece(to_sq, piece)
            board.set_piece(from_sq, None)

def apply_bomb_effects(
    board: 'Board',
    cache: 'OptimizedCacheManager',  # CORRECTED: Type hint shows it's the manager
    mv: 'Move',
    moving_piece: 'Piece',
    captured_piece: Optional['Piece'],
    removed_pieces: List[Tuple[Tuple[int, int, int], Piece]],
    is_self_detonate: bool
) -> bool:
    """Apply bomb detonation effects efficiently."""
    from game3d.common.enums import PieceType

    enemy_color = moving_piece.color.opposite()

    # Handle captured bomb explosion
    if captured_piece and captured_piece.ptype == PieceType.BOMB and captured_piece.color == enemy_color:
        for sq in detonate(board, mv.to_coord, moving_piece.color):
            # CORRECTED: Access through manager
            piece = cache.occupancy.get(sq)
            if piece:
                removed_pieces.append((sq, piece))
            board.set_piece(sq, None)

    # Handle self-detonation
    if (moving_piece.ptype == PieceType.BOMB and
        getattr(mv, 'is_self_detonate', False)):
        for sq in detonate(board, mv.to_coord, moving_piece.color):
            # CORRECTED: Access through manager
            piece = cache.occupancy.get(sq)
            if piece:
                removed_pieces.append((sq, piece))
            board.set_piece(sq, None)
        return True

    return False

def apply_trailblaze_effect(
    board: 'Board',
    cache: 'OptimizedCacheManager',  # CORRECTED: Type hint shows it's the manager
    mv: 'Move',  # FIXED: was target, now mv
    color: 'Color',
    removed_pieces: List[Tuple[Tuple[int, int, int], Piece]]
) -> None:
    """Apply trailblaze effect efficiently."""
    from game3d.common.enums import PieceType

    # CORRECTED: Access effect cache through manager
    trail_cache = cache.trailblaze_cache
    enemy_color = color.opposite()
    enemy_slid = extract_enemy_slid_path(mv)
    squares_to_check = set(enemy_slid) | {mv.to_coord}

    for sq in squares_to_check:
        if trail_cache.increment_counter(sq, enemy_color, board):
            # CORRECTED: Access through manager
            victim = cache.occupancy.get(sq)
            if victim:
                # Kings only removed if no priest alive
                if victim.ptype == PieceType.KING:
                    if not _any_priest_alive(board, victim.color):  # FIXED: Use victim.color
                        removed_pieces.append((sq, victim))
                        board.set_piece(sq, None)
                else:
                    removed_pieces.append((sq, victim))
                    board.set_piece(sq, None)

def reconstruct_trailblazer_path(
    from_coord: Tuple[int, int, int],
    to_coord: Tuple[int, int, int],
    include_start: bool = False,
    include_end: bool = True
) -> Set[Tuple[int, int, int]]:
    """Reconstruct the path of a trailblazer move."""
    return reconstruct_path(from_coord, to_coord, include_start=include_start, include_end=include_end, as_set=True)

def extract_enemy_slid_path(mv: 'Move') -> List[Tuple[int, int, int]]:
    """Extract enemy sliding path for trailblaze effect."""
    # Check if move has metadata about enemy slide
    if hasattr(mv, 'metadata') and mv.metadata:
        enemy_path = mv.metadata.get('enemy_slide_path', [])
        if enemy_path:
            return enemy_path

    # Reconstruct
    return list(reconstruct_trailblazer_path(mv.from_coord, mv.to_coord, include_start=False, include_end=False))

def apply_geomancy_effect(
    board: 'Board',
    cache: 'OptimizedCacheManager',
    target: Tuple[int, int, int],
    halfmove_clock: int
) -> None:
    """Block a square via the geomancy cache."""
    cache.block_square(target, halfmove_clock)

def apply_swap_move(board: 'Board', mv: 'Move') -> None:
    # CORRECT - cleaner through piece_cache
    cache = board.cache_manager
    target_piece = cache.occupancy.get(mv.to_coord)
    board.set_piece(mv.to_coord, cache.occupancy.get(mv.from_coord))
    board.set_piece(mv.from_coord, target_piece)

def apply_promotion_move(board: 'Board', mv: 'Move', piece: 'Piece') -> None:
    """Replace pawn with promoted piece."""
    from game3d.pieces.piece import Piece
    promoted = Piece(piece.color, mv.promotion_type)
    board.set_piece(mv.from_coord, None)
    board.set_piece(mv.to_coord, promoted)
