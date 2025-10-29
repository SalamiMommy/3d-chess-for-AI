# game3d/common/move_utils.py - ENHANCED VECTORIZED VERSION
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Set, Optional, Union, TYPE_CHECKING

from game3d.common.constants import SIZE, VOLUME
from game3d.common.coord_utils import in_bounds_vectorised, filter_valid_coords, add_coords, reconstruct_path, get_path_squares
from game3d.common.piece_utils import get_player_pieces
from game3d.common.enums import Color, PieceType
from game3d.pieces.piece import Piece
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

Coord = Tuple[int, int, int]

def extract_directions_and_steps_vectorized(start, to_coords):
    # Always use NumPy path
    to_coords_np = np.asarray(to_coords, dtype=np.int8)
    start_arr = np.asarray(start, dtype=np.int8)
    deltas = to_coords_np - start_arr
    abs_deltas = np.abs(deltas)
    norms = np.max(abs_deltas, axis=1, keepdims=True)
    norms_no_zero = np.where(norms == 0, 1, norms)
    unit_dirs = (deltas // norms_no_zero).astype(np.int8)
    uniq_dirs = np.unique(unit_dirs, axis=0) if len(unit_dirs) > 0 else unit_dirs
    max_steps = int(np.max(norms)) if len(norms) > 0 else 0
    return uniq_dirs, max_steps

def extract_directions_and_steps(to_coords: np.ndarray, start: Union[Coord, np.ndarray]) -> Tuple[np.ndarray, int]:
    return extract_directions_and_steps_vectorized(start, to_coords)

def rebuild_moves_from_directions(
    starts: Union[np.ndarray, List[Coord]],
    directions_batch: Union[List[np.ndarray], List[List[np.ndarray]]],
    max_steps_batch: Union[List[int], List[List[int]]],
    capture_sets: Union[List[Set[Coord]], List[List[Set[Coord]]]]
) -> Union[List[Move], List[List[Move]]]:
    """Batch rebuild moves from directions for multiple pieces - supports scalar and batch mode."""
    from game3d.movement.movepiece import Move

    # Handle scalar input
    if not isinstance(starts, (list, np.ndarray)) or (isinstance(starts, np.ndarray) and starts.ndim == 1):
        # Scalar mode
        if not isinstance(directions_batch, list) or not directions_batch:
            return []

        directions = directions_batch[0] if isinstance(directions_batch[0], (list, np.ndarray)) else directions_batch
        max_steps = max_steps_batch[0] if isinstance(max_steps_batch, list) else max_steps_batch
        capture_set = capture_sets[0] if isinstance(capture_sets, list) else capture_sets

        if len(directions) == 0 or max_steps <= 0:
            return []

        if isinstance(starts, np.ndarray):
            sx, sy, sz = starts.tolist()
        else:
            sx, sy, sz = starts

        rebuilt = []
        capture_frozen = frozenset(capture_set)

        for dx, dy, dz in directions:
            for step in range(1, max_steps + 1):
                to = (sx + step * dx, sy + step * dy, sz + step * dz)
                if all(0 <= c < SIZE for c in to):
                    rebuilt.append(Move.create_simple((sx, sy, sz), to, to in capture_frozen))

        return rebuilt
    else:
        # Batch mode
        all_moves = []

        # Convert to list for consistent processing
        if isinstance(starts, np.ndarray):
            starts_list = [tuple(starts[i].tolist()) for i in range(starts.shape[0])]
        else:
            starts_list = starts

        for i, (start, directions, max_steps, capture_set) in enumerate(
            zip(starts_list, directions_batch, max_steps_batch, capture_sets)
        ):
            if len(directions) == 0 or max_steps <= 0:
                all_moves.append([])
                continue

            sx, sy, sz = start
            rebuilt = []
            capture_frozen = frozenset(capture_set)

            for dx, dy, dz in directions:
                for step in range(1, max_steps + 1):
                    to = (sx + step * dx, sy + step * dy, sz + step * dz)
                    if all(0 <= c < SIZE for c in to):
                        rebuilt.append(Move.create_simple(start, to, to in capture_frozen))

            all_moves.append(rebuilt)

        return all_moves

def extend_move_range(move: Union[Move, List[Move]], start: Union[Coord, List[Coord]], max_steps: Union[int, List[int]] = 1, debuffed: Union[bool, List[bool]] = False) -> Union[List[Move], List[List[Move]]]:
    """Extend move range for buffed/debuffed pieces - supports scalar and batch mode."""
    from game3d.movement.movepiece import Move

    # Handle batch mode
    if isinstance(move, list):
        # Batch mode
        if isinstance(start, list) and isinstance(max_steps, list) and isinstance(debuffed, list):
            # Full batch with all parameters as lists
            results = []
            for m, s, ms, d in zip(move, start, max_steps, debuffed):
                results.append(extend_move_range(m, s, ms, d))
            return results
        else:
            # Mixed batch - use scalar parameters for all
            results = []
            for m in move:
                results.append(extend_move_range(m, start, max_steps, debuffed))
            return results

    # Scalar mode
    direction = tuple((b - a) for a, b in zip(start, move.to_coord))
    norm = max(abs(d) for d in direction) if direction else 0
    if norm == 0:
        return [move]

    unit_dir = tuple(d // norm for d in direction)
    extended_moves = [move]

    # Precompute bounds check values
    for step in range(1, max_steps + 1):
        next_step = tuple(a + step * b for a, b in zip(move.to_coord, unit_dir))
        if all(0 <= c < SIZE for c in next_step):
            extended_moves.append(Move.create_simple(start, next_step, is_capture=move.is_capture, debuffed=debuffed))
        else:
            break
    return extended_moves

def prepare_batch_data(state: "GameState") -> Tuple[List[Coord], List[PieceType], List[int]]:
    """
    Prepare coords, types, debuffed indices for batch dispatch.
    FIXED: Properly handles Piece objects.
    """
    coords, types, debuffed = [], [], []
    cache = state.cache_manager

    for idx, (coord, piece) in enumerate(get_player_pieces(state, state.color)):
        if not isinstance(piece, Piece):
            # Skip logging in optimized version
            continue

        coords.append(coord)
        types.append(piece.ptype)

        if cache.is_movement_debuffed(coord, state.color) and piece.ptype != PieceType.PAWN:
            debuffed.append(idx)

    return coords, types, debuffed

def filter_none_moves(moves: Union[List["Move"], List[List["Move"]]]) -> Union[List["Move"], List[List["Move"]]]:
    """Filter None values from move lists - handles both single and batch."""
    if not moves:
        return moves

    # Check if it's a batch (list of lists)
    if isinstance(moves[0], list):
        return [
            [m for m in move_list if m is not None and hasattr(m, 'from_coord') and hasattr(m, 'to_coord')]
            for move_list in moves
        ]
    else:
        # Single list
        return [m for m in moves if m is not None and hasattr(m, 'from_coord') and hasattr(m, 'to_coord')]

def apply_special_effects(
    game_state: "GameState",
    moves: List["Move"],
    moving_pieces: List["Piece"],
    captured_pieces: List[Optional["Piece"]]
) -> Tuple[List[List[Tuple]], List[List[Tuple]], List[bool]]:
    """Optimized special effects with batched operations."""
    if not moves:
        return [], [], []

    cache = game_state.cache_manager

    # Group moves by effect type to process in batches
    bomb_moves = []
    trail_moves = []

    for i, (move, piece) in enumerate(zip(moves, moving_pieces)):
        if piece.ptype == PieceType.BOMB:
            bomb_moves.append((i, move, piece, captured_pieces[i]))
        if piece.color in cache.trailblaze_cache._active_colors:
            trail_moves.append((i, move, piece))

    # Process each effect type in batch
    all_removed = [[] for _ in moves]
    all_moved = [[] for _ in moves]
    all_detonate = [False] * len(moves)

    # Batch process bombs
    for i, move, piece, captured in bomb_moves:
        removed = []
        detonate = apply_bomb_effects(
            game_state.board, cache, move, piece, captured, removed, False
        )
        all_removed[i] = removed
        all_detonate[i] = detonate

    # Batch process trailblaze
    for i, move, piece in trail_moves:
        apply_trailblaze_effect(
            game_state.board, cache, move, piece.color, all_removed[i]
        )

    return all_removed, all_moved, all_detonate

def create_enriched_move(game_state, move, removed_pieces, moved_pieces,
                        is_self_detonate, captured_piece=None):
    """Unified enriched move creation - scalar mode only."""
    from game3d.movement.movepiece import Move

    is_capture_flag = move.is_capture or (captured_piece is not None)
    is_pawn = (captured_piece and captured_piece.ptype == PieceType.PAWN) if captured_piece else False

    core_move = Move(
        from_coord=move.from_coord,
        to_coord=move.to_coord,
        is_capture=is_capture_flag,
        captured_piece=captured_piece,
        is_promotion=getattr(move, 'is_promotion', False),
        promotion_type=getattr(move, 'promotion_type', None),
        is_en_passant=False,
        is_castle=False,
    )

    # Import here to avoid circular imports
    from game3d.movement.movepiece import EnrichedMove
    return EnrichedMove(
        core_move=core_move,
        removed_pieces=removed_pieces,
        moved_pieces=moved_pieces,
        is_self_detonate=is_self_detonate,
        is_pawn_move=is_pawn,
        is_capture=is_capture_flag
    )

def apply_hole_effects(
    board: "Board",
    cache: 'OptimizedCacheManager',
    color: Union[Color, List[Color]],
    moved_pieces: Union[List[Tuple[Tuple[int, int, int], Tuple[int, int, int], Piece]], List[List[Tuple[Tuple[int, int, int], Tuple[int, int, int], Piece]]]]
) -> None:
    """Batch apply black-hole pulls & white-hole pushes - supports scalar and batch mode."""
    # Handle batch mode
    if isinstance(color, list):
        for c in color:
            apply_hole_effects(board, cache, c, moved_pieces)
        return

    # Scalar mode
    enemy_color = color.opposite()

    # Get maps directly
    pull_map = cache.black_hole_pull_map(color)
    push_map = cache.white_hole_push_map(color)

    # Combine operations and process in batch
    all_updates = []
    for from_sq, to_sq in {**pull_map, **push_map}.items():
        piece = cache.occupancy.get(from_sq)
        if piece and piece.color == enemy_color:
            all_updates.append((from_sq, to_sq, piece))

    # Apply all updates at once
    if all_updates:
        cache.batch_set_pieces([
            (from_sq, None) for from_sq, _, _ in all_updates
        ] + [
            (to_sq, piece) for _, to_sq, piece in all_updates
        ])

        moved_pieces.extend(all_updates)

def apply_bomb_effects(
    board: 'Board',
    cache: 'OptimizedCacheManager',
    mv: Union['Move', List['Move']],
    moving_piece: Union['Piece', List['Piece']],
    captured_piece: Union[Optional['Piece'], List[Optional['Piece']]],
    removed_pieces: Union[List[Tuple[Tuple[int, int, int], Piece]], List[List[Tuple[Tuple[int, int, int], Piece]]]],
    is_self_detonate: Union[bool, List[bool]]
) -> Union[bool, List[bool]]:
    """Apply bomb detonation effects efficiently - supports scalar and batch mode."""
    # Handle batch mode
    if isinstance(mv, list):
        results = []
        for i, (single_mv, single_piece, single_captured) in enumerate(zip(mv, moving_piece, captured_piece)):
            single_removed = removed_pieces[i] if isinstance(removed_pieces, list) and i < len(removed_pieces) else []
            single_detonate = is_self_detonate[i] if isinstance(is_self_detonate, list) and i < len(is_self_detonate) else False
            result = apply_bomb_effects(board, cache, single_mv, single_piece, single_captured, single_removed, single_detonate)
            results.append(result)
        return results

    # Scalar mode
    from game3d.common.enums import PieceType

    enemy_color = moving_piece.color.opposite()

    # Handle captured bomb explosion
    if captured_piece and captured_piece.ptype == PieceType.BOMB and captured_piece.color == enemy_color:
        for sq in detonate(board, mv.to_coord, moving_piece.color):
            piece = cache.occupancy.get(sq)
            if piece:
                removed_pieces.append((sq, piece))
            board.set_piece(sq, None)

    # Handle self-detonation
    if (moving_piece.ptype == PieceType.BOMB and
        getattr(mv, 'is_self_detonate', False)):
        for sq in detonate(board, mv.to_coord, moving_piece.color):
            piece = cache.occupancy.get(sq)
            if piece:
                removed_pieces.append((sq, piece))
            board.set_piece(sq, None)
        return True

    return False

def apply_trailblaze_effect(
    board: 'Board',
    cache: 'OptimizedCacheManager',
    mv: Union['Move', List['Move']],
    color: Union['Color', List['Color']],
    removed_pieces: Union[List[Tuple[Tuple[int, int, int], Piece]], List[List[Tuple[Tuple[int, int, int], Piece]]]]
) -> None:
    """Apply trailblaze effect efficiently - supports scalar and batch mode."""
    # Handle batch mode
    if isinstance(mv, list):
        for i, (single_mv, single_color) in enumerate(zip(mv, color)):
            single_removed = removed_pieces[i] if isinstance(removed_pieces, list) and i < len(removed_pieces) else []
            apply_trailblaze_effect(board, cache, single_mv, single_color, single_removed)
        return

    # Scalar mode
    from game3d.common.enums import PieceType

    # Get enemy sliding path
    enemy_color = color.opposite()
    enemy_slid = extract_enemy_slid_path(mv)
    squares_to_check = set(enemy_slid) | {mv.to_coord}

    for sq in squares_to_check:
        if cache.trailblaze_cache.increment_counter(sq, enemy_color, board):
            victim = cache.occupancy.get(sq)
            if victim:
                # Kings only removed if no priest alive
                if victim.ptype == PieceType.KING:
                    if not _any_priest_alive(board, victim.color):
                        removed_pieces.append((sq, victim))
                        board.set_piece(sq, None)
                else:
                    removed_pieces.append((sq, victim))
                    board.set_piece(sq, None)

def reconstruct_trailblazer_path(
    from_coord: Union[Tuple[int, int, int], np.ndarray],
    to_coord: Union[Tuple[int, int, int], np.ndarray],
    include_start: bool = False,
    include_end: bool = True
) -> Union[Set[Tuple[int, int, int]], List[Set[Tuple[int, int, int]]]]:
    """Reconstruct the path of a trailblazer move - supports scalar and batch mode."""
    if isinstance(from_coord, np.ndarray) and from_coord.ndim > 1:
        # Batch mode
        results = []
        for i in range(from_coord.shape[0]):
            single_from = tuple(from_coord[i].tolist())
            single_to = tuple(to_coord[i].tolist()) if isinstance(to_coord, np.ndarray) and to_coord.ndim > 1 else to_coord
            results.append(reconstruct_trailblazer_path(single_from, single_to, include_start, include_end))
        return results

    # Scalar mode
    return reconstruct_path(from_coord, to_coord, include_start=include_start, include_end=include_end, as_set=True)

def extract_enemy_slid_path(mv: Union['Move', List['Move']]) -> Union[List[Tuple[int, int, int]], List[List[Tuple[int, int, int]]]]:
    """Extract enemy sliding path for trailblaze effect - supports scalar and batch mode."""
    # Handle batch mode
    if isinstance(mv, list):
        return [extract_enemy_slid_path(single_mv) for single_mv in mv]

    # Scalar mode
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
    target: Union[Tuple[int, int, int], np.ndarray],
    halfmove_clock: Union[int, np.ndarray]
) -> None:
    """Block a square via the geomancy cache - supports scalar and batch mode."""
    if isinstance(target, np.ndarray) and target.ndim > 1:
        # Batch mode
        for i in range(target.shape[0]):
            single_target = tuple(target[i].tolist())
            single_clock = halfmove_clock[i] if isinstance(halfmove_clock, np.ndarray) and halfmove_clock.size > 1 else halfmove_clock
            apply_geomancy_effect(board, cache, single_target, single_clock)
    else:
        # Scalar mode
        cache.block_square(target, halfmove_clock)

def apply_swap_move(board: 'Board', mv: Union['Move', List['Move']]) -> None:
    """Optimized swap move application - supports scalar and batch mode."""
    if isinstance(mv, list):
        for single_mv in mv:
            apply_swap_move(board, single_mv)
        return

    # Scalar mode
    cache = board.cache_manager
    from_piece = cache.occupancy.get(mv.from_coord)
    to_piece = cache.occupancy.get(mv.to_coord)

    if from_piece:
        board.set_piece(mv.to_coord, from_piece)
    else:
        board.set_piece(mv.to_coord, None)

    if to_piece:
        board.set_piece(mv.from_coord, to_piece)
    else:
        board.set_piece(mv.from_coord, None)

def apply_promotion_move(board: 'Board', mv: Union['Move', List['Move']], piece: Union['Piece', List['Piece']]) -> None:
    """Replace pawn with promoted piece - supports scalar and batch mode."""
    from game3d.pieces.piece import Piece

    if isinstance(mv, list):
        for i, single_mv in enumerate(mv):
            single_piece = piece[i] if isinstance(piece, list) else piece
            apply_promotion_move(board, single_mv, single_piece)
        return

    # Scalar mode
    promoted = Piece(piece.color, mv.promotion_type)
    board.set_piece(mv.from_coord, None)
    board.set_piece(mv.to_coord, promoted)

def detonate(board: 'Board', center: Union[Coord, np.ndarray], friendly_color: Union[Color, np.ndarray]) -> Union[Set[Coord], List[Set[Coord]]]:
    """
    Get all squares in Manhattan distance 1 from center.
    Returns coords of enemy pieces that would be destroyed - supports scalar and batch mode.
    """
    from game3d.common.coord_utils import manhattan_distance, in_bounds

    # Handle batch mode
    if isinstance(center, np.ndarray) and center.ndim > 1:
        results = []
        for i in range(center.shape[0]):
            single_center = tuple(center[i].tolist())
            single_color = friendly_color[i] if isinstance(friendly_color, np.ndarray) and friendly_color.size > 1 else friendly_color
            results.append(detonate(board, single_center, single_color))
        return results

    # Scalar mode
    destroyed = set()
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == dy == dz == 0:
                    continue

                coord = (center[0] + dx, center[1] + dy, center[2] + dz)
                if not in_bounds(coord):
                    continue

                piece = board.cache_manager.occupancy.get(coord)
                if piece and piece.color != friendly_color:
                    destroyed.add(coord)

    return destroyed

def get_processing_mode(item_count: int, threshold: Optional[int] = None, breakpoints: Tuple[int, int] = (30, 100)) -> str:
    """
    Autodetect or force processing mode based on count/threshold.
    - breakpoints: (batch_threshold, mega_threshold)
    Returns: 'scalar', 'batch', or 'mega'
    """
    low, high = breakpoints
    if threshold is None:
        if item_count > high:
            return 'mega'
        elif item_count > low:
            return 'batch'
        else:
            return 'scalar'
    else:
        if threshold == 0:
            return 'mega'
        elif threshold <= low:
            return 'scalar'
        elif threshold <= high:
            return 'batch'
        else:
            return 'mega'

def create_filtered_moves_batch(
    from_coord: Tuple[int, int, int],
    to_coords: np.ndarray,
    captures: np.ndarray,
    state: 'GameState',
    apply_effects: bool = True
) -> List[Move]:
    """
    Filter coords/captures by bounds/effects, then create batch.
    """
    valid_mask = in_bounds_vectorised(to_coords)  # From coord_utils
    if apply_effects:
        cache = state.cache_manager
        geomancy_mask = ~cache.batch_get_geomancy_blocked(to_coords, state.ply)
        valid_mask &= geomancy_mask
        # Add debuff distance filter if needed
    filtered_to = to_coords[valid_mask]
    filtered_caps = captures[valid_mask]
    return Move.create_batch(from_coord, filtered_to, filtered_caps)
