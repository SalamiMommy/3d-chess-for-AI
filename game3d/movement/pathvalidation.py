"""
game3d/movement/pathvalidation.py — Centralized movement path and blocking logic for 3D chess.

Handles:
- Sliding pieces (bishop, rook, queen, zigzag, etc.)
- Jumping pieces (knight, custom leapers)
- Directional movement with blocking rules
- Capture/ally collision logic
- Bounds checking

Designed to be reused by all piece move generators.
"""

from typing import List, Tuple, Optional, Callable
from game3d.game.gamestate import GameState
from game3d.movement.movepiece import Move
from game3d.common.common import in_bounds, add_coords
from game3d.game.gamestate import GameState
from game3d.movement.movepiece import Move

# ────────────────────────────────
# CORE BLOCKING DECISION HELPER
# ────────────────────────────────
def is_edge_square(x: int, y: int, z: int, board_size: int = 9) -> bool:
    """Return True if square is on the edge of a board_size³ board."""
    edge = board_size - 1
    return x in (0, edge) or y in (0, edge) or z in (0, edge)

def is_path_blocked(
    state: GameState,
    target: Tuple[int, int, int],
    allow_capture: bool = True,
    allow_self_block: bool = False,
) -> bool:
    """
    Determine if moving to `target` is blocked by a piece.

    Returns:
        True → movement is blocked (stop pathing).
        False → movement is allowed (continue or land).

    Rules:
        - Empty square → not blocked.
        - Friendly piece → blocked unless `allow_self_block`.
        - Enemy piece → blocked unless `allow_capture` (then capture allowed, but still ends path).
    """
    piece = state.board.piece_at(target)
    if piece is None:
        return False  # Open square — not blocked

    current_color = state.color

    if piece.color == current_color:
        return not allow_self_block  # Blocked by self unless allowed
    else:
        # Enemy piece
        if allow_capture:
            return True  # Not "blocked" per se, but ends path after capture
        else:
            return True  # Truly blocked — can't capture

    # Should never reach here
    return True


# ────────────────────────────────
# SLIDING MOVEMENT (ROOK, BISHOP, QUEEN, ZIGZAG, ETC.)
# ────────────────────────────────

def slide_along_directions(
    state: GameState,
    start: Tuple[int, int, int],
    directions: List[Tuple[int, int, int]],
    allow_capture: bool = True,
    allow_self_block: bool = False,
    max_steps: Optional[int] = None,
    edge_only: bool = False,  # ← NEW PARAMETER
) -> List[Move]:
    """
    Generate all legal sliding moves from `start` along each direction until blocked.

    Args:
        state: Current game state.
        start: Starting coordinate (x, y, z).
        directions: List of unit direction vectors, e.g., [(1,0,1), (0,1,0), ...].
        allow_capture: Whether capturing enemy pieces is allowed.
        allow_self_block: Whether sliding through friendly pieces is allowed (rare).
        max_steps: Optional limit on steps per direction (e.g., for limited-range pieces).

    Returns:
        List of legal Move objects.
    """
    moves = []
    current_color = state.color
    board_size = 9  # or get from state if dynamic

    for dx, dy, dz in directions:
        step = 1
        while max_steps is None or step <= max_steps:
            offset = (dx * step, dy * step, dz * step)
            target = add_coords(start, offset)

            if not in_bounds(target):
                break

            # ✅ NEW: If edge_only, stop if target is not on edge
            if edge_only and not is_edge_square(*target, board_size):
                break

            blocked = is_path_blocked(
                state,
                target,
                allow_capture=allow_capture,
                allow_self_block=allow_self_block,
            )

            target_piece = state.board.piece_at(target)

            if target_piece is not None:
                if target_piece.color != current_color and allow_capture:
                    moves.append(Move(
                        from_coord=start,
                        to_coord=target,
                        is_capture=True
                    ))
                break
            else:
                moves.append(Move(
                    from_coord=start,
                    to_coord=target,
                    is_capture=False
                ))

            step += 1

    return moves


# ────────────────────────────────
# JUMPING MOVEMENT (KNIGHT, CUSTOM LEAPERS)
# ────────────────────────────────

def jump_to_targets(
    state: GameState,
    start: Tuple[int, int, int],
    offsets: List[Tuple[int, int, int]],
    allow_capture: bool = True,
    allow_self_block: bool = False,
) -> List[Move]:
    """
    Generate legal jump moves (ignores path, lands directly on target).

    Used for knights, fairy leapers, or any piece that doesn't slide.

    Args:
        state: Current game state.
        start: Starting coordinate (x, y, z).
        offsets: List of absolute jump offsets, e.g., [(2,1,0), (1,2,-1), ...].
        allow_capture: Whether capturing enemy pieces is allowed.
        allow_self_block: Whether landing on friendly pieces is allowed.

    Returns:
        List of legal Move objects.
    """
    moves = []
    current_color = state.color

    for offset in offsets:
        target = add_coords(start, offset)

        if not in_bounds(target):
            continue

        # Check if landing is blocked
        blocked = is_path_blocked(
            state,
            target,
            allow_capture=allow_capture,
            allow_self_block=allow_self_block,
        )

        target_piece = state.board.piece_at(target)

        if target_piece is not None:
            if target_piece.color == current_color and not allow_self_block:
                continue
            if target_piece.color != current_color and not allow_capture:
                continue

        is_capture = (target_piece is not None and target_piece.color != current_color)

        # Final safety: if it's a friendly and we're not allowed to land, skip
        if target_piece is not None and target_piece.color == current_color and not allow_self_block:
            continue

        moves.append(Move(
            from_coord=start,
            to_coord=target,
            is_capture=is_capture
        ))

    return moves


# ────────────────────────────────
# UTILITY: PIECE VALIDATION AT START
# ────────────────────────────────

def validate_piece_at(
    state: GameState,
    pos: Tuple[int, int, int],
    expected_type: Optional[int] = None,
) -> bool:
    piece = state.board.piece_at(pos)
    if piece is None:
        return False
    if piece.color != state.color:          # ← changed current → color
        return False
    if expected_type is not None and piece.ptype != expected_type:
        return False
    return True
