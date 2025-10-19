# game3d/game/move_utils.py
"""Shared move utilities to break circular dependencies between turnmove and moveeffects."""
from __future__ import annotations
from typing import List, Tuple, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from game3d.board.board import Board
    from game3d.pieces.piece import Piece
    from game3d.movement.movepiece import Move
    from game3d.common.enums import Color, PieceType
    from game3d.cache.manager import OptimizedCacheManager

from game3d.pieces.pieces.bomb import detonate
from game3d.attacks.check import _any_priest_alive
from game3d.common.common import reconstruct_path

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
        piece = cache.piece_cache.get(from_sq)
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
            piece = cache.piece_cache.get(sq)
            if piece:
                removed_pieces.append((sq, piece))
            board.set_piece(sq, None)

    # Handle self-detonation
    if (moving_piece.ptype == PieceType.BOMB and
        getattr(mv, 'is_self_detonate', False)):
        for sq in detonate(board, mv.to_coord, moving_piece.color):
            # CORRECTED: Access through manager
            piece = cache.piece_cache.get(sq)
            if piece:
                removed_pieces.append((sq, piece))
            board.set_piece(sq, None)
        return True

    return False

def apply_trailblaze_effect(
    board: 'Board',
    cache: 'OptimizedCacheManager',  # CORRECTED: Type hint shows it's the manager
    mv: 'Move',
    color: 'Color',
    removed_pieces: List[Tuple[Tuple[int, int, int], Piece]]
) -> None:
    """Apply trailblaze effect efficiently."""
    from game3d.common.enums import PieceType

    # CORRECTED: Access effect cache through manager
    trail_cache = cache.effects._effect_caches["trailblaze"]
    enemy_color = color.opposite()
    enemy_slid = extract_enemy_slid_path(mv)
    squares_to_check = set(enemy_slid) | {mv.to_coord}

    for sq in squares_to_check:
        if trail_cache.increment_counter(sq, enemy_color, board):
            # CORRECTED: Access through manager
            victim = cache.piece_cache.get(sq)
            if victim:
                # Kings only removed if no priest alive
                if victim.ptype == PieceType.KING:
                    if not _any_priest_alive(board, enemy_color):
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
    cache: 'OptimizedCacheManager',  # CORRECTED: Type hint shows it's the manager
    target: Tuple[int, int, int],
    halfmove_clock: int
) -> None:
    """Block a square via the geomancy cache."""
    # CORRECTED: Use manager method
    cache.block_square(target, halfmove_clock)

def apply_swap_move(board: 'Board', mv: 'Move') -> None:
    # CORRECT - cleaner through piece_cache
    cache = board.cache_manager
    target_piece = cache.piece_cache.get(mv.to_coord)
    board.set_piece(mv.to_coord, cache.piece_cache.get(mv.from_coord))
    board.set_piece(mv.from_coord, target_piece)

def apply_promotion_move(board: 'Board', mv: 'Move', piece: 'Piece') -> None:
    """Replace pawn with promoted piece."""
    from game3d.pieces.piece import Piece
    promoted = Piece(piece.color, mv.promotion_type)
    board.set_piece(mv.from_coord, None)
    board.set_piece(mv.to_coord, promoted)
