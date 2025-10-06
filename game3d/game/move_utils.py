# game3d/game/move_utils.py
"""Shared move utilities to break circular dependencies between turnmove and moveeffects."""
from __future__ import annotations
from typing import List, Tuple, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from game3d.board.board import Board
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.pieces.piece import Piece
    from game3d.movement.movepiece import Move
    from game3d.pieces.enums import Color, PieceType

from game3d.effects.bomb import detonate
from game3d.attacks.check import _any_priest_alive

def apply_hole_effects(board: 'Board', cache: 'OptimizedCacheManager',
                       color: 'Color', moved_pieces: List) -> None:
    """Apply black hole and white hole effects efficiently."""
    enemy_color = color.opposite()
    pull_map = cache.black_hole_pull_map(color)
    push_map = cache.white_hole_push_map(color)

    all_hole_moves = {**pull_map, **push_map}

    for from_sq, to_sq in all_hole_moves.items():
        piece = cache.piece_cache.get(from_sq)
        if piece and piece.color == enemy_color:
            moved_pieces.append((from_sq, to_sq, piece))
            board.set_piece(to_sq, piece)
            board.set_piece(from_sq, None)

def apply_bomb_effects(board: 'Board', cache: 'OptimizedCacheManager',
                       mv: 'Move', moving_piece: 'Piece',
                       captured_piece: Optional['Piece'], removed_pieces: List,
                       is_self_detonate: bool) -> bool:
    """Apply bomb detonation effects efficiently."""
    from game3d.pieces.enums import PieceType

    enemy_color = moving_piece.color.opposite()

    # Handle captured bomb explosion
    if captured_piece and captured_piece.ptype == PieceType.BOMB and captured_piece.color == enemy_color:
        for sq in detonate(board, mv.to_coord):
            piece = cache.piece_cache.get(sq)
            if piece:
                removed_pieces.append((sq, piece))
            board.set_piece(sq, None)

    # Handle self-detonation
    if (moving_piece.ptype == PieceType.BOMB and
        getattr(mv, 'is_self_detonate', False)):
        for sq in detonate(board, mv.to_coord):
            piece = cache.piece_cache.get(sq)
            if piece:
                removed_pieces.append((sq, piece))
            board.set_piece(sq, None)
        return True
    return False

def apply_trailblaze_effect(board: 'Board', cache: 'OptimizedCacheManager',
                            mv: 'Move', color: 'Color',
                            removed_pieces: List) -> None:
    """Apply trailblaze effect efficiently."""
    from game3d.pieces.enums import PieceType

    trail_cache = cache.effects["trailblaze"]
    enemy_color = color.opposite()
    enemy_slid = extract_enemy_slid_path(mv)
    squares_to_check = set(enemy_slid) | {mv.to_coord}

    for sq in squares_to_check:
        if trail_cache.increment_counter(sq, enemy_color, board):
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

def reconstruct_trailblazer_path(from_coord: Tuple[int, int, int],
                                to_coord: Tuple[int, int, int]) -> set:
    """Reconstruct the path of a trailblazer move."""
    fx, fy, fz = from_coord
    tx, ty, tz = to_coord

    if from_coord == to_coord:
        return set()

    dx = tx - fx
    dy = ty - fy
    dz = tz - fz

    step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
    step_z = 0 if dz == 0 else (1 if dz > 0 else -1)

    max_steps = max(abs(dx), abs(dy), abs(dz))

    path = set()
    x, y, z = fx, fy, fz

    for _ in range(max_steps):
        x += step_x
        y += step_y
        z += step_z
        path.add((x, y, z))
        if (x, y, z) == to_coord:
            break

    return path

def extract_enemy_slid_path(mv: 'Move') -> List[Tuple[int, int, int]]:
    """Extract enemy sliding path for trailblaze effect."""
    # Check if move has metadata about enemy slide
    if hasattr(mv, 'metadata') and mv.metadata:
        enemy_path = mv.metadata.get('enemy_slide_path', [])
        if enemy_path:
            return enemy_path

    # If no metadata, reconstruct from move coordinates
    path = []
    fx, fy, fz = mv.from_coord
    tx, ty, tz = mv.to_coord

    dx = tx - fx
    dy = ty - fy
    dz = tz - fz

    # Normalize to unit direction
    max_delta = max(abs(dx), abs(dy), abs(dz))
    if max_delta == 0:
        return []

    step_x = dx // max_delta if dx != 0 else 0
    step_y = dy // max_delta if dy != 0 else 0
    step_z = dz // max_delta if dz != 0 else 0

    # Build path (excluding start and end)
    x, y, z = fx, fy, fz
    for _ in range(max_delta - 1):
        x += step_x
        y += step_y
        z += step_z
        path.append((x, y, z))

    return path
