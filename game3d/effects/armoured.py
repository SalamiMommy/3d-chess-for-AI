"""Armoured – ARMOUR pieces are immune to pawn captures."""

from __future__ import annotations
from typing import Dict, Set, Tuple, Optional, TYPE_CHECKING
from game3d.pieces.enums import Color, PieceType
from game3d.common.protocols import BoardProto

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager

def is_armour_protected(sq: Tuple[int, int, int], board: BoardProto,
                        cache_manager: Optional[OptimizedCacheManager] = None) -> bool:
    """
    Check if a square contains an ARMOUR piece, making it immune to pawn captures.
    """
    # Use cache manager if available
    if cache_manager:
        piece = cache_manager.piece_cache.get(sq)
    else:
        # Fallback to board method if cache manager not available
        piece = board.get_piece(sq)

    return piece is not None and piece.ptype == PieceType.ARMOUR


def can_pawn_capture(pawn_sq: Tuple[int, int, int], target_sq: Tuple[int, int, int],
                   board: BoardProto, cache_manager: Optional[OptimizedCacheManager] = None) -> bool:
    """
    Check if a pawn can capture at the target square, considering armour protection.
    """
    # First check if target is armoured
    if is_armour_protected(target_sq, board, cache_manager):
        return False

    # Use cache manager to get the pawn piece
    if cache_manager:
        pawn = cache_manager.piece_cache.get(pawn_sq)
    else:
        pawn = board.get_piece(pawn_sq)

    # Verify it's actually a pawn
    if pawn is None or pawn.ptype != PieceType.PAWN:
        return False

    # Use cache manager to get the target piece
    if cache_manager:
        target = cache_manager.piece_cache.get(target_sq)
    else:
        target = board.get_piece(target_sq)

    # Verify target exists and is enemy piece
    if target is None or target.color == pawn.color:
        return False

    return True


def get_armoured_squares(board: BoardProto, controller: Color,
                        cache_manager: Optional[OptimizedCacheManager] = None) -> Set[Tuple[int, int, int]]:
    """
    Get all squares containing ARMOUR pieces for the given controller.
    """
    armoured_squares: Set[Tuple[int, int, int]] = set()

    # Use cache manager if available
    if cache_manager:
        # Iterate over all board coordinates
        for x in range(9):
            for y in range(9):
                for z in range(9):
                    coord = (x, y, z)
                    piece = cache_manager.piece_cache.get(coord)
                    if piece and piece.color == controller and piece.ptype == PieceType.ARMOUR:
                        armoured_squares.add(coord)
    else:
        # Fallback to board method if cache manager not available
        for coord, piece in board.list_occupied():
            if piece.color == controller and piece.ptype == PieceType.ARMOUR:
                armoured_squares.add(coord)

    return armoured_squares


def get_vulnerable_enemies(board: BoardProto, controller: Color,
                          cache_manager: Optional[OptimizedCacheManager] = None) -> Set[Tuple[int, int, int]]:
    """
    Get all enemy pieces that are vulnerable to pawn captures (not armoured).
    """
    vulnerable_squares: Set[Tuple[int, int, int]] = set()
    enemy_color = Color.BLACK if controller == Color.WHITE else Color.WHITE

    # Use cache manager if available
    if cache_manager:
        # Iterate over all board coordinates
        for x in range(9):
            for y in range(9):
                for z in range(9):
                    coord = (x, y, z)
                    piece = cache_manager.piece_cache.get(coord)
                    if piece and piece.color == enemy_color and piece.ptype != PieceType.ARMOUR:
                        vulnerable_squares.add(coord)
    else:
        # Fallback to board method if cache manager not available
        for coord, piece in board.list_occupied():
            if piece.color == enemy_color and piece.ptype != PieceType.ARMOUR:
                vulnerable_squares.add(coord)

    return vulnerable_squares
