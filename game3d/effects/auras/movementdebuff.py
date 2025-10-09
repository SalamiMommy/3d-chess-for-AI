"""Movement Debuff – enemy pieces inside 2-sphere aura move like pawns."""
from __future__ import annotations
from typing import List, Tuple, Set
from game3d.pieces.enums import Color, PieceType
from game3d.effects.auras.aura import sphere_centre, BoardProto

def debuffed_squares(board: BoardProto, debuffer_colour: Color, cache_manager) -> Set[Tuple[int, int, int]]:
    """Return enemy squares within 2-sphere of any friendly SLOWER."""
    debuffed: Set[Tuple[int, int, int]] = set()
    for coord, piece in board.list_occupied():
        if piece.color == debuffer_colour and piece.ptype == PieceType.SLOWER:
            for sq in sphere_centre(board, coord, radius=2):
                # Access cache through cache_manager parameter
                if cache_manager:
                    target = cache_manager.piece_cache.get(sq)
                else:
                    # Fallback to board method if cache manager not available
                    target = board.cache_manager.occupancy.get(sq) if cache_manager else board.get_piece(sq)
                if target is not None and target.color != debuffer_colour:
                    debuffed.add(sq)
    return debuffed
