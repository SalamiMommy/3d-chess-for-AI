"""Movement Buff – friendly pieces inside 2-sphere have extended movement."""
from __future__ import annotations
from typing import List, Tuple, Set
from game3d.pieces.enums import Color, PieceType
from game3d.effects.auras.aura import sphere_centre, BoardProto

def buffed_squares(board: BoardProto, buffer_colour: Color, cache_manager) -> Set[Tuple[int, int, int]]:
    """Return friendly squares within 2-sphere of any friendly SPEEDER."""
    buffed: Set[Tuple[int, int, int]] = set()
    for coord, piece in board.list_occupied():
        if piece.color == buffer_colour and piece.ptype == PieceType.SPEEDER:
            for sq in sphere_centre(board, coord, radius=2):
                # Access cache through cache_manager parameter
                if cache_manager:
                    target = cache_manager.piece_cache.get(sq)
                else:
                    # Fallback to board method if cache manager not available
                    target = board.get_piece(sq)
                if target is not None and target.color == buffer_colour:
                    buffed.add(sq)
    return buffed
