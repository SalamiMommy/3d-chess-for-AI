"""Swap-Move — instantly swap positions with any friendly piece on the board.
Pure movement logic — no GameState.
"""

from typing import List
from game3d.pieces.enums import Color
from game3d.movement.movepiece import Move

def generate_swapper_moves(board, color: Color, x: int, y: int, z: int) -> List['Move']:
    """
    Generate swap moves with any friendly piece on the board.
    The piece at (x,y,z) and the chosen friendly piece exchange positions.
    """
    self_pos = (x, y, z)

    # Validate piece at start
    piece = cache.piece_cache.get(self_pos)
    if piece is None or piece.color != color:
        return []

    moves: List['Move'] = []

    # Iterate over occupied squares
    for target_pos, target_piece in board.list_occupied():
        if target_pos == self_pos:
            continue
        if target_piece.color != color:
            continue

        moves.append(Move(
            from_coord=self_pos,
            to_coord=target_pos,
            is_capture=False
        ))

    return moves
