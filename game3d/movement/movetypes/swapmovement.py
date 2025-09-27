"""Swap-Move — instantly swap positions with any friendly piece on the board."""

from typing import List
from game3d.pieces.enums import Color
from game3d.movement.movepiece import Move

def generate_swap_moves(board, color: Color, x: int, y: int, z: int) -> List['Move']:
    """
    Generate swap moves with any friendly piece on the board.
    The piece at (x,y,z) and the chosen friendly piece exchange positions.
    """
    self_pos = (x, y, z)
    current_color = state.color

    # Validate piece
    piece = state.board.piece_at(self_pos)
    if piece is None or piece.color != current_color:
        return []

    moves: List['Move'] = []

    # ✅ Only iterate over occupied squares (not entire board!)
    for target_pos, target_piece in state.board.list_occupied():
        if target_pos == self_pos:
            continue  # cannot swap with itself
        if target_piece.color != current_color:
            continue  # must be friendly

        # Optional: respect freeze effects
        # if state.cache.is_frozen(target_pos, current_color):
        #     continue

        moves.append(Move(
            from_coord=self_pos,
            to_coord=target_pos,
            is_capture=False
        ))

    return moves
