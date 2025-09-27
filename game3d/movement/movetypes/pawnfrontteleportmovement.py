#game3d/movement/movetypes/pawnfrontteleportermovement.py
"""Pawn-Front Teleporter — teleports to any empty square directly in front of an enemy pawn.
   Pawns advance along Z: White +Z, Black -Z.
"""

from typing import List, Set
from game3d.pieces.enums import PieceType, Color
from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move
from game3d.common.common import in_bounds

def generate_pawn_front_teleport_moves(board, color: Color, x: int, y: int, z: int) -> List['Move']:
    """
    Generate teleport moves to any EMPTY square directly in front of an enemy pawn.
    Pawns move in Z-direction:
      - White pawns: front = (x, y, z+1)
      - Black pawns: front = (x, y, z-1)
    """
    self_pos = (x, y, z)
    current_color = state.color

    # Validate piece
    piece = state.board.piece_at(self_pos)
    if piece is None or piece.color != current_color:
        return []

    candidate_targets: Set[tuple] = set()
    board = state.board

    # ✅ Only iterate over occupied squares (not entire board!)
    for pos, other_piece in board.list_occupied():
        if other_piece.ptype != PieceType.PAWN:
            continue
        if other_piece.color == current_color:
            continue  # must be enemy pawn

        # Determine front square based on enemy pawn's color
        px, py, pz = pos
        if other_piece.color == Color.WHITE:
            front = (px, py, pz + 1)
        else:
            front = (px, py, pz - 1)

        if not in_bounds(front):
            continue
        if board.piece_at(front) is not None:
            continue  # must be empty

        candidate_targets.add(front)

    return [
        Move(from_coord=self_pos, to_coord=target, is_capture=False)
        for target in candidate_targets
    ]
