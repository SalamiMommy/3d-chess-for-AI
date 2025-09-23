"""Pawn-Front Teleporter — teleports to any empty square directly in front of an enemy pawn.
   Pawns advance along Z: White +Z, Black -Z.
"""

from typing import List, Set, Tuple
from pieces.enums import PieceType, Color
from game.state import GameState
from game.move import Move
from common import in_bounds


def generate_pawn_front_teleport_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """
    Generate teleport moves to any EMPTY square directly in front of an enemy pawn.
    Pawns move in Z-direction:
      - White pawns: front = (x, y, z+1)
      - Black pawns: front = (x, y, z-1)
    """
    moves: List[Move] = []
    board = state.board
    current_color = state.current
    self_pos = (x, y, z)

    # Validate piece exists and belongs to current player
    piece = board.piece_at(self_pos)
    if piece is None or piece.color != current_color:
        return moves

    candidate_targets: Set[Tuple[int, int, int]] = set()

    # Scan entire board for enemy pawns
    for px in range(9):
        for py in range(9):
            for pz in range(9):
                pos = (px, py, pz)
                target_piece = board.piece_at(pos)

                if target_piece is None:
                    continue
                if target_piece.ptype != PieceType.PAWN:
                    continue
                if target_piece.color == current_color:
                    continue  # must be enemy

                # Determine "in front" based on enemy pawn's color
                if target_piece.color == Color.WHITE:
                    front = (px, py, pz + 1)   # White moves +Z
                else:
                    front = (px, py, pz - 1)   # Black moves -Z

                if not in_bounds(front):
                    continue
                if board.piece_at(front) is not None:
                    continue  # empty only

                candidate_targets.add(front)

    # Build unique teleport moves
    for target in candidate_targets:
        moves.append(Move(
            from_coord=self_pos,
            to_coord=target,
            is_capture=False,
            metadata={
                "is_teleport": True,
                "mechanic": "pawn_front_z",
                "flavor": "Phasing into the enemy's front line"
            }
        ))

    return moves
