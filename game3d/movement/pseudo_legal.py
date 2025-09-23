"""
Pseudo-legal move generator for 9×9×9 board.
Does NOT validate king safety – only blocking & bounds.
"""

from typing import List, Dict, Tuple
from pieces.enums import PieceType
from game.state import GameState
from game.move import Move
from game3d.movement.registry import get_dispatcher
from game3d.board.board import Board


def generate_pseudo_legal_moves(state: GameState) -> List[Move]:
    """
    All moves that ignore king-in-check.
    Delegates to per-piece dispatchers registered in movement.registry.
    """
    board: Board = state.board
    current_color = state.current
    all_moves: List[Move] = []

    # iterate only occupied squares (zero-copy tensor view)
    for coord, piece in board.list_occupied():
        if piece.color != current_color:
            continue

        dispatcher = get_dispatcher(piece.ptype)
        if dispatcher is None:          # safety – should never happen
            continue

        # let the piece-specific module do the work
        piece_moves = dispatcher(state, *coord)
        all_moves.extend(piece_moves)

    return all_moves
