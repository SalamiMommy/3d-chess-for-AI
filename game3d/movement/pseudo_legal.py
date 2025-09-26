# game3d/movement/pseudo_legal.py
from typing import List
from game3d.game.gamestate import GameState
from game3d.movement.registry import get_dispatcher

def generate_pseudo_legal_moves(board: Board, color: Color) -> List[Move]:
    all_moves: List[Move] = []

    for coord, piece in board.list_occupied():
        if piece.color != color:
            continue
        dispatcher = get_dispatcher(piece.ptype)
        if dispatcher is None:
            continue
        piece_moves = dispatcher(board, color, *coord)

        # 🔥 CRITICAL: Validate from_coord matches actual piece location
        for mv in piece_moves:
            if mv.from_coord != coord:
                print(f"BUG: {piece.ptype} at {coord} generated move from {mv.from_coord}")
                print(f"Board at {mv.from_coord}: {board.piece_at(mv.from_coord)}")
                continue  # Skip invalid moves
            all_moves.append(mv)

    return all_moves
