from typing import List, Optional
from game3d.game.gamestate import GameState
from game3d.movement.registry import get_dispatcher
from game3d.board.board import Board
from game3d.pieces.enums import Color
from game3d.movement.movepiece import Move
from game3d.cache.manager import CacheManager

def generate_pseudo_legal_moves(state: GameState) -> List[Move]:
    all_moves: List[Move] = []

    for coord, piece in state.board.list_occupied():
        if piece.color != state.color:
            continue
        dispatcher = get_dispatcher(piece.ptype)
        if dispatcher is None:
            continue

        piece_moves = dispatcher(state, *coord)

        for mv in piece_moves:
            if mv.from_coord != coord:
                print(f"BUG: {piece.ptype} at {coord} generated move from {mv.from_coord}")
                continue
            all_moves.append(mv)

    return all_moves
