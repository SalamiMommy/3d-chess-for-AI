"""Registry + dispatcher for 42 move generators."""

from typing import Callable
from pieces.enums import PieceType
from game.state import GameState
from game.move import Move

# type alias
Generator = Callable[[GameState, int, int, int], list[Move]]

_REGISTRY: dict[PieceType, Generator] = {}

def register(pt: PieceType, gen: Generator):
    """Decorator / func registrar."""
    _REGISTRY[pt] = gen
    return gen

def generate_legal_moves(state: GameState) -> list[Move]:
    """Dispatch to piece-specific generators."""
    moves: list[Move] = []
    for z in range(9):
        for y in range(9):
            for x in range(9):
                p = state.board.pieces[z][y][x]
                if p is None or p.color != state.current:
                    continue
                gen = _REGISTRY.get(p.ptype)
                if gen:
                    moves.extend(gen(state, x, y, z))
    return moves
