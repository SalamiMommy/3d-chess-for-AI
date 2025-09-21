"""Generate games via MCTS + NN; store to data/db.sqlite."""

from game.state import GameState
from game.board import Board
from pieces.enums import Color

def play_game(net) -> list:
    """Single self-play game; returns list of (tensor, pi, z)."""
    state = GameState(Board(), Color.WHITE)
    examples = []
    while not state.is_terminal():           # TODO: implement
        pi = mcts_search(net, state)         # TODO
        examples.append((state.to_tensor(), pi, None))
        move = sample_pi(pi)
        state = state.make_move(move)
    z = state.outcome()                      # TODO
    return [(x, pi, z) for x, pi, _ in examples]
