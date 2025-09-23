"""Generate games via MCTS + NN; store to data/db.sqlite."""

from game3d.game3d import Game3D
from game3d.game.gamestate import GameState
from pieces.enums import Color

def mcts_search(net, state):
    # TODO: Implement MCTS using net and state
    # Should return a policy vector (pi) for legal moves
    # For now, stub: uniform over legal moves
    legal_moves = state.legal_moves()
    if not legal_moves:
        return None
    pi = [1.0/len(legal_moves) for _ in legal_moves]
    return pi

def sample_pi(pi, legal_moves):
    # TODO: Sample a move from policy vector pi
    # For now, stub: pick max probability (or first if uniform)
    return legal_moves[0] if legal_moves else None

def play_game(net) -> list:
    """Single self-play game; returns list of (tensor, pi, z)."""
    game = Game3D()  # uses GameState.empty()
    examples = []
    while not game.is_game_over():
        state = game.state
        legal_moves = state.legal_moves()
        if not legal_moves:
            break
        pi = mcts_search(net, state)
        move = sample_pi(pi, legal_moves)
        if move is None:
            break
        examples.append((state.to_tensor(), pi, None))
        receipt = game.submit_move(move)
        if not receipt.is_legal:
            break
    z = game.state.outcome()
    return [(x, pi, z) for x, pi, _ in examples]
