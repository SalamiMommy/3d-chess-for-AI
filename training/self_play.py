from game3d.game3d import Game3D
from game3d.game.gamestate import GameState
from game3d.pieces.enums import Color
from game3d.common.common import N_PLANES_PER_SIDE
import random

# ------------------------------------------------------------------
# 1.  tiny helper – crash early with full context
# ------------------------------------------------------------------
def _assert_legal(state: GameState, legal_moves: list) -> None:
    board = state.board
    WHITE_SLICE = slice(0, N_PLANES_PER_SIDE)
    BLACK_SLICE = slice(N_PLANES_PER_SIDE, 2 * N_PLANES_PER_SIDE)

    for m in legal_moves:
        from_coord = m.from_coord
        piece = board.piece_at(from_coord)
        if piece is None:
            x, y, z = from_coord
            white_vals = board._tensor[WHITE_SLICE, z, y, x]
            black_vals = board._tensor[BLACK_SLICE, z, y, x]

            w_max = white_vals.max().item()
            b_max = black_vals.max().item()
            occ = (white_vals.sum() + black_vals.sum()).item() > 0

            print("=" * 79)
            print(f"BUGGY LEGAL MOVE: {m}")
            print(f"board key        : 0x{board.byte_hash():016x}")
            print(f"history length   : {len(state.history)}")
            print(f"occupancy (sum)  : {white_vals.sum().item():.6f} + {black_vals.sum().item():.6f} = {white_vals.sum().item() + black_vals.sum().item():.6f}")
            print(f"white max        : {w_max:.6f}")
            print(f"black max        : {b_max:.6f}")
            print(f"is occupied?     : {occ}")
            print("=" * 79)
            raise AssertionError(f"Legal list contains empty-square move: {m}")

# ------------------------------------------------------------------
# 2.  MCTS stub – depth == 0  →  uniform random, no cache touched
# ------------------------------------------------------------------
def mcts_search(net, state, depth: int):
    legal_moves = state.legal_moves()
    if not legal_moves:
        return None
    _assert_legal(state, legal_moves)

    if depth == 0:                      # MCTS disabled
        pi = [1.0 / len(legal_moves) for _ in legal_moves]
        return pi

    # -----  depth > 0  →  placeholder MCTS (no game-cache used) -----
    pi = [1.0 / len(legal_moves) for _ in legal_moves]
    return pi

# ------------------------------------------------------------------
# 3.  sampler – only pick moves that *really* start on a piece
# ------------------------------------------------------------------
def sample_pi(pi, legal_moves, state: GameState):
    """Return a move whose from-square is *still* occupied, or None."""
    if not legal_moves:
        return None
    choices = list(zip(legal_moves, pi))
    random.shuffle(choices)
    for mv, _ in choices:
        if state.board.piece_at(mv.from_coord) is not None:
            return mv
    return None

# ------------------------------------------------------------------
# 4.  self-play loop – FULL rebuild before every move
# ------------------------------------------------------------------
def play_game(net, mcts_depth: int) -> list:
    # Start with initial state
    initial_state = GameState.start()
    current_state = initial_state
    examples = []

    while not current_state.is_game_over():
        board_hash_before = current_state.board.byte_hash()
        print(f"Board hash: {board_hash_before:016x}")

        legal_moves = current_state.legal_moves()
        if not legal_moves:
            break

        pi = mcts_search(net, current_state, depth=mcts_depth)
        move = sample_pi(pi, legal_moves, current_state)
        if move is None:
            break

        # Validate move before applying
        if current_state.board.piece_at(move.from_coord) is None:
            print(f"CRITICAL: Selected move from empty square: {move}")
            break

        examples.append((current_state.to_tensor(), pi, None))
        piece_before = current_state.board.piece_at(move.from_coord)
        print(f"Piece at {move.from_coord}: {piece_before}")

        # Apply move to create NEW state
        try:
            new_state = current_state.make_move(move)
            current_state = new_state  # Only update if successful
        except Exception as e:
            print(f"Move application failed: {e}")
            print(f"Board hash after failed move: {current_state.board.byte_hash():016x}")
            print(f"Piece at {move.from_coord} after: {current_state.board.piece_at(move.from_coord)}")
            break

    z = current_state.outcome()
    return [(x, pi, z) for x, pi, _ in examples]
