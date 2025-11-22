import sys
import os
import numpy as np
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game3d.game.terminal import is_game_over, result, get_draw_reason
from game3d.common.shared_types import FIFTY_MOVE_RULE, REPETITION_LIMIT, Result

# Mock GameState
class MockGameState:
    def __init__(self):
        self.halfmove_clock = 0
        self.zkey = 0
        self._position_keys = np.array([], dtype=np.int64)
        self._position_counts = np.array([], dtype=np.int32)
        self.legal_moves = np.array([1]) # Dummy legal moves
        self.color = 1 # White
        self.board = None
        self.cache_manager = None

def test_move_rule():
    print(f"Testing Move Rule (Limit: {FIFTY_MOVE_RULE} half-moves)")
    state = MockGameState()
    
    # Test below limit
    state.halfmove_clock = FIFTY_MOVE_RULE - 1
    assert not is_game_over(state), f"Game should NOT be over at {state.halfmove_clock} half-moves"
    print(f"PASS: Game continues at {state.halfmove_clock} half-moves")

    # Test at limit
    state.halfmove_clock = FIFTY_MOVE_RULE
    assert is_game_over(state), f"Game SHOULD be over at {state.halfmove_clock} half-moves"
    assert result(state) == Result.DRAW, "Result should be DRAW"
    assert get_draw_reason(state) == "Move rule draw", f"Reason should be 'Move rule draw', got '{get_draw_reason(state)}'"
    print(f"PASS: Game ends at {state.halfmove_clock} half-moves with 'Move rule draw'")

def test_repetition_rule():
    print(f"Testing Repetition Rule (Limit: {REPETITION_LIMIT} repetitions)")
    state = MockGameState()
    state.zkey = 12345
    state._position_keys = np.array([12345], dtype=np.int64)
    
    # Test below limit
    state._position_counts = np.array([REPETITION_LIMIT - 1], dtype=np.int32)
    assert not is_game_over(state), f"Game should NOT be over at {REPETITION_LIMIT - 1} repetitions"
    print(f"PASS: Game continues at {REPETITION_LIMIT - 1} repetitions")

    # Test at limit
    state._position_counts = np.array([REPETITION_LIMIT], dtype=np.int32)
    assert is_game_over(state), f"Game SHOULD be over at {REPETITION_LIMIT} repetitions"
    assert result(state) == Result.DRAW, "Result should be DRAW"
    assert get_draw_reason(state) == "Repetition draw", f"Reason should be 'Repetition draw', got '{get_draw_reason(state)}'"
    print(f"PASS: Game ends at {REPETITION_LIMIT} repetitions with 'Repetition draw'")

if __name__ == "__main__":
    try:
        test_move_rule()
        test_repetition_rule()
        print("\nALL TESTS PASSED")
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
