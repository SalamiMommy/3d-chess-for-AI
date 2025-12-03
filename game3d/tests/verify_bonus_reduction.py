
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game3d.common.shared_types import Color, PieceType, COORD_DTYPE, PIECE_TYPE_DTYPE
from game3d.game.gamestate import GameState
from training.opponents import AdaptiveOpponent, PriestHunterOpponent

def verify_bonuses():
    print("Verifying capture bonuses...")
    
    # Setup state
    state = GameState.from_startpos()
    cache_manager = state.cache_manager
    
    # Define moves that capture Priest and Freezer
    # We don't need actual pieces on board for the reward function to calculate rewards based on captured_types
    # We can mock the inputs to _apply_capture_rewards or just call batch_reward with a state that has pieces
    
    # Let's place a Priest and Freezer for Black
    priest_pos = np.array([5, 5, 5], dtype=COORD_DTYPE)
    freezer_pos = np.array([6, 6, 6], dtype=COORD_DTYPE)
    
    cache_manager.occupancy_cache.set_position(priest_pos, np.array([PieceType.PRIEST.value, Color.BLACK.value], dtype=PIECE_TYPE_DTYPE))
    cache_manager.occupancy_cache.set_position(freezer_pos, np.array([PieceType.FREEZER.value, Color.BLACK.value], dtype=PIECE_TYPE_DTYPE))
    
    # Define moves: White moves to capture them
    # Move 1: Capture Priest
    move_priest = np.array([[0,0,0, 5,5,5]], dtype=COORD_DTYPE) # From 0,0,0 to 5,5,5
    
    # Move 2: Capture Freezer
    move_freezer = np.array([[0,0,0, 6,6,6]], dtype=COORD_DTYPE) # From 0,0,0 to 6,6,6
    
    # 1. Test AdaptiveOpponent (uses defaults)
    opponent = AdaptiveOpponent(Color.WHITE)
    
    # We need to call batch_reward, but it does a lot of things.
    # We can isolate the capture reward logic or just check the total reward difference vs a non-capture
    # But batch_reward calculates everything.
    # Let's try to isolate by looking at the reward value.
    # Base capture reward is 0.1
    # Priest bonus should be 5.0 -> Total ~5.1
    # Freezer bonus should be 3.0 -> Total ~3.1
    
    # Note: Other rewards (distance, diversity etc) might interfere.
    # However, we can check if the reward is roughly what we expect or check the specific function.
    # Actually, let's just inspect the opponent instance's behavior by calling _apply_capture_rewards directly if possible?
    # No, it's an internal method.
    
    # Let's just run batch_reward and print the values.
    # We need to make sure the moves are "legal" enough for the function to run without crashing, 
    # but since we pass moves array directly, it should be fine.
    
    # We need to ensure from_coords have a piece so attributes can be fetched
    cache_manager.occupancy_cache.set_position(np.array([0,0,0], dtype=COORD_DTYPE), np.array([PieceType.PAWN.value, Color.WHITE.value], dtype=PIECE_TYPE_DTYPE))
    
    # Mock moves array (N, 6)
    moves_p = np.array([[0,0,0, 5,5,5]], dtype=COORD_DTYPE)
    moves_f = np.array([[0,0,0, 6,6,6]], dtype=COORD_DTYPE)
    
    # We need to pad moves to 6 columns if the code expects it?
    # OpponentBase.reward creates 6-col array. batch_reward expects (N, 6).
    # Wait, the code says: moves[:, :3] and moves[:, 3:6]. So 6 columns.
    
    # We need to construct full moves
    # We'll just use dummy values for flags (last columns if any? No, moves is just coords in batch_reward usually?
    # OpponentBase.reward constructs: [fx, fy, fz, tx, ty, tz]
    
    print("\n--- AdaptiveOpponent (Defaults) ---")
    rewards_p = opponent.batch_reward(state, moves_p)
    rewards_f = opponent.batch_reward(state, moves_f)
    
    print(f"Priest Capture Reward: {rewards_p[0]:.4f}")
    print(f"Freezer Capture Reward: {rewards_f[0]:.4f}")
    
    # Expected: ~5.1 and ~3.1 (plus/minus other small factors)
    # Let's verify they are close to expected values
    
    if abs(rewards_p[0] - 5.1) > 1.0: # Allow some slack for other factors
        print("WARNING: Priest reward seems off (expected ~5.1)")
    else:
        print("SUCCESS: Priest reward is in expected range.")
        
    if abs(rewards_f[0] - 3.1) > 1.0:
        print("WARNING: Freezer reward seems off (expected ~3.1)")
    else:
        print("SUCCESS: Freezer reward is in expected range.")

    # 2. Test PriestHunterOpponent
    print("\n--- PriestHunterOpponent ---")
    hunter = PriestHunterOpponent(Color.WHITE)
    
    rewards_p_hunter = hunter.batch_reward(state, moves_p)
    rewards_f_hunter = hunter.batch_reward(state, moves_f)
    
    print(f"Priest Capture Reward: {rewards_p_hunter[0]:.4f}")
    print(f"Freezer Capture Reward: {rewards_f_hunter[0]:.4f}")
    
    # Expected: 
    # Priest: 5.0 (base) + 0.1 (capture) = 5.1
    # Freezer: 0.1 (base) + 0.1 (capture) = 0.2
    
    if abs(rewards_p_hunter[0] - 5.1) > 1.0:
        print("WARNING: PriestHunter Priest reward seems off (expected ~5.1)")
    else:
        print("SUCCESS: PriestHunter Priest reward is in expected range.")
        
    if abs(rewards_f_hunter[0] - 0.2) > 0.5:
        print("WARNING: PriestHunter Freezer reward seems off (expected ~0.2)")
    else:
        print("SUCCESS: PriestHunter Freezer reward is in expected range.")

if __name__ == "__main__":
    verify_bonuses()
