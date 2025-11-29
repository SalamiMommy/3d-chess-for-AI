
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.parallel_self_play import generate_training_data_parallel
import logging

# Configure logging to avoid cluttering output, but show INFO
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    print("Starting test...")
    try:
        # Run with a very small number of games and moves to be quick
        examples = generate_training_data_parallel(
            model_checkpoint_path="dummy_path_not_used_mock", # We might need to mock this if it actually loads
            num_games=2,
            device="cpu", # Use CPU for test
            opponent_types=["random", "random"],
            epsilon=0.1,
            num_parallel=2,
            max_moves=5, # Short game
            model_size="small" # Assuming 'small' exists or we mock
        )
        print(f"Test finished. Generated {len(examples)} examples.")
    except Exception as e:
        # It might fail on model loading if we don't have a checkpoint.
        # Let's see if we can mock the model loading part or if we need a real checkpoint.
        print(f"Test failed (expected if no model): {e}")
