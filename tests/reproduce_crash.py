
import torch
import numpy as np
from training.optim_train import ChessTrainer, TrainingConfig, TrainingExample
from game3d.common.shared_types import SIZE, N_TOTAL_PLANES

def reproduce_crash():
    # Minimal config
    config = TrainingConfig(
        batch_size=2,
        epochs=1,
        validate_every=1,
        save_every=10,
        model_size="small", # Use small model for speed
        device="cpu" # Use CPU for reproduction to avoid CUDA setup issues
    )
    
    # Create trainer
    trainer = ChessTrainer(config)
    
    # Create a SINGLE example
    # This will force a very small dataset. 
    # With default train_split=0.9, 1 example -> 0 train, 1 val OR 1 train, 0 val depending on rounding/logic
    # Let's try to trigger the 0 batches case.
    
    # Create dummy data with valid policy targets (sum to 1.0)
    state = np.zeros((N_TOTAL_PLANES, SIZE, SIZE, SIZE), dtype=np.float32)
    from_target = np.zeros(SIZE**3, dtype=np.float32)
    from_target[0] = 1.0
    to_target = np.zeros(SIZE**3, dtype=np.float32)
    to_target[0] = 1.0
    
    example = TrainingExample(
        state_tensor=state,
        from_target=from_target,
        to_target=to_target,
        value_target=0.0
    )
    
    print("Starting training with 1 example...")
    try:
        trainer.train([example])
        print("Training completed successfully (Fix verified!)")
    except ZeroDivisionError:
        print("Caught ZeroDivisionError! Fix FAILED.")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"Caught unexpected exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reproduce_crash()
