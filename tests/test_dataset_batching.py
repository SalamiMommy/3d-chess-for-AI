import torch
import numpy as np
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from training.optim_train import GraphTransformerDataset
from training.training_types import TrainingExample, POLICY_DIM
from game3d.common.shared_types import N_TOTAL_PLANES, SIZE

def test_dataset_batching():
    print("Testing GraphTransformerDataset batching optimization...")
    
    # Mock examples
    examples = []
    for i in range(5):
        # Create VALID state shape
        # TrainingExample expects Normalized internal checks
        # But optimize_dataset uses ex.state_tensor
        state = np.zeros((N_TOTAL_PLANES, SIZE, SIZE, SIZE), dtype=np.uint8)
        state[0,0,0,0] = i + 10 # Marker
        
        from_target = np.zeros(POLICY_DIM, dtype=np.float32)
        to_target = np.zeros(POLICY_DIM, dtype=np.float32)
        # Validate requires sum ~ 1.0
        from_target[0] = 1.0
        to_target[0] = 1.0
        
        value = float(i) / 10.0
        
        ex = TrainingExample(
            state_tensor=state, 
            from_target=from_target,
            to_target=to_target,
            value_target=value,
            move_count=i,
            player_sign=1.0,
            game_id=f"game_{i}"
        )
        # Skip validate call here, but dataset will call it. 
        # So we must ensure it passes or mock validate.
        # But validate checks sums.
        
        examples.append(ex)
        
    ds = GraphTransformerDataset(examples, device='cpu') # Test on CPU
    
    # 1. Check internal storage
    print("Checking internal storage...")
    assert isinstance(ds.states, torch.Tensor), "Internal states should be a Tensor"
    assert ds.states.dtype == torch.uint8, "Internal states should be uint8"
    assert ds.states.shape == (5, N_TOTAL_PLANES, SIZE, SIZE, SIZE), f"Shape mismatch: {ds.states.shape}"
    # Check content
    # state[0,0,0,0] was i+10.
    # Ex 3 -> 13
    assert ds.states[3, 0, 0, 0, 0] == 13, "Data content verification failed"
    
    # 2. Check __getitem__ normalization
    print("Checking __getitem__...")
    item = ds[2]
    # item is tuple (state, from, to, val)
    state, f, t, v = item
    assert state.dtype == torch.float32, "__getitem__ should return float state"
    
    # Expected: (2+10) / 255.0 = 12/255
    expected = 12.0 / 255.0
    actual = state[0,0,0,0].item()
    assert abs(actual - expected) < 1e-6, f"Normalization error: got {actual}, expected {expected}"
    
    # 3. Check get_batch_tensor
    print("Checking get_batch_tensor...")
    batch = ds.get_batch_tensor([1, 4])
    assert batch.states.shape[0] == 2
    assert batch.states.dtype == torch.float32
    assert abs(batch.value_targets[1].item() - 0.4) < 1e-6, "Value target mismatch in batch"
    
    print("All checks passed!")

if __name__ == "__main__":
    test_dataset_batching()
