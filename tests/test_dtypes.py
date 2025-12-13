
import unittest
import numpy as np
import torch
import sys
import os

# Adjust path
sys.path.append(os.getcwd())

from training.training_types import TrainingExample, StateTensorPool
from game3d.common.shared_types import SIZE, N_TOTAL_PLANES, POLICY_DIM

class TestDataTypes(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        self.state_uint8 = np.random.randint(0, 256, (N_TOTAL_PLANES, SIZE, SIZE, SIZE), dtype=np.uint8)
        self.from_target = np.zeros(POLICY_DIM, dtype=np.float32)
        self.to_target = np.zeros(POLICY_DIM, dtype=np.float32)
        self.value_target = 0.5
        
        self.example = TrainingExample(
            state_tensor=self.state_uint8,
            from_target=self.from_target,
            to_target=self.to_target,
            value_target=self.value_target,
            game_id="test_game"
        )

    def test_to_tensor_normalization(self):
        """Verify that to_tensor transfers uint8 and normalizes correctly."""
        device = "cpu" # Test on CPU
        tensors = self.example.to_tensor(device=device)
        
        state_tensor = tensors['state']
        
        # Check dtype
        self.assertEqual(state_tensor.dtype, torch.float32)
        
        # Check normalization
        expected = self.state_uint8.astype(np.float32) / 255.0
        
        # Compare
        # Note: floating point differences might exist slightly, but direct division should be consistent
        self.assertTrue(np.allclose(state_tensor.numpy(), expected, atol=1e-6))
        
        print("State tensor shape:", state_tensor.shape)
        print("State tensor mean:", state_tensor.mean())

    def tearDown(self):
        # Clean up pool
        if self.example.game_id:
             StateTensorPool.release(self.example.game_id)

if __name__ == '__main__':
    unittest.main()
