
import sys
from unittest.mock import MagicMock

# Mock tensorboard
sys.modules["torch.utils.tensorboard"] = MagicMock()
sys.modules["tensorboard"] = MagicMock()

# Mock tqdm
tqdm_mock = MagicMock()
tqdm_mock.tqdm = lambda x, **kwargs: x
sys.modules["tqdm"] = tqdm_mock

import torch
from torch.utils.data import Dataset, DataLoader
from training.optim_train import ChessTrainer
from training.training_types import TrainingConfig, TrainingExample
import numpy as np

# Mock Dataset
class MockDataset(Dataset):
    def __init__(self, size):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Return dummy data matching the expected shapes
        # states: (89, 9, 9, 9)
        # from_targets: (729,)
        # to_targets: (729,)
        # value_targets: scalar
        return (
            torch.zeros((89, 9, 9, 9)),
            torch.zeros((729,)),
            torch.zeros((729,)),
            torch.tensor(0.0)
        )

def reproduce_crash():
    # Create a config with default train_split=0.9
    config = TrainingConfig(
        batch_size=4,
        train_split=0.9,
        model_type="transformer", # Use transformer to avoid ChessDataset logic which might be different
        device="cpu"
    )
    
    # Create a trainer
    trainer = ChessTrainer(config)
    
    # Create a dataset with very few examples
    # If size is 1: train_len = int(1 * 0.9) = 0
    dataset = MockDataset(size=1)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Train split: {config.train_split}")
    
    try:
        # This should crash
        trainer.train(dataset)
    except ZeroDivisionError:
        print("Caught expected ZeroDivisionError!")
    except Exception as e:
        print(f"Caught unexpected exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reproduce_crash()
