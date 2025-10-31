# types.py - UPDATED to support both numpy and tensor
from dataclasses import dataclass
from typing import Union
import torch
import numpy as np

@dataclass
class TrainingExample:
    state_tensor: Union[torch.Tensor, np.ndarray]      # (81, 9, 9, 9) - can be either
    from_target: Union[torch.Tensor, np.ndarray]       # (729,) - can be either
    to_target: Union[torch.Tensor, np.ndarray]         # (729,) - can be either
    value_target: float
    move_count: int
    player_sign: float
