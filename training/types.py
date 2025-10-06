from dataclasses import dataclass
import torch

@dataclass
class TrainingExample:
    state_tensor: torch.Tensor      # (81, 9, 9, 9)
    from_target: torch.Tensor       # (729,)
    to_target: torch.Tensor         # (729,)
    value_target: float
    move_count: int
    player_sign: float
