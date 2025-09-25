import torch
import torch.nn as nn
from game3d.pieces.features import N_CHANNELS
from game3d.common.common import SIZE_X, SIZE_Y, SIZE_Z, N_TOTAL_PLANES
SIZE = 9

def conv3x3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv3d:
    return nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)

class ResBlock(nn.Module):
    """Pre-activation residual block (3D)."""
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = conv3x3x3(ch, ch)
        self.bn1   = nn.BatchNorm3d(ch)
        self.conv2 = conv3x3x3(ch, ch)
        self.bn2   = nn.BatchNorm3d(ch)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class PolicyHead(nn.Module):
    def __init__(self, ch: int, n_moves: int):
        super().__init__()
        self.conv = nn.Conv3d(ch, 8, 1)        # shrink
        self.fc   = nn.Linear(8*SIZE*SIZE*SIZE, n_moves)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return torch.log_softmax(self.fc(x), dim=1)

class ValueHead(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv3d(ch, 4, 1)
        self.fc1  = nn.Linear(4*SIZE*SIZE*SIZE, 256)
        self.fc2  = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))          # (−1,1)

class ResNet3D(nn.Module):
    """Tower + two heads."""
    def __init__(self, blocks: int = 15, n_moves: int = 10_000):
        super().__init__()
        self.in_conv = nn.Conv3d(N_CHANNELS, 256, 3, padding=1, bias=False)
        self.bn      = nn.BatchNorm3d(256)
        self.relu    = nn.ReLU(inplace=True)
        self.tower   = nn.Sequential(*[ResBlock(256) for _ in range(blocks)])
        self.policy  = PolicyHead(256, n_moves)
        self.value   = ValueHead(256)

    def forward(self, x: torch.Tensor):
        """x: (N,C,9,9,9)"""
        feat = self.relu(self.bn(self.in_conv(x)))
        feat = self.tower(feat)
        return self.policy(feat), self.value(feat)
