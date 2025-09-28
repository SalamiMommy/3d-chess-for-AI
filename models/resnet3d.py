import torch
import torch.nn as nn
import torch.nn.functional as F
from game3d.common.common import SIZE_X, SIZE_Y, SIZE_Z, N_TOTAL_PLANES, N_CHANNELS
from typing import Optional, Tuple
SIZE = 9

# ==============================================================================
# OPTIMIZED 3D RESNET FOR CHESS
# ==============================================================================

class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation block for 3D - helps focus on important features."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class GroupNorm3D(nn.Module):
    """Group Normalization for 3D - better for small batch sizes than BatchNorm."""

    def __init__(self, num_channels: int, num_groups: int = 32):
        super().__init__()
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.size()
        assert c % self.num_groups == 0

        x = x.view(b, self.num_groups, c // self.num_groups, d, h, w)
        mean = x.mean(dim=[2, 3, 4, 5], keepdim=True)
        var = x.var(dim=[2, 3, 4, 5], keepdim=True, unbiased=False)
        x = (x - mean) / (var + 1e-5).sqrt()
        x = x.view(b, c, d, h, w)

        return x * self.weight + self.bias

def conv3x3x3(in_ch: int, out_ch: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv3d:
    """Optimized 3x3x3 convolution with configurable groups and dilation."""
    return nn.Conv3d(
        in_ch, out_ch, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, dilation=dilation, bias=False
    )

def conv1x1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv3d:
    """1x1x1 convolution for channel adjustment."""
    return nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)

class Bottleneck3D(nn.Module):
    """Bottleneck residual block with SE and GroupNorm - more efficient than basic block."""

    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: nn.Module = None, groups: int = 1, base_width: int = 64,
                 dilation: int = 1, norm_layer: nn.Module = None):
        super().__init__()

        if norm_layer is None:
            norm_layer = GroupNorm3D

        width = int(out_channels * (base_width / 64.)) * groups

        # Main path with bottleneck design
        self.conv1 = conv1x1x1(in_channels, width)
        self.norm1 = norm_layer(width)

        self.conv2 = conv3x3x3(width, width, stride, groups, dilation)
        self.norm2 = norm_layer(width)

        self.conv3 = conv1x1x1(width, out_channels * self.expansion)
        self.norm3 = norm_layer(out_channels * self.expansion)

        # Squeeze-and-Excitation
        self.se = SEBlock3D(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # 1x1x1 reduce
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        # 3x3x3 conv
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        # 1x1x1 expand
        out = self.conv3(out)
        out = self.norm3(out)

        # Apply SE
        out = self.se(out)

        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class AttentionBlock3D(nn.Module):
    """3D Attention block combining channel and spatial attention."""

    def __init__(self, channels: int):
        super().__init__()
        # Channel attention
        self.channel_avg = nn.AdaptiveAvgPool3d(1)
        self.channel_max = nn.AdaptiveMaxPool3d(1)
        self.channel_fc = nn.Sequential(
            nn.Conv3d(channels, channels // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 16, channels, 1, bias=False)
        )

        # Spatial attention
        self.spatial_conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        avg_out = self.channel_fc(self.channel_avg(x))
        max_out = self.channel_fc(self.channel_max(x))
        channel_att = torch.sigmoid(avg_out + max_out)
        x = x * channel_att

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = torch.sigmoid(self.spatial_conv(spatial_input))
        x = x * spatial_att

        return x

class OptimizedPolicyHead(nn.Module):
    """Optimized policy head with spatial attention and multiple output formats."""

    def __init__(self, in_channels: int, n_moves: int):
        super().__init__()

        # Feature extraction with attention
        self.conv1 = nn.Conv3d(in_channels, 128, 1, bias=False)
        self.norm1 = GroupNorm3D(128)
        self.attention = AttentionBlock3D(128)

        # Multi-scale feature extraction
        self.conv3 = nn.Conv3d(128, 64, 3, padding=1, bias=False)
        self.conv5 = nn.Conv3d(128, 64, 5, padding=2, bias=False)
        self.conv7 = nn.Conv3d(128, 64, 7, padding=3, bias=False)

        # Final feature combination
        self.final_conv = nn.Conv3d(192, 32, 1, bias=False)  # 64*3 = 192
        self.norm_final = GroupNorm3D(32)

        # Policy output
        self.fc_policy = nn.Linear(32 * SIZE * SIZE * SIZE, n_moves)

        # Optional: add spatial policy output for move probabilities per square
        self.spatial_conv = nn.Conv3d(32, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial feature extraction
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.attention(x)

        # Multi-scale features
        feat3 = F.relu(self.conv3(x))
        feat5 = F.relu(self.conv5(x))
        feat7 = F.relu(self.conv7(x))

        # Combine multi-scale features
        combined = torch.cat([feat3, feat5, feat7], dim=1)
        x = F.relu(self.norm_final(self.final_conv(combined)))

        # Global average pooling for policy
        policy_logits = self.fc_policy(x.view(x.size(0), -1))

        return F.log_softmax(policy_logits, dim=1)

class OptimizedValueHead(nn.Module):
    """Optimized value head with uncertainty estimation."""

    def __init__(self, in_channels: int):
        super().__init__()

        # Feature extraction
        self.conv1 = nn.Conv3d(in_channels, 32, 1, bias=False)
        self.norm1 = GroupNorm3D(32)

        # Attention for important regions
        self.attention = AttentionBlock3D(32)

        # Global feature extraction
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # Value regression with dropout for uncertainty
        self.fc1 = nn.Linear(32, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        # Optional: uncertainty head
        self.uncertainty_fc = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Feature extraction
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.attention(x)

        # Global pooling
        x = self.global_pool(x).view(x.size(0), -1)

        # Value regression
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))

        value = torch.tanh(self.fc3(x))

        # Optional uncertainty estimation
        uncertainty = torch.sigmoid(self.uncertainty_fc(x))

        return value, uncertainty

class OptimizedResNet3D(nn.Module):
    """Optimized 3D ResNet for chess with modern improvements."""

    def __init__(self, blocks: int = 15, n_moves: int = 10_000,
                 channels: int = 256, groups: int = 32, width_per_group: int = 64):
        super().__init__()

        # Initial convolution with larger kernel for better spatial coverage
        self.in_conv = nn.Conv3d(N_CHANNELS, channels, kernel_size=5,
                                 padding=2, bias=False)
        self.bn1 = GroupNorm3D(channels, groups)
        self.relu = nn.ReLU(inplace=True)

        # Optional: initial max pooling for dimension reduction
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)

        # Residual blocks with bottleneck design
        self.layer1 = self._make_layer(channels, channels//4, blocks,
                                       stride=1, groups=groups,
                                       base_width=width_per_group)

        # Optional: feature pyramid for multi-scale processing
        self.feature_pyramid = nn.ModuleList([
            nn.Conv3d(channels, channels//2, 1) for _ in range(3)
        ])

        # Attention mechanism
        self.global_attention = AttentionBlock3D(channels)

        # Heads
        self.policy_head = OptimizedPolicyHead(channels, n_moves)
        self.value_head = OptimizedValueHead(channels)

        # Initialize weights properly
        self._initialize_weights()

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int,
                    stride: int = 1, groups: int = 1, base_width: int = 64):
        """Create a layer of bottleneck blocks."""
        layers = []

        for i in range(blocks):
            layers.append(
                Bottleneck3D(
                    in_channels if i == 0 else out_channels * Bottleneck3D.expansion,
                    out_channels, stride if i == 0 else 1,
                    groups=groups, base_width=base_width
                )
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Improved weight initialization for better training stability."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # Kaiming initialization for ReLU
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            x = self.relu(self.bn1(self.in_conv(x)))
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.global_attention(x)
            policy = self.policy_head(x)
            value, uncertainty = self.value_head(x)
            return policy, value, uncertainty  # uncertainty can be ignored

# ==============================================================================
# LIGHTWEIGHT VERSION FOR FASTER INFERENCE
# ==============================================================================

class LightweightResNet3D(nn.Module):
    """Lightweight version for fast inference during search."""

    def __init__(self, blocks: int = 8, n_moves: int = 10_000, channels: int = 128):
        super().__init__()

        # Reduced initial channels
        self.in_conv = nn.Conv3d(N_CHANNELS, channels, 3, padding=1, bias=False)
        self.bn1 = GroupNorm3D(channels, 16)  # Smaller groups
        self.relu = nn.ReLU(inplace=True)

        # Basic blocks instead of bottleneck for speed
        self.blocks = nn.ModuleList([
            ResBlock(channels) for _ in range(blocks)
        ])

        # Simple heads
        self.policy_conv = nn.Conv3d(channels, 32, 1)
        self.policy_fc = nn.Linear(32 * SIZE * SIZE * SIZE, n_moves)

        self.value_conv = nn.Conv3d(channels, 16, 1)
        self.value_fc = nn.Linear(16 * SIZE * SIZE * SIZE, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Feature extraction
        x = self.relu(self.bn1(self.in_conv(x)))

        # Residual blocks
        for block in self.blocks:
            x = block(x)

        # Policy head
        policy = self.relu(self.policy_conv(x))
        policy = self.policy_fc(policy.view(policy.size(0), -1))
        policy = F.log_softmax(policy, dim=1)

        # Value head
        value = self.relu(self.value_conv(x))
        value = torch.tanh(self.value_fc(value.view(value.size(0), -1)))

        return policy, value

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model: nn.Module, input_size: tuple = (1, N_CHANNELS, 9, 9, 9)):
    """Print model summary."""
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(input_size)
    with torch.no_grad():
        outputs = model(x)
        policy, value = outputs[:2]  # Uncertainty optional
    print(f"Input shape: {x.shape}")
    print(f"Policy output shape: {policy.shape}")
    print(f"Value output shape: {value.shape}")

# Example usage
if __name__ == "__main__":
    # Create optimized model
    model = OptimizedResNet3D(blocks=15, n_moves=10000, channels=256)
    model_summary(model)

    # Create lightweight model for search
    light_model = LightweightResNet3D(blocks=8, n_moves=10000, channels=128)
    model_summary(light_model)
