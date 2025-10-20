import torch
import torch.nn as nn
import torch.nn.functional as F
from game3d.common.constants import SIZE_X, SIZE_Y, SIZE_Z, N_CHANNELS, SIZE
from typing import Optional, Tuple

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
        num_groups = self.num_groups if c % self.num_groups == 0 else (1 if c < 32 else c // 32 * 32 // c * c)
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

class AttentionBlock3D(nn.Module):  # Fix: Use Linear for channel att
    def __init__(self, channels: int):
        super().__init__()
        reduced = channels // 16
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),  # Linear after pool
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )
        self.spatial_conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *spatial = x.size()
        # Channel attention (after avg/max pool to (B,C,1,1,1))
        avg_out = F.adaptive_avg_pool3d(x, 1).view(b, c)
        max_out = F.adaptive_max_pool3d(x, 1).view(b, c)
        channel_att = self.channel_fc(avg_out + max_out).view(b, c, 1, 1, 1)
        x = x * torch.sigmoid(channel_att)

        # Spatial attention
        avg_sp = torch.mean(x, dim=1, keepdim=True)
        max_sp, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_sp, max_sp], dim=1)
        spatial_att = torch.sigmoid(self.spatial_conv(spatial_input))
        x = x * spatial_att
        return x

class FactorizedPolicyHead(nn.Module):
    """Outputs separate 'from' and 'to' logits (729 each)."""

    def __init__(self, in_channels: int):
        super().__init__()
        # Shared feature extractor
        self.conv = nn.Conv3d(in_channels, 32, kernel_size=1, bias=False)
        self.norm = GroupNorm3D(32)

        # From head
        self.from_conv = nn.Conv3d(32, 1, kernel_size=1, bias=False)
        # To head
        self.to_conv = nn.Conv3d(32, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            from_logits: (B, 729)
            to_logits:   (B, 729)
        """
        x = F.relu(self.norm(self.conv(x)))  # (B, 32, 9,9,9)

        from_logits = self.from_conv(x).view(x.size(0), -1)  # (B, 729)
        to_logits = self.to_conv(x).view(x.size(0), -1)      # (B, 729)

        return from_logits, to_logits

class OptimizedValueHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, 1, bias=False)
        self.norm1 = GroupNorm3D(32)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(32, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        # Drop uncertainty for now

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Return value only
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.global_pool(x).view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        value = torch.tanh(self.fc3(x))
        return value

class OptimizedResNet3D(nn.Module):
    def __init__(self, blocks: int = 15, n_moves: int = 10_000,
                 channels: int = 256, groups: int = 32, width_per_group: int = 64):
        super().__init__()
        self.in_conv = nn.Conv3d(N_CHANNELS, channels, kernel_size=5, padding=2, bias=False)  # <-- Fix: Use N_CHANNELS instead of 81
        self.bn1 = GroupNorm3D(channels, groups)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        self.layer1 = self._make_layer(channels, channels // 4, blocks, stride=1, groups=groups, base_width=width_per_group)

        # Attention (now fixed; optional - comment out if issues)
        self.global_attention = AttentionBlock3D(channels)

        # Heads (input: channels post-layer)
        self.policy_head = FactorizedPolicyHead(channels)
        self.value_head = OptimizedValueHead(channels)

        self._initialize_weights()

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int,
                    stride: int = 1, groups: int = 1, base_width: int = 64):
        layers = []
        downsample = None
        # FIXED: Proper downsample
        if stride != 1 or in_channels != out_channels * Bottleneck3D.expansion:
            downsample = nn.Sequential(
                conv1x1x1(in_channels, out_channels * Bottleneck3D.expansion, stride),
                GroupNorm3D(out_channels * Bottleneck3D.expansion, groups)
            )
        current_in = in_channels
        for i in range(blocks):
            stride_i = stride if i == 0 else 1
            layers.append(Bottleneck3D(
                current_in, out_channels, stride_i,
                downsample if i == 0 else None,
                groups=groups, base_width=base_width  # FIX: Changed width_per_group to base_width
            ))
            current_in = out_channels * Bottleneck3D.expansion  # Update for next block
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.relu(self.bn1(self.in_conv(x)))
        # x = self.maxpool(x)  # <-- Fix: Remove this (undefined); add if needed, e.g., self.maxpool = nn.MaxPool3d(kernel_size=3, stride=1, padding=1) in __init__
        x = self.layer1(x)
        x = self.global_attention(x)

        # NEW: factorized policy
        from_logits, to_logits = self.policy_head(x)
        value = self.value_head(x)  # <-- Fix: Only returns value
        return from_logits, to_logits, value  # <-- Fix: Return only 3 values

# ==============================================================================
# LIGHTWEIGHT VERSION FOR FASTER INFERENCE
# ==============================================================================
class ResBlock(nn.Module):
    """Basic residual block for lightweight model."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = conv3x3x3(channels, channels)
        self.norm1 = GroupNorm3D(channels)
        self.conv2 = conv3x3x3(channels, channels)
        self.norm2 = GroupNorm3D(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

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

        self.global_attention = AttentionBlock3D(channels)  # <-- Add if needed; was referenced but not defined

        self.policy_head = FactorizedPolicyHead(channels)  # <-- Fix: Add this (copied from optimized; adjust if needed)
        self.value_head = OptimizedValueHead(channels)  # <-- Fix: Use same as optimized

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.relu(self.bn1(self.in_conv(x)))
        for block in self.blocks:  # <-- Fix: Loop over self.blocks (no self.layer1)
            x = block(x)
        x = self.global_attention(x)  # FIXED/optional

        from_logits, to_logits = self.policy_head(x)
        value = self.value_head(x)
        return from_logits, to_logits, value

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def model_summary(model: nn.Module, input_size: tuple = (1, N_CHANNELS, 9, 9, 9)):
#     """Print model summary."""
#     print(f"Model: {model.__class__.__name__}")
#     print(f"Total parameters: {count_parameters(model):,}")
#
#     # Test forward pass
#     x = torch.randn(input_size)
#     with torch.no_grad():
#         outputs = model(x)
#         print(f"Number of outputs: {len(outputs)}")
#         if len(outputs) == 3:
#             from_logits, to_logits, value = outputs
#             print(f"Input shape: {x.shape}")
#             print(f"From logits shape: {from_logits.shape}")
#             print(f"To logits shape: {to_logits.shape}")
#             print(f"Value shape: {value.shape}")
#         else:
#             print(f"Unexpected number of outputs: {len(outputs)}")

# Example usage
if __name__ == "__main__":
    # Create optimized model
    model = OptimizedResNet3D(blocks=15, n_moves=10000, channels=256)
    model_summary(model)

    # Create lightweight model for search
    light_model = LightweightResNet3D(blocks=8, n_moves=10000, channels=128)
    model_summary(light_model)
