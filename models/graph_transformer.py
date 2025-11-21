# models/graph_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
import warnings

# Import N_TOTAL_PLANES from shared_types
try:
    from game3d.common.shared_types import N_TOTAL_PLANES, SIZE as BOARD_SIZE
except ImportError:
    # Fallback values if import fails
    N_TOTAL_PLANES = 89
    BOARD_SIZE = 9

def setup_rocm_optimizations():
    """Setup ROCm-specific optimizations."""
    if torch.cuda.is_available() and torch.version.hip:
        # Use new API (PyTorch 2.9+) to avoid deprecation warnings
        # Configure TF32 for matmul operations
        if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
            torch.backends.cuda.matmul.fp32_precision = 'ieee'
        
        # Configure TF32 for convolution operations (if applicable)
        if hasattr(torch.backends.cudnn.conv, 'fp32_precision'):
            torch.backends.cudnn.conv.fp32_precision = 'ieee'
        
        # ROCm memory optimizations
        torch.cuda.set_per_process_memory_fraction(0.85)  # Leave some headroom
        print("ROCm optimizations enabled")

class GraphAttention(nn.Module):
    """Optimized multi-head attention with flash attention support."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Try to use flash attention if available
        self.use_flash_attention = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(t.shape[0], -1, self.heads, t.shape[-1] // self.heads).transpose(1, 2), qkv)

        if self.use_flash_attention and mask is None:
            # Use PyTorch's optimized attention
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
        else:
            # Manual attention implementation
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            if mask is not None:
                dots.masked_fill_(mask == 0, -1e9)
            attn = dots.softmax(dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(x.shape[0], -1, self.heads * (x.shape[-1] // self.heads))
        return self.to_out(out)

class FeedForward(nn.Module):
    """Gated feed-forward network with GLU activation."""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class GraphTransformerBlock(nn.Module):
    """Graph Transformer block with pre-normalization."""

    def __init__(self, dim: int, heads: int, dim_head: int, ff_mult: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GraphAttention(dim, heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_mult, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x

class PositionalEncoding3D(nn.Module):
    """3D positional encoding for spatial coordinates."""

    def __init__(self, dim: int, max_size: int = 9):
        super().__init__()
        self.dim = dim
        self.max_size = max_size

        # Create positional embeddings for each dimension
        pe_x = torch.zeros(max_size, dim)
        pe_y = torch.zeros(max_size, dim)
        pe_z = torch.zeros(max_size, dim)

        position_x = torch.arange(0, max_size, dtype=torch.float).unsqueeze(1)
        position_y = torch.arange(0, max_size, dtype=torch.float).unsqueeze(1)
        position_z = torch.arange(0, max_size, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe_x[:, 0::2] = torch.sin(position_x * div_term)
        pe_x[:, 1::2] = torch.cos(position_x * div_term)
        pe_y[:, 0::2] = torch.sin(position_y * div_term)
        pe_y[:, 1::2] = torch.cos(position_y * div_term)
        pe_z[:, 0::2] = torch.sin(position_z * div_term)
        pe_z[:, 1::2] = torch.cos(position_z * div_term)

        self.register_buffer('pe_x', pe_x)
        self.register_buffer('pe_y', pe_y)
        self.register_buffer('pe_z', pe_z)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to coordinates.

        Args:
            coords: Tensor of shape (batch_size, num_nodes, 3) with x,y,z coordinates

        Returns:
            Tensor of shape (batch_size, num_nodes, dim) with positional encodings
        """
        x_idx = coords[..., 0].long()  # (batch_size, num_nodes)
        y_idx = coords[..., 1].long()  # (batch_size, num_nodes)
        z_idx = coords[..., 2].long()  # (batch_size, num_nodes)

        # Get positional encodings for each dimension
        pe_x = self.pe_x[x_idx]  # (batch_size, num_nodes, dim)
        pe_y = self.pe_y[y_idx]  # (batch_size, num_nodes, dim)
        pe_z = self.pe_z[z_idx]  # (batch_size, num_nodes, dim)

        # Combine positional encodings (simple addition works well)
        return pe_x + pe_y + pe_z

class GraphTransformer3D(nn.Module):
    """Graph Transformer for 3D chess with optimized memory usage."""

    def __init__(
        self,
        dim: int = 512,
        depth: int = 12,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = True,
        use_flash_attention: bool = True,
        num_piece_types: int = 40,
        num_channels: int = None,  # Auto-detect from N_TOTAL_PLANES
        SIZE: int = None  # Auto-detect from BOARD_SIZE
    ):
        super().__init__()
        self.dim = dim
        # Use provided values or auto-detect from shared_types
        self.SIZE = SIZE if SIZE is not None else BOARD_SIZE
        num_channels = num_channels if num_channels is not None else N_TOTAL_PLANES
        self.num_nodes = self.SIZE ** 3  # 729 nodes for 9x9x9 board
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Input projection - convert board tensor to node features
        self.input_proj = nn.Sequential(
            nn.Linear(num_channels, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )

        # 3D positional encoding
        self.pos_encoding = PositionalEncoding3D(dim, max_size=self.SIZE)

        # Graph transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerBlock(dim, heads, dim_head, ff_mult, dropout)
            for _ in range(depth)
        ])

        # Output heads
        self.from_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, self.num_nodes)  # 729 output classes for from squares
        )

        self.to_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, self.num_nodes)  # 729 output classes for to squares
        )

        self.value_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Tanh()
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _create_coordinate_grid(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create 3D coordinate grid for positional encoding."""
        coords = torch.stack(torch.meshgrid(
            torch.arange(self.SIZE, device=device),
            torch.arange(self.SIZE, device=device),
            torch.arange(self.SIZE, device=device),
            indexing='ij'
        ), dim=-1).float()  # (9, 9, 9, 3)

        coords = coords.reshape(-1, 3)  # (729, 3)
        coords = coords.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, 729, 3)
        return coords

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Graph Transformer.

        Args:
            x: Input tensor of shape (batch_size, num_channels, 9, 9, 9)

        Returns:
            from_logits: Logits for from squares (batch_size, 729)
            to_logits: Logits for to squares (batch_size, 729)
            value_pred: Value prediction (batch_size, 1)
        """
        batch_size = x.shape[0]
        device = x.device

        # Clone input immediately to prevent CUDA Graphs overwrites
        # This is critical when using torch.compile() with CUDA Graphs
        x = x.clone()

        # Reshape input: (batch_size, num_channels, 9, 9, 9) -> (batch_size, 729, num_channels)
        x_flat = x.reshape(batch_size, x.shape[1], -1).transpose(1, 2).contiguous()

        # Project input features
        node_features = self.input_proj(x_flat)  # (batch_size, 729, dim)

        # Add positional encoding - create fresh coordinate grid to avoid CUDA Graphs issues
        coords = self._create_coordinate_grid(batch_size, device)
        pos_enc = self.pos_encoding(coords)  # (batch_size, 729, dim)

        # Combine features and positional encoding
        x = node_features + pos_enc  # (batch_size, 729, dim)

        # Apply transformer layers
        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        # Global pooling for value head
        global_features = x.mean(dim=1)  # (batch_size, dim)

        # Output heads
        from_logits = self.from_head(x).mean(dim=1)  # (batch_size, 729)
        to_logits = self.to_head(x).mean(dim=1)      # (batch_size, 729)
        value_pred = self.value_head(global_features)  # (batch_size, 1)

        return from_logits, to_logits, value_pred

def create_optimized_model(
    model_size: str = "large",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> GraphTransformer3D:
    """Create optimized Graph Transformer model based on size specification."""

    configs = {
        "small": {
            "dim": 384,
            "depth": 8,
            "heads": 6,
            "dim_head": 64,
            "ff_mult": 4,
            "dropout": 0.1,
            "use_gradient_checkpointing": True
        },
        "default": {
            "dim": 512,
            "depth": 12,
            "heads": 8,
            "dim_head": 64,
            "ff_mult": 4,
            "dropout": 0.1,
            "use_gradient_checkpointing": True
        },
        "large": {
            "dim": 896,  # Increased from 768 for better capacity
            "depth": 20,  # Increased from 16 for deeper learning
            "heads": 14,  # Increased from 12 for more attention capacity
            "dim_head": 64,
            "ff_mult": 4,
            "dropout": 0.1,
            "use_gradient_checkpointing": True
        }
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")

    config = configs[model_size]

    # Adjust for available VRAM
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        if vram_gb < 16:
            # Scale down for lower VRAM
            config["dim"] = max(256, config["dim"] // 2)
            config["depth"] = max(6, config["depth"] - 2)
            config["heads"] = max(4, config["heads"] - 2)
            print(f"Scaled down model for {vram_gb:.1f}GB VRAM")
        elif vram_gb >= 24:
            # Use full configuration for 24GB+ VRAM
            print(f"Using full model configuration for {vram_gb:.1f}GB VRAM")

    model = GraphTransformer3D(**config)

    # Model memory optimization
    if torch.cuda.is_available():
        # Enable memory efficient attention
        if hasattr(F, 'scaled_dot_product_attention'):
            print("Using memory efficient attention")

        # Set model to use less memory
        model = model.to(device)
        
        # Apply torch.compile for ROCm optimization (PyTorch 2.0+)
        if hasattr(torch, 'compile') and torch.__version__ >= '2.0':
            try:
                # Use default mode for better stability on ROCm
                model = torch.compile(
                    model,
                    mode='default',
                    fullgraph=False,
                    dynamic=False,
                )
                print("Model compiled with torch.compile (default mode for ROCm)")
            except Exception as e:
                print(f"Warning: Could not compile model: {e}")

        # Print model size
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Created {model_size} Graph Transformer with {param_count:,} parameters")
        print(f"Model configuration: {config}")

    return model

class MemoryOptimizedWrapper(nn.Module):
    """Wrapper for memory optimization techniques."""

    def __init__(self, model: GraphTransformer3D):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Mark CUDA graph step boundary to prevent tensor overwriting
        # This is critical when using torch.compile() with CUDA Graphs
        if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
            torch.compiler.cudagraph_mark_step_begin()
        
        # Use mixed precision if available
        if torch.cuda.is_available() and x.dtype == torch.float32:
            with torch.amp.autocast('cuda'):
                return self.model(x)
        else:
            return self.model(x)

def create_memory_optimized_model(model_size: str = "large", device: torch.device = None) -> MemoryOptimizedWrapper:
    """Create memory-optimized model wrapper."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_optimized_model(model_size, device)
    return MemoryOptimizedWrapper(model)

# Utility functions for model management
def save_model_checkpoint(model: nn.Module, path: str, optimizer: torch.optim.Optimizer = None, scheduler=None, epoch: int = 0):
    """Save model checkpoint with optimization for ROCm."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # Use efficient saving for ROCm
    torch.save(checkpoint, path, _use_new_zipfile_serialization=True)

def load_model_checkpoint(model: nn.Module, path: str, device: torch.device, optimizer: torch.optim.Optimizer = None, scheduler=None):
    """Load model checkpoint with ROCm compatibility."""
    checkpoint = torch.load(path, map_location=device)

    # Handle potential weight mismatches
    model_state_dict = checkpoint['model_state_dict']
    current_state_dict = model.state_dict()

    # Filter out incompatible keys
    filtered_state_dict = {k: v for k, v in model_state_dict.items() if k in current_state_dict and v.shape == current_state_dict[k].shape}

    model.load_state_dict(filtered_state_dict, strict=False)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint.get('epoch', 0)

# Test function
def test_model_memory_usage():
    """Test model memory usage on available device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing model on: {device}")

    # Test different model sizes
    for size in ["small", "default", "large"]:
        print(f"\nTesting {size} model...")

        # Create model
        model = create_optimized_model(size, device)

        # Test forward pass with typical batch size
        batch_size = 16 if size == "small" else 8 if size == "default" else 4
        dummy_input = torch.randn(batch_size, N_TOTAL_PLANES, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE, device=device)

        # Memory usage before
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

        # Forward pass
        with torch.no_grad():
            from_logits, to_logits, value = model(dummy_input)

        # Memory usage after
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            memory_used = (final_memory - initial_memory) / (1024**3)  # GB
            print(f"Batch size {batch_size}: {memory_used:.2f} GB memory used")

        print(f"Output shapes: from{from_logits.shape}, to{to_logits.shape}, value{value.shape}")

if __name__ == "__main__":
    setup_rocm_optimizations()
    test_model_memory_usage()
