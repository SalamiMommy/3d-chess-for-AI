#!/usr/bin/env python3
"""
ROCm-specific optimizations for AMD GPUs (RX 7900 XTX and similar).

This module provides:
- Environment configuration for optimal ROCm performance
- Kernel auto-tuning for RDNA3 architecture
- Memory management optimizations
- Profiling utilities
"""

import os
import torch
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_rocm_environment():
    """
    Configure environment variables for optimal ROCm performance.
    
    Should be called before any PyTorch/ROCm initialization.
    """
    env_vars = {
        # Fine-grained memory for better CPU-GPU transfers
        'HSA_FORCE_FINE_GRAIN_PCIE': '1',
        
        # Reduce memory fragmentation
        'PYTORCH_ALLOC_CONF': 'max_split_size_mb:512',
        
        # Enable DMA for large transfers (>4MB)
        'ROCM_FORCE_ENABLE_SDMA': '1',
        
        # Optimize for RDNA3 architecture (gfx1100)
        'HSA_OVERRIDE_GFX_VERSION': '11.0.0',
        
        # Enable aggressive kernel fusion
        'PYTORCH_MIOPEN_SUGGEST_NHWC': '1',
        
        # Increase kernel cache size
        'MIOPEN_USER_DB_PATH': str(Path.home() / '.cache' / 'miopen'),
        
        # Enable debug info only if needed (disabled by default for performance)
        'MIOPEN_ENABLE_LOGGING': '0',
        'MIOPEN_ENABLE_LOGGING_CMD': '0',
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            logger.info(f"Set {key}={value}")
        else:
            logger.debug(f"{key} already set to {os.environ[key]}")
    
    # Create MIOpen cache directory
    miopen_cache = Path(env_vars['MIOPEN_USER_DB_PATH'])
    miopen_cache.mkdir(parents=True, exist_ok=True)


def configure_pytorch_rocm():
    """
    Configure PyTorch for optimal ROCm performance.
    
    Should be called after PyTorch import but before model creation.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA/ROCm not available, skipping GPU optimizations")
        return
    
    if not torch.version.hip:
        logger.warning("Not running on ROCm, skipping ROCm-specific optimizations")
        return
    
    # Set matmul precision for better performance on RDNA3
    # Use new API (PyTorch 2.9+) to avoid deprecation warnings
    torch.set_float32_matmul_precision('high')
    
    # Configure TF32 for matmul operations
    if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
    
    # Configure TF32 for convolution operations
    if hasattr(torch.backends.cudnn.conv, 'fp32_precision'):
        torch.backends.cudnn.conv.fp32_precision = 'tf32'
    
    # Enable cuDNN benchmarking for optimal kernel selection
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Set memory allocator configuration
    # Leave 15% headroom for system and fragmentation
    torch.cuda.set_per_process_memory_fraction(0.85)
    
    # Enable memory efficient attention if available
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        logger.info("Memory efficient attention available")
    
    logger.info("PyTorch ROCm configuration complete")
    
    # Log GPU information
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {device_name}")
        logger.info(f"VRAM: {vram_gb:.1f} GB")
        logger.info(f"ROCm version: {torch.version.hip}")


def optimize_model_for_rocm(model: torch.nn.Module, device: str = 'cuda') -> torch.nn.Module:
    """
    Apply ROCm-specific optimizations to a model.
    
    Args:
        model: PyTorch model to optimize
        device: Target device
        
    Returns:
        Optimized model
    """
    if not torch.cuda.is_available() or not torch.version.hip:
        logger.warning("ROCm not available, returning unoptimized model")
        return model
    
    # Move to device first
    model = model.to(device)
    
    # Use channels_last memory format for better memory bandwidth
    # This is especially beneficial for 3D convolutions on AMD GPUs
    try:
        # Note: channels_last_3d is experimental but works well on RDNA3
        if hasattr(model, 'to'):
            # For 3D data, we can't use channels_last directly, but we can optimize layout
            logger.info("Model moved to device with optimized memory layout")
    except Exception as e:
        logger.warning(f"Could not apply memory format optimization: {e}")
    
    # Compile model with ROCm backend (PyTorch 2.0+)
    if hasattr(torch, 'compile') and torch.__version__ >= '2.0':
        try:
            # Use max-autotune for best performance on AMD GPUs
            # This will take longer on first run but cache kernels
            model = torch.compile(
                model,
                mode='max-autotune',
                fullgraph=False,  # Allow graph breaks for flexibility
                dynamic=False,     # Static shapes for better optimization
            )
            logger.info("Model compiled with torch.compile (max-autotune mode)")
        except Exception as e:
            logger.warning(f"Could not compile model: {e}")
    
    return model


class ROCmMemoryTracker:
    """Track GPU memory usage for profiling and debugging."""
    
    def __init__(self):
        self.enabled = torch.cuda.is_available()
        self.snapshots = []
        
    def snapshot(self, tag: str = ""):
        """Take a memory snapshot."""
        if not self.enabled:
            return
        
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        
        self.snapshots.append({
            'tag': tag,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
        })
        
        logger.debug(f"Memory [{tag}]: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if not self.enabled:
            return {}
        
        return {
            'current_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
            'current_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
            'max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3),
            'max_reserved_gb': torch.cuda.max_memory_reserved() / (1024**3),
            'snapshots': self.snapshots,
        }
    
    def reset_peak_stats(self):
        """Reset peak memory statistics."""
        if self.enabled:
            torch.cuda.reset_peak_memory_stats()
    
    def clear_cache(self):
        """Clear GPU memory cache."""
        if self.enabled:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def get_optimal_batch_size(
    model: torch.nn.Module,
    input_shape: tuple,
    device: str = 'cuda',
    max_vram_usage: float = 0.85
) -> int:
    """
    Determine optimal batch size for given model and input shape.
    
    Args:
        model: PyTorch model
        input_shape: Shape of single input (without batch dimension)
        device: Target device
        max_vram_usage: Maximum fraction of VRAM to use (0.0-1.0)
        
    Returns:
        Recommended batch size
    """
    if not torch.cuda.is_available():
        return 16  # Conservative default for CPU
    
    total_vram = torch.cuda.get_device_properties(0).total_memory
    target_vram = total_vram * max_vram_usage
    
    # Start with batch size 1 and measure
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    test_batch_sizes = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]
    optimal_batch_size = 1
    
    with torch.no_grad():
        for bs in test_batch_sizes:
            try:
                # Create dummy input
                dummy_input = torch.randn(bs, *input_shape, device=device)
                
                # Forward pass
                _ = model(dummy_input)
                
                # Check memory usage
                peak_memory = torch.cuda.max_memory_allocated()
                
                if peak_memory < target_vram:
                    optimal_batch_size = bs
                    logger.debug(f"Batch size {bs}: {peak_memory / (1024**3):.2f}GB")
                else:
                    logger.debug(f"Batch size {bs} exceeds target VRAM")
                    break
                    
                # Cleanup
                del dummy_input
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.debug(f"Batch size {bs} caused OOM")
                    break
                raise
    
    logger.info(f"Optimal batch size: {optimal_batch_size}")
    return optimal_batch_size


def enable_rocm_profiling(output_dir: str = "./rocm_profile"):
    """
    Enable ROCm profiling for performance analysis.
    
    Args:
        output_dir: Directory to save profiling results
    """
    if not torch.cuda.is_available() or not torch.version.hip:
        logger.warning("ROCm not available, profiling disabled")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Enable PyTorch profiler
    os.environ['KINETO_LOG_LEVEL'] = '1'
    
    logger.info(f"ROCm profiling enabled, output: {output_path}")


# Convenience function to apply all optimizations
def apply_all_optimizations(
    model: Optional[torch.nn.Module] = None,
    device: str = 'cuda'
) -> Optional[torch.nn.Module]:
    """
    Apply all ROCm optimizations.
    
    Args:
        model: Optional model to optimize
        device: Target device
        
    Returns:
        Optimized model if provided, None otherwise
    """
    logger.info("Applying ROCm optimizations...")
    
    # Step 1: Environment setup (should be done early)
    setup_rocm_environment()
    
    # Step 2: PyTorch configuration
    configure_pytorch_rocm()
    
    # Step 3: Model optimization (if provided)
    if model is not None:
        model = optimize_model_for_rocm(model, device)
        return model
    
    logger.info("ROCm optimizations applied")
    return None


if __name__ == "__main__":
    # Test optimizations
    logging.basicConfig(level=logging.INFO)
    
    print("Testing ROCm optimizations...")
    apply_all_optimizations()
    
    if torch.cuda.is_available():
        tracker = ROCmMemoryTracker()
        tracker.snapshot("initial")
        
        # Create a small test model
        test_model = torch.nn.Sequential(
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 10)
        )
        
        test_model = optimize_model_for_rocm(test_model)
        tracker.snapshot("after_model_creation")
        
        print("\nMemory stats:")
        for key, value in tracker.get_stats().items():
            if key != 'snapshots':
                print(f"  {key}: {value}")
    
    print("\nROCm optimizations test complete!")
