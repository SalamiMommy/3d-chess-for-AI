
import torch
import numpy as np
from game3d.common.shared_types import (
    N_PIECE_TYPES, N_COLOR_PLANES, N_TOTAL_PLANES,
    SIZE, Color
)

def batch_sparse_to_dense_gpu(requests, device, model_config=None):
    """
    Reconstruct dense tensor batch from sparse representations directly on GPU.
    
    Args:
        requests: List of (worker_id, (coords, types, colors)) tuples.
        device: Torch device (cuda/cpu).
        model_config: Optional config dict (unused for now, checks consistency).
        
    Returns:
        (state_tensor, worker_ids)
        state_tensor: (B, C, D, H, W) float32 tensor
        worker_ids: List of worker IDs corresponding to batch dimension
    """
    batch_size = len(requests)
    worker_ids = [r[0] for r in requests]
    
    # 1. Concatenate all sparse data into big arrays
    all_coords_list = []
    all_types_list = []
    all_colors_list = []
    batch_indices_list = []
    
    total_pieces = 0
    
    for b_idx, (_, (coords, types, colors)) in enumerate(requests):
        n = coords.shape[0]
        if n > 0:
            all_coords_list.append(coords)
            all_types_list.append(types)
            all_colors_list.append(colors)
            batch_indices = np.full(n, b_idx, dtype=np.int32)
            batch_indices_list.append(batch_indices)
            total_pieces += n
            
    # 2. Allocate output tensor
    # Ensure float32 for model
    state_tensor = torch.zeros(
        (batch_size, N_TOTAL_PLANES, SIZE, SIZE, SIZE),
        dtype=torch.float32,
        device=device
    )
    
    if total_pieces == 0:
        return state_tensor, worker_ids
        
    # 3. Flatten and transfer to GPU
    # Utilize numpy for efficient flattening
    flat_coords = np.concatenate(all_coords_list)
    flat_types = np.concatenate(all_types_list)
    flat_colors = np.concatenate(all_colors_list)
    flat_batch_indices = np.concatenate(batch_indices_list)
    
    # Transfer to GPU (non_blocking=True if pinned, but here we create fresh tensors)
    # OPTIMIZATION: Transfer as smaller dtypes, then cast to long on device
    # This reduces PCIe bandwidth by 2x-8x
    t_coords = torch.from_numpy(flat_coords).to(device) # Keeps int16
    t_types = torch.from_numpy(flat_types).to(device)   # Keeps int8
    t_colors = torch.from_numpy(flat_colors).to(device) # Keeps uint8
    t_batch_idxs = torch.from_numpy(flat_batch_indices).to(device) # Keeps int32
    
    # Now cast to long for indexing
    t_coords = t_coords.long()
    t_types = t_types.long()
    t_colors = t_colors.long()
    t_batch_idxs = t_batch_idxs.long()
    
    # 4. Compute Plane Indices
    # plane_idx = (type - 1) + (color_offset * N_PIECE_TYPES)
    # Color: White=1, Black=2. Offset: White=0, Black=1.
    color_offsets = t_colors - int(Color.WHITE)
    plane_indices = (t_types - 1) + (color_offsets * N_PIECE_TYPES)
    
    # 5. Scatter values
    # We assign 1.0 to: state_tensor[b, plane, x, y, z]
    # We can use advanced indexing or scatter.
    # Advanced indexing:
    x = t_coords[:, 0]
    y = t_coords[:, 1]
    z = t_coords[:, 2]
    
    # Validate bounds (optional but safe)
    # Assuming input is valid from game engine
    
    state_tensor[t_batch_idxs, plane_indices, x, y, z] = 1.0
    
    return state_tensor, worker_ids

