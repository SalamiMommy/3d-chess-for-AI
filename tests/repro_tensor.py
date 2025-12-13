
import numpy as np
import sys
import os

# Adjust path
sys.path.append(os.getcwd())

from game3d.game.factory import start_game_state
from game3d.common.shared_types import N_TOTAL_PLANES, SIZE

def check_tensor():
    state = start_game_state()
    arr = state.board.array()
    
    print(f"Tensor shape: {arr.shape}")
    print(f"Tensor dtype: {arr.dtype}")
    
    # Check if metadata planes are empty
    # Planes: [0..79] = pieces
    # [80..81] = color planes
    # [82] = current player
    # [83..88] = effects
    
    piece_planes = arr[0:80]
    color_planes = arr[80:82]
    current_plane = arr[82:83]
    effect_planes = arr[83:89]
    
    print(f"Piece planes sum: {np.sum(piece_planes)}")
    print(f"Color planes sum: {np.sum(color_planes)}")
    print(f"Current player plane sum: {np.sum(current_plane)}")
    print(f"Effect planes sum: {np.sum(effect_planes)}")
    
    if np.sum(color_planes) == 0:
        print("WARNING: Color planes are empty!")
    if np.sum(current_plane) == 0:
        print("WARNING: Current player plane is empty!")

    # Test sparse reconstruction
    print("\nTesting sparse reconstruction...")
    try:
        from training.tensor_utils import batch_sparse_to_dense_gpu
        import torch
        
        # Extract sparse data
        oc = state.cache_manager.occupancy_cache
        _, _, coords, types, colors = oc.export_buffer_data()
        
        # Create dummy request
        requests = [(0, (coords, types, colors))]
        device = "cpu" # Test on CPU for simplicity
        
        # Reconstruct
        t_state, ids = batch_sparse_to_dense_gpu(requests, device)
        
        # Compare
        # t_state is (1, 89, 9, 9, 9)
        reconstructed = t_state[0].numpy()
        
        print(f"Reconstructed shape: {reconstructed.shape}")
        
        if np.allclose(arr, reconstructed):
            print("SUCCESS: Reconstructed tensor matches Board.array()")
        else:
            print("FAILURE: Mismatch!")
            diff = np.abs(arr - reconstructed)
            print(f"Max diff: {np.max(diff)}")
            print(f"Sum diff: {np.sum(diff)}")

    except ImportError:
        print("Could not import tensor_utils. Skipping test.")
    except Exception as e:
        print(f"Error during reconstruction test: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    check_tensor()
