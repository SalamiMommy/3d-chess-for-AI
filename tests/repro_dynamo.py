
import torch
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.graph_transformer import create_optimized_model, N_TOTAL_PLANES, BOARD_SIZE

def test_recompilation():
    print("Testing Dynamo recompilation behavior...")
    
    # Force cpu for this test to avoid needing ROCm specifically if not available,
    # but the issue is generic to torch.compile.
    # However, torch.compile behavior might depend on the backend.
    # The user is on linux with ROCm.
    # We will try to use cuda if available, else cpu.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = create_optimized_model("small", device=device)
    
    # Ensure model is compiled
    # The create_optimized_model function already calls torch.compile with dynamic=False
    # if torch version >= 2.9 (wait, the code checked for 2.0) and has compile.
    
    # Let's inspect if it is compiled
    # Inspecting is hard, but we can just run it.
    
    print("Running with 10 different batch sizes to hit limit...")
    for i in range(1, 15):
        bs = i
        print(f"Batch size {bs}")
        x = torch.randn(bs, N_TOTAL_PLANES, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE, device=device)
        model(x)

    print("Done.")

if __name__ == "__main__":
    test_recompilation()
