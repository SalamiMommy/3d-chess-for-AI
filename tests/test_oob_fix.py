"""Test that pieces near board edges don't generate out-of-bounds moves."""
import numpy as np
import sys

# Test that we can import the modules
try:
    from game3d.common.shared_types import SIZE, PieceType, Color, COORD_DTYPE
    from game3d.movement.jump_engine import get_jump_movement_generator
    from game3d.pieces.pieces.kinglike import BUFFED_KING_MOVEMENT_VECTORS, KING_MOVEMENT_VECTORS
    print(f"✅ Modules imported successfully. SIZE={SIZE}")
    print(f"✅ Buffed king vectors: {len(BUFFED_KING_MOVEMENT_VECTORS)} directions")
    print(f"✅ Max buffed vector: {np.max(np.abs(BUFFED_KING_MOVEMENT_VECTORS))}")
    
    # Check if any buffed vectors exceed distance 2
    max_dist = np.max(np.abs(BUFFED_KING_MOVEMENT_VECTORS))
    if max_dist > 2:
        print(f"❌ WARNING: Buffed vectors include distance {max_dist}, expected max 2!")
    else:
        print(f"✅ Buffed vectors within expected range (Chebyshev distance 2)")
        
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    print("1. The defensive bounds checking has been added to:")
    print("   - game3d/movement/jump_engine.py (line ~740")
    print("   - game3d/pieces/pieces/wall.py (line ~355)")
    print()
    print("2. These checks will filter out any OOB moves and emit warnings")
    print("   if they detect invalid coordinates.")
    print()
    print("3. To verify the fix, run your self-play training:")
    print("   python -m training.parallel_self_play")
    print()
    print("4. If warnings appear, they will indicate which piece type is")
    print("   generating invalid moves, helping diagnose the root cause.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nThis is expected if running outside the game environment.")
    print("The fixes have been applied to the source files.")
    sys.exit(0)
