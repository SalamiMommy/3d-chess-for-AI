"""Debug script to test Wall move generation at y=7."""
import numpy as np

SIZE = 9
COORD_DTYPE = np.int16

# Test the bounds check logic
test_positions = [
    [4, 7, 3],  # Should generate move to [4, 8, 3]?
    [4, 8, 3],  # Should NOT generate any moves (already at edge)
]

for pos in test_positions:
    x, y, z = pos
    print(f"\nTesting Wall at position {pos}:")
    print(f"  Current anchor: ({x}, {y}, {z})")
    print(f"  Occupies: ({x},{y}), ({x+1},{y}), ({x},{y+1}), ({x+1},{y+1})")
    
    # If this wall tries to move +1 in y direction:
    new_y = y + 1
    print(f"\n  If moving +Y to anchor ({x}, {new_y}, {z}):")
    print(f"    Would occupy: ({x},{new_y}), ({x+1},{new_y}), ({x},{new_y+1}), ({x+1},{new_y+1})")
    
    # Check bounds for the Numba kernel
    kernel_check = (0 <= x < SIZE - 1 and 0 <= new_y < SIZE - 1 and 0 <= z < SIZE)
    print(f"    Kernel bounds check (0 <= {new_y} < {SIZE-1}): {kernel_check}")
    
    # Check bounds for defensive filter
    filter_check = (0 <= x < SIZE - 1 and 0 <= new_y < SIZE - 1 and 0 <= z < SIZE)
    print(f"    Defensive filter check: {filter_check}")
    
    if new_y + 1 >= SIZE:
        print(f"    ❌ Would place square at y={new_y+1} (out of bounds!)")
    else:
        print(f"    ✓ All squares in bounds")

print("\n" + "="*60)
print("ANALYSIS:")
print("="*60)
print("Wall at [4, 7, 3] moving +1 in Y:")
print("  - New anchor: [4, 8, 3]")
print(f"  - Kernel check: 0 <= 8 < 8 = FALSE ❌")
print(f"  - Should be filtered by kernel")
print()
print("If the move is still being generated, the issue is:")
print("  1. Numba kernel caching (old compiled version)")
print("  2. Buffed move using different vectors")
print("  3. Move cache not invalidated")
