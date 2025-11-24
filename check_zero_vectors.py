#!/usr/bin/env python3
"""Check all jumping pieces for (0, 0, 0) direction vectors."""

import numpy as np
import sys
import importlib

# List of pieces that use jump_engine with their direction vector names
JUMP_PIECES = [
    ("game3d.pieces.pieces.bomb", "BOMB_MOVEMENT_VECTORS"),
    ("game3d.pieces.pieces.kinglike", "KING_DIRECTIONS"),
    ("game3d.pieces.pieces.knight", "KNIGHT_DIRECTIONS"),
    ("game3d.pieces.pieces.bigknights", "KNIGHT_3_2_DIRECTIONS"),
    ("game3d.pieces.pieces.bigknights", "KNIGHT_3_1_DIRECTIONS"),
    ("game3d.pieces.pieces.archer", "_KING_DIRECTIONS"),
    ("game3d.pieces.pieces.archer", "_ARCHERY_DIRECTIONS"),
    ("game3d.pieces.pieces.pawn", "WHITE_PAWN_PUSH"),
    ("game3d.pieces.pieces.pawn", "BLACK_PAWN_PUSH"),
    ("game3d.pieces.pieces.pawn", "WHITE_PAWN_TWO_STEP"),
    ("game3d.pieces.pieces.pawn", "BLACK_PAWN_TWO_STEP"),
    ("game3d.pieces.pieces.pawn", "WHITE_PAWN_CAPTURES"),
    ("game3d.pieces.pieces.pawn", "BLACK_PAWN_CAPTURES"),
    ("game3d.pieces.pieces.freezer", "KING_DIRECTIONS"),
    ("game3d.pieces.pieces.mirror", "MIRROR_DIRECTIONS"),
    ("game3d.pieces.pieces.nebula", "NEBULA_DIRECTIONS"),
    ("game3d.pieces.pieces.echo", "ECHO_DIRECTIONS"),
    ("game3d.pieces.pieces.panel", "PANEL_DIRECTIONS"),
    ("game3d.pieces.pieces.orbiter", "ORBITER_OFFSETS"),
    ("game3d.pieces.pieces.infiltrator", "STANDARD_KING_DIRECTIONS"),
    ("game3d.pieces.pieces.spiral", "SPIRAL_OFFSETS"),
]

def check_zero_vector(module_name, vector_name):
    """Check if a direction vector contains (0, 0, 0)."""
    try:
        module = importlib.import_module(module_name)
        if not hasattr(module, vector_name):
            return f"⚠️  {module_name}.{vector_name}: NOT FOUND"
        
        vectors = getattr(module, vector_name)
        if not isinstance(vectors, np.ndarray):
            return f"⚠️  {module_name}.{vector_name}: Not a numpy array"
        
        # Check for zero vectors
        zero_mask = (vectors[:, 0] == 0) & (vectors[:, 1] == 0) & (vectors[:, 2] == 0)
        has_zero = np.any(zero_mask)
        
        if has_zero:
            count = np.sum(zero_mask)
            return f"❌ {module_name}.{vector_name}: Contains {count} zero vector(s)"
        else:
            return f"✅ {module_name}.{vector_name}: Clean (no zero vectors)"
    
    except Exception as e:
        return f"⚠️  {module_name}.{vector_name}: Error - {str(e)}"

def main():
    print("Checking all jumping pieces for (0, 0, 0) direction vectors...\n")
    
    issues_found = []
    clean = []
    errors = []
    
    for module_name, vector_name in JUMP_PIECES:
        result = check_zero_vector(module_name, vector_name)
        print(result)
        
        if result.startswith("❌"):
            issues_found.append(result)
        elif result.startswith("✅"):
            clean.append(result)
        else:
            errors.append(result)
    
    print("\n" + "="*80)
    print("SUMMARY:")
    print(f"  Clean: {len(clean)}")
    print(f"  Issues: {len(issues_found)}")
    print(f"  Errors: {len(errors)}")
    
    if issues_found:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues_found:
            print(f"  {issue}")
        return 1
    
    if errors:
        print("\n⚠️  ERRORS:")
        for error in errors:
            print(f"  {error}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
