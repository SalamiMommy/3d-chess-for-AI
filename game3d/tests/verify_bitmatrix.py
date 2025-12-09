
import sys
import os
import numpy as np

# Adjust path to find game3d
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from game3d.cache.caches.movecache import MoveCache
from game3d.common.shared_types import SIZE, Color, COORD_DTYPE

class MockBoard:
    generation = 0

class MockOccupancyCache:
    def get_positions(self, color): return []
    def get_priest_count(self, color): return 0
    def validate_consistency(self): return True, ""

class MockCacheManager:
    def __init__(self):
        self.board = MockBoard()
        self.occupancy_cache = MockOccupancyCache()

def verify_bitmatrix():
    cm = MockCacheManager()
    mc = MoveCache(cm)
    
    print("Testing implementation...")
    
    # 1. Store moves for White Piece A (Key 10) attacking Target X (Key 100)
    white_piece_key = 10
    target_key = 100
    
    # Create fake moves array: [fx, fy, fz, tx, ty, tz]
    # We need to reverse-engineer keys to coords
    # Key 10: x=10? No key is packed. 
    # Let's use coordinate keys directly from known coords
    
    def to_key(x, y, z):
        return x | (y << 9) | (z << 18)
        
    def to_moves(fx, fy, fz, targets):
        moves = []
        for (tx, ty, tz) in targets:
            moves.append([fx, fy, fz, tx, ty, tz])
        return np.array(moves, dtype=COORD_DTYPE)

    # Piece A at (0,0,0) -> Key 0
    pA_key = to_key(0,0,0)
    
    # Targets (1,1,1) -> Key T1
    t1_coord = (1,1,1)
    t1_key = to_key(*t1_coord)
    
    movesA = to_moves(0,0,0, [t1_coord])
    
    print(f"Storing moves for White Piece A -> {t1_coord}")
    mc.store_piece_moves(Color.WHITE, pA_key, movesA)
    
    # Verify get_pieces_targeting
    targeting = mc.get_pieces_targeting(np.array([t1_key]))
    print(f"Targeting {t1_coord}: {targeting}")
    assert len(targeting) == 1
    assert targeting[0] == (0, pA_key)
    
    # 2. Add another attacker Piece B (White) at (2,2,2) -> T1
    pB_key = to_key(2,2,2)
    movesB = to_moves(2,2,2, [t1_coord])
    
    print(f"Storing moves for White Piece B -> {t1_coord}")
    mc.store_piece_moves(Color.WHITE, pB_key, movesB)
    
    targeting = mc.get_pieces_targeting(np.array([t1_key]))
    print(f"Targeting {t1_coord}: {targeting}")
    assert len(targeting) == 2
    
    # 3. Check has_other_attackers
    t1_coord_arr = np.array(t1_coord)
    
    # Exclude A, check if B is there
    has = mc.has_other_attackers(t1_coord_arr, Color.WHITE, np.array([pA_key], dtype=np.int64))
    print(f"Has other attackers (excluding A): {has}")
    assert has == True
    
    # Exclude A and B
    has = mc.has_other_attackers(t1_coord_arr, Color.WHITE, np.array([pA_key, pB_key], dtype=np.int64))
    print(f"Has other attackers (excluding A, B): {has}")
    assert has == False
    
    # 4. Remove moves for B
    print("Clearing moves for B")
    mc.store_piece_moves(Color.WHITE, pB_key, np.empty((0,6), dtype=COORD_DTYPE))
    
    has = mc.has_other_attackers(t1_coord_arr, Color.WHITE, np.array([pA_key], dtype=np.int64))
    print(f"Has other attackers (excluding A): {has}")
    assert has == False
    
    print("\n✅ Verification SUCCESS!")

if __name__ == "__main__":
    verify_bitmatrix()
