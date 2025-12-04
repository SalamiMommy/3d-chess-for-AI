
import numpy as np
import random
from game3d.game.gamestate import GameState
from game3d.board.board import Board
from game3d.cache.manager import get_cache_manager
from game3d.common.shared_types import Color, PieceType, SIZE, COORD_DTYPE
from game3d.attacks.check import _square_attacked_by_slow
from game3d.attacks.fast_attack import square_attacked_by_fast

def verify_fast_attack():
    print("Verifying fast_attack correctness...")
    board = Board()
    cache_manager = get_cache_manager(board)
    
    def add_piece(pos, ptype, color):
        coord = np.array(pos, dtype=COORD_DTYPE)
        piece_data = np.array([ptype, color], dtype=np.int32)
        cache_manager.occupancy_cache.set_position(coord, piece_data)
        
    # Test single pieces
    piece_types = [
        PieceType.PAWN, PieceType.KNIGHT, PieceType.BISHOP, 
        PieceType.ROOK, PieceType.QUEEN, PieceType.KING,
        PieceType.TRIGONALBISHOP
    ]
    
    mismatches = 0
    checked = 0
    for ptype in piece_types:
        print(f"Testing {ptype.name}...")
        board = Board()
        cache_manager = get_cache_manager(board)
        
        for _ in range(50):
            # Clear board
            cache_manager.occupancy_cache._occ.fill(0)
            cache_manager.occupancy_cache._ptype.fill(0)
            
            # Place attacker
            attacker_pos = np.array([random.randint(0, 8) for _ in range(3)], dtype=COORD_DTYPE)
            attacker_color = random.choice([Color.WHITE, Color.BLACK])
            add_piece(attacker_pos, ptype, attacker_color)
            
            # Place target (as a piece of opposite color to attacker)
            target = np.array([random.randint(0, 8) for _ in range(3)], dtype=COORD_DTYPE)
            
            # Ensure target is not same as attacker
            if np.array_equal(target, attacker_pos):
                continue
                
            # Place dummy piece at target so pawn captures work
            # Attacker color is 'attacker_color'. Target should be opposite.
            target_color = attacker_color.opposite()
            # Use a Pawn as dummy target
            target_data = np.array([PieceType.PAWN.value, target_color.value], dtype=np.int32)
            cache_manager.occupancy_cache.set_position(target, target_data)
            
            slow = _square_attacked_by_slow(board, target, attacker_color.value, cache_manager)
            fast = square_attacked_by_fast(board, target, attacker_color.value, cache_manager)
            
            if slow != fast:
                print(f"MISMATCH for {ptype.name} at {attacker_pos} attacking {target}")
                print(f"Slow: {slow}, Fast: {fast}")
                mismatches += 1
            checked += 1
                
    # Test with blockers
    print("Testing with blockers...")
    for _ in range(50):
        board = Board()
        cache_manager = get_cache_manager(board)
        
        # Attacker (Rook)
        attacker_pos = np.array([0, 0, 0], dtype=COORD_DTYPE)
        add_piece(attacker_pos, PieceType.ROOK, Color.WHITE)
        
        # Target
        target = np.array([0, 0, 5], dtype=COORD_DTYPE)
        
        # Blocker
        blocker_pos = np.array([0, 0, 3], dtype=COORD_DTYPE)
        add_piece(blocker_pos, PieceType.PAWN, Color.BLACK)
        
        slow = _square_attacked_by_slow(board, target, Color.WHITE.value, cache_manager)
        fast = square_attacked_by_fast(board, target, Color.WHITE.value, cache_manager)
        
        if slow != fast:
             print(f"MISMATCH with blocker: Slow {slow}, Fast {fast}")
             mismatches += 1
            
    if mismatches == 0:
        print(f"✅ SUCCESS: {checked} checks passed.")
    else:
        print(f"❌ FAILURE: {mismatches} mismatches found.")

if __name__ == "__main__":
    verify_fast_attack()
