
import numpy as np
from game3d.game.gamestate import GameState
from game3d.common.shared_types import PieceType, Color, SIZE
from game3d.pieces.pieces.wall import generate_wall_moves

def reproduce():
    print("Initializing GameState...")
    game = GameState.from_startpos()
    
    # Clear board
    game.cache_manager.occupancy_cache.clear()
    
    # Place a White Wall at [5, 7, 5]
    # This is a valid anchor: x=5 (valid < 8), y=7 (valid < 8), z=5
    # It occupies: (5,7,5), (6,7,5), (5,8,5), (6,8,5)
    
    anchor = np.array([5, 7, 5], dtype=np.int16)
    
    # Manually set occupancy for the wall parts
    # We need to set PieceType.WALL
    # And we need to ensure the anchor check passes (left and up neighbors are not WALL)
    
    # Set the wall parts
    parts = [
        (5, 7, 5), (6, 7, 5),
        (5, 8, 5), (6, 8, 5)
    ]
    
    for x, y, z in parts:
        game.cache_manager.occupancy_cache._occ[x, y, z] = Color.WHITE
        # We might need to set the piece type in a way that get_type_at works
        # But generate_wall_moves uses _occ directly for blocking, 
        # and uses batch_get_attributes for anchor validation.
        
    # For anchor validation:
    # is_wall_anchor checks if left (x-1) and up (y-1) are NOT WALL.
    # (4, 7, 5) should not be WALL
    # (5, 6, 5) should not be WALL
    # They are 0 (EMPTY) by default.
    
    # However, generate_wall_moves calls cache_manager.occupancy_cache.batch_get_attributes
    # We need to make sure that returns WALL for the wall parts if we want it to be realistic,
    # but strictly speaking it only checks neighbors.
    
    # Wait, generate_wall_moves takes `pos` which are candidate anchors.
    # It then filters them using `is_anchor` logic inside `generate_wall_moves`.
    # It checks `left_types` and `up_types`.
    
    # We need to mock the cache or set it up correctly.
    # Let's try to use the public API to place the piece if possible, or just hack the cache.
    
    # Hack the cache to return WALL type for the wall parts
    # The cache might use a separate array for types or pack it.
    # Let's check OccupancyCache implementation if needed.
    # But for now, let's assume we can just pass the position to generate_wall_moves
    # and ensure the neighbors are empty.
    
    print(f"Testing Wall at anchor {anchor}")
    
    # Generate moves
    moves = generate_wall_moves(game.cache_manager, Color.WHITE, anchor)
    
    print(f"Generated {len(moves)} moves")
    
    for i in range(len(moves)):
        m = moves[i]
        # m is [fx, fy, fz, tx, ty, tz]
        tx, ty, tz = m[3], m[4], m[5]
        print(f"Move to: [{tx}, {ty}, {tz}]")
        
        # Check validity
        if not (0 <= tx < SIZE - 1 and 0 <= ty < SIZE - 1 and 0 <= tz < SIZE):
            print(f"❌ INVALID ANCHOR: [{tx}, {ty}, {tz}]")
        
        # Check parts
        parts_valid = True
        if not (0 <= tx+1 < SIZE): parts_valid = False
        if not (0 <= ty+1 < SIZE): parts_valid = False
        
        if not parts_valid:
             print(f"❌ PARTS OUT OF BOUNDS for move to [{tx}, {ty}, {tz}]")
             
        if tx == 5 and ty == 8 and tz == 5:
            print("!!! REPRODUCED SPECIFIC ERROR CASE !!!")

if __name__ == "__main__":
    reproduce()
