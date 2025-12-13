
import sys
import numpy as np
import logging
from game3d.game.gamestate import GameState
from game3d.common.shared_types import Color, PieceType, PIECE_TYPE_DTYPE
from game3d.game.turnmove import make_move
from game3d.movement.generator import generate_legal_moves, initialize_generator

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def run_repro():
    print("Initializing Generator...")
    initialize_generator()
    
    print("Initializing GameState...")
    state = GameState.from_startpos()
    
    # Clear board
    print("Clearing board...")
    state.cache_manager.occupancy_cache.clear()
    state.cache_manager.move_cache.clear()
    
    # Add Kings to satisfy safety checks
    print("Placing White King at [0, 0, 0]...")
    king_white_pos = np.array([0, 0, 0], dtype=np.int16)
    king_white_data = np.array([PieceType.KING.value, Color.WHITE.value], dtype=PIECE_TYPE_DTYPE)
    state.cache_manager.occupancy_cache.set_position(king_white_pos, king_white_data)

    print("Placing Black King at [7, 7, 7]...")
    king_black_pos = np.array([7, 7, 7], dtype=np.int16)
    king_black_data = np.array([PieceType.KING.value, Color.BLACK.value], dtype=PIECE_TYPE_DTYPE)
    state.cache_manager.occupancy_cache.set_position(king_black_pos, king_black_data)

    # Place White ECHO at [3, 0, 0]
    print("Placing White Echo at [3, 0, 0]...")
    echo_pos = np.array([3, 0, 0], dtype=np.int16)
    
    # Use correct type for data
    piece_data = np.array([PieceType.ECHO.value, Color.WHITE.value], dtype=PIECE_TYPE_DTYPE)
                       
    state.cache_manager.occupancy_cache.set_position(echo_pos, piece_data)
    
    # DEBUG: Inspect export_buffer_data
    print("DEBUG: exporting buffer data...")
    occ, ptype, coords, types, colors = state.cache_manager.occupancy_cache.export_buffer_data()
    print(f"Buffer Coords Shape: {coords.shape}")
    print(f"Buffer Types Shape: {types.shape}")
    print(f"Buffer Colors Shape: {colors.shape}")
    
    if coords.shape[0] > 0:
        print(f"First element: Coord={coords[0]}, Type={types[0]}, Color={colors[0]}")
    else:
        print("BUFFER IS EMPTY!")

    # Verify Occ Grid directly
    pt, pc = state.cache_manager.occupancy_cache.get_fast(echo_pos)
    print(f"Direct Get Fast checked at {echo_pos}: Type={pt}, Color={pc}")
    
    # DEBUG: Check Echo Vectors
    try:
        from game3d.pieces.pieces.echo import _ECHO_DIRECTIONS
        print(f"DEBUG: _ECHO_DIRECTIONS length: {len(_ECHO_DIRECTIONS)}")
        
        # Check specific vectors relative to pos [3,0,0]
        # Target [5, 1, 3] -> delta [2, 1, 3]
        d1 = np.array([2, 1, 3])
        if np.any(np.all(_ECHO_DIRECTIONS == d1, axis=1)):
             print(f"Vector {d1} FOUND in _ECHO_DIRECTIONS")
        else:
             print(f"Vector {d1} NOT found in _ECHO_DIRECTIONS")
             
        # Target [5, 1, 1] -> delta [2, 1, 1] (Should be valid: Anchor [2,0,0] + Bubble [0,1,1])
        d2 = np.array([2, 1, 1])
        if np.any(np.all(_ECHO_DIRECTIONS == d2, axis=1)):
             print(f"Vector {d2} FOUND in _ECHO_DIRECTIONS")
        else:
             print(f"Vector {d2} NOT found in _ECHO_DIRECTIONS")
             
    except Exception as e:
        print(f"Failed to inspect Echo vectors: {e}")

    # Generate moves
    print("Generating moves for Echo...")
    moves = generate_legal_moves(state)
    print(f"Generated {len(moves)} moves.")
    
    # Check for [5, 1, 1] as valid target
    target = np.array([5, 1, 1])
    found = False
    for m in moves:
        if np.array_equal(m[3:], target):
            found = True
            break
            
    if not found:
        print(f"Expected move {echo_pos}->{target} NOT found!")
        # Print all moves
        print("Moves found:")
        for m in moves:
            print(m)
        # sys.exit(1) # Don't exit yet, let's see what happens
    else:
        print(f"Found target move {echo_pos}->{target}. Executing...")
        # Execute move
        move_arr = np.concatenate([echo_pos, target])
        state = make_move(state, move_arr)
        print("Move executed.")
    
    # Check if we can generate moves again (simulate game loop continuing)
    print("Generating moves for next turn (Black)...")
    try:
        moves_black = generate_legal_moves(state)
        print(f"Generated {len(moves_black)} moves for Black.")
    except Exception as e:
        print(f"Crash during Black move generation: {e}")
        raise

    state.color = Color.WHITE
    print("Switching to White, generating moves...")
    try:
        moves_white = generate_legal_moves(state)
        print(f"Generated {len(moves_white)} moves for White.")
    except Exception as e:
        print(f"Crash during White move generation: {e}")
        raise
        
    print("Reproduction successful (no crash observed).")

if __name__ == "__main__":
    run_repro()
