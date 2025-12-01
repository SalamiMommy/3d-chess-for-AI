
import time
import numpy as np
from game3d.cache.manager import OptimizedCacheManager
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE, SIZE
from game3d.pieces.pieces.bomb import generate_bomb_moves
from game3d.pieces.pieces.swapper import generate_swapper_moves
from game3d.pieces.pieces.freezer import get_all_frozen_squares_numpy
from game3d.pieces.pieces.friendlytp import generate_friendlytp_moves
from game3d.pieces.pieces.infiltrator import generate_infiltrator_moves
from game3d.pieces.pieces.mirror import generate_mirror_moves
from game3d.pieces.pieces.reflector import generate_reflecting_bishop_moves
from game3d.pieces.pieces.geomancer import generate_geomancer_moves
from game3d.pieces.pieces.echo import generate_echo_moves
from game3d.pieces.pieces.edgerook import generate_edgerook_moves
from game3d.pieces.pieces.wall import generate_wall_moves

def setup_cache_manager():
    from game3d.board.board import Board
    board = Board.startpos()
    cm = OptimizedCacheManager(board)
    # Populate board with some pieces
    # Add some friendly pieces
    for i in range(20):
        pos = np.array([i % SIZE, (i // SIZE) % SIZE, 0], dtype=COORD_DTYPE)
        cm.occupancy_cache.set_position(pos, np.array([PieceType.PAWN, Color.WHITE]))
    
    # Add some enemy pieces
    for i in range(20):
        pos = np.array([i % SIZE, (i // SIZE) % SIZE, SIZE - 1], dtype=COORD_DTYPE)
        cm.occupancy_cache.set_position(pos, np.array([PieceType.PAWN, Color.BLACK]))
        
    return cm

def benchmark_bomb(cm, n_iter=100):
    # Place bombs
    bombs = np.random.randint(0, SIZE, (100, 3)).astype(COORD_DTYPE)
    
    start = time.time()
    for _ in range(n_iter):
        generate_bomb_moves(cm, Color.WHITE, bombs)
    end = time.time()
    print(f"Bomb (100 pieces, {n_iter} iter): {end - start:.4f}s")

def benchmark_infiltrator(cm, n_iter=100):
    # Place infiltrators
    infiltrators = np.random.randint(0, SIZE, (100, 3)).astype(COORD_DTYPE)
    
    # Place enemy pawns
    enemy_pawns = np.random.randint(0, SIZE, (50, 3)).astype(COORD_DTYPE)
    cm.occupancy_cache.batch_set_positions(enemy_pawns, np.full((50, 2), [PieceType.PAWN, Color.BLACK]))
    
    start = time.time()
    total_moves = 0
    for _ in range(n_iter):
        moves = generate_infiltrator_moves(cm, Color.WHITE, infiltrators)
        total_moves += moves.shape[0]
    end = time.time()
    print(f"Infiltrator (100 pieces, {n_iter} iter): {end - start:.4f}s, Total moves: {total_moves}")

def benchmark_mirror(cm, n_iter=100):
    # Place mirrors
    mirrors = np.random.randint(0, SIZE, (100, 3)).astype(COORD_DTYPE)
    
    start = time.time()
    total_moves = 0
    for _ in range(n_iter):
        # Mirror currently might fail with batch, so we might see an error or weird behavior
        try:
            moves = generate_mirror_moves(cm, Color.WHITE, mirrors)
            total_moves += moves.shape[0]
        except Exception as e:
            print(f"Mirror batch failed: {e}")
            break
    end = time.time()
    print(f"Mirror (100 pieces, {n_iter} iter): {end - start:.4f}s, Total moves: {total_moves}")

def benchmark_reflector(cm, n_iter=100):
    # Place reflectors
    reflectors = np.random.randint(0, SIZE, (100, 3)).astype(COORD_DTYPE)
    
    start = time.time()
    total_moves = 0
    for _ in range(n_iter):
        try:
            moves = generate_reflecting_bishop_moves(cm, Color.WHITE, reflectors)
            total_moves += moves.shape[0]
        except Exception as e:
            print(f"Reflector batch failed: {e}")
            break
    end = time.time()
    print(f"Reflector (100 pieces, {n_iter} iter): {end - start:.4f}s, Total moves: {total_moves}")

def benchmark_geomancer(cm, n_iter=100):
    # Place geomancers
    geomancers = np.random.randint(0, SIZE, (100, 3)).astype(COORD_DTYPE)
    
    start = time.time()
    total_moves = 0
    for _ in range(n_iter):
        try:
            moves = generate_geomancer_moves(cm, Color.WHITE, geomancers)
            total_moves += moves.shape[0]
        except Exception as e:
            print(f"Geomancer batch failed: {e}")
            break
    end = time.time()
    print(f"Geomancer (100 pieces, {n_iter} iter): {end - start:.4f}s, Total moves: {total_moves}")

def benchmark_echo(cm, n_iter=100):
    # Place echoes
    echoes = np.random.randint(0, SIZE, (100, 3)).astype(COORD_DTYPE)
    
    start = time.time()
    total_moves = 0
    for _ in range(n_iter):
        try:
            moves = generate_echo_moves(cm, Color.WHITE, echoes)
            total_moves += moves.shape[0]
        except Exception as e:
            print(f"Echo batch failed: {e}")
            break
    end = time.time()
    print(f"Echo (100 pieces, {n_iter} iter): {end - start:.4f}s, Total moves: {total_moves}")

def benchmark_edgerook(cm, n_iter=100):
    # Place edgerooks on edges
    # Randomly pick edge coordinates
    edge_coords = []
    for _ in range(100):
        # Pick a random face/edge
        axis = np.random.randint(0, 3)
        val = np.random.choice([0, SIZE-1])
        coord = np.random.randint(0, SIZE, 3)
        coord[axis] = val
        edge_coords.append(coord)
    edgerooks = np.array(edge_coords, dtype=COORD_DTYPE)
    
    start = time.time()
    total_moves = 0
    for _ in range(n_iter):
        try:
            moves = generate_edgerook_moves(cm, Color.WHITE, edgerooks)
            total_moves += moves.shape[0]
        except Exception as e:
            print(f"EdgeRook batch failed: {e}")
            break
    end = time.time()
    print(f"EdgeRook (100 pieces, {n_iter} iter): {end - start:.4f}s, Total moves: {total_moves}")

def benchmark_wall(cm, n_iter=100):
    # Place walls (anchors)
    # Ensure anchors are within bounds for 2x2 blocks (0 to SIZE-2)
    walls = np.random.randint(0, SIZE-1, (100, 3)).astype(COORD_DTYPE)
    
    start = time.time()
    total_moves = 0
    for _ in range(n_iter):
        try:
            moves = generate_wall_moves(cm, Color.WHITE, walls)
            total_moves += moves.shape[0]
        except Exception as e:
            print(f"Wall batch failed: {e}")
            break
    end = time.time()
    print(f"Wall (100 pieces, {n_iter} iter): {end - start:.4f}s, Total moves: {total_moves}")

if __name__ == "__main__":
    cm = setup_cache_manager()
    
    # Run benchmarks
    print("Benchmarking...")
    benchmark_infiltrator(cm)
    benchmark_mirror(cm)
    benchmark_reflector(cm)
    benchmark_geomancer(cm)
    benchmark_echo(cm)
    benchmark_edgerook(cm)
    benchmark_wall(cm)
