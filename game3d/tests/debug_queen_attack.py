
import numpy as np
from game3d.common.shared_types import COORD_DTYPE

# Copy vectors
ROOK_VECTORS = np.array([
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
], dtype=COORD_DTYPE)

BISHOP_VECTORS = np.array([
    [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
    [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
    [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1]
], dtype=COORD_DTYPE)

QUEEN_VECTORS = np.concatenate((ROOK_VECTORS, BISHOP_VECTORS))

def check_slider_attack_debug(target, attacker, vectors):
    tx, ty, tz = target
    ax, ay, az = attacker
    
    dx = tx - ax
    dy = ty - ay
    dz = tz - az
    
    print(f"Delta: {dx}, {dy}, {dz}")
    
    n_vecs = vectors.shape[0]
    for i in range(n_vecs):
        vx, vy, vz = vectors[i]
        
        k = -1
        valid = True
        
        if vx != 0:
            if dx % vx != 0: valid = False
            else: k = dx // vx
        elif dx != 0: valid = False
        
        if valid:
            if vy != 0:
                if dy % vy != 0: valid = False
                else:
                    k2 = dy // vy
                    if k == -1: k = k2
                    elif k != k2: valid = False
            elif dy != 0: valid = False
            
        if valid:
            if vz != 0:
                if dz % vz != 0: valid = False
                else:
                    k3 = dz // vz
                    if k == -1: k = k3
                    elif k != k3: valid = False
            elif dz != 0: valid = False
            
        if valid and k > 0:
            print(f"MATCHED vector {vectors[i]} with k={k}")
            return True
            
    print("No match found")
    return False

if __name__ == "__main__":
    attacker = np.array([2, 7, 2])
    target = np.array([3, 7, 0])
    check_slider_attack_debug(target, attacker, QUEEN_VECTORS)
