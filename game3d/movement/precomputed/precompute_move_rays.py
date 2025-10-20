"""
Precompute move rays for all slider pieces.
Each ray is a list of squares (in order) from a given start in a given direction.
Output: game3d/movement/movetypes/precomputed/{piece}_rays.npy
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import numpy as np

from game3d.movement.movetypes.slidermovement import (
    ROOK_DIRS, BISHOP_DIRS, QUEEN_DIRS
)
from game3d.movement.movetypes.trigonalbishopmovement import TRIGONAL_BISHOP_DIRECTIONS
from game3d.movement.movetypes.xyqueenmovement import XY_QUEEN_DIRECTIONS
from game3d.movement.movetypes.xzqueenmovement import XZ_QUEEN_DIRECTIONS
from game3d.movement.movetypes.yzqueenmovement import YZ_QUEEN_DIRECTIONS
from game3d.movement.movetypes.vectorslidermovement import VECTOR_SLIDER_DIRECTIONS
from game3d.movement.movetypes.spiralmovement import SPIRAL_DIRECTIONS
from game3d.movement.movetypes.xzzigzagmovement import XZ_ZIGZAG_DIRECTIONS
from game3d.movement.movetypes.yzzigzagmovement import YZ_ZIGZAG_DIRECTIONS

# Output directory
OUTDIR = os.path.join(os.path.dirname(__file__), "precomputed")
os.makedirs(OUTDIR, exist_ok=True)

BOARD_SIZE = 9

def in_bounds(x, y, z):
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and 0 <= z < BOARD_SIZE

def build_rays_for_piece(directions, max_steps):
    rays = []
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            for z in range(BOARD_SIZE):
                per_dir = []
                for d in directions:
                    dx, dy, dz = int(d[0]), int(d[1]), int(d[2])
                    ray = []
                    for step in range(1, max_steps+1):
                        tx, ty, tz = x + dx*step, y + dy*step, z + dz*step
                        if not in_bounds(tx, ty, tz):
                            break
                        ray.append((tx, ty, tz))
                    per_dir.append(ray)
                rays.append(per_dir)
    return rays  # shape: [729][num_dirs][variable ray length]

def save_rays(piece_name, directions, max_steps=8):
    print(f"Precomputing rays for {piece_name}...")
    rays = build_rays_for_piece(directions, max_steps)
    np.save(os.path.join(OUTDIR, f"{piece_name}_rays.npy"), np.array(rays, dtype=object))
    print(f"Saved {piece_name}_rays.npy")

def main():
    save_rays("rook", ROOK_DIRS, max_steps=8)
    save_rays("bishop", BISHOP_DIRS, max_steps=8)
    save_rays("queen", QUEEN_DIRS, max_steps=8)
    save_rays("trigonal_bishop", TRIGONAL_BISHOP_DIRECTIONS, max_steps=8)
    save_rays("xy_queen", XY_QUEEN_DIRECTIONS, max_steps=8)
    save_rays("xz_queen", XZ_QUEEN_DIRECTIONS, max_steps=8)
    save_rays("yz_queen", YZ_QUEEN_DIRECTIONS, max_steps=8)
    save_rays("vector_slider", VECTOR_SLIDER_DIRECTIONS, max_steps=8)
    save_rays("spiral", SPIRAL_DIRECTIONS, max_steps=16)
    save_rays("xz_zigzag", XZ_ZIGZAG_DIRECTIONS, max_steps=16)
    save_rays("yz_zigzag", YZ_ZIGZAG_DIRECTIONS, max_steps=16)

if __name__ == "__main__":
    main()
