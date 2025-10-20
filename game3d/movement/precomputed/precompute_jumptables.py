"""
Precompute jump tables for all jumper pieces (knight, king, echo, panel, orbital, nebula, etc.)
Each entry is a list of destination squares (x, y, z) for each start square.
Output: game3d/movement/movetypes/precomputed/{piece}_jumptable.npy
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import numpy as np

from game3d.movement.movetypes.knightmovement import KNIGHT_OFFSETS
from game3d.movement.movetypes.kingmovement import KING_DIRECTIONS_3D
from game3d.movement.movetypes.knight31movement import VECTORS_31
from game3d.movement.movetypes.knight32movement import VECTORS_32
from game3d.movement.movetypes.echomovement import get_echo_directions
from game3d.movement.movetypes.panelmovement import get_panel_directions
from game3d.movement.movetypes.orbitalmovement import get_orbital_directions
from game3d.movement.movetypes.nebulamovement import get_nebula_offsets

OUTDIR = os.path.join(os.path.dirname(__file__), "precomputed")
os.makedirs(OUTDIR, exist_ok=True)

BOARD_SIZE = 9

def in_bounds(x, y, z):
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and 0 <= z < BOARD_SIZE

def build_jump_table(directions):
    jump_table = []
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            for z in range(BOARD_SIZE):
                moves = []
                for dx, dy, dz in directions:
                    tx, ty, tz = x + int(dx), y + int(dy), z + int(dz)
                    if in_bounds(tx, ty, tz):
                        moves.append((tx, ty, tz))
                jump_table.append(moves)
    return jump_table  # shape: [729][variable num moves]

def save_jumptable(piece_name, directions):
    print(f"Precomputing jump table for {piece_name}...")
    table = build_jump_table(directions)
    np.save(os.path.join(OUTDIR, f"{piece_name}_jumptable.npy"), np.array(table, dtype=object))
    print(f"Saved {piece_name}_jumptable.npy")

def main():
    save_jumptable("knight", KNIGHT_OFFSETS)
    save_jumptable("king", KING_DIRECTIONS_3D)
    save_jumptable("knight31", VECTORS_31)
    save_jumptable("knight32", VECTORS_32)
    save_jumptable("echo", get_echo_directions())
    save_jumptable("panel", get_panel_directions())
    save_jumptable("orbital", get_orbital_directions())
    save_jumptable("nebula", get_nebula_offsets())

if __name__ == "__main__":
    main()
