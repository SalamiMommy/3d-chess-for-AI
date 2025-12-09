"""
Debug infiltrator teleport move generation in generator.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from game3d.game.gamestate import GameState
from game3d.core.buffer import state_to_buffer
from game3d.pieces.pieces.infiltrator import _get_pawn_targets_kernel, _generate_infiltrator_teleport_moves
from game3d.common.shared_types import PieceType, COORD_DTYPE

state = GameState.from_startpos()
buffer = state_to_buffer(state)

print("=== Debug Infiltrator in Generator ===\n")

# Replicate what generator does
active_color = buffer.meta[0]  # Should be 1 (White)
print(f"Active color: {active_color}")

# Find infiltrators (type 38)
infiltrator_positions = []
for i in range(buffer.occupied_count):
    if buffer.occupied_types[i] == 38 and buffer.occupied_colors[i] == active_color:
        infiltrator_positions.append(buffer.occupied_coords[i])

print(f"Infiltrators found: {len(infiltrator_positions)}")
if infiltrator_positions:
    pos = np.array(infiltrator_positions, dtype=COORD_DTYPE)
    print(f"Positions: {pos[:3]}")
    
    # Check aura
    from game3d.core.generator_functional import _compute_auras
    is_buffed, is_debuffed, is_frozen = _compute_auras(
        buffer.occupied_types, buffer.occupied_coords, buffer.occupied_colors,
        buffer.occupied_count, active_color
    )
    
    # Check buff status
    cnt_inf = len(infiltrator_positions)
    is_buffed_mask = np.empty(cnt_inf, dtype=np.bool_)
    for k in range(cnt_inf):
        px, py, pz = pos[k]
        is_buffed_mask[k] = is_buffed[px, py, pz]
    
    print(f"is_buffed_mask: {is_buffed_mask}")
    
    unbuffed_mask = ~is_buffed_mask
    print(f"unbuffed_mask: {unbuffed_mask}")
    print(f"np.any(unbuffed_mask): {np.any(unbuffed_mask)}")
    
    if np.any(unbuffed_mask):
        ub_pos = pos[unbuffed_mask]
        print(f"Unbuffed positions: {ub_pos}")
        
        opp_color = 2 if active_color == 1 else 1
        dz_front = -1 if opp_color == 2 else 1
        
        print(f"opp_color: {opp_color}, dz_front: {dz_front}")
        
        targets = _get_pawn_targets_kernel(buffer.board_color, buffer.board_type, opp_color, dz_front)
        print(f"Targets found: {len(targets)}")
        
        if targets.shape[0] > 0:
            m_ub = _generate_infiltrator_teleport_moves(ub_pos.astype(COORD_DTYPE), targets)
            print(f"Moves generated: {len(m_ub)}")
