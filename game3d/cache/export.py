# game3d/cache/export.py

"""AI export utilities for cache manager state."""
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from game3d.common.enums import Color, PieceType
from game3d.common.constants import N_TOTAL_PLANES

# Updated function in export.py
def export_state_for_ai(board, piece_cache, move_cache, zobrist_hash, current_player: Color, move_number: int = 0) -> Dict[str, Any]:
    """
    Export core cache state safely for AI consumption.
    Only includes essential data: pieces, occupancy, legal moves, and metadata.
    All data is copied to prevent mutation issues.
    """

    # Reconstruct occupancy_arr and piece_type_arr using public methods
    occupancy_arr = np.zeros((9, 9, 9), dtype=np.uint8)
    piece_type_arr = np.zeros((9, 9, 9), dtype=np.uint8)
    for color in [Color.WHITE, Color.BLACK]:
        code = 1 if color == Color.WHITE else 2
        for coord, piece in piece_cache.iter_color(color):
            x, y, z = coord
            occupancy_arr[z, y, x] = code
            piece_type_arr[z, y, x] = piece.ptype.value

    piece_colors = np.zeros((9, 9, 9), dtype=np.int8)
    for (x, y, z), piece in board.list_occupied():
        piece_colors[x, y, z] = Color.WHITE.value if piece.color == Color.WHITE else Color.BLACK.value  # Use enum values

    legal_moves_white = []
    legal_moves_black = []

    if move_cache is not None:
        try:
            white_moves = move_cache.legal_moves(Color.WHITE, parallel=False)
            black_moves = move_cache.legal_moves(Color.BLACK, parallel=False)

            legal_moves_white = [
                (m.from_coord, m.to_coord, getattr(m, 'is_capture', False),
                    getattr(m, 'is_promotion', False))
                for m in white_moves
            ]
            legal_moves_black = [
                (m.from_coord, m.to_coord, getattr(m, 'is_capture', False),
                    getattr(m, 'is_promotion', False))
                for m in black_moves
            ]
        except Exception as e:
            print(f"[AI Export] Move cache error: {e}")
            legal_moves_white = []
            legal_moves_black = []

    piece_counts = {
        'white': {ptype: 0 for ptype in PieceType},
        'black': {ptype: 0 for ptype in PieceType}
    }
    for (coord, piece) in board.list_occupied():
        color_key = 'white' if piece.color == Color.WHITE else 'black'
        piece_counts[color_key][piece.ptype] += 1

    return {
        'occupancy': occupancy_arr.copy(),
        'piece_types': piece_type_arr.copy(),
        'piece_colors': piece_colors.copy(),
        'legal_moves_white': tuple(legal_moves_white),
        'legal_moves_black': tuple(legal_moves_black),
        'piece_counts': piece_counts,
        'zobrist_hash': zobrist_hash,
        'current_player': current_player,
        'move_number': move_number,
    }

def export_tensor_for_ai(board, piece_cache, move_cache, zobrist_hash, current_player: Color, move_number: int = 0,
                         device: str = "cpu") -> torch.Tensor:
    state = export_state_for_ai(board, piece_cache, move_cache, zobrist_hash, current_player, move_number)

    tensor = torch.zeros((N_TOTAL_PLANES, 9, 9, 9), dtype=torch.float32, device=device)

    player_offset = 0 if current_player == Color.WHITE else 40
    opponent_offset = 40 if current_player == Color.WHITE else 0

    # Optimized: Vectorized assignment
    mask = state['piece_colors'] > 0
    coords = np.argwhere(mask)
    for coord in coords:
        x, y, z = coord
        col = int(state['piece_colors'][x, y, z])
        ptype_val = int(state['piece_types'][x, y, z])
        if 0 <= ptype_val < 40:
            if (col == Color.WHITE.value and current_player == Color.WHITE) or (col == Color.BLACK.value and current_player == Color.BLACK):
                tensor[ptype_val + player_offset, z, y, x] = 1.0
            else:
                tensor[ptype_val + opponent_offset, z, y, x] = 1.0

    tensor[80, :, :, :] = 1.0 if current_player == Color.WHITE else 0.0

    return tensor

def get_legal_move_indices(move_cache, color: Color) -> Tuple[List[int], List[int]]:
    if move_cache is None:
        return ([], [])

    try:
        legal_moves = move_cache.legal_moves(color, parallel=False)

        from_indices = []
        to_indices = []

        for move in legal_moves:
            fx, fy, fz = move.from_coord
            tx, ty, tz = move.to_coord

            from_idx = fx * 81 + fy * 9 + fz
            to_idx = tx * 81 + ty * 9 + tz

            from_indices.append(from_idx)
            to_indices.append(to_idx)

        return (from_indices, to_indices)

    except Exception as e:
        print(f"[AI Export] Error getting legal move indices: {e}")
        return ([], [])

def get_legal_moves_as_policy_target(move_cache, color: Color,
                                     move_probabilities: Optional[Dict['Move', float]] = None) -> torch.Tensor:
    policy_target = torch.zeros(531_441, dtype=torch.float32)

    if move_cache is None:
        return policy_target

    try:
        legal_moves = move_cache.legal_moves(color, parallel=False)

        if not legal_moves:
            return policy_target

        if move_probabilities is None:
            prob = 1.0 / len(legal_moves)
            move_probabilities = {move: prob for move in legal_moves}

        for move, prob in move_probabilities.items():
            fx, fy, fz = move.from_coord
            tx, ty, tz = move.to_coord

            from_idx = fx * 81 + fy * 9 + fz
            to_idx = tx * 81 + ty * 9 + tz

            full_idx = from_idx * 729 + to_idx
            policy_target[full_idx] = prob

        return policy_target

    except Exception as e:
        print(f"[AI Export] Error creating policy target: {e}")
        return policy_target

def validate_export_integrity(board, piece_cache, move_cache, zobrist_hash) -> Dict[str, bool]:
    results = {}

    try:
        state = export_state_for_ai(board, piece_cache, move_cache, zobrist_hash, Color.WHITE, 0)

        occ_count = np.sum(state['occupancy'] > 0)
        piece_count = len(list(board.list_occupied()))
        results['occupancy_count_match'] = (occ_count == piece_count)

        color_mismatches = 0
        for (x, y, z), piece in board.list_occupied():
            expected_color = Color.WHITE.value if piece.color == Color.WHITE else Color.BLACK.value
            actual_color = state['piece_colors'][x, y, z]
            if expected_color != actual_color:
                color_mismatches += 1
        results['piece_colors_match'] = (color_mismatches == 0)

        type_mismatches = 0
        for (x, y, z), piece in board.list_occupied():
            expected_type = piece.ptype.value
            actual_type = state['piece_types'][x, y, z]
            if expected_type != actual_type:
                type_mismatches += 1
        results['piece_types_match'] = (type_mismatches == 0)

        results['zobrist_nonzero'] = (state['zobrist_hash'] != 0)

        has_moves = len(state['legal_moves_white']) > 0 or len(state['legal_moves_black']) > 0
        results['has_legal_moves'] = has_moves

        try:
            tensor = export_tensor_for_ai(board, piece_cache, move_cache, zobrist_hash, Color.WHITE, 0, device="cpu")
            results['tensor_export_works'] = (tensor.shape == (N_TOTAL_PLANES, 9, 9, 9))
        except Exception:
            results['tensor_export_works'] = False

        # Added: Check legal moves count
        results['legal_moves_count_positive'] = len(state['legal_moves_white']) + len(state['legal_moves_black']) > 0

    except Exception as e:
        print(f"[Validation] Error: {e}")
        results['validation_error'] = False

    return results
