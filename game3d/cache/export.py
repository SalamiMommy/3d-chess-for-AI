# game3d/cache/export.py - OPTIMIZED VERSION
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from game3d.common.enums import Color, PieceType
from game3d.common.constants import N_TOTAL_PLANES

# Use uint8 for everything possible to reduce memory and speed up transfers
def export_state_for_ai(board, piece_cache, move_cache, zobrist_hash,
                                current_player: Color, move_number: int = 0) -> Dict[str, Any]:
    """
    Ultra-optimized export using direct cache access and minimal conversions.
    """
    # DIRECT ACCESS to occupancy cache arrays (no reconstruction)
    occupancy = piece_cache._occ  # shape (9,9,9) uint8
    piece_types = piece_cache._ptype  # shape (9,9,9) uint8

    # Convert to piece_colors using vectorized operations
    piece_colors = occupancy.astype(np.int8)  # Already has 0,1,2 values

    # Fast legal moves extraction
    legal_moves_white = []
    legal_moves_black = []

    if move_cache is not None:
        try:
            # Use existing cached moves without regeneration
            white_moves = move_cache._legal_by_color.get(Color.WHITE, [])
            black_moves = move_cache._legal_by_color.get(Color.BLACK, [])

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
        except Exception:
            legal_moves_white = []
            legal_moves_black = []

    # Fast piece counts using vectorized operations
    white_mask = occupancy == 1
    black_mask = occupancy == 2

    piece_counts = {
        'white': {ptype: np.sum(piece_types[white_mask] == ptype.value) for ptype in PieceType},
        'black': {ptype: np.sum(piece_types[black_mask] == ptype.value) for ptype in PieceType}
    }

    return {
        'occupancy': occupancy,  # No copy if possible
        'piece_types': piece_types,  # No copy if possible
        'piece_colors': piece_colors,
        'legal_moves_white': tuple(legal_moves_white),
        'legal_moves_black': tuple(legal_moves_black),
        'piece_counts': piece_counts,
        'zobrist_hash': zobrist_hash,
        'current_player': current_player,
        'move_number': move_number,
    }

def export_tensor_for_ai(board, piece_cache, move_cache, zobrist_hash,
                                 current_player: Color, move_number: int = 0,
                                 device: str = "cpu") -> torch.Tensor:
    """
    Ultra-fast tensor export using vectorized operations and direct memory access.
    """
    # Direct access to avoid function call overhead
    occupancy = piece_cache._occ
    piece_types = piece_cache._ptype

    # Pre-allocate tensor with optimal memory layout
    tensor = torch.zeros((N_TOTAL_PLANES, 9, 9, 9), dtype=torch.bool, device=device)  # Use bool for one-hot

    player_offset = 0 if current_player == Color.WHITE else 40
    opponent_offset = 40 if current_player == Color.WHITE else 0

    # VECTORIZED assignment using advanced indexing
    white_pieces = occupancy == 1
    black_pieces = occupancy == 2

    # Process white pieces
    white_coords = np.argwhere(white_pieces)
    if len(white_coords) > 0:
        z, y, x = white_coords.T
        types = piece_types[white_pieces]
        if current_player == Color.WHITE:
            indices = types + player_offset
        else:
            indices = types + opponent_offset
        tensor[indices, z, y, x] = True

    # Process black pieces
    black_coords = np.argwhere(black_pieces)
    if len(black_coords) > 0:
        z, y, x = black_coords.T
        types = piece_types[black_pieces]
        if current_player == Color.BLACK:
            indices = types + player_offset
        else:
            indices = types + opponent_offset
        tensor[indices, z, y, x] = True

    # Current player plane
    tensor[80, :, :, :] = (current_player == Color.WHITE)

    return tensor

def export_tensor_direct(occupancy_cache, current_player: Color, device: str = "cpu") -> torch.Tensor:
    """
    Direct tensor export from occupancy cache - fastest possible version.
    Assumes occupancy_cache has _occ and _ptype arrays.
    """
    occupancy = occupancy_cache._occ
    piece_types = occupancy_cache._ptype

    tensor = torch.zeros((N_TOTAL_PLANES, 9, 9, 9), dtype=torch.bool, device=device)

    player_offset = 0 if current_player == Color.WHITE else 40
    opponent_offset = 40 if current_player == Color.WHITE else 0

    # Single pass vectorized processing
    white_mask = occupancy == 1
    black_mask = occupancy == 2

    # White pieces
    if np.any(white_mask):
        z, y, x = np.where(white_mask)
        types = piece_types[white_mask]
        offsets = player_offset if current_player == Color.WHITE else opponent_offset
        tensor[types + offsets, z, y, x] = True

    # Black pieces
    if np.any(black_mask):
        z, y, x = np.where(black_mask)
        types = piece_types[black_mask]
        offsets = player_offset if current_player == Color.BLACK else opponent_offset
        tensor[types + offsets, z, y, x] = True

    # Current player plane
    tensor[80, :, :, :] = (current_player == Color.WHITE)

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

def batch_export_tensors(occupancy_caches: List, current_players: List[Color],
                        device: str = "cpu") -> torch.Tensor:
    """
    Batch export multiple positions for efficient training.
    Returns tensor of shape (batch_size, N_TOTAL_PLANES, 9, 9, 9)
    """
    batch_size = len(occupancy_caches)
    batch_tensor = torch.zeros((batch_size, N_TOTAL_PLANES, 9, 9, 9),
                              dtype=torch.bool, device=device)

    for i, (occ_cache, player) in enumerate(zip(occupancy_caches, current_players)):
        occupancy = occ_cache._occ
        piece_types = occ_cache._ptype

        player_offset = 0 if player == Color.WHITE else 40
        opponent_offset = 40 if player == Color.WHITE else 0

        # White pieces
        white_mask = occupancy == 1
        if np.any(white_mask):
            z, y, x = np.where(white_mask)
            types = piece_types[white_mask]
            offsets = player_offset if player == Color.WHITE else opponent_offset
            batch_tensor[i, types + offsets, z, y, x] = True

        # Black pieces
        black_mask = occupancy == 2
        if np.any(black_mask):
            z, y, x = np.where(black_mask)
            types = piece_types[black_mask]
            offsets = player_offset if player == Color.BLACK else opponent_offset
            batch_tensor[i, types + offsets, z, y, x] = True

        # Current player plane
        batch_tensor[i, 80, :, :, :] = (player == Color.WHITE)

    return batch_tensor
