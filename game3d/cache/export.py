# game3d/cache/export.py

"""AI export utilities for cache manager state."""

import threading
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from functools import lru_cache
from game3d.pieces.enums import Color, PieceType
from game3d.common.common import N_TOTAL_PLANES

def export_state_for_ai(board, piece_cache, move_cache, zobrist_hash, current_player: Color, move_number: int = 0) -> Dict[str, Any]:
    """
    Export core cache state safely for AI consumption.
    Only includes essential data: pieces, occupancy, legal moves, and metadata.
    All data is copied to prevent mutation issues.

    Args:
        current_player: Current player color
        move_number: Current move number

    Returns:
        Dictionary with core cache data as numpy arrays/immutable structures
    """
    with threading.RLock():  # Thread-safe snapshot
        # 1. PIECE AND OCCUPANCY DATA (safe copies)
        occupancy_arr, piece_type_arr = piece_cache.export_arrays()

        # Build color array
        piece_colors = np.zeros((9, 9, 9), dtype=np.int8)
        for (x, y, z), piece in board.list_occupied():
            piece_colors[x, y, z] = 1 if piece.color == Color.WHITE else 2

        # 2. LEGAL MOVES (safe copies)
        legal_moves_white = []
        legal_moves_black = []

        if move_cache is not None:
            try:
                # Get legal moves for both colors
                white_moves = move_cache.legal_moves(Color.WHITE, parallel=False)
                black_moves = move_cache.legal_moves(Color.BLACK, parallel=False)

                # Store as tuples for immutability
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
                # Fallback to empty if move cache fails
                print(f"[AI Export] Move cache error: {e}")
                legal_moves_white = []
                legal_moves_black = []

        # 3. ZOBRIST HASH
        zobrist = zobrist_hash

        # 4. PIECE COUNTS BY TYPE (useful for game phase estimation)
        piece_counts = {
            'white': {ptype: 0 for ptype in PieceType},
            'black': {ptype: 0 for ptype in PieceType}
        }
        for (coord, piece) in board.list_occupied():
            color_key = 'white' if piece.color == Color.WHITE else 'black'
            piece_counts[color_key][piece.ptype] += 1

        return {
            # Core board state (safe copies)
            'occupancy': occupancy_arr.copy(),      # (9,9,9) uint8 - 0=empty, 1=white, 2=black
            'piece_types': piece_type_arr.copy(),   # (9,9,9) uint8 - PieceType.value
            'piece_colors': piece_colors.copy(),    # (9,9,9) int8 - 0=empty, 1=white, 2=black

            # Legal moves (immutable tuples)
            'legal_moves_white': tuple(legal_moves_white),
            'legal_moves_black': tuple(legal_moves_black),

            # Piece counts
            'piece_counts': piece_counts,

            # Metadata
            'zobrist_hash': zobrist,
            'current_player': current_player,
            'move_number': move_number,
        }

def export_tensor_for_ai(board, piece_cache, move_cache, zobrist_hash, current_player: Color, move_number: int = 0,
                         device: str = "cpu") -> torch.Tensor:
    state = export_state_for_ai(board, piece_cache, move_cache, zobrist_hash, current_player, move_number)

    # Initialize tensor
    tensor = torch.zeros((N_TOTAL_PLANES, 9, 9, 9), dtype=torch.float32, device=device)

    player_offset = 0 if current_player == Color.WHITE else 40
    opponent_offset = 40 if current_player == Color.WHITE else 0

    for z in range(9):
        for y in range(9):
            for x in range(9):
                col = int(state['piece_colors'][x, y, z])  # 0=empty, 1=white, 2=black
                if col > 0:
                    ptype_val = int(state['piece_types'][x, y, z])
                    if 0 <= ptype_val < 40:  # Safety check
                        # Map color to offset relative to current player
                        if (col == 1 and current_player == Color.WHITE) or (col == 2 and current_player == Color.BLACK):
                            # Current player's pieces
                            tensor[ptype_val + player_offset, z, y, x] = 1.0
                        else:
                            # Opponent's pieces
                            tensor[ptype_val + opponent_offset, z, y, x] = 1.0

    # === Side-to-move plane (80) ===
    tensor[80, :, :, :] = 1.0 if current_player == Color.WHITE else 0.0

    return tensor

def get_legal_move_indices(move_cache, color: Color) -> Tuple[List[int], List[int]]:
    """
    Get legal move indices for factorized policy head.
    Returns separate from_indices and to_indices for all legal moves.

    Args:
        color: Player color

    Returns:
        Tuple of (from_indices, to_indices) where each is List[int] in range [0, 728]
        Indices map to flattened 9x9x9 grid: index = x*81 + y*9 + z
    """
    if move_cache is None:
        return ([], [])

    try:
        legal_moves = move_cache.legal_moves(color, parallel=False)

        from_indices = []
        to_indices = []

        for move in legal_moves:
            # Convert 3D coords to flat index (0-728)
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
    """
    Convert legal moves to policy target tensor for training.

    Args:
        color: Player color
        move_probabilities: Optional dict mapping Move -> probability
                           If None, uniform distribution over legal moves

    Returns:
        Tensor of shape (531_441,) with probabilities for all possible moves
        Index = from_flat_idx * 729 + to_flat_idx where flat_idx = x*81 + y*9 + z
    """
    policy_target = torch.zeros(531_441, dtype=torch.float32)

    if move_cache is None:
        return policy_target

    try:
        legal_moves = move_cache.legal_moves(color, parallel=False)

        if not legal_moves:
            return policy_target

        # Uniform distribution if no probabilities given
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
    """
    Validate that exported data matches internal state.
    Useful for debugging and ensuring AI sees correct data.

    Returns:
        Dict mapping check_name -> passed (bool)
    """
    results = {}

    try:
        state = export_state_for_ai(board, piece_cache, move_cache, zobrist_hash, Color.WHITE, 0)

        # Check 1: Occupancy matches piece cache
        occ_count = np.sum(state['occupancy'] > 0)
        piece_count = len(list(board.list_occupied()))
        results['occupancy_count_match'] = (occ_count == piece_count)

        # Check 2: Piece colors match
        color_mismatches = 0
        for (x, y, z), piece in board.list_occupied():
            expected_color = 1 if piece.color == Color.WHITE else 2
            actual_color = state['piece_colors'][x, y, z]
            if expected_color != actual_color:
                color_mismatches += 1
        results['piece_colors_match'] = (color_mismatches == 0)

        # Check 3: Piece types match
        type_mismatches = 0
        for (x, y, z), piece in board.list_occupied():
            expected_type = piece.ptype.value
            actual_type = state['piece_types'][x, y, z]
            if expected_type != actual_type:
                type_mismatches += 1
        results['piece_types_match'] = (type_mismatches == 0)

        # Check 4: Zobrist hash is non-zero
        results['zobrist_nonzero'] = (state['zobrist_hash'] != 0)

        # Check 5: Legal moves exist (if game not over)
        has_moves = len(state['legal_moves_white']) > 0 or len(state['legal_moves_black']) > 0
        results['has_legal_moves'] = has_moves

        # Check 6: Tensor export works
        try:
            tensor = export_tensor_for_ai(board, piece_cache, move_cache, zobrist_hash, Color.WHITE, 0, device="cpu")
            results['tensor_export_works'] = (tensor.shape == (81, 9, 9, 9))
        except Exception:
            results['tensor_export_works'] = False

    except Exception as e:
        print(f"[Validation] Error: {e}")
        results['validation_error'] = False

    return results
