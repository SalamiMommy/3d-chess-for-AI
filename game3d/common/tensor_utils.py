# game3d/common/tensor_utils.py
# ------------------------------------------------------------------
# Tensor utilities (updated for new structure)
# ------------------------------------------------------------------
from __future__ import annotations
import torch
from typing import List, Tuple, Optional

from game3d.common.constants import PIECE_SLICE, N_PIECE_TYPES, COLOR_SLICE
from game3d.common.coord_utils import Coord

def hash_board_tensor(tensor: torch.Tensor) -> int:
    """Content-based hash — slow but correct."""
    # For performance, you might want to use a rolling hash in Board class instead
    return hash(tensor.cpu().numpy().tobytes())

def create_occupancy_mask_tensor(board_tensor: torch.Tensor) -> torch.Tensor:
    """Create boolean occupancy mask from board tensor."""
    pieces = board_tensor[PIECE_SLICE].sum(dim=0)
    return pieces > 0

def get_current_player(board_tensor: torch.Tensor) -> int:
    """Get current player (1=white, 0=black)."""
    return int(board_tensor[N_PIECE_TYPES + COLOR_SLICE.start, 0, 0, 0].item() > 0.5)

def get_piece_color(board_tensor: torch.Tensor, coord: Coord) -> Optional[int]:
    """Get color of piece at coordinate (1=white, 0=black, None=empty)."""
    x, y, z = coord
    # Check if any piece exists at this coordinate
    piece_planes = board_tensor[PIECE_SLICE, z, y, x]
    if piece_planes.sum() == 0:
        return None

    # Get color from color mask
    color_value = board_tensor[N_PIECE_TYPES, z, y, x].item()
    return 1 if color_value > 0.5 else 0

def find_pieces_by_type(board_tensor: torch.Tensor, piece_type: int, color: int) -> List[Coord]:
    """Find all pieces of given type and color."""
    piece_plane = board_tensor[piece_type]
    color_plane = board_tensor[N_PIECE_TYPES]  # color mask

    # Find positions where piece exists
    piece_positions = piece_plane > 0.5

    # Filter by color
    if color == 1:  # white
        color_positions = color_plane > 0.5
    else:  # black
        color_positions = color_plane <= 0.5

    # Combine conditions
    target_positions = piece_positions & color_positions
    positions = torch.nonzero(target_positions, as_tuple=False)
    return [(int(x), int(y), int(z)) for z, y, x in positions.tolist()]

def find_all_pieces_of_color(board_tensor: torch.Tensor, color: int) -> List[Tuple[int, Coord]]:
    """Find all pieces of given color, returning (piece_type, coordinate) pairs."""
    color_plane = board_tensor[N_PIECE_TYPES]  # color mask

    # Filter by color
    if color == 1:  # white
        color_positions = color_plane > 0.5
    else:  # black
        color_positions = color_plane <= 0.5

    results = []
    for piece_type in range(N_PIECE_TYPES):
        piece_plane = board_tensor[piece_type]
        # Find positions where both piece exists and color matches
        target_positions = (piece_plane > 0.5) & color_positions
        positions = torch.nonzero(target_positions, as_tuple=False)
        for z, y, x in positions.tolist():
            results.append((piece_type, (int(x), int(y), int(z))))

    return results

def get_piece_locations_by_color(board_tensor: torch.Tensor, color: int) -> List[Tuple[int, Coord]]:
    color_plane = board_tensor[COLOR_SLICE.start + (0 if color == 1 else 1)]
    color_positions = color_plane > 0.5 if color == 1 else color_plane <= 0.5
    results = []
    for piece_type in range(N_PIECE_TYPES):
        piece_plane = board_tensor[piece_type]
        target_positions = (piece_plane > 0.5) & color_positions
        positions = torch.nonzero(target_positions, as_tuple=False)
        for z, y, x in positions.tolist():
            results.append((piece_type, (int(x), int(y), int(z))))
    return results
