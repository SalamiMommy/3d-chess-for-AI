# game3d/common/tensor_utils.py
# ------------------------------------------------------------------
# Tensor utilities (updated for new structure)
# ------------------------------------------------------------------
from __future__ import annotations
import torch
from typing import List, Tuple, Optional, Union

from game3d.common.constants import PIECE_SLICE, N_PIECE_TYPES, COLOR_SLICE
from game3d.common.coord_utils import Coord

def hash_board_tensor(tensor: torch.Tensor) -> Union[int, List[int]]:
    """Content-based hash — slow but correct. Supports scalar and batch mode."""
    if tensor.ndim == 4:
        # Single board: [C, D, H, W]
        return hash(tensor.cpu().numpy().tobytes())
    elif tensor.ndim == 5:
        # Batch of boards: [B, C, D, H, W]
        return [hash(tensor[i].cpu().numpy().tobytes()) for i in range(tensor.shape[0])]
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {tensor.ndim}D")

def create_occupancy_mask_tensor(board_tensor: torch.Tensor) -> torch.Tensor:
    """Create boolean occupancy mask from board tensor. Supports scalar and batch mode."""
    if board_tensor.ndim == 4:
        # Single board: [C, D, H, W]
        pieces = board_tensor[PIECE_SLICE].sum(dim=0 if board_tensor.ndim == 4 else 1)
        return (pieces > 0).bool()
    elif board_tensor.ndim == 5:
        # Batch of boards: [B, C, D, H, W]
        pieces = board_tensor[:, PIECE_SLICE].sum(dim=1)
        return pieces > 0
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {board_tensor.ndim}D")

def get_current_player(board_tensor: torch.Tensor) -> Union[int, torch.Tensor]:
    """Get current player (1=white, 0=black). Supports scalar and batch mode."""
    if board_tensor.ndim == 4:
        # Single board
        return int(board_tensor[N_PIECE_TYPES + COLOR_SLICE.start, 0, 0, 0].item() > 0.5)
    elif board_tensor.ndim == 5:
        # Batch of boards
        color_values = board_tensor[:, N_PIECE_TYPES + COLOR_SLICE.start, 0, 0, 0]
        return (color_values > 0.5).int()
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {board_tensor.ndim}D")

def get_piece_color(board_tensor: torch.Tensor, coord: Coord) -> Union[Optional[int], List[Optional[int]]]:
    """Get color of piece at coordinate (1=white, 0=black, None=empty). Supports scalar and batch mode."""
    if torch.is_tensor(coord) and coord.ndim > 1:
        # Batch mode: multiple coordinates
        x_coords, y_coords, z_coords = coord[:, 0], coord[:, 1], coord[:, 2]

        if board_tensor.ndim == 4:
            # Single board, multiple coords
            piece_planes = board_tensor[PIECE_SLICE, z_coords, y_coords, x_coords]  # [piece_types, N]
            piece_exists = piece_planes.sum(dim=0) > 0  # [N]
            color_values = board_tensor[N_PIECE_TYPES, z_coords, y_coords, x_coords]  # [N]

            colors = []
            for i in range(coord.shape[0]):
                if piece_exists[i]:
                    colors.append(1 if color_values[i].item() > 0.5 else 0)
                else:
                    colors.append(None)
            return colors
        elif board_tensor.ndim == 5:
            # Batch of boards, batch of coords
            batch_size = board_tensor.shape[0]
            result = []
            for i in range(batch_size):
                single_board = board_tensor[i]
                single_coord = coord[i] if coord.shape[0] == batch_size else coord
                result.append(get_piece_color(single_board, single_coord))
            return result
    else:
        # Scalar mode: single coordinate
        if torch.is_tensor(coord):
            x, y, z = coord.tolist()
        else:
            x, y, z = coord

        if board_tensor.ndim == 4:
            # Single board, single coord
            piece_planes = board_tensor[PIECE_SLICE, z, y, x]
            if piece_planes.sum() == 0:
                return None
            color_value = board_tensor[N_PIECE_TYPES, z, y, x].item()
            return 1 if color_value > 0.5 else 0
        elif board_tensor.ndim == 5:
            # Batch of boards, single coord
            piece_planes = board_tensor[:, PIECE_SLICE, z, y, x]  # [B, piece_types]
            piece_exists = piece_planes.sum(dim=1) > 0  # [B]
            color_values = board_tensor[:, N_PIECE_TYPES, z, y, x]  # [B]

            colors = []
            for i in range(board_tensor.shape[0]):
                if piece_exists[i]:
                    colors.append(1 if color_values[i].item() > 0.5 else 0)
                else:
                    colors.append(None)
            return colors

    raise ValueError(f"Unsupported tensor dimensions: board {board_tensor.ndim}D, coord with {coord.ndim if torch.is_tensor(coord) else 'scalar'} dims")

def find_pieces_by_type(board_tensor: torch.Tensor, piece_type: Union[int, torch.Tensor], color: Union[int, torch.Tensor]) -> Union[List[Coord], List[List[Coord]]]:
    """Find all pieces of given type and color. Supports scalar and batch mode."""
    if board_tensor.ndim == 4:
        # Single board
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

    elif board_tensor.ndim == 5:
        # Batch of boards
        batch_size = board_tensor.shape[0]

        # Handle batch inputs for piece_type and color
        if torch.is_tensor(piece_type) and piece_type.numel() > 1:
            piece_types = piece_type
        else:
            piece_types = torch.tensor([piece_type] * batch_size)

        if torch.is_tensor(color) and color.numel() > 1:
            colors = color
        else:
            colors = torch.tensor([color] * batch_size)

        results = []
        for i in range(batch_size):
            single_board = board_tensor[i]
            single_piece_type = piece_types[i].item()
            single_color = colors[i].item()
            results.append(find_pieces_by_type(single_board, single_piece_type, single_color))
        return results

    raise ValueError(f"Expected 4D or 5D tensor, got {board_tensor.ndim}D")

def find_all_pieces_of_color(board_tensor: torch.Tensor, color: Union[int, torch.Tensor]) -> Union[List[Tuple[int, Coord]], List[List[Tuple[int, Coord]]]]:
    """Find all pieces of given color, returning (piece_type, coordinate) pairs. Supports scalar and batch mode."""
    if board_tensor.ndim == 4:
        # Single board
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

    elif board_tensor.ndim == 5:
        # Batch of boards
        batch_size = board_tensor.shape[0]

        # Handle batch input for color
        if torch.is_tensor(color) and color.numel() > 1:
            colors = color
        else:
            colors = torch.tensor([color] * batch_size)

        results = []
        for i in range(batch_size):
            single_board = board_tensor[i]
            single_color = colors[i].item()
            results.append(find_all_pieces_of_color(single_board, single_color))
        return results

    raise ValueError(f"Expected 4D or 5D tensor, got {board_tensor.ndim}D")

def get_piece_locations_by_color(board_tensor: torch.Tensor, color: Union[int, torch.Tensor]) -> Union[List[Tuple[int, Coord]], List[List[Tuple[int, Coord]]]]:
    """Get piece locations by color. Supports scalar and batch mode."""
    if board_tensor.ndim == 4:
        # Single board
        color_idx = 0 if color == 1 else 1
        color_plane = board_tensor[COLOR_SLICE.start + color_idx]
        color_positions = color_plane > 0.5

        results = []
        for piece_type in range(N_PIECE_TYPES):
            piece_plane = board_tensor[piece_type]
            target_positions = (piece_plane > 0.5) & color_positions
            positions = torch.nonzero(target_positions, as_tuple=False)
            for z, y, x in positions.tolist():
                results.append((piece_type, (int(x), int(y), int(z))))
        return results

    elif board_tensor.ndim == 5:
        # Batch of boards
        batch_size = board_tensor.shape[0]

        # Handle batch input for color
        if torch.is_tensor(color) and color.numel() > 1:
            colors = color
        else:
            colors = torch.tensor([color] * batch_size)

        results = []
        for i in range(batch_size):
            single_board = board_tensor[i]
            single_color = colors[i].item()
            results.append(get_piece_locations_by_color(single_board, single_color))
        return results

    raise ValueError(f"Expected 4D or 5D tensor, got {board_tensor.ndim}D")
