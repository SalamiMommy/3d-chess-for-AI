"""
Structure Manager
Handles logic for multi-square structures (e.g., Walls).
Centralizes retrieval of structure components and atomic removal operations.
"""

from typing import List, Tuple, Optional, Set
import numpy as np
import logging
from game3d.common.shared_types import PieceType, SIZE, COORD_DTYPE, COLOR_DTYPE

logger = logging.getLogger(__name__)

# Constants
WALL_BLOCK_OFFSETS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=COORD_DTYPE)

class StructureManager:
    """Static manager for multi-square structure logic."""

    @staticmethod
    def is_structure_type(piece_type: int) -> bool:
        """Check if a piece type is part of a multi-square structure."""
        return piece_type == PieceType.WALL

    @staticmethod
    def is_wall_anchor(pos: np.ndarray, source) -> bool:
        """
        Check if pos is the top-left anchor of a 2x2 wall block.
        Args:
            source: Either OccupancyCache or board_type encoded numpy array (SIZE, SIZE, SIZE).
        """
        x, y, z = pos[0], pos[1], pos[2]
        
        # Helper to get type
        def get_type(cx, cy, cz):
            if isinstance(source, np.ndarray):
                return source[cx, cy, cz]
            elif hasattr(source, 'get_type_at'):
                return source.get_type_at(cx, cy, cz)
            else:
                arr = np.array([cx, cy, cz], dtype=COORD_DTYPE)
                t, _ = source.get_fast(arr)
                return t

        # Check left neighbor (x-1)
        if x > 0:
            if get_type(x-1, y, z) == PieceType.WALL:
                return False
                
        # Check up neighbor (y-1)
        if y > 0:
            if get_type(x, y-1, z) == PieceType.WALL:
                return False
                
        return True

    @staticmethod
    def find_anchor_for_square(square: np.ndarray, piece_type: int, source) -> Optional[np.ndarray]:
        """
        Given a square that contains a structure piece, find its anchor position.
        Args:
            source: Either OccupancyCache or board_type numpy array.
        """
        if piece_type != PieceType.WALL:
            return square # Single square structure is its own anchor
            
        wx, wy, wz = square[0], square[1], square[2]
        
        # Check 4 possible anchor positions for 2x2 block
        candidates = [
            (wx, wy, wz),
            (wx - 1, wy, wz),
            (wx, wy - 1, wz),
            (wx - 1, wy - 1, wz)
        ]
        
        # Helper to get type
        def get_type(cx, cy, cz):
            if isinstance(source, np.ndarray):
                return source[cx, cy, cz]
            elif hasattr(source, 'get_type_at'):
                return source.get_type_at(cx, cy, cz)
            else:
                arr = np.array([cx, cy, cz], dtype=COORD_DTYPE)
                t, _ = source.get_fast(arr)
                return t
        
        for cx, cy, cz in candidates:
            if 0 <= cx < SIZE - 1 and 0 <= cy < SIZE - 1 and 0 <= cz < SIZE:
                c_arr = np.array([cx, cy, cz], dtype=COORD_DTYPE)
                c_type = get_type(cx, cy, cz)
                    
                if c_type == piece_type and StructureManager.is_wall_anchor(c_arr, source):
                    # Verify if our hit square (wx, wy, wz) relative offsets match 2x2 block
                    dx = wx - cx
                    dy = wy - cy
                    if 0 <= dx <= 1 and 0 <= dy <= 1:
                        return c_arr
                        
        return None

    @staticmethod
    def get_full_structure_squares(anchor: np.ndarray, piece_type: int) -> np.ndarray:
        """
        Get all squares belonging to a structure given its anchor.
        Returns array of shape (N, 3).
        """
        if piece_type == PieceType.WALL:
            squares = np.empty((4, 3), dtype=COORD_DTYPE)
            for i in range(4):
                squares[i] = anchor + WALL_BLOCK_OFFSETS[i]
            return squares
        
        return anchor.reshape(1, 3)

    @staticmethod
    def get_structure_squares_from_component(
        square: np.ndarray, 
        piece_type: int, 
        occupancy_cache
    ) -> np.ndarray:
        """
        Given ANY square of a structure, return ALL squares of that structure.
        Handles finding the anchor automatically.
        """
        if piece_type != PieceType.WALL:
             return square.reshape(1, 3)
             
        anchor = StructureManager.find_anchor_for_square(square, piece_type, occupancy_cache)
        if anchor is None:
            # Fallback: Just return the single square if structure seems broken
            logger.warning(f"StructureManager: Could not find anchor for Wall at {square}. State might be corrupted.")
            return square.reshape(1, 3)
            
        return StructureManager.get_full_structure_squares(anchor, piece_type)
