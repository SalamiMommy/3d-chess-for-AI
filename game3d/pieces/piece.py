"""Optimized piece module with numpy-native operations."""

import numpy as np
from game3d.common.shared_types import COLOR_DTYPE, PIECE_TYPE_DTYPE, PieceDtype
from game3d.common.shared_types import Color, PieceType

def get_piece_by_index_optimized(index: int, pieces_array: np.ndarray) -> np.ndarray:
    """Get piece at specific index from pieces array."""
    if index < 0 or index >= len(pieces_array):
        return np.array([], dtype=PieceDtype)
    return pieces_array[index:index+1]

# Constants using proper enums
WHITE = Color.WHITE
BLACK = Color.BLACK
PAWN = PieceType.PAWN
KNIGHT = PieceType.KNIGHT
BISHOP = PieceType.BISHOP
ROOK = PieceType.ROOK
QUEEN = PieceType.QUEEN
KING = PieceType.KING
EMPTY = Color.EMPTY

def create_piece_from_array(piece_data: np.ndarray) -> np.ndarray:
    """Create validated piece from numpy array."""
    if piece_data.shape != () or piece_data.dtype != PieceDtype:
        raise ValueError("Invalid piece data")
    return piece_data.copy()

def extract_piece_properties(pieces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract color and type arrays from piece array."""
    return pieces['color'].copy(), pieces['piece_type'].copy()

def compare_pieces(pieces1: np.ndarray, pieces2: np.ndarray) -> np.ndarray:
    """Compare piece arrays."""
    if pieces1.shape != pieces2.shape:
        raise ValueError("pieces1 and pieces2 must have same shape")
    return (pieces1['color'] == pieces2['color']) & (pieces1['piece_type'] == pieces2['piece_type'])

def count_piece_types(pieces: np.ndarray) -> np.ndarray:
    """Count occurrences of each piece type."""
    return np.bincount(pieces['piece_type'], minlength=256)

def get_piece_stats(pieces: np.ndarray) -> dict:
    """Get comprehensive statistics."""
    colors = pieces['color']
    return {
        'total': len(pieces),
        'white_pieces': int(np.sum(colors == WHITE)),
        'black_pieces': int(np.sum(colors == BLACK)),
        'piece_counts': count_piece_types(pieces)
    }

# Export
__all__ = [
    'PieceDtype', 'create_piece_from_array', 'extract_piece_properties',
    'compare_pieces', 'count_piece_types', 'get_piece_stats', 'get_piece_by_index_optimized',
    'WHITE', 'BLACK', 'PAWN', 'KNIGHT', 'BISHOP', 'ROOK', 'QUEEN', 'KING', 'EMPTY'
]
