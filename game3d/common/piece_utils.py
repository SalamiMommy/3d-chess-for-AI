# game3d/common/piece_utils.py
# ------------------------------------------------------------------
# Piece-related utilities
# ------------------------------------------------------------------
from __future__ import annotations
from typing import List, Tuple, Optional, Iterable, TYPE_CHECKING
import torch

from game3d.common.constants import PIECE_SLICE
from game3d.common.enums import Color, PieceType
from game3d.common.coord_utils import Coord
from game3d.pieces.piece import Piece
from numba import njit

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
    from game3d.board.board import Board
    from game3d.cache.manager import OptimizedCacheManager

@njit(cache=True)
def color_to_code(color: "Color") -> int:
    """Return the occupancy-array code (1 or 2) for the given Color enum."""
    return 1 if color.value == 1 else 2

def find_king(state: "GameState", color: Color) -> Optional[Coord]:
    """Vectorised, lock-free search for the king of *color*."""
    for coord, piece in state.cache_manager.get_pieces_of_color(color):
        if piece.ptype == PieceType.KING:
            return coord
    return None

def infer_piece_from_cache(
    cache_manager: "OptimizedCacheManager",
    coord: Coord,
    fallback_type: PieceType = PieceType.PAWN
) -> Piece:
    """
    Infer piece from cache, fallback to given type.
    FIXED: Handles Piece objects correctly.
    """
    piece = cache_manager.get_piece(coord)
    if piece:
        return piece
    # Fallback - need to know the color
    # This is a limitation of the current API
    return Piece(Color.WHITE, fallback_type)

def get_player_pieces(state: GameState, color: Color) -> List[Tuple[Coord, Piece]]:
    """Get all pieces of a given color using standardized cache access."""
    result = []
    for coord, piece in state.cache_manager.get_pieces_of_color(color):
        result.append((coord, piece))
    return result

def iterate_occupied(board: "Board", color: Optional[Color] = None, cache_manager: Optional["OptimizedCacheManager"] = None):
    """
    Iterate through occupied squares with standardized cache access.

    Args:
        board: The board instance
        color: Optional color filter
        cache_manager: Optional cache manager (uses board.cache_manager if not provided)
    """
    if cache_manager is None:
        cache_manager = board.cache_manager

    # Fast path - cache is available
    if cache_manager:
        if color is None:
            # Iterate all pieces of both colors
            for c in [Color.WHITE, Color.BLACK]:
                for coord, piece in cache_manager.get_pieces_of_color(c):
                    yield coord, piece
        else:
            # Iterate pieces of specific color
            for coord, piece in cache_manager.get_pieces_of_color(color):
                yield coord, piece
    else:
        # Slow path - tensor scan (fallback)
        occ = board._tensor[PIECE_SLICE].sum(dim=0) > 0
        indices = torch.nonzero(occ, as_tuple=False)
        for z, y, x in indices.tolist():
            piece = board.piece_at((x, y, z))
            if piece and (color is None or piece.color == color):
                yield (x, y, z), piece

def get_pieces_by_type(
    board: "Board",
    ptype: PieceType,
    color: Optional[Color] = None,
    cache_manager: Optional["OptimizedCacheManager"] = None
) -> List[Tuple[Coord, Piece]]:
    """
    Return every (coord, piece) on *board* whose type == *ptype*
    (and optionally colour == *color*).

    Uses the provided cache_manager or board.cache_manager if available,
    otherwise falls back to a direct tensor scan.

    Args:
        board: The board instance
        ptype: The piece type to search for
        color: Optional color filter
        cache_manager: Optional cache manager (uses board.cache_manager if not provided)
    """
    if cache_manager is None:
        cache_manager = getattr(board, 'cache_manager', None)

    # Fast path – cache is available
    if cache_manager is not None:
        result = []
        # If color is specified, only search that color, otherwise search both
        colors_to_search = [color] if color is not None else [Color.WHITE, Color.BLACK]

        for search_color in colors_to_search:
            for coord, piece in cache_manager.get_pieces_of_color(search_color):
                if piece.ptype == ptype:
                    result.append((coord, piece))
        return result

    # Slow path – cache not yet attached (e.g., during initial aura rebuild)
    # Note: You'll need to define N_PIECE_TYPES or adjust this fallback
    tensor = board._tensor
    piece_planes = tensor[PIECE_SLICE]
    col_plane = tensor[80]  # Assuming color plane is at index 80 (adjust as needed)

    wanted_type = ptype.value
    wanted_color = 1 if color is Color.WHITE else 0

    mask = (piece_planes[wanted_type] > 0.5)
    if color is not None:
        mask &= (col_plane > 0.5) if color is Color.WHITE else (col_plane <= 0.5)

    indices = torch.nonzero(mask, as_tuple=False)
    return [
        ((int(x), int(y), int(z)),
         Piece(color if color is not None else Color.WHITE, ptype))
        for z, y, x in indices.tolist()
    ]

def get_pieces_by_type_from_cache(
    cache_manager: "OptimizedCacheManager",
    ptype: PieceType,
    color: Color
) -> List[Tuple[Coord, Piece]]:
    """
    Direct cache-based lookup for pieces by type and color.
    More efficient when you already have the cache manager.
    """
    result = []
    for coord, piece in cache_manager.get_pieces_of_color(color):
        if piece.ptype == ptype:
            result.append((coord, piece))
    return result

AURA_EFFECT_MAP = {
    PieceType.SPEEDER: "aura",
    PieceType.SLOWER: "aura",
    PieceType.FREEZER: "aura",
    PieceType.WHITEHOLE: "aura",
    PieceType.BLACKHOLE: "aura",
    PieceType.TRAILBLAZER: "trailblaze",
    PieceType.GEOMANCER: "geomancy"
}

def get_piece_effect_type(piece_type: PieceType) -> Optional[str]:
    """Get standardized effect type for piece."""
    return AURA_EFFECT_MAP.get(piece_type)

def is_effect_piece(piece_type: PieceType) -> bool:
    """Check if piece type has special effects."""
    return piece_type in AURA_EFFECT_MAP

# Backward compatibility alias
list_occupied = iterate_occupied
