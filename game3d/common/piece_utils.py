# game3d/common/piece_utils.py
# ------------------------------------------------------------------
# Piece-related utilities - NUMPY OPTIMIZED VERSION
# ------------------------------------------------------------------
from __future__ import annotations
from typing import List, Tuple, Optional, Iterable, TYPE_CHECKING, Union
import numpy as np

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

def find_king(state: "GameState", color: Union[Color, np.ndarray]) -> Union[Optional[Coord], List[Optional[Coord]]]:
    if isinstance(color, np.ndarray) and color.ndim > 0:
        colors = color.astype(np.int8)
        return [find_king(state, Color(c.item())) for c in colors]
    # Scalar implementation here
    return None

def infer_piece_from_cache(
    cache_manager: "OptimizedCacheManager",
    coord: Union[Coord, np.ndarray],
    fallback_type: PieceType = PieceType.PAWN
) -> Union[Piece, List[Piece]]:
    """
    Infer piece from cache, fallback to given type - optimized numpy version.
    """
    if isinstance(coord, np.ndarray) and coord.ndim > 1:
        coord = coord.astype(np.int16)
        pieces = []
        for i in range(coord.shape[0]):
            single_coord = tuple(coord[i].tolist())
            piece = cache_manager.get_piece(single_coord)
            pieces.append(piece if piece else Piece(Color.WHITE, fallback_type))
        return pieces
    else:
        if isinstance(coord, np.ndarray):
            coord = tuple(coord.tolist())
        piece = cache_manager.get_piece(coord)
        return piece if piece else Piece(Color.WHITE, fallback_type)

def get_player_pieces(state: GameState, color: Union[Color, np.ndarray]) -> Union[List[Tuple[Coord, Piece]], List[List[Tuple[Coord, Piece]]]]:
    """Get all pieces of a given color - numpy optimized."""
    if isinstance(color, np.ndarray) and color.ndim > 0:
        return [get_player_pieces(state, Color(c.item())) for c in color]
    return list(state.cache_manager.get_pieces_of_color(color))

def iterate_occupied(board: "Board", color: Optional[Union[Color, np.ndarray]] = None, cache_manager: Optional["OptimizedCacheManager"] = None) -> Union[Iterable[Tuple[Coord, Piece]], List[Iterable[Tuple[Coord, Piece]]]]:
    """
    Iterate through occupied squares with numpy optimization.
    """
    if isinstance(color, np.ndarray) and color.ndim > 0:
        results = []
        for c in color:
            results.append(list(iterate_occupied(board, Color(c.item()), cache_manager)))
        return results

    cache_mgr = cache_manager or getattr(board, 'cache_manager', None)

    if cache_mgr:
        if color is None:
            for c in (Color.WHITE, Color.BLACK):
                yield from cache_mgr.get_pieces_of_color(c)
        else:
            yield from cache_mgr.get_pieces_of_color(color)
    else:
        # NUMPY OPTIMIZED: Use numpy for board scanning
        occ = board._array[PIECE_SLICE].sum(axis=0) > 0
        indices = np.argwhere(occ)

        color_plane = None
        if color is not None:
            color_plane = board._array[80]  # Color plane

        for idx in indices:
            z, y, x = idx
            piece = board.piece_at((x, y, z))
            if piece and (color is None or piece.color == color):
                yield (x, y, z), piece

def get_pieces_by_type(
    board: "Board",
    ptype: Union[PieceType, np.ndarray],
    color: Optional[Union[Color, np.ndarray]] = None,
    cache_manager: Optional["OptimizedCacheManager"] = None
) -> Union[List[Tuple[Coord, Piece]], List[List[Tuple[Coord, Piece]]]]:
    """
    Return every (coord, piece) on *board* by type - numpy optimized.
    """
    # Handle batch mode for ptype
    if isinstance(ptype, np.ndarray) and ptype.ndim > 0:
        results = []
        for single_ptype in ptype:
            single_color = color[0] if isinstance(color, np.ndarray) and color.ndim > 0 else color
            results.append(get_pieces_by_type(board, PieceType(single_ptype.item()), single_color, cache_manager))
        return results

    # Handle batch mode for color
    if isinstance(color, np.ndarray) and color.ndim > 0:
        results = []
        for single_color in color:
            results.append(get_pieces_by_type(board, ptype, Color(single_color.item()), cache_manager))
        return results

    # Scalar mode - CACHE FIRST APPROACH
    cache_mgr = cache_manager or getattr(board, 'cache_manager', None)

    if cache_mgr:
        colors_to_search = [color] if color is not None else [Color.WHITE, Color.BLACK]
        result = []
        for search_color in colors_to_search:
            result.extend([(coord, piece) for coord, piece in cache_mgr.get_pieces_of_color(search_color)
                          if piece.ptype == ptype])
        return result

    # NUMPY FALLBACK - Optimized for 9x9x9 board
    array = board._array
    piece_planes = array[PIECE_SLICE]

    # Create mask for the wanted piece type
    mask = (piece_planes[ptype.value] > 0.5)

    if color is not None:
        color_plane = array[80]  # Color plane
        color_mask = (color_plane > 0.5) if color is Color.WHITE else (color_plane <= 0.5)
        mask &= color_mask

    indices = np.argwhere(mask)
    coord_color = color if color is not None else Color.WHITE

    return [((int(x), int(y), int(z)), Piece(coord_color, ptype))
            for z, y, x in indices.tolist()]

def get_pieces_by_type_from_cache(
    cache_manager: "OptimizedCacheManager",
    ptype: Union[PieceType, np.ndarray],
    color: Union[Color, np.ndarray]
) -> Union[List[Tuple[Coord, Piece]], List[List[Tuple[Coord, Piece]]]]:
    """
    Direct cache-based lookup for pieces by type and color - numpy optimized.
    """
    if isinstance(ptype, np.ndarray) and ptype.ndim > 0:
        results = []
        for i, single_ptype in enumerate(ptype):
            single_color = color[i] if isinstance(color, np.ndarray) and color.ndim > 0 else color
            results.append(get_pieces_by_type_from_cache(cache_manager, PieceType(single_ptype.item()), single_color))
        return results

    if isinstance(color, np.ndarray):
        color = Color(color.item())

    return [(coord, piece) for coord, piece in cache_manager.get_pieces_of_color(color)
            if piece.ptype == ptype]

# Pre-computed effect mapping for faster lookups
AURA_EFFECT_MAP = {
    PieceType.SPEEDER: "aura",
    PieceType.SLOWER: "aura",
    PieceType.FREEZER: "aura",
    PieceType.WHITEHOLE: "aura",
    PieceType.BLACKHOLE: "aura",
    PieceType.TRAILBLAZER: "trailblaze",
    PieceType.GEOMANCER: "geomancy"
}

EFFECT_PIECE_TYPES = frozenset(AURA_EFFECT_MAP.keys())

def get_piece_effect_type(piece_type: Union[PieceType, np.ndarray]) -> Union[Optional[str], List[Optional[str]]]:
    """Get standardized effect type for piece - numpy optimized."""
    if isinstance(piece_type, np.ndarray) and piece_type.ndim > 0:
        return [AURA_EFFECT_MAP.get(PieceType(pt.item())) for pt in piece_type]
    else:
        if isinstance(piece_type, np.ndarray):
            piece_type = PieceType(piece_type.item())
        return AURA_EFFECT_MAP.get(piece_type)

def is_effect_piece(piece_type: Union[PieceType, np.ndarray]) -> Union[bool, np.ndarray]:
    """Check if piece type has special effects - numpy optimized."""
    if isinstance(piece_type, np.ndarray) and piece_type.ndim > 0:
        results = []
        for pt in piece_type:
            pt_enum = PieceType(pt.item())
            results.append(pt_enum in EFFECT_PIECE_TYPES)
        return np.array(results, dtype=bool)
    else:
        if isinstance(piece_type, np.ndarray):
            piece_type = PieceType(piece_type.item())
        return piece_type in EFFECT_PIECE_TYPES

# Backward compatibility alias
list_occupied = iterate_occupied
