# game3d/common/piece_utils.py
# ------------------------------------------------------------------
# Piece-related utilities - OPTIMIZED VERSION
# ------------------------------------------------------------------
from __future__ import annotations
from typing import List, Tuple, Optional, Iterable, TYPE_CHECKING, Union
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

def find_king(state: "GameState", color: Union[Color, torch.Tensor]) -> Union[Optional[Coord], List[Optional[Coord]]]:
    if isinstance(color, torch.Tensor) and color.ndim > 0:
        # Vectorized batch: Avoid loop, but since cache access isn't vectorized, loop is fine. Use torch.int8 for colors.
        colors = color.to(torch.int8)
        return [find_king(state, Color(c.item())) for c in colors]

def infer_piece_from_cache(
    cache_manager: "OptimizedCacheManager",
    coord: Union[Coord, torch.Tensor],
    fallback_type: PieceType = PieceType.PAWN
) -> Union[Piece, List[Piece]]:
    """
    Infer piece from cache, fallback to given type - supports scalar and batch mode.
    FIXED: Handles Piece objects correctly - optimized with direct cache access.
    """
    if isinstance(coord, torch.Tensor) and coord.ndim > 1:
            coord = coord.to(torch.int8)  # Minimize: Cast once.
            pieces = []
            for i in range(coord.shape[0]):
                single_coord = tuple(coord[i].tolist())
                piece = cache_manager.get_piece(single_coord)
                pieces.append(piece if piece else Piece(Color.WHITE, fallback_type))
                return pieces
    else:
        # Scalar mode
        if isinstance(coord, torch.Tensor):
            coord = tuple(coord.tolist())
        piece = cache_manager.get_piece(coord)
        return piece if piece else Piece(Color.WHITE, fallback_type)

def get_player_pieces(state: GameState, color: Union[Color, torch.Tensor]) -> Union[List[Tuple[Coord, Piece]], List[List[Tuple[Coord, Piece]]]]:
    """Get all pieces of a given color - supports scalar and batch mode."""
    if isinstance(color, torch.Tensor) and color.ndim > 0:
        # Batch mode
        return [get_player_pieces(state, Color(c.item())) for c in color]

    # Scalar mode
    # Convert generator to list directly for better performance
    return list(state.cache_manager.get_pieces_of_color(color))

def iterate_occupied(board: "Board", color: Optional[Union[Color, torch.Tensor]] = None, cache_manager: Optional["OptimizedCacheManager"] = None) -> Union[Iterable[Tuple[Coord, Piece]], List[Iterable[Tuple[Coord, Piece]]]]:
    """
    Iterate through occupied squares with standardized cache access - optimized color handling.
    Supports scalar and batch mode.
    """
    if isinstance(color, torch.Tensor) and color.ndim > 0:
        # Batch mode
        results = []
        for c in color:
            results.append(list(iterate_occupied(board, Color(c.item()), cache_manager)))
        return results

    # Scalar mode
    cache_mgr = cache_manager or getattr(board, 'cache_manager', None)

    if cache_mgr:
        # Fast path - cache is available
        if color is None:
            # Single loop for both colors
            for c in (Color.WHITE, Color.BLACK):
                yield from cache_mgr.get_pieces_of_color(c)
        else:
            yield from cache_mgr.get_pieces_of_color(color)
    else:
        # Slow path - tensor scan (fallback) - optimized tensor operations
        occ = board._tensor[PIECE_SLICE].sum(dim=0) > 0
        indices = torch.nonzero(occ, as_tuple=False)

        # Pre-compute color plane if needed
        color_plane = None
        if color is not None:
            color_plane = board._tensor[80]  # Assuming color plane at index 80

        for z, y, x in indices.tolist():
            piece = board.piece_at((x, y, z))
            if piece and (color is None or piece.color == color):
                yield (x, y, z), piece

def get_pieces_by_type(
    board: "Board",
    ptype: Union[PieceType, torch.Tensor],
    color: Optional[Union[Color, torch.Tensor]] = None,
    cache_manager: Optional["OptimizedCacheManager"] = None
) -> Union[List[Tuple[Coord, Piece]], List[List[Tuple[Coord, Piece]]]]:
    """
    Return every (coord, piece) on *board* whose type == *ptype*
    (and optionally colour == *color*) - optimized cache usage.
    Supports scalar and batch mode.
    """
    # Handle batch mode for ptype
    if isinstance(ptype, torch.Tensor) and ptype.ndim > 0:
        # Batch mode
        results = []
        for single_ptype in ptype:
            single_color = color[0] if isinstance(color, torch.Tensor) and color.ndim > 0 else color
            results.append(get_pieces_by_type(board, PieceType(single_ptype.item()), single_color, cache_manager))
        return results

    # Handle batch mode for color
    if isinstance(color, torch.Tensor) and color.ndim > 0:
        # Batch mode
        results = []
        for single_color in color:
            results.append(get_pieces_by_type(board, ptype, Color(single_color.item()), cache_manager))
        return results

    # Scalar mode
    cache_mgr = cache_manager or getattr(board, 'cache_manager', None)

    if cache_mgr:
        # Fast path – cache is available
        result = []
        colors_to_search = [color] if color is not None else [Color.WHITE, Color.BLACK]

    return [(coord, piece) for coord, piece in cache_mgr.get_pieces_of_color(color)
            if piece.ptype == ptype]

    # Slow path – fallback to tensor operations
    tensor = board._tensor
    piece_planes = tensor[PIECE_SLICE]

    # Create mask for the wanted piece type
    mask = (piece_planes[ptype.value] > 0.5).bool()

    if color is not None:
        color_plane = tensor[80]  # Assuming color plane at index 80
        color_mask = (color_plane > 0.5) if color is Color.WHITE else (color_plane <= 0.5)
        mask &= color_mask

    indices = torch.nonzero(mask, as_tuple=False)
    coord_color = color if color is not None else Color.WHITE

    return [((int(x), int(y), int(z)), Piece(coord_color, ptype))
            for z, y, x in indices.tolist()]

def get_pieces_by_type_from_cache(
    cache_manager: "OptimizedCacheManager",
    ptype: Union[PieceType, torch.Tensor],
    color: Union[Color, torch.Tensor]
) -> Union[List[Tuple[Coord, Piece]], List[List[Tuple[Coord, Piece]]]]:
    """
    Direct cache-based lookup for pieces by type and color - optimized list comprehension.
    Supports scalar and batch mode.
    """
    # Handle batch mode
    if isinstance(ptype, torch.Tensor) and ptype.ndim > 0:
        # Batch mode
        results = []
        for i, single_ptype in enumerate(ptype):
            single_color = color[i] if isinstance(color, torch.Tensor) and color.ndim > 0 else color
            results.append(get_pieces_by_type_from_cache(cache_manager, PieceType(single_ptype.item()), single_color))
        return results

    # Scalar mode
    if isinstance(color, torch.Tensor):
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

# Pre-computed set for membership testing
EFFECT_PIECE_TYPES = frozenset(AURA_EFFECT_MAP.keys())

def get_piece_effect_type(piece_type: Union[PieceType, torch.Tensor]) -> Union[Optional[str], List[Optional[str]]]:
    """Get standardized effect type for piece - optimized dict lookup. Supports scalar and batch mode."""
    if isinstance(piece_type, torch.Tensor) and piece_type.ndim > 0:
        # Batch mode
        return [AURA_EFFECT_MAP.get(PieceType(pt.item())) for pt in piece_type]
    else:
        # Scalar mode
        if isinstance(piece_type, torch.Tensor):
            piece_type = PieceType(piece_type.item())
        return AURA_EFFECT_MAP.get(piece_type)

def is_effect_piece(piece_type: Union[PieceType, torch.Tensor]) -> Union[bool, torch.Tensor]:
    """Check if piece type has special effects - optimized set membership. Supports scalar and batch mode."""
    if isinstance(piece_type, torch.Tensor) and piece_type.ndim > 0:
        # Batch mode
        results = []
        for pt in piece_type:
            pt_enum = PieceType(pt.item())
            results.append(pt_enum in EFFECT_PIECE_TYPES)
        return torch.tensor(results, dtype=torch.bool)
    else:
        # Scalar mode
        if isinstance(piece_type, torch.Tensor):
            piece_type = PieceType(piece_type.item())
        return piece_type in EFFECT_PIECE_TYPES

# Backward compatibility alias
list_occupied = iterate_occupied
