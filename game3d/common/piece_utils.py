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

@njit(cache=True)
def color_to_code(color: "Color") -> int:
    """Return the occupancy-array code (1 or 2) for the given Color enum."""
    return 1 if color.value == 1 else 2

def find_king(state: "GameState", color: Color) -> Optional[Coord]:
    """Vectorised, lock-free search for the king of *color*."""
    for coord, piece in state.cache.piece_cache.iter_color(color):
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
    piece = cache_manager.occupancy.get(coord)
    if piece:
        return piece
    # Fallback - need to know the color
    # This is a limitation of the current API
    return Piece(Color.WHITE, fallback_type)

def get_player_pieces(state: GameState, color: Color) -> List[Tuple[Coord, Piece]]:
    result = []
    # BETTER - use piece_cache property
    for coord, piece_data in state.cache.piece_cache.iter_color(color):
        if not isinstance(piece_data, Piece):
            continue
        result.append((coord, piece_data))
    return result

def iterate_occupied(board: "Board", color: Optional[Color] = None):
    # print(f"[ITER-ENTER] cache_manager exists: {board.cache_manager is not None}")
    # if board.cache_manager:  # fast path
    #     print("[ITER-ENTER] taking fast path")
    #     if color is None:
    #         for c in [Color.WHITE, Color.BLACK]:
    #             print(f"[ITER] iter_color {c}:")
    #             for coord, piece_data in board.cache_manager.occupancy.iter_color(c):
    #                 print(f"[ITER]   got {coord} -> {piece_data} (type {type(piece_data)})")
    if board.cache_manager:  # fast path
        if color is None:
            for c in [Color.WHITE, Color.BLACK]:
                for coord, piece_data in board.cache_manager.occupancy.iter_color(c):
                    # piece_data should already be a Piece object from iter_color
                    if not isinstance(piece_data, Piece):
                        # Defensive fallback
                        if isinstance(piece_data, PieceType):
                            piece = Piece(c, piece_data)
                        else:
                            print(f"[ERROR] Unexpected data type in iterate_occupied: {type(piece_data)}")
                            continue
                    else:
                        piece = piece_data
                    yield coord, piece
        else:
            for coord, piece_data in board.cache_manager.occupancy.iter_color(color):
                # piece_data should already be a Piece object
                if not isinstance(piece_data, Piece):
                    # Defensive fallback
                    if isinstance(piece_data, PieceType):
                        piece = Piece(color, piece_data)
                    else:
                        print(f"[ERROR] Unexpected data type in iterate_occupied: {type(piece_data)}")
                        continue
                else:
                    piece = piece_data

                if piece and (color is None or piece.color == color):
                    yield coord, piece
    else:  # slow path, tensor scan
        occ = board._tensor[PIECE_SLICE].sum(dim=0) > 0
        indices = torch.nonzero(occ, as_tuple=False)
        for z, y, x in indices.tolist():
            piece = board.piece_at((x, y, z))
            if piece and (color is None or piece.color == color):
                yield (x, y, z), piece

def get_pieces_by_type(
    board: "Board",
    ptype: PieceType,
    color: Optional[Color] = None
) -> List[Tuple[Coord, Piece]]:
    """
    Return every (coord, piece) on *board* whose type == *ptype*
    (and optionally colour == *color*).
    Uses the occupancy cache if already available, otherwise falls back
    to a direct tensor scan.
    FIXED: Properly handles Piece objects from iter_color.
    """
    # Fast path – cache is ready
    if board.cache_manager is not None:
        result = []
        for sq, piece_data in board.cache_manager.occupancy.iter_color(color):
            # piece_data is a Piece object from iter_color
            if not isinstance(piece_data, Piece):
                # Defensive fallback - should not happen
                if isinstance(piece_data, PieceType):
                    if piece_data == ptype:
                        result.append((sq, Piece(color, ptype)))
                continue

            # Normal case: piece_data is a Piece
            if piece_data.ptype == ptype:
                result.append((sq, piece_data))
        return result

    # Slow path – cache not yet attached (e.g. during initial aura rebuild)
    tensor = board._tensor
    piece_planes = tensor[PIECE_SLICE]
    col_plane = tensor[N_PIECE_TYPES]

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
