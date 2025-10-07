from __future__ import annotations
from typing import Callable, List, Dict, TYPE_CHECKING, Tuple
from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move  # ADDED: Missing import
import numpy as np
from numba import njit
from numba.typed import List as NbList

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

_REGISTRY: Dict[PieceType, Callable[["GameState", int, int, int], List]] = {}

def register(pt: PieceType):
    """Decorator that stores a move-generator for a piece-type."""
    def _decorator(fn):
        if pt in _REGISTRY:
            raise ValueError(f"Dispatcher for {pt} already registered.")
        _REGISTRY[pt] = fn
        return fn
    return _decorator

def get_dispatcher(pt: PieceType):
    """Return the move-generator function registered for *pt*."""
    try:
        return _REGISTRY[pt]
    except KeyError:
        raise ValueError(f"No dispatcher registered for {pt}.") from None

def get_all_dispatchers() -> Dict[PieceType, Callable]:
    """Return a shallow copy of the whole registry."""
    return _REGISTRY.copy()

def dispatch_batch(
    state: "GameState",
    piece_coords: List[Tuple[int, int, int]],
    piece_types: List[PieceType],
    color: Color,
) -> List[Move]:
    """Generate every pseudo-legal move for all pieces in the two input lists
    in one go.  Returns a flat Python list[Move]."""
    if not piece_coords:
        return []

    nb_coords = NbList([(x, y, z) for x, y, z in piece_coords])
    nb_types  = NbList(piece_types)

    raw = _batch_kernel(
        nb_coords,
        nb_types,
        state.cache.piece_cache.get_occupancy_view(),
        color.value,
    )

    return [
        Move(
            from_coord=(fr_x, fr_y, fr_z),
            to_coord  =(to_x, to_y, to_z),
            flags=MOVE_FLAGS['CAPTURE'] if ic else 0,
            captured_piece=None,
        )
        for fr_x, fr_y, fr_z, to_x, to_y, to_z, ic in raw
    ]

@njit(cache=True, nogil=True)
def _batch_kernel(
    coords: NbList[Tuple[int, int, int]],
    types:  NbList[PieceType],
    occ:    np.ndarray,
    color:  int,
):
    out_raw = []
    for i in range(len(coords)):
        cx, cy, cz = coords[i]
        pt         = types[i]

        dirs   = _get_directions(pt)
        for d in range(dirs.shape[0]):
            dx, dy, dz = dirs[d]
            for step in range(1, 9):  # Board is 9x9x9
                nx = cx + dx * step
                ny = cy + dy * step
                nz = cz + dz * step
                if not (0 <= nx < 9 and 0 <= ny < 9 and 0 <= nz < 9):
                    break

                occ_code = occ[nz, ny, nx]
                if occ_code == 0:  # Empty
                    out_raw.append((cx, cy, cz, nx, ny, nz, 0))
                elif occ_code != color:  # Enemy
                    out_raw.append((cx, cy, cz, nx, ny, nz, 1))
                    break
                else:  # Friendly piece
                    break
    return out_raw

@njit(cache=True)
def _get_directions(pt: PieceType) -> np.ndarray:
    if pt.value == PieceType.ROOK.value:
        return np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=np.int8)
    if pt.value == PieceType.BISHOP.value:
        return np.array([[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]], dtype=np.int8)
    if pt.value == PieceType.QUEEN.value:
        return np.array([
            [1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],
            [1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]
        ], dtype=np.int8)
    return np.empty((0, 3), dtype=np.int8)

__all__ = ["register", "get_dispatcher", "get_all_dispatchers", "dispatch_batch"]
