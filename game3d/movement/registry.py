from __future__ import annotations
from typing import Callable, List, Dict, TYPE_CHECKING, Tuple
from game3d.common.enums import PieceType, Color
from game3d.movement.movepiece import Move
import numpy as np
from numba import njit
from numba.typed import List as NbList

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

_REGISTRY: Dict[PieceType, Callable[["GameState", int, int, int], List]] = {}

_PIECE_DIRECTIONS = {  # Precompute directions
    # Add directions for each PieceType
}

def register(pt: PieceType):
    def _decorator(fn):
        if pt in _REGISTRY:
            raise ValueError(f"Dispatcher for {pt} already registered.")
        _REGISTRY[pt] = fn
        return fn
    return _decorator

def get_dispatcher(pt: PieceType):
    try:
        return _REGISTRY[pt]
    except KeyError:
        raise ValueError(f"No dispatcher registered for {pt}.") from None

def get_all_dispatchers() -> Dict[PieceType, Callable]:
    return _REGISTRY.copy()

def dispatch_batch(
    state: "GameState",
    piece_coords: List[Tuple[int, int, int]],
    piece_types: List[PieceType],
    color: Color,
) -> List[Move]:
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

@njit(cache=True, nogil=True, parallel=False)
def _batch_kernel(
    coords: NbList[Tuple[int, int, int]],
    types:  NbList[PieceType],
    occ:    np.ndarray,
    color:  int,
):
    out_raw = NbList()
    for i in range(len(coords)):
        cx, cy, cz = coords[i]
        pt         = types[i]

        dirs   = _PIECE_DIRECTIONS[pt]  # Use precomputed
        local_out = NbList()
        for d in range(dirs.shape[0]):
            dx, dy, dz = dirs[d]
            for step in range(1, 9):
                nx = cx + dx * step
                ny = cy + dy * step
                nz = cz + dz * step
                if not (0 <= nx < 9 and 0 <= ny < 9 and 0 <= nz < 9):
                    break

                occ_code = occ[nz, ny, nx]
                if occ_code == 0:
                    local_out.append((cx, cy, cz, nx, ny, nz, 0))
                elif occ_code != color:
                    local_out.append((cx, cy, cz, nx, ny, nz, 1))
                    break
                else:
                    break
        out_raw.append(local_out)
    return out_raw

__all__ = ["register", "get_dispatcher", "get_all_dispatchers", "dispatch_batch"]
