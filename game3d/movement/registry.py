from __future__ import annotations
from typing import Callable, List, Dict, TYPE_CHECKING, Tuple
from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move
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

# ------------------------------------------------------------------
# Fast batch driver – one Numba call for every slider of one colour
# ------------------------------------------------------------------
def dispatch_batch(
    state: "GameState",
    piece_coords: List[Tuple[int, int, int]],
    piece_types: List[PieceType],
    color: Color,
) -> List[Move]:
    """Generate every pseudo-legal move for all pieces in the two input lists
    in one go.  Returns a flat Python list[Move]."""
    if not piece_coords:                       # fast-path empty
        return []

    # convert to Numba-typed containers once
    nb_coords = NbList([(x, y, z) for x, y, z in piece_coords])
    nb_types  = NbList(piece_types)

    # call the JIT-ed batch kernel
    raw = _batch_kernel(
        nb_coords,
        nb_types,
        state.cache.piece_cache.get_occupancy_view(),
        color.value,
    )

    # build Move objects in Python (single linear pass)
    return [
        Move(
            from_coord=(fr_x, fr_y, fr_z),
            to_coord  =(to_x, to_y, to_z),
            is_capture=ic,
            captured_piece=None,
        )
        for fr_x, fr_y, fr_z, to_x, to_y, to_z, ic in raw
    ]

# ------------------------------------------------------------------
# Numba kernel – loops inside Numba land, zero Python loops
# ------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _batch_kernel(
    coords: NbList[Tuple[int, int, int]],
    types:  NbList[PieceType],
    occ:    np.ndarray,          # (9,9,9) uint8
    color:  int,                 # 1 or 2
):
    out_raw = []
    for i in range(len(coords)):
        cx, cy, cz = coords[i]
        pt         = types[i]

        # ---- re-use the already JIT-ed sliding kernel ----
        dirs   = _get_directions(pt)          # small const array
        starts = np.array([[cx, cy, cz]], dtype=np.int8)
        # import at top of file:  from game3d.movement.movetypes.slidermovement import _slide_kernel
        coords3, valid, hit = _slide_kernel(starts, dirs, 8, occ)

        # ---- unpack the result exactly like _build_compact ----
        for d in range(coords3.shape[0]):
            for s in range(coords3.shape[1]):
                if not valid[d, s]:
                    continue
                tx, ty, tz = coords3[d, s]
                h = hit[d, s]
                if h == 0:                       # empty square
                    out_raw.append((cx, cy, cz, tx, ty, tz, False))
                else:                            # capture or block
                    if h != color:               # not our own piece
                        out_raw.append((cx, cy, cz, tx, ty, tz, True))
                    break                        # ray blocked
    return out_raw

# ------------------------------------------------------------------
# tiny helper – directions per piece type (same table the dispatcher uses)
# ------------------------------------------------------------------
@njit(cache=True)
def _get_directions(pt: PieceType) -> np.ndarray:
    # keep this in sync with the real direction tables
    if pt.value == PieceType.ROOK.value:
        return np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=np.int8)
    if pt.value == PieceType.BISHOP.value:
        return np.array([[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]], dtype=np.int8)
    if pt.value == PieceType.QUEEN.value:
        return np.array([
            [1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],
            [1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]
        ], dtype=np.int8)
    # fallback – empty
    return np.empty((0, 3), dtype=np.int8)

__all__ = ["register", "get_dispatcher", "get_all_dispatchers", "dispatch_batch"]
