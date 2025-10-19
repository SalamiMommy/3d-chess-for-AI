"""White-Hole – moves like a Speeder and pushes enemies 1 step away at turn end."""

from __future__ import annotations
from typing import List, Dict, Tuple, TYPE_CHECKING
import torch

from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movepiece import Move
from game3d.common.common import add_coords, in_bounds
from game3d.pieces.piece import Piece
if TYPE_CHECKING:
    from game3d.pieces.pieces.auras.aura import BoardProto
    from game3d.movement.cache import OptimizedCacheManager

# ------------------------------------------------------------------
#  Internal helpers
# ------------------------------------------------------------------
def _away(pos: Tuple[int, int, int], target: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """1 Chebyshev step from pos *away* from target."""
    x, y, z = pos
    tx, ty, tz = target
    dx = 0 if x == tx else (1 if tx < x else -1)
    dy = 0 if y == ty else (1 if ty < y else -1)
    dz = 0 if z == tz else (1 if tz < z else -1)
    return add_coords(pos, (dx, dy, dz))

def _pieces_from_board(board):
    """Cold-start fallback: produce (coord, Piece) tuples."""
    tensor = board.tensor()          # (C, D, H, W)
    occ = tensor[:40].sum(dim=0)     # any piece plane > 0
    idx = torch.nonzero(occ, as_tuple=False)  # (N, 3)  -> z,y,x
    for z, y, x in idx.tolist():
        coord = (x, y, z)
        # find first plane that is 1.0 for this square
        for p in range(40):
            if tensor[p, z, y, x] > 0.5:
                # determine colour
                white_plane = tensor[80, z, y, x]
                color = Color.WHITE if white_plane > 0.5 else Color.BLACK
                yield coord, Piece(color, PieceType(p))   # ← Piece object, not PieceType
                break
# ------------------------------------------------------------------
#  Public API
# ------------------------------------------------------------------
def generate_whitehole_moves(
    cache_manager,
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """White-Hole moves exactly like a Speeder (king single steps)."""
    return generate_king_moves(cache_manager, color, x, y, z)


def push_candidates(
    board: BoardProto,
    controller: Color,
    cache_manager: OptimizedCacheManager | None = None
) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Return dict {enemy_square: push_target} for every enemy within
    2-sphere of any friendly WHITE_HOLE.  Push target is 1 step away
    from the nearest hole (first hole found).
    """
    from game3d.common.common import get_pieces_by_type, chebyshev_distance

    out: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    holes: list[Tuple[int, int, int]] = [
        coord for coord, _ in get_pieces_by_type(board, PieceType.WHITEHOLE, controller)
    ]
    if not holes:
        return out

    if board.cache_manager is not None:                      # fast path
        iterable = board.cache_manager.occupancy.iter_color(None)
    else:                                                    # cold-start path
        iterable = _pieces_from_board(board)

    for coord, piece in iterable:
        if piece.color == controller:
            continue
        for hole in holes:
            if chebyshev_distance(coord, hole) <= 2:
                push = _away(coord, hole)
                if in_bounds(push):
                    out[coord] = push
                break  # push away from first hole only
    return out

# ------------------------------------------------------------------
#  Dispatcher registration
# ------------------------------------------------------------------
@register(PieceType.WHITEHOLE)
def whitehole_move_dispatcher(state, x: int, y: int, z: int) -> List[Move]:
    return generate_whitehole_moves(state.cache, state.color, x, y, z)
