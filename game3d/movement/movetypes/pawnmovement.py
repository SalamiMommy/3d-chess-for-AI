from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move
from game3d.common.common import in_bounds
from game3d.cache.manager import OptimizedCacheManager

def _is_on_start_rank(z: int, color: Color) -> bool:
    return (color == Color.WHITE and z == 1) or (color == Color.BLACK and z == 7)

def _create_pawn_move(
    from_pos: tuple[int, int, int],
    to_pos: tuple[int, int, int],
    color: Color,
    is_capture: bool = False,
    is_en_passant: bool = False
) -> Move:
    _, _, tz = to_pos
    is_promotion = (color == Color.WHITE and tz == 8) or (color == Color.BLACK and tz == 0)
    return Move(
        from_coord=from_pos,
        to_coord=to_pos,
        is_capture=is_capture,
        is_promotion=is_promotion,
        is_en_passant=is_en_passant
    )

def generate_pawn_moves(
    cache: OptimizedCacheManager,  # ← CHANGED: board → cache
    color: Color,    # ← current player color
    x: int, y: int, z: int
) -> List['Move']:
    """
    Generate all legal pawn moves from (x, y, z).
    """
    pos = (x, y, z)
    piece = cache.piece_cache.get(pos)
    if piece is None or piece.color != color or piece.ptype != PieceType.PAWN:
        return []

    dz = 1 if color == Color.WHITE else -1
    moves: List['Move'] = []

    # --- Forward moves ---
    forward = (x, y, z + dz)
    if in_bounds(forward) and cache.piece_cache.get(forward) is None:
        moves.append(_create_pawn_move(pos, forward, color))
        if _is_on_start_rank(z, color):
            double_forward = (x, y, z + 2 * dz)
            if in_bounds(double_forward) and cache.piece_cache.get(double_forward) is None:
                moves.append(_create_pawn_move(pos, double_forward, color))

    # --- Capture logic ---
    def _can_capture_pawn(target_piece) -> bool:
        if target_piece is None or target_piece.color == color:
            return False
        if target_piece.ptype == PieceType.ARMOUR:
            return False
        return True

    # XZ, YZ, and 3D diagonals
    for dx in (-1, 1):
        for dy in (-1, 0, 1):  # include dy=0 for XZ
            for dz_offset in [dz]:  # only one step forward
                if dx == 0 and dy == 0:
                    continue  # skip forward (already handled)
                target = (x + dx, y + dy, z + dz)
                if in_bounds(target):
                    if _can_capture_pawn(cache.piece_cache.get(target)):
                        moves.append(_create_pawn_move(pos, target, color, is_capture=True))

    return moves
