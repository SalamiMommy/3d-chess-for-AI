from typing import List
from game3d.pieces.enums import PieceType
from game3d.board.board import Board          # <─ keep
from game3d.movement.registry import register
from game3d.movement.movetypes.pawnfrontteleportmovement import generate_pawn_front_teleport_moves
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movepiece import Move

def generate_pawn_front_teleport_with_king_moves(
    state,          # <-- now takes a GameState directly
    *coord
) -> List[Move]:
    """Combine pawn-front teleport moves + 1-step king moves, deduplicated."""
    teleport_moves = generate_pawn_front_teleport_moves(state, *coord)
    king_moves       = generate_king_moves(state, *coord)

    seen = {m.to_coord for m in teleport_moves}
    combined = teleport_moves[:]
    for m in king_moves:
        if m.to_coord not in seen:
            combined.append(m)
            seen.add(m.to_coord)
    return combined


@register(PieceType.INFILTRATOR)
def pawn_front_teleport_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
    from game3d.game.gamestate import GameState
    real_board = board if isinstance(board, Board) else Board(board)
    state = GameState(real_board, color, cache=cache)
    # pass the GameState, not the raw board
    return generate_pawn_front_teleport_with_king_moves(state, *coord)

__all__ = ['generate_pawn_front_teleport_with_king_moves']
