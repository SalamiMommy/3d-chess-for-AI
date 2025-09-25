# game3d/movement/piecemoves/xyqueenmoves.py

from typing import List
from game3d.pieces.enums import PieceType
from game3d.board.board import Board          # <-- add this line
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.xyqueenmovement import generate_xy_queen_moves
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movepiece import Move

def generate_xy_queen_with_king_moves(state, *coord) -> List[Move]:
    """
    Combines XY queen sliding moves + 1-step king moves within XY plane (Z fixed).
    Deduplicates by target coordinate.
    """
    # use the state that was passed in
    queen_moves = generate_xy_queen_moves(state, *coord)
    king_moves  = generate_king_moves(state, *coord)

    # Filter king moves to only those in XY plane (dz = 0)
    z = coord[2]                                    # need original z
    in_plane_king_moves = [
        move for move in king_moves
        if move.to_coord[2] == z
    ]

    # Deduplicate by target square
    seen_targets = {move.to_coord for move in queen_moves}
    combined_moves = queen_moves[:]

    for move in in_plane_king_moves:
        if move.to_coord not in seen_targets:
            combined_moves.append(move)
            seen_targets.add(move.to_coord)

    return combined_moves


@register(PieceType.XYQUEEN)
def xy_queen_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
    from game3d.game.gamestate import GameState
    real_board = board if isinstance(board, Board) else Board(board)
    state = GameState(real_board, color, cache=cache)
    return generate_xy_queen_with_king_moves(state, *coord)


# Re-export for external use
__all__ = ['generate_xy_queen_with_king_moves']
