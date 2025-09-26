from __future__ import annotations
#game3d/movement/legal.py
"""Legal-move filter – now just a thin cache accessor."""


from typing import List
from game3d.game.gamestate import GameState
from game3d.movement.movepiece import Move
# REMOVED: from game3d.cache.movecache import get_cache
from game3d.movement.pseudo_legal import generate_pseudo_legal_moves
from game3d.attacks.check import king_in_check

def generate_legal_moves(state: GameState) -> List[Move]:
    legal: List[Move] = []
    freeze = state.cache._effect["freeze"]
    color = state.color

    for mv in generate_pseudo_legal_moves(state):
        if freeze.is_frozen(mv.from_coord, color):
            continue

        tmp_board = state.board.clone()
        tmp_board.apply_move(mv)

        # Pass the cache to king_in_check
        if not king_in_check(tmp_board, color, color, state.cache):
            legal.append(mv)

    return legal
