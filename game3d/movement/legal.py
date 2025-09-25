"""Legal-move filter – now just a thin cache accessor."""
#game3d/movement/legal.py
from __future__ import annotations
from typing import List
from game3d.cache.movecache import get_cache
from game3d.cache.effectscache.freezecache import get_freeze_cache
from game3d.movement.pseudo_legal import generate_pseudo_legal_moves

def generate_legal_moves(state: GameState) -> List[Move]:
    print("generate_pseudo_legal_moves:", generate_pseudo_legal_moves)
    legal = []
    freeze = get_freeze_cache()
    for mv in generate_pseudo_legal_moves(state):
        if freeze.is_frozen(mv.from_coord, state.current):
            continue                     # piece is paralysed
        tmp = state.clone()
        tmp.board.apply_move(mv)
        tmp.current = state.current
        if not king_in_check(tmp.board, tmp.current, state.current):
            legal.append(mv)
    return legal
