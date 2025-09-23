# generator.py  (keep the registrar, swap the implementation)
from __future__ import annotations
from typing import Callable, List
from pieces.enums import PieceType
from game.state import GameState
from game.move import Move
from game3d.movement.pseudo_legal import generate_pseudo_legal_moves  # NEW
from game3d.cache.manager import get_cache_manager


_REGISTRY: dict[PieceType, Callable[[GameState, int, int, int], List[Move]]] = {}

def register(pt: PieceType):
    """Decorator."""
    def _decorator(fn):
        _REGISTRY[pt] = fn
        return fn
    return _decorator

def get_dispatcher(pt: PieceType) -> Callable[[GameState, int, int, int], List[Move]] | None:
    """Expose lookup for pseudo-legal generator."""
    return _REGISTRY.get(pt)

# ---- public API ----
def generate_legal_moves(state: GameState) -> List[Move]:
    """Full legal moves (pseudo-legal → legality filter)."""
    # local import → breaks any potential circle
    from game3d.movement.legal import generate_legal_moves as _legal_gen
    return _legal_gen(state)

max_steps = 3
if get_cache_manager().is_movement_buffed(start_sq, state.current):
    max_steps += 1
