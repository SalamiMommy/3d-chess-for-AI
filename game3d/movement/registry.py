# game3d/movement/registry.py
from __future__ import annotations
from typing import Callable, List, Dict, TYPE_CHECKING
from game3d.pieces.enums import PieceType
if TYPE_CHECKING:
    from game3d.game.gamestate import GameState   # noqa: F401

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
