# game3d/movement/registry.py
from typing import Dict, Callable, Optional, List
from game3d.pieces.enums import PieceType
from game3d.movement.movepiece import Move

# Registry mapping PieceType to move generation functions
_move_dispatchers: Dict[PieceType, Callable[["GameState", int, int, int], List[Move]]] = {}

def register(piece_type: PieceType):
    """Decorator to register a move generator for a piece type."""
    def decorator(func: Callable[["GameState", int, int, int], List[Move]]) -> Callable:
        _move_dispatchers[piece_type] = func
        return func
    return decorator

def get_dispatcher(piece_type: PieceType) -> Optional[Callable[["GameState", int, int, int], List[Move]]]:
    """Get the move generator for a piece type, or None if not registered."""
    return _move_dispatchers.get(piece_type)

def get_all_dispatchers() -> Dict[PieceType, Callable[["GameState", int, int, int], List[Move]]]:
    """Get all registered dispatchers."""
    return _move_dispatchers.copy()
