# cache_protocols.py - NEW FILE
"""Standardized protocols for cache manager interfaces — NUMPY-NATIVE."""
from __future__ import annotations
from typing import Protocol, Optional, Dict, Set, List, TYPE_CHECKING
import numpy as np
from abc import ABC, abstractmethod

from game3d.common.shared_types import Color, PieceType, Result
from game3d.common.shared_types import (
    EMPTY, WHITE, BLACK,
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    PRIEST, KNIGHT32, KNIGHT31, TRIGONALBISHOP, HIVE, ORBITER, NEBULA,
    ECHO, PANEL, EDGEROOK, XYQUEEN, XZQUEEN, YZQUEEN, VECTORSLIDER,
    CONESLIDER, MIRROR, FREEZER, WALL, ARCHER, BOMB, FRIENDLYTELEPORTER,
    ARMOUR, SPEEDER, SLOWER, GEOMANCER, SWAPPER, XZZIGZAG, YZZIGZAG,
    REFLECTOR, BLACKHOLE, WHITEHOLE, INFILTRATOR, TRAILBLAZER, SPIRAL
)


if TYPE_CHECKING:
    from board.board import Board


class CacheManagerProtocol(Protocol):
    """Standardized interface for cache manager access.

    All coordinates are np.ndarray of shape (3,) and dtype int16 or int32.
    """

    # Core cache properties
    @property
    def occupancy(self): ...

    @property
    def piece_cache(self): ...

    # Movement effect queries
    def is_frozen(self, coord: np.ndarray, color: int) -> bool: ...
    def is_movement_buffed(self, coord: np.ndarray, color: int) -> bool: ...
    def is_movement_debuffed(self, coord: np.ndarray, color: int) -> bool: ...

    # Special effect maps
    def black_hole_pull_map(self, controller: Color) -> Dict[np.ndarray, np.ndarray]: ...
    def white_hole_push_map(self, controller: Color) -> Dict[np.ndarray, np.ndarray]: ...

    # Geomancy and trail effects
    def is_geomancy_blocked(self, sq: np.ndarray, current_ply: int) -> bool: ...
    def current_trail_squares(self, controller: Color) -> Set[np.ndarray]: ...

    # Piece queries
    def get_piece(self, coord: np.ndarray) -> Optional[np.ndarray]: ...
    def set_piece(self, coord: np.ndarray, piece: Optional[np.ndarray]) -> None: ...
    def get_pieces_of_color(self, color: int) -> List[tuple[np.ndarray, np.ndarray]]: ...

    # Board state
    @property
    def board(self) -> Board: ...


class MovementCacheProtocol(Protocol):
    """Standardized interface for movement caches.

    All coordinates in Move objects and methods are np.ndarray (3,).
    """

    def apply_move(self, mv: np.ndarray, color: int) -> bool: ...
    def undo_move(self, mv: np.ndarray, color: int) -> bool: ...
    def legal_moves(self, color: int, **kwargs) -> List[np.ndarray]: ...
    def invalidate_square(self, coord: np.ndarray) -> None: ...
    def invalidate_attacked_squares(self, color: int) -> None: ...


class EffectCacheProtocol(Protocol):
    """Standardized interface for effect caches.

    Coordinates in Move and board state are np.ndarray (3,).
    """

    def apply_move(self, mv: np.ndarray, mover: Color, current_ply: int, board: Board) -> None: ...
    def undo_move(self, mv: np.ndarray, mover: Color, current_ply: int, board: Board) -> None: ...
    def clear(self) -> None: ...
    def invalidate(self) -> None: ...


class CacheListener(ABC):
    """Abstract base class for caches that want to receive dependency notifications."""
    
    @abstractmethod
    def on_occupancy_changed(self, changed_coords: np.ndarray, pieces: np.ndarray) -> None:
        """Called when occupancy changes. Should update internal cache accordingly."""
        raise NotImplementedError("Subclasses must implement on_occupancy_changed")

    @abstractmethod
    def on_batch_occupancy_changed(self, coords: np.ndarray, pieces: np.ndarray) -> None:
        """Called when batch occupancy changes. Should update internal cache accordingly."""
        raise NotImplementedError("Subclasses must implement on_batch_occupancy_changed")

    @abstractmethod
    def get_priority(self) -> int:
        """Return priority for update order (lower = higher priority)."""
        raise NotImplementedError("Subclasses must implement get_priority")
