"""Central cache manager – owns movement + effects + occupancy view."""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import torch
from pieces.enums import Color
from game.move import Move
from game3d.board.board import Board
from game3d.cache.movecache import MoveCache, init_cache as _init_move
from game3d.cache.effectscache.freezecache import FreezeCache, init_freeze_cache as _init_freeze
from game3d.cache.effectscache.slowcache import SlowCache, init_slow_cache as _init_slow
# … import other 8 effect caches the same way


from game3d.cache.occupancycache import OccupancyCache, init_occupancy_cache, get_occupancy_cache

class CacheManager:
    def __init__(self, board: Board) -> None:
        self.board = board
        self.occupancy = OccupancyCache(board)   # *** NEW ***
        self.move = MoveCache(board, Color.WHITE)
        self._init_effects()

    def _rebuild_occupancy(self) -> None:
        """Called after every make/undo."""
        self.occupancy.rebuild(self.board)

    # ----------------------------------------------------------
    # effect caches – each receives the same occupancy view
    # ----------------------------------------------------------
    def _init_effects(self) -> None:
        self._effect: Dict[str, object] = {
            "freeze": FreezeCache(board, self.occupancy),
            "slow":   SlowCache(board, self.occupancy),
            "movement_buff": MovementBuffCache(board),   # <-- NEW
        }

    # ----------------------------------------------------------
    # public accessors
    # ----------------------------------------------------------
    def legal_moves(self, color: Color) -> List[Move]:
        return self.move.legal_moves(color)

    def is_frozen(self, sq: Tuple[int, int, int], victim: Color) -> bool:
        return self._effect["freeze"].is_frozen(sq, victim)

    def is_slowed(self, sq: Tuple[int, int, int], victim: Color) -> bool:
        return self._effect["slow"].is_slowed(sq, victim)
    # … add thin wrappers for the other 8

    # ----------------------------------------------------------
    # make / undo – single entry point for GameState
    # ----------------------------------------------------------
    def apply_move(self, mv: Move, mover: Color) -> None:
        self.board.apply_move(mv)
        self._rebuild_occupancy()          # *** ONE scan ***
        self.move.apply_move(mv, mover)    # incremental
        for cache in self._effect.values():
            cache.apply_move(mv, mover, self.occupancy)  # pass view
        self._effect["movement_debuff"].apply_move(mv, mover)
        self._effect["movement_buff"].apply_move(mv, mover)
        self._effect["freeze"].apply_move(mv, mover)

    def undo_move(self, mv: Move, mover: Color) -> None:
        self.board.undo_move(mv)
        self._rebuild_occupancy()
        self.move.undo_move(mv, mover)
        for cache in self._effect.values():
            cache.undo_move(mv, mover, self.occupancy)
        self._effect["movement_debuff"].apply_move(mv, mover)
        self._effect["movement_buff"].apply_move(mv, mover)
        self._effect["freeze"].apply_move(mv, mover)

    def is_movement_buffed(self, sq: Tuple[int, int, int], friendly: Color) -> bool:
        return self._effect["movement_buff"].is_buffed(sq, friendly)

    def is_movement_debuffed(self, sq: Tuple[int, int, int], victim: Color) -> bool:
        return self._effect["movement_debuff"].is_debuffed(sq, victim)
# ------------------------------------------------------------------
# module-level singleton
# ------------------------------------------------------------------
_manager: Optional[CacheManager] = None


def init_cache_manager(board: Board) -> None:
    global _manager
    _manager = CacheManager(board)


def get_cache_manager() -> CacheManager:
    if _manager is None:
        raise RuntimeError("CacheManager not initialised")
    return _manager
