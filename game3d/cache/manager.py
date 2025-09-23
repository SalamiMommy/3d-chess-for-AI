"""Central cache manager – owns movement + effects + occupancy view."""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Set
import torch
from pieces.enums import Color, PieceType
from game.move import Move
from game3d.board.board import Board
from game3d.cache.movecache import MoveCache, init_cache as _init_move
from game3d.cache.occupancycache import OccupancyCache, init_occupancy_cache, get_occupancy_cache
from game3d.cache.effectscache.freezecache import FreezeCache, init_freeze_cache, get_freeze_cache
from game3d.cache.effectscache.blackholesuckcache import BlackHoleSuckCache, init_black_hole_suck_cache, get_black_hole_suck_cache
from game3d.cache.effectscache.movementdebuffcache import MovementDebuffCache, init_movement_debuff_cache, get_movement_debuff_cache
from game3d.cache.effectscache.movementbuffcache import MovementBuffCache, init_movement_buff_cache, get_movement_buff_cache
from game3d.cache.effectscache.whiteholepushcache import WhiteHolePushCache, init_white_hole_push_cache, get_white_hole_push_cache
from game3d.cache.effectscache.trailblazecache import TrailblazeCache, init_trailblaze_cache, get_trailblaze_cache
from game3d.cache.effectscache.capturefrombehindcache import ArmouredCache, init_armoured_cache, get_armoured_cache
from game3d.cache.effectscache.geomancycache import GeomancyCache, init_geomancy_cache, get_geomancy_cache
from game3d.cache.effectscache.archerycache import ArcheryCache, init_archery_cache, get_archery_cache
from game3d.cache.effectscache.sharesquarecache import ShareSquareCache, init_share_square_cache, get_share_square_cache

class CacheManager:
    def __init__(self, board: Board) -> None:
        self.board = board
        self.occupancy = OccupancyCache(board)
        self.move = MoveCache(board, Color.WHITE)
        self._init_effects()

    def _init_effects(self) -> None:
        self._effect: Dict[str, object] = {
            "freeze":          FreezeCache(self.board),
            "movement_buff":   MovementBuffCache(self.board),
            "movement_debuff": MovementDebuffCache(self.board),
            "black_hole_suck": BlackHoleSuckCache(self.board),
            "white_hole_push": WhiteHolePushCache(self.board),
            "trailblaze":      TrailblazeCache(self.board),
            "geomancy":        GeomancyCache(self.board),
            "archery":         ArcheryCache(self.board),
            "armoured":        ArmouredCache(self.board),
            "share_square":    ShareSquareCache(self.board),
        }

    def _rebuild_occupancy(self) -> None:
        self.occupancy.rebuild(self.board)

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
        self._effect["black_hole_suck"].apply_move(mv, mover)
        self._effect["white_hole_push"].apply_move(mv, mover)
        self._effect["trailblaze"].apply_move(mv, mover)
        self._effect["geomancy"].apply_move(mv, mover, current_ply)
        self._effect["archery"].apply_move(mv, mover)
        self._effect["armoured"].apply_move(mv, mover)
        self._effect["share_square"].apply_move(mv, mover)
        self._effect["capture_from_behind"].apply_move(mv, mover)

    def undo_move(self, mv: Move, mover: Color) -> None:
        self.board.undo_move(mv)
        self._rebuild_occupancy()
        self.move.undo_move(mv, mover)
        for cache in self._effect.values():
            cache.undo_move(mv, mover, self.occupancy)
        self._effect["movement_debuff"].apply_move(mv, mover)
        self._effect["movement_buff"].apply_move(mv, mover)
        self._effect["freeze"].apply_move(mv, mover)
        self._effect["black_hole_suck"].undo_move(mv, mover)
        self._effect["white_hole_push"].undo_move(mv, mover)
        self._effect["trailblaze"].undo_move(mv, mover)
        self._effect["geomancy"].undo_move(mv, mover, current_ply)
        self._effect["archery"].undo_move(mv, mover)
        self._effect["armoured"].undo_move(mv, mover)
        self._effect["share_square"].undo_move(mv, mover)
        self._effect["capture_from_behind"].undo_move(mv, mover)

    def legal_moves(self, color: Color) -> List[Move]:
        return self.move.legal_moves(color)

    def is_frozen(self, sq: Tuple[int, int, int], victim: Color) -> bool:
        return self._effect["freeze"].is_frozen(sq, victim)

    def is_movement_buffed(self, sq: Tuple[int, int, int], friendly: Color) -> bool:
        return self._effect["movement_buff"].is_buffed(sq, friendly)

    def is_movement_debuffed(self, sq: Tuple[int, int, int], victim: Color) -> bool:
        return self._effect["movement_debuff"].is_debuffed(sq, victim)

    def black_hole_pull_map(self, controller: Color) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
        return self._effect["black_hole_suck"].pull_map(controller)

    def white_hole_push_map(self, controller: Color) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
        return self._effect["white_hole_push"].push_map(controller)

    def mark_trail(self, trailblazer_sq: Tuple[int, int, int], slid_squares: Set[Tuple[int, int, int]]) -> None:
        self._effect["trailblaze"].mark_trail(trailblazer_sq, slid_squares)

    def current_trail_squares(self, controller: Color) -> Set[Tuple[int, int, int]]:
        return self._effect["trailblaze"].current_trail_squares(controller)

    def is_geomancy_blocked(self, sq: Tuple[int, int, int], current_ply: int) -> bool:
        return self._effect["geomancy"].is_blocked(sq, current_ply)

    def block_square(self, sq: Tuple[int, int, int], current_ply: int) -> bool:
        return self._effect["geomancy"].block_square(sq, current_ply)

    def archery_targets(self, controller: Color) -> List[Tuple[int, int, int]]:
        return self._effect["archery"].attack_targets(controller)

    def is_valid_archery_attack(self, sq: Tuple[int, int, int], controller: Color) -> bool:
        return self._effect["archery"].is_valid_attack(sq, controller)

    def can_capture_wall(self, attacker_sq: Tuple[int, int, int], wall_sq: Tuple[int, int, int], controller: Color) -> bool:
        return self._effect["armoured"].can_capture(attacker_sq, wall_sq, controller)

    def pieces_at(self, sq: Tuple[int, int, int]) -> List['Piece']:
        return self._effect["share_square"].pieces_at(sq)

    def top_piece(self, sq: Tuple[int, int, int]) -> Optional['Piece']:
        return self._effect["share_square"].top_piece(sq)
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

# Effect cache direct accessors (for use by board, etc.)
def get_share_square_cache() -> ShareSquareCache:
    return get_cache_manager()._effect["share_square"]

def get_armoured_cache() -> ArmouredCache:
    return get_cache_manager()._effect["armoured"]

def get_freeze_cache() -> FreezeCache:
    return get_cache_manager()._effect["freeze"]

def get_trailblaze_cache() -> TrailblazeCache:
    return get_cache_manager()._effect["trailblaze"]

def get_movement_buff_cache() -> MovementBuffCache:
    return get_cache_manager()._effect["movement_buff"]

def get_movement_debuff_cache() -> MovementDebuffCache:
    return get_cache_manager()._effect["movement_debuff"]

def get_black_hole_suck_cache() -> BlackHoleSuckCache:
    return get_cache_manager()._effect["black_hole_suck"]

def get_white_hole_push_cache() -> WhiteHolePushCache:
    return get_cache_manager()._effect["white_hole_push"]

def get_geomancy_cache() -> GeomancyCache:
    return get_cache_manager()._effect["geomancy"]

def get_archery_cache() -> ArcheryCache:
    return get_cache_manager()._effect["archery"]
