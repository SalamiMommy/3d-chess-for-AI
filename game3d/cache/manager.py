"""Central cache manager – owns movement + effects + occupancy view."""
#game3d/cache/manager.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Set
import torch
from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move
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
        self._effect: Dict[str, Any] = {}
        self._init_effects()

        # create the object but do NOT rebuild yet
        self._move_cache = MoveCache(board, Color.WHITE, self, _defer_rebuild=True)

    def initialise(self, current: Color) -> None:
        from game3d.cache.movecache import init_cache as init_move_cache
        from game3d.cache.occupancycache import init_occupancy_cache
        from game3d.cache.effectscache.freezecache import init_freeze_cache

        # Initialize caches
        init_occupancy_cache(self.board)
        init_freeze_cache()  # ← No board argument
        init_move_cache(self.board, current)

    def _init_effects(self) -> None:
        self._effect = {
            "freeze":          FreezeCache(),          # ← removed board
            "movement_buff":   MovementBuffCache(),
            "movement_debuff": MovementDebuffCache(),
            "black_hole_suck": BlackHoleSuckCache(),
            "white_hole_push": WhiteHolePushCache(),
            "trailblaze":      TrailblazeCache(),
            "armoured":        ArmouredCache(),
            "geomancy":        GeomancyCache(),
            "archery":         ArcheryCache(),
            "share_square":    ShareSquareCache(),
        }

    def _rebuild_occupancy(self) -> None:
        self.occupancy.rebuild(self.board)

    def sync_board(self, board: Board) -> None:
        """Replace every cache’s internal board with a clone of *board*."""
        self.board = board.clone()
        for cache in self._effect.values():
            if hasattr(cache, "_board"):   # every effect cache has its own mirror
                cache._board = self.board.clone()

    def replace_board(self, board: Board) -> None:
        """Throw away the old tensor and adopt *board*."""
        self.board = board.clone()          # authoritative source
        self.occupancy.rebuild(self.board)  # occupancy view
        # optional: force every effect cache to rebuild now
        for cache in self._effect.values():
            if hasattr(cache, "_rebuild"):
                cache._rebuild(self.board)

    def apply_move(self, mv: Move, mover: Color, current_ply: int = 0) -> None:
        if self.board.piece_at(mv.from_coord) is None:
            raise AssertionError(
                f"Illegal move requested: {mv}  (square {mv.from_coord} empty)"
            )
        # 1.  single source of truth: mutate the shared tensor
        self.board.apply_move(mv)
        self._rebuild_occupancy()

        # 2.  push change to every effect cache (board given explicitly)
        for name, cache in self._effect.items():
            if name == "geomancy":
                cache.apply_move(mv, mover, current_ply, self.board)
            elif name in ("archery", "black_hole_suck", "armoured", "freeze",
                        "movement_buff", "movement_debuff", "share_square",
                        "trailblaze", "white_hole_push"):
                cache.apply_move(mv, mover, self.board)
            else:                       # future caches
                cache.apply_move(mv, mover)

        # 3.  move cache (mirror already synced by caller)
        self.move.apply_move(mv, mover)


    def undo_move(self, mv: Move, mover: Color, current_ply: int = 0) -> None:
        """Authoritative board undo + incremental cache refresh."""
        # 1.  manually reverse the move on the shared tensor
        piece = self.board.piece_at(mv.to_coord)
        if piece is not None:
            self.board.set_piece(mv.from_coord, piece)
            self.board.set_piece(mv.to_coord,   None)
            if getattr(mv, "is_capture", False):
                captured_type = getattr(mv, "captured_ptype", None)
                if captured_type is not None:
                    captured_color = piece.color.opposite()
                    self.board.set_piece(
                        mv.to_coord, Piece(captured_color, captured_type)
                    )
            if (getattr(mv, "is_promotion", False) and
                getattr(mv, "promotion_type", None)):
                self.board.set_piece(
                    mv.from_coord, Piece(piece.color, PieceType.PAWN)
                )
        self._rebuild_occupancy()

        # 2.  propagate undo to every effect cache
        for name, cache in self._effect.items():
            if name == "geomancy":
                cache.undo_move(mv, mover, current_ply, self.board)
            elif name in ("archery", "black_hole_suck", "armoured", "freeze",
                        "movement_buff", "movement_debuff", "share_square",
                        "trailblaze", "white_hole_push"):
                cache.undo_move(mv, mover, self.board)
            else:
                cache.undo_move(mv, mover)

        # 3.  move cache (mirror already synced by caller)
        self.move.undo_move(mv, mover)

    @property
    def move(self) -> MoveCache:
        """Guarantee that the move-cache is available."""
        if self._move_cache is None:
            raise RuntimeError(
                "MoveCache not ready – forgot to call cache_manager.initialise() ?"
            )
        return self._move_cache

    def legal_moves(self, color: Color) -> List[Move]:
        return self.move.legal_moves(color)   # now self.move is valid

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

_manager: Optional[CacheManager] = None



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

def init_cache_manager(board: Board, current: Color) -> None:
    global _manager
    if _manager is None:
        _manager = CacheManager(board)
    else:
        _manager.replace_board(board)
    _manager.initialise(current)  # ← pass current color

def get_cache_manager() -> CacheManager:
    if _manager is None:
        raise RuntimeError("CacheManager not initialised")
    return _manager
