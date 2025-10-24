# manager.py (updated with common modules)
from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Set, Any, TYPE_CHECKING, Iterator, Callable, Iterable

import numpy as np

from game3d.common.constants import N_TOTAL_PLANES, SIZE
from game3d.common.enums import Color, PieceType
from game3d.pieces.piece import Piece
from game3d.common.coord_utils import Coord, in_bounds
from game3d.common.piece_utils import get_piece_effect_type, iterate_occupied, find_king
from game3d.common.move_utils import filter_none_moves
from game3d.common.debug_utils import CacheStatsMixin

if TYPE_CHECKING:
    from game3d.board.board import Board
    from game3d.movement.movepiece import Move

from game3d.cache.caches.movecache import (
    OptimizedMoveCache,
    CompactMove,
    create_optimized_move_cache,
)
from game3d.cache.caches.occupancycache import OccupancyCache
from game3d.cache.caches.transposition import TTEntry
from game3d.cache.caches.zobrist import ZobristHash, compute_zobrist, _PIECE_KEYS, _SIDE_KEY
from .managerconfig import ManagerConfig
from .managerperformance import CachePerformanceMonitor, CacheEventType, MemoryManager
from .parallelmanager import ParallelManager
from .export import (
    export_state_for_ai,
    export_tensor_for_ai,
    get_legal_move_indices,
    get_legal_moves_as_policy_target,
    validate_export_integrity,
)
from .diagnostics import record_cache_creation
from game3d.cache.effectscache.auracache import UnifiedAuraCache
from game3d.cache.effectscache.trailblazecache import TrailblazeCache
from game3d.cache.effectscache.geomancycache import GeomancyCache
from game3d.cache.caches.attackscache import AttacksCache
from game3d.cache.cache_protocols import CacheManagerProtocol, MovementCacheProtocol, EffectCacheProtocol


class CacheDesyncError(Exception):
    pass


class OptimizedCacheManager(CacheManagerProtocol, CacheStatsMixin):
    __slots__ = (
        "config", "board", "occupancy", "performance_monitor",
        "_zobrist", "_current_zobrist_hash", "parallel", "_move_cache",
        "_move_counter", "_age_counter", "_current", "_needs_rebuild",
        "_skip_effect_updates", "_effect_update_counter",
        "_effect_update_interval", "_check_summary_cache", "_check_summary_age",
        "_network_teleport_dirty", "_swap_targets_dirty", "_memory_manager",
        "__weakref__", "_integrated_jump_gen", "_swap_targets",
        "_swap_targets_dirty",
        "_network_teleport_targets",
        "_network_teleport_dirty",
        "_reflecting_bishop_gen", "_board",
        "aura_cache", "trailblaze_cache", "geomancy_cache", "attacks_cache"
    )

    def __init__(self, board: Board, current: Color = Color.WHITE) -> None:
        self.config = ManagerConfig()
        self.board = board
        self._current = current
        self.occupancy = OccupancyCache(board)
        self.performance_monitor = CachePerformanceMonitor()
        self._memory_manager = MemoryManager(self.config, lambda: self._move_cache)
        self.parallel = ParallelManager(self.config)
        self._zobrist = ZobristHash()

        self._current_zobrist_hash = 0

        self._move_cache: Optional[OptimizedMoveCache] = None
        self._move_counter = 0
        self._age_counter = 0

        self._skip_effect_updates = False
        self._effect_update_counter = 0
        self._effect_update_interval = 10
        self._check_summary_cache: Optional[Dict[str, Any]] = None
        self._check_summary_age = -1

        self._integrated_jump_gen: Optional[Any] = None
        self._swap_targets = {Color.WHITE: set(), Color.BLACK: set()}

        self._network_teleport_targets = {Color.WHITE: set(), Color.BLACK: set()}

        self._integrated_jump_gen: Optional[IntegratedJumpMovementGenerator] = None
        self._reflecting_bishop_gen: Optional[_ReflectingBishopGen] = None
        record_cache_creation(self, board)
        if board is not None:
            board.cache_manager = self
        self._initialize_effect_caches()

        _dirty_flags: Dict[str, bool] = {
            'moves': False,
            'attacks': False,
            'effects': False,
            'network_teleport': {Color.WHITE: False, Color.BLACK: False},
            'swap_targets': {Color.WHITE: False, Color.BLACK: False}
        }

    def initialise(self, current: Color) -> None:
        self._current_zobrist_hash = compute_zobrist(self.board, current)
        self._move_cache = create_optimized_move_cache(self.board, current, self)

    def _initialize_effect_caches(self) -> None:
        self.aura_cache = UnifiedAuraCache(self.board, self)
        self.trailblaze_cache = TrailblazeCache(self)
        self.geomancy_cache = GeomancyCache(self)
        self.attacks_cache = AttacksCache(self.board)

        self.aura_cache._cache_manager = self
        self.trailblaze_cache._cache_manager = self
        self.geomancy_cache._cache_manager = self
        self.attacks_cache._manager = self

    def apply_move(self, mv: Move, mover: Color) -> bool:
        board_gen = getattr(self.board, 'generation', 0)
        if (hasattr(self._move_cache, '_gen') and
            self._move_cache._gen != board_gen):
            self._move_cache._needs_rebuild = True
        from_piece = self.occupancy.get(mv.from_coord)
        captured_piece = self.occupancy.get(mv.to_coord) if getattr(mv, 'is_capture', False) else None

        if from_piece is None:
            return False

        success = self.board.apply_move(mv)
        if not success:
            return False

        new_hash = self._current_zobrist_hash
        new_hash ^= _PIECE_KEYS[(from_piece.ptype, from_piece.color, mv.from_coord)]
        if captured_piece is not None:
            new_hash ^= _PIECE_KEYS[(captured_piece.ptype, captured_piece.color, mv.to_coord)]
        add_ptype = getattr(mv, 'promotion_type', from_piece.ptype) if getattr(mv, 'is_promotion', False) else from_piece.ptype
        new_hash ^= _PIECE_KEYS[(add_ptype, from_piece.color, mv.to_coord)]
        new_hash ^= _SIDE_KEY
        self.sync_zobrist(new_hash)

        self.aura_cache.apply_move(mv, mover, self.board)
        self.trailblaze_cache.apply_move(mv, mover, self.halfmove_clock, self.board)
        self.geomancy_cache.apply_move(mv, mover, self.halfmove_clock, self.board)
        self.attacks_cache.apply_move(mv, mover, self.board)
        self._move_cache.apply_move(mv, mover)

        self._move_counter += 1
        self._effect_update_counter += 1
        if self._effect_update_counter >= self._effect_update_interval:
            self._effect_update_counter = 0

        self._needs_rebuild = False

        if hasattr(self._move_cache, '_gen'):
            self._move_cache._gen = getattr(self.board, 'generation', board_gen + 1)

        return True

    def undo_move(self, mv: Move, mover: Color) -> bool:
        original_tensor = self.board.tensor().clone()

        try:
            moving_piece = self.occupancy.get(mv.to_coord)
            if moving_piece is None:
                return False

            self.board.set_piece(mv.from_coord, moving_piece)
            self.board.set_piece(mv.to_coord, None)

            if getattr(mv, 'is_capture', False) and hasattr(mv, 'captured_piece'):
                self.board.set_piece(mv.to_coord, mv.captured_piece)

            self._move_cache.undo_move(mv, mover)
            self.attacks_cache.undo_move(mv, mover, self.board)
            self.geomancy_cache.undo_move(mv, mover, self.halfmove_clock - 1, self.board)
            self.trailblaze_cache.undo_move(mv, mover, self.halfmove_clock - 1, self.board)
            self.aura_cache.undo_move(mv, mover, self.board)

            self.sync_zobrist(compute_zobrist(self.board, mover))

            self._move_counter -= 1
            self._effect_update_counter -= 1

            return True

        except Exception as e:
            self.board._tensor = original_tensor
            print(f"[ERROR] Undo move failed: {e}")
            return False

    def _get_cache_by_name(self, name: str):
        if name == "aura":
            return self.aura_cache
        elif name == "trailblaze":
            return self.trailblaze_cache
        elif name == "geomancy":
            return self.geomancy_cache
        elif name == "attacks":
            return self.attacks_cache
        elif name == "occupancy":
            return self.occupancy_cache
        elif name == "moves":
            return self.move_cache
        return None

    def get_affected_caches(
        self,
        mv: 'Move',
        mover: Color,
        from_piece: Optional[Piece],
        to_piece: Optional[Piece],
        captured_piece: Optional[Piece],
        is_undo: bool = False
    ) -> Set[str]:
        affected = set()

        effect_map = {
            PieceType.FREEZER:   "aura",
            PieceType.SPEEDER:   "aura",
            PieceType.SLOWER:    "aura",
            PieceType.BLACKHOLE: "aura",
            PieceType.WHITEHOLE: "aura",
            PieceType.TRAILBLAZER: "trailblaze",
            PieceType.GEOMANCER:   "geomancy",
        }

        if from_piece and from_piece.ptype in effect_map:
            affected.add(effect_map[from_piece.ptype])

        if captured_piece and captured_piece.ptype in effect_map:
            affected.add(effect_map[captured_piece.ptype])

        self._add_affected_effects_from_pos(mv.from_coord, affected)
        self._add_affected_effects_from_pos(mv.to_coord, affected)
        if is_undo:
            self._add_affected_effects_from_pos(mv.to_coord, affected)

        return affected

    def get_affected_caches_for_undo(self, mv: 'Move', mover: Color) -> Set[str]:
        return self.get_affected_caches(mv, mover, None, None, None, is_undo=True)

    def _add_affected_effects_from_pos(self, pos: Coord, affected: Set[str]) -> None:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    check_pos = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
                    if all(0 <= c < SIZE for c in check_pos):
                        piece = self.occupancy.get(check_pos)
                        if piece and piece.ptype in {PieceType.FREEZER, PieceType.BLACKHOLE,
                                                     PieceType.WHITEHOLE, PieceType.GEOMANCER}:
                            effect_map = {
                                PieceType.FREEZER: "aura",
                                PieceType.BLACKHOLE: "aura",
                                PieceType.WHITEHOLE: "aura",
                                PieceType.GEOMANCER: "geomancy"
                            }
                            affected.add(effect_map[piece.ptype])

    def update_effect_caches(
            self,
            mv: "Move",
            mover: Color,
            affected_caches: set[str],
            current_ply: int,
    ) -> None:
        for cache_name in affected_caches:
            cache = self._get_cache_by_name(cache_name)
            if cache and hasattr(cache, 'apply_move'):
                cache.apply_move(mv, mover, current_ply, self.board)

    def is_movement_buffed(self, sq: Coord, color: Color) -> bool:
        return self.aura_cache.is_buffed(sq, color)

    def is_movement_debuffed(self, sq: Coord, color: Color) -> bool:
        return self.aura_cache.is_debuffed(sq, color)

    def is_frozen(self, sq: Coord, color: Color) -> bool:
        return self.aura_cache.is_frozen(sq, color)

    def black_hole_pull_map(self, controller: Color) -> Dict[Coord, Coord]:
        return self.aura_cache.pull_map(controller)

    def white_hole_push_map(self, controller: Color) -> Dict[Coord, Coord]:
        return self.aura_cache.push_map(controller)

    def mark_trail(self, trailblazer_sq: Coord, slid_squares: Set[Coord]) -> None:
        self.trailblaze_cache.mark_trail(trailblazer_sq, slid_squares)

    def current_trail_squares(self, controller: Color) -> Set[Coord]:
        return self.trailblaze_cache.current_trail_squares(controller, self.board)

    def is_geomancy_blocked(self, sq: Coord, current_ply: int) -> bool:
        return self.geomancy_cache.is_blocked(sq, current_ply)

    def block_square(self, sq: Coord, current_ply: int) -> bool:
        return self.geomancy_cache.block_square(sq, current_ply, self.board)

    def get_attacked_squares(self, color: Color) -> Set[Coord]:
        cached = self.attacks_cache.get_for_color(color)
        return cached if cached is not None else set()

    def store_attacked_squares(self, color: Color, attacked: Set[Coord]) -> None:
        self.attacks_cache.store_for_color(color, attacked)

    def clear_all_effects(self) -> None:
        for cache in [self.aura_cache, self.trailblaze_cache, self.geomancy_cache, self.attacks_cache]:
            if hasattr(cache, 'clear'):
                cache.clear()
            elif hasattr(cache, 'invalidate'):
                cache.invalidate()

    def apply_blackhole_pulls(self, controller: Color) -> None:
        dirty_squares = self.aura_cache.apply_pull_effects(controller, self.board)
        if dirty_squares:
            self.occupancy.batch_set_positions(
                [(sq, self.occupancy.get(sq)) for sq in dirty_squares]
            )

        self.aura_cache.apply_move(None, controller, self.halfmove_clock, self.board)

        self._move_cache.invalidate_attacked_squares(controller)
        self._move_cache.invalidate_attacked_squares(controller.opposite())
        self._move_cache._lazy_revalidate()

    def apply_whitehole_pushes(self, controller: Color) -> None:
        dirty_squares = self.aura_cache.apply_push_effects(controller, self.board)
        if dirty_squares:
            self.occupancy.batch_set_positions(
                [(sq, self.occupancy.get(sq)) for sq in dirty_squares]
            )

        self.aura_cache.apply_move(None, controller, self.halfmove_clock, self.board)

        self._move_cache.invalidate_attacked_squares(controller)
        self._move_cache.invalidate_attacked_squares(controller.opposite())
        self._move_cache._lazy_revalidate()

    def apply_freeze_effects(self, controller: Color) -> None:
        frozen_squares = self.aura_cache.apply_freeze_effects(controller, self.board)
        if frozen_squares:
            self.occupancy.batch_set_positions(
                [(sq, self.occupancy.get(sq)) for sq in frozen_squares]
            )
        self.aura_cache.apply_move(None, controller, self.halfmove_clock, self.board)

    def get_check_summary(self, has_priest: Callable[[Color], bool]) -> Dict[str, Any]:
        w_k = find_king(self, Color.WHITE)
        b_k = find_king(self, Color.BLACK)
        w_at = self.get_attacked_squares(Color.WHITE)
        b_at = self.get_attacked_squares(Color.BLACK)

        w_checkers = [] if has_priest(Color.WHITE) else [sq for sq in b_at if sq == w_k] if w_k else []
        b_checkers = [] if has_priest(Color.BLACK) else [sq for sq in w_at if sq == b_k] if b_k else []

        return {
            "white_check": bool(w_checkers),
            "black_check": bool(b_checkers),
            "white_king_position": w_k,
            "black_king_position": b_k,
            "white_checkers": w_checkers,
            "black_checkers": b_checkers,
            "attacked_squares_white": w_at,
            "attacked_squares_black": b_at,
            "white_priests_alive": self.has_priest(Color.WHITE),
            "black_priests_alive": self.has_priest(Color.BLACK),
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        base = self.performance_monitor.get_performance_stats()
        if self._move_cache:
            base.update({
                "move_cache_stats": self._move_cache.get_stats(),
                "zobrist_hash": self._current_zobrist_hash,
                "main_tt_size_mb": self.config.main_tt_size_mb,
                "sym_tt_size_mb": self.config.sym_tt_size_mb,
                "main_tt_capacity_estimate": (self.config.main_tt_size_mb * 1024 * 1024) // 32,
                "sym_tt_capacity_estimate": (self.config.sym_tt_size_mb * 1024 * 1024) // 32,
                "enable_parallel": self.config.enable_parallel,
                "max_workers": self.config.max_workers,
                "enable_vectorization": self.config.enable_vectorization,
            })
        base["effect_caches"] = {
            "aura": {"type": type(self.aura_cache).__name__},
            "trailblaze": {"type": type(self.trailblaze_cache).__name__},
            "geomancy": {"type": type(self.geomancy_cache).__name__},
            "attacks": {"type": type(self.attacks_cache).__name__},
        }
        return base

    def get_optimization_suggestions(self) -> List[str]:
        return self.performance_monitor.get_optimization_suggestions()

    def _log_cache_stats(self, context: str) -> None:
        stats = self.get_cache_stats()
        suggestions = self.get_optimization_suggestions()

    @property
    def move(self) -> OptimizedMoveCache:
        if self._move_cache is None:
            raise RuntimeError("MoveCache not initialised. Call initialise() first.")
        return self._move_cache

    def has_priest(self, color: Color) -> bool:
        return self.occupancy.has_priest(color)

    def any_priest_alive(self) -> bool:
        return self.occupancy.any_priest_alive()

    def priest_status(self) -> Dict[str, bool]:
        return {
            "white_priests_alive": self.has_priest(Color.WHITE),
            "black_priests_alive": self.has_priest(Color.BLACK),
        }

    def rebuild(self, board: Board, color: Color) -> None:
        if self.board is board and self._current == color and not self._needs_rebuild:
            return
        self.board = board
        self._current = color
        if self.board is not board:
            self.occupancy.rebuild(board)
        self._current_zobrist_hash = compute_zobrist(board, color)
        self._needs_rebuild = False

    def sync_zobrist(self, new_hash: int) -> None:
        self._current_zobrist_hash = new_hash

    def is_movement_blocked_for_hive(self, sq: Tuple[int, int, int], color: Color) -> bool:
        ply = getattr(self.board, "halfmove_clock", 0)
        return (
            self.is_frozen(sq, color) or
            self.is_movement_debuffed(sq, color) or
            self.is_geomancy_blocked(sq, ply)
        )

    @property
    def halfmove_clock(self) -> int:
        return getattr(self.board, 'halfmove_clock', 0)

    def _attach_board(self, b: Board) -> None:
        self.board = b
        b.cache_manager = self

    def update_occupancy_incrementally(
        self,
        board: "Board",
        moved_sq: Coord,
        captured_sq: Optional[Coord] = None
    ) -> None:
        updates = []

        updates.append((moved_sq, None))

        if captured_sq is not None:
            updates.append((captured_sq, None))

        to_piece = self.occupancy.get(moved_sq)
        if to_piece:
            updates.append((moved_sq, to_piece))

        self.batch_set_pieces(updates)

    def _fast_occupancy_update(self, mv: Move, piece: Piece, mover: Color) -> None:
        captured_sq = mv.to_coord if mv.is_capture else None
        self.occupancy.set_position(mv.from_coord, None)
        if captured_sq:
            self.occupancy.set_position(captured_sq, None)
        self.occupancy.set_position(mv.to_coord, piece)

    def get_piece(self, coord: Coord) -> Optional[Piece]:
        return self.occupancy.get(coord)

    def set_piece(self, coord: Coord, piece: Optional[Piece]) -> None:
        self.occupancy.set_position(coord, piece)

    def get_pieces_of_color(self, color: Color) -> Iterable[Tuple[Coord, Piece]]:
        return self.occupancy.iter_color(color)

    def find_king(self, color: Color) -> Optional[Coord]:
        return self.occupancy.find_king(color)

    def get_occupancy_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.occupancy._occ.copy(), self.occupancy._ptype.copy()

    def batch_set_pieces(self, updates: List[Tuple[Coord, Optional[Piece]]]) -> None:
        self.occupancy.batch_set_positions(updates)

    def get_occupancy_array(self) -> np.ndarray:
        return self.occupancy._occ.copy()

    def get_occupancy_array_readonly(self) -> np.ndarray:
        return self.occupancy._occ

    def get_piece_type_array(self) -> np.ndarray:
        return self.occupancy._ptype.copy()

    @property
    def occupancy_cache(self):
        return self.occupancy

    @property
    def move_cache(self):
        return self._move_cache

def get_cache_manager(board: Board, current: Color) -> OptimizedCacheManager:
    if board is None:
        raise ValueError("get_cache_manager requires a real Board instance")
    cm = OptimizedCacheManager(board, current)
    cm.initialise(current)
    board.cache_manager = cm
    return cm

CacheManager = OptimizedCacheManager
