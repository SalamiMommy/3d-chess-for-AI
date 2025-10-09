# game3d/cache/manager.py
# ---------------------------------------------------------------------------
#  OptimisedCacheManager – one source of truth for the whole cache stack
# ---------------------------------------------------------------------------
from __future__ import annotations

import gc
import time
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Set, Any, TYPE_CHECKING

import numpy as np

from game3d.common.common import N_TOTAL_PLANES
from game3d.pieces.enums import Color, PieceType
from game3d.pieces.piece import Piece

if TYPE_CHECKING:
    from game3d.board.board import Board
    from game3d.movement.movepiece import Move

# ---------- cache sub-modules ----------
from game3d.cache.caches.movecache import (
    OptimizedMoveCache,
    CompactMove,
    create_optimized_move_cache,
)
from game3d.cache.caches.occupancycache import OccupancyCache
from game3d.cache.caches.transposition import TTEntry          # ← was missing
from game3d.game.zobrist import ZobristHash, compute_zobrist
# ---------- helpers ----------
from .managerconfig import ManagerConfig
from .managerperformance import CachePerformanceMonitor, CacheEventType, MemoryManager
from .effects_cache import EffectsCache
from .parallelmanager import ParallelManager
from .export import (
    export_state_for_ai,
    export_tensor_for_ai,
    get_legal_move_indices,
    get_legal_moves_as_policy_target,
    validate_export_integrity,
)
from .diagnostics import record_cache_creation


class CacheDesyncError(Exception):
    """Raised when cache/board state diverge."""


# ===========================================================================
#  OptimisedCacheManager
# ===========================================================================
class OptimizedCacheManager:
    """
    Central cache manager – thin façade that owns every sub-cache.
    All heavy lifting is delegated to specialised classes.
    """

    __slots__ = (
        "config", "board", "occupancy", "effects", "performance_monitor",
        "_zobrist", "_current_zobrist_hash", "parallel", "_move_cache",
        "_move_counter", "_age_counter", "_current", "_needs_rebuild",
        "_save_thread", "_skip_effect_updates", "_effect_update_counter",
        "_effect_update_interval", "_check_summary_cache", "_check_summary_age",
        "_network_teleport_dirty", "_swap_targets_dirty", "_memory_manager",
        "__weakref__", "_integrated_jump_gen", "_swap_targets",               # swap-move cache
        "_swap_targets_dirty",
        "_network_teleport_targets",   # net-teleport cache
        "_network_teleport_dirty",     # net-teleport invalidation flag
        "_reflecting_bishop_gen"      # reflecting-bishop generator
    )

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #
    def __init__(self, board: Board) -> None:
        self.config = ManagerConfig()
        self.board = board
        self.board.cache_manager = self  # back-ref

        # ----- sub-caches -----
        self.occupancy = OccupancyCache(board)
        self.effects = EffectsCache(board, self)

        # ----- performance & infra -----
        self.performance_monitor = CachePerformanceMonitor()
        self._memory_manager = MemoryManager(self.config, lambda: self._move_cache)
        self.parallel = ParallelManager(self.config)

        # ----- zobrist -----
        self._zobrist = ZobristHash()
        self._current_zobrist_hash = compute_zobrist(board, Color.WHITE)

        # ----- move cache (lazy) -----
        self._move_cache: Optional[OptimizedMoveCache] = None

        # ----- misc -----
        self._move_counter = 0
        self._age_counter = 0
        self._current = Color.WHITE
        self._needs_rebuild = False
        self._save_thread = None
        self._skip_effect_updates = False
        self._effect_update_counter = 0
        self._effect_update_interval = 10
        self._check_summary_cache: Optional[Dict[str, Any]] = None
        self._check_summary_age = -1
        self._network_teleport_dirty = {Color.WHITE: False, Color.BLACK: False}
        self._swap_targets_dirty = {Color.WHITE: False, Color.BLACK: False}
        self._integrated_jump_gen: Optional[IntegratedJumpMovementGenerator] = None
        self._swap_targets = {Color.WHITE: set(), Color.BLACK: set()}
        self._swap_targets_dirty = {Color.WHITE: True, Color.BLACK: True}

        self._network_teleport_targets = {Color.WHITE: set(), Color.BLACK: set()}
        self._network_teleport_dirty = {Color.WHITE: True, Color.BLACK: True}

        self._reflecting_bishop_gen: Optional[ReflectingBishopGenerator] = None
        record_cache_creation(self, board)  # diagnostics

    # ------------------------------------------------------------------ #
    #  Initialisation
    # ------------------------------------------------------------------ #
    def initialise(self, current: Color) -> None:
        from game3d.game.zobrist import compute_zobrist   # ← local import
        self._current_zobrist_hash = compute_zobrist(self.board, current)
        self._move_cache = create_optimized_move_cache(self.board, current, self)
        self._move_cache._full_rebuild()

        if self.config.enable_disk_cache:
            self._move_cache._load_from_disk()

        self._log_cache_stats("initialisation")

    # ------------------------------------------------------------------ #
    #  Move application / undo
    # ------------------------------------------------------------------ #
    def apply_move(self, mv: Move, mover: Color, current_ply: int) -> None:
        """Fast, incremental move application."""
        start = time.perf_counter()
        try:
            self._fast_occupancy_update(mv, self.occupancy.get(mv.from_coord), mover)
            self.board.apply_move(mv)

            captured = self.occupancy.get(mv.to_coord) if mv.is_capture else None
            self._current_zobrist_hash = self._zobrist.update_hash_move(
                self._current_zobrist_hash, mv,
                self.occupancy.get(mv.to_coord), captured, cache=self
            )

            self._current = mover.opposite()
            self._age_counter += 1
            self._needs_rebuild = False

            # batched effect update
            self._effect_update_counter += 1
            if self._effect_update_counter % self._effect_update_interval == 0:
                self._update_effects(mv, self.occupancy.get(mv.to_coord),
                                   captured, mover, current_ply)

            if self._move_cache and not self._needs_rebuild:
                self._move_cache.apply_move(mv, mover)

            self.performance_monitor.record_move_apply_time(
                time.perf_counter() - start)
        except Exception as e:
            self.performance_monitor.record_event(CacheEventType.CACHE_ERROR,
                                                {"error": str(e)})
            raise

    def undo_move(self, mv: Move, mover: Color, current_ply: int = 0) -> None:
        """Undo with proper Zobrist rollback."""
        start = time.time()
        try:
            piece = self.occupancy.get(mv.to_coord)
            captured = None
            if mv.is_capture:
                captured = Piece(mover.opposite(), mv.captured_ptype)

            self._current_zobrist_hash = self._zobrist.update_hash_move(
                self._current_zobrist_hash, mv, piece, captured, cache=self)

            self.board.undo_move(mv)
            self.occupancy.set_position(mv.from_coord,
                                       Piece(piece.color, PieceType.PAWN)
                                       if mv.is_promotion else piece)
            self.occupancy.set_position(mv.to_coord, captured)

            self.effects.update_effect_caches_for_undo(mv, mover,
                self.effects.get_affected_caches_for_undo(mv, mover),
                current_ply)

            if self._move_cache:
                self._move_cache.undo_move(mv, mover)

            self._current = mover
            self._age_counter += 1
            self._mark_network_teleport_dirty()
            self._mark_swap_dirty()

            dur = time.time() - start
            self.performance_monitor.record_move_undo_time(dur)
            self.performance_monitor.record_event(CacheEventType.MOVE_UNDONE,
                                                {"move": str(mv),
                                                 "color": mover.name,
                                                 "duration_ms": dur * 1000})
        except Exception as e:
            self.performance_monitor.record_event(CacheEventType.CACHE_ERROR,
                                                {"error": str(e),
                                                 "move": str(mv),
                                                 "color": mover.name})
            raise

    # ------------------------------------------------------------------ #
    #  Public query API
    # ------------------------------------------------------------------ #
    def legal_moves(self, color: Color) -> List[Move]:
        """Lazy-rebuild + parallel generation."""
        if self._move_cache is None:
            raise RuntimeError("MoveCache not initialised. Call initialise() first.")
        if self._needs_rebuild:
            self._move_cache._full_rebuild()
            self._needs_rebuild = False

        start = time.perf_counter()
        moves = self._move_cache.legal_moves(color,
                                           parallel=self.config.enable_parallel,
                                           max_workers=self.config.max_workers)
        self.performance_monitor.record_legal_move_generation_time(
            time.perf_counter() - start)
        return moves

    # ------------------------------------------------------------------ #
    #  Transposition table façade
    # ------------------------------------------------------------------ #
    def probe_transposition_table(self, hash_value: int) -> Optional[TTEntry]:
        if not self._move_cache:
            return None
        result = self._move_cache.get_cached_evaluation(hash_value)
        if result:
            score, depth, best_move = result
            self.performance_monitor.record_event(CacheEventType.TT_HIT,
                                                {"hash": hash_value,
                                                 "depth": depth,
                                                 "score": score})
            return TTEntry(hash_value, depth, score, 0, best_move, 0)
        self.performance_monitor.record_event(CacheEventType.TT_MISS,
                                            {"hash": hash_value})
        return None

    def store_transposition_table(self, hash_value: int, depth: int, score: int,
                                node_type: int,
                                best_move: Optional[CompactMove] = None) -> None:
        if self._move_cache:
            self._move_cache.store_evaluation(hash_value, depth, score,
                                            node_type, best_move)

    def get_current_zobrist_hash(self) -> int:
        return self._current_zobrist_hash

    # ------------------------------------------------------------------ #
    #  Effect-cache delegation (one-liners)
    # ------------------------------------------------------------------ #
    def is_frozen(self, sq: Tuple[int, int, int], victim: Color) -> bool:
        return self.effects.is_frozen(sq, victim)

    def is_movement_buffed(self, sq: Tuple[int, int, int], friendly: Color) -> bool:
        return self.effects.is_movement_buffed(sq, friendly)

    def is_movement_debuffed(self, sq: Tuple[int, int, int], victim: Color) -> bool:
        return self.effects.is_movement_debuffed(sq, victim)

    def black_hole_pull_map(self, controller: Color) -> Dict[Tuple[int, int, int],
                                                             Tuple[int, int, int]]:
        return self.effects.black_hole_pull_map(controller)

    def white_hole_push_map(self, controller: Color) -> Dict[Tuple[int, int, int],
                                                            Tuple[int, int, int]]:
        return self.effects.white_hole_push_map(controller)

    def mark_trail(self, trailblazer_sq: Tuple[int, int, int],
                 slid_squares: Set[Tuple[int, int, int]]) -> None:
        self.effects.mark_trail(trailblazer_sq, slid_squares)

    def current_trail_squares(self, controller: Color) -> Set[Tuple[int, int, int]]:
        return self.effects.current_trail_squares(controller)

    def is_geomancy_blocked(self, sq: Tuple[int, int, int], current_ply: int) -> bool:
        return self.effects.is_geomancy_blocked(sq, current_ply)

    def block_square(self, sq: Tuple[int, int, int], current_ply: int) -> bool:
        return self.effects.block_square(sq, current_ply)

    def archery_targets(self, controller: Color) -> List[Tuple[int, int, int]]:
        return self.effects.archery_targets(controller)

    def is_valid_archery_attack(self, sq: Tuple[int, int, int],
                               controller: Color) -> bool:
        return self.effects.is_valid_archery_attack(sq, controller)

    def can_capture_wall(self, attacker_sq: Tuple[int, int, int],
                        wall_sq: Tuple[int, int, int], controller: Color) -> bool:
        return self.effects.can_capture_wall(attacker_sq, wall_sq, controller)

    def pieces_at(self, sq: Tuple[int, int, int]) -> List[Piece]:
        return self.effects.pieces_at(sq)

    def top_piece(self, sq: Tuple[int, int, int]) -> Optional[Piece]:
        return self.effects.top_piece(sq)

    def get_attacked_squares(self, color: Color) -> Set[Tuple[int, int, int]]:
        return self._move_cache.get_attacked_squares(color) if self._move_cache else set()

    def store_attacked_squares(self, color: Color,
                             attacked: Set[Tuple[int, int, int]]) -> None:
        self.effects.store_attacked_squares(color, attacked)

    # ------------------------------------------------------------------ #
    #  Configuration
    # ------------------------------------------------------------------ #
    def configure_transposition_table(self, size_mb: int) -> None:
        import psutil
        if psutil.virtual_memory().percent >= 85.0:
            print("[CacheManager] Refused TT expansion: RAM ≥ 85 %")
            return
        self.config.main_tt_size_mb = size_mb
        if self._move_cache:
            cur = self._move_cache._current
            self._move_cache = create_optimized_move_cache(
                self.board, cur, self,
                main_tt_size_mb=size_mb,
                sym_tt_size_mb=self.config.sym_tt_size_mb)

    def configure_symmetry_tt(self, size_mb: int) -> None:
        self.config.sym_tt_size_mb = size_mb
        if self._move_cache:
            cur = self._move_cache._current
            self._move_cache = create_optimized_move_cache(
                self.board, cur, self,
                main_tt_size_mb=self.config.main_tt_size_mb,
                sym_tt_size_mb=size_mb)

    def set_parallel_processing(self, enabled: bool) -> None:
        self.config.enable_parallel = enabled

    def set_vectorization(self, enabled: bool) -> None:
        self.config.enable_vectorization = enabled

    # ------------------------------------------------------------------ #
    #  Utility
    # ------------------------------------------------------------------ #
    def clear_all_caches(self) -> None:
        if self._move_cache:
            self._move_cache.clear()
        self.effects.clear_all_effects()
        self.occupancy.rebuild(self.board)
        self.performance_monitor = CachePerformanceMonitor()
        self._move_counter = 0
        gc.collect()
        self.parallel.shutdown()

    def export_cache_state(self) -> Dict[str, Any]:
        return {
            "zobrist_hash": self._current_zobrist_hash,
            "performance_stats": self.get_cache_stats(),
            "board_state": {
                "occupied_squares": len(list(self.board.list_occupied())),
                "current_player": self._move_cache._current.name if self._move_cache else None,
            },
            "effect_cache_status": {
                name: {"type": type(cache).__name__}
                for name, cache in self.effects._effect_caches.items()
            },
        }

    def validate_cache_consistency(self) -> bool:
        for coord, piece in self.board.list_occupied():
            if self.occupancy.get(coord) != piece:
                print(f"[INCONSISTENCY] Board has {piece} at {coord}, cache has {self.occupancy.get(coord)}")
                return False
        return True

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #
    def _fast_occupancy_update(self, mv: Move, from_piece: Optional[Piece], mover: Color) -> None:
        if from_piece is None:
            return
        promotion_type = getattr(mv, "promotion_ptype", None)
        to_piece = (Piece(mover, PieceType(promotion_type)) if promotion_type else from_piece)
        self.occupancy.set_position(mv.from_coord, None)
        self.occupancy.set_position(mv.to_coord, to_piece)

    def _update_effects(self, mv: Move, from_piece: Optional[Piece],
                       captured_piece: Optional[Piece], mover: Color,
                       current_ply: int) -> None:
        relevant = {
            PieceType.FREEZER: "freeze",
            PieceType.SPEEDER: "movement_buff",
            PieceType.SLOWER: "movement_debuff",
            PieceType.BLACKHOLE: "black_hole_suck",
            PieceType.WHITEHOLE: "white_hole_push",
            PieceType.TRAILBLAZER: "trailblaze",
            PieceType.WALL: "behind",
            PieceType.ARMOUR: "armour",
            PieceType.GEOMANCER: "geomancy",
            PieceType.ARCHER: "archery",
            PieceType.KNIGHT: "share_square",
        }.get(getattr(from_piece, "ptype", None))
        if relevant:
            try:
                cache = self.effects._effect_caches[relevant]
                if hasattr(cache, "apply_move"):
                    cache.apply_move(mv, mover, self.board)
            except Exception as e:
                print(f"Effect {relevant} update failed: {e}")

    def _mark_network_teleport_dirty(self) -> None:
        self._network_teleport_dirty[Color.WHITE] = True
        self._network_teleport_dirty[Color.BLACK] = True

    def _mark_swap_dirty(self) -> None:
        self._swap_targets_dirty[Color.WHITE] = True
        self._swap_targets_dirty[Color.BLACK] = True

    # ---------- check summary ----------
    def get_check_summary(self) -> Dict[str, Any]:
        if self._check_summary_age != self._age_counter:
            self._check_summary_cache = self._recompute_check_summary()
            self._check_summary_age = self._age_counter
        return self._check_summary_cache  # type: ignore

    def _recompute_check_summary(self) -> Dict[str, Any]:
        def king_pos(color: Color) -> Optional[Tuple[int, int, int]]:
            for sq, p in self.board.list_occupied():
                if p.color == color and p.ptype is PieceType.KING:
                    return sq
            return None

        def has_priest(color: Color) -> bool:
            return any(
                p.ptype is PieceType.PRIEST
                for _, p in self.board.list_occupied()
                if p.color == color
            )

        w_k, b_k = king_pos(Color.WHITE), king_pos(Color.BLACK)
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
        }

    # ---------- stats ----------
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
            name: {"type": type(cache).__name__}
            for name, cache in self.effects._effect_caches.items()
        }
        return base

    def get_optimization_suggestions(self) -> List[str]:
        return self.performance_monitor.get_optimization_suggestions()

    def _log_cache_stats(self, context: str) -> None:
        stats = self.get_cache_stats()
        suggestions = self.get_optimization_suggestions()
        if suggestions:
            print(f"  Optimisation hints ({len(suggestions)}):")
            for s in suggestions[:3]:
                print(f"    - {s}")

    # ---------- property shims for backward compatibility ----------
    @property
    def cache(self):
        """Legacy shim: some code expects game.cache.piece_cache"""
        return self

    @property
    def piece_cache(self):
        """Legacy shim: old name for occupancy cache"""
        return self.occupancy

    @property
    def move(self) -> OptimizedMoveCache:
        if self._move_cache is None:
            raise RuntimeError("MoveCache not initialised. Call initialise() first.")
        return self._move_cache


# =============================================================================
#  Factory
# =============================================================================
def get_cache_manager(board: Board, current: Color) -> OptimizedCacheManager:
    cm = OptimizedCacheManager(board)
    cm.initialise(current)
    return cm


# =============================================================================
#  Legacy aliases (zero-cost)
# =============================================================================
CacheManager = OptimizedCacheManager
