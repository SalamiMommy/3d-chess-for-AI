# manager.py
# game3d/cache/manager.py
# ---------------------------------------------------------------------------
#  OptimisedCacheManager – one source of truth for the whole cache stack
# ---------------------------------------------------------------------------
from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Set, Any, TYPE_CHECKING

import numpy as np

from game3d.common.common import N_TOTAL_PLANES
from game3d.common.enums import Color, PieceType
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
        "_skip_effect_updates", "_effect_update_counter",
        "_effect_update_interval", "_check_summary_cache", "_check_summary_age",
        "_network_teleport_dirty", "_swap_targets_dirty", "_memory_manager",
        "__weakref__", "_integrated_jump_gen", "_swap_targets",               # swap-move cache
        "_swap_targets_dirty",
        "_network_teleport_targets",   # net-teleport cache
        "_network_teleport_dirty",     # net-teleport invalidation flag
        "_reflecting_bishop_gen", "_board"      # reflecting-bishop generator
    )

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #
    def __init__(self, board: Board, current: Color = Color.WHITE) -> None:
        self.config = ManagerConfig()
        self.board = board
        self._current = current
        self.occupancy = OccupancyCache(board)
        self.effects = EffectsCache(board, self)
        self.performance_monitor = CachePerformanceMonitor()
        self._memory_manager = MemoryManager(self.config, lambda: self._move_cache)
        self.parallel = ParallelManager(self.config)
        self._zobrist = ZobristHash()
        self._current_zobrist_hash = compute_zobrist(board, current)  # Fixed: use current
        self._move_cache: Optional[OptimizedMoveCache] = None
        self._move_counter = 0
        self._age_counter = 0
        self._needs_rebuild = False
        self._skip_effect_updates = False
        self._effect_update_counter = 0
        self._effect_update_interval = 10
        self._check_summary_cache: Optional[Dict[str, Any]] = None
        self._check_summary_age = -1
        self._network_teleport_dirty = {Color.WHITE: False, Color.BLACK: False}
        self._swap_targets_dirty = {Color.WHITE: False, Color.BLACK: False}
        self._integrated_jump_gen: Optional[Any] = None  # Assume type from context
        self._swap_targets = {Color.WHITE: set(), Color.BLACK: set()}
        self._swap_targets_dirty = {Color.WHITE: True, Color.BLACK: True}
        self._network_teleport_targets = {Color.WHITE: set(), Color.BLACK: set()}
        self._network_teleport_dirty = {Color.WHITE: True, Color.BLACK: True}
        self._reflecting_bishop_gen: Optional[Any] = None
        record_cache_creation(self, board)
        self._board = board
        if board is not None:
            board.cache_manager = self
    # ------------------------------------------------------------------ #
    #  Initialisation
    # ------------------------------------------------------------------ #
    def initialise(self, current: Color) -> None:
        self._current_zobrist_hash = compute_zobrist(self.board, current)  # Avoid redundant compute if same
        self._move_cache = create_optimized_move_cache(self.board, current, self)
    # ------------------------------------------------------------------ #
    #  Move application / undo
    # ------------------------------------------------------------------ #
    def apply_move(self, mv: Move, *args, **kwargs) -> bool:
        """Apply the move *and* resolve friendly auras atomically."""
        from_coord = mv.from_coord
        to_coord = mv.to_coord

        mover = args[0] if args else self._current
        piece = self.occupancy.get(from_coord)
        if piece is None:
            raise CacheDesyncError(f"apply_move: empty from-square {from_coord}")

        start = time.perf_counter()
        affected: set[Tuple[int, int, int]] = {from_coord, to_coord}

        try:
            # 1️⃣ Standard move ------------------------------------------------
            self._fast_occupancy_update(mv, piece, mover)
            self.board.apply_move(mv)

            captured = self.occupancy.get(to_coord) if mv.is_capture else None
            self._current_zobrist_hash = self._zobrist.update_hash_move(
                self._current_zobrist_hash, mv, piece, captured, cache=self
            )

            # 2️⃣ === ATOMIC AURA PHASE ====================================
            # 2-a Freeze - ANY move triggers freeze re-emission
            freeze_cache = self.effects._effect_caches["freeze"]
            freeze_cache.apply_freeze_effects(mover, self.board)
            frozen_squares = freeze_cache.get_frozen_squares(mover.opposite())
            affected.update(frozen_squares)

            # 2-b Black-hole suck
            bh_cache = self.effects._effect_caches["black_hole_suck"]
            for fr, to in bh_cache.pull_map(mover).items():
                victim = self.occupancy.get(fr)
                if victim and victim.color != mover:
                    self.board.move_piece(fr, to)           # physical
                    self.occupancy.set_position(fr, None)   # mirror
                    self.occupancy.set_position(to, victim)
                    affected.update((fr, to))

            # 2-c White-hole push
            wh_cache = self.effects._effect_caches["white_hole_push"]
            for fr, to in wh_cache.push_map(mover).items():
                victim = self.occupancy.get(fr)
                if victim and victim.color != mover:
                    self.board.move_piece(fr, to)
                    self.occupancy.set_position(fr, None)
                    self.occupancy.set_position(to, victim)
                    affected.update((fr, to))


            self.performance_monitor.record_event(CacheEventType.MOVE_APPLIED)
            self._move_counter += 1
            self.effects.apply_freeze_effects(self._current, self.board)
            # 3️⃣ Incremental cache invalidation ------------------------------
            self._current = mover.opposite()
            self._age_counter += 1
            self._needs_rebuild = False

            if self._move_cache:
                self._move_cache.invalidate_squares(affected)
                self._move_cache.invalidate_attacked_squares(mover)
                self._move_cache.invalidate_attacked_squares(mover.opposite())

            self.performance_monitor.record_move_apply_time(
                time.perf_counter() - start)
            return True

        except Exception as e:
            self.performance_monitor.record_event(CacheEventType.CACHE_ERROR,
                                                {"error": str(e)})
            raise

    def undo_move(self, mv: Move, color: Color, halfmove_delta: int, undo_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Incremental undo for cache state, reversing the effects of a move.
        Relies on stored undo_info from make_move for precise reversals.
        If undo_info is None, falls back to full rebuild (less efficient).

        Args:
            mv: The Move being undone.
            color: The color that made the move (pre-undo current player).
            halfmove_delta: Change in halfmove clock (usually -1 or 0).
            undo_info: Dict with reversal data (e.g., 'occupancy_updates', 'removed_pieces', etc.).
        """
        if undo_info is None:
            # Fallback: Full rebuild (expensive, but safe)
            self.rebuild(self.board, color)
            self._current = color
            self.sync_zobrist(compute_zobrist(self.board, color))
            return

        # 1. Reverse occupancy updates (batch reverse)
        if 'occupancy_updates' in undo_info:
            # Reverse the list to undo in opposite order
            reverse_updates = list(reversed(undo_info['occupancy_updates']))
            self.occupancy.batch_set_positions(reverse_updates)
        else:
            # Minimal reverse: Swap back from/to, restore captured
            captured_piece = undo_info.get('captured_piece')
            reverse_updates = [
                (mv.to_coord, None),  # Clear to
                (mv.from_coord, undo_info['moving_piece']),  # Restore from
            ]
            if captured_piece:
                reverse_updates.append((mv.to_coord, captured_piece))  # Restore captured if any
            self.occupancy.batch_set_positions(reverse_updates)

        # 2. Reverse effects (decrement counters, clear flags)
        # Assuming EffectsCache has a method to revert per-move effects
        self.effects.clear_effects_for_move(mv, undo_info)
        # Specific reversals:
        # - Restore removed_pieces from bombs/trailblaze/holes
        if 'removed_pieces' in undo_info:
            restore_updates = [(sq, piece) for sq, piece in undo_info['removed_pieces']]
            self.occupancy.batch_set_positions(restore_updates)
        # - Reverse moved_pieces (holes)
        if 'moved_pieces' in undo_info:
            reverse_moves = [(to_sq, from_sq, piece) for from_sq, to_sq, piece in reversed(undo_info['moved_pieces'])]
            move_updates = []
            for from_sq, to_sq, piece in reverse_moves:
                move_updates.append((to_sq, None))
                move_updates.append((from_sq, piece))
            self.occupancy.batch_set_positions(move_updates)

        # 3. Undo move cache incrementally
        self.move._optimized_incremental_undo(mv, color)

        # 4. Sync Zobrist from stored or recompute
        if 'original_zkey' in undo_info:
            original_hash = undo_info['original_zkey']
        else:
            original_hash = compute_zobrist(self.board, color)
        self._current_zobrist_hash = original_hash
        self.sync_zobrist(original_hash)

        # 5. Minimal invalidation (affected squares from undo_info)
        affected = {mv.from_coord, mv.to_coord}
        if 'affected_squares' in undo_info:
            affected.update(undo_info['affected_squares'])
        self.move.invalidate_squares(affected)
        self.move.invalidate_attacked_squares(color)
        self.move.invalidate_attacked_squares(color.opposite())

        # 6. Update internal state
        self._current = color
        # Sync generation counters
        if hasattr(self.board, 'generation'):
            self.occupancy._gen = self.board.generation
            self.move._gen = self.board.generation

        # Diagnostics
        self.performance_monitor.record_event(CacheEventType.UNDO_MOVE, len(affected) if 'affected_squares' in undo_info else 2)

    def _undo_white_hole_push(self, mover: Color) -> None:
        cache = self.effects._effect_caches["white_hole_push"]
        if hasattr(cache, "_undo_stack"):
            cache._restore_undo_snapshot(mover, self.board)

    def _undo_black_hole_suck(self, mover: Color) -> None:
        cache = self.effects._effect_caches["black_hole_suck"]
        if hasattr(cache, "_undo_stack"):
            cache._restore_undo_snapshot(mover, self.board)

    def _undo_freeze(self, mover: Color) -> None:
        # freeze is state-less; nothing to roll back
        pass
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

    def get_check_summary(self) -> Dict[str, Any]:
        if self._check_summary_age == self._age_counter:
            return self._check_summary_cache
        # Compute summary
        summary = {
            "white_priests_alive": self.has_priest(Color.WHITE),
            "black_priests_alive": self.has_priest(Color.BLACK),
        }
        self._check_summary_cache = summary
        self._check_summary_age = self._age_counter
        return summary

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
            # >>>>>>  ADD THESE TWO LINES  <<<<<<
            "white_priests_alive": self.has_priest(Color.WHITE),
            "black_priests_alive": self.has_priest(Color.BLACK),
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

    def has_priest(self, color: Color) -> bool:
        return self.occupancy.has_priest(color)

    def any_priest_alive(self) -> bool:
        return self.occupancy.any_priest_alive()

    def priest_status(self) -> Dict[str, bool]:
        """Return both priest-alive flags in one call."""
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

    # ADDED: New method for incremental Zobrist sync (called in make_move)
    def sync_zobrist(self, new_hash: int) -> None:
        self._current_zobrist_hash = new_hash

    def is_movement_blocked_for_hive(self, sq: Tuple[int, int, int], color: Color) -> bool:
        # ------------------------------------------------------------------
        # geomancy-block check needs “current ply”; use 0 when unknown
        # ------------------------------------------------------------------
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
        occupancy: OccupancyCache,
        board: "Board",
        moved_sq: Tuple[int, int, int],
        captured_sq: Optional[Tuple[int, int, int]] = None
    ) -> None:
        """
        Mirror the board state into the occupancy cache without a full rebuild.
        """
        # 1. Old piece left the from-square
        occupancy.set_position(moved_sq, None)

        # 2. Captured piece disappeared (if any)
        if captured_sq is not None:
            occupancy.set_position(captured_sq, None)

        # 3. Moved piece arrived at the to-square
        to_piece = board.get(moved_sq)  # board already reflects the move (fallback if needed, but prefer occupancy)
        occupancy.set_position(moved_sq, to_piece)

    def get_piece(self, coord: Coord) -> Optional[Piece]:
        """Get piece at coordinate."""
        return self.occupancy.get(coord)

    def set_piece(self, coord: Coord, piece: Optional[Piece]) -> None:
        """Set piece at coordinate."""
        self.occupancy.set_position(coord, piece)

    def get_pieces_of_color(self, color: Color) -> Iterable[Tuple[Coord, Piece]]:
        """Iterate over all pieces of given color."""
        return self.occupancy.iter_color(color)

    def find_king(self, color: Color) -> Optional[Coord]:
        """Find king position for given color."""
        return self.occupancy.find_king(color)

    def get_occupancy_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get occupancy and piece-type arrays for JIT kernels."""
        return self.occupancy._occ.copy(), self.occupancy._ptype.copy()

    def batch_set_pieces(self, updates: List[Tuple[Coord, Optional[Piece]]]) -> None:
        """Batch update multiple positions."""
        self.occupancy.batch_set_positions(updates)
# =============================================================================
#  Factory
# =============================================================================
def get_cache_manager(board: Board, current: Color) -> OptimizedCacheManager:
    """Factory that guarantees the board carries the returned manager."""
    if board is None:
        raise ValueError("get_cache_manager requires a real Board instance")
    cm = OptimizedCacheManager(board, current)
    cm.initialise(current)
    board.cache_manager = cm
    return cm
# =============================================================================
#  Legacy aliases (zero-cost)
# =============================================================================
CacheManager = OptimizedCacheManager
