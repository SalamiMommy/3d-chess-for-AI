# manager.py
# game3d/cache/manager.py (optimized)

from __future__ import annotations
"""Optimized Central cache manager - reduced background thread overhead."""
# game3d/cache/manager.py

import gc
import time
from typing import Dict, List, Tuple, Optional, Set, Any, TYPE_CHECKING
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import threading

from game3d.common.common import N_TOTAL_PLANES
from game3d.pieces.enums import Color, PieceType
from game3d.pieces.piece import Piece

# TYPE_CHECKING imports - no runtime fallbacks needed
if TYPE_CHECKING:
    from game3d.board.board import Board
    from game3d.movement.movepiece import Move
    from game3d.cache.caches.transposition import CompactMove

# Import the optimized cache
from game3d.cache.caches.movecache import (
    OptimizedMoveCache,
    CompactMove,
    create_optimized_move_cache
)

from game3d.cache.caches.occupancycache import OccupancyCache
from game3d.game.zobrist import compute_zobrist, ZobristHash

# Refactored imports
from .managerconfig import ManagerConfig
from .managerperformance import CachePerformanceMonitor, CacheEventType, MemoryManager
from .effects_cache import EffectsCache
from .parallelmanager import ParallelManager
from .export import (
    export_state_for_ai,
    export_tensor_for_ai,
    get_legal_move_indices,
    get_legal_moves_as_policy_target,
    validate_export_integrity
)


class CacheDesyncError(Exception):
    """Exception raised when cache desynchronization is detected."""
    pass


class OptimizedCacheManager:
    """Advanced cache manager with reduced background overhead."""

    def __init__(self, board: Board) -> None:
        global OptimizedMoveCache, CompactMove, TTEntry
        if OptimizedMoveCache is None:
            from game3d.cache.caches.movecache import (
                OptimizedMoveCache as OMC,
                CompactMove as CM,
                TTEntry as TTE,
                create_optimized_move_cache
            )
            OptimizedMoveCache = OMC
            CompactMove = CM
            TTEntry = TTE

        self.config = ManagerConfig()
        self.board = board
        self._board = board
        self.board.cache_manager = self  # Set reference on board

        # Now initialize other caches that depend on piece_cache
        self.occupancy = OccupancyCache(board)
        self.effects = EffectsCache(board, self)

        # Performance monitoring
        self.performance_monitor = CachePerformanceMonitor()

        # Zobrist hashing
        self._zobrist = ZobristHash()
        self._current_zobrist_hash = self._zobrist.compute_from_scratch(board, Color.WHITE)

        self.parallel = ParallelManager(self.config)
        self.occupancy = OccupancyCache(board)
        self._move_cache: Optional[OptimizedMoveCache] = None
        self._move_counter = 0
        self._age_counter = 0  # Add missing attribute
        self._current = Color.WHITE  # Add missing attribute

        # Memory management - disable background monitoring to reduce overhead
        self.memory_manager = None  # We'll handle memory management differently
        self._needs_rebuild = False

        # Disable background save thread - save only on explicit calls
        self._save_thread = None
        self._skip_effect_updates = False
        self._effect_update_counter = 0
        self._effect_update_interval = 10
        self._check_summary_cache: dict[str, Any] | None = None
        self._check_summary_age = -1


    @property
    def cache(self):
        """Return self for backward compatibility with code expecting .cache.piece_cache"""
        return self

    def initialise(self, current: Color) -> None:
        self._current = current
        self._current_zobrist_hash = self._zobrist.compute_from_scratch(self.board, current)
        self._move_cache = create_optimized_move_cache(
            self.board, current, self
        )

        # Perform rebuild after initialization
        if self._move_cache:
            self._move_cache._full_rebuild()

        # Load from disk only if explicitly requested
        if self.config.enable_disk_cache and self._move_cache:
            self._move_cache._load_from_disk()

        self._log_cache_stats("initialization")

    # REMOVE the _periodic_save method entirely - save manually when needed
    def save_to_disk(self) -> None:
        """Manual save to disk - call when appropriate."""
        if self._move_cache and self.config.enable_disk_cache:
            self._move_cache._save_to_disk()

    # --------------------------------------------------------------------------
    # OPTIMIZED MOVE APPLICATION & UNDO
    # --------------------------------------------------------------------------
    def apply_move(self, mv, mover, current_ply):
        """Optimized apply_move - minimal work."""
        start_time = time.perf_counter()

        try:
            # Fast validation using occupancy cache only
            from_piece = self.occupancy.get(mv.from_coord)
            if from_piece is None:
                # Mark for rebuild but continue
                self._needs_rebuild = True
                self.board.apply_move(mv)
                self._current = mover.opposite()
                self._age_counter += 1
                return

            dest_piece = self.occupancy.get(mv.to_coord)
            if (dest_piece and dest_piece.color == mover and
                    from_piece.ptype is not PieceType.KNIGHT):
                raise ValueError(
                    f"Illegal move: {from_piece} may not land on friendly {dest_piece}"
                )

            if from_piece.color != mover:
                raise ValueError(f"Wrong color piece at {mv.from_coord}")

            # Get captured piece from cache
            captured_piece = self.occupancy.get(mv.to_coord) if mv.is_capture else None

            # Update Zobrist (fast)
            self._current_zobrist_hash = self._zobrist.update_hash_move(
                self._current_zobrist_hash, mv, from_piece, captured_piece, cache=self)

            # Apply to board
            self.board.apply_move(mv)

            # CRITICAL: Update occupancy FIRST (fastest)
            self._fast_occupancy_update(mv, from_piece, mover)

            # Skip effect updates most of the time
            self._effect_update_counter += 1
            if self._effect_update_counter % self._effect_update_interval == 0:
                self._update_effects(mv, from_piece, captured_piece, mover, current_ply)

            # Update move cache WITHOUT checking board state
            if self._move_cache and not self._needs_rebuild:
                try:
                    self._move_cache.apply_move(mv, mover)
                except Exception:
                    self._needs_rebuild = True

            self._current = mover.opposite()
            self._age_counter += 1

            duration = time.perf_counter() - start_time
            self.performance_monitor.record_move_apply_time(duration)

        except Exception as e:
            self.performance_monitor.record_event(CacheEventType.CACHE_ERROR, {
                'error': str(e)
            })
            raise

    def undo_move(self, mv: 'Move', mover: Color, current_ply: int = 0) -> None:
        """
        Undo a move with proper Zobrist hash rollback and cache updates.
        CRITICAL: Order matters - hash update uses current board state.
        """
        start_time = time.time()
        self.memory_manager.check_and_gc_if_needed()

        try:
            # FIXED: Read state BEFORE mutating board
            piece = self.occupancy.get(mv.to_coord)  # Piece that arrived here
            if piece is None:
                raise ValueError(f"No piece at move target {mv.to_coord} during undo")

            captured_piece = None
            if getattr(mv, "is_capture", False):
                captured_type = getattr(mv, "captured_ptype", None)
                if captured_type is not None:
                    captured_piece = Piece(mover.opposite(), captured_type)

            # Update Zobrist hash BEFORE board mutation
            self._current_zobrist_hash = self._zobrist.update_hash_move(
                self._current_zobrist_hash, mv, from_piece, captured_piece, cache=self)

            # Now mutate the board
            self._undo_move_optimized(mv, mover, piece, captured_piece)

            # Update caches incrementally
            unpromoted_piece = Piece(piece.color, PieceType.PAWN) if getattr(mv, "is_promotion", False) else piece
            self.occupancy.set_position(mv.from_coord, unpromoted_piece)
            self.occupancy.set_position(mv.to_coord, captured_piece)

            # Update effect caches
            affected_caches = self.effects.get_affected_caches_for_undo(mv, mover)
            affected_caches.add("attacks")
            self.effects.update_effect_caches_for_undo(mv, mover, affected_caches, current_ply)

            # Update move cache
            if self._move_cache:
                self._move_cache.undo_move(mv, mover)

            # Mark network teleport targets as dirty
            self._mark_network_teleport_dirty()
            self._mark_swap_dirty()

            self._current = mover
            self._age_counter += 1

            duration = time.time() - start_time
            self.performance_monitor.record_move_undo_time(duration)
            self.performance_monitor.record_event(CacheEventType.MOVE_UNDONE, {
                'move': str(mv),
                'color': mover.name,
                'duration_ms': duration * 1000
            })

        except Exception as e:
            self.performance_monitor.record_event(CacheEventType.CACHE_ERROR, {
                'error': str(e),
                'move': str(mv),
                'color': mover.name
            })
            raise

    def _fast_occupancy_update(self, mv, from_piece, mover):
        """Fast occupancy update without board queries."""
        # Determine destination piece
        promotion_type = getattr(mv, "promotion_ptype", None)
        if getattr(mv, "is_promotion", False) and promotion_type:
            to_piece = Piece(mover, PieceType(promotion_type))
        else:
            to_piece = from_piece

        # Update occupancy cache directly
        self.occupancy.set_position(mv.from_coord, None)
        self.occupancy.set_position(mv.to_coord, to_piece)

    def _is_move_legal_fast(self, mv, mover, from_piece) -> bool:
        """Fast move legality check using cached results."""
        if self._move_cache is None:
            return False

        # Use a faster method to check move legality without generating all moves
        return self._move_cache.is_move_legal_fast(mv, mover, from_piece)

    def _light_memory_check(self):
        """Lightweight memory check instead of full psutil calls."""
        import sys
        # Simple check based on object count
        if len(gc.get_objects()) > 500000:  # Arbitrary threshold, adjust based on usage
            gc.collect()

    def _undo_move_optimized(self, mv: 'Move', mover: Color, piece: Piece, captured_piece: Optional[Piece]) -> None:
        """FIXED: Handle piece restoration properly."""
        # Move piece back to original position
        self.board.set_piece(mv.from_coord, piece)
        # Restore captured or clear
        self.board.set_piece(mv.to_coord, captured_piece)

        # Handle un-promotion if applicable
        if getattr(mv, "is_promotion", False):
            # The piece was a pawn before promotion
            self.board.set_piece(mv.from_coord, Piece(piece.color, PieceType.PAWN))

    def _should_store_in_tt(self, mv: 'Move', from_piece: Piece) -> bool:
        return True

    # --------------------------------------------------------------------------
    # TRANSPOSITION TABLE INTERFACE
    # --------------------------------------------------------------------------
    def probe_transposition_table(self, hash_value: int) -> Optional[TTEntry]:
        if not self._move_cache:
            return None
        result = self._move_cache.get_cached_evaluation(hash_value)
        if result:
            score, depth, best_move = result
            self.performance_monitor.record_event(CacheEventType.TT_HIT, {
                'hash_value': hash_value,
                'depth': depth,
                'score': score
            })
            return TTEntry(hash_value, depth, score, 0, best_move, 0)
        else:
            self.performance_monitor.record_event(CacheEventType.TT_MISS, {
                'hash_value': hash_value
            })
            return None

    def store_transposition_table(self, hash_value: int, depth: int, score: int,
                                node_type: int, best_move: Optional[CompactMove] = None) -> None:
        if self._move_cache:
            self._move_cache.store_evaluation(hash_value, depth, score, node_type, best_move)

    def get_current_zobrist_hash(self) -> int:
        return self._current_zobrist_hash

    # --------------------------------------------------------------------------
    # PARALLEL LEGAL MOVE GENERATION — KEY OPTIMIZATION FOR 5600X
    # --------------------------------------------------------------------------
    def legal_moves(self, color: Color) -> List['Move']:
        """Get legal moves with lazy rebuild."""
        # Check if rebuild needed
        if self._needs_rebuild:
            print(f"[REBUILD] Performing lazy rebuild for {color}")
            rebuild_start = time.perf_counter()
            self._move_cache._full_rebuild()
            self._needs_rebuild = False
            rebuild_time = time.perf_counter() - rebuild_start
            if rebuild_time > 0.05:  # Log slow rebuilds
                print(f"[REBUILD] Took {rebuild_time*1000:.1f}ms")

        start_time = time.perf_counter()
        moves = self._move_cache.legal_moves(
            color,
            parallel=self.config.enable_parallel,
            max_workers=self.config.max_workers
        )

        duration = time.perf_counter() - start_time
        self.performance_monitor.record_legal_move_generation_time(duration)
        return moves

    @property
    def move(self) -> OptimizedMoveCache:
        if self._move_cache is None:
            raise RuntimeError("MoveCache not initialized. Call initialise() first.")
        return self._move_cache

    # --------------------------------------------------------------------------
    # STATS & CONFIGURATION
    # --------------------------------------------------------------------------
    def get_cache_stats(self) -> Dict[str, Any]:
        base_stats = self.performance_monitor.get_performance_stats()
        if self._move_cache:
            move_cache_stats = self._move_cache.get_stats()
            base_stats.update({
                'move_cache_stats': move_cache_stats,
                'zobrist_hash': self._current_zobrist_hash,
                'main_tt_size_mb': self.config.main_tt_size_mb,
                'sym_tt_size_mb': self.config.sym_tt_size_mb,
                'main_tt_capacity_estimate': (self.config.main_tt_size_mb * 1024 * 1024) // 32,
                'sym_tt_capacity_estimate': (self.config.sym_tt_size_mb * 1024 * 1024) // 32,
                'enable_parallel': self.config.enable_parallel,
                'max_workers': self.config.max_workers,
                'enable_vectorization': self.config.enable_vectorization
            })
        base_stats['effect_caches'] = {
            name: {'type': type(cache).__name__}
            for name, cache in self.effects._effect_caches.items()
        }
        return base_stats

    def get_optimization_suggestions(self) -> List[str]:
        return self.performance_monitor.get_optimization_suggestions()

    def _log_cache_stats(self, context: str) -> None:
        stats = self.get_cache_stats()
        # print(f"[CacheManager] {context} stats:")
        # print(f"  Main TT Size: {stats['main_tt_size_mb']} MB (~{stats.get('main_tt_capacity_estimate', 0):,} entries)")
        # print(f"  Sym TT Size: {stats['sym_tt_size_mb']} MB (~{stats.get('sym_tt_capacity_estimate', 0):,} entries)")
        # print(f"  TT Hit Rate: {stats['tt_hit_rate']:.3f}")
        # print(f"  Parallel Workers: {stats.get('max_workers', 'N/A')}")
        # print(f"  Avg Move Apply Time: {stats['avg_move_apply_time_ms']:.2f}ms")
        # print(f"  Avg Legal Move Time: {stats['avg_legal_gen_time_ms']:.2f}ms")

        suggestions = self.get_optimization_suggestions()
        if suggestions:
            print(f"  Optimization Suggestions ({len(suggestions)}):")
            for suggestion in suggestions[:3]:
                print(f"    - {suggestion}")

    # --------------------------------------------------------------------------
    # EFFECT CACHE INTERFACE (DELEGATED)
    # --------------------------------------------------------------------------
    def is_frozen(self, sq: Tuple[int, int, int], victim: Color) -> bool:
        return self.effects.is_frozen(sq, victim)

    def is_movement_buffed(self, sq: Tuple[int, int, int], friendly: Color) -> bool:
        return self.effects.is_movement_buffed(sq, friendly)

    def is_movement_debuffed(self, sq: Tuple[int, int, int], victim: Color) -> bool:
        return self.effects.is_movement_debuffed(sq, victim)

    def black_hole_pull_map(self, controller: Color) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
        return self.effects.black_hole_pull_map(controller)

    def white_hole_push_map(self, controller: Color) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
        return self.effects.white_hole_push_map(controller)

    def mark_trail(self, trailblazer_sq: Tuple[int, int, int], slid_squares: Set[Tuple[int, int, int]]) -> None:
        self.effects.mark_trail(trailblazer_sq, slid_squares)

    def current_trail_squares(self, controller: Color) -> Set[Tuple[int, int, int]]:
        return self.effects.current_trail_squares(controller)

    def is_geomancy_blocked(self, sq: Tuple[int, int, int], current_ply: int) -> bool:
        return self.effects.is_geomancy_blocked(sq, current_ply)

    def block_square(self, sq: Tuple[int, int, int], current_ply: int) -> bool:
        return self.effects.block_square(sq, current_ply)

    def archery_targets(self, controller: Color) -> List[Tuple[int, int, int]]:
        return self.effects.archery_targets(controller)

    def is_valid_archery_attack(self, sq: Tuple[int, int, int], controller: Color) -> bool:
        return self.effects.is_valid_archery_attack(sq, controller)

    def can_capture_wall(self, attacker_sq: Tuple[int, int, int], wall_sq: Tuple[int, int, int], controller: Color) -> bool:
        return self.effects.can_capture_wall(attacker_sq, wall_sq, controller)

    def pieces_at(self, sq: Tuple[int, int, int]) -> List['Piece']:
        return self.effects.pieces_at(sq)

    def top_piece(self, sq: Tuple[int, int, int]) -> Optional['Piece']:
        return self.effects.top_piece(sq)

    def get_attacked_squares(self, color: Color) -> Set[Tuple[int, int, int]]:
        """Get all squares attacked by pieces of the given color."""
        if self._move_cache:
            return self._move_cache.get_attacked_squares(color)
        return set()

    def is_pinned(self, coord: Tuple[int, int, int], color: Optional[Color] = None) -> bool:
        if color is None:
            piece = self.occupancy.get(coord)
            if piece is None:
                return False
            color = piece.color
        return False

    def store_attacked_squares(self, color: Color, attacked: Set[Tuple[int, int, int]]) -> None:
        self.effects.store_attacked_squares(color, attacked)

    # --------------------------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------------------------
    def configure_transposition_table(self, size_mb: int) -> None:
        """Grow TT only if we are still below 85 % physical RAM."""
        import psutil
        if psutil.virtual_memory().percent >= 85.0:
            print("[CacheManager] Refused TT expansion: RAM already at 85 %")
            return
        self.config.main_tt_size_mb = size_mb
        if self._move_cache:
            current_color = self._move_cache._current
            self._move_cache = create_optimized_move_cache(
                self.board, current_color, self,
                main_tt_size_mb=size_mb,
                sym_tt_size_mb=self.config.sym_tt_size_mb
            )

    def configure_symmetry_tt(self, size_mb: int) -> None:  # Add this method
        self.config.sym_tt_size_mb = size_mb
        if self._move_cache:
            current_color = self._move_cache._current
            self._move_cache = create_optimized_move_cache(
                self.board, current_color, self,
                main_tt_size_mb=self.config.main_tt_size_mb,  # Preserve main size
                sym_tt_size_mb=size_mb
            )

    def set_parallel_processing(self, enabled: bool) -> None:
        self.config.enable_parallel = enabled

    def set_vectorization(self, enabled: bool) -> None:
        self.config.enable_vectorization = enabled

    # --------------------------------------------------------------------------
    # UTILITY
    # --------------------------------------------------------------------------
    def clear_all_caches(self) -> None:
        if self._move_cache:
            self._move_cache.clear()
        self.effects.clear_all_effects()
        self.occupancy.rebuild(self.board)
        self.performance_monitor = CachePerformanceMonitor()
        self.performance_monitor.record_event(CacheEventType.CACHE_CLEARED, {})
        self._move_counter = 0
        gc.collect()
        self.parallel.shutdown()

    def export_cache_state(self) -> Dict[str, Any]:
        return {
            'zobrist_hash': self._current_zobrist_hash,
            'performance_stats': self.get_cache_stats(),
            'board_state': {
                'occupied_squares': len(list(self.board.list_occupied())),
                'current_player': self._move_cache._current.name if self._move_cache else None
            },
            'effect_cache_status': {
                name: {'type': type(cache).__name__}
                for name, cache in self.effects._effect_caches.items()
            }
        }

    def _mark_network_teleport_dirty(self) -> None:
        """Mark network teleport targets as dirty for both colors."""
        if hasattr(self, '_network_teleport_dirty'):
            self._network_teleport_dirty[Color.WHITE] = True
            self._network_teleport_dirty[Color.BLACK] = True


    def _update_effects(self, mv, from_piece, captured_piece, mover, current_ply):
        """Update effects only when needed."""
        # Determine which effects might be affected
        relevant_effects = set()

        if from_piece.ptype == PieceType.FREEZER:
            relevant_effects.add("freeze")
        elif from_piece.ptype == PieceType.ARCHER:
            relevant_effects.add("archery")
        elif from_piece.ptype in {PieceType.BLACKHOLE, PieceType.WHITEHOLE}:
            relevant_effects.add("black_hole_suck")
            relevant_effects.add("white_hole_push")

        # Only update relevant effects
        for effect_name in relevant_effects:
            try:
                cache = self.effects._effect_caches[effect_name]
                if hasattr(cache, 'apply_move'):
                    cache.apply_move(mv, mover, self.board)
            except Exception as e:
                print(f"Effect {effect_name} update failed: {e}")
# ==============================================================================
# AI-SAFE DATA EXPORT METHODS (DELEGATED TO export.py)
# ==============================================================================

    def export_state_for_ai(self, current_player: Color, move_number: int = 0) -> Dict[str, Any]:
        return export_state_for_ai(self.board, self.occupancy, self._move_cache, self._current_zobrist_hash, current_player, move_number)

    def export_tensor_for_ai(self, current_player: Color, move_number: int = 0,
                             device: str = "cpu") -> torch.Tensor:
        return export_tensor_for_ai(self.board, self.occupancy, self._move_cache, self._current_zobrist_hash, current_player, move_number, device)

    def get_legal_move_indices(self, color: Color) -> Tuple[List[int], List[int]]:
        return get_legal_move_indices(self._move_cache, color)

    def get_legal_moves_as_policy_target(self, color: Color,
                                         move_probabilities: Optional[Dict['Move', float]] = None) -> torch.Tensor:
        return get_legal_moves_as_policy_target(self._move_cache, color, move_probabilities)

    def validate_export_integrity(self) -> Dict[str, bool]:
        return validate_export_integrity(self.board, self.occupancy, self._move_cache, self._current_zobrist_hash)

    @lru_cache(maxsize=128*1024)
    def _cached_legal_moves(self, zkey: int, color_value: int) -> Tuple['Move',...]:
        return tuple(self._generate_legal_moves_raw(Color(color_value)))

    def validate_cache_consistency(self) -> bool:
        """Validate cache matches board state."""
        for coord, piece in self.board.list_occupied():
            cached_piece = self.occupancy.get(coord)
            if cached_piece != piece:
                print(f"[INCONSISTENCY] Board has {piece} at {coord}, cache has {cached_piece}")
                return False
        return True

    def validate_cache_state(self):
        """Check for cache consistency"""
        if self._move_cache:
            board_pieces = set(coord for coord, _ in self.board.list_occupied())
            cache_pieces = set(self.occupancy._white_pieces.keys()) | set(self.occupancy._black_pieces.keys())
            if board_pieces != cache_pieces:
                print(f"[ERROR] Cache inconsistency detected")
                return False
        return True

    def get_share_square_cache(self) -> 'ShareSquareCache':
        return self.effects._effect_caches["share_square"]

    @property
    def piece_cache(self):
        """Return the occupancy cache for backward compatibility."""
        return self.occupancy

    def _mark_swap_dirty(self) -> None:
        """Mark swap targets as dirty for both colors."""
        if hasattr(self, '_swap_targets_dirty'):
            self._swap_targets_dirty[Color.WHITE] = True
            self._swap_targets_dirty[Color.BLACK] = True

    def get_check_summary(self) -> dict[str, Any]:
        if self._check_summary_age != self._age_counter:
            self._check_summary_cache = self._recompute_check_summary()
            self._check_summary_age = self._age_counter
        return self._check_summary_cache


    def get_check_summary(self) -> dict[str, Any]:
        if self._check_summary_age != self._age_counter:
            self._check_summary_cache = self._recompute_check_summary()
            self._check_summary_age = self._age_counter
        return self._check_summary_cache

    # ---------- private ----------

    def _recompute_check_summary(self) -> dict[str, Any]:
        # --- helpers ---
        def king_pos(color: Color) -> tuple[int, int, int] | None:
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

        # --- king squares ---
        w_k = king_pos(Color.WHITE)
        b_k = king_pos(Color.BLACK)

        # --- attacked squares (always cheap; we still need them for other logic) ---
        w_at = self.get_attacked_squares(Color.WHITE)
        b_at = self.get_attacked_squares(Color.BLACK)

        # --- checkers / check flags (skip when priests guard the king) ---
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
# ==============================================================================
# FACTORY & BACKWARD COMPATIBILITY
# ==============================================================================

def get_cache_manager(board: Board, current: Color) -> OptimizedCacheManager:
    cache = OptimizedCacheManager(board)
    cache.initialise(current)
    return cache


class CacheManager(OptimizedCacheManager):
    def __init__(self, board: Board) -> None:
        super().__init__(board)

    def sync_board(self, new_board: Board) -> None:
        import warnings
        warnings.warn("sync_board is deprecated. Create a new CacheManager instead.", DeprecationWarning)
        self.board = new_board
        self.occupancy.rebuild(new_board)
        if self._move_cache:
            self._move_cache._board = new_board

    def replace_board(self, new_board: Board) -> None:
        import warnings
        warnings.warn("replace_board is deprecated. Create a new CacheManager instead.", DeprecationWarning)
        self.board = new_board
        self.occupancy.rebuild(new_board)
        if self._move_cache:
            self._move_cache._board = new_board
