# manager.py
# game3d/cache/manager.py (refactored main module)

from __future__ import annotations
"""Optimized Central cache manager – supports advanced move caching, transposition tables, and full 5600X/64GB RAM utilization."""
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
    from game3d.movement.movepiece import Move  # Add this
    from game3d.cache.caches.transposition import CompactMove
# Import the optimized cache
from game3d.cache.caches.movecache import (
    OptimizedMoveCache,
    CompactMove,
    create_optimized_move_cache
)

from game3d.cache.caches.piececache import PieceCache
from game3d.cache.caches.occupancycache import OccupancyCache
from game3d.game.zobrist import compute_zobrist, ZobristHash
from game3d.cache.caches.transposition import TTEntry
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


class OptimizedCacheManager:
    """Advanced cache manager with TT, parallel move gen, and NUMA-friendly design."""

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
        self._board = board  # Add this line
        self.occupancy = OccupancyCache(board)
        self.piece_cache = PieceCache(board)
        self.effects = EffectsCache(board)

        # Performance monitoring
        self.performance_monitor = CachePerformanceMonitor()

        # Zobrist hashing - FIXED: Use the ZobristHash class
        self._zobrist = ZobristHash()
        self._current_zobrist_hash = self._zobrist.compute_from_scratch(board, Color.WHITE)

        self.parallel = ParallelManager(self.config)

        self._move_cache: Optional[OptimizedMoveCache] = None
        self._move_counter = 0

        # Memory management
        self.memory_manager = MemoryManager(self.config, self._move_cache if hasattr(self, '_move_cache') else None)
        self._needs_rebuild = False

    @property
    def cache(self):
        """Return self for backward compatibility with code expecting .cache.piece_cache"""
        return self

    def initialise(self, current: Color) -> None:
        self._current = current  # Add this line
        self._current_zobrist_hash = self._zobrist.compute_from_scratch(self.board, current)
        self._move_cache = create_optimized_move_cache(
            self.board, current, self
        )
        # ADD this line after the move cache is created:
        self._move_cache._full_rebuild()  # <-- Perform rebuild after initialization

        self.memory_manager.move_cache_ref = self._move_cache
        self._log_cache_stats("initialization")

        # Load from disk
        if self._move_cache:
            self._move_cache._load_from_disk()

        # Start background save thread
        self._save_thread = threading.Thread(target=self._periodic_save, daemon=True)
        self._save_thread.start()

    def _periodic_save(self) -> None:
        while True:
            time.sleep(3600)  # Every hour
            if self._move_cache:
                self._move_cache._save_to_disk()

    # --------------------------------------------------------------------------
    # MOVE APPLICATION & UNDO (UNCHANGED LOGIC, KEPT FOR COMPLETENESS)
    # --------------------------------------------------------------------------
    def apply_move(self, mv, mover, current_ply):
        if not self.validate_cache_state():
            raise CacheDesyncError("Cache inconsistent before move")
        start_time = time.time()
        self.memory_manager.check_and_gc_if_needed()

        try:
            # 验证移动
            from_piece = self.piece_cache.get(mv.from_coord)
            if from_piece is None:
                # 不要直接重建，抛出异常让上层处理
                raise CacheDesyncError(f"Cache desync detected at {mv.from_coord}. Rebuild required.")
            if from_piece is None:
                print(f"[CACHE DESYNC] No piece at {mv.from_coord}, rebuilding all caches...")
                self.clear_all_caches()
                self.initialise(mover)
                raise ValueError(f"Cache desync detected at {mv.from_coord}. Caches rebuilt. Please retry move.")

            if from_piece.color != mover:
                raise ValueError(f"Piece at {mv.from_coord} belongs to {from_piece.color}, not {mover}")

            # Validate move is legal
            legal_moves = self._move_cache.legal_moves(mover)
            if mv not in legal_moves:
                print(f"[ILLEGAL MOVE] {mv} not in {len(legal_moves)} legal moves")
                self._move_cache._full_rebuild()
                legal_moves = self._move_cache.legal_moves(mover)
                if mv not in legal_moves:
                    raise ValueError(f"Move {mv} is not legal for {mover}")

            # Get captured piece BEFORE board mutation
            captured_piece = self.piece_cache.get(mv.to_coord) if getattr(mv, 'is_capture', False) else None

            # Update Zobrist hash BEFORE board mutation
            self._current_zobrist_hash = self._zobrist.update_hash_move(
                self._current_zobrist_hash, mv, from_piece, captured_piece,
                old_castling=0, new_castling=0,
                old_ep=None, new_ep=None,
                old_ply=current_ply, new_ply=current_ply + 1
            )

            # Apply move to board
            self.board.apply_move(mv)

            # Update caches incrementally
            self.occupancy.update_for_move(mv.from_coord, mv.to_coord)
            self.piece_cache.update_for_move(mv.from_coord, mv.to_coord, from_piece, captured_piece)

            # Update effect caches
            affected_caches = self.effects.get_affected_caches(mv, mover, from_piece, None, captured_piece)
            affected_caches.add("attacks")

            # Incremental update instead of full rebuild
            self.effects.update_effect_caches(mv, mover, affected_caches, current_ply)

            # Only update move cache for affected pieces
            if self._move_cache:
                self._move_cache.apply_move(mv, mover)
                # Only rebuild if necessary
                if self._needs_rebuild:
                    self._move_cache._full_rebuild()
                    self._needs_rebuild = False

                if self._should_store_in_tt(mv, from_piece):
                    # Use the CompactMove from movecache import, not transposition
                    compact_move = CompactMove(
                        mv.from_coord, mv.to_coord, from_piece.ptype,
                        getattr(mv, 'is_capture', False),
                        captured_piece.ptype if captured_piece else None,
                        getattr(mv, 'is_promotion', False)
                    )
                    self._move_cache.store_evaluation(
                        self._current_zobrist_hash, 1, 0, 0, compact_move
                    )

            self._current = mover.opposite()
            self._age_counter += 1

            duration = time.time() - start_time
            self.performance_monitor.record_move_apply_time(duration)
            self.performance_monitor.record_event(CacheEventType.MOVE_APPLIED, {
                'move': str(mv),
                'color': mover.name,
                'duration_ms': duration * 1000,
                'affected_caches': list(affected_caches)
            })

            self._move_counter += 1
            if self._move_counter % self.config.cache_stats_interval == 0:
                self._log_cache_stats("periodic")

        except CacheDesyncError as e:
            # 记录错误并重新抛出
            print(f"[ERROR] {str(e)}")
            raise
        except Exception as e:
            # 其他异常处理
            self.performance_monitor.record_event(CacheEventType.CACHE_ERROR, {
                'error': str(e),
                'move': str(mv),
                'color': mover.name
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
            piece = self.piece_cache.get(mv.to_coord)  # Piece that arrived here
            if piece is None:
                raise ValueError(f"No piece at move target {mv.to_coord} during undo")

            captured_piece = None
            if getattr(mv, "is_capture", False):
                captured_type = getattr(mv, "captured_ptype", None)
                if captured_type is not None:
                    captured_piece = Piece(mover.opposite(), captured_type)

            # Update Zobrist hash BEFORE board mutation
            self._current_zobrist_hash = self._zobrist.update_hash_move(
                self._current_zobrist_hash, mv, piece, captured_piece
            )

            # Now mutate the board
            self._undo_move_optimized(mv, mover)

            # Update caches incrementally
            self.occupancy.update_for_move(mv.to_coord, mv.from_coord)
            self.piece_cache.update_for_move(mv.to_coord, mv.from_coord, piece, None)

            # Update effect caches
            affected_caches = self.effects.get_affected_caches_for_undo(mv, mover)
            affected_caches.add("attacks")
            self.effects.update_effect_caches_for_undo(mv, mover, affected_caches, current_ply)

            # Update move cache
            if self._move_cache:
                self._move_cache.undo_move(mv, mover)

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

    def _undo_move_optimized(self, mv: 'Move', mover: Color) -> None:
        """FIXED: Handle piece restoration properly."""
        # Restore captured piece first if needed
        if getattr(mv, "is_capture", False):
            captured_type = getattr(mv, "captured_ptype", None)
            if captured_type is not None:
                self.board.set_piece(mv.to_coord, Piece(mover.opposite(), captured_type))

        # Get the piece at destination (what was moved)
        piece = self.piece_cache.get(mv.to_coord)  # FIXED: Use self.piece_cache directly
        if piece:
            # Move piece back to original position
            self.board.set_piece(mv.from_coord, piece)
            self.board.set_piece(mv.to_coord, None)

        # Handle promotion undo
        if getattr(mv, "is_promotion", False) and piece:
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
        start_time = time.time()
        self.memory_manager.check_and_gc_if_needed()

        if self._move_cache is None:
            raise RuntimeError("Move cache not initialized")

        # Delegate to move cache, which now supports parallel mode
        moves = self._move_cache.legal_moves(
            color,
            parallel=self.config.enable_parallel,
            max_workers=self.config.max_workers
        )

        duration = time.time() - start_time
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
        return self.effects.get_attacked_squares(color)

    def is_pinned(self, coord: Tuple[int, int, int], color: Optional[Color] = None) -> bool:
        if color is None:
            piece = self.piece_cache.get(coord)
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

# ==============================================================================
# AI-SAFE DATA EXPORT METHODS (DELEGATED TO export.py)
# ==============================================================================

    def export_state_for_ai(self, current_player: Color, move_number: int = 0) -> Dict[str, Any]:
        return export_state_for_ai(self.board, self.piece_cache, self._move_cache, self._current_zobrist_hash, current_player, move_number)

    def export_tensor_for_ai(self, current_player: Color, move_number: int = 0,
                             device: str = "cpu") -> torch.Tensor:
        return export_tensor_for_ai(self.board, self.piece_cache, self._move_cache, self._current_zobrist_hash, current_player, move_number, device)

    def get_legal_move_indices(self, color: Color) -> Tuple[List[int], List[int]]:
        return get_legal_move_indices(self._move_cache, color)

    def get_legal_moves_as_policy_target(self, color: Color,
                                         move_probabilities: Optional[Dict['Move', float]] = None) -> torch.Tensor:
        return get_legal_moves_as_policy_target(self._move_cache, color, move_probabilities)

    def validate_export_integrity(self) -> Dict[str, bool]:
        return validate_export_integrity(self.board, self.piece_cache, self._move_cache, self._current_zobrist_hash)

    @lru_cache(maxsize=128*1024)
    def _cached_legal_moves(self, zkey: int, color_value: int) -> Tuple['Move',...]:
        return tuple(self._generate_legal_moves_raw(Color(color_value)))

    def validate_cache_consistency(self) -> bool:
        """Validate cache matches board state."""
        for coord, piece in self.board.list_occupied():
            cached_piece = self.piece_cache.get(coord)
            if cached_piece != piece:
                print(f"[INCONSISTENCY] Board has {piece} at {coord}, cache has {cached_piece}")
                return False
        return True

    def validate_cache_state(self):
        """Check for cache consistency"""
        if self._move_cache:
            board_pieces = set(self.board.list_occupied())
            cache_pieces = set(self.piece_cache._pieces.keys())
            if board_pieces != cache_pieces:
                print(f"[ERROR] Cache inconsistency detected")
                return False
        return True

    def get_share_square_cache(self) -> 'ShareSquareCache':
        return self.effects._effect_caches["share_square"]
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
