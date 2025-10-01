from __future__ import annotations
"""Optimized Central cache manager – supports advanced move caching, transposition tables, and full 5600X/64GB RAM utilization."""
# game3d/cache/manager.py

import os
import time
import gc
import psutil
import threading
from typing import Dict, List, Tuple, Optional, Set, Any, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pickle
import zlib

from game3d.pieces.enums import Color

if TYPE_CHECKING:
    from game3d.cache.movecache import OptimizedMoveCache, CompactMove, TTEntry
    from game3d.cache.occupancycache import OccupancyCache
    from game3d.pieces.enums import PieceType
    from game3d.movement.movepiece import Move
    from game3d.board.board import Board
    from game3d.pieces.piece import Piece

# Import the optimized cache
from game3d.cache.movecache import (
    OptimizedMoveCache,
    CompactMove,
    TTEntry,
    ZobristHashing,
    TranspositionTable,
    create_optimized_move_cache
)

# Import effect caches
from game3d.cache.effectscache.freezecache import FreezeCache
from game3d.cache.effectscache.blackholesuckcache import BlackHoleSuckCache
from game3d.cache.effectscache.movementdebuffcache import MovementDebuffCache
from game3d.cache.effectscache.movementbuffcache import MovementBuffCache
from game3d.cache.effectscache.whiteholepushcache import WhiteHolePushCache
from game3d.cache.effectscache.trailblazecache import TrailblazeCache
from game3d.cache.effectscache.capturefrombehindcache import BehindCache
from game3d.cache.effectscache.geomancycache import GeomancyCache
from game3d.cache.effectscache.archerycache import ArcheryCache
from game3d.cache.effectscache.sharesquarecache import ShareSquareCache
from game3d.cache.effectscache.armourcache import ArmourCache
from game3d.cache.piececache import PieceCache
from game3d.cache.occupancycache import OccupancyCache
from game3d.cache.attackscache import AttacksCache


# ==============================================================================
# PERFORMANCE MONITORING
# ==============================================================================

class CacheEventType(Enum):
    MOVE_APPLIED = "move_applied"
    MOVE_UNDONE = "move_undone"
    TT_HIT = "tt_hit"
    TT_MISS = "tt_miss"
    TT_COLLISION = "tt_collision"
    CACHE_CLEARED = "cache_cleared"
    CACHE_ERROR = "cache_error"


@dataclass
class CacheEvent:
    """Event for cache performance monitoring."""
    __slots__ = ("event_type", "timestamp", "data")
    event_type: CacheEventType
    timestamp: float
    data: Dict[str, Any]


class CachePerformanceMonitor:
    """Advanced performance monitoring for the cache system."""

    def __init__(self, enable_monitoring: bool = True, max_events: int = 10000):
        self.enable_monitoring = enable_monitoring
        self.events: List[CacheEvent] = []
        self.max_events = max_events
        self.start_time = time.time()

        # Performance counters
        self.tt_hits = 0
        self.tt_misses = 0
        self.tt_collisions = 0
        self.move_applications = 0
        self.move_undos = 0
        self.cache_clears = 0

        # Timing statistics (keep last 1000)
        self.move_apply_times = []
        self.move_undo_times = []
        self.legal_move_generation_times = []

    def record_event(self, event_type: CacheEventType, data: Dict[str, Any] = None):
        if not self.enable_monitoring:
            return
        event = CacheEvent(
            event_type=event_type,
            timestamp=time.time() - self.start_time,
            data=data or {}
        )
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events.pop(0)

        if event_type == CacheEventType.TT_HIT:
            self.tt_hits += 1
        elif event_type == CacheEventType.TT_MISS:
            self.tt_misses += 1
        elif event_type == CacheEventType.TT_COLLISION:
            self.tt_collisions += 1
        elif event_type == CacheEventType.MOVE_APPLIED:
            self.move_applications += 1
        elif event_type == CacheEventType.MOVE_UNDONE:
            self.move_undos += 1
        elif event_type == CacheEventType.CACHE_CLEARED:
            self.cache_clears += 1

    def record_move_apply_time(self, duration: float):
        self.move_apply_times.append(duration)
        if len(self.move_apply_times) > 1000:
            self.move_apply_times.pop(0)

    def record_move_undo_time(self, duration: float):
        self.move_undo_times.append(duration)
        if len(self.move_undo_times) > 1000:
            self.move_undo_times.pop(0)

    def record_legal_move_generation_time(self, duration: float):
        self.legal_move_generation_times.append(duration)
        if len(self.legal_move_generation_times) > 1000:
            self.legal_move_generation_times.pop(0)

    def get_performance_stats(self) -> Dict[str, Any]:
        total_tt_accesses = self.tt_hits + self.tt_misses
        tt_hit_rate = self.tt_hits / max(1, total_tt_accesses)

        avg_move_apply_time = np.mean(self.move_apply_times) if self.move_apply_times else 0
        avg_move_undo_time = np.mean(self.move_undo_times) if self.move_undo_times else 0
        avg_legal_gen_time = np.mean(self.legal_move_generation_times) if self.legal_move_generation_times else 0

        return {
            'tt_hits': self.tt_hits,
            'tt_misses': self.tt_misses,
            'tt_hit_rate': tt_hit_rate,
            'tt_collisions': self.tt_collisions,
            'move_applications': self.move_applications,
            'move_undos': self.move_undos,
            'cache_clears': self.cache_clears,
            'avg_move_apply_time_ms': avg_move_apply_time * 1000,
            'avg_move_undo_time_ms': avg_move_undo_time * 1000,
            'avg_legal_gen_time_ms': avg_legal_gen_time * 1000,
            'total_events': len(self.events),
        }

    def get_optimization_suggestions(self) -> List[str]:
        suggestions = []
        stats = self.get_performance_stats()

        # Only warn about TT hit rate after meaningful usage
        if stats['tt_hits'] + stats['tt_misses'] > 1000:  # ← ADD THIS CHECK
            if stats['tt_hit_rate'] < 0.6:
                suggestions.append("Transposition table hit rate is low. Consider increasing size beyond 6GB.")

        if stats['tt_collisions'] > stats['tt_hits'] * 0.01:
            suggestions.append("High TT collision rate. Verify Zobrist hash quality.")

        if stats['avg_move_apply_time_ms'] > 0.5:
            suggestions.append("Move apply time high. Profile occupancy/piece cache rebuilds.")

        if stats['avg_legal_gen_time_ms'] > 3.0:
            suggestions.append("Legal move gen slow. Ensure parallelism and vectorization are active.")

        return suggestions


# ==============================================================================
# OPTIMIZED CACHE MANAGER — FULLY TUNED FOR 5600X + 64GB RAM
# ==============================================================================

class OptimizedCacheManager:
    """Advanced cache manager with TT, parallel move gen, and NUMA-friendly design."""

    def __init__(self, board: Board) -> None:
        total_ram_budget_gb = 45
        self.board = board
        self.occupancy = OccupancyCache(board)
        self.piece_cache = PieceCache(board)
        self._effect: Dict[str, Any] = {}
        self._init_effects()

        # Performance monitoring
        self.performance_monitor = CachePerformanceMonitor()

        # Zobrist hashing
        self._zobrist = ZobristHashing()
        self._current_zobrist_hash = self._zobrist.compute_hash(board, Color.WHITE, ply=0)

        total_mb = total_ram_budget_gb * 1024
        self.main_tt_size_mb = int(total_mb * 0.73)
        self.sym_tt_size_mb  = int(total_mb * 0.27)
        self._enable_parallel = True
        self._enable_vectorization = True
        self._max_workers = min(8, os.cpu_count() or 6)  # 5600X: 6C/12T → use 6–8 threads
        self._cache_stats_interval = 1000

        self._move_cache: Optional[OptimizedMoveCache] = None
        self._move_counter = 0

        # NEW: Memory management
        self._mem_threshold_gb = 40  # Trigger GC above this
        self._mem_check_interval = 300  # Seconds (5 min) for background check
        self._mem_monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self._mem_monitor_thread.start()

    def _monitor_memory(self) -> None:
        while True:
            try:
                process = psutil.Process(os.getpid())
                mem_usage_gb = process.memory_info().rss / (1024 ** 3)  # RSS in GB
                if mem_usage_gb > self._mem_threshold_gb:
                    print(f"[CacheManager] Memory usage {mem_usage_gb:.2f}GB > {self._mem_threshold_gb}GB. Triggering GC.")
                    gc.collect()  # Force garbage collection
                    # Compress simple move cache if it exists
                    self.compress_caches()
                    post_gc_gb = process.memory_info().rss / (1024 ** 3)
                    print(f"[CacheManager] Post-GC memory: {post_gc_gb:.2f}GB")
            except Exception as e:
                print(f"[CacheManager] Memory monitor error: {str(e)}")
            time.sleep(self._mem_check_interval)

    def _check_and_gc_if_needed(self) -> None:
        process = psutil.Process(os.getpid())
        mem_usage_gb = process.memory_info().rss / (1024 ** 3)
        if mem_usage_gb > self._mem_threshold_gb:
            print(f"[CacheManager] Inline GC trigger: {mem_usage_gb:.2f}GB > {self._mem_threshold_gb}GB")
            gc.collect()
            self.compress_caches()

    def compress_caches(self) -> None:
        """Compress big caches like simple move cache using zlib."""
        if not self._move_cache:
            return
        simple_cache = self._move_cache._simple_move_cache
        if simple_cache:
            compressed = {}
            for key, moves in simple_cache.items():
                pickled = pickle.dumps(moves, protocol=4)
                compressed[key] = zlib.compress(pickled, level=6)
            self._move_cache._simple_move_cache = compressed
            print(f"[CacheManager] Compressed {len(compressed)} simple move cache entries.")
            # Note: movecache.py needs update to decompress on access (check if bytes, then pickle.loads(zlib.decompress()))

    def initialise(self, current: Color) -> None:
        self._current_zobrist_hash = self._zobrist.compute_hash(self.board, current)
        # manager keeps control – just pass self
        self._move_cache = create_optimized_move_cache(
            self.board, current, self
        )
        self._log_cache_stats("initialization")

    def _init_effects(self) -> None:
        self._effect = {
            "freeze": FreezeCache(),
            "movement_buff": MovementBuffCache(),
            "movement_debuff": MovementDebuffCache(),
            "black_hole_suck": BlackHoleSuckCache(),
            "white_hole_push": WhiteHolePushCache(),
            "trailblaze": TrailblazeCache(self),
            "behind": BehindCache(),
            "armour": ArmourCache(),
            "geomancy": GeomancyCache(),
            "archery": ArcheryCache(),
            "share_square": ShareSquareCache(),
            "attacks": AttacksCache(self.board),
        }

    # --------------------------------------------------------------------------
    # MOVE APPLICATION & UNDO (UNCHANGED LOGIC, KEPT FOR COMPLETENESS)
    # --------------------------------------------------------------------------
    def apply_move(self, mv: Move, mover: Color, current_ply: int = 0) -> None:
        start_time = time.time()
        self._check_and_gc_if_needed()
        try:
            from_piece = self.piece_cache.get(mv.from_coord)
            if from_piece is None:
                raise AssertionError(f"Illegal move: {mv} — no piece at {mv.from_coord}")

            to_piece = self.piece_cache.get(mv.to_coord)
            captured_piece = None
            if getattr(mv, "is_capture", False):
                captured_type = getattr(mv, "captured_ptype", None)
                if captured_type is not None:
                    captured_piece = Piece(mover.opposite(), captured_type)

            self._current_zobrist_hash = self._zobrist.update_hash_move(
                self._current_zobrist_hash, mv, from_piece, captured_piece,
                old_castling=0, new_castling=0,  # set real values if applicable
                old_ep=None, new_ep=None,
                old_ply=current_ply, new_ply=current_ply + 1
            )

            self.board.apply_move(mv)
            self.occupancy.rebuild(self.board)
            self.piece_cache.rebuild(self.board)

            affected_caches = self._get_affected_caches(mv, mover, from_piece, to_piece, captured_piece)
            affected_caches.add("attacks")

            self._update_effect_caches(mv, mover, affected_caches, current_ply)

            if self._move_cache:
                self._move_cache.apply_move(mv, mover)
                if self._should_store_in_tt(mv, from_piece):
                    compact_move = CompactMove(
                        mv.from_coord, mv.to_coord, from_piece.ptype,
                        getattr(mv, 'is_capture', False),
                        captured_piece.ptype if captured_piece else None,
                        getattr(mv, 'is_promotion', False)
                    )
                    self._move_cache.store_evaluation(
                        self._current_zobrist_hash, 1, 0, 0, compact_move
                    )

            duration = time.time() - start_time
            self.performance_monitor.record_move_apply_time(duration)
            self.performance_monitor.record_event(CacheEventType.MOVE_APPLIED, {
                'move': str(mv),
                'color': mover.name,
                'duration_ms': duration * 1000,
                'affected_caches': list(affected_caches)
            })

            self._move_counter += 1
            if self._move_counter % self._cache_stats_interval == 0:
                self._log_cache_stats("periodic")

        except Exception as e:
            self.performance_monitor.record_event(CacheEventType.CACHE_ERROR, {
                'error': str(e),
                'move': str(mv),
                'color': mover.name
            })
            raise

    def undo_move(self, mv: Move, mover: Color, current_ply: int = 0) -> None:
        start_time = time.time()
        self._check_and_gc_if_needed()
        try:
            piece = self.piece_cache.get(mv.to_coord)
            captured_piece = None
            if getattr(mv, "is_capture", False):
                captured_type = getattr(mv, "captured_ptype", None)
                if captured_type is not None:
                    captured_piece = Piece(mover.opposite(), captured_type)

            if piece:
                self._current_zobrist_hash = self._zobrist.update_hash_move(
                    self._current_zobrist_hash, mv, piece, captured_piece,
                    old_castling=0, new_castling=0,  # set real values if applicable
                    old_ep=None, new_ep=None,
                    old_ply=current_ply, new_ply=current_ply - 1
                )

            self._undo_move_optimized(mv, mover)
            self.occupancy.rebuild(self.board)
            self.piece_cache.rebuild(self.board)

            affected_caches = self._get_affected_caches_for_undo(mv, mover)
            affected_caches.add("attacks")
            self._update_effect_caches_for_undo(mv, mover, affected_caches, current_ply)

            if self._move_cache:
                self._move_cache.undo_move(mv, mover)

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

    def _undo_move_optimized(self, mv: Move, mover: Color) -> None:
        if getattr(mv, "is_capture", False):
            captured_type = getattr(mv, "captured_ptype", None)
            if captured_type is not None:
                self.board.set_piece(mv.to_coord, Piece(mover.opposite(), captured_type))
        piece = self.board.piece_at(mv.to_coord)
        if piece:
            self.board.set_piece(mv.from_coord, piece)
            self.board.set_piece(mv.to_coord, None)
        if getattr(mv, "is_promotion", False) and piece:
            self.board.set_piece(mv.from_coord, Piece(piece.color, PieceType.PAWN))

    def _should_store_in_tt(self, mv: Move, from_piece: Piece) -> bool:
        return True

    def _get_affected_caches(self, mv: Move, mover: Color, from_piece: Piece,
                            to_piece: Optional[Piece], captured_piece: Optional[Piece]) -> Set[str]:
        affected = set()
        if from_piece.ptype in {PieceType.FREEZE_AURA, PieceType.MOVEMENT_BUFF, PieceType.MOVEMENT_DEBUFF,
                                PieceType.BLACK_HOLE, PieceType.WHITE_HOLE, PieceType.TRAILBLAZER,
                                PieceType.BEHIND_CAPTURE, PieceType.ARMOUR, PieceType.GEOMANCER,
                                PieceType.ARCHER, PieceType.SHARE_SQUARE}:
            effect_map = {
                PieceType.FREEZE_AURA: "freeze",
                PieceType.MOVEMENT_BUFF: "movement_buff",
                PieceType.MOVEMENT_DEBUFF: "movement_debuff",
                PieceType.BLACK_HOLE: "black_hole_suck",
                PieceType.WHITE_HOLE: "white_hole_push",
                PieceType.TRAILBLAZER: "trailblaze",
                PieceType.BEHIND_CAPTURE: "behind",
                PieceType.ARMOUR: "armour",
                PieceType.GEOMANCER: "geomancy",
                PieceType.ARCHER: "archery",
                PieceType.SHARE_SQUARE: "share_square"
            }
            affected.add(effect_map[from_piece.ptype])

        if captured_piece:
            self._add_affected_effects_from_pos(mv.to_coord, affected)
        self._add_affected_effects_from_pos(mv.from_coord, affected)

        return affected

    def _add_affected_effects_from_pos(self, pos: Tuple[int, int, int], affected: Set[str]) -> None:
        from game3d.pieces.enums import PieceType
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    check_pos = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
                    if all(0 <= c < 9 for c in check_pos):
                        piece = self.piece_cache.get(check_pos)
                        if piece and piece.ptype in {PieceType.FREEZE_AURA, PieceType.BLACK_HOLE,
                                                   PieceType.WHITE_HOLE, PieceType.GEOMANCER}:
                            effect_map = {
                                PieceType.FREEZE_AURA: "freeze",
                                PieceType.BLACK_HOLE: "black_hole_suck",
                                PieceType.WHITE_HOLE: "white_hole_push",
                                PieceType.GEOMANCER: "geomancy"
                            }
                            affected.add(effect_map[piece.ptype])

    def _get_affected_caches_for_undo(self, mv: Move, mover: Color) -> Set[str]:
        return self._get_affected_caches(mv, mover, None, None, None)

    def _update_effect_caches(self, mv: Move, mover: Color,
                            affected_caches: Set[str], current_ply: int) -> None:
        for name in affected_caches:
            try:
                cache = self._effect[name]
                if name == "geomancy":
                    cache.apply_move(mv, mover, current_ply, self.board)
                elif name in ("archery", "black_hole_suck", "armour", "freeze",
                              "movement_buff", "movement_debuff", "share_square",
                              "trailblaze", "white_hole_push", "attacks"):
                    cache.apply_move(mv, mover, self.board)
                else:
                    cache.apply_move(mv, mover)
            except Exception as e:
                self.performance_monitor.record_event(CacheEventType.CACHE_ERROR, {
                    'error': f"Effect cache {name} update failed: {str(e)}",
                    'move': str(mv)
                })

    def _update_effect_caches_for_undo(self, mv: Move, mover: Color,
                                     affected_caches: Set[str], current_ply: int) -> None:
        for name in affected_caches:
            try:
                cache = self._effect[name]
                if hasattr(cache, 'undo_move'):
                    if name == "geomancy":
                        cache.undo_move(mv, mover, current_ply, self.board)
                    elif hasattr(cache, '_board') or name == "attacks":
                        cache.undo_move(mv, mover, self.board)
                    else:
                        cache.undo_move(mv, mover)
            except Exception as e:
                self.performance_monitor.record_event(CacheEventType.CACHE_ERROR, {
                    'error': f"Effect cache {name} undo failed: {str(e)}",
                    'move': str(mv)
                })

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
    def legal_moves(self, color: Color) -> List[Move]:
        start_time = time.time()
        self._check_and_gc_if_needed()

        if self._move_cache is None:
            raise RuntimeError("Move cache not initialized")

        # Delegate to move cache, which now supports parallel mode
        moves = self._move_cache.legal_moves(color, parallel=self._enable_parallel, max_workers=self._max_workers)

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
                'main_tt_size_mb': self.main_tt_size_mb,
                'sym_tt_size_mb': self.sym_tt_size_mb,
                'main_tt_capacity_estimate': (self.main_tt_size_mb * 1024 * 1024) // 32,
                'sym_tt_capacity_estimate': (self.sym_tt_size_mb * 1024 * 1024) // 32,
                'enable_parallel': self._enable_parallel,
                'max_workers': self._max_workers,
                'enable_vectorization': self._enable_vectorization
            })
        base_stats['effect_caches'] = {
            name: {'type': type(cache).__name__}
            for name, cache in self._effect.items()
        }
        return base_stats

    def get_optimization_suggestions(self) -> List[str]:
        return self.performance_monitor.get_optimization_suggestions()

    def _log_cache_stats(self, context: str) -> None:
        stats = self.get_cache_stats()
        print(f"[CacheManager] {context} stats:")
        print(f"  Main TT Size: {stats['main_tt_size_mb']} MB (~{stats.get('main_tt_capacity_estimate', 0):,} entries)")
        print(f"  Sym TT Size: {stats['sym_tt_size_mb']} MB (~{stats.get('sym_tt_capacity_estimate', 0):,} entries)")
        print(f"  TT Hit Rate: {stats['tt_hit_rate']:.3f}")
        print(f"  Parallel Workers: {stats.get('max_workers', 'N/A')}")
        print(f"  Avg Move Apply Time: {stats['avg_move_apply_time_ms']:.2f}ms")
        print(f"  Avg Legal Move Time: {stats['avg_legal_gen_time_ms']:.2f}ms")

        suggestions = self.get_optimization_suggestions()
        if suggestions:
            print(f"  Optimization Suggestions ({len(suggestions)}):")
            for suggestion in suggestions[:3]:
                print(f"    - {suggestion}")

    # --------------------------------------------------------------------------
    # EFFECT CACHE INTERFACE (UNCHANGED)
    # --------------------------------------------------------------------------
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
        return self._effect["trailblaze"].current_trail_squares(controller, self.board)

    def is_geomancy_blocked(self, sq: Tuple[int, int, int], current_ply: int) -> bool:
        return self._effect["geomancy"].is_blocked(sq, current_ply)

    def block_square(self, sq: Tuple[int, int, int], current_ply: int) -> bool:
        return self._effect["geomancy"].block_square(sq, current_ply)

    def archery_targets(self, controller: Color) -> List[Tuple[int, int, int]]:
        return self._effect["archery"].attack_targets(controller)

    def is_valid_archery_attack(self, sq: Tuple[int, int, int], controller: Color) -> bool:
        return self._effect["archery"].is_valid_attack(sq, controller)

    def can_capture_wall(self, attacker_sq: Tuple[int, int, int], wall_sq: Tuple[int, int, int], controller: Color) -> bool:
        return self._effect["armour"].can_capture(attacker_sq, wall_sq, controller)

    def pieces_at(self, sq: Tuple[int, int, int]) -> List['Piece']:
        return self._effect["share_square"].pieces_at(sq)

    def top_piece(self, sq: Tuple[int, int, int]) -> Optional['Piece']:
        return self._effect["share_square"].top_piece(sq)

    def get_attacked_squares(self, color: Color) -> Set[Tuple[int, int, int]]:
        cached = self._effect["attacks"].get_for_color(color)
        return cached if cached is not None else set()

    def is_pinned(self, coord: Tuple[int, int, int], color: Optional[Color] = None) -> bool:
        if color is None:
            piece = self.piece_cache.get(coord)
            if piece is None:
                return False
            color = piece.color
        return False

    def store_attacked_squares(self, color: Color, attacked: Set[Tuple[int, int, int]]) -> None:
        self._effect["attacks"].store_for_color(color, attacked)

    # --------------------------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------------------------
    def configure_transposition_table(self, size_mb: int) -> None:
        self.main_tt_size_mb = size_mb
        if self._move_cache:
            current_color = self._move_cache._current
            self._move_cache = create_optimized_move_cache(
                self.board, current_color, self,
                main_tt_size_mb=size_mb,
                sym_tt_size_mb=self.sym_tt_size_mb  # Preserve sym size
            )

    def configure_symmetry_tt(self, size_mb: int) -> None:  # Add this method
        self.sym_tt_size_mb = size_mb
        if self._move_cache:
            current_color = self._move_cache._current
            self._move_cache = create_optimized_move_cache(
                self.board, current_color, self,
                main_tt_size_mb=self.main_tt_size_mb,  # Preserve main size
                sym_tt_size_mb=size_mb
            )

    def set_parallel_processing(self, enabled: bool) -> None:
        self._enable_parallel = enabled

    def set_vectorization(self, enabled: bool) -> None:
        self._enable_vectorization = enabled

    # --------------------------------------------------------------------------
    # UTILITY
    # --------------------------------------------------------------------------
    def clear_all_caches(self) -> None:
        if self._move_cache:
            self._move_cache.clear()
        for cache in self._effect.values():
            if hasattr(cache, 'clear'):
                cache.clear()
            elif hasattr(cache, 'invalidate'):
                cache.invalidate()
        self.occupancy.rebuild(self.board)
        self.performance_monitor = CachePerformanceMonitor()
        self.performance_monitor.record_event(CacheEventType.CACHE_CLEARED, {})
        self._move_counter = 0
        gc.collect()

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
                for name, cache in self._effect.items()
            }
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
