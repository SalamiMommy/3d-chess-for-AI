from __future__ import annotations
"""Optimized Central cache manager – supports advanced move caching and transposition tables."""
# game3d/cache/manager.py

from typing import Dict, List, Tuple, Optional, Set, Any, TYPE_CHECKING
import time
from dataclasses import dataclass
import numpy as np
from enum import Enum
from game3d.pieces.enums import Color
if TYPE_CHECKING:
    from game3d.cache.movecache import OptimizedMoveCache, CompactMove, TTEntry
    from game3d.cache.occupancycache import OccupancyCache
    from game3d.pieces.enums import Color, PieceType
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
# Import occupancy cache (this was missing!)
from game3d.cache.occupancycache import OccupancyCache
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

@dataclass
class CacheEvent:
    """Event for cache performance monitoring."""
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

        # Timing statistics
        self.move_apply_times = []
        self.move_undo_times = []
        self.legal_move_generation_times = []

    def record_event(self, event_type: CacheEventType, data: Dict[str, Any] = None):
        """Record a cache event."""
        event = CacheEvent(
            event_type=event_type,
            timestamp=time.time() - self.start_time,
            data=data or {}
        )

        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events.pop(0)

        # Update counters
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
        """Record time taken for move application."""
        self.move_apply_times.append(duration)
        if len(self.move_apply_times) > 1000:
            self.move_apply_times.pop(0)

    def record_move_undo_time(self, duration: float):
        """Record time taken for move undo."""
        self.move_undo_times.append(duration)
        if len(self.move_undo_times) > 1000:
            self.move_undo_times.pop(0)

    def record_legal_move_generation_time(self, duration: float):
        """Record time taken for legal move generation."""
        self.legal_move_generation_times.append(duration)
        if len(self.legal_move_generation_times) > 1000:
            self.legal_move_generation_times.pop(0)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
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
            'avg_move_apply_time_ms': avg_move_apply_time,
            'avg_move_undo_time_ms': avg_move_undo_time,
            'avg_legal_gen_time_ms': avg_legal_gen_time,
            'total_events': len(self.events),
        }

    def get_optimization_suggestions(self) -> List[str]:
        """Get optimization suggestions based on performance data."""
        suggestions = []
        stats = self.get_performance_stats()

        if stats['tt_hit_rate'] < 0.4:
            suggestions.append("Low transposition table hit rate. Consider increasing table size or improving replacement strategy.")

        if stats['tt_collisions'] > stats['tt_hits'] * 0.02:
            suggestions.append("High collision rate. Consider using larger hash keys or better hash function.")

        if stats['avg_move_apply_time_ms'] > 1.0:
            suggestions.append("Slow move application. Consider optimizing incremental updates.")

        if stats['avg_legal_gen_time_ms'] > 5.0:
            suggestions.append("Slow move generation. Consider vectorization or better caching strategies.")

        return suggestions

# ==============================================================================
# OPTIMIZED CACHE MANAGER
# ==============================================================================

class OptimizedCacheManager:
    """Advanced cache manager with integrated transposition tables and performance monitoring."""

    def __init__(self, board: Board) -> None:
        # 🟢 SINGLE source of truth: no cloning
        self.board = board
        self.occupancy = OccupancyCache(board)
        self.piece_cache = PieceCache(board)  # 👈 NEW
        self._effect: Dict[str, Any] = {}
        self._init_effects()

        # Performance monitoring
        self.performance_monitor = CachePerformanceMonitor()

        # Zobrist hashing for transposition tables
        self._zobrist = ZobristHashing()
        self._current_zobrist_hash = self._zobrist.compute_hash(board, Color.WHITE)

        # Create optimized move cache but defer rebuild
        self._move_cache: Optional[OptimizedMoveCache] = None

        # Cache configuration
        self._tt_size_mb = 512  # 512MB transposition table
        self._enable_parallel = True
        self._enable_vectorization = True
        self._cache_stats_interval = 1000  # Log stats every N moves

        # Move counter for periodic stats
        self._move_counter = 0

    def initialise(self, current: Color) -> None:
        """Initialize with current player and create optimized move cache."""
        self._current_zobrist_hash = self._zobrist.compute_hash(self.board, current)
        self._move_cache = create_optimized_move_cache(self.board, current, self)

        # Log initial stats
        self._log_cache_stats("initialization")

    def _init_effects(self) -> None:
        """Initialize all effect caches."""
        self._effect = {
            "freeze":          FreezeCache(),
            "movement_buff":   MovementBuffCache(),
            "movement_debuff": MovementDebuffCache(),
            "black_hole_suck": BlackHoleSuckCache(),
            "white_hole_push": WhiteHolePushCache(),
            "trailblaze":      TrailblazeCache(),
            "behind":          BehindCache(),
            "armour":          ArmourCache(),
            "geomancy":        GeomancyCache(),
            "archery":         ArcheryCache(),
            "share_square":    ShareSquareCache(),
        }

    # --------------------------------------------------------------------------
    # ADVANCED MOVE APPLICATION WITH TRANSPOSITION TABLE SUPPORT
    # --------------------------------------------------------------------------
    def apply_move(self, mv: Move, mover: Color, current_ply: int = 0) -> None:
        """Apply move with full transposition table and performance monitoring."""
        start_time = time.time()

        try:
            # 1. Validate move on CURRENT board state
            from_piece = self.piece_cache.get(mv.from_coord)
            if from_piece is None:
                raise AssertionError(f"Illegal move: {mv} — no piece at {mv.from_coord}")

            # 2. Capture pre-move state
            to_piece = self.piece_cache.get(mv.to_coord)
            captured_piece = None

            if getattr(mv, "is_capture", False):
                captured_type = getattr(mv, "captured_piece", None)
                if captured_type is not None:
                    captured_piece = Piece(mover.opposite(), captured_type)

            # 3. Update Zobrist hash incrementally
            self._current_zobrist_hash = self._zobrist.update_hash_move(
                self._current_zobrist_hash, mv, from_piece, captured_piece
            )

            # 4. Apply move to shared board
            self.board.apply_move(mv)
            self.occupancy.rebuild(self.board)
            self.piece_cache.rebuild(self.board)

            # 5. Determine affected caches
            affected_caches = self._get_affected_caches(
                mv, mover, from_piece, to_piece, captured_piece
            )

            # 6. Update affected effect caches
            self._update_effect_caches(mv, mover, affected_caches, current_ply)

            # 7. Update optimized move cache
            if self._move_cache:
                self._move_cache.apply_move(mv, mover)

                # Store in transposition table if this is a significant move
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

            # 8. Performance monitoring
            duration = time.time() - start_time
            self.performance_monitor.record_move_apply_time(duration)
            self.performance_monitor.record_event(CacheEventType.MOVE_APPLIED, {
                'move': str(mv),
                'color': mover.name,
                'duration_ms': duration * 1000,
                'affected_caches': list(affected_caches)
            })

            # 9. Periodic stats logging
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
        """Undo move with full cache restoration and hash updates."""
        start_time = time.time()

        try:
            # Restore Zobrist hash (XOR is its own inverse)
            piece = self.piece_cache.get(mv.to_coord)
            captured_piece = None

            if getattr(mv, "is_capture", False):
                captured_type = getattr(mv, "captured_ptype", None)
                if captured_type is not None:
                    captured_piece = Piece(mover.opposite(), captured_type)

            if piece:
                self._current_zobrist_hash = self._zobrist.update_hash_move(
                    self._current_zobrist_hash, mv, piece, captured_piece
                )

            # Apply undo to board
            self._undo_move_optimized(mv, mover)

            # Update occupancy
            self.occupancy.rebuild(self.board)
            self.piece_cache.rebuild(self.board)

            # Update effect caches
            affected_caches = self._get_affected_caches_for_undo(mv, mover)
            self._update_effect_caches_for_undo(mv, mover, affected_caches, current_ply)

            # Update move cache
            if self._move_cache:
                self._move_cache.undo_move(mv, mover)

            # Performance monitoring
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

    def _undo_move_optimized(self, mv: Move, color: Color) -> None:
        """Optimized undo implementation."""
        # Handle captures
        if getattr(mv, "is_capture", False):
            captured_type = getattr(mv, "captured_ptype", None)
            if captured_type is not None:
                captured_color = color.opposite()
                self.board.set_piece(mv.to_coord, Piece(captured_color, captured_type))

        # Move piece back
        piece = self.piece_cache.get(mv.to_coord)  # Fixed from undefined coord
        if piece:
            self.board.set_piece(mv.from_coord, piece)
            self.board.set_piece(mv.to_coord, None)

        # Handle promotions
        if getattr(mv, "is_promotion", False) and piece:
            self.board.set_piece(mv.from_coord, Piece(piece.color, PieceType.PAWN))

    def _should_store_in_tt(self, mv: Move, piece: Piece) -> bool:
        """Determine if move should be stored in transposition table."""
        # Store significant moves: captures, promotions, checks
        return (getattr(mv, 'is_capture', False) or
                getattr(mv, 'is_promotion', False) or
                piece.ptype in {PieceType.KING, PieceType.QUEEN, PieceType.ROOK})

    def _get_affected_caches(self, mv: Move, mover: Color,
                           from_piece: Optional[Piece], to_piece: Optional[Piece],
                           captured_piece: Optional[Piece]) -> Set[str]:
        """Determine affected caches using dependency analysis."""
        affected = {"trailblaze"}  # trailblaze always updates

        pieces_to_check = [p for p in [from_piece, to_piece, captured_piece] if p is not None]

        # Add pieces that might be affected by the move
        for piece in pieces_to_check:
            ptype = piece.ptype
            if ptype == PieceType.FREEZE_AURA:
                affected.add("freeze")
            elif ptype == PieceType.BLACK_HOLE:
                affected.add("black_hole_suck")
            elif ptype == PieceType.WHITE_HOLE:
                affected.add("white_hole_push")
            elif ptype == PieceType.GEOMANCER:
                affected.add("geomancy")
            elif ptype == PieceType.ARCHER:
                affected.add("archery")
            elif ptype == PieceType.WALL:
                affected.add("armoured")
            elif ptype == PieceType.TRAILBLAZER:
                affected.add("trailblaze")
            elif ptype in {PieceType.SPEEDER, PieceType.XZQUEEN, PieceType.YZQUEEN, PieceType.XYQUEEN}:
                affected.add("movement_buff")
            elif ptype in {PieceType.SLOWER, PieceType.CONESLIDER}:
                affected.add("movement_debuff")
            elif ptype == PieceType.KNIGHT:
                affected.add("share_square")

        # Add caches that might be affected by position changes
        self._add_position_dependent_caches(affected, mv.from_coord)
        self._add_position_dependent_caches(affected, mv.to_coord)

        return affected

    def _add_position_dependent_caches(self, affected: Set[str], pos: Tuple[int, int, int]) -> None:
        """Add caches that depend on specific positions."""
        # Check for effect pieces near the position
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == dy == dz == 0:
                        continue

                    check_pos = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
                    if (0 <= check_pos[0] < 9 and 0 <= check_pos[1] < 9 and 0 <= check_pos[2] < 9):
                        piece = self.piece_cache.get(coord)
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
        """Get affected caches for undo operation."""
        # Similar to regular affected caches but considers undo state
        return self._get_affected_caches(mv, mover, None, None, None)

    def _update_effect_caches(self, mv: Move, mover: Color,
                            affected_caches: Set[str], current_ply: int) -> None:
        """Update effect caches with error handling."""
        for name in affected_caches:
            try:
                cache = self._effect[name]
                if name == "geomancy":
                    cache.apply_move(mv, mover, current_ply, self.board)
                elif name in ("archery", "black_hole_suck", "armoured", "freeze",
                              "movement_buff", "movement_debuff", "share_square",
                              "trailblaze", "white_hole_push"):
                    cache.apply_move(mv, mover, self.board)
                else:
                    cache.apply_move(mv, mover)
            except Exception as e:
                # Log error but don't crash - effect caches are non-critical
                self.performance_monitor.record_event(CacheEventType.CACHE_ERROR, {
                    'error': f"Effect cache {name} update failed: {str(e)}",
                    'move': str(mv)
                })

    def _update_effect_caches_for_undo(self, mv: Move, mover: Color,
                                     affected_caches: Set[str], current_ply: int) -> None:
        """Update effect caches for undo with proper error handling."""
        for name in affected_caches:
            try:
                cache = self._effect[name]
                if hasattr(cache, 'undo_move'):
                    if name == "geomancy":
                        cache.undo_move(mv, mover, current_ply, self.board)
                    elif hasattr(cache, '_board'):
                        cache.undo_move(mv, mover, self.board)
                    else:
                        cache.undo_move(mv, mover)
            except Exception as e:
                self.performance_monitor.record_event(CacheEventType.CACHE_ERROR, {
                    'error': f"Effect cache {name} undo failed: {str(e)}",
                    'move': str(mv)
                })

    # --------------------------------------------------------------------------
    # ADVANCED TRANSPOSITION TABLE INTERFACE
    # --------------------------------------------------------------------------
    def probe_transposition_table(self, hash_value: int) -> Optional[TTEntry]:
        """Probe transposition table for cached evaluation."""
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
        """Store evaluation in transposition table."""
        if self._move_cache:
            self._move_cache.store_evaluation(hash_value, depth, score, node_type, best_move)

    def get_current_zobrist_hash(self) -> int:
        """Get current Zobrist hash."""
        return self._current_zobrist_hash

    # --------------------------------------------------------------------------
    # PROPERTIES AND HELPERS - ENHANCED WITH PERFORMANCE MONITORING
    # --------------------------------------------------------------------------
    @property
    def move(self) -> OptimizedMoveCache:
        """Get the optimized move cache."""
        if self._move_cache is None:
            raise RuntimeError("MoveCache not initialized. Call initialise() first.")
        return self._move_cache

    def legal_moves(self, color: Color) -> List[Move]:
        """Get legal moves with performance monitoring."""
        start_time = time.time()

        moves = self.move.legal_moves(color)

        duration = time.time() - start_time
        self.performance_monitor.record_legal_move_generation_time(duration)

        return moves

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        base_stats = self.performance_monitor.get_performance_stats()

        # Add move cache specific stats
        if self._move_cache:
            move_cache_stats = self._move_cache.get_stats()
            base_stats.update({
                'move_cache_stats': move_cache_stats,
                'zobrist_hash': self._current_zobrist_hash,
                'tt_size_mb': self._tt_size_mb,
                'enable_parallel': self._enable_parallel,
                'enable_vectorization': self._enable_vectorization
            })

        # Add effect cache counts
        base_stats['effect_caches'] = {
            name: {'type': type(cache).__name__}
            for name, cache in self._effect.items()
        }

        return base_stats

    def get_optimization_suggestions(self) -> List[str]:
        """Get optimization suggestions from performance monitor."""
        suggestions = self.performance_monitor.get_optimization_suggestions()

        # Add cache-specific suggestions
        stats = self.get_cache_stats()
        if 'move_cache_stats' in stats:
            move_stats = stats['move_cache_stats']
            if move_stats.get('simple_move_cache_size', 0) < 50:
                suggestions.append("Simple move cache underutilized - consider more simple move patterns")

            if stats['avg_legal_gen_time_ms'] > 5.0:
                suggestions.append("Slow move generation. Consider vectorization or better caching strategies.")

        return suggestions

    def _log_cache_stats(self, context: str) -> None:
        """Log cache statistics for debugging."""
        stats = self.get_cache_stats()
        print(f"[CacheManager] {context} stats:")
        print(f"  TT Hit Rate: {stats['tt_hit_rate']:.3f}")
        print(f"  Avg Move Apply Time: {stats['avg_move_apply_time_ms']:.2f}ms")
        print(f"  Avg Legal Move Time: {stats['avg_legal_gen_time_ms']:.2f}ms")

        suggestions = self.get_optimization_suggestions()
        if suggestions:
            print(f"  Optimization Suggestions: {len(suggestions)}")
            for suggestion in suggestions[:3]:  # Show top 3
                print(f"    - {suggestion}")

    # --------------------------------------------------------------------------
    # EFFECT CACHE INTERFACE METHODS - ENHANCED
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
        return self._effect["armoured"].can_capture(attacker_sq, wall_sq, controller)

    def pieces_at(self, sq: Tuple[int, int, int]) -> List['Piece']:
        return self._effect["share_square"].pieces_at(sq)

    def top_piece(self, sq: Tuple[int, int, int]) -> Optional['Piece']:
        return self._effect["share_square"].top_piece(sq)

    # --------------------------------------------------------------------------
    # CONFIGURATION METHODS
    # --------------------------------------------------------------------------
    def configure_transposition_table(self, size_mb: int) -> None:
        """Configure transposition table size."""
        self._tt_size_mb = size_mb
        if self._move_cache:
            # Recreate with new size
            current_color = self._move_cache._current
            self._move_cache = create_optimized_move_cache(self.board, current_color, self)

    def set_parallel_processing(self, enabled: bool) -> None:
        """Enable/disable parallel move generation."""
        self._enable_parallel = enabled
        if self._move_cache:
            self._move_cache._enable_parallel = enabled

    def set_vectorization(self, enabled: bool) -> None:
        """Enable/disable vectorized operations."""
        self._enable_vectorization = enabled
        if self._move_cache:
            self._move_cache._enable_vectorization = enabled

    # --------------------------------------------------------------------------
    # UTILITY METHODS
    # --------------------------------------------------------------------------
    def clear_all_caches(self) -> None:
        """Clear all caches and reset statistics."""
        # Clear move cache
        if self._move_cache:
            self._move_cache._simple_move_cache.clear()
            self._move_cache._vectorized_cache.clear()
            self._move_cache._legal_per_piece.clear()
            self._move_cache._rebuild_color_lists()

        # Clear effect caches
        for cache in self._effect.values():
            if hasattr(cache, 'clear'):
                cache.clear()

        # Clear occupancy cache
        self.occupancy.rebuild(self.board)

        # Reset performance monitor
        self.performance_monitor = CachePerformanceMonitor()
        self.performance_monitor.record_event(CacheEventType.CACHE_CLEARED, {})

        # Reset move counter
        self._move_counter = 0

    def export_cache_state(self) -> Dict[str, Any]:
        """Export current cache state for analysis."""
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
# FACTORY FUNCTION FOR BACKWARD COMPATIBILITY
# ==============================================================================

def get_cache_manager(board: Board, current: Color) -> OptimizedCacheManager:
    """Create and initialize a new OptimizedCacheManager for the given board and player."""
    cache = OptimizedCacheManager(board)
    cache.initialise(current)
    return cache

# ==============================================================================
# BACKWARD COMPATIBILITY LAYER
# ==============================================================================

class CacheManager(OptimizedCacheManager):
    """Backward compatibility wrapper for the original CacheManager."""

    def __init__(self, board: Board) -> None:
        super().__init__(board)

    # Alias methods for backward compatibility
    def sync_board(self, new_board: Board) -> None:
        """Deprecated - use new board reference instead."""
        import warnings
        warnings.warn("sync_board is deprecated. Create a new CacheManager instead.", DeprecationWarning)
        self.board = new_board
        self.occupancy.rebuild(new_board)
        if self._move_cache:
            self._move_cache._board = new_board

    def replace_board(self, new_board: Board) -> None:
        """Deprecated - use new board reference instead."""
        import warnings
        warnings.warn("replace_board is deprecated. Create a new CacheManager instead.", DeprecationWarning)
        self.board = new_board
        self.occupancy.rebuild(new_board)
        if self._move_cache:
            self._move_cache._board = new_board
