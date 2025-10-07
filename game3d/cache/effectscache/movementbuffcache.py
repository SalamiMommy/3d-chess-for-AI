"""Optimized incremental cache for Movement-Buff squares with performance improvements."""

from __future__ import annotations
from typing import Set, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import weakref

from game3d.pieces.enums import Color
from game3d.effects.auras.movementbuff import buffed_squares
from game3d.movement.movepiece import Move
from game3d.pieces.piece import Piece

# ==============================================================================
# OPTIMIZATION CONSTANTS
# ==============================================================================

@dataclass
class BuffEffect:
    """Represents a movement buff effect."""
    source_square: Tuple[int, int, int]
    affected_squares: Set[Tuple[int, int, int]]
    priority: int
    buff_type: str  # Type of buff: speed, range, etc.

class BuffPriority(Enum):
    """Priority levels for buff effects."""
    HIGH = 3    # Direct aura effects
    MEDIUM = 2  # Overlapping auras
    LOW = 1     # Indirect effects

# ==============================================================================
# OPTIMIZED MOVEMENT BUFF CACHE
# ==============================================================================

class OptimizedMovementBuffCache:
    """Optimized incremental cache for Movement-Buff squares with smart updates."""

    __slots__ = (
        "_buffed", "_buff_sources", "_affected_squares",
        "_board_ref", "_last_board_hash", "_dirty_flags", "_source_tracking",
        "_buff_types", "_cache_manager"
    )

    def __init__(self, board: Optional[Board] = None, cache_manager=None) -> None:
        # Core buffed squares
        self._buffed: Dict[Color, Set[Tuple[int, int, int]]] = {
            Color.WHITE: set(),
            Color.BLACK: set(),
        }

        # Buff source tracking - which pieces are causing buffs
        self._buff_sources: Dict[Color, Set[Tuple[int, int, int]]] = {
            Color.WHITE: set(),  # Squares with buff-causing pieces
            Color.BLACK: set(),
        }

        # Squares affected by buff sources (for incremental updates)
        self._affected_squares: Dict[Color, Set[Tuple[int, int, int]]] = {
            Color.WHITE: set(),
            Color.BLACK: set(),
        }

        # Source tracking for each buffed square
        self._source_tracking: Dict[Color, Dict[Tuple[int, int, int], Set[Tuple[int, int, int]]]] = {
            Color.WHITE: {},  # Maps buffed square -> set of source squares
            Color.BLACK: {},
        }

        # Buff type tracking
        self._buff_types: Dict[Color, Dict[Tuple[int, int, int], str]] = {
            Color.WHITE: {},  # Maps buffed square -> buff type
            Color.BLACK: {},
        }

        # Board reference and change tracking
        self._board_ref: Optional[weakref.ref] = weakref.ref(board) if board else None
        self._last_board_hash: int = 0
        self._dirty_flags: Dict[str, bool] = {
            'sources': True,
            'buffed': True,
        }

        # Cache manager reference
        self._cache_manager = cache_manager

        if board:
            self._full_rebuild(board)

    # ---------- PUBLIC INTERFACE ----------
    def is_buffed(self, sq: Tuple[int, int, int], friendly_color: Color) -> bool:
        """Check if square is buffed for the given color."""
        if self._dirty_flags['buffed']:
            self._incremental_rebuild()
        return sq in self._buffed[friendly_color]

    def get_buff_sources(self, sq: Tuple[int, int, int], friendly_color: Color) -> Set[Tuple[int, int, int]]:
        """Get source squares that are buffing this square."""
        if self._dirty_flags['buffed']:
            self._incremental_rebuild()
        return self._source_tracking[friendly_color].get(sq, set())

    def get_buffed_squares(self, controller: Color) -> Set[Tuple[int, int, int]]:
        """Get all buffed squares for the given color."""
        if self._dirty_flags['buffed']:
            self._incremental_rebuild()
        return self._buffed[controller].copy()

    def get_buff_type(self, sq: Tuple[int, int, int], friendly_color: Color) -> Optional[str]:
        """Get the type of buff affecting this square."""
        if self._dirty_flags['buffed']:
            self._incremental_rebuild()
        return self._buff_types[friendly_color].get(sq)

    def get_buff_source_count(self, controller: Color) -> int:
        """Get number of buff sources for controller."""
        return len(self._buff_sources[controller])

    # ---------- MOVE HANDLING ----------
    def apply_move(self, mv: Move, mover: Color, board: Board) -> None:
        """Smart incremental update based on move impact."""
        # Quick check if move affects buff sources
        if self._move_affects_buff_sources(mv, board):
            self._incremental_update(mv, mover, board)
        else:
            # Move doesn't affect buff sources, no rebuild needed
            pass

    def undo_move(self, mv: Move, mover: Color, board: Board) -> None:
        """Smart undo with minimal rebuilding."""
        self._incremental_update(mv, mover, board)

    # ---------- INCREMENTAL UPDATES ----------
    def _move_affects_buff_sources(self, mv: Move, board: Board) -> bool:
        """Check if move affects buff sources (piece moved, captured, or near buff sources)."""
        # Check if moved piece is a buff source
        if self._cache_manager:
            moved_piece = self._cache_manager.piece_cache.get(mv.from_coord)
        else:
            # Fallback to board method if cache manager not available
            moved_piece = board.get_piece(mv.from_coord)

        if moved_piece and self._is_buff_source(moved_piece):
            return True

        # Check if destination had a buff source (captured)
        if self._cache_manager:
            dest_piece = self._cache_manager.piece_cache.get(mv.to_coord)
        else:
            # Fallback to board method if cache manager not available
            dest_piece = board.get_piece(mv.to_coord)

        if dest_piece and self._is_buff_source(dest_piece):
            return True

        # Check if move affects squares near buff sources
        for color in (Color.WHITE, Color.BLACK):
            if mv.from_coord in self._affected_squares[color]:
                return True
            if mv.to_coord in self._affected_squares[color]:
                return True

        return False

    def _is_buff_source(self, piece: Piece) -> bool:
        """Check if piece can cause movement buffs."""
        # This should match the logic in movementbuff.buffed_squares
        # Common buff sources: speed aura pieces, command pieces, etc.
        return hasattr(piece, 'causes_movement_buff') and piece.causes_movement_buff

    def _incremental_update(self, mv: Move, mover: Color, board: Board) -> None:
        """Smart incremental update instead of full rebuild."""
        current_hash = board.byte_hash()

        # If board hash hasn't changed significantly, do minimal update
        if abs(current_hash - self._last_board_hash) < 1000:
            self._minimal_rebuild(mv, mover, board)
        else:
            # Significant change, full rebuild
            self._full_rebuild(board)

        self._last_board_hash = current_hash

    def _minimal_rebuild(self, mv: Move, mover: Color, board: Board) -> None:
        """Minimal rebuild affecting only changed areas."""
        # Update buff sources
        old_sources = self._buff_sources.copy()
        self._update_buff_sources(board)

        # Only rebuild buffed squares for colors that had changes
        for color in (Color.WHITE, Color.BLACK):
            if old_sources[color] != self._buff_sources[color]:
                self._rebuild_buffed_squares_for_color(board, color)
                self._dirty_flags['buffed'] = True

    def _update_buff_sources(self, board: Board) -> None:
        """Update buff source tracking."""
        for color in (Color.WHITE, Color.BLACK):
            old_sources = self._buff_sources[color].copy()
            self._buff_sources[color].clear()

            # Find all buff sources of this color
            for coord, piece in board.list_occupied():
                if piece.color == color and self._is_buff_source(piece):
                    self._buff_sources[color].add(coord)

            # Mark dirty if sources changed
            if old_sources != self._buff_sources[color]:
                self._dirty_flags['sources'] = True

    def _rebuild_buffed_squares_for_color(self, board: Board, color: Color) -> None:
        """Rebuild buffed squares only for specific color."""
        # Clear old data
        self._buffed[color].clear()
        self._affected_squares[color].clear()
        self._source_tracking[color].clear()
        self._buff_types[color].clear()

        # Rebuild from scratch for this color
        if self._cache_manager:
            buffed = buffed_squares(board, color, self._cache_manager)
        else:
            # Fallback if cache manager not available
            buffed = buffed_squares(board, color, None)

        # Build buffed set and track affected squares
        for sq in buffed:
            self._buffed[color].add(sq)
            self._affected_squares[color].add(sq)
            self._source_tracking[color][sq] = set()
            self._buff_types[color][sq] = self._determine_buff_type(board, sq, color)

        # Build source tracking
        self._build_source_tracking(board, color)

    def _determine_buff_type(self, board: Board, sq: Tuple[int, int, int], color: Color) -> str:
        """Determine the type of buff affecting this square."""
        # This would analyze the board to determine buff type
        # For now, return default
        return "speed"

    def _build_source_tracking(self, board: Board, color: Color) -> None:
        """Build source tracking for each buffed square."""
        # This can be expanded to track which specific sources affect which squares
        for source_sq in self._buff_sources[color]:
            # Determine which squares this source affects
            affected = self._get_squares_affected_by_source(board, source_sq, color)
            for affected_sq in affected:
                if affected_sq in self._source_tracking[color]:
                    self._source_tracking[color][affected_sq].add(source_sq)

    def _get_squares_affected_by_source(self, board: Board, source_sq: Tuple[int, int, int], source_color: Color) -> Set[Tuple[int, int, int]]:
        """Get squares affected by a specific buff source."""
        # This would need to be implemented based on the actual buff mechanics
        # For now, return empty set as placeholder
        return set()

    # ---------- FULL REBUILD (FALLBACK) ----------
    def _full_rebuild(self, board: Board) -> None:
        """Complete rebuild of all data structures."""
        for color in (Color.WHITE, Color.BLACK):
            self._rebuild_buffed_squares_for_color(board, color)

        self._dirty_flags['sources'] = False
        self._dirty_flags['buffed'] = False

    def _incremental_rebuild(self) -> None:
        """Smart incremental rebuild based on dirty flags."""
        board = self._get_board()
        if board is None:
            return

        if self._dirty_flags['sources']:
            self._update_buff_sources(board)

        if self._dirty_flags['buffed']:
            for color in (Color.WHITE, Color.BLACK):
                if self._dirty_flags['sources'] or len(self._buffed[color]) == 0:
                    self._rebuild_buffed_squares_for_color(board, color)

        # Clear dirty flags
        for flag in self._dirty_flags:
            self._dirty_flags[flag] = False

    # ---------- UTILITY METHODS ----------
    def _get_board(self) -> Optional[Board]:
        """Get board from weak reference."""
        if self._board_ref is None:
            return None
        board = self._board_ref()
        if board is None:
            self._board_ref = None
        return board

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics for performance monitoring."""
        board = self._get_board()
        return {
            'buff_sources_white': len(self._buff_sources[Color.WHITE]),
            'buff_sources_black': len(self._buff_sources[Color.BLACK]),
            'buffed_squares_white': len(self._buffed[Color.WHITE]),
            'buffed_squares_black': len(self._buffed[Color.BLACK]),
            'affected_squares_white': len(self._affected_squares[Color.WHITE]),
            'affected_squares_black': len(self._affected_squares[Color.BLACK]),
            'dirty_flags': self._dirty_flags.copy(),
            'board_hash': board.byte_hash() if board else 0,
        }

    def clear(self) -> None:
        """Clear all cached data."""
        for color in (Color.WHITE, Color.BLACK):
            self._buffed[color].clear()
            self._buff_sources[color].clear()
            self._affected_squares[color].clear()
            self._source_tracking[color].clear()
            self._buff_types[color].clear()

        self._last_board_hash = 0
        for flag in self._dirty_flags:
            self._dirty_flags[flag] = True

# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_optimized_movement_buff_cache(board: Optional[Board] = None, cache_manager=None) -> OptimizedMovementBuffCache:
    """Factory function for creating optimized cache."""
    return OptimizedMovementBuffCache(board, cache_manager)

# ==============================================================================
# BACKWARD COMPATIBILITY
# ==============================================================================

class MovementBuffCache(OptimizedMovementBuffCache):
    """Backward compatibility wrapper."""

    def __init__(self, cache_manager=None) -> None:
        super().__init__(None, cache_manager)  # Don't pass board to avoid immediate rebuild

# ------------------------------------------------------------------
# singleton
# ------------------------------------------------------------------
_buff_cache: Optional[MovementBuffCache] = None

def init_movement_buff_cache(cache_manager=None) -> None:
    global _buff_cache
    _buff_cache = MovementBuffCache(cache_manager)

def get_movement_buff_cache() -> MovementBuffCache:
    if _buff_cache is None:
        raise RuntimeError("MovementBuffCache not initialised")
    return _buff_cache
