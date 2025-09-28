#game3d/cache/effects/freezcache.py
"""Optimized incremental cache for frozen enemy squares with performance improvements."""
#game3d/cache/effects/freezcache.py
from __future__ import annotations
from typing import Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import weakref

from game3d.pieces.enums import Color
from game3d.board.board import Board
from game3d.effects.auras.freeze import frozen_squares
from game3d.movement.movepiece import Move
from game3d.pieces.piece import Piece

# ==============================================================================
# OPTIMIZATION CONSTANTS
# ==============================================================================

@dataclass
class FreezeEffect:
    """Represents a freeze effect."""
    source_square: Tuple[int, int, int]
    affected_squares: Set[Tuple[int, int, int]]
    priority: int
    duration: int  # Turns remaining

class FreezePriority(Enum):
    """Priority levels for freeze effects."""
    HIGH = 3    # Direct freeze aura effects
    MEDIUM = 2  # Chain freeze effects
    LOW = 1     # Residual freeze effects

# ==============================================================================
# OPTIMIZED FREEZE CACHE
# ==============================================================================

class OptimizedFreezeCache:
    """Optimized incremental cache for frozen enemy squares with smart updates."""

    __slots__ = (
        "_frozen", "_freeze_sources", "_affected_squares",
        "_board_ref", "_last_board_hash", "_dirty_flags", "_source_tracking",
        "_freeze_durations"
    )

    def __init__(self, board: Optional[Board] = None) -> None:
        # Core frozen squares
        self._frozen: Dict[Color, Set[Tuple[int, int, int]]] = {
            Color.WHITE: set(),
            Color.BLACK: set(),
        }

        # Freeze source tracking - which pieces are causing freezes
        self._freeze_sources: Dict[Color, Set[Tuple[int, int, int]]] = {
            Color.WHITE: set(),  # Squares with freeze-causing pieces (enemy POV)
            Color.BLACK: set(),
        }

        # Squares affected by freeze sources (for incremental updates)
        self._affected_squares: Dict[Color, Set[Tuple[int, int, int]]] = {
            Color.WHITE: set(),
            Color.BLACK: set(),
        }

        # Source tracking for each frozen square
        self._source_tracking: Dict[Color, Dict[Tuple[int, int, int], Set[Tuple[int, int, int]]]] = {
            Color.WHITE: {},  # Maps frozen square -> set of source squares
            Color.BLACK: {},
        }

        # Freeze duration tracking
        self._freeze_durations: Dict[Color, Dict[Tuple[int, int, int], int]] = {
            Color.WHITE: {},  # Maps frozen square -> remaining duration
            Color.BLACK: {},
        }

        # Board reference and change tracking
        self._board_ref: Optional[weakref.ref] = weakref.ref(board) if board else None
        self._last_board_hash: int = 0
        self._dirty_flags: Dict[str, bool] = {
            'sources': True,
            'frozen': True,
        }

        if board:
            self._full_rebuild(board)

    # ---------- PUBLIC INTERFACE ----------
    def is_frozen(self, sq: Tuple[int, int, int], victim: Color) -> bool:
        """Check if square is frozen for the given color."""
        if self._dirty_flags['frozen']:
            self._incremental_rebuild()
        return sq in self._frozen[victim]

    def get_freeze_sources(self, sq: Tuple[int, int, int], victim: Color) -> Set[Tuple[int, int, int]]:
        """Get source squares that are freezing this square."""
        if self._dirty_flags['frozen']:
            self._incremental_rebuild()
        return self._source_tracking[victim].get(sq, set())

    def get_frozen_squares(self, victim: Color) -> Set[Tuple[int, int, int]]:
        """Get all frozen squares for the given color."""
        if self._dirty_flags['frozen']:
            self._incremental_rebuild()
        return self._frozen[victim].copy()

    def get_freeze_duration(self, sq: Tuple[int, int, int], victim: Color) -> int:
        """Get remaining freeze duration for this square."""
        if self._dirty_flags['frozen']:
            self._incremental_rebuild()
        return self._freeze_durations[victim].get(sq, 0)

    def get_freeze_source_count(self, controller: Color) -> int:
        """Get number of freeze sources for controller."""
        return len(self._freeze_sources[controller])

    # ---------- MOVE HANDLING ----------
    def apply_move(self, mv: Move, mover: Color, board: Board) -> None:
        """Smart incremental update based on move impact."""
        # Quick check if move affects freeze sources
        if self._move_affects_freeze_sources(mv, board):
            self._incremental_update(mv, mover, board)
        else:
            # Move doesn't affect freeze sources, no rebuild needed
            pass

    def undo_move(self, mv: Move, mover: Color, board: Board) -> None:
        """Smart undo with minimal rebuilding."""
        self._incremental_update(mv, mover, board)

    # ---------- INCREMENTAL UPDATES ----------
    def _move_affects_freeze_sources(self, mv: Move, board: Board) -> bool:
        """Check if move affects freeze sources (piece moved, captured, or near freeze sources)."""
        # Check if moved piece is a freeze source
        moved_piece = cache.piece_cache.get(mv.from_coord)
        if moved_piece and self._is_freeze_source(moved_piece):
            return True

        # Check if destination had a freeze source (captured)
        dest_piece = cache.piece_cache.get(mv.to_coord)
        if dest_piece and self._is_freeze_source(dest_piece):
            return True

        # Check if move affects squares near freeze sources
        for color in (Color.WHITE, Color.BLACK):
            if mv.from_coord in self._affected_squares[color]:
                return True
            if mv.to_coord in self._affected_squares[color]:
                return True

        return False

    def _is_freeze_source(self, piece: Piece) -> bool:
        """Check if piece can cause freezes."""
        # This should match the logic in freeze.frozen_squares
        # Common freeze sources: ice pieces, aura pieces, etc.
        return hasattr(piece, 'causes_freeze') and piece.causes_freeze

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
        # Update freeze sources
        old_sources = self._freeze_sources.copy()
        self._update_freeze_sources(board)

        # Only rebuild frozen squares for colors that had changes
        for color in (Color.WHITE, Color.BLACK):
            if old_sources[color] != self._freeze_sources[color]:
                self._rebuild_frozen_squares_for_color(board, color)
                self._dirty_flags['frozen'] = True

    def _update_freeze_sources(self, board: Board) -> None:
        """Update freeze source tracking."""
        for color in (Color.WHITE, Color.BLACK):
            old_sources = self._freeze_sources[color].copy()
            self._freeze_sources[color].clear()

            # Find all freeze sources of this color (enemy perspective)
            enemy_color = Color.BLACK if color == Color.WHITE else Color.WHITE
            for coord, piece in board.list_occupied():
                if piece.color == enemy_color and self._is_freeze_source(piece):
                    self._freeze_sources[color].add(coord)

            # Mark dirty if sources changed
            if old_sources != self._freeze_sources[color]:
                self._dirty_flags['sources'] = True

    def _rebuild_frozen_squares_for_color(self, board: Board, color: Color) -> None:
        """Rebuild frozen squares only for specific color."""
        # Clear old data
        self._frozen[color].clear()
        self._affected_squares[color].clear()
        self._source_tracking[color].clear()
        self._freeze_durations[color].clear()

        # Rebuild from scratch for this color
        enemy_color = Color.BLACK if color == Color.WHITE else Color.WHITE
        frozen = frozen_squares(board, enemy_color)

        # Build frozen set and track affected squares
        for sq in frozen:
            self._frozen[color].add(sq)
            self._affected_squares[color].add(sq)
            self._source_tracking[color][sq] = set()
            self._freeze_durations[color][sq] = self._determine_freeze_duration(board, sq, enemy_color)

        # Build source tracking
        self._build_source_tracking(board, color)

    def _determine_freeze_duration(self, board: Board, sq: Tuple[int, int, int], source_color: Color) -> int:
        """Determine freeze duration for this square."""
        # This would analyze the board to determine duration
        # For now, return default
        return 1

    def _build_source_tracking(self, board: Board, color: Color) -> None:
        """Build source tracking for each frozen square."""
        # This can be expanded to track which specific sources affect which squares
        for source_sq in self._freeze_sources[color]:
            # Determine which squares this source affects
            affected = self._get_squares_affected_by_source(board, source_sq, color)
            for affected_sq in affected:
                if affected_sq in self._source_tracking[color]:
                    self._source_tracking[color][affected_sq].add(source_sq)

    def _get_squares_affected_by_source(self, board: Board, source_sq: Tuple[int, int, int], source_color: Color) -> Set[Tuple[int, int, int]]:
        """Get squares affected by a specific freeze source."""
        # This would need to be implemented based on the actual freeze mechanics
        # For now, return empty set as placeholder
        return set()

    # ---------- FULL REBUILD (FALLBACK) ----------
    def _full_rebuild(self, board: Board) -> None:
        """Complete rebuild of all data structures."""
        for color in (Color.WHITE, Color.BLACK):
            self._rebuild_frozen_squares_for_color(board, color)

        self._dirty_flags['sources'] = False
        self._dirty_flags['frozen'] = False

    def _incremental_rebuild(self) -> None:
        """Smart incremental rebuild based on dirty flags."""
        board = self._get_board()
        if board is None:
            return

        if self._dirty_flags['sources']:
            self._update_freeze_sources(board)

        if self._dirty_flags['frozen']:
            for color in (Color.WHITE, Color.BLACK):
                if self._dirty_flags['sources'] or len(self._frozen[color]) == 0:
                    self._rebuild_frozen_squares_for_color(board, color)

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
            'freeze_sources_white': len(self._freeze_sources[Color.WHITE]),
            'freeze_sources_black': len(self._freeze_sources[Color.BLACK]),
            'frozen_squares_white': len(self._frozen[Color.WHITE]),
            'frozen_squares_black': len(self._frozen[Color.BLACK]),
            'affected_squares_white': len(self._affected_squares[Color.WHITE]),
            'affected_squares_black': len(self._affected_squares[Color.BLACK]),
            'dirty_flags': self._dirty_flags.copy(),
            'board_hash': board.byte_hash() if board else 0,
        }

    def clear(self) -> None:
        """Clear all cached data."""
        for color in (Color.WHITE, Color.BLACK):
            self._frozen[color].clear()
            self._freeze_sources[color].clear()
            self._affected_squares[color].clear()
            self._source_tracking[color].clear()
            self._freeze_durations[color].clear()

        self._last_board_hash = 0
        for flag in self._dirty_flags:
            self._dirty_flags[flag] = True

# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_optimized_freeze_cache(board: Optional[Board] = None) -> OptimizedFreezeCache:
    """Factory function for creating optimized cache."""
    return OptimizedFreezeCache(board)

# ==============================================================================
# BACKWARD COMPATIBILITY
# ==============================================================================

class FreezeCache(OptimizedFreezeCache):
    """Backward compatibility wrapper."""

    def __init__(self) -> None:
        super().__init__(None)  # Don't pass board to avoid immediate rebuild


