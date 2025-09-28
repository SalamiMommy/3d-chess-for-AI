"""Optimized incremental cache for Movement-Debuff squares with performance improvements."""

from __future__ import annotations
from typing import Set, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import weakref

from game3d.pieces.enums import Color
from game3d.board.board import Board
from game3d.effects.auras.movementdebuff import debuffed_squares
from game3d.movement.movepiece import Move
from game3d.pieces.piece import Piece

# ==============================================================================
# OPTIMIZATION CONSTANTS
# ==============================================================================

@dataclass
class DebuffEffect:
    """Represents a movement debuff effect."""
    source_square: Tuple[int, int, int]
    affected_squares: Set[Tuple[int, int, int]]
    priority: int

class DebuffPriority(Enum):
    """Priority levels for debuff effects."""
    HIGH = 3    # Direct aura effects
    MEDIUM = 2  # Overlapping auras
    LOW = 1     # Indirect effects

# ==============================================================================
# OPTIMIZED MOVEMENT DEBUFF CACHE
# ==============================================================================

class OptimizedMovementDebuffCache:
    """Optimized incremental cache for Movement-Debuff squares with smart updates."""

    __slots__ = (
        "_debuffed", "_debuff_sources", "_affected_squares",
        "_board_ref", "_last_board_hash", "_dirty_flags", "_source_tracking"
    )

    def __init__(self, board: Optional[Board] = None) -> None:
        # Core debuffed squares
        self._debuffed: Dict[Color, Set[Tuple[int, int, int]]] = {
            Color.WHITE: set(),
            Color.BLACK: set(),
        }

        # Source tracking - which pieces are causing debuffs
        self._debuff_sources: Dict[Color, Set[Tuple[int, int, int]]] = {
            Color.WHITE: set(),  # Squares with debuff-causing pieces (enemy POV)
            Color.BLACK: set(),
        }

        # Squares affected by debuff sources (for incremental updates)
        self._affected_squares: Dict[Color, Set[Tuple[int, int, int]]] = {
            Color.WHITE: set(),
            Color.BLACK: set(),
        }

        # Source tracking for each debuffed square
        self._source_tracking: Dict[Color, Dict[Tuple[int, int, int], Set[Tuple[int, int, int]]]] = {
            Color.WHITE: {},  # Maps debuffed square -> set of source squares
            Color.BLACK: {},
        }

        # Board reference and change tracking
        self._board_ref: Optional[weakref.ref] = weakref.ref(board) if board else None
        self._last_board_hash: int = 0
        self._dirty_flags: Dict[str, bool] = {
            'sources': True,
            'debuffed': True,
        }

        if board:
            self._full_rebuild(board)

    # ---------- PUBLIC INTERFACE ----------
    def is_debuffed(self, sq: Tuple[int, int, int], victim_color: Color) -> bool:
        """Check if square is debuffed for the given color."""
        if self._dirty_flags['debuffed']:
            self._incremental_rebuild()
        return sq in self._debuffed[victim_color]

    def get_debuff_sources(self, sq: Tuple[int, int, int], victim_color: Color) -> Set[Tuple[int, int, int]]:
        """Get source squares that are debuffing this square."""
        if self._dirty_flags['debuffed']:
            self._incremental_rebuild()
        return self._source_tracking[victim_color].get(sq, set())

    def get_debuffed_squares(self, controller: Color) -> Set[Tuple[int, int, int]]:
        """Get all debuffed squares for the given color."""
        if self._dirty_flags['debuffed']:
            self._incremental_rebuild()
        return self._debuffed[controller].copy()

    def get_debuff_source_count(self, controller: Color) -> int:
        """Get number of debuff sources for controller."""
        return len(self._debuff_sources[controller])

    # ---------- MOVE HANDLING ----------
    def apply_move(self, mv: Move, mover: Color, board: Board) -> None:
        """Smart incremental update based on move impact."""
        # Quick check if move affects debuff sources
        if self._move_affects_debuff_sources(mv, board):
            self._incremental_update(mv, mover, board)
        else:
            # Move doesn't affect debuff sources, no rebuild needed
            pass

    def undo_move(self, mv: Move, mover: Color, board: Board) -> None:
        """Smart undo with minimal rebuilding."""
        self._incremental_update(mv, mover, board)

    # ---------- INCREMENTAL UPDATES ----------
    def _move_affects_debuff_sources(self, mv: Move, board: Board) -> bool:
        """Check if move affects debuff sources (piece moved, captured, or near debuff sources)."""
        # Check if moved piece is a debuff source
        moved_piece = cache.piece_cache.get(mv.from_coord)
        if moved_piece and self._is_debuff_source(moved_piece):
            return True

        # Check if destination had a debuff source (captured)
        dest_piece = cache.piece_cache.get(mv.to_coord)
        if dest_piece and self._is_debuff_source(dest_piece):
            return True

        # Check if move affects squares near debuff sources
        for color in (Color.WHITE, Color.BLACK):
            if mv.from_coord in self._affected_squares[color]:
                return True
            if mv.to_coord in self._affected_squares[color]:
                return True

        return False

    def _is_debuff_source(self, piece: Piece) -> bool:
        """Check if piece can cause movement debuffs."""
        # This should match the logic in movementdebuff.debuffed_squares
        # Common debuff sources: swamp pieces, aura pieces, etc.
        return hasattr(piece, 'causes_movement_debuff') and piece.causes_movement_debuff

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
        # Update debuff sources
        old_sources = self._debuff_sources.copy()
        self._update_debuff_sources(board)

        # Only rebuild debuffed squares for colors that had changes
        for color in (Color.WHITE, Color.BLACK):
            if old_sources[color] != self._debuff_sources[color]:
                self._rebuild_debuffed_squares_for_color(board, color)
                self._dirty_flags['debuffed'] = True

    def _update_debuff_sources(self, board: Board) -> None:
        """Update debuff source tracking."""
        for color in (Color.WHITE, Color.BLACK):
            old_sources = self._debuff_sources[color].copy()
            self._debuff_sources[color].clear()

            # Find all debuff sources of this color (enemy perspective)
            enemy_color = Color.BLACK if color == Color.WHITE else Color.WHITE
            for coord, piece in board.list_occupied():
                if piece.color == enemy_color and self._is_debuff_source(piece):
                    self._debuff_sources[color].add(coord)

            # Mark dirty if sources changed
            if old_sources != self._debuff_sources[color]:
                self._dirty_flags['sources'] = True

    def _rebuild_debuffed_squares_for_color(self, board: Board, color: Color) -> None:
        """Rebuild debuffed squares only for specific color."""
        # Clear old data
        self._debuffed[color].clear()
        self._affected_squares[color].clear()
        self._source_tracking[color].clear()

        # Rebuild from scratch for this color
        enemy_color = Color.BLACK if color == Color.WHITE else Color.WHITE
        debuffed = debuffed_squares(board, enemy_color)

        # Build debuffed set and track affected squares
        for sq in debuffed:
            self._debuffed[color].add(sq)
            self._affected_squares[color].add(sq)
            self._source_tracking[color][sq] = set()  # Will be populated if needed

        # Build source tracking (optional optimization)
        self._build_source_tracking(board, color)

    def _build_source_tracking(self, board: Board, color: Color) -> None:
        """Build source tracking for each debuffed square."""
        # This can be expanded to track which specific sources affect which squares
        # For now, just establish the framework
        enemy_color = Color.BLACK if color == Color.WHITE else Color.WHITE
        for source_sq in self._debuff_sources[color]:
            # Determine which squares this source affects
            # This would need coordination with the movementdebuff module
            affected = self._get_squares_affected_by_source(board, source_sq, enemy_color)
            for affected_sq in affected:
                if affected_sq in self._source_tracking[color]:
                    self._source_tracking[color][affected_sq].add(source_sq)

    def _get_squares_affected_by_source(self, board: Board, source_sq: Tuple[int, int, int], source_color: Color) -> Set[Tuple[int, int, int]]:
        """Get squares affected by a specific debuff source."""
        # This would need to be implemented based on the actual debuff mechanics
        # For now, return empty set as placeholder
        return set()

    # ---------- FULL REBUILD (FALLBACK) ----------
    def _full_rebuild(self, board: Board) -> None:
        """Complete rebuild of all data structures."""
        for color in (Color.WHITE, Color.BLACK):
            self._rebuild_debuffed_squares_for_color(board, color)

        self._dirty_flags['sources'] = False
        self._dirty_flags['debuffed'] = False

    def _incremental_rebuild(self) -> None:
        """Smart incremental rebuild based on dirty flags."""
        board = self._get_board()
        if board is None:
            return

        if self._dirty_flags['sources']:
            self._update_debuff_sources(board)

        if self._dirty_flags['debuffed']:
            for color in (Color.WHITE, Color.BLACK):
                if self._dirty_flags['sources'] or len(self._debuffed[color]) == 0:
                    self._rebuild_debuffed_squares_for_color(board, color)

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
            'debuff_sources_white': len(self._debuff_sources[Color.WHITE]),
            'debuff_sources_black': len(self._debuff_sources[Color.BLACK]),
            'debuffed_squares_white': len(self._debuffed[Color.WHITE]),
            'debuffed_squares_black': len(self._debuffed[Color.BLACK]),
            'affected_squares_white': len(self._affected_squares[Color.WHITE]),
            'affected_squares_black': len(self._affected_squares[Color.BLACK]),
            'dirty_flags': self._dirty_flags.copy(),
            'board_hash': board.byte_hash() if board else 0,
        }

    def clear(self) -> None:
        """Clear all cached data."""
        for color in (Color.WHITE, Color.BLACK):
            self._debuffed[color].clear()
            self._debuff_sources[color].clear()
            self._affected_squares[color].clear()
            self._source_tracking[color].clear()

        self._last_board_hash = 0
        for flag in self._dirty_flags:
            self._dirty_flags[flag] = True

# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_optimized_movement_debuff_cache(board: Optional[Board] = None) -> OptimizedMovementDebuffCache:
    """Factory function for creating optimized cache."""
    return OptimizedMovementDebuffCache(board)

# ==============================================================================
# BACKWARD COMPATIBILITY
# ==============================================================================

class MovementDebuffCache(OptimizedMovementDebuffCache):
    """Backward compatibility wrapper."""

    def __init__(self) -> None:
        super().__init__(None)  # Don't pass board to avoid immediate rebuild

# ------------------------------------------------------------------
# singleton
# ------------------------------------------------------------------
_debuff_cache: Optional[MovementDebuffCache] = None

def init_movement_debuff_cache() -> None:
    global _debuff_cache
    _debuff_cache = MovementDebuffCache()

def get_movement_debuff_cache() -> MovementDebuffCache:
    if _debuff_cache is None:
        raise RuntimeError("MovementDebuffCache not initialised")
    return _debuff_cache
