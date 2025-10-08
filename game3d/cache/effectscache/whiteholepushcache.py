"""Optimized incremental cache for White-Hole push map with performance improvements."""

from __future__ import annotations
from typing import Dict, Tuple, Optional, Set, List, Any
from dataclasses import dataclass
from enum import Enum
import weakref

from game3d.pieces.enums import Color, PieceType
from game3d.effects.auras.whiteholepush import push_candidates
from game3d.movement.movepiece import Move
from game3d.pieces.piece import Piece

# ==============================================================================
# OPTIMIZATION CONSTANTS
# ==============================================================================

@dataclass
class PushEffect:
    """Represents a white hole push effect."""
    from_square: Tuple[int, int, int]
    to_square: Tuple[int, int, int]
    affected_pieces: Set[Tuple[int, int, int]]  # Pieces that would be pushed
    priority: int  # Higher priority pushes override lower priority ones

class PushPriority(Enum):
    """Priority levels for push effects."""
    HIGH = 3    # Direct white hole pushes
    MEDIUM = 2  # Chain reaction pushes
    LOW = 1     # Indirect effects

# ==============================================================================
# OPTIMIZED WHITE HOLE PUSH CACHE
# ==============================================================================

class OptimizedWhiteHolePushCache:
    """Optimized incremental cache for White-Hole push map with smart updates."""

    __slots__ = (
        "_push_map", "_white_hole_positions", "_affected_squares",
        "_board_ref", "_last_board_hash", "_dirty_flags", "_push_chains",
        "_cache_manager"
    )

    def __init__(self, board: Optional[Board] = None, cache_manager=None) -> None:
        # Core push maps
        self._push_map: Dict[Color, Dict[Tuple[int, int, int], Tuple[int, int, int]]] = {
            Color.WHITE: {},
            Color.BLACK: {},
        }

        # White hole positions for quick lookup
        self._white_hole_positions: Dict[Color, Set[Tuple[int, int, int]]] = {
            Color.WHITE: set(),
            Color.BLACK: set(),
        }

        # Squares affected by white holes (for incremental updates)
        self._affected_squares: Dict[Color, Set[Tuple[int, int, int]]] = {
            Color.WHITE: set(),
            Color.BLACK: set(),
        }

        # Push chains (cascading effects)
        self._push_chains: Dict[Color, List[PushEffect]] = {
            Color.WHITE: [],
            Color.BLACK: [],
        }

        # Board reference and change tracking
        self._board_ref: Optional[weakref.ref] = weakref.ref(board) if board else None
        self._last_board_hash: int = 0
        self._dirty_flags: Dict[str, bool] = {
            'white_holes': True,
            'push_map': True,
            'chains': True,
        }

        # Cache manager reference
        self._cache_manager = cache_manager

        if board:
            self._full_rebuild(board)

    # ---------- PUBLIC INTERFACE ----------
    def push_map(self, controller: Color) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
        """Get push map for controller, rebuilding if necessary."""
        if self._dirty_flags['push_map']:
            self._incremental_rebuild()
        return self._push_map[controller]

    def get_push_targets(self, controller: Color, from_square: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Get all possible push targets from a square (including chains)."""
        if self._dirty_flags['chains']:
            self._rebuild_push_chains()

        targets = []
        main_target = self._push_map[controller].get(from_square)
        if main_target:
            targets.append(main_target)

        # Add chain reaction targets
        for effect in self._push_chains[controller]:
            if effect.from_square == from_square:
                targets.append(effect.to_square)

        return targets

    def is_affected_by_white_hole(self, square: Tuple[int, int, int], controller: Color) -> bool:
        """Quick check if square is affected by white holes."""
        return square in self._affected_squares[controller]

    def get_white_hole_count(self, controller: Color) -> int:
        """Get number of white holes for controller."""
        return len(self._white_hole_positions[controller])

    # ---------- MOVE HANDLING ----------
    def apply_move(self, mv: Move, mover: Color, board: Board) -> None:
        """Smart incremental update based on move impact."""
        # Quick check if move affects white holes
        if self._move_affects_white_holes(mv, board):
            self._incremental_update(mv, mover, board)
        else:
            # Move doesn't affect white holes, no rebuild needed
            pass

    def undo_move(self, mv: Move, mover: Color, board: Board) -> None:
        """Smart undo with minimal rebuilding."""
        # For undo, we need to check if the move affected white holes
        # This is more complex - for now, rebuild
        self._incremental_update(mv, mover, board)

    # ---------- INCREMENTAL UPDATES ----------
    def _move_affects_white_holes(self, mv: Move, board: Board) -> bool:
        """Check if move affects white holes (piece moved, captured, or white hole moved)."""
        # Check if moved piece is a white hole
        if self._cache_manager:
            moved_piece = self._cache_manager.piece_cache.get(mv.from_coord)
        else:
            # Fallback to board method if cache manager not available
            moved_piece = board.cache_manager.occupancy.get(mv.from_coord)

        if moved_piece and moved_piece.ptype == PieceType.WHITE_HOLE:
            return True

        # Check if destination had a white hole (captured)
        if self._cache_manager:
            dest_piece = self._cache_manager.piece_cache.get(mv.to_coord)
        else:
            # Fallback to board method if cache manager not available
            dest_piece   = board.cache_manager.occupancy.get(mv.to_coord)

        if dest_piece and dest_piece.ptype == PieceType.WHITE_HOLE:
            return True

        # Check if move affects squares near white holes
        for color in (Color.WHITE, Color.BLACK):
            if mv.from_coord in self._affected_squares[color]:
                return True
            if mv.to_coord in self._affected_squares[color]:
                return True

        return False

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
        # Update white hole positions
        old_white_holes = self._white_hole_positions.copy()
        self._update_white_hole_positions(board)

        # Only rebuild push maps for colors that had changes
        for color in (Color.WHITE, Color.BLACK):
            if old_white_holes[color] != self._white_hole_positions[color]:
                self._rebuild_push_map_for_color(board, color)
                self._dirty_flags['push_map'] = True

    def _update_white_hole_positions(self, board: Board) -> None:
        """Update white hole position tracking."""
        for color in (Color.WHITE, Color.BLACK):
            old_positions = self._white_hole_positions[color].copy()
            self._white_hole_positions[color].clear()

            # Find all white holes of this color
            for coord, piece in board.list_occupied():
                if piece.color == color and piece.ptype == PieceType.WHITE_HOLE:
                    self._white_hole_positions[color].add(coord)

            # Mark dirty if positions changed
            if old_positions != self._white_hole_positions[color]:
                self._dirty_flags['white_holes'] = True

    def _rebuild_push_map_for_color(self, board: Board, color: Color) -> None:
        """Rebuild push map only for specific color."""
        # Clear old data
        self._push_map[color].clear()
        self._affected_squares[color].clear()
        self._push_chains[color].clear()

        # Rebuild from scratch for this color
        if self._cache_manager:
            candidates = push_candidates(board, color, self._cache_manager)
        else:
            # Fallback if cache manager not available
            candidates = push_candidates(board, color, None)

        # Build push map and track affected squares
        for from_sq, to_sq in candidates.items():
            self._push_map[color][from_sq] = to_sq

            # Track affected squares (for incremental updates)
            self._affected_squares[color].add(from_sq)
            self._affected_squares[color].add(to_sq)

            # Build push chains (cascading effects)
            self._build_push_chain(board, color, from_sq, to_sq)

    def _build_push_chain(self, board: Board, color: Color, from_sq: Tuple[int, int, int],
                         to_sq: Tuple[int, int, int]) -> None:
        """Build chain reaction effects for complex push scenarios."""
        # This would handle cases where pushing one piece causes a chain reaction
        # For now, just track the direct effect
        effect = PushEffect(
            from_square=from_sq,
            to_square=to_sq,
            affected_pieces={from_sq},  # Could be expanded for chains
            priority=PushPriority.HIGH
        )
        self._push_chains[color].append(effect)

    # ---------- FULL REBUILD (FALLBACK) ----------
    def _full_rebuild(self, board: Board) -> None:
        """Complete rebuild of all data structures."""
        for color in (Color.WHITE, Color.BLACK):
            self._rebuild_push_map_for_color(board, color)

        self._dirty_flags['push_map'] = False
        self._dirty_flags['white_holes'] = False
        self._dirty_flags['chains'] = False

    def _incremental_rebuild(self) -> None:
        """Smart incremental rebuild based on dirty flags."""
        board = self._get_board()
        if board is None:
            return

        if self._dirty_flags['white_holes']:
            self._update_white_hole_positions(board)

        if self._dirty_flags['push_map']:
            for color in (Color.WHITE, Color.BLACK):
                if self._dirty_flags['white_holes'] or len(self._push_map[color]) == 0:
                    self._rebuild_push_map_for_color(board, color)

        if self._dirty_flags['chains']:
            self._rebuild_push_chains()

        # Clear dirty flags
        for flag in self._dirty_flags:
            self._dirty_flags[flag] = False

    def _rebuild_push_chains(self) -> None:
        """Rebuild push chain effects."""
        board = self._get_board()
        if board is None:
            return

        for color in (Color.WHITE, Color.BLACK):
            self._push_chains[color].clear()
            # Rebuild chains based on current push map
            for from_sq, to_sq in self._push_map[color].items():
                self._build_push_chain(board, color, from_sq, to_sq)

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
            'white_holes_white': len(self._white_hole_positions[Color.WHITE]),
            'white_holes_black': len(self._white_hole_positions[Color.BLACK]),
            'push_entries_white': len(self._push_map[Color.WHITE]),
            'push_entries_black': len(self._push_map[Color.BLACK]),
            'affected_squares_white': len(self._affected_squares[Color.WHITE]),
            'affected_squares_black': len(self._affected_squares[Color.BLACK]),
            'dirty_flags': self._dirty_flags.copy(),
            'board_hash': board.byte_hash() if board else 0,
        }

    def clear(self) -> None:
        """Clear all cached data."""
        for color in (Color.WHITE, Color.BLACK):
            self._push_map[color].clear()
            self._white_hole_positions[color].clear()
            self._affected_squares[color].clear()
            self._push_chains[color].clear()

        self._last_board_hash = 0
        for flag in self._dirty_flags:
            self._dirty_flags[flag] = True

# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_optimized_white_hole_cache(board: Optional[Board] = None, cache_manager=None) -> OptimizedWhiteHolePushCache:
    """Factory function for creating optimized cache."""
    return OptimizedWhiteHolePushCache(board, cache_manager)

# ==============================================================================
# BACKWARD COMPATIBILITY
# ==============================================================================

class WhiteHolePushCache(OptimizedWhiteHolePushCache):
    """Backward compatibility wrapper."""

    def __init__(self, cache_manager=None) -> None:
        super().__init__(None, cache_manager)  # Don't pass board to avoid immediate rebuild

# ------------------------------------------------------------------
# singleton
# ------------------------------------------------------------------
_white_hole_cache: Optional[WhiteHolePushCache] = None

def init_white_hole_push_cache(cache_manager=None) -> None:
    global _white_hole_cache
    _white_hole_cache = WhiteHolePushCache(cache_manager)

def get_white_hole_push_cache() -> WhiteHolePushCache:
    if _white_hole_cache is None:
        raise RuntimeError("WhiteHolePushCache not initialised")
    return _white_hole_cache
