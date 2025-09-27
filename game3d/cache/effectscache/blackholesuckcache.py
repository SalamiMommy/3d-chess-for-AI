"""Optimized incremental cache for Black-Hole suck pull map with performance improvements."""

from __future__ import annotations
from typing import Dict, Tuple, Optional, Set, List, Any
from dataclasses import dataclass
from enum import Enum
import weakref

from game3d.pieces.enums import Color
from game3d.board.board import Board
from game3d.effects.auras.blackholesuck import suck_candidates
from game3d.movement.movepiece import Move
from game3d.pieces.piece import Piece

# ==============================================================================
# OPTIMIZATION CONSTANTS
# ==============================================================================

@dataclass
class PullEffect:
    """Represents a black hole pull effect."""
    from_square: Tuple[int, int, int]
    to_square: Tuple[int, int, int]
    affected_pieces: Set[Tuple[int, int, int]]  # Pieces that would be pulled
    priority: int  # Higher priority pulls override lower priority ones

class PullPriority(Enum):
    """Priority levels for pull effects."""
    HIGH = 3    # Direct black hole pulls
    MEDIUM = 2  # Chain reaction pulls
    LOW = 1     # Indirect effects

# ==============================================================================
# OPTIMIZED BLACK HOLE SUCK CACHE
# ==============================================================================

class OptimizedBlackHoleSuckCache:
    """Optimized incremental cache for Black-Hole suck pull map with smart updates."""

    __slots__ = (
        "_pull_map", "_black_hole_positions", "_affected_squares",
        "_board_ref", "_last_board_hash", "_dirty_flags", "_pull_chains"
    )

    def __init__(self, board: Optional[Board] = None) -> None:
        # Core pull maps
        self._pull_map: Dict[Color, Dict[Tuple[int, int, int], Tuple[int, int, int]]] = {
            Color.WHITE: {},
            Color.BLACK: {},
        }

        # Black hole positions for quick lookup
        self._black_hole_positions: Dict[Color, Set[Tuple[int, int, int]]] = {
            Color.WHITE: set(),
            Color.BLACK: set(),
        }

        # Squares affected by black holes (for incremental updates)
        self._affected_squares: Dict[Color, Set[Tuple[int, int, int]]] = {
            Color.WHITE: set(),
            Color.BLACK: set(),
        }

        # Pull chains (cascading effects)
        self._pull_chains: Dict[Color, List[PullEffect]] = {
            Color.WHITE: [],
            Color.BLACK: [],
        }

        # Board reference and change tracking
        self._board_ref: Optional[weakref.ref] = weakref.ref(board) if board else None
        self._last_board_hash: int = 0
        self._dirty_flags: Dict[str, bool] = {
            'black_holes': True,
            'pull_map': True,
            'chains': True,
        }

        if board:
            self._full_rebuild(board)

    # ---------- PUBLIC INTERFACE ----------
    def pull_map(self, controller: Color) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
        """Get pull map for controller, rebuilding if necessary."""
        if self._dirty_flags['pull_map']:
            self._incremental_rebuild()
        return self._pull_map[controller]

    def get_pull_targets(self, controller: Color, from_square: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Get all possible pull targets from a square (including chains)."""
        if self._dirty_flags['chains']:
            self._rebuild_pull_chains()

        targets = []
        main_target = self._pull_map[controller].get(from_square)
        if main_target:
            targets.append(main_target)

        # Add chain reaction targets
        for effect in self._pull_chains[controller]:
            if effect.from_square == from_square:
                targets.append(effect.to_square)

        return targets

    def is_affected_by_black_hole(self, square: Tuple[int, int, int], controller: Color) -> bool:
        """Quick check if square is affected by black holes."""
        return square in self._affected_squares[controller]

    def get_black_hole_count(self, controller: Color) -> int:
        """Get number of black holes for controller."""
        return len(self._black_hole_positions[controller])

    # ---------- MOVE HANDLING ----------
    def apply_move(self, mv: Move, mover: Color, board: Board) -> None:
        """Smart incremental update based on move impact."""
        # Quick check if move affects black holes
        if self._move_affects_black_holes(mv, board):
            self._incremental_update(mv, mover, board)
        else:
            # Move doesn't affect black holes, no rebuild needed
            pass

    def undo_move(self, mv: Move, mover: Color, board: Board) -> None:
        """Smart undo with minimal rebuilding."""
        # For undo, we need to check if the move affected black holes
        # This is more complex - for now, rebuild
        self._incremental_update(mv, mover, board)

    # ---------- INCREMENTAL UPDATES ----------
    def _move_affects_black_holes(self, mv: Move, board: Board) -> bool:
        """Check if move affects black holes (piece moved, captured, or black hole moved)."""
        # Check if moved piece is a black hole
        moved_piece = board.piece_at(mv.from_coord)
        if moved_piece and moved_piece.ptype == PieceType.BLACK_HOLE:
            return True

        # Check if destination had a black hole (captured)
        dest_piece = board.piece_at(mv.to_coord)
        if dest_piece and dest_piece.ptype == PieceType.BLACK_HOLE:
            return True

        # Check if move affects squares near black holes
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
        # Update black hole positions
        old_black_holes = self._black_hole_positions.copy()
        self._update_black_hole_positions(board)

        # Only rebuild pull maps for colors that had changes
        for color in (Color.WHITE, Color.BLACK):
            if old_black_holes[color] != self._black_hole_positions[color]:
                self._rebuild_pull_map_for_color(board, color)
                self._dirty_flags['pull_map'] = True

    def _update_black_hole_positions(self, board: Board) -> None:
        """Update black hole position tracking."""
        for color in (Color.WHITE, Color.BLACK):
            old_positions = self._black_hole_positions[color].copy()
            self._black_hole_positions[color].clear()

            # Find all black holes of this color
            for coord, piece in board.list_occupied():
                if piece.color == color and piece.ptype == PieceType.BLACK_HOLE:
                    self._black_hole_positions[color].add(coord)

            # Mark dirty if positions changed
            if old_positions != self._black_hole_positions[color]:
                self._dirty_flags['black_holes'] = True

    def _rebuild_pull_map_for_color(self, board: Board, color: Color) -> None:
        """Rebuild pull map only for specific color."""
        # Clear old data
        self._pull_map[color].clear()
        self._affected_squares[color].clear()
        self._pull_chains[color].clear()

        # Rebuild from scratch for this color
        candidates = suck_candidates(board, color)

        # Build pull map and track affected squares
        for from_sq, to_sq in candidates.items():
            self._pull_map[color][from_sq] = to_sq

            # Track affected squares (for incremental updates)
            self._affected_squares[color].add(from_sq)
            self._affected_squares[color].add(to_sq)

            # Build pull chains (cascading effects)
            self._build_pull_chain(board, color, from_sq, to_sq)

    def _build_pull_chain(self, board: Board, color: Color, from_sq: Tuple[int, int, int],
                         to_sq: Tuple[int, int, int]) -> None:
        """Build chain reaction effects for complex pull scenarios."""
        # This would handle cases where pulling one piece causes a chain reaction
        # For now, just track the direct effect
        effect = PullEffect(
            from_square=from_sq,
            to_square=to_sq,
            affected_pieces={from_sq},  # Could be expanded for chains
            priority=PullPriority.HIGH
        )
        self._pull_chains[color].append(effect)

    # ---------- FULL REBUILD (FALLBACK) ----------
    def _full_rebuild(self, board: Board) -> None:
        """Complete rebuild of all data structures."""
        for color in (Color.WHITE, Color.BLACK):
            self._rebuild_pull_map_for_color(board, color)

        self._dirty_flags['pull_map'] = False
        self._dirty_flags['black_holes'] = False
        self._dirty_flags['chains'] = False

    def _incremental_rebuild(self) -> None:
        """Smart incremental rebuild based on dirty flags."""
        board = self._get_board()
        if board is None:
            return

        if self._dirty_flags['black_holes']:
            self._update_black_hole_positions(board)

        if self._dirty_flags['pull_map']:
            for color in (Color.WHITE, Color.BLACK):
                if self._dirty_flags['black_holes'] or len(self._pull_map[color]) == 0:
                    self._rebuild_pull_map_for_color(board, color)

        if self._dirty_flags['chains']:
            self._rebuild_pull_chains()

        # Clear dirty flags
        for flag in self._dirty_flags:
            self._dirty_flags[flag] = False

    def _rebuild_pull_chains(self) -> None:
        """Rebuild pull chain effects."""
        board = self._get_board()
        if board is None:
            return

        for color in (Color.WHITE, Color.BLACK):
            self._pull_chains[color].clear()
            # Rebuild chains based on current pull map
            for from_sq, to_sq in self._pull_map[color].items():
                self._build_pull_chain(board, color, from_sq, to_sq)

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
            'black_holes_white': len(self._black_hole_positions[Color.WHITE]),
            'black_holes_black': len(self._black_hole_positions[Color.BLACK]),
            'pull_entries_white': len(self._pull_map[Color.WHITE]),
            'pull_entries_black': len(self._pull_map[Color.BLACK]),
            'affected_squares_white': len(self._affected_squares[Color.WHITE]),
            'affected_squares_black': len(self._affected_squares[Color.BLACK]),
            'dirty_flags': self._dirty_flags.copy(),
            'board_hash': board.byte_hash() if board else 0,
        }

    def clear(self) -> None:
        """Clear all cached data."""
        for color in (Color.WHITE, Color.BLACK):
            self._pull_map[color].clear()
            self._black_hole_positions[color].clear()
            self._affected_squares[color].clear()
            self._pull_chains[color].clear()

        self._last_board_hash = 0
        for flag in self._dirty_flags:
            self._dirty_flags[flag] = True

# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_optimized_black_hole_cache(board: Optional[Board] = None) -> OptimizedBlackHoleSuckCache:
    """Factory function for creating optimized cache."""
    return OptimizedBlackHoleSuckCache(board)

# ==============================================================================
# BACKWARD COMPATIBILITY
# ==============================================================================

class BlackHoleSuckCache(OptimizedBlackHoleSuckCache):
    """Backward compatibility wrapper."""

    def __init__(self) -> None:
        super().__init__(None)  # Don't pass board to avoid immediate rebuild
