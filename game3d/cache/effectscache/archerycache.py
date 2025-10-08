"""Optimized incremental cache for Archery attack targets on 2-radius sphere surface."""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set, Any, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
import weakref
import math

from game3d.pieces.enums import Color, PieceType
from game3d.effects.archery import archery_targets
from game3d.movement.movepiece import Move
from game3d.pieces.piece import Piece

if TYPE_CHECKING:
    from game3d.board.board import Board

# ==============================================================================
# OPTIMIZATION CONSTANTS
# ==============================================================================

@dataclass
class ArcheryEffect:
    """Represents an archery attack effect."""
    archer_square: Tuple[int, int, int]
    target_square: Tuple[int, int, int]
    distance: float
    priority: int

class ArcheryPriority(Enum):
    """Priority levels for archery attacks."""
    HIGH = 3    # Direct line of sight
    MEDIUM = 2  # Indirect shots
    LOW = 1     # Blocked or difficult shots

# ==============================================================================
# OPTIMIZED ARCHERY CACHE
# ==============================================================================

class OptimizedArcheryCache:
    """Optimized incremental cache for Archery attack targets with 2-radius sphere surface."""

    __slots__ = (
        "_targets", "_archer_positions", "_sphere_surface_squares",
        "_board_ref", "_last_board_hash", "_dirty_flags", "_target_tracking",
        "_line_of_sight", "_attack_ranges", "_cache_manager"
    )

    def __init__(self, board: Optional[Board] = None, cache_manager=None) -> None:
        # Core attack targets
        self._targets: Dict[Color, List[Tuple[int, int, int]]] = {
            Color.WHITE: [],
            Color.BLACK: [],
        }

        # Archer positions for quick lookup
        self._archer_positions: Dict[Color, Set[Tuple[int, int, int]]] = {
            Color.WHITE: set(),
            Color.BLACK: set(),
        }

        # Pre-calculated sphere surface squares for 2-radius
        self._sphere_surface_squares: Dict[Tuple[int, int, int], Set[Tuple[int, int, int]]] = {}

        # Line of sight tracking
        self._line_of_sight: Dict[Color, Dict[Tuple[int, int, int], Dict[Tuple[int, int, int], bool]]] = {
            Color.WHITE: {},
            Color.BLACK: {},
        }

        # Attack range tracking
        self._attack_ranges: Dict[Color, Dict[Tuple[int, int, int], float]] = {
            Color.WHITE: {},
            Color.BLACK: {},
        }

        # Board reference and change tracking
        self._board_ref: Optional[weakref.ref] = weakref.ref(board) if board else None
        self._last_board_hash: int = 0
        self._dirty_flags: Dict[str, bool] = {
            'archers': True,
            'targets': True,
            'los': True,
        }

        self._cache_manager = cache_manager
        self._last_archer_count = 0  # Track archer count changes
        self._rebuild_threshold = 100  # Only rebuild every N moves
        self._moves_since_rebuild = 0

    # ---------- PUBLIC INTERFACE ----------
    def attack_targets(self, controller: Color) -> List[Tuple[int, int, int]]:
        """Get attack targets - rebuild only if dirty."""
        if self._dirty_flags['targets']:
            # Lazy rebuild - only when actually needed
            self._rebuild_targets_for_color(self._get_board(), controller)
            self._dirty_flags['targets'] = False
        return self._targets[controller].copy()

    def is_valid_attack(self, sq: Tuple[int, int, int], controller: Color) -> bool:
        """Check if square is a valid attack target."""
        if self._dirty_flags['targets']:
            self._incremental_rebuild()
        return sq in self._targets[controller]

    def get_archer_count(self, controller: Color) -> int:
        """Get number of archers for controller."""
        return len(self._archer_positions[controller])

    def get_sphere_surface_squares(self, center: Tuple[int, int, int]) -> Set[Tuple[int, int, int]]:
        """Get squares on 2-radius sphere surface around center."""
        if center not in self._sphere_surface_squares:
            self._sphere_surface_squares[center] = self._calculate_sphere_surface(center, 2.0)
        return self._sphere_surface_squares[center]

    def has_line_of_sight(self, from_sq: Tuple[int, int, int], to_sq: Tuple[int, int, int], controller: Color) -> bool:
        """Check if there's line of sight between squares."""
        if self._dirty_flags['los']:
            self._incremental_rebuild()
        return self._line_of_sight[controller].get(from_sq, {}).get(to_sq, False)

    # ---------- MOVE HANDLING ----------
    def apply_move(self, mv: Move, mover: Color, board: Board) -> None:
        """Smart incremental update - avoid expensive rebuilds."""
        self._moves_since_rebuild += 1

        # Only check if move affects cache if it involves relevant pieces
        if not self._quick_relevance_check(mv):
            # Move doesn't affect archery - skip update entirely
            return

        # Check if archer count changed (fast check)
        current_archer_count = self._count_archers()
        if current_archer_count != self._last_archer_count:
            # Archer was added/removed - need to rebuild
            self._last_archer_count = current_archer_count
            self._minimal_rebuild(mv, mover, board)
            self._moves_since_rebuild = 0
        elif self._moves_since_rebuild > self._rebuild_threshold:
            # Periodic rebuild to prevent drift
            self._full_rebuild(board)
            self._moves_since_rebuild = 0
        else:
            # Just mark as dirty - rebuild happens on next query
            self._dirty_flags['targets'] = True

    def undo_move(self, mv: Move, mover: Color, board: Board) -> None:
        """Smart undo with minimal rebuilding."""
        self._incremental_update(mv, mover, board)

    def _quick_relevance_check(self, mv: Move) -> bool:
        """Fast check if move could affect archery cache."""
        # Check if move involves archer positions or target squares
        return (mv.from_coord in self._archer_positions[Color.WHITE] or
                mv.from_coord in self._archer_positions[Color.BLACK] or
                mv.to_coord in self._archer_positions[Color.WHITE] or
                mv.to_coord in self._archer_positions[Color.BLACK])

    def _count_archers(self) -> int:
        """Fast archer count using cache manager."""
        if not self._cache_manager:
            return 0
        return (len(self._archer_positions[Color.WHITE]) +
                len(self._archer_positions[Color.BLACK]))

    # ---------- INCREMENTAL UPDATES ----------
    def _incremental_update(self, mv: Move, mover: Color, board: Board) -> None:
        """SIMPLIFIED - no more board hash checks."""
        # Just update archer positions
        self._update_archer_positions(board)
        # Mark dirty - actual rebuild happens on query
        self._dirty_flags['targets'] = True
        self._dirty_flags['los'] = True

    def _minimal_rebuild(self, mv: Move, mover: Color, board: Board) -> None:
        """Minimal rebuild affecting only changed areas."""
        # Update archer positions
        old_archers = self._archer_positions.copy()
        self._update_archer_positions(board)

        # Only rebuild targets for colors that had changes
        for color in (Color.WHITE, Color.BLACK):
            if old_archers[color] != self._archer_positions[color]:
                self._rebuild_targets_for_color(board, color)
                self._dirty_flags['targets'] = True
                self._dirty_flags['los'] = True

    def _update_archer_positions(self, board: Board) -> None:
        """Update archer position tracking."""
        for color in (Color.WHITE, Color.BLACK):
            old_positions = self._archer_positions[color].copy()
            self._archer_positions[color].clear()

            # Find all archers of this color
            for coord, piece in board.list_occupied():
                if piece.color == color and piece.ptype == PieceType.ARCHER:
                    self._archer_positions[color].add(coord)

            # Mark dirty if positions changed
            if old_positions != self._archer_positions[color]:
                self._dirty_flags['archers'] = True

    def _rebuild_targets_for_color(self, board: Board, color: Color) -> None:
        """Optimized rebuild using direct cache access."""
        if board is None:
            return

        # Clear old data
        self._targets[color].clear()

        # Fast path: no archers = no targets
        if not self._archer_positions[color]:
            return

        # Use set for deduplication
        all_targets = set()

        # For each archer, get sphere surface squares
        for archer_sq in self._archer_positions[color]:
            sphere_surface = self.get_sphere_surface_squares(archer_sq)

            # Check each square in sphere
            for target_sq in sphere_surface:
                # Use cache manager for fast piece lookup
                if self._cache_manager:
                    piece = self._cache_manager.piece_cache.get(target_sq)
                else:
                    piece = None  # Instead of board.get_piece(target_sq)

                # Valid target: enemy piece
                if piece and piece.color != color:
                    all_targets.add(target_sq)

        # Convert to list
        self._targets[color] = list(all_targets)

    def _is_valid_target(self, board: Board, target_sq: Tuple[int, int, int], controller: Color) -> bool:
        """Check if square is a valid attack target."""
        # Must be occupied by enemy piece
        if self._cache_manager:
            piece = self._cache_manager.piece_cache.get(target_sq)
        else:
            piece = None

        if not piece or piece.color == controller:
            return False

        # Must be on 2-radius sphere surface from at least one archer
        for archer_sq in self._archer_positions[controller]:
            if self._is_on_sphere_surface(archer_sq, target_sq, 2.0):
                return True

        return False

    def _has_line_of_sight(self, board: Board, from_sq: Tuple[int, int, int], to_sq: Tuple[int, int, int]) -> bool:
        """Check if there's clear line of sight between squares."""
        # Simple line of sight check - can be enhanced with actual 3D ray casting
        # For now, check if path is clear of blocking pieces
        path_squares = self._get_path_squares(from_sq, to_sq)

        for sq in path_squares[1:-1]:
            if self._cache_manager:
                if self._cache_manager.piece_cache.get(sq) is not None:
                    return False
            else:
                return False

        return True

    def _calculate_sphere_surface(self, center: Tuple[int, int, int], radius: float) -> Set[Tuple[int, int, int]]:
        """Calculate squares on sphere surface with given radius."""
        surface_squares = set()
        cx, cy, cz = center

        # Check squares in a reasonable range around center
        range_size = int(radius) + 2

        for dx in range(-range_size, range_size + 1):
            for dy in range(-range_size, range_size + 1):
                for dz in range(-range_size, range_size + 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue

                    target = (cx + dx, cy + dy, cz + dz)
                    distance = math.sqrt(dx*dx + dy*dy + dz*dz)

                    # Check if distance is close to radius (within tolerance)
                    if abs(distance - radius) < 0.1:
                        surface_squares.add(target)

        return surface_squares

    def _is_on_sphere_surface(self, center: Tuple[int, int, int], target: Tuple[int, int, int], radius: float) -> bool:
        """Check if target is on sphere surface around center."""
        if center not in self._sphere_surface_squares:
            self._sphere_surface_squares[center] = self._calculate_sphere_surface(center, radius)
        return target in self._sphere_surface_squares[center]

    def _calculate_distance(self, from_sq: Tuple[int, int, int], to_sq: Tuple[int, int, int]) -> float:
        """Calculate Euclidean distance between squares."""
        dx = to_sq[0] - from_sq[0]
        dy = to_sq[1] - from_sq[1]
        dz = to_sq[2] - from_sq[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def _get_path_squares(self, from_sq: Tuple[int, int, int], to_sq: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Get squares along path from from_sq to to_sq."""
        # Simple 3D line algorithm
        path = []
        dx = to_sq[0] - from_sq[0]
        dy = to_sq[1] - from_sq[1]
        dz = to_sq[2] - from_sq[2]

        steps = max(abs(dx), abs(dy), abs(dz))
        if steps == 0:
            return [from_sq]

        for i in range(steps + 1):
            t = i / steps
            x = round(from_sq[0] + dx * t)
            y = round(from_sq[1] + dy * t)
            z = round(from_sq[2] + dz * t)
            path.append((x, y, z))

        return path

    # ---------- FULL REBUILD (FALLBACK) ----------
    def _full_rebuild(self, board: Board) -> None:
        """Complete rebuild of all data structures."""
        for color in (Color.WHITE, Color.BLACK):
            self._rebuild_targets_for_color(board, color)

        self._dirty_flags['archers'] = False
        self._dirty_flags['targets'] = False
        self._dirty_flags['los'] = False

    def _incremental_rebuild(self) -> None:
        """Smart incremental rebuild based on dirty flags."""
        board = self._get_board()
        if board is None:
            return

        if self._dirty_flags['archers']:
            self._update_archer_positions(board)

        if self._dirty_flags['targets'] or self._dirty_flags['los']:
            for color in (Color.WHITE, Color.BLACK):
                if self._dirty_flags['archers'] or len(self._targets[color]) == 0:
                    self._rebuild_targets_for_color(board, color)

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
        total_targets = len(self._targets[Color.WHITE]) + len(self._targets[Color.BLACK])
        total_archers = len(self._archer_positions[Color.WHITE]) + len(self._archer_positions[Color.BLACK])

        return {
            'archers_white': len(self._archer_positions[Color.WHITE]),
            'archers_black': len(self._archer_positions[Color.BLACK]),
            'targets_white': len(self._targets[Color.WHITE]),
            'targets_black': len(self._targets[Color.BLACK]),
            'total_targets': total_targets,
            'total_archers': total_archers,
            'sphere_cache_size': len(self._sphere_surface_squares),
            'dirty_flags': self._dirty_flags.copy(),
            'board_hash': board.byte_hash() if board else 0,
        }

    def clear(self) -> None:
        """Clear all cached data."""
        for color in (Color.WHITE, Color.BLACK):
            self._targets[color].clear()
            self._archer_positions[color].clear()
            self._line_of_sight[color].clear()
            self._attack_ranges[color].clear()

        self._sphere_surface_squares.clear()
        self._last_board_hash = 0

        for flag in self._dirty_flags:
            self._dirty_flags[flag] = True



# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_optimized_archery_cache(board: Optional[Board] = None, cache_manager=None) -> OptimizedArcheryCache:
    """Factory function for creating optimized cache."""
    return OptimizedArcheryCache(board, cache_manager)

# ==============================================================================
# BACKWARD COMPATIBILITY
# ==============================================================================

class ArcheryCache(OptimizedArcheryCache):
    """Backward compatibility wrapper."""

    def __init__(self, cache_manager=None) -> None:
        super().__init__(None, cache_manager)  # Don't pass board to avoid immediate rebuild

# ------------------------------------------------------------------
# singleton
# ------------------------------------------------------------------
_archery_cache: Optional[ArcheryCache] = None

def init_archery_cache(cache_manager=None) -> None:
    global _archery_cache
    _archery_cache = ArcheryCache(cache_manager)

def get_archery_cache() -> ArcheryCache:
    if _archery_cache is None:
        raise RuntimeError("ArcheryCache not initialised")
    return _archery_cache
