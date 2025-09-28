"""Optimized incremental cache for Archery attack targets on 2-radius sphere surface."""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set, Any
from dataclasses import dataclass
from enum import Enum
import weakref
import math

from game3d.pieces.enums import Color, PieceType
from game3d.board.board import Board
from game3d.effects.archery import archery_targets
from game3d.movement.movepiece import Move
from game3d.pieces.piece import Piece

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
        "_line_of_sight", "_attack_ranges"
    )

    def __init__(self, board: Optional[Board] = None) -> None:
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

        if board:
            self._full_rebuild(board)

    # ---------- PUBLIC INTERFACE ----------
    def attack_targets(self, controller: Color) -> List[Tuple[int, int, int]]:
        """Get attack targets for controller, rebuilding if necessary."""
        if self._dirty_flags['targets']:
            self._incremental_rebuild()
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
        """Smart incremental update based on move impact."""
        if self._move_affects_cache(mv, board):
            self._incremental_update(mv, mover, board)

    def undo_move(self, mv: Move, mover: Color, board: Board) -> None:
        """Smart undo with minimal rebuilding."""
        self._incremental_update(mv, mover, board)

    def _move_affects_cache(self, mv: Move, board: Board) -> bool:
        """Check if move involves pieces that affect this cache."""
        from_piece = cache.piece_cache.get(mv.from_coord)
        to_piece = cache.piece_cache.get(mv.to_coord)

        # Archers and high-value targets affect cache
        relevant_types = {PieceType.ARCHER, PieceType.KING, PieceType.QUEEN, PieceType.ROOK}

        affects_cache = (
            (from_piece and from_piece.ptype in relevant_types) or
            (to_piece and to_piece.ptype in relevant_types)
        )

        # Also check if move affects line of sight
        if not affects_cache:
            for color in (Color.WHITE, Color.BLACK):
                if (mv.from_coord in self._affected_squares[color] or
                    mv.to_coord in self._affected_squares[color]):
                    return True

        return affects_cache

    # ---------- INCREMENTAL UPDATES ----------
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
        """Rebuild attack targets only for specific color."""
        # Clear old data
        self._targets[color].clear()
        self._line_of_sight[color].clear()
        self._attack_ranges[color].clear()

        # Get all potential targets on 2-radius sphere surfaces
        all_targets = set()
        for archer_sq in self._archer_positions[color]:
            sphere_surface = self.get_sphere_surface_squares(archer_sq)
            all_targets.update(sphere_surface)

        # Filter targets and check line of sight
        for target_sq in all_targets:
            if self._is_valid_target(board, target_sq, color):
                self._targets[color].append(target_sq)
                # Build line of sight tracking
                for archer_sq in self._archer_positions[color]:
                    if self._has_line_of_sight(board, archer_sq, target_sq):
                        if archer_sq not in self._line_of_sight[color]:
                            self._line_of_sight[color][archer_sq] = {}
                        self._line_of_sight[color][archer_sq][target_sq] = True
                        # Calculate attack range
                        distance = self._calculate_distance(archer_sq, target_sq)
                        self._attack_ranges[color][target_sq] = distance

    def _is_valid_target(self, board: Board, target_sq: Tuple[int, int, int], controller: Color) -> bool:
        """Check if square is a valid attack target."""
        # Must be occupied by enemy piece
        piece = cache.piece_cache.get(target_sq)
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

        for sq in path_squares[1:-1]:  # Exclude start and end
            if cache.piece_cache.get(sq) is not None:
                return False

        return True

    def _calculate_sphere_surface_squares(self, center: Tuple[int, int, int], radius: float) -> Set[Tuple[int, int, int]]:
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
            self._sphere_surface_squares[center] = self._calculate_sphere_surface_squares(center, radius)
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

def create_optimized_archery_cache(board: Optional[Board] = None) -> OptimizedArcheryCache:
    """Factory function for creating optimized cache."""
    return OptimizedArcheryCache(board)

# ==============================================================================
# BACKWARD COMPATIBILITY
# ==============================================================================

class ArcheryCache(OptimizedArcheryCache):
    """Backward compatibility wrapper."""

    def __init__(self) -> None:
        super().__init__(None)  # Don't pass board to avoid immediate rebuild
