# game3d/cache/effectscache/attackscache.py
from __future__ import annotations
from typing import Dict, Set, Tuple, Optional, TYPE_CHECKING, List
from dataclasses import dataclass, field
from game3d.pieces.enums import Color, PieceType
from game3d.common.common import Coord

if TYPE_CHECKING:
    from game3d.board.board import Board
    from game3d.movement.movepiece import Move
    from game3d.pieces.piece import Piece

@dataclass(slots=True)
class AttacksCache:
    """Incremental cache for attacked squares by each color."""

    board: 'Board'
    # Main cache: attacked squares for each color
    attacked_squares: Dict[Color, Set[Tuple[int, int, int]]] = field(default_factory=dict)

    # Per-piece attack contributions for incremental updates
    piece_attacks: Dict[Tuple[int, int, int], Set[Tuple[int, int, int]]] = field(default_factory=dict)

    # Track last known positions of pieces for efficient updates
    last_positions: Dict[Color, Set[Tuple[int, int, int]]] = field(default_factory=dict)

    # Invalidation tracking
    is_valid: Dict[Color, bool] = field(default_factory=dict)

    def __post_init__(self):
        self.attacked_squares[Color.WHITE] = set()
        self.attacked_squares[Color.BLACK] = set()
        self.last_positions[Color.WHITE] = set()
        self.last_positions[Color.BLACK] = set()
        self.is_valid[Color.WHITE] = False
        self.is_valid[Color.BLACK] = False

    def get_for_color(self, color: Color) -> Optional[Set[Tuple[int, int, int]]]:
        """Get attacked squares for a color if cache is valid."""
        if self.is_valid.get(color, False):
            return self.attacked_squares.get(color, set()).copy()
        return None

    def store_for_color(self, color: Color, attacked: Set[Tuple[int, int, int]]) -> None:
        """Store attacked squares for a color and mark as valid."""
        self.attacked_squares[color] = attacked.copy()
        self.is_valid[color] = True

        # Rebuild piece_attacks mapping for this color
        self._rebuild_piece_attacks_for_color(color)

    def invalidate(self, color: Optional[Color] = None) -> None:
        """Invalidate cache for a color or both."""
        if color is None:
            self.is_valid[Color.WHITE] = False
            self.is_valid[Color.BLACK] = False
            self.piece_attacks.clear()
        else:
            self.is_valid[color] = False
            # Clear piece attacks for invalidated color
            coords_to_remove = []
            for coord in self.piece_attacks.keys():
                piece = self.board.piece_at(coord)
                if piece and piece.color == color:
                    coords_to_remove.append(coord)
            for coord in coords_to_remove:
                del self.piece_attacks[coord]

    def apply_move(self, mv: 'Move', mover: Color, board: 'Board') -> None:
        """Incrementally update attacked squares on move."""
        # Update both colors since move affects both
        self._incremental_update(mv, mover, board, is_undo=False)

    def undo_move(self, mv: 'Move', mover: Color, board: 'Board') -> None:
        """Incrementally update on undo."""
        self._incremental_update(mv, mover, board, is_undo=True)

    def _incremental_update(self, mv: 'Move', mover: Color, board: 'Board', is_undo: bool) -> None:
        """
        Perform incremental update of attacked squares.

        Strategy:
        1. Remove attacks from piece's old position
        2. Add attacks from piece's new position
        3. Update attacks from pieces that might be affected (discovered attacks/blocks)
        """
        from_coord = mv.from_coord
        to_coord = mv.to_coord

        # Get the moving piece (from destination if undo, from source if apply)
        if is_undo:
            moving_piece = board.piece_at(from_coord)
        else:
            moving_piece = board.piece_at(to_coord)

        if moving_piece is None:
            # Full rebuild if we can't determine piece
            self.invalidate()
            return

        # Update attacker's color
        self._update_piece_attacks(from_coord, to_coord, moving_piece, mover, board, is_undo)

        # Update affected sliding pieces (discovered attacks/blocks)
        self._update_affected_sliders(from_coord, to_coord, mover, board)

        # Mark both colors as valid after incremental update
        self.is_valid[mover] = True
        self.is_valid[mover.opposite()] = True

    def _update_piece_attacks(self, from_coord: Coord, to_coord: Coord,
                            piece: 'Piece', color: Color, board: 'Board', is_undo: bool) -> None:
        """Update attacks for a specific piece movement."""
        # Remove old attacks
        if from_coord in self.piece_attacks:
            old_attacks = self.piece_attacks[from_coord]
            self.attacked_squares[color] -= old_attacks
            del self.piece_attacks[from_coord]

        # Add new attacks (if not undo to empty square)
        if not is_undo or board.piece_at(from_coord) is not None:
            new_attacks = self._calculate_piece_attacks(to_coord if not is_undo else from_coord,
                                                       piece, board)
            self.piece_attacks[to_coord if not is_undo else from_coord] = new_attacks
            self.attacked_squares[color] |= new_attacks

    def _update_affected_sliders(self, from_coord: Coord, to_coord: Coord,
                                color: Color, board: 'Board') -> None:
        """
        Update sliding pieces that might be affected by the move.
        This handles discovered attacks and blocked attacks.
        """
        # Get all sliding pieces that might be affected
        affected_coords = self._get_potentially_affected_sliders(from_coord, to_coord, board)

        for coord in affected_coords:
            piece = board.piece_at(coord)
            if piece is None:
                continue

            # Recalculate attacks for this slider
            if coord in self.piece_attacks:
                old_attacks = self.piece_attacks[coord]
                self.attacked_squares[piece.color] -= old_attacks

            new_attacks = self._calculate_piece_attacks(coord, piece, board)
            self.piece_attacks[coord] = new_attacks
            self.attacked_squares[piece.color] |= new_attacks

    def _get_potentially_affected_sliders(self, from_coord: Coord, to_coord: Coord,
                                         board: 'Board') -> Set[Coord]:
        """
        Get sliding pieces that might be affected by the move.
        Returns pieces on the same rays as from_coord or to_coord.
        """
        affected = set()

        # Sliding piece types
        sliding_types = {
            PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP,
            PieceType.XYQUEEN, PieceType.XZQUEEN, PieceType.YZQUEEN,
            PieceType.TRIGONALBISHOP, PieceType.EDGEROOK,
            PieceType.VECTORSLIDER, PieceType.CONESLIDER,
            PieceType.SPIRAL, PieceType.XZZIGZAG, PieceType.YZZIGZAG
        }

        # Check all 26 directions from both coordinates
        for coord in [from_coord, to_coord]:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == dy == dz == 0:
                            continue

                        # Scan along this direction
                        curr_x, curr_y, curr_z = coord
                        for step in range(1, 9):
                            curr_x += dx
                            curr_y += dy
                            curr_z += dz

                            if not (0 <= curr_x < 9 and 0 <= curr_y < 9 and 0 <= curr_z < 9):
                                break

                            check_coord = (curr_x, curr_y, curr_z)
                            piece = board.piece_at(check_coord)

                            if piece is not None:
                                if piece.ptype in sliding_types:
                                    affected.add(check_coord)
                                break  # Stop at first piece in this direction

        return affected

    def _calculate_piece_attacks(self, coord: Coord, piece: 'Piece', board: 'Board') -> Set[Coord]:
        """
        Calculate all squares attacked by a piece at a given coordinate.
        This uses the move generation system but only for attack squares.
        """
        from game3d.movement.registry import get_dispatcher

        attacks = set()

        # Get move dispatcher for this piece type
        dispatcher = get_dispatcher(piece.ptype)
        if dispatcher is None:
            return attacks

        try:
            # Create minimal state for move generation
            from game3d.game.gamestate import GameState
            from game3d.cache.manager import get_cache_manager

            # Use a minimal cache to avoid recursion
            temp_cache = get_cache_manager(board, piece.color)
            temp_state = GameState.__new__(GameState)
            temp_state.board = board
            temp_state.color = piece.color
            temp_state.cache = temp_cache

            # Generate pseudo-legal moves
            moves = dispatcher(temp_state, coord[0], coord[1], coord[2])

            # Extract attack squares
            for move in moves:
                attacks.add(move.to_coord)

        except Exception:
            # If move generation fails, fall back to empty set
            pass

        return attacks

    def _rebuild_piece_attacks_for_color(self, color: Color) -> None:
        """Rebuild the piece_attacks mapping for a specific color."""
        # Clear existing entries for this color
        coords_to_remove = []
        for coord in self.piece_attacks.keys():
            piece = self.board.piece_at(coord)
            if piece and piece.color == color:
                coords_to_remove.append(coord)
        for coord in coords_to_remove:
            del self.piece_attacks[coord]

        # Rebuild from current attacked_squares
        # This is approximate - we know the total but not per-piece breakdown
        # For full accuracy, would need to recalculate from scratch
        for coord, piece in self.board.list_occupied():
            if piece.color == color:
                attacks = self._calculate_piece_attacks(coord, piece, self.board)
                self.piece_attacks[coord] = attacks

    def force_rebuild(self, color: Optional[Color] = None) -> None:
        """Force a complete rebuild of the cache."""
        if color is None:
            self.invalidate()
            for c in [Color.WHITE, Color.BLACK]:
                self._full_rebuild(c)
        else:
            self.invalidate(color)
            self._full_rebuild(color)

    def _full_rebuild(self, color: Color) -> None:
        """Perform a full rebuild of attacked squares for a color."""
        attacked = set()
        self.piece_attacks.clear()

        for coord, piece in self.board.list_occupied():
            if piece.color == color:
                attacks = self._calculate_piece_attacks(coord, piece, self.board)
                self.piece_attacks[coord] = attacks
                attacked |= attacks

        self.attacked_squares[color] = attacked
        self.is_valid[color] = True

    def get_stats(self) -> Dict[str, any]:
        """Get cache statistics for monitoring."""
        return {
            'white_attacked_count': len(self.attacked_squares.get(Color.WHITE, set())),
            'black_attacked_count': len(self.attacked_squares.get(Color.BLACK, set())),
            'white_valid': self.is_valid.get(Color.WHITE, False),
            'black_valid': self.is_valid.get(Color.BLACK, False),
            'piece_attacks_tracked': len(self.piece_attacks),
        }
