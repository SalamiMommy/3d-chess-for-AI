# game3d/cache/caches/attackscache.py - FIXED
from __future__ import annotations
from typing import Dict, Set, Tuple, Optional, TYPE_CHECKING, List
from dataclasses import dataclass, field
from game3d.common.enums import Color, PieceType
from game3d.common.coord_utils import Coord, validate_and_sanitize_coord
from game3d.common.debug_utils import CacheStatsMixin
from game3d.common.piece_utils import get_piece_effect_type, is_effect_piece
from game3d.common.cache_utils import get_piece

if TYPE_CHECKING:
    from game3d.board.board import Board
    from game3d.movement.movepiece import Move
    from game3d.pieces.piece import Piece
    from game3d.cache.manager import OptimizedCacheManager

@dataclass(slots=True)
class AttacksCache(CacheStatsMixin):
    """Incremental cache for attacked squares by each color."""

    board: "Board"
    _manager: Optional["OptimizedCacheManager"] = field(init=False, repr=False)
    attacked_squares: Dict[Color, Set[Tuple[int, int, int]]] = field(default_factory=dict)
    piece_attacks: Dict[Tuple[int, int, int], Set[Tuple[int, int, int]]] = field(default_factory=dict)
    last_positions: Dict[Color, Set[Tuple[int, int, int]]] = field(default_factory=dict)
    is_valid: Dict[Color, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # FIX: Call the parent mixin's __init__ method explicitly
        CacheStatsMixin.__init__(self)

        self._board = self.board
        self._manager = getattr(self.board, "cache_manager", None)

        # Initialize with proper structure
        self.attacked_squares[Color.WHITE] = set()
        self.attacked_squares[Color.BLACK] = set()
        self.last_positions[Color.WHITE] = set()
        self.last_positions[Color.BLACK] = set()
        self.is_valid[Color.WHITE] = False
        self.is_valid[Color.BLACK] = False

    def get_for_color(self, color: Color) -> Optional[Set[Tuple[int, int, int]]]:
        """Get attacked squares for a color if cache is valid."""
        if not self._manager:
            return None
        if self._manager.has_priest(color.opposite()):
            return set()
        if not self.is_valid.get(color, False):
            self._full_rebuild(color)
        return self.attacked_squares.get(color, set()).copy()

    def store_for_color(self, color: Color, attacked: Set[Tuple[int, int, int]]) -> None:
        """Store attacked squares for a color and mark as valid."""
        self.attacked_squares[color] = attacked.copy()
        self.is_valid[color] = True
        self._rebuild_piece_attacks_for_color(color)

    def invalidate(self, color: Optional[Color] = None) -> None:
        """Invalidate cache for a color or both."""
        if color is None:
            self.is_valid[Color.WHITE] = False
            self.is_valid[Color.BLACK] = False
            self.piece_attacks.clear()
        else:
            self.is_valid[color] = False
            coords_to_remove = []
            for coord in self.piece_attacks.keys():
                piece = get_piece(self._manager, coord)
                if piece and piece.color == color:
                    coords_to_remove.append(coord)
            for coord in coords_to_remove:
                del self.piece_attacks[coord]

    def apply_move(self, mv: 'Move', mover: Color, board: 'Board') -> None:
        """Incrementally update attacked squares on move."""
        if self._manager.has_priest(mover.opposite()):
            return
        self._incremental_update(mv, mover, board, is_undo=False)
        self.is_valid[mover] = True

    def undo_move(self, mv: 'Move', mover: Color, board: 'Board') -> None:
        """Incrementally update on undo."""
        if self._manager.has_priest(mover.opposite()):
            return
        self._incremental_update(mv, mover, board, is_undo=True)
        self.is_valid[mover] = True

    def _incremental_update(
        self,
        mv: "Move",
        mover: Color,
        board: "Board",
        is_undo: bool,
    ) -> None:
        """Incrementally update attacked squares after move."""
        from_coord = mv.from_coord
        to_coord = mv.to_coord

        if is_undo:
            moving_piece = get_piece(self._manager, from_coord)
        else:
            moving_piece = get_piece(self._manager, to_coord)

        if moving_piece is None:
            self.invalidate()
            return

        self._update_piece_attacks(from_coord, to_coord, moving_piece, mover, board, is_undo)
        self._update_affected_sliders(from_coord, to_coord, mover, board)

    def _full_rebuild(self, color: Color) -> None:
        """Complete rebuild for *color*."""
        attacked: Set[Tuple[int, int, int]] = set()

        if self._manager:
            for coord, piece in self._manager.occupancy.iter_color(color):
                attacks = self._calculate_piece_attacks(coord, piece, self.board)
                self.piece_attacks[coord] = attacks
                attacked |= attacks

        self.attacked_squares[color] = attacked
        self.is_valid[color] = True

    def _update_piece_attacks(self, from_coord: Coord, to_coord: Coord,
                            piece: 'Piece', color: Color, board: 'Board', is_undo: bool) -> None:
        """Update attacks for a specific piece movement."""
        if from_coord in self.piece_attacks:
            old_attacks = self.piece_attacks[from_coord]
            self.attacked_squares[color] -= old_attacks
            del self.piece_attacks[from_coord]

        if not is_undo or get_piece(self._manager, from_coord) is not None:
            new_attacks = self._calculate_piece_attacks(
                to_coord if not is_undo else from_coord,
                piece,
                board
            )
            self.piece_attacks[to_coord if not is_undo else from_coord] = new_attacks
            self.attacked_squares[color] |= new_attacks

    def _update_affected_sliders(self, from_coord: Coord, to_coord: Coord,
                                color: Color, board: 'Board') -> None:
        """Update sliding pieces that might be affected by the move."""
        affected_coords = self._get_potentially_affected_sliders(from_coord, to_coord, board)

        for coord in affected_coords:
            if not self._manager:
                continue

            piece = get_piece(self._manager, coord)
            if piece is None:
                continue

            if self._manager.has_priest(piece.color.opposite()):
                if coord in self.piece_attacks:
                    self.attacked_squares[piece.color] -= self.piece_attacks[coord]
                    del self.piece_attacks[coord]
                continue

            if coord in self.piece_attacks:
                old_attacks = self.piece_attacks[coord]
                self.attacked_squares[piece.color] -= old_attacks

            new_attacks = self._calculate_piece_attacks(coord, piece, board)
            self.piece_attacks[coord] = new_attacks
            self.attacked_squares[piece.color] |= new_attacks

    def _get_potentially_affected_sliders(self, from_coord: Coord, to_coord: Coord,
                                         board: 'Board') -> Set[Coord]:
        """Get sliding pieces that might be affected by the move."""
        affected = set()

        sliding_types = {
            PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP,
            PieceType.XYQUEEN, PieceType.XZQUEEN, PieceType.YZQUEEN,
            PieceType.TRIGONALBISHOP, PieceType.EDGEROOK,
            PieceType.VECTORSLIDER, PieceType.CONESLIDER,
            PieceType.SPIRAL, PieceType.XZZIGZAG, PieceType.YZZIGZAG
        }

        if not self._manager:
            return affected

        for coord in [from_coord, to_coord]:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == dy == dz == 0:
                            continue

                        curr_x, curr_y, curr_z = coord
                        for step in range(1, 9):
                            curr_x += dx
                            curr_y += dy
                            curr_z += dz

                            if not (0 <= curr_x < 9 and 0 <= curr_y < 9 and 0 <= curr_z < 9):
                                break

                            check_coord = (curr_x, curr_y, curr_z)
                            piece = get_piece(self._manager, check_coord)

                            if piece is not None:
                                if piece.ptype in sliding_types:
                                    affected.add(check_coord)
                                break

        return affected

    def _calculate_piece_attacks(self, coord: Coord, piece: 'Piece', board: 'Board') -> Set[Coord]:
        """Calculate all squares attacked by a piece at a given coordinate."""
        from game3d.movement.registry import get_dispatcher

        attacks = set()
        coord = validate_and_sanitize_coord(coord)
        if coord is None:
            return attacks

        dispatcher = get_dispatcher(piece.ptype)
        if dispatcher is None:
            return attacks

        try:
            from game3d.game.gamestate import GameState
            from game3d.cache.manager import get_cache_manager

            temp_cache = self._manager if self._manager else get_cache_manager(board, piece.color)
            temp_state = GameState.__new__(GameState)
            temp_state.board = board
            temp_state.color = piece.color
            temp_state.cache_manager = temp_cache

            moves = dispatcher(temp_state, coord[0], coord[1], coord[2])

            for move in moves:
                attacks.add(move.to_coord)

        except Exception:
            pass

        return attacks

    def _rebuild_piece_attacks_for_color(self, color: Color) -> None:
        """Rebuild the piece_attacks mapping for a specific colour."""
        if not self._manager:
            return

        for c in list(self.piece_attacks):
            p = get_piece(self._manager, c)
            if p and p.color == color:
                del self.piece_attacks[c]

        for coord, piece in self._manager.occupancy.iter_color(color):
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

    def get_stats(self) -> Dict[str, any]:
        """Get cache statistics for monitoring."""
        return {
            'white_attacked_count': len(self.attacked_squares.get(Color.WHITE, set())),
            'black_attacked_count': len(self.attacked_squares.get(Color.BLACK, set())),
            'white_valid': self.is_valid.get(Color.WHITE, False),
            'black_valid': self.is_valid.get(Color.BLACK, False),
            'piece_attacks_tracked': len(self.piece_attacks),
        }

    def set_cache_manager(self, manager: "OptimizedCacheManager") -> None:
        """Allow the cache manager to set itself after creation."""
        self._manager = manager
