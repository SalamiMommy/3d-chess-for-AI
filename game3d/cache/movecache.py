"""Incremental legal-move cache for 9×9×9."""
# game3d/cache/movecache.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from game3d.cache.manager import CacheManager
from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move
from game3d.movement.registry import get_dispatcher
from game3d.attacks.check import king_in_check
from game3d.board.board import Board
from game3d.pieces.piece import Piece

# global singleton
_move_cache: Optional[MoveCache] = None


class MoveCache:
    __slots__ = (
        "_board",
        "_current",
        "_cache",
        "_legal_per_piece",      # Dict[Coord, List[Move]]
        "_legal_by_color",       # Dict[Color, List[Move]]
        "_king_pos",             # Dict[Color, Coord]
        "_priest_count",         # Dict[Color, int]
    )

    def __init__(self,
                 board: Board,
                 current: Color,
                 cache: CacheManager,
                 *,
                 _defer_rebuild: bool = False) -> None:
        self._board = board
        self._current = current
        self._cache = cache
        self._legal_per_piece: Dict[Tuple[int, int, int], List[Move]] = {}
        self._legal_by_color: Dict[Color, List[Move]] = {
            Color.WHITE: [],
            Color.BLACK: []
        }
        self._king_pos: Dict[Color, Optional[Tuple[int, int, int]]] = {
            Color.WHITE: None,
            Color.BLACK: None
        }
        self._priest_count: Dict[Color, int] = {
            Color.WHITE: 0,
            Color.BLACK: 0
        }
        if not _defer_rebuild:
            self._full_rebuild()

    def _resync_mirror(self) -> None:
        """Replace internal board with a clone of the **current** real board."""
        self._board = self._cache.board.clone()

    # ----------------------------------------------------------
    # public API
    # ----------------------------------------------------------
    def legal_moves(self, color: Color) -> List[Move]:
        """Return cached legal moves for side `color`."""
        return self._legal_by_color[color]

    def apply_move(self, mv: Move, color: Color) -> None:
        """Apply move and incrementally update legal moves."""
        from_coord = mv.from_coord
        to_coord = mv.to_coord

        # 1. Apply the move to the board
        self._board.apply_move(mv)
        self._current = color.opposite()

        # 2. Refresh priest counts and king positions
        self._refresh_counts()

        # 3. Find affected squares:
        #    - from_coord: piece left
        #    - to_coord: piece arrived (and possibly captured piece)
        #    - Kings: their safety might be affected
        affected_squares = {from_coord, to_coord}
        for col in (Color.WHITE, Color.BLACK):
            king_pos = self._find_king(col)
            if king_pos is not None:
                affected_squares.add(king_pos)

        # 4. Remove old legal moves for affected squares
        for coord in affected_squares:
            self._legal_per_piece.pop(coord, None)

        # 5. Regenerate legal moves for affected squares
        for coord in affected_squares:
            piece = self._board.piece_at(coord)
            if piece is not None:
                self._legal_per_piece[coord] = self._generate_piece_moves(coord)

        # 6. Rebuild color move lists
        self._rebuild_color_lists()



    def undo_move(self, mv: Move, color: Color) -> None:
        """Undo move and incrementally update legal moves."""
        self._resync_mirror()

        # ------------------------------------------------------------------
        # 1. Handle captures first – restore captured piece on the square
        #    where it was taken (mv.to_coord) **before** we put the mover back.
        # ------------------------------------------------------------------
        if getattr(mv, "is_capture", False):
            captured_type = getattr(mv, "captured_ptype", None)
            if captured_type is not None:
                captured_color = color.opposite()
                self._board.set_piece(
                    mv.to_coord, Piece(captured_color, captured_type)
                )

        # ------------------------------------------------------------------
        # 2. Now move the piece back to its origin square.
        # ------------------------------------------------------------------
        piece = self._board.piece_at(mv.to_coord)   # still the mover here
        if piece is not None:
            self._board.set_piece(mv.from_coord, piece)
            self._board.set_piece(mv.to_coord, None)

        # ------------------------------------------------------------------
        # 3. Handle promotions – demote to pawn.
        # ------------------------------------------------------------------
        if (getattr(mv, "is_promotion", False) and
            getattr(mv, "promotion_type", None)):
            self._board.set_piece(
                mv.from_coord, Piece(piece.color, PieceType.PAWN)
            )

        # ------------------------------------------------------------------
        # 4. Finish incremental update (unchanged).
        # ------------------------------------------------------------------
        self._current = color
        self._refresh_counts()
        affected_squares = {mv.from_coord, mv.to_coord}
        for col in (Color.WHITE, Color.BLACK):
            king_pos = self._find_king(col)
            if king_pos is not None:
                affected_squares.add(king_pos)

        for coord in affected_squares:
            self._legal_per_piece.pop(coord, None)
        for coord in affected_squares:
            piece = self._board.piece_at(coord)
            if piece is not None:
                self._legal_per_piece[coord] = self._generate_piece_moves(coord)
        self._rebuild_color_lists()

    # ----------------------------------------------------------
    # internals
    # ----------------------------------------------------------
    def _full_rebuild(self) -> None:
        """Brute-force rebuild all legal moves – with paranoid filter."""
        self._refresh_counts()
        self._legal_per_piece.clear()
        white_moves: list[Move] = []
        black_moves: list[Move] = []

        for coord, piece in self._board.list_occupied():
            pseudo = self._generate_piece_moves(coord)   # already filtered inside
            # EXTRA PARANOID: drop anything that slipped through
            safe = [m for m in pseudo if m.from_coord == coord]
            self._legal_per_piece[coord] = safe
            if piece.color == Color.WHITE:
                white_moves.extend(safe)
            else:
                black_moves.extend(safe)

        self._legal_by_color[Color.WHITE] = white_moves
        self._legal_by_color[Color.BLACK] = black_moves

    def _generate_piece_moves(self, coord: Tuple[int, int, int]) -> List[Move]:
        """Return *fully* legal moves for the piece on `coord`."""
        piece = self._board.piece_at(coord)
        if piece is None:                       # nothing here
            return []

        dispatcher = get_dispatcher(piece.ptype)
        if dispatcher is None:                  # unknown piece
            return []

        pseudo = dispatcher(self._board, piece.color, *coord)
        # 🔥  PARANOID:  drop dispatcher bugs
        pseudo = [m for m in pseudo if m.from_coord == coord]

        if self._cache.is_frozen(coord, piece.color):
            return []

        # king-safety check WITHOUT cloning – mutate real board temporarily
        legal = []
        skip_check = self._priest_count[piece.color] > 0

        for mv in pseudo:
            # quick sanity: target square must not contain own piece
            victim = self._board.piece_at(mv.to_coord)
            if victim is not None and victim.color == piece.color:
                continue

            # apply move temporarily
            moving_piece = self._board.piece_at(mv.from_coord)
            self._board.set_piece(mv.from_coord, None)
            self._board.set_piece(mv.to_coord, moving_piece)

            safe = skip_check or not king_in_check(self._board, piece.color, piece.color)

            # undo the temporary move
            self._board.set_piece(mv.from_coord, moving_piece)
            self._board.set_piece(mv.to_coord, victim)

            if safe:
                legal.append(mv)

        return legal

        # ------------------------------------------------------------------
        # King-safety check WITHOUT cloning – do it **once** on the real board
        # ------------------------------------------------------------------
        legal = []
        skip_check = self._priest_count[piece.color] > 0

        for mv in pseudo:
            # 1.  quick sanity: target square must not contain own piece
            victim = self._board.piece_at(mv.to_coord)
            if victim is not None and victim.color == piece.color:
                continue

            # 2.  apply move temporarily
            moving_piece = self._board.piece_at(mv.from_coord)
            self._board.set_piece(mv.from_coord, None)
            self._board.set_piece(mv.to_coord, moving_piece)

            # 3.  king still safe?
            safe = skip_check or not king_in_check(self._board, piece.color, piece.color)

            # 4.  undo the temporary move
            self._board.set_piece(mv.from_coord, moving_piece)
            self._board.set_piece(mv.to_coord, victim)

            if safe:
                legal.append(mv)

        return legal

    def _find_king(self, color: Color) -> Optional[Tuple[int, int, int]]:
        """Find king position for color."""
        # Use cached king position if available
        if self._king_pos[color] is not None:
            king_piece = self._board.piece_at(self._king_pos[color])
            if (king_piece is not None and
                king_piece.color == color and
                king_piece.ptype == PieceType.KING):
                return self._king_pos[color]

        # Otherwise scan board
        for c, p in self._board.list_occupied():
            if p.color == color and p.ptype == PieceType.KING:
                self._king_pos[color] = c
                return c

        self._king_pos[color] = None
        return None

    def _refresh_counts(self) -> None:
        """Refresh priest counts and king positions."""
        # Reset counts
        self._priest_count[Color.WHITE] = 0
        self._priest_count[Color.BLACK] = 0
        self._king_pos[Color.WHITE] = None
        self._king_pos[Color.BLACK] = None

        # Scan board once
        for coord, piece in self._board.list_occupied():
            if piece.ptype == PieceType.PRIEST:
                self._priest_count[piece.color] += 1
            elif piece.ptype == PieceType.KING:
                self._king_pos[piece.color] = coord

    def _rebuild_color_lists(self) -> None:
        """Rebuild color move lists from per-piece moves."""
        white_moves = []
        black_moves = []

        for coord, moves in self._legal_per_piece.items():
            piece = self._board.piece_at(coord)
            if piece is None:
                continue
            if piece.color == Color.WHITE:
                white_moves.extend(moves)
            else:
                black_moves.extend(moves)

        self._legal_by_color[Color.WHITE] = white_moves
        self._legal_by_color[Color.BLACK] = black_moves


# ------------------------------------------------------------------
# global singleton helpers
# ------------------------------------------------------------------
def init_cache(board: Board, current: Color) -> None:
    global _move_cache
    from game3d.cache.manager import get_cache_manager
    _move_cache = MoveCache(board, current, get_cache_manager())


def get_cache() -> MoveCache:
    if _move_cache is None:
        raise RuntimeError("MoveCache not initialized")
    return _move_cache
