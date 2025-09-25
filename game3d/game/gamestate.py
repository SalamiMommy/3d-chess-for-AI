"""
Tiny 64-bit Zobrist hasher for GameState.
Works for any board size (x,y,z) and any PieceType / Color.
"""
#game3d/game/gamestate.py
from __future__ import annotations
from dataclasses import dataclass, field   #  <-- add this line
from functools import lru_cache
from typing import List, Tuple, Optional
import random
import torch
from game3d.board.board import Board
from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move
from game3d.common.common import SIZE_X, SIZE_Y, SIZE_Z
from game3d.cache.manager import CacheManager
from game3d.attacks.check import king_in_check          # <-- fixes NameError

from game3d.pieces.piece import Piece  # Assuming this is where Piece is defined
from game3d.cache.effectscache.trailblazecache import TrailblazeCache  # If needed
from game3d.cache.manager import get_cache_manager, get_trailblaze_cache  # If needed
from game3d.effects.bomb import detonate  # Add this import for the detonate function
# global tables filled once at import time
_PIECE_KEYS: Dict[Tuple[PieceType, Color, Tuple[int, int, int]], int] = {}
_EN_PASSANT_KEYS: Dict[Tuple[int, int, int], int] = {}
_CASTLE_KEYS: Dict[str, int] = {}
_SIDE_KEY: int = 0
_INITIALIZED: bool = False

def _init_zobrist(width: int = 9, height: int = 9, depth: int = 9) -> None:
    """Call once per program start (automatically done on first key request)."""
    global _PIECE_KEYS, _EN_PASSANT_KEYS, _CASTLE_KEYS, _SIDE_KEY, _INITIALIZED
    if _INITIALIZED:
        return

    # 1. piece on square
    for ptype in PieceType:
        for color in Color:
            for x in range(width):
                for y in range(height):
                    for z in range(depth):
                        _PIECE_KEYS[(ptype, color, (x, y, z))] = random.getrandbits(64)

    # 2. en-passant file (we use the square itself)
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                _EN_PASSANT_KEYS[(x, y, z)] = random.getrandbits(64)

    # 3. castling rights – 16 combinations (4 bits)
    for cr in range(16):
        _CASTLE_KEYS[f"{cr}"] = random.getrandbits(64)

    # 4. side to move
    _SIDE_KEY = random.getrandbits(64)

    _INITIALIZED = True

# ------------------------------------------------------------------
# public helpers
# ------------------------------------------------------------------

def start_key(game_state: GameState, turn: Color) -> int:
    _init_zobrist(SIZE_X, SIZE_Y, SIZE_Z)
    key = 0
    for sq, piece in game_state.board.list_occupied():
        key ^= _PIECE_KEYS[(piece.ptype, piece.color, sq)]
    if turn == Color.BLACK:
        key ^= _SIDE_KEY
    return key

def update_move(key: int, mv: Move, captured: Optional[Piece], board: Board,
                turn: Color, new_turn: Color) -> int:
    """Update Zobrist key after a move.  If piece is missing, return old key."""
    piece = board.piece_at(mv.from_coord)
    if piece is None:                       # 🔥 guard
        return key                            # nothing changed – key stays

    # ----------  original logic unchanged  ----------
    _init_zobrist(SIZE_X, SIZE_Y, SIZE_Z)
    key ^= _PIECE_KEYS[(piece.ptype, piece.color, mv.from_coord)]
    if captured is not None:
        key ^= _PIECE_KEYS[(captured.ptype, captured.color, mv.to_coord)]
    key ^= _PIECE_KEYS[(piece.ptype, piece.color, mv.to_coord)]
    key ^= _SIDE_KEY
    return key

def _insufficient_material(board: Board) -> bool:
    total = 0
    for _, p in board.list_occupied():
        if p.ptype not in (PieceType.KING, PieceType.PRIEST):
            total += 1
    return total == 0

def extract_enemy_slid_path(mv: Move):
    # TODO: Implement logic to extract the path of squares the enemy slid through (Trailblazer etc.)
    return []

def _any_priest_alive(board: Board, color: Color) -> bool:
    for _, piece in board.list_occupied():
        if piece.color == color and piece.ptype == PieceType.PRIEST:
            return True
    return False

@lru_cache(maxsize=262_144)
def _legal_moves_from_key(zkey: int, color: Color, cache: CacheManager) -> Tuple[Move, ...]:
    """Return legal moves for the position represented by *zkey*."""
    return tuple(cache.legal_moves(color))

@dataclass
class GameState:
    board: Board
    color: Color
    cache: CacheManager
    history: Tuple[Move, ...] = field(default_factory=tuple)
    halfmove_clock: int = 0
    _zkey: int = field(init=False)          # not passed, built in __post_init__

    def __post_init__(self) -> None:
        # build initial key
        object.__setattr__(self, '_zkey', start_key(self, self.color))

    @property                      # <── inside GameState
    def zkey(self) -> int:
        return self._zkey

    @staticmethod
    def start(cache: "CacheManager | None" = None) -> "GameState":
        from game3d.board.board import Board
        from game3d.cache.manager import CacheManager

        board = Board.empty()
        board.init_startpos()
        cache = cache or CacheManager(board)   # create one if caller didn’t bring it
        return GameState(
            board=board,
            color=Color.WHITE,
            cache=cache,
            history=(),
            halfmove_clock=0,
        )

    def to_tensor(self) -> torch.Tensor:
        board_t = self.board.tensor()
        player_t = torch.full((1, 9, 9, 9), float(self.color))
        return torch.cat([board_t, player_t], dim=0)

    # In game3d/game/gamestate.py
    def legal_moves(self) -> List[Move]:
        from game3d.cache.movecache import get_cache
        return get_cache().legal_moves(self.color)

    def pseudo_legal_moves(self) -> List[Move]:
        from game3d.movement.pseudo_legal import generate_pseudo_legal_moves
        return generate_pseudo_legal_moves(self)

    def make_move(self, mv: Move) -> GameState:
        old_board = self.board

        # 0.  FINAL PHYSICAL GUARD – piece must still be there
        if old_board.piece_at(mv.from_coord) is None:
            raise ValueError(
                f"make_move: piece disappeared from {mv.from_coord} – move {mv}"
            )

        # 1.  synchronise **all** cache mirrors
        self.cache.sync_board(old_board)

        # 2.  apply move to cache first (this will validate and update cache state)
        self.cache.apply_move(mv, self.color)

        # 3.  build the new position (immutable copy) - apply move to board
        captured = old_board.piece_at(mv.to_coord)
        new_board = old_board.clone()

        # 🔥 NEW:  if board ignored the move, stop here
        if not new_board.apply_move(mv):
            raise ValueError(
                f"make_move: board refused move {mv} (empty start square)"
            )

        # ----------  rest of method unchanged  ----------
        new_key = update_move(self._zkey, mv, captured, old_board,
                            self.color, self.color.opposite())

        moved_piece = old_board.piece_at(mv.from_coord)
        target_piece = old_board.piece_at(mv.to_coord)
        is_pawn = moved_piece is not None and moved_piece.ptype == PieceType.PAWN
        is_capture = target_piece is not None
        new_clock = 0 if (is_pawn or is_capture) else self.halfmove_clock + 1

        # Fix history concatenation (tuples don't have copy() or append())
        new_hist = self.history + (mv,)  # Concatenate tuple with single move

        # Black-hole suck
        pull_map = self.cache.black_hole_pull_map(self.color)
        for from_sq, to_sq in pull_map.items():
            piece = new_board.piece_at(from_sq)
            if piece is None or piece.color == self.color:
                continue
            new_board.set_piece(to_sq, piece)
            new_board.set_piece(from_sq, None)

        # White-hole push
        push_map = self.cache.white_hole_push_map(self.color)
        for from_sq, to_sq in push_map.items():
            piece = new_board.piece_at(from_sq)
            if piece is None or piece.color == self.color:
                continue
            new_board.set_piece(to_sq, piece)
            new_board.set_piece(from_sq, None)

        # Bomb detonation
        trigger_piece = old_board.piece_at(mv.from_coord)  # Use old_board here
        target_piece = old_board.piece_at(mv.to_coord)     # Use old_board here
        if (target_piece is not None and
            target_piece.ptype == PieceType.BOMB and
            target_piece.color != self.color):
            cleared = detonate(new_board, mv.to_coord)
            for sq in cleared:
                new_board.set_piece(sq, None)
        if (trigger_piece is not None and
            trigger_piece.ptype == PieceType.BOMB and
            mv.from_coord == mv.to_coord):
            cleared = detonate(new_board, mv.to_coord)
            for sq in cleared:
                new_board.set_piece(sq, None)

        # Trailblaze counters
        trail_cache = self.cache._effect["trailblaze"]
        enemy_color = self.color.opposite()
        enemy_slid = extract_enemy_slid_path(mv)
        for sq in enemy_slid:
            if trail_cache.increment_counter(sq, enemy_color, new_board):  # Add new_board parameter
                victim = new_board.piece_at(sq)
                if victim is not None and victim.ptype != PieceType.KING:
                    new_board.set_piece(sq, None)
                elif victim is not None and victim.ptype == PieceType.KING:
                    if not _any_priest_alive(new_board, enemy_color):
                        new_board.set_piece(sq, None)
        if trail_cache.increment_counter(mv.to_coord, enemy_color, new_board):  # Add new_board parameter
            victim = new_board.piece_at(mv.to_coord)
            if victim is not None and victim.ptype != PieceType.KING:
                new_board.set_piece(mv.to_coord, None)
            elif victim is not None and victim.ptype == PieceType.KING:
                if not _any_priest_alive(new_board, enemy_color):
                    new_board.set_piece(mv.to_coord, None)

        # Create new GameState with the same cache reference
        new_state = GameState(
            board=new_board,
            color=self.color.opposite(),
            cache=self.cache,  # Use the same cache instance
            history=new_hist,
            halfmove_clock=new_clock
        )

        # Set the zkey using object.__setattr__ to bypass frozen dataclass
        object.__setattr__(new_state, '_zkey', new_key)

        return new_state

    def undo_move(self) -> Optional["GameState"]:
        if not self.history:
            return None
        last_move = self.history[-1]

        # 1.  start from the current board
        new_board = self.board.clone()

        # 2.  reverse the last move manually
        piece = new_board.piece_at(last_move.to_coord)
        if piece is None:          # should never happen
            raise RuntimeError("Corrupt history: destination square empty on undo")

        # 3.  put the piece back where it started
        new_board.set_piece(last_move.from_coord, piece)
        new_board.set_piece(last_move.to_coord,   None)

        # 4.  restore capture (if any)
        if getattr(last_move, "is_capture", False):
            # captured piece is always the opposite color
            captured_type = getattr(last_move, "captured_ptype", None)
            if captured_type is not None:
                captured_color = piece.color.opposite()
                new_board.set_piece(
                    last_move.to_coord,
                    Piece(captured_color, captured_type)
                )

        # 5.  undo promotion (if any)
        if (getattr(last_move, "is_promotion", False) and
            getattr(last_move, "promotion_type", None)):
            new_board.set_piece(
                last_move.from_coord,
                Piece(piece.color, PieceType.PAWN)
            )

        # 6.  build the new state
        return GameState(
            board=new_board,
            color=self.color.opposite(),
            cache=self.cache,               # same global cache instance
            history=self.history[:-1],
            halfmove_clock=max(0, self.halfmove_clock - 1),
        )

    def submit_block(self, sq: Tuple[int, int, int]) -> bool:
        # TODO: Implement color_player_controls_geomancer and board state logic.
        # For now, always False.
        return False

    def is_check(self) -> bool:
        return king_in_check(self.board, self.color, self.color)

    def is_stalemate(self) -> bool:
        return not self.is_check() and not self.legal_moves()

    def is_insufficient_material(self) -> bool:
        return _insufficient_material(self.board)

    def is_fifty_move_draw(self) -> bool:
        return self.halfmove_clock >= 100

    def is_game_over(self) -> bool:
        return (
            self.is_fifty_move_draw()
            or self.is_insufficient_material()
            or not self.legal_moves()
        )

    def result(self) -> Optional[Result]:
        if not self.is_game_over():
            return None
        if self.is_fifty_move_draw() or self.is_insufficient_material() or self.is_stalemate():
            return Result.DRAW
        return Result.WHITE if self.color == Color.BLACK else Result.BLACK

    def is_terminal(self) -> bool:
        return self.is_game_over()

    def outcome(self) -> int:
        res = self.result()
        if res == Result.WHITE:
            return 1
        if res == Result.BLACK:
            return -1
        if res == Result.DRAW:
            return 0
        return None

    def sample_pi(self, pi):
        moves = self.legal_moves()
        if not moves:
            return None
        return moves[0]

    def clone(self) -> 'GameState':
        return GameState(
            board=self.board.clone(),  # This should be deep
            current=self.current,
            cache=self._cache  # But cache should NOT be cloned!
        )

    def __repr__(self) -> str:
        return (f"GameState(color={self.color}, "
                f"history={len(self.history)}, "
                f"clock={self.halfmove_clock}, "
                f"legal={len(self.legal_moves())})")
