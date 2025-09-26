# game3d/game/gamestate.py
from __future__ import annotations
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Tuple, Optional, Dict   # <─ added Dict
import random
import torch

from game3d.board.board import Board
from game3d.pieces.enums import Color, PieceType, Result
from game3d.movement.movepiece import Move
from game3d.common.common import SIZE_X, SIZE_Y, SIZE_Z
from game3d.cache.manager import CacheManager
from game3d.attacks.check import king_in_check
from game3d.pieces.piece import Piece
from game3d.cache.effectscache.trailblazecache import TrailblazeCache
from game3d.cache.manager import get_cache_manager
from game3d.effects.bomb import detonate

# ------------------------------------------------------------------
# Zobrist tables
# ------------------------------------------------------------------
_PIECE_KEYS: Dict[Tuple[PieceType, Color, Tuple[int, int, int]], int] = {}
_EN_PASSANT_KEYS: Dict[Tuple[int, int, int], int] = {}
_CASTLE_KEYS: Dict[str, int] = {}
_SIDE_KEY: int = 0
_INITIALIZED: bool = False


def _init_zobrist(width: int = 9, height: int = 9, depth: int = 9) -> None:
    global _PIECE_KEYS, _EN_PASSANT_KEYS, _CASTLE_KEYS, _SIDE_KEY, _INITIALIZED
    if _INITIALIZED:
        return

    for ptype in PieceType:
        for color in Color:
            for x in range(width):
                for y in range(height):
                    for z in range(depth):
                        _PIECE_KEYS[(ptype, color, (x, y, z))] = random.getrandbits(64)

    for x in range(width):
        for y in range(height):
            for z in range(depth):
                _EN_PASSANT_KEYS[(x, y, z)] = random.getrandbits(64)

    for cr in range(16):
        _CASTLE_KEYS[f"{cr}"] = random.getrandbits(64)

    _SIDE_KEY = random.getrandbits(64)
    _INITIALIZED = True


# ------------------------------------------------------------------
# Zobrist helpers
# ------------------------------------------------------------------
def start_key(game_state: GameState, turn: Color) -> int:
    _init_zobrist(SIZE_X, SIZE_Y, SIZE_Z)
    key = 0
    # board is a field of GameState – call list_occupied on it
    for sq, piece in game_state.board.list_occupied():
        key ^= _PIECE_KEYS[(piece.ptype, piece.color, sq)]
    if turn == Color.BLACK:
        key ^= _SIDE_KEY
    return key


def update_move(
    key: int,
    mv: Move,
    captured: Optional[Piece],
    board: Board,
    turn: Color,
    new_turn: Color
) -> int:
    piece = board.piece_at(mv.from_coord)
    if piece is None:
        return key

    _init_zobrist(SIZE_X, SIZE_Y, SIZE_Z)
    key ^= _PIECE_KEYS[(piece.ptype, piece.color, mv.from_coord)]
    if captured is not None:
        key ^= _PIECE_KEYS[(captured.ptype, captured.color, mv.to_coord)]
    key ^= _PIECE_KEYS[(piece.ptype, piece.color, mv.to_coord)]
    key ^= _SIDE_KEY
    return key


# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------
def _insufficient_material(board: Board) -> bool:
    total = 0
    for _, p in board.list_occupied():
        if p.ptype not in (PieceType.KING, PieceType.PRIEST):
            total += 1
    return total == 0


def extract_enemy_slid_path(mv: Move) -> List[Tuple[int, int, int]]:
    return []


def _any_priest_alive(board: Board, color: Color) -> bool:
    for _, piece in board.list_occupied():
        if piece.color == color and piece.ptype == PieceType.PRIEST:
            return True
    return False


# ------------------------------------------------------------------
# Game-state dataclass
# ------------------------------------------------------------------
@dataclass(slots=True)
class GameState:
    board: Board
    color: Color
    cache: CacheManager
    history: Tuple[Move, ...] = field(default_factory=tuple)
    halfmove_clock: int = 0
    _zkey: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, '_zkey', start_key(self, self.color))

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------
    @property
    def zkey(self) -> int:
        return self._zkey

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @staticmethod
    def start(cache: Optional[CacheManager] = None) -> GameState:
        board = Board.empty()
        board.init_startpos()
        cache = cache or get_cache_manager(board, Color.WHITE)
        return GameState(
            board=board,
            color=Color.WHITE,
            cache=cache,
            history=(),
            halfmove_clock=0,
        )

    # ------------------------------------------------------------------
    # Tensor representation
    # ------------------------------------------------------------------
    def to_tensor(self) -> torch.Tensor:
        board_t = self.board.tensor()
        player_t = torch.full((1, 9, 9, 9), float(self.color))
        return torch.cat([board_t, player_t], dim=0)

    # ------------------------------------------------------------------
    # Move generation (keeps new dispatcher contract)
    # ------------------------------------------------------------------
    def legal_moves(self) -> List[Move]:
        from game3d.movement.legal import generate_legal_moves
        return generate_legal_moves(self)

    def pseudo_legal_moves(self) -> List[Move]:
        from game3d.movement.pseudo_legal import generate_pseudo_legal_moves
        return generate_pseudo_legal_moves(self.board, self.color, self.cache)

    # ------------------------------------------------------------------
    # Make / undo moves
    # ------------------------------------------------------------------
    def make_move(self, mv: Move) -> GameState:
        if self.board.piece_at(mv.from_coord) is None:
            raise ValueError(f"make_move: piece disappeared from {mv.from_coord}")

        captured = self.board.piece_at(mv.to_coord)
        new_board = self.board.clone()
        if not new_board.apply_move(mv):
            raise ValueError(f"make_move: board refused move {mv}")

        new_key = update_move(self._zkey, mv, captured, self.board,
                              self.color, self.color.opposite())

        moved_piece = self.board.piece_at(mv.from_coord)
        target_piece = self.board.piece_at(mv.to_coord)
        is_pawn = moved_piece is not None and moved_piece.ptype == PieceType.PAWN
        is_capture = target_piece is not None
        new_clock = 0 if (is_pawn or is_capture) else self.halfmove_clock + 1
        new_hist = self.history + (mv,)

        # Black-hole suck
        pull_map = self.cache.black_hole_pull_map(self.color)
        for from_sq, to_sq in pull_map.items():
            piece = new_board.piece_at(from_sq)
            if piece is not None and piece.color != self.color:
                new_board.set_piece(to_sq, piece)
                new_board.set_piece(from_sq, None)

        # White-hole push
        push_map = self.cache.white_hole_push_map(self.color)
        for from_sq, to_sq in push_map.items():
            piece = new_board.piece_at(from_sq)
            if piece is not None and piece.color != self.color:
                new_board.set_piece(to_sq, piece)
                new_board.set_piece(from_sq, None)

        # Bomb detonation
        if (target_piece is not None and
            target_piece.ptype == PieceType.BOMB and
            target_piece.color != self.color):
            for sq in detonate(new_board, mv.to_coord):
                new_board.set_piece(sq, None)

        if (moved_piece is not None and
            moved_piece.ptype == PieceType.BOMB and
            mv.from_coord == mv.to_coord):
            for sq in detonate(new_board, mv.to_coord):
                new_board.set_piece(sq, None)

        # Trailblaze
        trail_cache = self.cache._effect["trailblaze"]
        enemy_color = self.color.opposite()
        enemy_slid = extract_enemy_slid_path(mv)
        for sq in enemy_slid:
            if trail_cache.increment_counter(sq, enemy_color, new_board):
                victim = new_board.piece_at(sq)
                if victim is not None and victim.ptype != PieceType.KING:
                    new_board.set_piece(sq, None)
                elif victim is not None and victim.ptype == PieceType.KING:
                    if not _any_priest_alive(new_board, enemy_color):
                        new_board.set_piece(sq, None)

        if trail_cache.increment_counter(mv.to_coord, enemy_color, new_board):
            victim = new_board.piece_at(mv.to_coord)
            if victim is not None and victim.ptype != PieceType.KING:
                new_board.set_piece(mv.to_coord, None)
            elif victim is not None and victim.ptype == PieceType.KING:
                if not _any_priest_alive(new_board, enemy_color):
                    new_board.set_piece(mv.to_coord, None)

        new_state = GameState(
            board=new_board,
            color=self.color.opposite(),
            cache=self.cache,
            history=new_hist,
            halfmove_clock=new_clock,
        )
        object.__setattr__(new_state, '_zkey', new_key)
        return new_state

    def undo_move(self) -> Optional[GameState]:
        if not self.history:
            return None
        last_move = self.history[-1]
        new_board = self.board.clone()

        piece = new_board.piece_at(last_move.to_coord)
        if piece is None:
            raise RuntimeError("Corrupt history: destination square empty on undo")

        new_board.set_piece(last_move.from_coord, piece)
        new_board.set_piece(last_move.to_coord, None)

        if getattr(last_move, "is_capture", False):
            captured_type = getattr(last_move, "captured_ptype", None)
            if captured_type is not None:
                captured_color = piece.color.opposite()
                new_board.set_piece(
                    last_move.to_coord,
                    Piece(captured_color, captured_type)
                )

        if (getattr(last_move, "is_promotion", False) and
            getattr(last_move, "promotion_type", None)):
            new_board.set_piece(
                last_move.from_coord,
                Piece(piece.color, PieceType.PAWN)
            )

        return GameState(
            board=new_board,
            color=self.color.opposite(),
            cache=self.cache,
            history=self.history[:-1],
            halfmove_clock=max(0, self.halfmove_clock - 1),
        )
    # ------------------------------------------------------------------
    # Game status
    # ------------------------------------------------------------------
    def is_check(self) -> bool:
        return king_in_check(self.board, self.color, self.color, self.cache)

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
            return Result.IN_PROGRESS
        if self.is_fifty_move_draw() or self.is_insufficient_material() or self.is_stalemate():
            return Result.DRAW
        return Result.BLACK_WON if self.color == Color.WHITE else Result.WHITE_WON

    def is_terminal(self) -> bool:
        return self.is_game_over()

    def outcome(self) -> int:
        res = self.result()
        if res == Result.WHITE_WON:
            return 1
        elif res == Result.BLACK_WON:
            return -1
        elif res == Result.DRAW:
            return 0
        raise ValueError("outcome() called on non-terminal state")

    def sample_pi(self, pi):
        moves = self.legal_moves()
        return moves[0] if moves else None

    def clone(self) -> GameState:
        return GameState(
            board=self.board.clone(),
            color=self.color,
            cache=self.cache,
            history=self.history,
            halfmove_clock=self.halfmove_clock,
        )

    def __repr__(self) -> str:
        return (f"GameState(color={self.color}, "
                f"history={len(self.history)}, "
                f"clock={self.halfmove_clock}, "
                f"legal={len(self.legal_moves())})")
