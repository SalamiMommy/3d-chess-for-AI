"""Full GameState with 50-move rule, insufficient-material, stalemate."""

from __future__ import annotations
import torch
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Tuple, Optional

from pieces.enums import Color, PieceType, Result
from game3d.board.board import Board
from game3d.game.move import Move
from game3d.cache.movecache import init_cache, get_cache, MoveCache
from game3d.movement.check import king_in_check
from game3d.effects.bomb import detonate
from game3d.cache.manager import init_cache_manager, get_cache_manager, get_trailblaze_cache
from game3d.effects.geomancy import block_candidates

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

@dataclass(slots=True)
class GameState:
    board: Board
    current: Color
    history: List[Move] = field(default_factory=list)
    halfmove_clock: int = 0

    def __post_init__(self) -> None:
        init_cache_manager(self.board)

    @staticmethod
    def empty() -> "GameState":
        # TODO: implement an empty starting position (or standard chess start if desired)
        return GameState(Board.empty(), Color.WHITE)

    def to_tensor(self) -> torch.Tensor:
        board_t = self.board.tensor()
        player_t = torch.full((1, 9, 9, 9), float(self.current))
        return torch.cat([board_t, player_t], dim=0)

    @lru_cache(maxsize=32_768)
    def legal_moves(self) -> Tuple[Move, ...]:
        return tuple(get_cache().legal_moves(self.current))

    def pseudo_legal_moves(self) -> List[Move]:
        from game3d.movement.pseudo_legal import generate_pseudo_legal_moves
        return generate_pseudo_legal_moves(self)

    def make_move(self, mv: Move) -> "GameState":
        new_board = self.board.clone()
        new_board.apply_move(mv)
        moved_piece = self.board.piece_at(mv.from_coord)
        target_piece = self.board.piece_at(mv.to_coord)
        is_pawn = moved_piece is not None and moved_piece.ptype == PieceType.PAWN
        is_capture = target_piece is not None
        new_clock = 0 if (is_pawn or is_capture) else self.halfmove_clock + 1
        get_cache().apply_move(mv, self.current)
        new_hist = self.history.copy()
        new_hist.append(mv)

        # Black-hole suck
        pull_map = get_cache_manager().black_hole_pull_map(self.current)
        for from_sq, to_sq in pull_map.items():
            piece = new_board.piece_at(from_sq)
            if piece is None or piece.color == self.current:
                continue
            new_board.set_piece(to_sq, piece)
            new_board.set_piece(from_sq, None)
        # White-hole push
        push_map = get_cache_manager().white_hole_push_map(self.current)
        for from_sq, to_sq in push_map.items():
            piece = new_board.piece_at(from_sq)
            if piece is None or piece.color == self.current:
                continue
            new_board.set_piece(to_sq, piece)
            new_board.set_piece(from_sq, None)
        # Bomb detonation
        trigger_piece = self.board.piece_at(mv.from_coord)
        target_piece  = self.board.piece_at(mv.to_coord)
        if (target_piece is not None and
            target_piece.ptype == PieceType.BOMB and
            target_piece.color != self.current):
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
        trail_cache = get_trailblaze_cache()
        enemy_color = self.current.opposite()
        enemy_slid = extract_enemy_slid_path(mv)
        for sq in enemy_slid:
            if trail_cache.increment_counter(sq, enemy_color):
                victim = new_board.piece_at(sq)
                if victim is not None and victim.ptype != PieceType.KING:
                    new_board.set_piece(sq, None)
                elif victim is not None and victim.ptype == PieceType.KING:
                    if not _any_priest_alive(new_board, enemy_color):
                        new_board.set_piece(sq, None)
        if trail_cache.increment_counter(mv.to_coord, enemy_color):
            victim = new_board.piece_at(mv.to_coord)
            if victim is not None and victim.ptype != PieceType.KING:
                new_board.set_piece(mv.to_coord, None)
            elif victim is not None and victim.ptype == PieceType.KING:
                if not _any_priest_alive(new_board, enemy_color):
                    new_board.set_piece(mv.to_coord, None)
        return GameState(
            board=new_board,
            current=self.current.opposite(),
            history=new_hist,
            halfmove_clock=new_clock,
        )

    def undo_move(self) -> Optional["GameState"]:
        if not self.history:
            return None
        mv = self.history[-1]
        new_board = self.board.clone()
        new_board.undo_move(mv)
        new_clock = max(0, self.halfmove_clock - 1)
        get_cache().undo_move(mv, self.current.opposite())
        return GameState(
            board=new_board,
            current=self.current.opposite(),
            history=self.history[:-1],
            halfmove_clock=new_clock,
        )

    def submit_block(self, sq: Tuple[int, int, int]) -> bool:
        # TODO: Implement current_player_controls_geomancer and board state logic.
        # For now, always False.
        return False

    def is_check(self) -> bool:
        return king_in_check(self.board, self.current, self.current)

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
        return Result.WHITE if self.current == Color.BLACK else Result.BLACK

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

    def __repr__(self) -> str:
        return (f"GameState(current={self.current}, "
                f"history={len(self.history)}, "
                f"clock={self.halfmove_clock}, "
                f"legal={len(self.legal_moves())})")
