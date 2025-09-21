from __future__ import annotations
import torch
from dataclasses import dataclass, field
from functools import lru_cache
from pieces.enums import Color
from game.board import Board
from game.move import Move
from movement.generator import generate_legal_moves

@dataclass(slots=True)
class GameState:
    board: Board
    current: Color
    history: list[Move] = field(default_factory=list)

    # ---------- tensor for NN ----------
    def to_tensor(self) -> torch.Tensor:
        """(C,9,9,9) tensor + current-player plane."""
        board_t = self.board.to_tensor()          # (84,9,9,9)
        player_t = torch.full((1, 9, 9, 9), float(self.current))
        return torch.cat([board_t, player_t], dim=0)

    # ---------- move helpers ----------
    @lru_cache(maxsize=32_768)
    def legal_moves(self) -> tuple[Move, ...]:
        return tuple(generate_legal_moves(self))

    def make_move(self, m: Move) -> GameState:
        # shallow copy + apply
        new_board = Board()
        # naive deep-copy for now (will be optimised later)
        for z in range(9):
            for y in range(9):
                for x in range(9):
                    new_board.set_piece(x, y, z, self.board.pieces[z][y][x])
        # apply move logic (placeholder)
        # TODO: movement.apply_move(new_board, m)
        new_history = self.history.copy() + [m]
        return GameState(new_board, Color(1 - self.current.value), new_history)
