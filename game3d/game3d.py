"""Game controller – turn order, move submission, outcome."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

from pieces.enums import Color, Result
from game.move import Move
from game3d.game.gamestate import GameState


@dataclass(slots=True, frozen=True)
class MoveReceipt:
    """Return value from submit_move."""
    new_state: GameState
    is_legal: bool
    is_game_over: bool
    result: Optional[Result] = None
    message: str = ""


class Game3D:
    """Minimal game controller."""

    __slots__ = ("_state",)

    def __init__(self, start_state: Optional[GameState] = None) -> None:
        self._state = start_state or GameState.empty()

    # ----------------------------------------------------------
    # public read-only API
    # ----------------------------------------------------------
    @property
    def state(self) -> GameState:
        """Current position (immutable)."""
        return self._state

    @property
    def current_player(self) -> Color:
        return self._state.current

    def is_game_over(self) -> bool:
        return self._state.is_game_over()

    def result(self) -> Optional[Result]:
        return self._state.result()

    # ----------------------------------------------------------
    # move submission
    # ----------------------------------------------------------
    def submit_move(self, move: Move) -> MoveReceipt:
        """Attempt to play a move.  Always returns a fresh state."""
        # 1. game already over → reject
        if self.is_game_over():
            return MoveReceipt(
                new_state=self._state,
                is_legal=False,
                is_game_over=True,
                result=self.result(),
                message="Game already finished.",
            )

        # 2. not the mover's turn → reject
        # (optional – you can remove if client enforces turn order)
        # if move.color != self.current_player:
        #     return MoveReceipt(..., message="Not your turn.")

        # 3. legality test
        if move not in self._state.legal_moves():
            return MoveReceipt(
                new_state=self._state,
                is_legal=False,
                is_game_over=False,
                message="Illegal move.",
            )

        # 4. play and return
        new_state = self._state.make_move(move)
        self._state = new_state          # atomically update

        return MoveReceipt(
            new_state=new_state,
            is_legal=True,
            is_game_over=new_state.is_game_over(),
            result=new_state.result(),
            message="",
        )

    # ----------------------------------------------------------
    # utilities
    # ----------------------------------------------------------
    def reset(self, start_state: Optional[GameState] = None) -> None:
        """Start a new game (or from custom position)."""
        self._state = start_state or GameState.empty()

    def __repr__(self) -> str:
        return f"Game3D({self._state})"
