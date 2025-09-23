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

    def submit_archery_attack(self, target_sq: Tuple[int, int, int]) -> MoveReceipt:
        """Archer player fires at target_sq instead of moving a piece."""
        if not self.state.current_player_controls_archer():   # you’ll add this helper
            return MoveReceipt(
                new_state=self.state,
                is_legal=False,
                is_game_over=False,
                message="No archer controlled.",
            )
        if not get_cache_manager().is_valid_archery_attack(target_sq, self.state.current):
            return MoveReceipt(
                new_state=self.state,
                is_legal=False,
                is_game_over=False,
                message="Invalid archery target.",
            )

        # build a synthetic "move" that only captures
        archer_move = Move(
            from_coord=target_sq,   # archer doesn't move – use target as from/to
            to_coord=target_sq,
            is_capture=True,
            metadata={"is_archery": True},
        )

        new_board = self.state.board.clone()
        new_board.set_piece(target_sq, None)  # remove victim
        new_hist = self.state.history.copy() + [archer_move]

        # update caches (normal move semantics)
        get_cache_manager().apply_move(archer_move, self.state.current)

        new_state = GameState(
            board=new_board,
            current=self.state.current.opposite(),
            history=new_hist,
            halfmove_clock=0,  # capture resets 50-move rule
        )

        return MoveReceipt(
            new_state=new_state,
            is_legal=True,
            is_game_over=new_state.is_game_over(),
            result=new_state.result(),
            message="Archery strike!",
        )

    def submit_hive_turn(self, moves: List[Move]) -> MoveReceipt:
        """
        Player submits **any subset** of Hive moves (≥1, ≤all).
        Must be **only** Hive pieces moved this turn.
        """
        if not moves:
            return MoveReceipt(self.state, False, False, message="No moves submitted.")

        # 1. must be current player's turn
        if self.state.current not in (m.metadata.get("color", self.state.current) for m in moves):
            return MoveReceipt(self.state, False, False, message="Not your turn.")

        # 2. every moved piece must be a Hive
        for mv in moves:
            p = self.state.board.piece_at(mv.from_coord)
            if p is None or p.ptype != PieceType.HIVE or p.color != self.state.current:
                return MoveReceipt(self.state, False, False, message="Only Hive pieces may move.")

        # 3. no other piece type moved this ply
        all_legal = self.state.legal_moves()
        non_hive_moves = [m for m in all_legal if self.state.board.piece_at(m.from_coord).ptype != PieceType.HIVE]
        if non_hive_moves:
            return MoveReceipt(self.state, False, False, message="You must move **only** Hive pieces this turn.")

        # 4. every submitted move must be individually legal
        for mv in moves:
            if mv not in all_legal:
                return MoveReceipt(self.state, False, False, message=f"Illegal move: {mv}")

        # 5. apply all moves atomically
        new_board = self.state.board.clone()
        new_hist = self.state.history.copy()
        for mv in moves:
            new_board.apply_move(mv)
            new_hist.append(mv)

        new_state = GameState(
            board=new_board,
            current=self.state.current.opposite(),
            history=new_hist,
            halfmove_clock=0,  # any Hive move resets 50-move rule
        )

        return MoveReceipt(
            new_state=new_state,
            is_legal=True,
            is_game_over=new_state.is_game_over(),
            result=new_state.result(),
            message="Hive turn completed.",
        )
    # ----------------------------------------------------------
    # utilities
    # ----------------------------------------------------------
    def reset(self, start_state: Optional[GameState] = None) -> None:
        """Start a new game (or from custom position)."""
        self._state = start_state or GameState.empty()

    def __repr__(self) -> str:
        return f"Game3D({self._state})"
