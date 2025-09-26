"""Game controller – turn order, move submission, outcome."""
#game3d/game3d.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
from game3d.pieces.enums import Color, Result, PieceType  # Added PieceType
from game3d.movement.movepiece import Move
from game3d.game.gamestate import GameState
from game3d.cache.manager import CacheManager, init_cache_manager, get_cache_manager
from game3d.board.board import Board

@dataclass(slots=True, frozen=True)
class MoveReceipt:
    """Return value from submit_move."""
    new_state: GameState
    is_legal: bool
    is_game_over: bool
    result: Optional[Result] = None
    message: str = ""

class Game3D:
    __slots__ = ("_state", "_cache")

    def __init__(self) -> None:
        board = Board.startpos()
        current_color = Color.WHITE


        init_cache_manager(board, current_color)
        cache_manager = get_cache_manager()  # Get the initialized manager

        # Now create GameState with cache
        self._state = GameState(board, current_color, cache_manager)

    # rebuild cache whenever we replace the board
    def _rebuild_cache(self, board: Board) -> None:
        self._cache.occupancy.rebuild(board)

    # helper used inside submit_move, submit_archery_attack, …
    def _make_next_state(self, board: Board, move: Move, mover: Color) -> GameState:
        self._cache.apply_move(move, mover)          # incremental update
        next_player = mover.opposite()
        return GameState(
            board=board,
            color=next_player,
            cache=self._cache,        # always the same instance
        )

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
        # 0.  last-second physical validation
        if self._state.board.piece_at(move.from_coord) is None:
            return MoveReceipt(
                new_state=self._state,
                is_legal=False,
                is_game_over=self.is_game_over(),
                result=self.result(),
                message=f"No piece at {move.from_coord}",
            )

        # 1.  game already over → reject
        if self.is_game_over():
            return MoveReceipt(
                new_state=self._state,
                is_legal=False,
                is_game_over=True,
                result=self.result(),
                message="Game already finished.",
            )

        # 2.  standard legality check
        if move not in self._state.legal_moves():
            return MoveReceipt(
                new_state=self._state,
                is_legal=False,
                is_game_over=False,
                message="Illegal move.",
            )
        piece_check = self._state.board.piece_at(move.from_coord)
        if piece_check is None:
            print(f"PRE-APPLY: No piece at {move.from_coord}")
            print(f"Board hash: {self._state.board.byte_hash():016x}")
            # List nearby pieces
            for dz in [-1,0,1]:
                for dy in [-1,0,1]:
                    for dx in [-1,0,1]:
                        cx, cy, cz = move.from_coord[0]+dx, move.from_coord[1]+dy, move.from_coord[2]+dz
                        if 0 <= cx < 9 and 0 <= cy < 9 and 0 <= cz < 9:
                            p = self._state.board.piece_at((cx, cy, cz))
                            if p:
                                print(f"  Neighbor at {(cx,cy,cz)}: {p}")
            raise AssertionError("Piece missing at submission time!")
        # 3.  try to make the move – catch "piece disappeared" errors
        try:
            new_state = self._state.make_move(move)
        except ValueError as e:
            if "empty start square" in str(e):
                return MoveReceipt(
                    new_state=self._state,
                    is_legal=False,
                    is_game_over=False,
                    message=str(e),
                )
            raise   # any other error is real – re-raise

        self._state = new_state
        return MoveReceipt(
            new_state=new_state,
            is_legal=True,
            is_game_over=new_state.is_game_over(),
            result=new_state.result(),
            message="",
        )

    def submit_archery_attack(self, target_sq: Tuple[int, int, int]) -> MoveReceipt:
        """Archer player fires at target_sq instead of moving a piece."""
        # Assuming there's a method to check if current player controls an archer
        # You'll need to implement this helper in GameState
        if not self._state.current_player_controls_archer():
            return MoveReceipt(
                new_state=self._state,
                is_legal=False,
                is_game_over=False,
                message="No archer controlled.",
            )

        if not self._cache.is_valid_archery_attack(target_sq, self._state.current):
            return MoveReceipt(
                new_state=self._state,
                is_legal=False,
                is_game_over=False,
                message="Invalid archery target.",
            )

        # Create an archery move
        archer_move = Move(
            from_coord=target_sq,
            to_coord=target_sq,
            is_capture=True,
            metadata={"is_archery": True},
        )

        # Apply the attack to the board
        new_board = self._state.board.clone()
        new_board.set_piece(target_sq, None)

        # Update history
        new_history = self._state.history + (archer_move,)  # Using tuple concatenation

        # Apply move to cache
        self._cache.apply_move(archer_move, self._state.current)

        # Create new state
        new_state = GameState(
            board=new_board,
            color=self._state.current.opposite(),
            cache=self._cache,
            history=new_history,
            halfmove_clock=self._state.halfmove_clock + 1,  # Increment clock
        )

        # Update internal state
        self._state = new_state

        return MoveReceipt(
            new_state=new_state,
            is_legal=True,
            is_game_over=new_state.is_game_over(),
            result=new_state.result(),
            message="Archery strike!",
        )

    def submit_hive_turn(self, moves: list) -> MoveReceipt:
        if not moves:
            return MoveReceipt(self._state, False, False, message="No moves submitted.")

        # Check if it's the current player's turn
        if not all(self._state.board.piece_at(mv.from_coord).color == self._state.current for mv in moves if self._state.board.piece_at(mv.from_coord)):
            return MoveReceipt(self._state, False, False, message="Not your turn.")

        # Check if all moves are Hive pieces
        for mv in moves:
            p = self._state.board.piece_at(mv.from_coord)
            if p is None or p.ptype != PieceType.HIVE or p.color != self._state.current:
                return MoveReceipt(self._state, False, False, message="Only Hive pieces may move.")

        # Get all legal moves and check if non-hive moves exist
        all_legal = self._state.legal_moves()
        non_hive_moves = [m for m in all_legal if self._state.board.piece_at(m.from_coord).ptype != PieceType.HIVE]
        if non_hive_moves:
            return MoveReceipt(self._state, False, False, message="You must move **only** Hive pieces this turn.")

        # Validate each move
        for mv in moves:
            if mv not in all_legal:
                return MoveReceipt(self._state, False, False, message=f"Illegal move: {mv}")

        # Apply all moves
        new_board = self._state.board.clone()
        new_history = list(self._state.history)  # Convert to list for modification
        for mv in moves:
            new_board.apply_move(mv)
            new_history.append(mv)

        # Convert back to tuple
        new_history = tuple(new_history)

        new_state = GameState(
            board=new_board,
            color=self._state.current.opposite(),
            cache=self._cache,
            history=new_history,
            halfmove_clock=0,
        )

        # Update internal state
        self._state = new_state

        return MoveReceipt(
            new_state=new_state,
            is_legal=True,
            is_game_over=new_state.is_game_over(),
            result=new_state.result(),
            message="Hive turn completed.",
        )

    def reset(self, start_state: Optional[GameState] = None) -> None:
        """Start a new game (or from custom position)."""
        board = Board.startpos()
        init_cache_manager(board)  # Reinitialize cache manager
        self._cache = get_cache_manager()
        self._state = start_state or GameState.start(cache=self._cache)

    def __repr__(self) -> str:
        return f"Game3D({self._state})"
