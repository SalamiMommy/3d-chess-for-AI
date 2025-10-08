from __future__ import annotations
"""
game3d/game/game3d.py
Game engine that manages game rules and provides high-level API.
"""

import time
from typing import List, Tuple, Optional, Dict, Any

from game3d.pieces.enums import Color, Result
from game3d.movement.movepiece import Move, MoveReceipt
from game3d.cache.manager import get_cache_manager
from game3d.board.board import Board
from game3d.game.gamestate import GameState
from game3d.game.gamestate import GameMode


class OptimizedGame3D:
    __slots__ = (
        "_state",
        "_game_mode",
        "_debug_turn_info",
        "_move_history",
    )

    def __init__(self,
                game_mode: GameMode = GameMode.STANDARD,
                *, debug_turn_info: bool = True) -> None:

        # Initialize board and cache
        board = Board.startpos()
        current_color = Color.WHITE

        # Initialize cache manager
        cache = get_cache_manager(board, current_color)

        # Initialize game state
        self._state = GameState(
            board=board,
            color=current_color,
            cache=cache,
            game_mode=game_mode
        )
        self._game_mode = game_mode
        self._move_history: List[Tuple[Move, float]] = []  # Move and processing time
        self._debug_turn_info: bool = debug_turn_info

    # ---------- PUBLIC API ----------
    @property
    def state(self) -> GameState:
        """Current position (immutable)."""
        return self._state

    @property
    def current_player(self) -> Color:
        return self._state.color

    def is_game_over(self) -> bool:
        return self._state.is_game_over()

    def result(self) -> Optional[Result]:
        return self._state.result()

    def toggle_debug_turn_info(self, value: bool | None = None) -> bool:
        """Toggle (or set) turn debug print-outs.  Returns new state."""
        if value is None:
            self._debug_turn_info = not self._debug_turn_info
        else:
            self._debug_turn_info = bool(value)
        return self._debug_turn_info

    # ---------- ENHANCED MOVE SUBMISSION ----------
    def submit_move(self, move: Move) -> MoveReceipt:
        start_time = time.perf_counter()

        if self._debug_turn_info:
            print(f"[Turn {self._state.turn_number}] {self.current_player.name} submits {move}")

        # Process move (validation handled in make_move)
        try:
            new_state = self._state.make_move(move)

            # Update history
            self._move_history.append((move, time.perf_counter() - start_time))
            self._state = new_state

            return MoveReceipt(
                new_state=new_state,
                is_legal=True,
                is_game_over=new_state.is_game_over(),
                result=new_state.result(),
                message="",
                move_time_ms=(time.perf_counter() - start_time) * 1000,
                cache_stats=self._get_cache_stats()
            )

        except ValueError as e:
            return self._create_error_receipt(str(e), start_time)

    # REMOVED: _validate_move_fast function (moved to validation.py)

    # ---------- UTILITY METHODS ----------
    def _create_error_receipt(self, message: str, start_time: float) -> MoveReceipt:
        """Create error move receipt with performance tracking."""
        return MoveReceipt(
            new_state=self._state,
            is_legal=False,
            is_game_over=self.is_game_over(),
            result=self.result(),
            message=message,
            move_time_ms=(time.perf_counter() - start_time) * 1000,
            cache_stats=self._get_cache_stats()
        )

    def _get_cache_stats(self) -> Dict[str, Any]:
        """Get current cache statistics."""
        if hasattr(self._state.cache, 'get_stats'):
            return self._state.cache.get_stats()
        return {}

    def reset(self, start_state: Optional[GameState] = None) -> None:
        """Reset game with optional starting state."""
        if start_state:
            self._state = start_state
        else:
            board = Board.startpos()
            current_color = Color.WHITE
            cache = get_cache_manager(board, current_color)
            self._state = GameState(
                board=board,
                color=current_color,
                cache=cache,
                game_mode=self._game_mode
            )

        self._move_history.clear()

    def get_move_history(self) -> List[Tuple[Move, float]]:
        """Get move history with processing times."""
        return self._move_history.copy()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        return self._state.get_performance_stats()

    def __repr__(self) -> str:
        return f"OptimizedGame3D({self._state}, mode={self._game_mode.value})"
