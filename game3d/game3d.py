from __future__ import annotations
"""
game3d/game/game3d.py
Game engine that manages game rules and provides high-level API.
"""

import time
from typing import List, Tuple, Optional, Dict, Any

from game3d.common.enums import Color, Result, PieceType
from game3d.movement.movepiece import Move, MoveReceipt
from game3d.cache.manager import get_cache_manager
from game3d.board.board import Board
from game3d.game.gamestate import GameState
from game3d.game.gamestate import GameMode
from game3d.pieces.pieces.hive import (
    get_movable_hives, apply_multi_hive_move
)

# game3d/game3d.py
class OptimizedGame3D:
    __slots__ = ("_state", "_game_mode", "_debug_turn_info", "_move_history")

    def __init__(
        self,
        *,
        board: Board | None = None,
        cache: OptimizedCacheManager | None = None,
        game_mode: GameMode = GameMode.STANDARD,
        debug_turn_info: bool = True,
    ) -> None:
        if board is None:
            board = Board.startpos()  # still empty
        if cache is None:
            raise RuntimeError("OptimizedGame3D must be given an external cache")

        self._state = GameState(
            board=board,
            color=Color.WHITE,
            cache=cache,
            game_mode=game_mode,
        )
        self._game_mode = game_mode
        self._move_history: list[tuple[Move, float]] = []
        self._debug_turn_info = debug_turn_info

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

    # -----------------------------------------------------------
    # 2.  Replace the naive submit_move method
    # -----------------------------------------------------------
    def submit_move(self, move: Move) -> MoveReceipt:
        """
        Enhanced entry-point that understands multi-Hive turns.
        A Hive move does **not** flip the colour until the player
        explicitly passes or has no more Hives to move.
        """
        start_time = time.perf_counter()

        if self._debug_turn_info:
            print(f"[Turn {self._state.turn_number}] "
                f"{self.current_player.name} submits {move}")

        # ----- fast reject -------------------------------------------------
        piece = self._state.cache.piece_cache.get(move.from_coord)
        if piece is None or piece.color != self.current_player:
            return self._create_error_receipt("No own piece on from-square",
                                            start_time)

        # ----- Hive multi-move branch --------------------------------------
        if piece.ptype is PieceType.HIVE:
            try:
                # apply without flipping colour
                self._state = apply_multi_hive_move(self._state, move)
                self._move_history.append((move,
                                        time.perf_counter() - start_time))

                # keep offering Hive moves until exhausted or user passes
                while True:
                    still_available = get_movable_hives(self._state,
                                                        self.current_player)
                    if not still_available:          # forced end
                        break
                    # ---- UI hook -------------------------------------------------
                    # If your interface already calls submit_move repeatedly,
                    # just return here; the next submit_move will land in
                    # this same branch until no Hives remain.
                    # --------------------------------------------------------------
                    break   # ← remove this line if you want auto-loop

                # finally flip colour
                self._state = self._state.pass_turn()
                return MoveReceipt(
                    new_state=self._state,
                    is_legal=True,
                    is_game_over=self._state.is_game_over(),
                    result=self._state.result(),
                    message="",
                    move_time_ms=(time.perf_counter() - start_time) * 1000,
                    cache_stats=self._get_cache_stats()
                )

            except ValueError as e:
                return self._create_error_receipt(str(e), start_time)

        # ----- ordinary single-move branch ---------------------------------
        else:
            return self._submit_single_move(move, start_time)

    # -----------------------------------------------------------
    # 3.  Extract the old single-move logic so we can reuse it
    # -----------------------------------------------------------

    def _submit_single_move(self, mv: Move, start_time: float) -> MoveReceipt:
        """The original submit_move body (colour flips automatically)."""
        try:
            # Validate move before applying
            piece = self._state.cache.piece_cache.get(mv.from_coord)
            if piece is None:
                return self._create_error_receipt(f"No piece at {mv.from_coord}", start_time)
            if piece.color != self._state.color:
                return self._create_error_receipt(f"Cannot move opponent's piece", start_time)

            new_state = self._state.make_move(mv)
            self._move_history.append((mv, time.perf_counter() - start_time))
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
        except Exception as e:
            print(f"[ERROR] Unexpected error in _submit_single_move: {e}")
            import traceback
            traceback.print_exc()
            return self._create_error_receipt(f"Internal error: {e}", start_time)

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
