"""Optimized 3D game engine - orchestration layer with delegation to specialized modules."""

from __future__ import annotations
import time
import numpy as np
from typing import Optional, Any
from collections import namedtuple
import logging
logger = logging.getLogger(__name__)

# Core imports for orchestration
from game3d.common.shared_types import Color, Result, INDEX_DTYPE, FLOAT_DTYPE, COORD_DTYPE, format_bounds_error, PieceType
from game3d.movement.movepiece import Move
from game3d.cache.manager import OptimizedCacheManager
from game3d.board.board import Board
from game3d.game.gamestate import GameState, PerformanceMetrics
from game3d.game.terminal import is_game_over as _terminal_is_game_over, result as _terminal_result

# Delegated functionality
from game3d.game import turnmove

# Frontend interface
MoveReceipt = namedtuple('MoveReceipt', [
    'new_state', 'is_legal', 'is_game_over', 'result',
    'message', 'move_time_ms', 'cache_stats'
])

class MoveValidationError(ValueError):
    """Raised when move validation fails."""
    pass

class InvalidMoveError(MoveValidationError):
    """Raised when a move is illegal."""
    pass

class OptimizedGame3D:
    """Orchestration layer for 3D game engine - delegates all logic to specialized modules."""

    __slots__ = ("_state", "_move_history", "_cache_manager")

    def __init__(self, *, board: Board, cache: OptimizedCacheManager) -> None:
        """Initialize game with board and cache manager."""
        if board is None:
            raise ValueError("OptimizedGame3D requires a valid board instance")
        if cache is None:
            raise ValueError("OptimizedGame3D requires a valid cache manager")

        # Link board and cache manager
        cache.board = board
        board.cache_manager = cache

        self._cache_manager = cache
        self._state = GameState(
            board=board,
            color=Color.WHITE,
            cache_manager=cache,
        )
        self._move_history = np.array([], dtype=object)  # Simplified history tracking

    @property
    def state(self) -> GameState:
        """Get current game state."""
        return self._state

    @property
    def current_player(self) -> Color:
        """Get current player color."""
        return self._state.color

    def submit_move(self, move: Move) -> MoveReceipt:
        """
        Submit a move for execution. RAISES InvalidMoveError on validation failure.

        Only returns receipt for LEGAL moves. All errors are exceptions.
        """
        start_time = time.perf_counter()

        # === CRITICAL VALIDATION: Delegated to turnmove ===
        validation_error = turnmove.validate_move_integrated(self._state, move)
        if validation_error:
            logger.error(f"❌ MOVE REJECTED - {validation_error}")
            raise InvalidMoveError(validation_error)

        # === EXECUTE MOVE: Delegated to turnmove ===
        move_array = np.concatenate([move.from_coord, move.to_coord])

        # Check if it's a hive move
        piece_data = self._state.cache_manager.occupancy_cache.get(move.from_coord)
        is_hive = piece_data and piece_data["piece_type"] == PieceType.HIVE

        try:
            if is_hive:
                new_state = turnmove.execute_hive_move(self._state, move)
            else:
                new_state = turnmove.make_move(self._state, move_array)
        except Exception as e:
            logger.critical(f"🚨 MOVE EXECUTION FAILED: {e}", exc_info=True)
            raise InvalidMoveError(f"Move execution failed: {e}") from e

        # Update internal state
        self._state = new_state
        move_time = time.perf_counter() - start_time

        # Append to history
        self._append_move_history(move, move_time)

        # 🎯 FIX: Fetch cache stats
        cache_stats = self._cache_manager.get_cache_statistics()

        return MoveReceipt(
            new_state=new_state,
            is_legal=True,
            is_game_over=_terminal_is_game_over(new_state),
            result=_terminal_result(new_state),
            message="",
            move_time_ms=move_time * 1000,
            # 🎯 FIX: Include the required argument
            cache_stats=cache_stats
        )

    def _append_move_history(self, move: Move, move_time: float) -> None:
        """Append move to history with timing."""
        # Simple tuple storage for frontend use
        entry = (move, move_time, self._state.turn_number, self._state.color)
        self._move_history = np.append(self._move_history, entry)

    def reset(self, start_state: Optional[GameState] = None) -> None:
        """
        Reset game to initial or specified state.
        Delegates cache rebuild to cache manager.
        """
        if start_state:
            self._state = start_state
            self._cache_manager = start_state.cache_manager
        else:
            # Use factory method for clean initialization
            self._state = GameState.from_startpos()
            self._cache_manager = self._state.cache_manager

        self._move_history = np.array([], dtype=object)

    def get_move_history(self) -> np.ndarray:
        """Get copy of move history."""
        return self._move_history.copy()

    def clone(self) -> 'OptimizedGame3D':
        """
        Create a deep copy of game state with properly synchronized cache.
        """
        # Clone state
        new_state = self._state.clone()

        # Create new game instance
        new_game = OptimizedGame3D(board=new_state.board, cache=new_state.cache_manager)
        new_game._state = new_state
        new_game._move_history = self._move_history.copy()

        return new_game

    # Delegated properties and checks
    @property
    def cache_manager(self) -> OptimizedCacheManager:
        return self._cache_manager

    def is_game_over(self) -> bool:
        """Check if game is over (delegated to terminal module)."""
        if not hasattr(self, '_state') or self._state is None:
            raise RuntimeError("Game state not initialized")
        return _terminal_is_game_over(self._state)

    def result(self) -> Any:
        """Get game result (delegated to terminal module)."""
        if not hasattr(self, '_state') or self._state is None:
            raise RuntimeError("Game state not initialized")
        return _terminal_result(self._state)

    def __repr__(self) -> str:
        return f"OptimizedGame3D(state={self._state})"

    def __str__(self) -> str:
        return f"OptimizedGame3D(current_player={self.current_player}, turn={self._state.turn_number})"
