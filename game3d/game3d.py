"""Optimized 3D game engine - orchestration layer with delegation to specialized modules."""

from __future__ import annotations
import time
import numpy as np
from typing import Optional, Any
from collections import namedtuple
import logging
logger = logging.getLogger(__name__)
# Core imports for orchestration
from game3d.common.shared_types import (
    Color, Result, INDEX_DTYPE, FLOAT_DTYPE, COORD_DTYPE,
    format_bounds_error, PieceType
)
from game3d.movement.movepiece import Move
from game3d.cache.manager import OptimizedCacheManager
from game3d.board.board import Board
from game3d.game.gamestate import GameState
from game3d.game.terminal import is_game_over as _terminal_is_game_over, result as _terminal_result
from game3d.common.coord_utils import in_bounds_vectorized
from game3d.common.validation import validate_move
from game3d.pieces.pieces.hive import get_movable_hives, apply_multi_hive_move

# Corrected delegation import
from game3d.game import turnmove

# Simple MoveReceipt class
MoveReceipt = namedtuple('MoveReceipt', [
    'new_state', 'is_legal', 'is_game_over', 'result',
    'message', 'move_time_ms', 'cache_stats'
])

# Constants for move history dtype
MOVE_HISTORY_DTYPE = np.dtype([
    ('move', object),
    ('time', FLOAT_DTYPE)
])

class MoveValidationError(ValueError):
    """Raised when move validation fails."""
    pass

class InvalidMoveError(MoveValidationError):
    """Raised when a move is illegal."""
    pass

class OptimizedGame3D:
    """Orchestration layer for 3D game engine."""

    __slots__ = ("_state", "_move_history", "_cache_manager")

    def __init__(
        self,
        *,
        board: Board,
        cache: OptimizedCacheManager,
    ) -> None:
        """Initialize game with board and cache manager."""
        if board is None:
            raise ValueError("OptimizedGame3D requires a valid board instance")
        if cache is None:
            raise ValueError("OptimizedGame3D requires a valid cache manager")

        # CRITICAL FIX: Link board and cache manager ↓↓↓
        board.cache_manager = cache  # This sets board._cache_manager via the property
        cache.board = board  # Ensure cache manager knows about board

        self._cache_manager = cache
        self._state = GameState(
            board=board,
            color=Color.WHITE,
            cache_manager=cache,
        )
        self._move_history = np.array([], dtype=MOVE_HISTORY_DTYPE)

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
        Submit a move for execution. RAISES InvalidMoveError on validation failure (LOUD FAILURE).

        Only returns receipt for LEGAL moves. All errors are exceptions.
        """
        start_time = time.perf_counter()

        # === CRITICAL VALIDATION: BOUNDS ===
        try:
            from_coord_valid = in_bounds_vectorized(move.from_coord)
            to_coord_valid = in_bounds_vectorized(move.to_coord)
        except Exception as e:
            logger.critical(f"🚨 COORDINATE VALIDATION CRASHED: {e}", exc_info=True)
            raise InvalidMoveError(f"Coordinate validation error: {e}") from e

        if not (from_coord_valid and to_coord_valid):
            error_msg = format_bounds_error(np.array([move.from_coord, move.to_coord]))
            logger.error(f"❌ MOVE REJECTED - Out of bounds: {error_msg}")
            logger.debug(f"   from: {move.from_coord}, to: {move.to_coord}")
            raise InvalidMoveError(error_msg)

        # === CRITICAL VALIDATION: PIECE EXISTENCE & OWNERSHIP ===
        # === CRITICAL VALIDATION: PIECE EXISTENCE & OWNERSHIP ===
        try:
            from_coord_batch = move.from_coord.reshape(1, 3)  # Make it a batch of size 1
            colors, types = self._state.cache_manager.occupancy_cache.batch_get_attributes(from_coord_batch)

            # ✅ ADDED: Debug logging for piece lookup
            # logger.debug(f"Piece lookup at {move.from_coord}: color={colors[0]}, type={types[0]}")

            # ✅ ADDED: Verify cache has pieces at all
            occ_coords, occ_types, occ_colors = self._state.cache_manager.occupancy_cache.get_all_occupied_vectorized()
            if len(occ_coords) == 0:
                logger.critical(f"🚨 OCCUPANCY CACHE IS EMPTY! Board generation={self._state.board.generation}")
                raise InvalidMoveError("Occupancy cache is empty - board not initialized")

            if colors[0] == 0:  # COLOR_EMPTY = 0
                piece = None
            else:
                piece = {
                    'piece_type': types[0],
                    'color': colors[0]
                }
        except Exception as e:
            logger.critical(f"🚨 OCCUPANCY CACHE LOOKUP FAILED: {e}", exc_info=True)
            raise InvalidMoveError("Cache system failure") from e

        if piece is None:
            logger.error(f"❌ MOVE REJECTED - No piece at {move.from_coord}")
            raise InvalidMoveError(f"No piece at from-square {move.from_coord}")

        if piece["color"] != self.current_player:
            logger.error(f"❌ MOVE REJECTED - Wrong color: {piece['color']} != {self.current_player}")
            raise InvalidMoveError("Not your piece")

        # === VALID MOVE: Continue with delegation ===
        new_state = (self._delegate_hive_move(move, start_time)
                    if piece["piece_type"] is PieceType.HIVE
                    else self._delegate_standard_move(move, start_time))

        self._state = new_state
        move_time = time.perf_counter() - start_time
        self._append_move_history(move, move_time)

        return MoveReceipt(
            new_state=new_state,
            is_legal=True,
            is_game_over=_terminal_is_game_over(new_state),  # FIXED
            result=_terminal_result(new_state),              # FIXED
            message="",
            move_time_ms=move_time * 1000,
            cache_stats=self._get_cache_stats()
        )

    def _delegate_hive_move(self, move: Move, start_time: float) -> GameState:
        """Delegate hive move execution to hive module via turnmove."""
        # Apply hive move through dedicated handler
        new_state = apply_multi_hive_move(self._state, move)

        # Handle pass-turn logic if no more movable hives
        if not get_movable_hives(new_state, self.current_player):
            if hasattr(new_state, 'pass_turn'):
                new_state = new_state.pass_turn()

        return new_state

    def _delegate_standard_move(self, move: Move, start_time: float) -> GameState:
        """Delegate standard move execution to turnmove module."""
        # Convert Move object to numpy array format expected by turnmove
        move_array = np.concatenate([move.from_coord, move.to_coord])
        return turnmove.make_move(self._state, move_array)

    def _append_move_history(self, move: Move, move_time: float) -> None:
        """Append move to history."""
        new_entry = np.array([(move, move_time)], dtype=MOVE_HISTORY_DTYPE)
        self._move_history = np.append(self._move_history, new_entry)

    def _create_error_receipt(self, message: str, start_time: float) -> MoveReceipt:
        """Create error receipt for invalid moves."""
        return MoveReceipt(
            new_state=self._state,
            is_legal=False,
            is_game_over=_terminal_is_game_over(self._state),  # FIXED
            result=_terminal_result(self._state),              # FIXED
            message=message,
            move_time_ms=(time.perf_counter() - start_time) * 1000,
            cache_stats=self._get_cache_stats()
        )

    def _get_cache_stats(self) -> dict:
        """Get cache statistics from cache manager."""
        return self._cache_manager.get_cache_statistics()

    def reset(self, start_state: Optional[GameState] = None) -> None:
        """
        Reset game to initial or specified state.

        Delegates cache rebuild to cache manager.
        """
        if start_state:
            self._state = start_state
            self._cache_manager = start_state.cache_manager
        else:
            board = Board.startpos()
            current_color = Color.WHITE

            # Use cache manager rebuild method
            self._cache_manager.rebuild(board, current_color)
            self._state = GameState(
                board=board,
                color=current_color,
                cache_manager=self._cache_manager,
            )

        self._move_history = np.array([], dtype=MOVE_HISTORY_DTYPE)

    def get_move_history(self) -> np.ndarray:
        """Get copy of move history."""
        return self._move_history.copy()

    def clone(self) -> 'OptimizedGame3D':
        """
        Create a deep copy of game state with properly synchronized cache.

        CRITICAL: Must rebuild cache from board data to ensure consistency.
        """
        # 1. Clone the board array first
        new_board = self._state.board.copy()

        # 2. Create fresh cache manager with the CLONED board
        #    This triggers _rebuild_occupancy_from_board() automatically
        new_cache = OptimizedCacheManager(board=new_board, color=self._state.color)

        # 3. Clone game state (this also copies turn number, history, etc.)
        new_state = self._state.clone()

        # 4. Ensure the cloned state uses our new cache
        new_state.cache_manager = new_cache

        # 5. Create new game instance
        new_game = OptimizedGame3D(board=new_board, cache=new_cache)
        new_game._state = new_state
        new_game._move_history = self._move_history.copy()

        # 6. CRITICAL: Re-sync Zobrist key (it's recomputed during cache rebuild)
        new_state._zkey = new_cache._zkey

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
