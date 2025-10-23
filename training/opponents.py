"""
training/opponents.py
Defines opponent types for self-play in 3D chess, with custom reward logic.
"""

from typing import List, Optional, Tuple, Dict, Any
from game3d.common.enums import Color, PieceType
from game3d.game.gamestate import GameState
from game3d.movement.movepiece import Move
from game3d.cache.manager import get_cache_manager
from game3d.board.board import Board

# Utility: Center squares (middle layer, center of board)
CENTER_SQUARES = [
    (x, y, z)
    for x in range(3, 5)
    for y in range(3, 5)
    for z in range(3, 5)
]

def is_center(coord: Tuple[int, int, int]) -> bool:
    return coord in CENTER_SQUARES

def is_priest(piece) -> bool:
    return piece is not None and piece.ptype == PieceType.PRIEST

def is_king(piece) -> bool:
    return piece is not None and piece.ptype == PieceType.KING


class OpponentBase:
    """Base class for all opponents."""
    def __init__(self, color: Color):
        self.color = color

    def _get_simulated_state(self, state: GameState, move: Move) -> GameState:
        """Create an isolated simulated state after the move."""
        temp_board = Board(state.board.tensor().clone())
        temp_cache = get_cache_manager(temp_board, state.color)
        temp_state = GameState(
            board=temp_board,
            color=state.color,
            cache=temp_cache,
            history=state.history,
            halfmove_clock=state.halfmove_clock,
            game_mode=state.game_mode,
            turn_number=state.turn_number,
        )
        next_state = temp_state.make_move(move)
        return next_state

    def reward(self, state: GameState, move: Move) -> float:
        """Compute reward for a move given the current state."""
        raise NotImplementedError

    def observe(self, state: GameState):
        """Update opponent's internal state if needed."""
        pass


class AdversarialOpponent(OpponentBase):
    """Adversarial opponent: tries to win, rewards for check, priest capture."""
    def reward(self, state: GameState, move: Move) -> float:
        reward = 0.0

        # Reward for capturing any piece
        captured = state.cache_manager_manager.occupancy.get(move.to_coord)
        if captured and captured.color == self.color.opposite():
            reward += 0.2
            if is_priest(captured):
                reward += 1.0  # Big reward for priests

        # Small reward for capturing opposite priest
        if captured and is_priest(captured) and captured.color == self.color.opposite():
            reward += 0.5

        # Small reward for putting king in check (after move)
        next_state = self._get_simulated_state(state, move)
        opp_king_pos = next_state.cache_manager.find_king(self.color.opposite())
        if opp_king_pos is not None:
            attacked = next_state.cache_manager.move.get_attacked_squares(self.color)
            if opp_king_pos in attacked:
                reward += 0.2

        # Penalty for being in check
        my_king_pos = next_state.cache_manager.find_king(self.color)
        if my_king_pos is not None:
            attacked = next_state.cache_manager.move.get_attacked_squares(self.color.opposite())
            if my_king_pos in attacked:
                reward -= 0.2

        return reward


class CenterControlOpponent(OpponentBase):
    """Opponent that favors moving to/controlling the center."""
    def reward(self, state: GameState, move: Move) -> float:
        reward = 0.0

        # Small reward for moving toward center
        if is_center(move.to_coord):
            reward += 0.3

        # Small reward for capturing any piece
        captured = state.cache_manager.occupancy.get(move.to_coord)
        if captured and captured.color == self.color.opposite():
            reward += 0.2
            if is_priest(captured):
                reward += 1.0

        # Small reward for capturing opposite priest
        if captured and is_priest(captured) and captured.color == self.color.opposite():
            reward += 0.5

        # Small reward for putting king in check
        next_state = self._get_simulated_state(state, move)
        opp_king_pos = next_state.cache_manager.find_king(self.color.opposite())
        if opp_king_pos is not None:
            attacked = next_state.cache_manager.move.get_attacked_squares(self.color)
            if opp_king_pos in attacked:
                reward += 0.2

        # Penalty for being in check
        my_king_pos = next_state.cache_manager.find_king(self.color)
        if my_king_pos is not None:
            attacked = next_state.cache_manager.move.get_attacked_squares(self.color.opposite())
            if my_king_pos in attacked:
                reward -= 0.2

        return reward


class PieceCaptureOpponent(OpponentBase):
    """Opponent that gets a small reward for capturing any piece."""
    def reward(self, state: GameState, move: Move) -> float:
        reward = 0.0

        captured = state.cache_manager.occupancy.get(move.to_coord)
        if captured and captured.color == self.color.opposite():
            reward += 0.5  # Small reward for any capture
            if is_priest(captured):
                reward += 1.0

        # Small reward for capturing opposite priest
        if captured and is_priest(captured) and captured.color == self.color.opposite():
            reward += 0.5

        # Small reward for putting king in check
        next_state = self._get_simulated_state(state, move)
        opp_king_pos = next_state.cache_manager.find_king(self.color.opposite())
        if opp_king_pos is not None:
            attacked = next_state.cache_manager.move.get_attacked_squares(self.color)
            if opp_king_pos in attacked:
                reward += 0.2

        # Penalty for being in check
        my_king_pos = next_state.cache_manager.find_king(self.color)
        if my_king_pos is not None:
            attacked = next_state.cache_manager.move.get_attacked_squares(self.color.opposite())
            if my_king_pos in attacked:
                reward -= 0.2

        return reward


class PriestHunterOpponent(OpponentBase):
    """Opponent that gets a big reward for capturing priests."""
    def reward(self, state: GameState, move: Move) -> float:
        reward = 0.0

        captured = state.cache_manager.occupancy.get(move.to_coord)
        if captured and captured.color == self.color.opposite():
            if is_priest(captured):
                reward += 3.0  # Big reward for priests
            else:
                reward += 0.2  # Small reward for other captures

        # Small reward for capturing opposite priest
        if captured and is_priest(captured) and captured.color == self.color.opposite():
            reward += 0.5

        # Small reward for putting king in check
        next_state = self._get_simulated_state(state, move)
        opp_king_pos = next_state.cache_manager.find_king(self.color.opposite())
        if opp_king_pos is not None:
            attacked = next_state.cache_manager.move.get_attacked_squares(self.color)
            if opp_king_pos in attacked:
                reward += 0.2

        # Penalty for being in check
        my_king_pos = next_state.cache_manager.find_king(self.color)
        if my_king_pos is not None:
            attacked = next_state.cache_manager.move.get_attacked_squares(self.color.opposite())
            if my_king_pos in attacked:
                reward -= 0.2

        return reward


# Factory to create opponents by name
def create_opponent(opponent_type: str, color: Color) -> OpponentBase:
    types = {
        'adversarial': AdversarialOpponent,
        'center_control': CenterControlOpponent,
        'piece_capture': PieceCaptureOpponent,
        'priest_hunter': PriestHunterOpponent,
    }
    if opponent_type not in types:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    return types[opponent_type](color)


# List of available opponent types
AVAILABLE_OPPONENTS = [
    'adversarial',
    'center_control',
    'piece_capture',
    'priest_hunter',
]

__all__ = [
    "OpponentBase",
    "AdversarialOpponent",
    "CenterControlOpponent",
    "PieceCaptureOpponent",
    "PriestHunterOpponent",
    "create_opponent",
    "AVAILABLE_OPPONENTS",
]
