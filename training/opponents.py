"""
training/opponents.py - FIXED to avoid creating temporary states
Defines opponent types with reward logic that doesn't clone states.
"""

from typing import List, Optional, Tuple, Dict, Any
from game3d.common.enums import Color, PieceType
from game3d.game.gamestate import GameState
from game3d.movement.movepiece import Move
from game3d.common.piece_utils import find_king

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
    def __init__(self, color: Color):
        self.color = color
        self.visited_positions = set()
        self.position_counter = {}
        self.repetition_penalty = -1.0

    def _get_position_key(self, state: GameState, move: Move) -> str:
        """Create a unique key for position + move combination."""
        # Use board hash + current move
        board_hash = state.board.byte_hash()
        move_key = f"{move.from_coord}-{move.to_coord}"
        return f"{board_hash}:{move_key}:{state.color}"

    def reward(self, state: GameState, move: Move) -> float:
        raise NotImplementedError

    def observe(self, state: GameState, move: Move):
        """Update opponent's internal state after move."""
        position_key = self._get_position_key(state, move)

        # Track position frequency
        self.position_counter[position_key] = self.position_counter.get(position_key, 0) + 1

        # Add to visited positions
        self.visited_positions.add(position_key)

    def get_repetition_penalty(self, state: GameState, move: Move) -> float:
        """Get penalty for repetitive positions."""
        position_key = self._get_position_key(state, move)
        count = self.position_counter.get(position_key, 0)

        if count >= 3:
            return self.repetition_penalty * 2  # Heavy penalty for 3+ repetitions
        elif count >= 2:
            return self.repetition_penalty  # Standard penalty for 2 repetitions
        return 0.0

class AdversarialOpponent(OpponentBase):
    """Adversarial opponent: rewards captures, checks, and tactical moves."""
    def reward(self, state: GameState, move: Move) -> float:
        reward = 0.0
        cache_manager = state.cache_manager

        repetition_penalty = self.get_repetition_penalty(state, move)
        reward += repetition_penalty

        # Reward for captures
        captured = cache_manager.occupancy_cache.get(move.to_coord)
        if captured and captured.color == self.color.opposite():
            reward += 0.2
            if is_priest(captured):
                reward += 1.0  # Big reward for priests

        # Reward for moving to attacked squares (aggressive)
        attacked_by_enemy = cache_manager.get_attacked_squares(self.color.opposite())
        if move.to_coord in attacked_by_enemy:
            # Only if we're capturing or it's a sacrifice
            if captured:
                reward += 0.1  # Aggressive capture

        # Reward for attacking enemy king square (without simulating)
        enemy_king_pos = find_king(state, self.color.opposite())
        if enemy_king_pos:
            # Check if this move would attack the king square
            # This is a heuristic - doesn't require full simulation
            from_piece = cache_manager.occupancy_cache.get(move.from_coord)
            if from_piece:
                # Simple heuristic: if move brings us closer to king
                from game3d.common.coord_utils import manhattan_distance
                old_dist = manhattan_distance(move.from_coord, enemy_king_pos)
                new_dist = manhattan_distance(move.to_coord, enemy_king_pos)
                if new_dist < old_dist:
                    reward += 0.05  # Small reward for approaching king

        # Penalty for moving into danger (leaving piece attacked)
        my_attacked_squares = cache_manager.get_attacked_squares(self.color)
        if move.from_coord in my_attacked_squares and move.to_coord not in my_attacked_squares:
            reward -= 0.1  # Moving out of danger is good

        return reward


class CenterControlOpponent(OpponentBase):
    """Opponent that favors controlling the center."""
    def reward(self, state: GameState, move: Move) -> float:
        reward = 0.0
        cache_manager = state.cache_manager

        repetition_penalty = self.get_repetition_penalty(state, move)
        reward += repetition_penalty

        # Reward for moving to center
        if is_center(move.to_coord):
            reward += 0.3

        # Reward for captures
        captured = cache_manager.occupancy_cache.get(move.to_coord)
        if captured and captured.color == self.color.opposite():
            reward += 0.2
            if is_priest(captured):
                reward += 1.0

        # Reward for controlling center squares with attacks
        # (without full simulation - use piece type heuristics)
        from_piece = cache_manager.occupancy_cache.get(move.from_coord)
        if from_piece and from_piece.ptype in [PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN]:
            # These pieces control center well
            if is_center(move.to_coord):
                reward += 0.1

        return reward


class PieceCaptureOpponent(OpponentBase):
    """Opponent that prioritizes capturing pieces."""
    def reward(self, state: GameState, move: Move) -> float:
        reward = 0.0
        cache_manager = state.cache_manager

        repetition_penalty = self.get_repetition_penalty(state, move)
        reward += repetition_penalty

        captured = cache_manager.occupancy_cache.get(move.to_coord)
        if captured and captured.color == self.color.opposite():
            reward += 0.5  # Base capture reward
            if is_priest(captured):
                reward += 1.0

            # Bonus for capturing high-value pieces
            if captured.ptype in [PieceType.QUEEN, PieceType.ROOK]:
                reward += 0.3

        return reward


class PriestHunterOpponent(OpponentBase):
    """Opponent that specifically hunts priests."""
    def reward(self, state: GameState, move: Move) -> float:
        reward = 0.0
        cache_manager = state.cache_manager

        repetition_penalty = self.get_repetition_penalty(state, move)
        reward += repetition_penalty

        captured = cache_manager.occupancy_cache.get(move.to_coord)
        if captured and captured.color == self.color.opposite():
            if is_priest(captured):
                reward += 3.0  # Huge reward for priests
            else:
                reward += 0.2  # Small reward for other captures

        # Reward for moving closer to enemy priests
        from game3d.common.piece_utils import get_pieces_by_type
        enemy_priests = get_pieces_by_type(state.board, PieceType.PRIEST, self.color.opposite(), cache_manager)

        if enemy_priests:
            from game3d.common.coord_utils import manhattan_distance
            # Find closest priest
            min_old_dist = float('inf')
            min_new_dist = float('inf')

            for priest_coord, _ in enemy_priests:
                old_dist = manhattan_distance(move.from_coord, priest_coord)
                new_dist = manhattan_distance(move.to_coord, priest_coord)
                min_old_dist = min(min_old_dist, old_dist)
                min_new_dist = min(min_new_dist, new_dist)

            if min_new_dist < min_old_dist:
                reward += 0.1  # Reward for getting closer to priest

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
