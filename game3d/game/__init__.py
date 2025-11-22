"""Game initialization with numpy array support."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

# Core classes - ENSURE GameState is imported first
from game3d.game.gamestate import GameState  # ADD THIS LINE AT THE TOP

# Factory functions
from game3d.game.factory import start_game_state, create_game_state_from_tensor, clone_game_state_for_search

# Terminal/outcome functions
from game3d.game.terminal import (
    is_check,
    is_stalemate,
    is_insufficient_material,
    is_move_rule_draw,
    is_repetition_draw,
    is_game_over,
    result,
    is_terminal,
    outcome,
    insufficient_material,
)

# Special move effects
# from game3d.game.moveeffects import apply_archery_attack, apply_hive_moves

# Move functions - imported lazily to avoid circular imports
def _get_turnmove_functions():
    """Get turn-move functions for numpy array compatibility."""
    from .turnmove import (
        legal_moves,
        make_move,
        undo_move,
    )
    # Import functions from the correct module
    from game3d.movement.generator import generate_legal_moves as generate_legal_moves_func
    from game3d.movement.generator import validate_moves as batch_validate_moves

    return legal_moves, generate_legal_moves_func, make_move, undo_move, batch_validate_moves


# Bind all methods to GameState class
def _bind_game_state_methods():
    """Bind all game methods to GameState class for numpy array support."""
    # Terminal/outcome methods
    GameState.is_check = staticmethod(is_check)
    GameState.is_stalemate = staticmethod(is_stalemate)
    GameState.is_insufficient_material = staticmethod(is_insufficient_material)
    GameState.is_move_rule_draw = staticmethod(is_move_rule_draw)
    GameState.is_repetition_draw = staticmethod(is_repetition_draw)
    GameState.is_terminal = staticmethod(is_terminal)
    GameState.outcome = staticmethod(outcome)

    # Move methods - imported on demand
    legal_moves, generate_legal_moves_func, make_move, undo_move, batch_validate_moves = _get_turnmove_functions()


    GameState.generate_legal_moves = staticmethod(generate_legal_moves_func)
    GameState.make_move = staticmethod(make_move)
    GameState.undo_move = staticmethod(undo_move)
    GameState.batch_validate_moves = staticmethod(batch_validate_moves)

    # Special move methods
    # GameState.apply_archery_attack = staticmethod(apply_archery_attack)
    # GameState.apply_hive_moves = staticmethod(apply_hive_moves)

    # Factory functions
    GameState.start_game_state = staticmethod(start_game_state)
    GameState.create_game_state_from_tensor = staticmethod(create_game_state_from_tensor)
    GameState.clone_game_state_for_search = staticmethod(clone_game_state_for_search)

    # Cache-related methods
    _bind_cache_methods()


def _bind_cache_methods():
    """Bind cache-related methods for numpy array operations."""
    if not hasattr(GameState, "has_priest"):
        GameState.has_priest = lambda self, color: self.cache_manager.has_priest(color)

    if not hasattr(GameState, "any_priest_alive"):
        GameState.any_priest_alive = lambda self: self.cache_manager.any_priest_alive()

    if not hasattr(GameState, "find_king"):
        GameState.find_king = lambda self, color: self.cache_manager.find_king(color)

    if not hasattr(GameState, "get_attacked_squares"):
        GameState.get_attacked_squares = (
            lambda self, color: self.cache_manager.get_attacked_squares(color)
        )


# Initialize all bindings
_bind_game_state_methods()

# Export all public interface
__all__ = [
    "GameState",
    "PerformanceMetrics",
    "start_game_state",
    "create_game_state_from_tensor",
    "clone_game_state_for_search",
    "is_check",
    "is_stalemate",
    "is_insufficient_material",
    "is_move_rule_draw",
    "is_repetition_draw",
    "is_game_over",
    "result",
    "is_terminal",
    "outcome",
    "insufficient_material",
    "legal_moves",
    "generate_legal_moves",
    "make_move",
    "undo_move",
    "batch_validate_moves",
    # "apply_archery_attack",
    # "apply_hive_moves",
    "track_operation_time",
    "track_performance",
]
