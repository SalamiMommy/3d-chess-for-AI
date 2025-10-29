# __init__.py - Fixed version
"""
game3d/game/__init__.py
Central initialization that binds all game functionality to GameState.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

# Import core classes first
from .gamestate import GameState
from .performance import PerformanceMetrics, track_operation_time, track_performance
from game3d.cache.caches.zobrist import compute_zobrist

# Import factory functions
from .factory import start_game_state, create_game_state_from_tensor, clone_game_state_for_search
# Import terminal/outcome functions
from .terminal import (
    is_check,
    is_stalemate,
    is_insufficient_material,
    is_fifty_move_draw,
    is_game_over,
    result,
    is_terminal,
    outcome,
    insufficient_material
)

# Import moveeffects (now has archery and hive moves)
from .moveeffects import apply_archery_attack, apply_hive_moves

# Removed redundant factory redefinitions - use from factory.py

# ------------------------------------------------------------------
# BIND ALL METHODS TO GAMESTATE CLASS
# ------------------------------------------------------------------

# Terminal/outcome methods
GameState.is_check = is_check
GameState.is_stalemate = is_stalemate
GameState.is_insufficient_material = is_insufficient_material
GameState.is_fifty_move_draw = is_fifty_move_draw
GameState.is_game_over = is_game_over
GameState.result = result
GameState.is_terminal = is_terminal
GameState.outcome = outcome

# Move generation and execution methods - imported on demand
def _import_turnmove_functions():
    """Import turn-move functions on demand to avoid circular imports."""
    from .turnmove import (
        legal_moves,
        make_move,
        undo_move,
        validate_legal_moves,
    )
    # add the missing name here
    from game3d.movement.pseudo_legal import generate_pseudo_legal_moves as pseudo_legal_moves

    return legal_moves, pseudo_legal_moves, make_move, undo_move, validate_legal_moves

legal_moves, pseudo_legal_moves, make_move, undo_move, validate_legal_moves = _import_turnmove_functions()

GameState.legal_moves = legal_moves
GameState.pseudo_legal_moves = pseudo_legal_moves
GameState.make_move = make_move
GameState.undo_move = undo_move

# Add special move methods
GameState.apply_archery_attack = apply_archery_attack
GameState.apply_hive_moves = apply_hive_moves

# Bind factory functions (from factory.py)
GameState.start_game_state = staticmethod(start_game_state)
GameState.create_game_state_from_tensor = staticmethod(create_game_state_from_tensor)
GameState.clone_game_state_for_search = staticmethod(clone_game_state_for_search)

# ------------------------------------------------------------------
# ADD MISSING BINDINGS FOR CACHE-RELATED METHODS
# ------------------------------------------------------------------

def _bind_cache_methods():
    """Bind cache-related methods that were missing."""
    # Import locally to avoid circular imports
    from game3d.common.enums import PieceType

    # Add methods that were referenced in other modules but not bound
    if not hasattr(GameState, 'has_priest'):
        GameState.has_priest = lambda self, color: self.cache_manager.has_priest(color)

    if not hasattr(GameState, 'any_priest_alive'):
        GameState.any_priest_alive = lambda self: self.cache_manager.any_priest_alive()

    if not hasattr(GameState, 'find_king'):
        GameState.find_king = lambda self, color: self.cache_manager.find_king(color)

    if not hasattr(GameState, 'get_attacked_squares'):
        GameState.get_attacked_squares = lambda self, color: self.cache_manager.get_attacked_squares(color)

# Call the binding function
_bind_cache_methods()

# ------------------------------------------------------------------
# EXPORTS
# ------------------------------------------------------------------

__all__ = [
    # Core classes
    'GameState',
    'PerformanceMetrics',

    # Factory functions
    'start_game_state',
    'create_game_state_from_tensor',
    'clone_game_state_for_search',

    # Zobrist
    'compute_zobrist',

    # Terminal functions
    'is_check',
    'is_stalemate',
    'is_insufficient_material',
    'is_fifty_move_draw',
    'is_game_over',
    'result',
    'is_terminal',
    'outcome',
    'insufficient_material',

    # Move functions
    'legal_moves',
    'pseudo_legal_moves',
    'make_move',
    'undo_move',
    'validate_legal_moves',

    # Special game modes
    'apply_archery_attack',
    'apply_hive_moves',

    # Performance
    'track_operation_time',
    'track_performance',
]
