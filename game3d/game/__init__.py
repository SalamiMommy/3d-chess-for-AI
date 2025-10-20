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
from .zobrist import compute_zobrist
from game3d.pieces.piece import Color
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

# Import move_utils (breaks circular dependency)
from .move_utils import (
    apply_hole_effects,
    apply_bomb_effects,
    apply_trailblaze_effect,
    reconstruct_trailblazer_path,
    extract_enemy_slid_path
)

# Import moveeffects (now has archery and hive moves)
from .moveeffects import apply_archery_attack, apply_hive_moves  # MODIFIED: Added apply_hive_moves

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
    """Import turnmove functions on demand to avoid circular imports."""
    from .turnmove import (
        legal_moves,
        pseudo_legal_moves,
        make_move,
        undo_move,
        validate_legal_moves
    )
    return legal_moves, pseudo_legal_moves, make_move, undo_move, validate_legal_moves

legal_moves, pseudo_legal_moves, make_move, undo_move, validate_legal_moves = _import_turnmove_functions()

GameState.legal_moves = legal_moves
GameState.pseudo_legal_moves = pseudo_legal_moves
GameState.make_move = make_move
GameState.undo_move = undo_move
GameState.apply_hive_moves = apply_hive_moves  # MODIFIED: Now using moveeffects import

# Add missing PieceType import for validation methods
from game3d.common.enums import PieceType

# Bind factory functions (from factory.py)
GameState.start_game_state = staticmethod(start_game_state)
GameState.create_game_state_from_tensor = staticmethod(create_game_state_from_tensor)
GameState.clone_game_state_for_search = staticmethod(clone_game_state_for_search)

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

    # Special effects
    'apply_hole_effects',
    'apply_bomb_effects',
    'apply_trailblaze_effect',
    'reconstruct_trailblazer_path',
    'extract_enemy_slid_path',

    # Special game modes
    'apply_archery_attack',
    'apply_hive_moves',
    # Removed undefined validate_archery_attack, validate_hive_moves - add if implemented

    # Performance
    'track_operation_time',
    'track_performance',
]
