# __init__.py - Fixed version
"""
game3d/game/__init__.py
Central initialization that binds all game functionality to GameState.
"""

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

# Import move functions - use late imports to avoid circular dependencies
def _import_turnmove_functions():
    """Import turnmove functions on demand to avoid circular imports."""
    from .turnmove import (
        legal_moves,
        pseudo_legal_moves,
        make_move,
        undo_move,
        apply_hive_moves,
        validate_legal_moves
    )
    return legal_moves, pseudo_legal_moves, make_move, undo_move, apply_hive_moves, validate_legal_moves

# Import move_utils (breaks circular dependency)
from .move_utils import (
    apply_hole_effects,
    apply_bomb_effects,
    apply_trailblaze_effect,
    reconstruct_trailblazer_path,
    extract_enemy_slid_path
)

# Import moveeffects (now only has archery)
from .moveeffects import apply_archery_attack

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
legal_moves, pseudo_legal_moves, make_move, undo_move, apply_hive_moves, validate_legal_moves = _import_turnmove_functions()

GameState.legal_moves = legal_moves
GameState.pseudo_legal_moves = pseudo_legal_moves
GameState.make_move = make_move
GameState.undo_move = undo_move
GameState.apply_hive_moves = apply_hive_moves

# Add missing PieceType import for validation methods
from game3d.pieces.enums import PieceType

# ------------------------------------------------------------------
# FACTORY FUNCTIONS (Add these since they're referenced but not defined)
# ------------------------------------------------------------------

def start_game_state(board_size: int = 9) -> GameState:
    """Create a new game state with initial position."""
    from game3d.board.board import Board
    from game3d.cache.manager import get_cache_manager
    from game3d.pieces.enums import Color

    # Create empty board
    board = Board.create_initial_position()
    cache = get_cache_manager(board, Color.WHITE)

    return GameState(
        board=board,
        color=Color.WHITE,
        cache=cache,
        history=(),
        halfmove_clock=0,
        turn_number=1
    )

def create_game_state_from_tensor(tensor, color: Color) -> GameState:
    """Create game state from tensor representation."""
    from game3d.board.board import Board
    from game3d.cache.manager import get_cache_manager

    board = Board(tensor)
    cache = get_cache_manager(board, color)

    return GameState(
        board=board,
        color=color,
        cache=cache,
        history=(),
        halfmove_clock=0,
        turn_number=1
    )

def clone_game_state_for_search(original: GameState) -> GameState:
    """Create a deep clone for search algorithms."""
    return original.clone_with_new_cache()

# Bind factory functions
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
    'validate_archery_attack',
    'validate_hive_moves',

    # Performance
    'track_operation_time',
    'track_performance',
]
