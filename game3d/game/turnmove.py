# turnmove.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, TYPE_CHECKING

from game3d.board.board import Board
from game3d.movement.movepiece import Move, convert_legacy_move_args  # FIXED: Import convert_legacy_move_args
from game3d.movement.generator import generate_legal_moves
from game3d.movement.pseudo_legal import generate_pseudo_legal_moves
from game3d.pieces.enums import Color, PieceType
from game3d.pieces.piece import Piece
from game3d.cache.manager import get_cache_manager
from game3d.attacks.check import _any_priest_alive
from .performance import track_operation_time
from .zobrist import compute_zobrist
from game3d.movement.movepiece import MOVE_FLAGS

# Import from move_utils instead of moveeffects
from .move_utils import (
    apply_hole_effects,
    apply_bomb_effects,
    apply_trailblaze_effect,
    reconstruct_trailblazer_path,
    extract_enemy_slid_path
)

if TYPE_CHECKING:
    from .gamestate import GameState

# ------------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------------
def validate_legal_moves(game_state: 'GameState', moves: List[Move], color: Color) -> List[Move]:
    """Paranoid validation of legal moves - remove any from empty squares."""
    valid_moves = []

    for move in moves:
        piece = game_state.cache.piece_cache.get(move.from_coord)
        if piece is None:
            continue  # Skip instead of raise to avoid crash
        # Additional validation
        if piece.color != color:
            continue
        valid_moves.append(move)

    return valid_moves

# ------------------------------------------------------------------
# MOVE GENERATION WITH ADVANCED CACHING
# ------------------------------------------------------------------
def legal_moves(game_state: 'GameState') -> List[Move]:
    """Fixed legal move generation with paranoid validation."""
    with track_operation_time(game_state._metrics, 'total_legal_moves_time'):
        game_state._metrics.legal_moves_calls += 1

        current_key = game_state.zkey
        if (game_state._legal_moves_cache is not None and
            game_state._legal_moves_cache_key == current_key):
            # CRITICAL: Validate cached moves before returning
            return validate_legal_moves(game_state, game_state._legal_moves_cache, game_state.color)

        moves = generate_legal_moves(game_state)

        # CRITICAL: Validate moves immediately after generation
        moves = validate_legal_moves(game_state, moves, game_state.color)

        # Cache result
        game_state._legal_moves_cache = moves
        game_state._legal_moves_cache_key = current_key

        return moves

def pseudo_legal_moves(game_state: 'GameState') -> List[Move]:
    """Fast pseudo-legal move generation."""
    return generate_pseudo_legal_moves(game_state.board, game_state.color, game_state.cache)




# ------------------------------------------------------------------
# OPTIMIZED MOVE MAKING WITH INCREMENTAL UPDATES
# ------------------------------------------------------------------
def make_move(game_state: 'GameState', mv: Move) -> 'GameState':
    """Fixed move making with proper cache invalidation and trailblaze support."""
    if game_state.cache.piece_cache.get(mv.from_coord) is None:
        raise ValueError(f"make_move: no piece at {mv.from_coord}")

    with track_operation_time(game_state._metrics, 'total_make_move_time'):
        game_state._metrics.make_move_calls += 1

        # Validate move before any changes
        moving_piece = game_state.cache.piece_cache.get(mv.from_coord)
        if moving_piece is None:
            raise ValueError(f"Cannot move from empty square: {mv.from_coord}")
        if moving_piece.color != game_state.color:
            raise ValueError(f"Cannot move opponent's piece: {mv.from_coord}")

        # Clone board
        new_board = Board(game_state.board.tensor().clone())

        # Pre-compute undo info
        captured_piece = game_state.cache.piece_cache.get(mv.to_coord)
        undo_info = _compute_undo_info(game_state, mv, moving_piece, captured_piece)

        # Apply move to board
        if not new_board.apply_move(mv):
            raise ValueError(f"Board refused move: {mv}")

        # Clear current state's caches
        game_state._clear_caches()

        # Initialize side-effect collections
        removed_pieces: List[Tuple[Tuple[int, int, int], Piece]] = []
        moved_pieces: List[Tuple[Tuple[int, int, int], Tuple[int, int, int], Piece]] = []
        is_self_detonate = False

        # Apply special effects
        apply_hole_effects(new_board, game_state.cache, game_state.color, moved_pieces)
        is_self_detonate = apply_bomb_effects(new_board, game_state.cache, mv, moving_piece,
                                              captured_piece, removed_pieces, is_self_detonate)
        apply_trailblaze_effect(new_board, game_state.cache, mv, game_state.color, removed_pieces)

        # Update halfmove clock
        is_pawn = moving_piece.ptype == PieceType.PAWN
        is_capture = captured_piece is not None
        new_clock = 0 if (is_pawn or is_capture) else game_state.halfmove_clock + 1

        # Create enriched move
        enriched_move = _create_enriched_move(
            game_state, mv, removed_pieces, moved_pieces, is_self_detonate, undo_info, captured_piece
        )

        # Create NEW cache for new state
        new_cache = get_cache_manager(new_board, game_state.color.opposite())

        # Handle Trailblazer path recording in the NEW cache
        if moving_piece.ptype == PieceType.TRAILBLAZER:
            path = reconstruct_trailblazer_path(mv.from_coord, mv.to_coord)
            trail_cache = new_cache.effects["trailblaze"]
            trail_cache.record_trail(mv.from_coord, path)

        # Compute new state
        new_color = game_state.color.opposite()
        new_key = compute_zobrist(new_board, new_color)

        # Import here to avoid circular dependency
        from .gamestate import GameState

        new_state = GameState(
            board=new_board,
            color=new_color,
            cache=new_cache,
            history=game_state.history + (enriched_move,),
            halfmove_clock=new_clock,
            game_mode=game_state.game_mode,
            turn_number=game_state.turn_number + 1,
        )
        object.__setattr__(new_state, '_zkey', new_key)

        return new_state

def _compute_undo_info(game_state: 'GameState', mv: Move, moving_piece: Piece,
                       captured_piece: Optional[Piece]) -> Dict[str, Any]:
    """Pre-compute information needed for efficient undo."""
    return {
        'original_board_tensor': game_state.board.tensor().clone(),
        'moving_piece': moving_piece,
        'captured_piece': captured_piece,
        'original_halfmove_clock': game_state.halfmove_clock,
        'original_zkey': game_state._zkey,
        'original_turn_number': game_state.turn_number,
    }

def _create_enriched_move(
    game_state: 'GameState',
    mv: Move,
    removed_pieces: List,
    moved_pieces: List,
    is_self_detonate: bool,
    undo_info: Dict[str, Any],
    captured_piece: Optional['Piece'] = None  # New param
) -> EnrichedMove:
    """Create enriched move with all side effects and undo information."""
    # Use converter for core Move (handles flags, ints)
    is_capture = mv.is_capture or (captured_piece is not None)
    core_move = convert_legacy_move_args(
        from_coord=mv.from_coord,
        to_coord=mv.to_coord,
        is_capture=is_capture,
        captured_piece=captured_piece,  # Piece -> converter handles .ptype.value
        is_promotion=mv.is_promotion,
        promotion_type=None,  # Fetch if needed: game_state.cache... but assume 0 for now
        is_en_passant=False,  # Set based on flags if needed
        is_castle=False,
        # ... other bools from mv.flags if needed
    )
    return EnrichedMove(
        core_move=core_move,
        removed_pieces=removed_pieces,
        moved_pieces=moved_pieces,
        is_self_detonate=is_self_detonate,
        undo_info=undo_info,
    )

# ------------------------------------------------------------------
# UNDO MOVE IMPLEMENTATION
# ------------------------------------------------------------------
def undo_move(game_state: 'GameState') -> 'GameState':
    """Undo the last move and return previous state."""
    if not game_state.history:
        raise ValueError("Cannot undo - no move history")

    with track_operation_time(game_state._metrics, 'total_undo_move_time'):
        game_state._metrics.undo_move_calls += 1

        # Try fast undo if available
        if game_state._undo_info is not None:
            return _fast_undo(game_state)
        else:
            return _full_undo(game_state)

def _fast_undo(game_state: 'GameState') -> 'GameState':
    """Fast undo using cached undo information."""
    if game_state._undo_info is None:
        raise ValueError("No undo info available for fast undo")

    undo_info = game_state._undo_info

    # Restore board state
    new_board = Board(undo_info['original_board_tensor'].clone())

    # Create cache for restored position
    prev_color = game_state.color.opposite()
    new_cache = get_cache_manager(new_board, prev_color)

    # Import here to avoid circular dependency
    from .gamestate import GameState

    # Create previous state
    prev_state = GameState(
        board=new_board,
        color=prev_color,
        cache=new_cache,
        history=game_state.history[:-1],
        halfmove_clock=undo_info['original_halfmove_clock'],
        game_mode=game_state.game_mode,
        turn_number=undo_info['original_turn_number'],
    )
    object.__setattr__(prev_state, '_zkey', undo_info['original_zkey'])

    return prev_state

def _full_undo(game_state: 'GameState') -> 'GameState':
    """Full undo by replaying moves from initial state."""
    # This is expensive but works when undo_info is not available
    # You would need to store initial state or replay from scratch
    raise NotImplementedError("Full undo requires storing initial state")

@dataclass
class EnrichedMove:
    """Move with side effects and undo information."""
    core_move: Move
    removed_pieces: List[Tuple[Tuple[int, int, int], Piece]]
    moved_pieces: List[Tuple[Tuple[int, int, int], Tuple[int, int, int], Piece]]
    is_self_detonate: bool
    undo_info: Dict[str, Any]
