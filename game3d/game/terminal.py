# terminal.py - FIXED with proper termination conditions
from __future__ import annotations

from typing import Optional, List
from collections import defaultdict

from game3d.board.board import Board
from game3d.common.enums import Color, PieceType, Result
from game3d.movement.movepiece import Move

# Import check functions
from game3d.attacks.check import king_in_check, get_check_summary

# ------------------------------------------------------------------
# GAME STATUS WITH ADVANCED CACHING
# ------------------------------------------------------------------
def is_check(game_state) -> bool:
    """Cached check detection using check.py logic."""
    current_key = game_state.zkey

    if (game_state._is_check_cache is not None and
        game_state._is_check_cache_key == current_key):
        return game_state._is_check_cache

    # Use the check.py implementation
    result = king_in_check(
        game_state.board,
        game_state.color,  # Current player (about to move)
        game_state.color,  # King color to check
        game_state.cache_manager
    )

    # Cache result
    game_state._is_check_cache = result
    game_state._is_check_cache_key = current_key

    return result

def is_stalemate(game_state) -> bool:
    """More robust stalemate detection for complex 3D chess."""
    # Must not be in check
    if is_check(game_state):
        return False

    # Get legal moves
    legal_moves = game_state.legal_moves()

    # If there are any legal moves, it's not stalemate
    if legal_moves:
        return False

    # Additional checks for complex game states
    cache_manager = game_state.cache_manager

    # Check if any non-frozen pieces exist that could theoretically move
    active_pieces_exist = False
    for coord, piece in cache_manager.occupancy.iter_color(game_state.color):
        if not cache_manager.is_frozen(coord, piece.color):
            active_pieces_exist = True
            break

    # If no active pieces exist and no legal moves, it's stalemate
    return not active_pieces_exist

def is_insufficient_material(game_state) -> bool:
    """Cached insufficient material detection."""
    current_key = game_state.zkey

    if (game_state._insufficient_material_cache is not None and
        game_state._insufficient_material_cache_key == current_key):
        return game_state._insufficient_material_cache

    result = _check_insufficient_material(game_state)

    # Cache result
    game_state._insufficient_material_cache = result
    game_state._insufficient_material_cache_key = current_key

    return result

def _check_insufficient_material(game_state) -> bool:
    """More conservative insufficient material check for 462-piece game."""
    piece_counts = {
        Color.WHITE: {ptype: 0 for ptype in PieceType},
        Color.BLACK: {ptype: 0 for ptype in PieceType}
    }

    # Count all pieces
    total_pieces = 0
    for _, piece in game_state.board.list_occupied():
        piece_counts[piece.color][piece.ptype] += 1
        total_pieces += 1

    # With 462 starting pieces, insufficient material is extremely unlikely
    # Only consider it when very few pieces remain
    if total_pieces > 10:  # Much higher threshold
        return False

    # 1. King vs King
    if (piece_counts[Color.WHITE][PieceType.KING] == 1 and
        piece_counts[Color.BLACK][PieceType.KING] == 1 and
        all(piece_counts[Color.WHITE][pt] == 0 for pt in PieceType if pt != PieceType.KING) and
        all(piece_counts[Color.BLACK][pt] == 0 for pt in PieceType if pt != PieceType.KING)):
        return True

    # 2. King + Bishop vs King
    for color in [Color.WHITE, Color.BLACK]:
        opponent = color.opposite()
        if (piece_counts[color][PieceType.KING] == 1 and
            piece_counts[color][PieceType.BISHOP] == 1 and
            all(piece_counts[color][pt] == 0 for pt in PieceType if pt not in [PieceType.KING, PieceType.BISHOP]) and
            piece_counts[opponent][PieceType.KING] == 1 and
            all(piece_counts[opponent][pt] == 0 for pt in PieceType if pt != PieceType.KING)):
            return True

    # 3. King + Knight vs King
    for color in [Color.WHITE, Color.BLACK]:
        opponent = color.opposite()
        if (piece_counts[color][PieceType.KING] == 1 and
            piece_counts[color][PieceType.KNIGHT] == 1 and
            all(piece_counts[color][pt] == 0 for pt in PieceType if pt not in [PieceType.KING, PieceType.KNIGHT]) and
            piece_counts[opponent][PieceType.KING] == 1 and
            all(piece_counts[opponent][pt] == 0 for pt in PieceType if pt != PieceType.KING)):
            return True

    # 4. King vs King + Bishop
    for color in [Color.WHITE, Color.BLACK]:
        opponent = color.opposite()
        if (piece_counts[color][PieceType.KING] == 1 and
            all(piece_counts[color][pt] == 0 for pt in PieceType if pt != PieceType.KING) and
            piece_counts[opponent][PieceType.KING] == 1 and
            piece_counts[opponent][PieceType.BISHOP] == 1 and
            all(piece_counts[opponent][pt] == 0 for pt in PieceType if pt not in [PieceType.KING, PieceType.BISHOP])):
            return True

    # 5. King vs King + Knight
    for color in [Color.WHITE, Color.BLACK]:
        opponent = color.opposite()
        if (piece_counts[color][PieceType.KING] == 1 and
            all(piece_counts[color][pt] == 0 for pt in PieceType if pt != PieceType.KING) and
            piece_counts[opponent][PieceType.KING] == 1 and
            piece_counts[opponent][PieceType.KNIGHT] == 1 and
            all(piece_counts[opponent][pt] == 0 for pt in PieceType if pt not in [PieceType.KING, PieceType.KNIGHT])):
            return True

    return False

def is_fifty_move_draw(game_state) -> bool:
    """
    Fifty-move rule: Game is drawn if 50 consecutive moves by EACH player
    (100 half-moves total) with no pawn move or capture.

    For a 462-piece game, we use the standard 50-move rule.
    halfmove_clock >= 100 means 50 full moves without progress.
    """
    return game_state.halfmove_clock >= 100

def is_threefold_repetition(game_state) -> bool:
    """
    Check for threefold repetition using Zobrist hashes.
    The same position must occur 3 times (not necessarily consecutively).
    """
    # Initialize position tracking if not exists
    if not hasattr(game_state, '_position_counts'):
        game_state._position_counts = defaultdict(int)
        # Count current position
        game_state._position_counts[game_state.zkey] = 1
        return False

    current_zkey = game_state.zkey
    count = game_state._position_counts.get(current_zkey, 0)

    # Position must occur at least 3 times for threefold repetition
    return count >= 3

def is_fivefold_repetition(game_state) -> bool:
    """
    Fivefold repetition is an automatic draw (no claim needed).
    This is stronger than threefold repetition.
    """
    if not hasattr(game_state, '_position_counts'):
        return False

    current_zkey = game_state.zkey
    count = game_state._position_counts.get(current_zkey, 0)
    return count >= 5

def is_seventy_five_move_draw(game_state) -> bool:
    """
    75-move rule: Automatic draw after 75 moves by each player (150 half-moves)
    with no pawn move or capture. This is stronger than the fifty-move rule.
    """
    return game_state.halfmove_clock >= 150

def is_game_over(game_state) -> bool:
    """
    More conservative game over detection.
    Game ends when:
    1. No legal moves (checkmate or stalemate)
    2. Fivefold repetition (automatic draw)
    3. 75-move rule (automatic draw)
    4. Insufficient material
    """
    # Check automatic draw conditions first (no claim needed)
    if is_fivefold_repetition(game_state):
        return True

    if is_seventy_five_move_draw(game_state):
        return True

    if is_insufficient_material(game_state):
        return True

    # Check for no legal moves
    legal_moves = game_state.legal_moves()

    # If there are legal moves, game continues
    if legal_moves:
        return False

    # No legal moves - check why
    if is_check(game_state):
        return True  # Checkmate

    # For stalemate, be more conservative in complex games
    # Use occupancy cache to count pieces efficiently
    total_pieces = game_state.cache_manager.occupancy.count

    if total_pieces > 20:  # With many pieces, stalemate is unlikely
        # Double-check by generating moves from scratch
        from game3d.movement.generator import generate_legal_moves
        fresh_moves = generate_legal_moves(game_state)
        if fresh_moves:
            return False

    return True  # True stalemate

def result(game_state) -> Optional[Result]:
    """
    Fast game result determination.
    Returns None if game is not over.
    """
    if not is_game_over(game_state):
        return None

    # Check automatic draw conditions first
    if is_fivefold_repetition(game_state):
        return Result.DRAW

    if is_seventy_five_move_draw(game_state):
        return Result.DRAW

    # Check claimable draw conditions
    if is_fifty_move_draw(game_state):
        return Result.DRAW

    if is_threefold_repetition(game_state):
        return Result.DRAW

    if is_insufficient_material(game_state):
        return Result.DRAW

    if is_stalemate(game_state):
        return Result.DRAW

    # If we're here and game is over, it must be checkmate
    # The player who cannot move has lost
    return Result.BLACK_WON if game_state.color == Color.WHITE else Result.WHITE_WON

def is_terminal(game_state) -> bool:
    """Fast terminal check - alias for is_game_over."""
    return is_game_over(game_state)

def outcome(game_state) -> int:
    """
    Fast outcome evaluation.
    Returns: 1 for white win, -1 for black win, 0 for draw.
    """
    res = result(game_state)
    if res == Result.WHITE_WON:
        return 1
    elif res == Result.BLACK_WON:
        return -1
    elif res == Result.DRAW:
        return 0
    raise ValueError("outcome() called on non-terminal state")

def insufficient_material(board: Board) -> bool:
    """Fast insufficient material check (backward compatibility)."""
    piece_types = set()
    for _, piece in board.list_occupied():
        if piece.ptype != PieceType.KING:
            piece_types.add(piece.ptype)

    # If only kings or kings + minor pieces that cannot checkmate
    if len(piece_types) == 0:
        return True  # Only kings

    if piece_types.issubset({PieceType.BISHOP, PieceType.KNIGHT}):
        return True  # Only kings and bishops/knights

    return False

def get_draw_reason(game_state) -> Optional[str]:
    """
    Get the reason for a draw, if applicable.
    Returns None if not a draw.
    """
    if not is_game_over(game_state):
        return None

    if result(game_state) != Result.DRAW:
        return None

    # Check all draw conditions and return the first that matches
    if is_fivefold_repetition(game_state):
        return "Fivefold repetition (automatic draw)"

    if is_seventy_five_move_draw(game_state):
        return "75-move rule (automatic draw)"

    if is_fifty_move_draw(game_state):
        return "50-move rule"

    if is_threefold_repetition(game_state):
        return "Threefold repetition"

    if is_insufficient_material(game_state):
        return "Insufficient material"

    if is_stalemate(game_state):
        return "Stalemate"

    return "Unknown draw reason"
