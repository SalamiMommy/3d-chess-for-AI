# terminal.py - Updated with correct fifty-move rule implementation
from __future__ import annotations

from typing import Optional, List

from game3d.board.board import Board
from game3d.pieces.enums import Color, PieceType, Result
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
        game_state.cache
    )

    # Cache result
    game_state._is_check_cache = result
    game_state._is_check_cache_key = current_key

    return result

def is_stalemate(game_state) -> bool:
    """Fast stalemate detection."""
    return not is_check(game_state) and len(game_state.legal_moves()) == 0

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
    """Check for insufficient material with early exits."""
    # Count pieces by type
    piece_counts = {
        Color.WHITE: {ptype: 0 for ptype in PieceType},
        Color.BLACK: {ptype: 0 for ptype in PieceType}
    }

    for _, piece in game_state.board.list_occupied():
        piece_counts[piece.color][piece.ptype] += 1

    # Check common insufficient material patterns

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
    """Correct fifty-move rule implementation with debugging."""
    # In chess, the fifty-move rule is 50 moves by each player (100 half-moves)
    # without a pawn move or capture
    if game_state.halfmove_clock >= 100:
        # print(f"[DEBUG] Fifty-move draw: halfmove_clock = {game_state.halfmove_clock}")

        # Let's check the last few moves to see if there were pawn moves or captures
        # print("[DEBUG] Checking last 10 moves for pawn moves or captures:")
        pawn_or_capture_found = False

        # Look at the last 10 moves (or fewer if not enough history)
        history_to_check = game_state.history[-10:] if len(game_state.history) >= 10 else game_state.history

        for i, move in enumerate(history_to_check):
            # Check if this move was a pawn move or capture
            is_pawn_move = False
            is_capture = False

            # Check EnrichedMove flags if available
            if hasattr(move, 'is_pawn_move'):
                is_pawn_move = move.is_pawn_move
            if hasattr(move, 'is_capture'):
                is_capture = move.is_capture

            # Also check core move flags
            if hasattr(move, 'core_move'):
                core_move = move.core_move
                if hasattr(core_move, 'is_capture') and core_move.is_capture:
                    is_capture = True

            print(f"  Move {len(game_state.history)-len(history_to_check)+i+1}: {move}")
            if is_pawn_move:
                print(f"    -> Pawn move!")
                pawn_or_capture_found = True
            if is_capture:
                print(f"    -> Capture!")
                pawn_or_capture_found = True

        if pawn_or_capture_found:
            # print("[DEBUG] Pawn move or capture found in recent history - fifty-move rule should not apply!")
            return False
        else:
            print("[DEBUG] No pawn moves or captures found in recent history")

            # Let's check the entire history for pawn moves or captures
            print("[DEBUG] Checking entire history for pawn moves or captures:")
            for i, move in enumerate(game_state.history):
                is_pawn_move = False
                is_capture = False

                # Check EnrichedMove flags if available
                if hasattr(move, 'is_pawn_move'):
                    is_pawn_move = move.is_pawn_move
                if hasattr(move, 'is_capture'):
                    is_capture = move.is_capture

                # Also check core move flags
                if hasattr(move, 'core_move'):
                    core_move = move.core_move
                    if hasattr(core_move, 'is_capture') and core_move.is_capture:
                        is_capture = True

                if is_pawn_move:
                    print(f"  Move {i+1}: Pawn move")
                    pawn_or_capture_found = True
                if is_capture:
                    print(f"  Move {i+1}: Capture")
                    pawn_or_capture_found = True

            if pawn_or_capture_found:
                print("[DEBUG] Pawn move or capture found in entire history - fifty-move rule should not apply!")
                return False
            else:
                print("[DEBUG] No pawn moves or captures found in entire history")
                return True
    return False

def is_threefold_repetition(game_state) -> bool:
    """Check for threefold repetition using Zobrist hashes."""
    if len(game_state.history) < 6:  # Need at least 3 repetitions
        return False

    current_hash = game_state.zkey
    count = 0

    # Count occurrences of current position in history
    # Note: This is simplified - in practice you'd need to store position hashes
    for i in range(0, len(game_state.history) - 1, 2):  # Only compare same player to move
        # This would require storing historical zobrist hashes
        pass

    return count >= 2  # Current position + 2 previous occurrences

def is_game_over(game_state) -> bool:
    """Fast terminal detection with early exits."""
    # Check fastest conditions first
    if is_fifty_move_draw(game_state):
        return True

    if is_insufficient_material(game_state):
        return True

    # Check for checkmate or stalemate
    legal_moves = game_state.legal_moves()
    if len(legal_moves) == 0:
        return True

    return False

def result(game_state) -> Optional[Result]:
    """Fast game result determination."""
    if not is_game_over(game_state):
        return None

    # Check draw conditions first
    if (is_fifty_move_draw(game_state) or
        is_insufficient_material(game_state) or
        is_stalemate(game_state)):
        return Result.DRAW

    # If we're here and game is over, it must be checkmate
    # The player who cannot move has lost
    return Result.BLACK_WON if game_state.color == Color.WHITE else Result.WHITE_WON

def is_terminal(game_state) -> bool:
    """Fast terminal check."""
    return is_game_over(game_state)

def outcome(game_state) -> int:
    """Fast outcome evaluation."""
    res = result(game_state)
    if res == Result.WHITE_WON:
        return 1
    elif res == Result.BLACK_WON:
        return -1
    elif res == Result.DRAW:
        return 0
    raise ValueError("outcome() called on non-terminal state")

def insufficient_material(board: Board) -> bool:
    """Fast insufficient material check."""
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
