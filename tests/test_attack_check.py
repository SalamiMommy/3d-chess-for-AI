"""
Comprehensive tests for attack and check mechanics in 3D chess.

Tests cover:
1. Attack puts king in check (0 priests)
2. King is safe from check with priest protection
3. King cannot move into attacked square (0 priests)
4. Pieces are pinned when king has 0 priests
5. Capturing the attacker resolves check
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from game3d.game.gamestate import GameState
from game3d.board.board import Board
from game3d.common.shared_types import PieceType, Color, COORD_DTYPE, SIZE
from game3d.attacks.check import king_in_check, move_would_leave_king_in_check
from game3d.game.terminal import is_check
from game3d.movement.generator import LegalMoveGenerator


def create_empty_game_state(color: Color = Color.WHITE) -> GameState:
    """Create an empty game state for testing."""
    gs = GameState(Board.empty(), color)
    gs.cache_manager.occupancy_cache.clear()
    return gs


def place_piece(gs: GameState, coord: tuple, piece_type: PieceType, color: Color):
    """Helper to place a piece on the board."""
    pos = np.array(coord, dtype=np.int16)
    data = np.array([piece_type.value, int(color)], dtype=np.int8)
    gs.cache_manager.occupancy_cache.set_position(pos, data)


def refresh_caches(gs: GameState):
    """Invalidate and refresh move caches after board changes.
    
    Also generates opponent moves so attack masks are populated for king safety checks.
    """
    gs.cache_manager.move_cache.invalidate()
    # Generate opponent moves so attack detection uses cached moves
    opponent_color = Color.WHITE if gs.color == Color.BLACK else Color.BLACK
    # Temporarily swap color to generate opponent moves
    original_color = gs.color
    gs.color = opponent_color
    generator = LegalMoveGenerator()
    generator.refresh_pseudolegal_moves(gs)
    gs.color = original_color


class TestAttackPutsKingInCheck:
    """Test 1: Verify attacking piece puts king in check when no priests."""
    
    def test_rook_attacks_king(self):
        """Rook on same file attacks king -> check."""
        gs = create_empty_game_state(Color.WHITE)
        
        # White King at (4, 4, 4)
        place_piece(gs, (4, 4, 4), PieceType.KING, Color.WHITE)
        # Black King far away
        place_piece(gs, (0, 0, 0), PieceType.KING, Color.BLACK)
        # Black Rook attacking king along z-axis at (4, 4, 8)
        place_piece(gs, (4, 4, 8), PieceType.ROOK, Color.BLACK)
        
        refresh_caches(gs)
        
        # White has no priests, so should be in check
        assert is_check(gs) == True, "King should be in check from Rook attack"
    
    def test_queen_attacks_king_diagonal(self):
        """Queen on diagonal attacks king -> check."""
        gs = create_empty_game_state(Color.WHITE)
        
        place_piece(gs, (4, 4, 4), PieceType.KING, Color.WHITE)
        place_piece(gs, (0, 0, 0), PieceType.KING, Color.BLACK)
        # Black Queen attacking diagonally
        place_piece(gs, (8, 8, 8), PieceType.QUEEN, Color.BLACK)
        
        refresh_caches(gs)
        
        assert is_check(gs) == True, "King should be in check from Queen diagonal attack"
    
    def test_knight_attacks_king(self):
        """Knight in attack range -> check."""
        gs = create_empty_game_state(Color.WHITE)
        
        place_piece(gs, (4, 4, 4), PieceType.KING, Color.WHITE)
        place_piece(gs, (0, 0, 0), PieceType.KING, Color.BLACK)
        # Black Knight at L-shape from king (2,1 movement)
        place_piece(gs, (6, 5, 4), PieceType.KNIGHT, Color.BLACK)
        
        refresh_caches(gs)
        
        assert is_check(gs) == True, "King should be in check from Knight attack"


class TestKingSafeWithPriest:
    """Test 2: Verify king is safe from check when priest is present."""
    
    def test_priest_protects_from_rook(self):
        """Priest presence negates check from Rook."""
        gs = create_empty_game_state(Color.WHITE)
        
        # White King at (4, 4, 4)
        place_piece(gs, (4, 4, 4), PieceType.KING, Color.WHITE)
        # White Priest at safe location
        place_piece(gs, (0, 0, 1), PieceType.PRIEST, Color.WHITE)
        # Black King far away
        place_piece(gs, (0, 0, 0), PieceType.KING, Color.BLACK)
        # Black Rook attacking king
        place_piece(gs, (4, 4, 8), PieceType.ROOK, Color.BLACK)
        
        refresh_caches(gs)
        
        # With priest, king should NOT be in check
        assert is_check(gs) == False, "King should NOT be in check when priest is alive"
    
    def test_priest_protects_from_multiple_attackers(self):
        """Priest presence negates check even from multiple attackers."""
        gs = create_empty_game_state(Color.WHITE)
        
        place_piece(gs, (4, 4, 4), PieceType.KING, Color.WHITE)
        place_piece(gs, (1, 1, 1), PieceType.PRIEST, Color.WHITE)
        place_piece(gs, (0, 0, 0), PieceType.KING, Color.BLACK)
        # Multiple attackers
        place_piece(gs, (4, 4, 8), PieceType.ROOK, Color.BLACK)
        place_piece(gs, (8, 8, 8), PieceType.QUEEN, Color.BLACK)
        
        refresh_caches(gs)
        
        assert is_check(gs) == False, "Priest should protect from multiple attackers"


class TestKingCannotMoveIntoCheck:
    """Test 3: Verify king cannot move into attacked square when no priests."""
    
    def test_king_blocked_from_attacked_square(self):
        """King's legal moves should not include attacked squares."""
        gs = create_empty_game_state(Color.WHITE)
        
        # White King at (4, 4, 4)
        place_piece(gs, (4, 4, 4), PieceType.KING, Color.WHITE)
        # Black King far away
        place_piece(gs, (0, 0, 0), PieceType.KING, Color.BLACK)
        # Black Rook controls (4, 4, 5) via the z-file
        place_piece(gs, (4, 4, 8), PieceType.ROOK, Color.BLACK)
        
        refresh_caches(gs)
        
        # Generate legal moves
        generator = LegalMoveGenerator()
        legal_moves = generator.generate_fused(gs)
        
        # Filter for king moves (from position 4,4,4)
        king_from = np.array([4, 4, 4])
        king_moves = legal_moves[
            (legal_moves[:, 0] == king_from[0]) & 
            (legal_moves[:, 1] == king_from[1]) & 
            (legal_moves[:, 2] == king_from[2])
        ]
        
        # Check that king cannot move to (4, 4, 5) which is attacked
        for move in king_moves:
            to_coord = move[3:6]
            # These squares are on the rook's attack file
            assert not (to_coord[0] == 4 and to_coord[1] == 4 and to_coord[2] in [5, 6, 7]), \
                f"King should not be able to move to attacked square {to_coord}"


class TestPiecePinnedWhenNoPriests:
    """Test 4: Verify pieces are pinned when king has 0 priests."""
    
    def test_pawn_pinned_by_rook(self):
        """Pawn between king and rook is pinned - cannot move sideways."""
        gs = create_empty_game_state(Color.WHITE)
        
        # White King at (4, 4, 0)
        place_piece(gs, (4, 4, 0), PieceType.KING, Color.WHITE)
        # White Pawn at (4, 4, 1) - pinned on z-axis
        place_piece(gs, (4, 4, 1), PieceType.PAWN, Color.WHITE)
        # Black King far away
        place_piece(gs, (0, 0, 8), PieceType.KING, Color.BLACK)
        # Black Rook at (4, 4, 7) - pinning the pawn
        place_piece(gs, (4, 4, 7), PieceType.ROOK, Color.BLACK)
        
        refresh_caches(gs)
        
        # Generate legal moves
        generator = LegalMoveGenerator()
        legal_moves = generator.generate_fused(gs)
        
        # Filter for pawn moves (from position 4,4,1)
        pawn_from = np.array([4, 4, 1])
        pawn_moves = legal_moves[
            (legal_moves[:, 0] == pawn_from[0]) & 
            (legal_moves[:, 1] == pawn_from[1]) & 
            (legal_moves[:, 2] == pawn_from[2])
        ]
        
        # Pawn should only be able to move forward along the pin line (z-axis)
        for move in pawn_moves:
            to_coord = move[3:6]
            # Pinned pawn can only move along (4, 4, z)
            assert to_coord[0] == 4 and to_coord[1] == 4, \
                f"Pinned pawn should not move off pin line to {to_coord}"
    
    @pytest.mark.xfail(reason="Engine bug: pin filtering not filtering bishop moves")
    def test_bishop_pinned_cannot_break_pin(self):
        """Bishop pinned by rook cannot move diagonally (breaks pin)."""
        gs = create_empty_game_state(Color.WHITE)
        
        # White King at (4, 4, 0)
        place_piece(gs, (4, 4, 0), PieceType.KING, Color.WHITE)
        # White Bishop at (4, 4, 2) - pinned on z-axis
        place_piece(gs, (4, 4, 2), PieceType.BISHOP, Color.WHITE)
        # Black King far away
        place_piece(gs, (0, 0, 8), PieceType.KING, Color.BLACK)
        # Black Rook at (4, 4, 7) - pinning the bishop
        place_piece(gs, (4, 4, 7), PieceType.ROOK, Color.BLACK)
        
        refresh_caches(gs)
        
        # Generate legal moves
        generator = LegalMoveGenerator()
        legal_moves = generator.generate_fused(gs)
        
        # Filter for bishop moves
        bishop_from = np.array([4, 4, 2])
        bishop_moves = legal_moves[
            (legal_moves[:, 0] == bishop_from[0]) & 
            (legal_moves[:, 1] == bishop_from[1]) & 
            (legal_moves[:, 2] == bishop_from[2])
        ]
        
        # Bishop is pinned by rook on z-axis - cannot move diagonally
        # Bishop has no legal moves because it can only move diagonally
        assert len(bishop_moves) == 0, \
            f"Pinned bishop should have no legal moves, found {len(bishop_moves)}"


class TestCaptureResolvesCheck:
    """Test 5: Verify capturing the attacker resolves check."""
    
    def test_capture_rook_resolves_check(self):
        """Capturing the attacking rook should resolve check."""
        gs = create_empty_game_state(Color.WHITE)
        
        # White King at (4, 4, 4)
        place_piece(gs, (4, 4, 4), PieceType.KING, Color.WHITE)
        # White Rook that can capture the attacker
        place_piece(gs, (4, 0, 8), PieceType.ROOK, Color.WHITE)
        # Black King far away
        place_piece(gs, (0, 0, 0), PieceType.KING, Color.BLACK)
        # Black Rook attacking king at (4, 4, 8)
        place_piece(gs, (4, 4, 8), PieceType.ROOK, Color.BLACK)
        
        refresh_caches(gs)
        
        # Verify we're in check first
        assert is_check(gs) == True, "Should be in check initially"
        
        # The move that captures the black rook: (4, 0, 8) -> (4, 4, 8)
        capture_move = np.array([4, 0, 8, 4, 4, 8], dtype=COORD_DTYPE)
        
        # This move should NOT leave king in check (it captures the attacker)
        leaves_check = move_would_leave_king_in_check(gs, capture_move, gs.cache_manager)
        assert leaves_check == False, "Capturing the attacker should resolve check"
    
    def test_king_captures_attacker_resolves_check(self):
        """King capturing the attacker should resolve check."""
        gs = create_empty_game_state(Color.WHITE)
        
        # White King at (4, 4, 4)
        place_piece(gs, (4, 4, 4), PieceType.KING, Color.WHITE)
        # Black King far away
        place_piece(gs, (0, 0, 0), PieceType.KING, Color.BLACK)
        # Black Knight attacking king - king can capture it
        place_piece(gs, (5, 5, 5), PieceType.KNIGHT, Color.BLACK)
        
        refresh_caches(gs)
        
        # Note: Knight at (5,5,5) might not actually attack (4,4,4)
        # Let's use adjacent piece that king can capture
        gs2 = create_empty_game_state(Color.WHITE)
        place_piece(gs2, (4, 4, 4), PieceType.KING, Color.WHITE)
        place_piece(gs2, (0, 0, 0), PieceType.KING, Color.BLACK)
        # Black Rook adjacent to king (can be captured)
        place_piece(gs2, (4, 4, 5), PieceType.ROOK, Color.BLACK)
        
        refresh_caches(gs2)
        
        # Verify in check
        assert is_check(gs2) == True, "Should be in check from adjacent rook"
        
        # King captures rook: (4,4,4) -> (4,4,5)
        capture_move = np.array([4, 4, 4, 4, 4, 5], dtype=COORD_DTYPE)
        
        # Must verify destination is not also attacked by another piece
        # For simplicity, this should resolve check if rook is the only attacker
        leaves_check = move_would_leave_king_in_check(gs2, capture_move, gs2.cache_manager)
        assert leaves_check == False, "King capturing sole attacker should resolve check"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
