"""
Precompute raw move generation tables for all piece types (40) on a 9x9x9 3D chess board,
and serialize them to disk for fast loading.

Usage:
    from game3d.movement.precompute import load_precomputed_moves, get_precomputed_moves
    moves = get_precomputed_moves(piece_type, (x, y, z))
"""

import os
import pickle
from typing import Dict, Tuple, List, Optional
import numpy as np

from game3d.pieces.enums import PieceType, Color
from game3d.common.common import in_bounds, SIZE_X, SIZE_Y, SIZE_Z

# Import all individual piece move generators
from game3d.movement.movetypes.bishopmovement import generate_bishop_moves
from game3d.movement.movetypes.rookmovement import generate_rook_moves
from game3d.movement.movetypes.trigonalbishopmovement import generate_trigonal_bishop_moves
from game3d.movement.movetypes.knightmovement import generate_knight_moves
from game3d.movement.movetypes.knight31movement import generate_knight31_moves
from game3d.movement.movetypes.knight32movement import generate_knight32_moves
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movetypes.xyqueenmovement import generate_xy_queen_moves
from game3d.movement.movetypes.xzqueenmovement import generate_xz_queen_moves
from game3d.movement.movetypes.yzqueenmovement import generate_yz_queen_moves
from game3d.movement.movetypes.orbitalmovement import generate_orbital_moves
from game3d.movement.movetypes.panelmovement import generate_panel_moves
from game3d.movement.movetypes.nebulamovement import generate_nebula_moves
from game3d.movement.movetypes.faceconemovement import generate_face_cone_slider_moves
from game3d.movement.movetypes.vectorslidermovement import generate_vector_slider_moves
from game3d.movement.movetypes.spiralmovement import generate_spiral_moves
from game3d.movement.movetypes.reflectingbishopmovement import generate_reflecting_bishop_moves
from game3d.movement.movetypes.edgerookmovement import generate_edgerook_moves
from game3d.movement.movetypes.echomovement import generate_echo_moves
from game3d.movement.movetypes.xzzigzagmovement import generate_xz_zigzag_moves
from game3d.movement.movetypes.yzzigzagmovement import generate_yz_zigzag_moves
from game3d.movement.movetypes.swapmovement import generate_swapper_moves
from game3d.movement.movetypes.networkteleportmovement import generate_network_teleport_moves
from game3d.movement.movetypes.mirrorteleportmovement import generate_mirror_teleport_move
from game3d.movement.movetypes.pawnfrontteleportmovement import generate_pawn_front_teleport_moves
from game3d.movement.movetypes.pawnmovement import generate_pawn_moves

# Add other PieceTypes as needed.

from game3d.cache.manager import get_cache_manager
from game3d.board.board import Board

_PRECOMPUTED_MOVES_PATH = os.path.expanduser("/home/salamimommy/Documents/code/3d-chess-for-AI/game3d/movement/movetypes/precomputed_cache/moves.pkl")

def get_dummy_cache():
    board = Board.empty()
    cache = get_cache_manager(board, Color.WHITE)
    return cache

PIECE_GENERATORS = {
    PieceType.BISHOP: generate_bishop_moves,
    PieceType.ROOK: generate_rook_moves,
    PieceType.TRIGONALBISHOP: generate_trigonal_bishop_moves,
    PieceType.KNIGHT: lambda cache, color, x, y, z: generate_knight_moves(cache, x, y, z),
    PieceType.KNIGHT31: generate_knight31_moves,
    PieceType.KNIGHT32: generate_knight32_moves,
    PieceType.KING: generate_king_moves,
    PieceType.QUEEN: lambda cache, color, x, y, z: (
        generate_rook_moves(cache, color, x, y, z) +
        generate_bishop_moves(cache, color, x, y, z)
    ),
    PieceType.XYQUEEN: generate_xy_queen_moves,
    PieceType.XZQUEEN: generate_xz_queen_moves,
    PieceType.YZQUEEN: generate_yz_queen_moves,
    PieceType.ORBITER: generate_orbital_moves,
    PieceType.PANEL: generate_panel_moves,
    PieceType.NEBULA: generate_nebula_moves,
    PieceType.FACECONESLIDER: generate_face_cone_slider_moves,
    PieceType.VECTORSLIDER: generate_vector_slider_moves,
    PieceType.SPIRAL: generate_spiral_moves,
    PieceType.REFLECTORBISHOP: generate_reflecting_bishop_moves,
    PieceType.EDGEROOK: generate_edgerook_moves,
    PieceType.ECHO: generate_echo_moves,
    PieceType.XZZIGZAG: generate_xz_zigzag_moves,
    PieceType.YZZIGZAG: generate_yz_zigzag_moves,
    PieceType.SWAPPER: generate_swapper_moves,
    PieceType.NETWORKTELEPORTER: generate_network_teleport_moves,
    PieceType.MIRROR: generate_mirror_teleport_move,
    PieceType.PAWNFRONTTELEPORTER: generate_pawn_front_teleport_moves,
    PieceType.PAWN: generate_pawn_moves,
    # Add custom pieces here (bomb, wall, geomancer, etc.)
}

def precompute_moves_for_piece(piece_type: PieceType, color: Color) -> Dict[Tuple[int, int, int], List]:
    """Precompute raw moves for given piece type and color on a 9x9x9 board."""
    cache = get_dummy_cache()
    moves_table = {}
    gen_fn = PIECE_GENERATORS.get(piece_type)
    if gen_fn is None:
        return {}
    for x in range(SIZE_X):
        for y in range(SIZE_Y):
            for z in range(SIZE_Z):
                if not in_bounds((x, y, z)):
                    continue
                try:
                    moves = gen_fn(cache, color, x, y, z)
                    # Store only the raw destination coordinates for compactness
                    moves_table[(x, y, z)] = [m.to_coord for m in moves]
                except Exception:
                    moves_table[(x, y, z)] = []
    return moves_table

def build_precomputed_moves() -> Dict[str, Dict[Tuple[int, int, int], List]]:
    """Precompute for all 40 pieces. Keyed by piece name (str)."""
    all_moves: Dict[str, Dict[Tuple[int, int, int], List]] = {}
    for pt in PieceType:
        # For color-sensitive pieces, choose default (WHITE) or precompute both if needed
        all_moves[pt.name] = precompute_moves_for_piece(pt, Color.WHITE)
    return all_moves

def save_precomputed_moves(path: Optional[str] = None) -> None:
    """Serialize precomputed moves to disk."""
    moves = build_precomputed_moves()
    if path is None:
        path = _PRECOMPUTED_MOVES_PATH
    with open(path, "wb") as f:
        pickle.dump(moves, f)
    print(f"Precomputed moves saved to {path}")

def load_precomputed_moves(path: Optional[str] = None) -> Dict[str, Dict[Tuple[int, int, int], List]]:
    """Load precomputed move table from disk."""
    if path is None:
        path = _PRECOMPUTED_MOVES_PATH
    if not os.path.exists(path):
        print("No precomputed moves found, generating now...")
        save_precomputed_moves(path)
    with open(path, "rb") as f:
        moves = pickle.load(f)
    return moves

# On import, try to load from disk:
try:
    _PRECOMPUTED_MOVES: Dict[str, Dict[Tuple[int, int, int], List]] = load_precomputed_moves()
except Exception:
    _PRECOMPUTED_MOVES = build_precomputed_moves()

def get_precomputed_moves(piece_type: PieceType, coord: Tuple[int, int, int]) -> List:
    """Get precomputed destinations for (piece_type, coord)."""
    piece_table = _PRECOMPUTED_MOVES.get(piece_type.name, {})
    return piece_table.get(coord, [])

# Utility for manual regeneration
def regenerate_precomputed_moves(path: Optional[str] = None):
    """Force regeneration and save to disk."""
    save_precomputed_moves(path)
    print("Regenerated and saved precomputed moves.")
