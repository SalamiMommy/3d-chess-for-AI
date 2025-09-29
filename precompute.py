"""
game3d/movement/precompute.py
Precompute all legal moves for every piece type at every position on 9×9×9 board.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Set
from game3d.pieces.enums import PieceType, Color
from game3d.common.common import VOLUME, SIZE_X, SIZE_Y, SIZE_Z

# Import all movement generators
from game3d.movement.movetypes.bishopmovement import generate_bishop_moves
from game3d.movement.movetypes.rookmovement import generate_rook_moves
from game3d.movement.movetypes.queenmovement import generate_queen_moves
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movetypes.knightmovement import generate_knight_moves
from game3d.movement.movetypes.knight31movement import generate_knight31_moves
from game3d.movement.movetypes.knight32movement import generate_knight32_moves
from game3d.movement.movetypes.pawnmovement import generate_pawn_moves
from game3d.movement.movetypes.trigonalbishopmovement import generate_trigonal_bishop_moves
from game3d.movement.movetypes.xyqueenmovement import generate_xy_queen_moves
from game3d.movement.movetypes.xzqueenmovement import generate_xz_queen_moves
from game3d.movement.movetypes.yzqueenmovement import generate_yz_queen_moves
from game3d.movement.movetypes.vectorslidermovement import generate_vector_slider_moves
from game3d.movement.movetypes.trailblazermovement import generate_trailblazer_moves
from game3d.movement.movetypes.nebulamovement import generate_nebula_moves
from game3d.movement.movetypes.orbitalmovement import generate_orbital_moves
from game3d.movement.movetypes.echomovement import generate_echo_moves
from game3d.movement.movetypes.panelmovement import generate_panel_moves
from game3d.movement.movetypes.mirrorteleportmovement import generate_mirror_teleport_move
from game3d.movement.movetypes.networkteleportmovement import generate_network_teleport_moves
from game3d.movement.movetypes.pawnfrontteleportmovement import generate_pawn_front_teleport_moves
from game3d.movement.movetypes.spiralmovement import generate_spiral_moves
from game3d.movement.movetypes.xzzigzagmovement import generate_xz_zigzag_moves
from game3d.movement.movetypes.yzzigzagmovement import generate_yz_zigzag_moves
from game3d.movement.movetypes.edgerookmovement import generate_edgerook_moves
from game3d.movement.movetypes.faceconemovement import generate_face_cone_slider_moves
from game3d.movement.movetypes.reflectingbishopmovement import generate_reflecting_bishop_moves
from game3d.movement.movetypes.swapmovement import generate_swapper_moves


class MockCache:
    """Minimal mock cache for precomputation - empty board."""
    def __init__(self):
        self.piece_cache = MockPieceCache()
        self.board = MockBoard()

class MockPieceCache:
    """Mock piece cache that returns None (empty squares)."""
    def __init__(self):
        self._cache = {}

    def get(self, pos):
        return None

class MockBoard:
    """Mock board for network teleporter (needs list_occupied)."""
    def list_occupied(self):
        return iter([])


# Map piece types to their move generators
MOVE_GENERATORS = {
    PieceType.PAWN: generate_pawn_moves,
    PieceType.KNIGHT: generate_knight_moves,
    PieceType.BISHOP: generate_bishop_moves,
    PieceType.ROOK: generate_rook_moves,
    PieceType.QUEEN: generate_queen_moves,
    PieceType.KING: generate_king_moves,
    PieceType.KNIGHT31: generate_knight31_moves,
    PieceType.KNIGHT32: generate_knight32_moves,
    PieceType.TRIGONALBISHOP: generate_trigonal_bishop_moves,
    PieceType.XYQUEEN: generate_xy_queen_moves,
    PieceType.XZQUEEN: generate_xz_queen_moves,
    PieceType.YZQUEEN: generate_yz_queen_moves,
    PieceType.VECTORSLIDER: generate_vector_slider_moves,
    PieceType.TRAILBLAZER: generate_trailblazer_moves,
    PieceType.NEBULA: generate_nebula_moves,
    PieceType.ORBITER: generate_orbital_moves,
    PieceType.ECHO: generate_echo_moves,
    PieceType.PANEL: generate_panel_moves,
    PieceType.MIRROR: generate_mirror_teleport_move,
    PieceType.FRIENDLYTELEPORTER: generate_network_teleport_moves,
    PieceType.HIVE: generate_pawn_front_teleport_moves,
    PieceType.SPIRAL: generate_spiral_moves,
    PieceType.XZZIGZAG: generate_xz_zigzag_moves,
    PieceType.YZZIGZAG: generate_yz_zigzag_moves,
    PieceType.EDGEROOK: generate_edgerook_moves,
    PieceType.CONESLIDER: generate_face_cone_slider_moves,
    PieceType.REFLECTOR: generate_reflecting_bishop_moves,
    PieceType.SWAPPER: generate_swapper_moves,
}

# Pieces that don't move (or have special behavior)
NON_MOVING_PIECES = {
    PieceType.WALL,
    PieceType.BOMB,
    PieceType.WHITEHOLE,
    PieceType.BLACKHOLE,
    PieceType.FREEZER,
    PieceType.SLOWER,
    PieceType.SPEEDER,
    PieceType.ARMOUR,
    PieceType.PRIEST,
    PieceType.ARCHER,
    PieceType.INFILTRATOR,
    PieceType.GEOMANCER,
}


def coord_to_index(x: int, y: int, z: int) -> int:
    """Convert 3D coordinate to flat index."""
    return z * (SIZE_Y * SIZE_X) + y * SIZE_X + x


def index_to_coord(idx: int) -> Tuple[int, int, int]:
    """Convert flat index to 3D coordinate."""
    z = idx // (SIZE_Y * SIZE_X)
    remainder = idx % (SIZE_Y * SIZE_X)
    y = remainder // SIZE_X
    x = remainder % SIZE_X
    return (x, y, z)


def precompute_all_moves() -> Dict[PieceType, np.ndarray]:
    """
    Precompute all legal moves for every piece type at every position.

    Returns:
        Dict mapping PieceType to 3D boolean array [729, 729] where
        arr[from_idx, to_idx] = True if move is legal (on empty board).
    """
    print("Starting move precomputation for 9×9×9 board...")

    mock_cache = MockCache()
    precomputed = {}

    for piece_type in PieceType:
        if piece_type in NON_MOVING_PIECES:
            print(f"Skipping {piece_type.name} (non-moving piece)")
            continue

        if piece_type not in MOVE_GENERATORS:
            print(f"WARNING: No generator for {piece_type.name}")
            continue

        print(f"Precomputing {piece_type.name}...", end=" ", flush=True)

        # Create move matrix: [from_square, to_square]
        move_matrix = np.zeros((VOLUME, VOLUME), dtype=bool)
        generator = MOVE_GENERATORS[piece_type]

        move_count = 0
        for z in range(SIZE_Z):
            for y in range(SIZE_Y):
                for x in range(SIZE_X):
                    from_idx = coord_to_index(x, y, z)

                    # Generate moves for WHITE (most pieces are symmetric)
                    # For pawns, we'll need both colors
                    try:
                        moves = generator(mock_cache, Color.WHITE, x, y, z)

                        for move in moves:
                            to_x, to_y, to_z = move.to_coord
                            to_idx = coord_to_index(to_x, to_y, to_z)
                            move_matrix[from_idx, to_idx] = True
                            move_count += 1
                    except Exception as e:
                        print(f"\nError generating moves for {piece_type.name} at ({x},{y},{z}): {e}")
                        continue

        precomputed[piece_type] = move_matrix
        print(f"{move_count} moves")

    return precomputed


def save_precomputed_moves(precomputed: Dict[PieceType, np.ndarray], filepath: str = "precomputed_moves.pkl"):
    """Save precomputed moves to disk."""
    print(f"\nSaving precomputed moves to {filepath}...")

    # Convert to serializable format
    data = {
        piece_type.value: moves.astype(np.uint8)  # Save as uint8 to reduce size
        for piece_type, moves in precomputed.items()
    }

    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size = Path(filepath).stat().st_size / (1024 * 1024)
    print(f"Saved {len(data)} piece types ({file_size:.2f} MB)")


def load_precomputed_moves(filepath: str = "precomputed_moves.pkl") -> Dict[PieceType, np.ndarray]:
    """Load precomputed moves from disk."""
    print(f"Loading precomputed moves from {filepath}...")

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    # Convert back to PieceType keys and bool arrays
    precomputed = {
        PieceType(piece_value): moves.astype(bool)
        for piece_value, moves in data.items()
    }

    print(f"Loaded {len(precomputed)} piece types")
    return precomputed


def analyze_precomputed_moves(precomputed: Dict[PieceType, np.ndarray]):
    """Print statistics about precomputed moves."""
    print("\n" + "="*70)
    print("PRECOMPUTED MOVE STATISTICS")
    print("="*70)

    stats = []
    for piece_type, move_matrix in sorted(precomputed.items(), key=lambda x: x[0].name):
        total_moves = move_matrix.sum()
        avg_moves = total_moves / VOLUME
        max_moves = move_matrix.sum(axis=1).max()
        min_moves = move_matrix.sum(axis=1).min()

        stats.append({
            'name': piece_type.name,
            'total': int(total_moves),
            'avg': avg_moves,
            'max': int(max_moves),
            'min': int(min_moves)
        })

    # Print table
    print(f"{'Piece Type':<20} {'Total Moves':>12} {'Avg/Square':>12} {'Max':>8} {'Min':>8}")
    print("-"*70)

    for stat in stats:
        print(f"{stat['name']:<20} {stat['total']:>12,} {stat['avg']:>12.1f} {stat['max']:>8} {stat['min']:>8}")

    print("="*70)
    total_all = sum(s['total'] for s in stats)
    print(f"{'TOTAL':<20} {total_all:>12,}")
    print("="*70)


class PrecomputedMoveTable:
    """Fast lookup table for precomputed moves."""

    def __init__(self, precomputed: Dict[PieceType, np.ndarray]):
        self.tables = precomputed

    def get_legal_targets(self, piece_type: PieceType, x: int, y: int, z: int) -> List[Tuple[int, int, int]]:
        """Get all legal target squares for a piece at (x, y, z)."""
        if piece_type not in self.tables:
            return []

        from_idx = coord_to_index(x, y, z)
        move_vector = self.tables[piece_type][from_idx]
        target_indices = np.nonzero(move_vector)[0]

        return [index_to_coord(idx) for idx in target_indices]

    def is_legal_move(self, piece_type: PieceType, from_pos: Tuple[int, int, int],
                      to_pos: Tuple[int, int, int]) -> bool:
        """Check if a move is legal (on empty board)."""
        if piece_type not in self.tables:
            return False

        from_idx = coord_to_index(*from_pos)
        to_idx = coord_to_index(*to_pos)

        return bool(self.tables[piece_type][from_idx, to_idx])


def main():
    """Generate and save precomputed move tables."""
    import time

    start = time.time()
    precomputed = precompute_all_moves()
    elapsed = time.time() - start

    print(f"\nPrecomputation completed in {elapsed:.2f} seconds")

    analyze_precomputed_moves(precomputed)
    save_precomputed_moves(precomputed)

    # Test loading
    loaded = load_precomputed_moves()
    print("\nVerification: All tables loaded successfully ✓")


if __name__ == "__main__":
    main()
