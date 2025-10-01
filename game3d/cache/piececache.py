# game3d/cache/piececache.py
from typing import Dict, Optional, Tuple, TYPE_CHECKING
from game3d.pieces.piece import Piece
from game3d.pieces.enums import Color, PieceType
from game3d.common.common import Coord, WHITE_SLICE, BLACK_SLICE, N_PLANES_PER_SIDE, SIZE_X, SIZE_Y, SIZE_Z
import numpy as np
import threading
if TYPE_CHECKING:
    from game3d.board.board import Board  # Only for type checking
else:
    Board = object  # Dummy type for runtime

class PieceCache:
    __slots__ = ("_white_pieces", "_black_pieces", "_valid", "_occ_view", "_lock")

    def __init__(self, board: "Board") -> None:
        self._white_pieces: Dict[Coord, PieceType] = {}
        self._black_pieces: Dict[Coord, PieceType] = {}
        self._valid = False
        self._occ_view: Optional[np.ndarray] = None  # <-- NEW
        self.rebuild(board)
        self._lock = threading.RLock()

    def get(self, coord: Coord) -> Optional[Piece]:
        """O(1) lookup with minimal overhead."""
        if coord in self._white_pieces:
            return Piece(Color.WHITE, self._white_pieces[coord])
        if coord in self._black_pieces:
            return Piece(Color.BLACK, self._black_pieces[coord])
        return None

    def get_type(self, coord: Coord, color: Color) -> Optional[PieceType]:
        """Faster lookup when you know the color."""
        if color == Color.WHITE:
            return self._white_pieces.get(coord)
        else:
            return self._black_pieces.get(coord)

    def iter_color(self, color: Color):
        """Iterate only pieces of specific color."""
        pieces_dict = self._white_pieces if color == Color.WHITE else self._black_pieces
        for coord, ptype in pieces_dict.items():
            yield coord, Piece(color, ptype)

    def rebuild(self, board: "Board") -> None:
        # Invalidate cached view
        if hasattr(self, '_occ_view'):
            delattr(self, '_occ_view')
        self._white_pieces.clear()
        self._black_pieces.clear()

        tensor_np = board._tensor.cpu().numpy() if board._tensor.is_cuda else board._tensor.numpy()

        # Allow 81 planes (80 piece + 1 side-to-move)
        total_piece_planes = 2 * N_PLANES_PER_SIDE  # 80
        expected_full_shape = (total_piece_planes + 1, SIZE_Z, SIZE_Y, SIZE_X)  # (81, 9, 9, 9)
        if tensor_np.shape != expected_full_shape:
            raise ValueError(f"Board tensor shape {tensor_np.shape} != expected {expected_full_shape}")

        # Extract ONLY the piece planes (first 80)
        piece_tensor = tensor_np[:total_piece_planes]  # shape (80, 9, 9, 9)

        # Process WHITE: planes 0 to 39
        white_planes = piece_tensor[0:N_PLANES_PER_SIDE]  # (40, 9, 9, 9)
        for ptype_idx in range(N_PLANES_PER_SIDE):
            mask = white_planes[ptype_idx]
            coords = np.argwhere(mask == 1.0)
            for z, y, x in coords:
                try:
                    ptype = PieceType(ptype_idx)
                    self._white_pieces[(int(x), int(y), int(z))] = ptype
                except ValueError:
                    continue

        # Process BLACK: planes 40 to 79
        black_planes = piece_tensor[N_PLANES_PER_SIDE : 2 * N_PLANES_PER_SIDE]
        for ptype_idx in range(N_PLANES_PER_SIDE):
            mask = black_planes[ptype_idx]
            coords = np.argwhere(mask == 1.0)
            for z, y, x in coords:
                try:
                    ptype = PieceType(ptype_idx)
                    self._black_pieces[(int(x), int(y), int(z))] = ptype
                except ValueError:
                    continue

        self._valid = True

    def export_arrays(self):
        """Snapshot current board into two NumPy arrays."""
        occ  = np.zeros((9, 9, 9), dtype=np.uint8)
        ptyp = np.zeros((9, 9, 9), dtype=np.uint8)

        # self._white_pieces[coord] -> PieceType directly
        for (x, y, z), ptype in self._white_pieces.items():
            occ[x, y, z]  = 1
            ptyp[x, y, z] = ptype.value

        for (x, y, z), ptype in self._black_pieces.items():
            occ[x, y, z]  = 2
            ptyp[x, y, z] = ptype.value

        return occ, ptyp

    def get_occupancy_view(self) -> np.ndarray:
        """Thread-safe occupancy view with copy-on-demand."""
        with self._lock:
            if not hasattr(self, '_occ_view') or not self._valid:
                occ = np.zeros((9, 9, 9), dtype=np.uint8)
                for (x, y, z), _ in self._white_pieces.items():
                    occ[x, y, z] = 1
                for (x, y, z), _ in self._black_pieces.items():
                    occ[x, y, z] = 2
                self._occ_view = occ
            # Return a COPY to prevent external mutation
            return self._occ_view.copy()  # ← Critical change
