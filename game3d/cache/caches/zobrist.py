# zobrist.py - PURE NUMPY ZOBRIST HASHING
from __future__ import annotations
from typing import TYPE_CHECKING, Union
import numpy as np
from numba import njit, prange

if TYPE_CHECKING:
    from game3d.board.board import Board
    from game3d.board.board import Board

# Import shared standardized types - single source of truth
from game3d.common.shared_types import (
    COORD_DTYPE, INDEX_DTYPE, BOOL_DTYPE, COLOR_DTYPE,
    PIECE_TYPE_DTYPE, FLOAT_DTYPE, HASH_DTYPE,
    Color, PieceType, Result, COLOR_WHITE, COLOR_BLACK, COLOR_EMPTY,
    SIZE, VOLUME, N_PIECE_TYPES, N_COLOR_PLANES, N_TOTAL_PLANES, MAX_COORD_VALUE, MIN_COORD_VALUE,
    PieceDtype, compute_board_index, PIECE_SLICE
)

# Pure numpy Zobrist hashing implementation - fully vectorized
_PIECE_KEYS: np.ndarray = None
_SIDE_KEY: HASH_DTYPE = 0
_ZOBRIST_BITS = 64  # Number of bits for Zobrist hash
_HASH_MASK = (1 << 63) - 1  # Mask for 63-bit signed values


def _validate_board_size(width: int, height: int, depth: int) -> tuple[int, int, int]:
    """Validate board dimensions using shared_types constants."""
    if width != SIZE or height != SIZE or depth != SIZE:
        raise ValueError(f"Board size must be {SIZE}x{SIZE}x{SIZE}")
    return width, height, depth



def _init_zobrist(width: int = SIZE, height: int = SIZE, depth: int = SIZE) -> None:
    """Initialize Zobrist keys with pure numpy and shared_types constants."""
    global _PIECE_KEYS, _SIDE_KEY

    if _PIECE_KEYS is not None:
        return

    width, height, depth = _validate_board_size(width, height, depth)

    _PIECE_KEYS = np.random.randint(
        0, 2**64,
        size=(N_PIECE_TYPES, N_COLOR_PLANES, SIZE, SIZE, SIZE),  # 5D shape
        dtype=np.uint64
    ).astype(HASH_DTYPE)

    # Use numpy's uint64 random generation
    _SIDE_KEY = HASH_DTYPE(np.random.randint(0, 1 << 63, dtype=np.uint64))


def compute_zobrist(board: Union["Board", np.ndarray], color: int) -> HASH_DTYPE:
    """Compute Zobrist hash for board state using full vectorization."""
    _init_zobrist()  # Ensures global _PIECE_KEYS is initialized

    # ✅ NEW: Handle numpy arrays directly
    if isinstance(board, np.ndarray):
        board_array = board
    elif hasattr(board, 'array'):
        board_array = board.array()
    else:
        raise TypeError(f"Board must be ndarray or have .array(), got {type(board)}")

    # Validate shape
    expected_shape = (N_TOTAL_PLANES, SIZE, SIZE, SIZE)
    if board_array.shape != expected_shape:
        raise ValueError(
            f"Board array shape must be {expected_shape}, got {board_array.shape}"
        )

    # Find occupied positions across all planes
    # Find occupied positions across all planes (only check piece planes)
    occupied_mask = board_array[PIECE_SLICE] > 0
    occupied_indices = np.where(occupied_mask.any(axis=0))

    if len(occupied_indices[0]) == 0:
        return _SIDE_KEY if color == Color.BLACK else HASH_DTYPE(0)

    # Convert to coordinates (z,y,x) then reorder to (x,y,z)
    coords_zyx = np.stack(occupied_indices, axis=1).astype(INDEX_DTYPE)
    coords_array = coords_zyx[:, [2, 1, 0]].astype(COORD_DTYPE)

    # Extract piece types and colors
    occupied_values = board_array[:, occupied_indices[0], occupied_indices[1], occupied_indices[2]]
    piece_planes = np.argmax(occupied_values, axis=0)

    white_mask = piece_planes < N_PIECE_TYPES
    colors_array = np.where(white_mask, Color.WHITE, Color.BLACK).astype(COLOR_DTYPE)
    piece_types_array = np.where(white_mask,
                                piece_planes + 1,
                                piece_planes - N_PIECE_TYPES + 1).astype(PIECE_TYPE_DTYPE)

    # Convert to 0-based indices
    color_indices = (colors_array - Color.WHITE).astype(COLOR_DTYPE)
    piece_types_arr = (piece_types_array - 1).astype(PIECE_TYPE_DTYPE)

    # Compute hash
    keys = _PIECE_KEYS[
        piece_types_arr,
        color_indices,
        coords_array[:, 0], coords_array[:, 1], coords_array[:, 2]
    ]

    zkey = np.bitwise_xor.reduce(keys)
    if color == Color.BLACK:
        zkey ^= _SIDE_KEY

    # Return Python int to ensure hashability
    return int(zkey)

class ZobristHash:
    """Pure numpy Zobrist hashing - fully vectorized and native."""

    def __init__(self):
        """Initialize Zobrist hashing with pure numpy using shared_types."""
        self._piece_keys = np.random.randint(
            0, 1 << 63,
            size=(N_PIECE_TYPES, N_COLOR_PLANES, SIZE, SIZE, SIZE),
            dtype=np.uint64
        ).astype(HASH_DTYPE)
        self._side_key = HASH_DTYPE(np.random.randint(0, 1 << 63, dtype=np.uint64))

    def compute_from_scratch(self, board: "Board", color: int) -> HASH_DTYPE:
        """Compute hash from scratch for board state."""
        # PRIMARY PATH: Use OccupancyCache directly
        if hasattr(board, 'cache_manager') and hasattr(board.cache_manager, 'occupancy_cache'):
            occ_cache = board.cache_manager.occupancy_cache
            coords, piece_types, colors = occ_cache.get_all_occupied_vectorized()

            if coords.shape[0] == 0:
                return self._side_key if color == Color.BLACK else HASH_DTYPE(0)

            piece_types_arr = (piece_types - 1).astype(PIECE_TYPE_DTYPE, copy=False)

            # FIX: Convert color VALUES (1,2) to INDICES (0,1)
            # COLOR_WHITE is likely 1, so subtract 1 to get index 0
            colors_arr = (colors - COLOR_WHITE).astype(COLOR_DTYPE, copy=False)  # ← ADD THIS

            coords_arr = coords.astype(COORD_DTYPE, copy=False)

            keys = self._piece_keys[
                piece_types_arr,
                colors_arr,  # ← Use converted indices, not raw values
                coords_arr[:, 0], coords_arr[:, 1], coords_arr[:, 2]
            ]

            zkey = np.bitwise_xor.reduce(keys)
            if color == Color.BLACK:
                zkey ^= self._side_key
            # Return Python int to ensure hashability
            return int(zkey)

        # Fallback to global function
        return compute_zobrist(board, color)

    def get_piece_key(self, ptype: int, color: int, coord: np.ndarray) -> HASH_DTYPE:
        """Get piece key for given piece type, color, and coordinate using shared_types."""
        coord_array = np.asarray(coord, dtype=COORD_DTYPE)
        if coord_array.ndim == 1 and coord_array.shape[0] == 3:
            return self._piece_keys[
                int(ptype), int(color), coord_array[0], coord_array[1], coord_array[2]
            ]
        raise ValueError("Coordinate must be a 3-element array")

    def update_hash_move(self, current_hash: HASH_DTYPE, mv: np.ndarray,
                        from_piece: object, captured_piece: object | None) -> HASH_DTYPE:
        """Update hash for a move operation using vectorized operations."""
        from_coord = np.asarray(mv[:3], dtype=COORD_DTYPE)
        to_coord = np.asarray(mv[3:], dtype=COORD_DTYPE)

        if from_piece is None:
            raise ValueError(f"from_piece cannot be None for move from {from_coord} to {to_coord}")

        # ✅ FIX: Use Color.WHITE (integer 1) not COLOR_WHITE (numpy array)
        from_piece_type = int(from_piece["piece_type"])
        from_piece_color_idx = int(from_piece["color"]) - Color.WHITE  # 1→0, 2→1

        new_hash = current_hash

        # XOR out piece at source
        new_hash ^= self._piece_keys[
            from_piece_type - 1, from_piece_color_idx,
            from_coord[0], from_coord[1], from_coord[2]
        ]

        if captured_piece:
            cap_piece_type = int(captured_piece["piece_type"])
            cap_piece_color_idx = int(captured_piece["color"]) - Color.WHITE
            # XOR out captured piece at destination
            new_hash ^= self._piece_keys[
                cap_piece_type - 1, cap_piece_color_idx,
                to_coord[0], to_coord[1], to_coord[2]
            ]

        # XOR in piece at destination (after promotion if any)
        # Promotion not currently supported in numpy array move format, assuming no promotion for now
        final_ptype = from_piece_type
        new_hash ^= self._piece_keys[
            int(final_ptype) - 1, from_piece_color_idx,
            to_coord[0], to_coord[1], to_coord[2]
        ]

        new_hash ^= self._side_key
        # Return Python int to ensure hashability in dictionary operations
        return int(new_hash)
    
    def undo_hash_move(self, current_hash: HASH_DTYPE, mv: np.ndarray,
                      moved_piece: object, captured_piece: object | None) -> HASH_DTYPE:
        """Undo hash for a move operation using incremental XOR (reverse of update_hash_move).
        
        OPTIMIZATION: This is O(1) instead of O(n) full recomputation.
        """
        from_coord = np.asarray(mv[:3], dtype=COORD_DTYPE)
        to_coord = np.asarray(mv[3:], dtype=COORD_DTYPE)

        if moved_piece is None:
            raise ValueError(f"moved_piece cannot be None for undo from {to_coord} to {from_coord}")

        # Convert to 0-based indexing - use Color.WHITE not COLOR_WHITE
        piece_type = int(moved_piece["piece_type"])
        piece_color_idx = int(moved_piece["color"]) - Color.WHITE
        
        new_hash = current_hash

        # Undo: Remove piece from destination (reverse of "XOR in at destination")
        final_ptype = piece_type
        new_hash ^= self._piece_keys[
            int(final_ptype) - 1, piece_color_idx,
            to_coord[0], to_coord[1], to_coord[2]
        ]

        # Undo: Restore captured piece if any
        if captured_piece:
            cap_piece_type = int(captured_piece["piece_type"])
            cap_piece_color_idx = int(captured_piece["color"]) - Color.WHITE
            new_hash ^= self._piece_keys[
                cap_piece_type - 1, cap_piece_color_idx,
                to_coord[0], to_coord[1], to_coord[2]
            ]

        # Undo: Restore piece at source (reverse of "XOR out from source")
        new_hash ^= self._piece_keys[
            piece_type - 1, piece_color_idx,
            from_coord[0], from_coord[1], from_coord[2]
        ]

        # Flip side
        new_hash ^= self._side_key
        # Return Python int to ensure hashability
        return int(new_hash)

    def update_hash_piece_placement(
        self,
        current_hash: HASH_DTYPE,
        coord: np.ndarray,
        old_piece: object | None,
        new_piece: object | None
    ) -> HASH_DTYPE:
        """Update hash for piece placement using shared_types."""
        coord_array = np.asarray(coord, dtype=COORD_DTYPE)
        if coord_array.ndim != 1 or coord_array.shape[0] != 3:
            raise ValueError("Coordinate must be a 3-element array")

        new_hash = current_hash

        if old_piece is not None:
            # ✅ FIX: Use Color.WHITE not COLOR_WHITE
            old_piece_type = int(old_piece["piece_type"]) - 1
            old_piece_color_idx = int(old_piece["color"]) - Color.WHITE
            new_hash ^= self._piece_keys[
                old_piece_type, old_piece_color_idx,
                coord_array[0], coord_array[1], coord_array[2]
            ]

        if new_piece is not None:
            # ✅ FIX: Use Color.WHITE not COLOR_WHITE
            new_piece_type = int(new_piece["piece_type"]) - 1
            new_piece_color_idx = int(new_piece["color"]) - Color.WHITE
            new_hash ^= self._piece_keys[
                new_piece_type, new_piece_color_idx,
                coord_array[0], coord_array[1], coord_array[2]
            ]

        # Return Python int to ensure hashability
        return int(new_hash)

    def flip_side(self, current_hash: HASH_DTYPE) -> HASH_DTYPE:
        """Flip the side to move using shared_types."""
        return HASH_DTYPE(current_hash ^ self._side_key)

    def get_side_key(self) -> HASH_DTYPE:
        """Get the side key using shared_types."""
        return HASH_DTYPE(self._side_key)

    def batch_get_piece_keys(self, ptypes: np.ndarray, colors: np.ndarray,
                           coords: np.ndarray) -> np.ndarray:
        """Batch get piece keys using full vectorization."""
        if not all(isinstance(arr, np.ndarray) for arr in [ptypes, colors, coords]):
            raise TypeError("All inputs must be numpy arrays")
        if ptypes.shape != colors.shape or ptypes.shape[0] != coords.shape[0]:
            raise ValueError("Mismatched array shapes")
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("Coordinates must be (N, 3) array")

        ptypes_arr = ptypes.astype(PIECE_TYPE_DTYPE, copy=False)
        colors_arr = colors.astype(COLOR_DTYPE, copy=False)
        coords_arr = coords.astype(COORD_DTYPE, copy=False)

        return self._piece_keys[
            ptypes_arr, colors_arr,
            coords_arr[:, 0], coords_arr[:, 1], coords_arr[:, 2]
        ].astype(HASH_DTYPE)

    def batch_update_hash_piece_placements(
        self,
        current_hashes: np.ndarray,
        coords: np.ndarray,
        old_pieces: np.ndarray | None = None,
        new_pieces: np.ndarray | None = None
    ) -> np.ndarray:
        """Batch update hash for piece placements using full vectorization."""
        if not isinstance(current_hashes, np.ndarray):
            raise TypeError("Current hashes must be numpy array")

        coords_array = coords.reshape(-1, 3)
        result_hashes = current_hashes.astype(HASH_DTYPE, copy=True)

        if old_pieces is not None and len(old_pieces) > 0:
            if not isinstance(old_pieces, np.ndarray):
                old_pieces = np.asarray(old_pieces)
            valid_old_mask = old_pieces != None
            if np.any(valid_old_mask):
                valid_indices = np.where(valid_old_mask)[0]
                valid_coords = coords_array[valid_indices]

                old_types = np.zeros(len(valid_old_mask), dtype=PIECE_TYPE_DTYPE)

                # Mask for non-None pieces
                non_none_mask = valid_old_mask & (old_pieces != None)

                # Vectorized extraction
                if np.any(non_none_mask):
                    old_types[non_none_mask] = (
                        np.array([p["piece_type"] for p in old_pieces[non_none_mask]], dtype=PIECE_TYPE_DTYPE) - 1
                    )
                old_colors = np.array([p["color"] - Color.WHITE if p is not None else 0 for p in old_pieces[valid_old_mask]], dtype=COLOR_DTYPE)

                result_hashes[valid_indices] ^= self._piece_keys[
                    old_types, old_colors,
                    valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]
                ]

        if new_pieces is not None and len(new_pieces) > 0:
            if not isinstance(new_pieces, np.ndarray):
                new_pieces = np.asarray(new_pieces)
            valid_new_mask = new_pieces != None
            if np.any(valid_new_mask):
                valid_indices = np.where(valid_new_mask)[0]
                valid_coords = coords_array[valid_indices]

                new_types = np.array([p["piece_type"] - 1 if p is not None else 0 for p in new_pieces[valid_new_mask]], dtype=PIECE_TYPE_DTYPE)
                new_colors = np.array([p["color"] - Color.WHITE if p is not None else 0 for p in new_pieces[valid_new_mask]], dtype=COLOR_DTYPE)

                result_hashes[valid_indices] ^= self._piece_keys[
                    new_types, new_colors,
                    valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]
                ]

        return result_hashes.astype(HASH_DTYPE)
