# symmetry_tt.py - NUMPY-NATIVE SYMMETRY-AWARE TRANSPOSITION TABLE

import numpy as np
from numba import njit, prange
from typing import Optional, Dict
import time

from game3d.cache.caches.transposition import TranspositionTable, TT_ENTRY_DTYPE, MOVE_DATA_DTYPE
from game3d.board.symmetry import SymmetryManager
from game3d.cache.caches.zobrist import compute_zobrist
from game3d.common.shared_types import Color, HASH_DTYPE, INDEX_DTYPE
from game3d.common.coord_utils import idx_to_coord, coord_to_idx

class SymmetryAwareTranspositionTable(TranspositionTable):
    """Fully vectorized symmetry-aware transposition table."""

    __slots__ = ("_sym", "_stats", "_sym_cache", "_sym_cache_max_size", "_transform_mats", "_inv_transforms")

    # Pre-computed transform metadata
    _TMAP = {
        "identity": 0, "rotate_x_90": 1, "rotate_x_270": 2, "rotate_x_180": 3,
        "rotate_y_90": 4, "rotate_y_270": 5, "rotate_y_180": 6,
        "rotate_z_90": 7, "rotate_z_270": 8, "rotate_z_180": 9
    }
    _INV = np.array([0, 2, 1, 3, 5, 4, 6, 8, 7, 9], dtype=INDEX_DTYPE)  # Inverses

    def __init__(self, symmetry_manager: SymmetryManager, size_mb: int = 6144):
        # Initialize base table with smaller size (symmetry expands coverage)
        super().__init__(size_mb=512)
        self._sym = symmetry_manager

        # Vectorized statistics: [probes, hits, stores, time, cache_hits]
        self._stats = np.zeros(5, dtype=np.float64)

        # ✅ OPTIMIZED: Dict-based symmetry cache for O(1) lookup
        # hash -> (entry_idx, transform_idx)
        self._sym_cache = {}
        self._sym_cache_max_size = 1000000  # 1M entries

        # Pre-computed transformation matrices (10, 3, 3)
        self._transform_mats = np.array([
            np.eye(3, dtype=INDEX_DTYPE),  # identity
            np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=INDEX_DTYPE),  # x90
            np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=INDEX_DTYPE),  # x270
            np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=INDEX_DTYPE), # x180
            np.array([[0,0,1],[0,1,0],[-1,0,0]], dtype=INDEX_DTYPE),  # y90
            np.array([[0,0,-1],[0,1,0],[1,0,0]], dtype=INDEX_DTYPE),  # y270
            np.array([[-1,0,0],[0,1,0],[0,0,-1]], dtype=INDEX_DTYPE), # y180
            np.array([[0,-1,0],[1,0,0],[0,0,1]], dtype=INDEX_DTYPE),  # z90
            np.array([[0,1,0],[-1,0,0],[0,0,1]], dtype=INDEX_DTYPE),  # z270
            np.array([[-1,0,0],[0,-1,0],[0,0,1]], dtype=INDEX_DTYPE)  # z180
        ])

    @njit(cache=True, nogil=True)
    def _apply_transform_vectorized(self, coord: np.ndarray, transform_idx: int) -> np.ndarray:
        """Apply 3D transformation to coordinate."""
        # Center at origin, apply matrix, recenter
        centered = coord - 4
        transformed = self._transform_mats[transform_idx] @ centered
        new_coord = np.round(transformed) + 4
        return np.clip(new_coord, 0, 8).astype(coord.dtype)

    def _transform_entry_back(self, entry: np.ndarray, transform_idx: int) -> np.ndarray:
        """Transform move coordinates from canonical back to original orientation."""
        if transform_idx == 0 or entry['best_move'] == 0:
            return entry

        # Decompose move data
        move_data = entry['best_move']
        from_idx = move_data & 0x3FF
        to_idx = (move_data >> 10) & 0x3FF

        # Convert to coordinates
        from_coord = idx_to_coord(from_idx)
        to_coord = idx_to_coord(to_idx)

        # Apply inverse transform
        inv_idx = self._INV[transform_idx]
        if inv_idx != 0:
            from_coord = self._apply_transform_vectorized(from_coord, inv_idx)
            to_coord = self._apply_transform_vectorized(to_coord, inv_idx)

        # Reconstruct move data with transformed coordinates
        new_move_data = MOVE_DATA_DTYPE(
            (coord_to_idx(from_coord) & 0x3FF) |
            ((coord_to_idx(to_coord) & 0x3FF) << 10) |
            (move_data & ~0xFFFFFFFFFF)  # Preserve flags/metadata
        )

        # Return transformed entry view
        new_entry = entry.copy()
        new_entry['best_move'] = new_move_data
        return new_entry.view()

    def probe_with_symmetry(self, hash_value: int, board) -> Optional[np.ndarray]:
        """
        Probe transposition table with symmetry-aware lookup.

        This method checks for equivalent positions under board symmetries
        by finding the canonical (smallest hash) representation.

        Args:
            hash_value: Zobrist hash of the current board orientation
            board: Board object OR numpy array

        Returns:
            View into TT entry structured array, or None
        """
        # 1. Fast path: Check exact hash match
        exact_idx = self._probe_idx(hash_value)
        if self.entries[exact_idx]['hash_key'] == hash_value:
            return self.entries[exact_idx].view()

        self._stats[0] += 1  # probes

        # ✅ OPTIMIZATION: O(1) dict lookup instead of linear scan
        if hash_value in self._sym_cache:
            self._stats[4] += 1  # cache_hits
            entry_idx, transform_idx = self._sym_cache[hash_value]
            return self._transform_entry_back(self.entries[entry_idx].view(), transform_idx)

        # 3. Compute canonical form using symmetry manager
        # Must return board array and transform index
        start = time.perf_counter()
        canonical_board, transform_idx = self._sym.get_canonical_form(board)
        self._stats[3] += time.perf_counter() - start

        # Compute hash of canonical form (WHITE perspective for consistency)
        canonical_hash = compute_zobrist(canonical_board, Color.WHITE)

        # 4. Probe transposition table with canonical hash
        canon_idx = self._probe_idx(canonical_hash)
        canonical_entry = self.entries[canon_idx]

        if canonical_entry['hash_key'] == canonical_hash:
            self._stats[1] += 1  # hits

            # ✅ OPTIMIZATION: Store in dict with LRU eviction
            if len(self._sym_cache) >= self._sym_cache_max_size:
                # Simple LRU: Remove first (oldest) entry
                first_key = next(iter(self._sym_cache))
                del self._sym_cache[first_key]
            
            self._sym_cache[hash_value] = (canon_idx, transform_idx)

            # 6. Transform move coordinates back to original orientation
            return self._transform_entry_back(canonical_entry.view(), transform_idx)

        return None

    def store_with_symmetry(self, hash_value: int, board, depth: int, score: int,
                           node_type: int, best_move: Optional[object] = None) -> None:
        """Store entry with symmetry canonicalization."""
        # Store in original orientation
        super().store(hash_value, depth, score, node_type, best_move, self.age_counter)
        self._stats[2] += 1  # stores

        # Store in canonical orientation for deep entries
        if depth >= 3:
            canonical_board, canonical_transform = self._sym.get_canonical_form(board)
            canonical_hash = compute_zobrist(canonical_board, Color.WHITE)

            # Transform move if needed
            canonical_move = best_move
            if canonical_transform != "identity":
                # Transform move coordinates
                from_coord = self._apply_transform_vectorized(
                    best_move.from_coord, self._TMAP[canonical_transform]
                )
                to_coord = self._apply_transform_vectorized(
                    best_move.to_coord, self._TMAP[canonical_transform]
                )
                canonical_move = type(best_move)(from_coord, to_coord, best_move.piece_type,
                                               best_move.is_capture, best_move.captured_type,
                                               best_move.is_promotion)

            super().store(canonical_hash, depth, score, node_type, canonical_move, self.age_counter)

    def get_symmetry_stats(self) -> Dict[str, float]:
        """Return statistics as numpy-compatible dictionary."""
        total_probes = int(self._stats[0])
        return {
            'probes': total_probes,
            'hits': int(self._stats[1]),
            'stores': int(self._stats[2]),
            'time_ms': self._stats[3] * 1000,
            'cache_hits': int(self._stats[4]),
            'hit_rate': self._stats[1] / max(total_probes, 1),
            'cache_size': len(self._sym_cache)
        }
