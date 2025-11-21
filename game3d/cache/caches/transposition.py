import numpy as np
from typing import Optional
from numba import njit, prange

from game3d.common.coord_utils import coord_to_idx, idx_to_coord
from game3d.common.shared_types import (
    COORD_DTYPE, INDEX_DTYPE, HASH_DTYPE, PIECE_TYPE_DTYPE, NODE_TYPE_DTYPE
)

# NUMPY-STRUCTURED TTEntry - zero Python overhead
TT_ENTRY_DTYPE = np.dtype([
    ('hash_key', HASH_DTYPE),
    ('depth', INDEX_DTYPE),
    ('score', INDEX_DTYPE),
    ('node_type', NODE_TYPE_DTYPE),
    ('best_move', np.uint64),  # Bit-packed move data
    ('age', INDEX_DTYPE)
])

MOVE_DATA_DTYPE = np.uint64

class CompactMove:
    """Bit-packed move representation - purely numpy."""
    __slots__ = ('data',)

    def __init__(self, from_coord: np.ndarray, to_coord: np.ndarray, piece_type,
                 is_capture: bool = False, captured_type: Optional[int] = None,
                 is_promotion: bool = False):
        # Vectorized conversion
        from_idx = coord_to_idx(np.asarray(from_coord, dtype=COORD_DTYPE))
        to_idx = coord_to_idx(np.asarray(to_coord, dtype=COORD_DTYPE))

        piece_val = piece_type.value if hasattr(piece_type, 'value') else piece_type
        captured_val = captured_type.value if captured_type and hasattr(captured_type, 'value') else (captured_type or 0)

        self.data = MOVE_DATA_DTYPE(
            (from_idx & 0x3FF) |
            ((to_idx & 0x3FF) << 10) |
            ((piece_val & 0x3F) << 20) |
            ((captured_val & 0x3F) << 26) |
            ((1 if is_capture else 0) << 32) |
            ((1 if is_promotion else 0) << 33)
        )

    @property
    def from_coord(self) -> np.ndarray:
        return idx_to_coord(int(self.data & 0x3FF))

    @property
    def to_coord(self) -> np.ndarray:
        return idx_to_coord(int((self.data >> 10) & 0x3FF))

    @property
    def is_capture(self) -> bool:
        return bool(self.data & (1 << 32))

    @property
    def is_promotion(self) -> bool:
        return bool(self.data & (1 << 33))

@njit(cache=True, nogil=True)
def _compute_probe_idx(size: int, hash_value: int) -> int:
    """Numba-compiled index calculation."""
    return hash_value & (size - 1)

class TranspositionTable:
    """Fully vectorized transposition table - zero Python loops."""
    __slots__ = ('size', 'entries', 'hits', 'misses', 'collisions', 'age_counter')

    def __init__(self, size_mb: int = 1024):
        raw_size = int(size_mb * 1024 * 1024 / TT_ENTRY_DTYPE.itemsize)
        self.size = 1 << (raw_size.bit_length() - 1)

        # SINGLE structured array for all entries
        self.entries = np.zeros(self.size, dtype=TT_ENTRY_DTYPE)
        self.hits = self.misses = self.collisions = self.age_counter = 0

    def _probe_idx(self, hash_value: int) -> int:
        """Index calculation wrapper."""
        return _compute_probe_idx(self.size, hash_value)

    def probe(self, hash_value: int) -> Optional[np.ndarray]:
        """Return TT entry as numpy structured array slice."""
        idx = self._probe_idx(hash_value)
        entry = self.entries[idx]

        if entry['hash_key'] == hash_value:
            self.hits += 1
            return entry.view()  # Return view, not copy
        self.misses += 1
        return None

    def store(self, hash_value: int, depth: int, score: int, node_type: int,
              best_move: Optional[CompactMove], age: int) -> None:
        """Vectorized store with numpy boolean masking."""
        idx = self._probe_idx(hash_value)
        entry = self.entries[idx]

        should_replace = (entry['hash_key'] == 0) or (node_type == 0) or \
                        (entry['depth'] <= depth - 2) or \
                        (entry['age'] < age - 8) or \
                        ((entry['depth'] <= depth) and (entry['age'] < age))

        if should_replace:
            if entry['hash_key'] != 0:
                self.collisions += 1

            # Vectorized assignment
            entry['hash_key'] = hash_value
            entry['depth'] = depth
            entry['score'] = score
            entry['node_type'] = node_type
            entry['best_move'] = best_move.data if best_move else 0
            entry['age'] = age

    def probe_batch(self, hash_values: np.ndarray) -> np.ndarray:
        """Fully vectorized batch probe."""
        indices = hash_values & (self.size - 1)
        matches = self.entries['hash_key'][indices] == hash_values

        results = np.empty(len(hash_values), dtype=object)
        results[~matches] = None

        # For matches, return entry views
        match_indices = np.where(matches)[0]
        for idx in match_indices:
            results[idx] = self.entries[indices[idx]].view()

        self.hits += matches.sum()
        self.misses += (~matches).sum()

        return results

    def get_stats(self) -> dict:
        total = self.hits + self.misses
        return {
            'hits': self.hits, 'misses': self.misses,
            'collisions': self.collisions,
            'hit_rate': self.hits / max(total, 1),
            'entries': int(np.count_nonzero(self.entries['hash_key']))
        }
