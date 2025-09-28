# game3d/cache/transposition.py

from __future__ import annotations
from typing import Dict, Any, List, Optional
from game3d.pieces.enums import PieceType

def get_stats(self) -> Dict[str, Any]:
    """Get basic statistics for the transposition table."""
    total_accesses = self.hits + self.misses
    return {
        'hits': self.hits,
        'misses': self.misses,
        'collisions': self.collisions,
        'hit_rate': self.hits / max(1, total_accesses),
        'size': self.size,
    }

def get_memory_usage(self) -> int:
    """Approximate memory usage in bytes."""
    return self.size * 32  # Assuming ~32 bytes per TTEntry

class CompactMove:
    """Ultra-compact move representation using bit packing."""
    __slots__ = ('data',)
    def __init__(self, from_coord: Tuple[int, int, int], to_coord: Tuple[int, int, int],
                 piece_type: PieceType, is_capture: bool = False,
                 captured_type: Optional[PieceType] = None,
                 is_promotion: bool = False):

        from_index = from_coord[0] * 81 + from_coord[1] * 9 + from_coord[2]
        to_index = to_coord[0] * 81 + to_coord[1] * 9 + to_coord[2]

        self.data = (from_index & 0x1FFFFF) | \
                   ((to_index & 0x1FFFFF) << 21) | \
                   ((piece_type.value & 0x3F) << 42) | \
                   ((captured_type.value & 0x3F) << 48 if captured_type else 0) | \
                   (1 << 54 if is_capture else 0) | \
                   (1 << 55 if is_promotion else 0)

    def unpack(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], bool, bool]:
        """Unpack move data."""
        from_index = self.data & 0x1FFFFF
        to_index = (self.data >> 21) & 0x1FFFFF
        is_capture = bool(self.data & (1 << 54))
        is_promotion = bool(self.data & (1 << 55))

        from_coord = (from_index // 81, (from_index % 81) // 9, from_index % 9)
        to_coord = (to_index // 81, (to_index % 81) // 9, to_index % 9)

        return from_coord, to_coord, is_capture, is_promotion

class TTEntry:
    """Transposition table entry with optimized memory layout."""
    __slots__ = ('hash_key', 'depth', 'score', 'node_type', 'best_move', 'age')
    def __init__(self, hash_key: int, depth: int, score: int, node_type: int,
                 best_move: Optional['CompactMove'], age: int):
        self.hash_key = hash_key
        self.depth = depth
        self.score = score
        self.node_type = node_type  # 0=exact, 1=lower, 2=upper
        self.best_move = best_move
        self.age = age

class TranspositionTable:
    """High-performance transposition table with advanced replacement strategy."""
    def __init__(self, size_mb: int = 256):
        # Size should be power of 2 for efficient masking
        self.size = (size_mb * 1024 * 1024) // 32  # 32 bytes per entry
        self.table: List[Optional[TTEntry]] = [None] * self.size
        self.mask = self.size - 1
        self.age_counter = 0

        # Statistics for performance monitoring
        self.hits = 0
        self.misses = 0
        self.collisions = 0

    def probe(self, hash_value: int) -> Optional[TTEntry]:
        index = hash_value & self.mask
        entry = self.table[index]
        if entry is not None and entry.hash_key == hash_value:
            self.hits += 1
            return entry
        else:
            self.misses += 1
            return None

    def store(self, hash_value: int, depth: int, score: int, node_type: int,
            best_move: Optional[CompactMove], age: int) -> None:
        index = hash_value & self.mask
        existing_entry = self.table[index]
        should_replace = (existing_entry is None or
                        existing_entry.depth <= depth or
                        existing_entry.age < age - 4)
        if should_replace:
            if existing_entry is not None:
                self.collisions += 1
            self.table[index] = TTEntry(hash_value, depth, score, node_type, best_move, age)
