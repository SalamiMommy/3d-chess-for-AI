# game3d/cache/transposition.py
from typing import Optional, List
import random

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

    def probe(self, hash_value: int) -> Optional[TTEntry]:
        """Look up cached evaluation for this position."""
        index = hash_value & self.mask
        entry = self.table[index]
        if entry is not None and entry.hash_key == hash_value:
            return entry
        return None

    def store(self, hash_value: int, depth: int, score: int, node_type: int,
              best_move: Optional['CompactMove'], age: int) -> None:
        """Store evaluation with advanced replacement strategy."""
        index = hash_value & self.mask
        existing_entry = self.table[index]
        # Replacement strategy: always replace if deeper, or if old entry is too old
        should_replace = (existing_entry is None or
                         existing_entry.depth <= depth or
                         existing_entry.age < age - 4)
        if should_replace:
            self.table[index] = TTEntry(hash_value, depth, score, node_type, best_move, age)
