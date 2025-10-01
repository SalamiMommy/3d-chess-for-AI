from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from game3d.pieces.enums import PieceType

# ==============================================================================
# COMPACT MOVE REPRESENTATION — 56 BITS TOTAL
# ==============================================================================
class CompactMove:
    """Ultra-compact move representation using bit packing (56 bits total)."""
    __slots__ = ('data',)

    def __init__(self, from_coord: Tuple[int, int, int], to_coord: Tuple[int, int, int],
                 piece_type: PieceType, is_capture: bool = False,
                 captured_type: Optional[PieceType] = None,
                 is_promotion: bool = False):
        # Encode 9x9x9 = 729 positions → 10 bits each (0–1023 covers 0–728)
        from_index = from_coord[0] * 81 + from_coord[1] * 9 + from_coord[2]  # 0–728
        to_index = to_coord[0] * 81 + to_coord[1] * 9 + to_coord[2]          # 0–728

        self.data = (from_index & 0x3FF) | \
                   ((to_index & 0x3FF) << 10) | \
                   ((piece_type.value & 0x3F) << 20) | \
                   ((captured_type.value & 0x3F) << 26 if captured_type else 0) | \
                   (1 << 32 if is_capture else 0) | \
                   (1 << 33 if is_promotion else 0)

    def unpack(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], bool, bool]:
        """Unpack move data."""
        from_index = self.data & 0x3FF
        to_index = (self.data >> 10) & 0x3FF
        is_capture = bool(self.data & (1 << 32))
        is_promotion = bool(self.data & (1 << 33))

        from_coord = (from_index // 81, (from_index % 81) // 9, from_index % 9)
        to_coord = (to_index // 81, (to_index % 81) // 9, to_index % 9)

        return from_coord, to_coord, is_capture, is_promotion


# ==============================================================================
# TRANSPOSITION TABLE ENTRY — 32 BYTES EXACT
# ==============================================================================
class TTEntry:
    """Transposition table entry with tight 32-byte memory layout."""
    __slots__ = ('hash_key', 'depth', 'score', 'node_type', 'best_move', 'age')

    def __init__(self, hash_key: int, depth: int, score: int, node_type: int,
                 best_move: Optional['CompactMove'], age: int):
        self.hash_key = hash_key      # 8 bytes
        self.depth = depth            # 4 bytes
        self.score = score            # 4 bytes
        self.node_type = node_type    # 4 bytes (0=exact, 1=lower, 2=upper)
        self.best_move = best_move    # 8 bytes (pointer)
        self.age = age                # 4 bytes → total = 32 bytes


# ==============================================================================
# HIGH-PERFORMANCE TRANSPOSITION TABLE — 6GB DEFAULT
# ==============================================================================
class TranspositionTable:
    """High-performance transposition table optimized for 64GB RAM systems."""
    __slots__ = ('size', 'table', 'mask', 'age_counter', 'hits', 'misses', 'collisions')

    def __init__(self, size_mb: int = 6144):  # 🔥 Default: 6 GB
        # Ensure size is power of 2 for fastest masking
        entry_size_bytes = 32
        raw_size = (size_mb * 1024 * 1024) // entry_size_bytes
        # Round down to nearest power of two
        self.size = 1 << (raw_size.bit_length() - 1)
        self.mask = self.size - 1

        # Pre-allocate table (6 GB → ~200 million entries)
        self.table: List[Optional[TTEntry]] = [None] * self.size
        self.age_counter = 0

        # Statistics
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
        existing = self.table[index]

        # 🔥 Advanced replacement: prefer deeper, newer, or exact entries
        should_replace = (
            existing is None or
            node_type == 0 or  # Always store exact scores
            existing.depth <= depth - 2 or  # Store if significantly deeper
            existing.age < age - 8 or  # Replace stale entries
            (existing.depth <= depth and existing.age < age)  # Tie-break by age
        )

        if should_replace:
            if existing is not None:
                self.collisions += 1
            self.table[index] = TTEntry(hash_value, depth, score, node_type, best_move, age)

    def get_stats(self) -> Dict[str, Any]:
        """Get basic statistics for the transposition table."""
        total_accesses = self.hits + self.misses
        return {
            'hits': self.hits,
            'misses': self.misses,
            'collisions': self.collisions,
            'hit_rate': self.hits / max(1, total_accesses),
            'size': self.size,
            'size_mb': (self.size * 32) // (1024 * 1024),
        }

    def get_memory_usage(self) -> int:
        """Approximate memory usage in bytes."""
        return self.size * 32  # 32 bytes per TTEntry

    def clear(self) -> None:
        """Clear the entire table (e.g., between games)."""
        self.table = [None] * self.size
        self.hits = self.misses = self.collisions = 0
        self.age_counter = 0
