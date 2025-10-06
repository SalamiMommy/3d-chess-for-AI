from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from game3d.pieces.enums import PieceType

# ==============================================================================
# COMPACT MOVE REPRESENTATION – 56 BITS TOTAL (FIXED COORDINATE ORDERING)
# ==============================================================================
class CompactMove:
    """Ultra-compact move representation using bit packing (56 bits total)."""
    __slots__ = ('data',)

    def __init__(self, from_coord: Tuple[int, int, int], to_coord: Tuple[int, int, int],
                 piece_type: PieceType, is_capture: bool = False,
                 captured_type: Optional[PieceType] = None,
                 is_promotion: bool = False):
        """
        Initialize compact move with proper coordinate ordering.

        CRITICAL: Uses z*81 + y*9 + x to match Board tensor indexing (z, y, x).
        """
        # Extract coordinates
        fx, fy, fz = from_coord
        tx, ty, tz = to_coord

        # FIXED: Use z*81 + y*9 + x (matches tensor [plane, z, y, x] layout)
        from_index = fz * 81 + fy * 9 + fx  # 0–728
        to_index = tz * 81 + ty * 9 + tx    # 0–728

        self.data = (from_index & 0x3FF) | \
                   ((to_index & 0x3FF) << 10) | \
                   ((piece_type.value & 0x3F) << 20) | \
                   ((captured_type.value & 0x3F) << 26 if captured_type else 0) | \
                   (1 << 32 if is_capture else 0) | \
                   (1 << 33 if is_promotion else 0)

    def unpack(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], bool, bool]:
        """
        Unpack move data with correct coordinate ordering.

        Returns: (from_coord, to_coord, is_capture, is_promotion)
        where coords are (x, y, z) tuples.
        """
        from_index = self.data & 0x3FF
        to_index = (self.data >> 10) & 0x3FF
        is_capture = bool(self.data & (1 << 32))
        is_promotion = bool(self.data & (1 << 33))

        # FIXED: Unpack using z*81 + y*9 + x formula
        # Given: index = z*81 + y*9 + x
        # Extract: z = index // 81, remainder = index % 81
        #          y = remainder // 9, x = remainder % 9

        from_z = from_index // 81
        remainder = from_index % 81
        from_y = remainder // 9
        from_x = remainder % 9

        to_z = to_index // 81
        remainder = to_index % 81
        to_y = remainder // 9
        to_x = remainder % 9

        # Return as (x, y, z) tuples for consistency with external API
        from_coord = (from_x, from_y, from_z)
        to_coord = (to_x, to_y, to_z)

        return from_coord, to_coord, is_capture, is_promotion

    def get_piece_type(self) -> PieceType:
        """Extract piece type from packed data."""
        piece_value = (self.data >> 20) & 0x3F
        return PieceType(piece_value)

    def get_captured_type(self) -> Optional[PieceType]:
        """Extract captured piece type if any."""
        if not (self.data & (1 << 32)):  # Check is_capture bit
            return None
        captured_value = (self.data >> 26) & 0x3F
        if captured_value == 0:
            return None
        return PieceType(captured_value)

    def __eq__(self, other):
        """Compare compact moves."""
        if not isinstance(other, CompactMove):
            return False
        return self.data == other.data

    def __hash__(self):
        """Hash for use in sets/dicts."""
        return hash(self.data)

    def __repr__(self):
        """String representation for debugging."""
        from_coord, to_coord, is_capture, is_promotion = self.unpack()
        flags = []
        if is_capture:
            flags.append("capture")
        if is_promotion:
            flags.append("promotion")
        flag_str = f" ({', '.join(flags)})" if flags else ""
        return f"CompactMove({from_coord} -> {to_coord}{flag_str})"


# ==============================================================================
# TRANSPOSITION TABLE ENTRY – 32 BYTES EXACT (NO CHANGES NEEDED)
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
# HIGH-PERFORMANCE TRANSPOSITION TABLE – 6GB DEFAULT (NO CHANGES NEEDED)
# ==============================================================================
class TranspositionTable:
    """High-performance transposition table optimized for 64GB RAM systems."""
    __slots__ = ('size', 'table', 'age_counter', 'hits', 'misses', 'collisions')

    def __init__(self, size_mb: int = 6144):  # 🔥 Default: 6 GB
        # Ensure size is power of 2 for fastest masking
        entry_size_bytes = 32
        raw_size = (size_mb * 1024 * 1024) // entry_size_bytes
        # Round down to nearest power of two
        self.size = 1 << (raw_size.bit_length() - 1)

        # Pre-allocate table as dict for sparse allocation
        self.table: Dict[int, TTEntry] = {}
        self.age_counter = 0

        # Statistics
        self.hits = 0
        self.misses = 0
        self.collisions = 0

    def probe(self, hash_value: int) -> Optional[TTEntry]:
        entry = self.table.get(hash_value)
        if entry:
            self.hits += 1
            return entry
        else:
            self.misses += 1
            return None

    def store(self, hash_value: int, depth: int, score: int, node_type: int,
              best_move: Optional[CompactMove], age: int) -> None:
        existing = self.table.get(hash_value)

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
            self.table[hash_value] = TTEntry(hash_value, depth, score, node_type, best_move, age)

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
        return len(self.table) * 32  # Approximate, since dict has overhead

    def clear(self) -> None:
        """Clear the entire table (e.g., between games)."""
        self.table.clear()
        self.hits = self.misses = self.collisions = 0
        self.age_counter = 0
