# transposition.py (updated with common modules)
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

from game3d.common.enums import PieceType
from game3d.common.coord_utils import coord_to_idx, idx_to_coord
from game3d.common.debug_utils import CacheStatsMixin

class CompactMove:
    __slots__ = ('data',)

    def __init__(self, from_coord: Tuple[int, int, int], to_coord: Tuple[int, int, int],
                 piece_type: PieceType, is_capture: bool = False,
                 captured_type: Optional[PieceType] = None,
                 is_promotion: bool = False):
        fx, fy, fz = from_coord
        tx, ty, tz = to_coord

        from_index = coord_to_idx((fx, fy, fz))
        to_index = coord_to_idx((tx, ty, tz))

        self.data = (from_index & 0x3FF) | \
                   ((to_index & 0x3FF) << 10) | \
                   ((piece_type.value & 0x3F) << 20) | \
                   ((captured_type.value & 0x3F) << 26 if captured_type else 0) | \
                   (1 << 32 if is_capture else 0) | \
                   (1 << 33 if is_promotion else 0)

    def unpack(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], bool, bool]:
        from_index = self.data & 0x3FF
        to_index = (self.data >> 10) & 0x3FF
        is_capture = bool(self.data & (1 << 32))
        is_promotion = bool(self.data & (1 << 33))

        from_coord = idx_to_coord(from_index)
        to_coord = idx_to_coord(to_index)

        return from_coord, to_coord, is_capture, is_promotion

    def get_piece_type(self) -> PieceType:
        piece_value = (self.data >> 20) & 0x3F
        return PieceType(piece_value)

    def get_captured_type(self) -> Optional[PieceType]:
        if not (self.data & (1 << 32)):
            return None
        captured_value = (self.data >> 26) & 0x3F
        if captured_value == 0:
            return None
        return PieceType(captured_value)

    def __eq__(self, other):
        if not isinstance(other, CompactMove):
            return False
        return self.data == other.data

    def __hash__(self):
        return hash(self.data)

    def __repr__(self):
        from_coord, to_coord, is_capture, is_promotion = self.unpack()
        flags = []
        if is_capture:
            flags.append("capture")
        if is_promotion:
            flags.append("promotion")
        flag_str = f" ({', '.join(flags)})" if flags else ""
        return f"CompactMove({from_coord} -> {to_coord}{flag_str})"


class TTEntry:
    __slots__ = ('hash_key', 'depth', 'score', 'node_type', 'best_move', 'age')

    def __init__(self, hash_key: int, depth: int, score: int, node_type: int,
                 best_move: Optional['CompactMove'], age: int):
        self.hash_key = hash_key
        self.depth = depth
        self.score = score
        self.node_type = node_type
        self.best_move = best_move
        self.age = age


class TranspositionTable(CacheStatsMixin):
    __slots__ = ('size', 'table', 'age_counter', 'hits', 'misses', 'collisions')

    def __init__(self, size_mb: int = 6144):
        entry_size_bytes = 32
        raw_size = (size_mb * 1024 * 1024) // entry_size_bytes
        self.size = 1 << (raw_size.bit_length() - 1)

        self.table: Dict[int, TTEntry] = {}
        self.age_counter = 0

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

        should_replace = (
            existing is None or
            node_type == 0 or
            existing.depth <= depth - 2 or
            existing.age < age - 8 or
            (existing.depth <= depth and existing.age < age)
        )

        if should_replace:
            if existing is not None:
                self.collisions += 1
            self.table[hash_value] = TTEntry(hash_value, depth, score, node_type, best_move, age)

    def get_stats(self) -> Dict[str, Any]:
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
        return len(self.table) * 32

    def clear(self) -> None:
        self.table.clear()
        self.hits = self.misses = self.collisions = 0
        self.age_counter = 0
