from __future__ import annotations
"""Optimized symmetry-aware transposition table – tuned for Ryzen 5 5600X + 64GB RAM."""
# game3d/cache/symmetry_tt.py

from typing import Optional, Dict, Any, Set, Tuple, List
from dataclasses import dataclass
import time
from collections import OrderedDict

from game3d.cache.caches.transposition import TranspositionTable, TTEntry, CompactMove
from game3d.board.symmetry import SymmetryManager, RotationType
from game3d.pieces.enums import Color
from game3d.game.zobrist import compute_zobrist

@dataclass
class SymmetryStats:
    """Statistics for symmetry operations."""
    __slots__ = ("probe_count", "hit_count", "store_count", "transform_time", "canonical_hits")
    probe_count: int
    hit_count: int
    store_count: int
    transform_time: float
    canonical_hits: int

    def __init__(self):
        self.probe_count = 0
        self.hit_count = 0
        self.store_count = 0
        self.transform_time = 0.0
        self.canonical_hits = 0


class LRUCache:
    """Simple LRU cache with size limit."""
    def __init__(self, maxsize: int = 5000):
        self.maxsize = maxsize
        self.cache = OrderedDict()

    def get(self, key: int) -> Optional[Tuple[int, str]]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: int, value: Tuple[int, str]) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.maxsize:
            self.cache.popitem(last=False)
        self.cache[key] = value

    def clear(self) -> None:
        self.cache.clear()


class SymmetryAwareTranspositionTable(TranspositionTable):
    """High-performance symmetry-aware TT – 6GB default, Zobrist-based, 3D-optimized."""
    __slots__ = (
        "_symmetry_manager", "_symmetry_stats", "_canonical_cache",
        "_inverse_transform_map", "_zobrist"
    )

    def __init__(self, symmetry_manager: SymmetryManager, size_mb: int = 6144):
        super().__init__(size_mb)
        self._symmetry_manager = symmetry_manager
        self._symmetry_stats = SymmetryStats()
        self._canonical_cache = LRUCache(maxsize=10000)

        # Precompute inverse transforms for all 24 rotations
        self._inverse_transform_map = {
            RotationType.IDENTITY: RotationType.IDENTITY,
            RotationType.ROTATE_X_90: RotationType.ROTATE_X_270,
            RotationType.ROTATE_X_270: RotationType.ROTATE_X_90,
            RotationType.ROTATE_X_180: RotationType.ROTATE_X_180,
            RotationType.ROTATE_Y_90: RotationType.ROTATE_Y_270,
            RotationType.ROTATE_Y_270: RotationType.ROTATE_Y_90,
            RotationType.ROTATE_Y_180: RotationType.ROTATE_Y_180,
            RotationType.ROTATE_Z_90: RotationType.ROTATE_Z_270,
            RotationType.ROTATE_Z_270: RotationType.ROTATE_Z_90,
            RotationType.ROTATE_Z_180: RotationType.ROTATE_Z_180,
            # Vertex rotations
            RotationType.ROTATE_XYZ_120: RotationType.ROTATE_XYZ_240,
            RotationType.ROTATE_XYZ_240: RotationType.ROTATE_XYZ_120,
            RotationType.ROTATE_XYmZ_120: RotationType.ROTATE_XYmZ_240,
            RotationType.ROTATE_XYmZ_240: RotationType.ROTATE_XYmZ_120,
            RotationType.ROTATE_XmYZ_120: RotationType.ROTATE_XmYZ_240,
            RotationType.ROTATE_XmYZ_240: RotationType.ROTATE_XmYZ_120,
            RotationType.ROTATE_XmYmZ_120: RotationType.ROTATE_XmYmZ_240,
            RotationType.ROTATE_XmYmZ_240: RotationType.ROTATE_XmYmZ_120,
            # Edge rotations (self-inverse)
            RotationType.ROTATE_XY_EDGE: RotationType.ROTATE_XY_EDGE,
            RotationType.ROTATE_XmY_EDGE: RotationType.ROTATE_XmY_EDGE,
            RotationType.ROTATE_XZ_EDGE: RotationType.ROTATE_XZ_EDGE,
            RotationType.ROTATE_XmZ_EDGE: RotationType.ROTATE_XmZ_EDGE,
            RotationType.ROTATE_YZ_EDGE: RotationType.ROTATE_YZ_EDGE,
            RotationType.ROTATE_YmZ_EDGE: RotationType.ROTATE_YmZ_EDGE,
        }

    # --------------------------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------------------------
    def probe_with_symmetry(self, hash_value: int, board) -> Optional[TTEntry]:
        # Fast path: exact match
        exact_hit = self.probe(hash_value)
        if exact_hit:
            return exact_hit

        self._symmetry_stats.probe_count += 1

        # Check canonical cache
        canonical_result = self._canonical_cache.get(hash_value)
        if canonical_result:
            canonical_hash, transform_name = canonical_result
            canonical_entry = self.probe(canonical_hash)
            if canonical_entry:
                self._symmetry_stats.canonical_hits += 1
                return self._transform_entry_back(canonical_entry, transform_name)

        # Compute canonical form ONCE
        start = time.perf_counter()
        try:
            canonical_board, canonical_transform = self._symmetry_manager.get_canonical_form(board)
            canonical_hash = self._compute_zobrist_hash(canonical_board)
        except Exception:
            self._symmetry_stats.transform_time += time.perf_counter() - start
            return None
        self._symmetry_stats.transform_time += time.perf_counter() - start

        # Probe canonical form
        canonical_entry = self.probe(canonical_hash)
        if canonical_entry:
            self._symmetry_stats.hit_count += 1
            self._canonical_cache.put(hash_value, (canonical_hash, canonical_transform))
            return self._transform_entry_back(canonical_entry, canonical_transform)

        return None

    def store_with_symmetry(self, hash_value: int, board, depth: int,
                        score: int, node_type: int, best_move: Optional[CompactMove] = None) -> None:
        # Store original
        self.store(hash_value, depth, score, node_type, best_move, self.age_counter)
        self._symmetry_stats.store_count += 1

        # Store canonical form for depth ≥ 3
        if depth >= 3:
            try:
                canonical_board, canonical_transform = self._symmetry_manager.get_canonical_form(board)
                canonical_hash = self._compute_zobrist_hash(canonical_board)
                canonical_move = best_move
                if best_move and canonical_transform != "identity":
                    canonical_move = self._transform_move(best_move, canonical_transform)
                self.store(canonical_hash, depth, score, node_type, canonical_move, self.age_counter)
                self._canonical_cache.put(hash_value, (canonical_hash, canonical_transform))
            except Exception:
                pass

    def get_symmetry_stats(self) -> Dict[str, Any]:
        total_probes = max(1, self._symmetry_stats.probe_count)
        return {
            'symmetry_probes': self._symmetry_stats.probe_count,
            'symmetry_hits': self._symmetry_stats.hit_count,
            'symmetry_hit_rate': self._symmetry_stats.hit_count / total_probes,
            'canonical_hits': self._symmetry_stats.canonical_hits,
            'transform_time': self._symmetry_stats.transform_time,
            'cache_size': len(self._canonical_cache.cache),
            'memory_usage_mb': self.get_memory_usage() / (1024 * 1024),
        }

    # --------------------------------------------------------------------------
    # INTERNAL HELPERS
    # --------------------------------------------------------------------------
    def _compute_zobrist_hash(self, board) -> int:
        """Compute Zobrist hash without side-to-move (for canonical form)."""
        return compute_zobrist(board, Color.WHITE)

    def _transform_entry_back(self, entry: TTEntry, transform_name: str) -> TTEntry:
        """Transform entry back using transformation name (string)."""
        if transform_name == "identity" or not entry.best_move:
            return entry
        original_move = self._transform_move_back(entry.best_move, transform_name)
        return TTEntry(
            hash_key=entry.hash_key,
            depth=entry.depth,
            score=entry.score,
            node_type=entry.node_type,
            best_move=original_move,
            age=entry.age
        )

    def _transform_move(self, move: CompactMove, transform_name: str) -> CompactMove:
        """Transform move TO canonical coordinates using string name."""
        if transform_name == "identity":
            return move
        from_coord, to_coord, is_capture, is_promotion = move.unpack()
        from_canon = self._apply_transform_to_coord(from_coord, transform_name)
        to_canon = self._apply_transform_to_coord(to_coord, transform_name)
        # Reconstruct with proper piece type handling
        from game3d.pieces.enums import PieceType
        piece_type_value = (move.data >> 20) & 0x3F
        piece_type = PieceType(piece_type_value)
        captured_type_value = (move.data >> 26) & 0x3F
        captured_type = PieceType(captured_type_value) if is_capture and captured_type_value > 0 else None
        return CompactMove(from_canon, to_canon, piece_type, is_capture, captured_type, is_promotion)

    def _transform_move_back(self, move: CompactMove, transform_name: str) -> CompactMove:
        """Transform move FROM canonical back to original using string name."""
        if transform_name == "identity":
            return move
        inverse_name = self._get_inverse_transform_name(transform_name)
        return self._transform_move(move, inverse_name)

    def _get_inverse_transform_name(self, transform_name: str) -> str:
        """Get inverse transformation name from string."""
        for rot_type in RotationType:
            if rot_type.value == transform_name:
                inverse_type = self._inverse_transform_map.get(rot_type, rot_type)
                return inverse_type.value
        return "identity"

    def _apply_transform_to_coord(self, coord: Tuple[int, int, int], transform_name: str) -> Tuple[int, int, int]:
        """Apply 3D rotation to coordinate using symmetry manager's matrices."""
        R = None
        for rot_type in RotationType:
            if rot_type.value == transform_name:
                R = self._symmetry_manager.rotation_matrices[rot_type]
                break
        if R is None:
            return coord

        x, y, z = coord
        center = 4  # (9-1)//2
        centered = [x - center, y - center, z - center]
        rotated = [
            R[0][0]*centered[0] + R[0][1]*centered[1] + R[0][2]*centered[2],
            R[1][0]*centered[0] + R[1][1]*centered[1] + R[1][2]*centered[2],
            R[2][0]*centered[0] + R[2][1]*centered[1] + R[2][2]*centered[2],
        ]
        new_coord = (
            int(round(rotated[0])) + center,
            int(round(rotated[1])) + center,
            int(round(rotated[2])) + center,
        )
        # Clamp to board
        return (
            max(0, min(8, new_coord[0])),
            max(0, min(8, new_coord[1])),
            max(0, min(8, new_coord[2]))
        )

    def clear_symmetry_cache(self) -> None:
        self._canonical_cache.clear()
        self._symmetry_stats = SymmetryStats()

    def get_detailed_stats(self) -> Dict[str, Any]:
        base = self.get_stats()
        sym = self.get_symmetry_stats()
        return {**base, **sym}
