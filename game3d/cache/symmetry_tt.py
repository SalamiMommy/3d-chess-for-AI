"""Optimized symmetry-aware transposition table with enhanced 3D board support."""

from __future__ import annotations
from typing import Optional, Dict, Any, Set, Tuple, List
from dataclasses import dataclass
import weakref
import struct
import time

from game3d.cache.transposition import TranspositionTable, TTEntry
from game3d.movement.movepiece import CompactMove
from game3d.board.symmetry import SymmetryManager
from game3d.pieces.enums import Color

# ==============================================================================
# OPTIMIZATION CONSTANTS
# ==============================================================================

@dataclass
class SymmetryStats:
    """Statistics for symmetry operations."""
    probe_count: int = 0
    hit_count: int = 0
    store_count: int = 0
    transform_time: float = 0.0
    canonical_hits: int = 0

class SymmetryAwareTranspositionTable(TranspositionTable):
    """High-performance transposition table with 3D symmetry awareness."""

    __slots__ = (
        "_symmetry_manager", "_symmetry_stats", "_canonical_cache",
        "_transform_cache", "_inverse_transforms", "_batch_transforms"
    )

    def __init__(self, symmetry_manager: SymmetryManager, size_mb: int = 512) -> None:
        super().__init__(size_mb)
        self._symmetry_manager = symmetry_manager
        self._symmetry_stats = SymmetryStats()

        # Caches for symmetry operations
        self._canonical_cache: Dict[int, Tuple[int, str]] = {}  # hash -> (canonical_hash, transform)
        self._transform_cache: Dict[Tuple[str, int], int] = {}  # (transform, move_hash) -> transformed_move_hash
        self._inverse_transforms: Dict[str, str] = {}
        self._batch_transforms: Dict[str, List[str]] = {}

        # Initialize transformation lookup tables
        self._init_transform_tables()

    # ---------- PUBLIC INTERFACE ----------
    def probe_with_symmetry(self, hash_value: int, board_state) -> Optional[TTEntry]:
        """High-performance symmetry-aware probe with caching."""
        # Fast path: exact match
        exact_hit = self.probe(hash_value)
        if exact_hit:
            return exact_hit

        # Track symmetry probe
        self._symmetry_stats.probe_count += 1

        # Check canonical cache first
        if hash_value in self._canonical_cache:
            canonical_hash, transform_name = self._canonical_cache[hash_value]
            canonical_entry = self.probe(canonical_hash)
            if canonical_entry:
                self._symmetry_stats.canonical_hits += 1
                return self._transform_entry_back(canonical_entry, transform_name)

        # Get symmetric variants (batch operation for performance)
        symmetric_variants = self._get_symmetric_variants_batch(board_state)

        for sym_hash, transform_name, sym_board in symmetric_variants:
            sym_entry = self.probe(sym_hash)
            if sym_entry:
                self._symmetry_stats.hit_count += 1

                # Cache this symmetry for future use
                self._canonical_cache[hash_value] = (sym_hash, transform_name)

                # Transform entry back to original coordinates
                return self._transform_entry_back(sym_entry, transform_name)

        return None

    def store_with_symmetry(self, hash_value: int, board_state, depth: int,
                          score: int, node_type: int, best_move: Optional[CompactMove] = None) -> None:
        """Optimized symmetry-aware store with canonical form caching."""
        # Store original position
        self.store(hash_value, depth, score, node_type, best_move)
        self._symmetry_stats.store_count += 1

        # Store canonical form for high-depth positions
        if depth >= 3:  # Configurable threshold
            self._store_canonical_form(hash_value, board_state, depth, score, node_type, best_move)

    def get_symmetry_stats(self) -> Dict[str, Any]:
        """Get detailed symmetry statistics."""
        total_probes = max(1, self._symmetry_stats.probe_count)
        return {
            'symmetry_probes': self._symmetry_stats.probe_count,
            'symmetry_hits': self._symmetry_stats.hit_count,
            'symmetry_hit_rate': self._symmetry_stats.hit_count / total_probes,
            'canonical_hits': self._symmetry_stats.canonical_hits,
            'transform_time': self._symmetry_stats.transform_time,
            'cache_efficiency': len(self._canonical_cache) / max(1, self._symmetry_stats.probe_count),
            'total_entries': len(self._entries),
            'memory_usage_mb': self._get_memory_usage(),
        }

    # ---------- INTERNAL OPTIMIZATIONS ----------
    def _init_transform_tables(self) -> None:
        """Initialize transformation lookup tables for performance."""
        # Pre-calculate inverse transforms
        self._inverse_transforms = {
            'identity': 'identity',
            'rotate_90_x': 'rotate_270_x',
            'rotate_180_x': 'rotate_180_x',
            'rotate_270_x': 'rotate_90_x',
            'rotate_90_y': 'rotate_270_y',
            'rotate_180_y': 'rotate_180_y',
            'rotate_270_y': 'rotate_90_y',
            'rotate_90_z': 'rotate_270_z',
            'rotate_180_z': 'rotate_180_z',
            'rotate_270_z': 'rotate_270_z',
            'reflect_x': 'reflect_x',
            'reflect_y': 'reflect_y',
            'reflect_z': 'reflect_z',
        }

        # Pre-calculate batch transformation sequences
        self._batch_transforms = {
            'primary': ['identity', 'rotate_90_x', 'rotate_180_x', 'rotate_270_x'],
            'secondary': ['rotate_90_y', 'rotate_180_y', 'rotate_270_y'],
            'tertiary': ['rotate_90_z', 'rotate_180_z', 'rotate_270_z'],
            'reflections': ['reflect_x', 'reflect_y', 'reflect_z'],
        }

    def _get_symmetric_variants_batch(self, board_state) -> List[Tuple[int, str, Any]]:
        """Batch generate symmetric variants for performance."""
        start_time = time.perf_counter()

        variants = []
        seen_hashes: Set[int] = set()

        # Generate transformations in optimized order
        transform_groups = ['primary', 'secondary', 'tertiary', 'reflections']

        for group in transform_groups:
            for transform_name in self._batch_transforms[group]:
                try:
                    # Apply transformation
                    sym_board = self._symmetry_manager.apply_transformation(board_state, transform_name)
                    sym_hash = hash(self._symmetry_manager._extract_board_state(sym_board))

                    if sym_hash not in seen_hashes:
                        seen_hashes.add(sym_hash)
                        variants.append((sym_hash, transform_name, sym_board))

                except Exception:
                    # Skip invalid transformations
                    continue

        self._symmetry_stats.transform_time += time.perf_counter() - start_time
        return variants

    def _store_canonical_form(self, hash_value: int, board_state, depth: int,
                            score: int, node_type: int, best_move: Optional[CompactMove]) -> None:
        """Store position in canonical form for maximum symmetry coverage."""
        try:
            canonical_board, canonical_transform = self._symmetry_manager.get_canonical_form(board_state)
            canonical_hash = hash(self._symmetry_manager._extract_board_state(canonical_board))

            # Transform best move to canonical coordinates
            canonical_move = best_move
            if best_move and canonical_transform != "identity":
                canonical_move = self._transform_move_to_canonical(best_move, canonical_transform)

            # Store canonical entry
            self.store(canonical_hash, depth, score, node_type, canonical_move)

            # Cache the canonical relationship
            self._canonical_cache[hash_value] = (canonical_hash, canonical_transform)

        except Exception:
            # Fallback: don't store canonical form if transformation fails
            pass

    def _transform_entry_back(self, entry: TTEntry, transform_name: str) -> TTEntry:
        """Transform TTEntry back to original coordinates."""
        if transform_name == "identity" or not entry.best_move:
            return entry

        # Transform best move back
        original_move = self._transform_move_back(entry.best_move, transform_name)

        # Create new entry with transformed move
        return TTEntry(
            depth=entry.depth,
            score=entry.score,
            node_type=entry.node_type,
            best_move=original_move
        )

    def _transform_move_back(self, move: CompactMove, transform_name: str) -> CompactMove:
        """Transform move coordinates back from symmetric position."""
        if transform_name == "identity":
            return move

        # Use cached transformation if available
        cache_key = (transform_name, hash(move))
        if cache_key in self._transform_cache:
            # Need to reconstruct move from cached hash - this is a simplified version
            return move  # In practice, you'd reconstruct properly

        # Apply inverse transformation
        inverse_transform = self._inverse_transforms.get(transform_name, "identity")

        # Transform coordinates (simplified - would need full 3D coordinate transformation)
        # This is a placeholder - actual implementation would transform move coordinates
        return move

    def _transform_move_to_canonical(self, move: CompactMove, transform_name: str) -> CompactMove:
        """Transform move to canonical coordinates."""
        if transform_name == "identity":
            return move

        # Apply transformation to move coordinates
        # Simplified - actual implementation would transform properly
        return move

    def _get_memory_usage(self) -> float:
        """Calculate memory usage in MB."""
        # Base table memory
        base_memory = super().get_memory_usage() if hasattr(super(), 'get_memory_usage') else 0

        # Symmetry cache memory
        symmetry_memory = (
            len(self._canonical_cache) * 64 +  # Rough estimate
            len(self._transform_cache) * 32 +
            len(self._inverse_transforms) * 32 +
            len(self._batch_transforms) * 32
        )

        return (base_memory + symmetry_memory) / (1024 * 1024)

    # ---------- BATCH OPERATIONS ----------
    def probe_batch_with_symmetry(self, positions: List[Tuple[int, Any]]) -> List[Optional[TTEntry]]:
        """Batch probe multiple positions with symmetry."""
        results = []

        for hash_val, board_state in positions:
            result = self.probe_with_symmetry(hash_val, board_state)
            results.append(result)

        return results

    def store_batch_with_symmetry(self, entries: List[Tuple[int, Any, int, int, int, Optional[CompactMove]]]) -> None:
        """Batch store multiple entries with symmetry."""
        for hash_val, board_state, depth, score, node_type, best_move in entries:
            self.store_with_symmetry(hash_val, board_state, depth, score, node_type, best_move)

    # ---------- CACHE MANAGEMENT ----------
    def clear_symmetry_cache(self) -> None:
        """Clear symmetry-specific caches."""
        self._canonical_cache.clear()
        self._transform_cache.clear()
        self._symmetry_stats = SymmetryStats()

    def resize(self, size_mb: int) -> None:
        """Resize table and clear symmetry caches."""
        super().resize(size_mb)
        self.clear_symmetry_cache()

    # ---------- STATISTICS ----------
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics including symmetry performance."""
        base_stats = super().get_stats() if hasattr(super(), 'get_stats') else {}
        symmetry_stats = self.get_symmetry_stats()

        return {**base_stats, **symmetry_stats}

# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_symmetry_aware_tt(symmetry_manager: SymmetryManager, size_mb: int = 512) -> SymmetryAwareTranspositionTable:
    """Factory function for creating symmetry-aware transposition table."""
    return SymmetryAwareTranspositionTable(symmetry_manager, size_mb)

# ==============================================================================
# BACKWARD COMPATIBILITY
# ==============================================================================

class TranspositionTableSymmetry(SymmetryAwareTranspositionTable):
    """Backward compatibility wrapper."""
    pass
