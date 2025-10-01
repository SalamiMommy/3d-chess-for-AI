from __future__ import annotations
"""Optimized incremental legal-move cache for 9×9×9 3D chess — symmetry-aware and 5600X-tuned."""
# game3d/cache/movecache.py

import os
import random
import time
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Set, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pickle
import gzip
import threading

if TYPE_CHECKING:
    from game3d.cache.manager import CacheManager
    from game3d.board.board import Board

from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move
from game3d.movement.registry import get_dispatcher
from game3d.attacks.check import king_in_check
from game3d.pieces.piece import Piece
from game3d.board.symmetry import SymmetryManager
from game3d.cache.symmetry_tt import SymmetryAwareTranspositionTable
from game3d.cache.transposition import CompactMove, TTEntry, TranspositionTable
from game3d.common.common import in_bounds


# ==============================================================================
# COMPACT ZOBRIST HASHING
# ==============================================================================

import random
from game3d.pieces.enums import Color, PieceType

class ZobristHashing:
    """Thread-safe, memory-efficient Zobrist hasher with full 3D game state support."""
    __slots__ = (
        "piece_keys",
        "side_to_move_key",
        "castling_keys",      # Even in 3D, some variants have castling-like moves
        "en_passant_keys",    # For pawn-like pieces
        "ply_parity_key"      # Helps distinguish repetition at odd/even plies
    )

    # Singleton pattern for global key table
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ZobristHashing, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if ZobristHashing._initialized:
            return
        ZobristHashing._initialized = True

        # Piece keys: [color][piece_type][x][y][z]
        self.piece_keys: Dict[Tuple[Color, PieceType, Tuple[int, int, int]], int] = {}
        for color in (Color.WHITE, Color.BLACK):
            for ptype in PieceType:
                for x in range(9):
                    for y in range(9):
                        for z in range(9):
                            self.piece_keys[(color, ptype, (x, y, z))] = random.getrandbits(64)

        # Game state keys
        self.side_to_move_key = random.getrandbits(64)
        self.castling_keys = [random.getrandbits(64) for _ in range(16)]  # 4 bits for castling rights
        self.en_passant_keys = [random.getrandbits(64) for _ in range(9)]  # 9 possible files
        self.ply_parity_key = random.getrandbits(64)

    def compute_hash(self, board, current_color: Color, ply: int = 0,
                    castling: int = 0, en_passant_file: Optional[int] = None) -> int:
        """Compute full Zobrist hash including game state."""
        h = 0

        # Board pieces
        for coord, piece in board.list_occupied():
            h ^= self.piece_keys[(piece.color, piece.ptype, coord)]

        # Side to move
        if current_color == Color.WHITE:
            h ^= self.side_to_move_key

        # Castling rights
        if castling:
            for i in range(16):
                if castling & (1 << i):
                    h ^= self.castling_keys[i]

        # En passant
        if en_passant_file is not None and 0 <= en_passant_file < 9:
            h ^= self.en_passant_keys[en_passant_file]

        # Ply parity (helps with 3-fold repetition)
        if ply & 1:
            h ^= self.ply_parity_key

        return h

    def update_hash_move(self, hash_value: int, move, piece, captured_piece,
                        old_castling: int = 0, new_castling: int = 0,
                        old_ep: Optional[int] = None, new_ep: Optional[int] = None,
                        old_ply: int = 0, new_ply: int = 1) -> int:
        """Fully incremental hash update."""
        h = hash_value

        # Remove piece from source
        h ^= self.piece_keys[(piece.color, piece.ptype, move.from_coord)]
        # Add piece to target
        h ^= self.piece_keys[(piece.color, piece.ptype, move.to_coord)]
        # Remove captured piece
        if captured_piece:
            h ^= self.piece_keys[(captured_piece.color, captured_piece.ptype, move.to_coord)]

        # Toggle side to move
        h ^= self.side_to_move_key

        # Update castling
        diff_castling = old_castling ^ new_castling
        for i in range(16):
            if diff_castling & (1 << i):
                h ^= self.castling_keys[i]

        # Update en passant
        if old_ep is not None and 0 <= old_ep < 9:
            h ^= self.en_passant_keys[old_ep]
        if new_ep is not None and 0 <= new_ep < 9:
            h ^= self.en_passant_keys[new_ep]

        # Update ply parity
        if (old_ply & 1) != (new_ply & 1):
            h ^= self.ply_parity_key

        return h
# ==============================================================================
# PARALLEL MOVE GENERATOR — TUNED FOR 5600X
# ==============================================================================
class ParallelMoveGenerator:
    __slots__ = ("max_workers",)

    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(8, os.cpu_count() or 6)

    def generate_moves_parallel(self, board: "Board", color: Color,
                               piece_positions: List[Tuple[int, int, int]],
                               cache_manager: "CacheManager") -> List[Move]:
        if len(piece_positions) < 6:
            return self._generate_moves_sequential(board, color, piece_positions, cache_manager)

        chunk_size = max(1, len(piece_positions) // self.max_workers)
        chunks = [piece_positions[i:i + chunk_size]
                  for i in range(0, len(piece_positions), chunk_size)]

        all_moves = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._generate_chunk_moves, board, color, chunk, cache_manager)
                for chunk in chunks
            ]
            for future in futures:
                all_moves.extend(future.result())
        return all_moves

    def _generate_moves_sequential(self, board: "Board", color: Color,
                                positions: List[Tuple[int, int, int]],
                                cache_manager: "CacheManager") -> List[Move]:
        moves = []
        for pos in positions:
            piece = cache_manager.piece_cache.get(pos)
            if piece and piece.color == color:
                dispatcher = get_dispatcher(piece.ptype)
                if dispatcher:
                    from game3d.game.gamestate import GameState
                    tmp_state = GameState.__new__(GameState)
                    tmp_state.board = board
                    tmp_state.color = color
                    tmp_state.cache = cache_manager
                    pseudo_moves = dispatcher(tmp_state, *pos)  # ✅ FIXED: was `piece_moves`
                    moves.extend([m for m in pseudo_moves if m.from_coord == pos])
        return moves

    _generate_chunk_moves = _generate_moves_sequential


# ==============================================================================
# OPTIMIZED MOVE CACHE — FULLY TUNED
# ==============================================================================
class OptimizedMoveCache:
    __slots__ = (
        "_current", "_cache", "_legal_per_piece", "_legal_by_color",
        "_king_pos", "_priest_count", "_has_priest", "_zobrist",
        "_transposition_table", "_parallel_generator",
        "_zobrist_hash", "_age_counter",
        "_simple_move_cache", "symmetry_manager", "symmetry_tt",
        "_save_interval", "_save_dir", "_save_counter", "_save_threshold",
        "_last_save_time", "_save_thread"
    )

    def __init__(
        self,
        board: "Board",
        current: Color,
        cache_manager: "CacheManager",
    ) -> None:
        self._current = current
        self._cache = cache_manager
        self._has_priest = {Color.WHITE: False, Color.BLACK: False}
        self._legal_per_piece: Dict[Tuple[int, int, int], List[Move]] = {}
        self._legal_by_color: Dict[Color, List[Move]] = {
            Color.WHITE: [], Color.BLACK: []
        }
        self._king_pos: Dict[Color, Optional[Tuple[int, int, int]]] = {
            Color.WHITE: None, Color.BLACK: None
        }
        self._priest_count: Dict[Color, int] = {
            Color.WHITE: 0, Color.BLACK: 0
        }

        self._zobrist = ZobristHashing()
        main_mb = cache_manager.main_tt_size_mb
        sym_mb = cache_manager.sym_tt_size_mb

        # 1. create the manager first
        self.symmetry_manager = SymmetryManager()

        # 2. now we can safely pass it to the symmetry TT
        self.symmetry_tt = SymmetryAwareTranspositionTable(
            self.symmetry_manager, size_mb=sym_mb
        )

        self._transposition_table = TranspositionTable(size_mb=main_mb)
        self._parallel_generator = ParallelMoveGenerator()
        self._zobrist_hash = self._zobrist.compute_hash(board, current)
        self._age_counter = 0
        self._simple_move_cache: Dict[
            Tuple[Tuple[int, int, int], Color], List[Move]
        ] = {}

        # Symmetry systems
        self.symmetry_manager = SymmetryManager()
        # 🔥 2 GB symmetry TT (separate to avoid polluting main TT)
        self.symmetry_tt = SymmetryAwareTranspositionTable(self.symmetry_manager, size_mb=2048)

        self._full_rebuild()

        # Disk cache setup
        self._save_interval = 3600  # Seconds (1 hour)
        self._save_dir = "tt_saves"  # Directory for save files
        self._save_counter = 0
        self._save_threshold = 1000000  # Save after this many stores (adjust as needed)
        self._last_save_time = time.time()
        os.makedirs(self._save_dir, exist_ok=True)

        # Load from disk if files exist
        self._load_from_disk()

        # Start background save thread
        self._save_thread = threading.Thread(target=self._periodic_save, daemon=True)
        self._save_thread.start()

    @property
    def _board(self) -> "Board":
        return self._cache.board

    @property
    def main_tt_size_mb(self) -> int:
        return self._main_tt_size_mb

    @property
    def sym_tt_size_mb(self) -> int:
        return self._sym_tt_size_mb

    # --------------------------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------------------------
    def legal_moves(self, color: Color, parallel: bool = True, max_workers: int = 8) -> List[Move]:
        return self._legal_by_color[color]

    def get_cached_evaluation(self, hash_value: int) -> Optional[Tuple[int, int, Optional[CompactMove]]]:
        # Try main TT first
        result = self._transposition_table.probe(hash_value)
        if result:
            return result.score, result.depth, result.best_move
        # Fall back to symmetry TT
        sym_result = self.symmetry_tt.probe_with_symmetry(hash_value, self._board)
        if sym_result:
            return sym_result.score, sym_result.depth, sym_result.best_move
        return None

    def store_evaluation(self, hash_value: int, depth: int, score: int,
                        node_type: int, best_move: Optional[CompactMove] = None) -> None:
        self._transposition_table.store(hash_value, depth, score, node_type, best_move, self._age_counter)
        if depth >= 3:  # Only store deep positions in symmetry TT
            self.symmetry_tt.store_with_symmetry(hash_value, self._board, depth, score, node_type, best_move)

        # Incremental save logic
        self._save_counter += 1
        if self._save_counter >= self._save_threshold:
            self._save_to_disk()

    def apply_move(self, mv: Move, color: Color) -> None:
        from_coord = mv.from_coord
        to_coord = mv.to_coord
        piece = self._board.piece_at(from_coord)
        captured_piece = self._board.piece_at(to_coord) if getattr(mv, 'is_capture', False) else None

        self._zobrist_hash = self._zobrist.update_hash_move(self._zobrist_hash, mv, piece, captured_piece)
        self._board.apply_move(mv)
        self._current = color.opposite()
        self._age_counter += 1

        self._optimized_incremental_update(mv, color, from_coord, to_coord, piece, captured_piece)

    def undo_move(self, mv: Move, color: Color) -> None:
        piece = self._board.piece_at(mv.to_coord)
        captured_piece = None
        if getattr(mv, "is_capture", False):
            captured_type = getattr(mv, "captured_ptype", None)
            if captured_type is not None:
                captured_piece = Piece(color.opposite(), captured_type)

        if piece:
            self._zobrist_hash = self._zobrist.update_hash_move(self._zobrist_hash, mv, piece, captured_piece)

        self._undo_move_optimized(mv, color)
        self._current = color
        self._age_counter += 1
        self._optimized_incremental_undo(mv, color)

    def _undo_move_optimized(self, mv: Move, color: Color) -> None:
        if getattr(mv, "is_capture", False):
            captured_type = getattr(mv, "captured_ptype", None)
            if captured_type is not None:
                self._board.set_piece(mv.to_coord, Piece(color.opposite(), captured_type))
        piece = self._board.piece_at(mv.to_coord)
        if piece:
            self._board.set_piece(mv.from_coord, piece)
            self._board.set_piece(mv.to_coord, None)
        if getattr(mv, "is_promotion", False) and piece:
            self._board.set_piece(mv.from_coord, Piece(piece.color, PieceType.PAWN))

    def clear(self) -> None:
        self._legal_per_piece.clear()
        self._legal_by_color = {Color.WHITE: [], Color.BLACK: []}
        self._simple_move_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        base_stats = {
            'zobrist_hash': self._zobrist_hash,
            'tt_hits': self._transposition_table.hits,
            'tt_misses': self._transposition_table.misses,
            'tt_collisions': self._transposition_table.collisions,
            'simple_move_cache_size': len(self._simple_move_cache),
            'age_counter': self._age_counter,
            'tt_size_mb': 6144,
            'symmetry_tt_size_mb': 2048
        }
        if hasattr(self.symmetry_tt, 'get_symmetry_stats'):
            base_stats['symmetry_stats'] = self.symmetry_tt.get_symmetry_stats()
        return base_stats

    def _save_to_disk(self) -> None:
        timestamp = int(time.time())
        for tt, prefix in [(self._transposition_table, "main"), (self.symmetry_tt, "sym")]:
            path = os.path.join(self._save_dir, f"{prefix}_tt_{timestamp}.pkl.gz")
            with gzip.open(path, 'wb') as f:
                for idx in range(tt.size):
                    entry = tt.table[idx]
                    if entry is not None:
                        pickle.dump((idx, entry), f, protocol=4)
        print(f"Saved TTs to {self._save_dir} with timestamp {timestamp}")
        self._last_save_time = time.time()
        self._save_counter = 0  # Reset counter

    def _load_from_disk(self) -> None:
        for prefix, tt in [("main", self._transposition_table), ("sym", self.symmetry_tt)]:
            files = [f for f in os.listdir(self._save_dir) if f.startswith(f"{prefix}_tt_") and f.endswith(".pkl.gz")]
            if not files:
                continue
            latest = max(files, key=lambda f: int(f.split('_')[2].split('.')[0]))
            path = os.path.join(self._save_dir, latest)
            with gzip.open(path, 'rb') as f:
                while True:
                    try:
                        idx, entry = pickle.load(f)
                        tt.table[idx] = entry
                    except EOFError:
                        break
            print(f"Loaded {prefix} TT from {path}")

    def _periodic_save(self) -> None:
        while True:
            time.sleep(60)  # Check every minute
            if time.time() - self._last_save_time >= self._save_interval:
                self._save_to_disk()

    # --------------------------------------------------------------------------
    # INTERNAL OPTIMIZATIONS
    # --------------------------------------------------------------------------
    def _optimized_incremental_update(self, mv: Move, color: Color, from_coord: Tuple[int, int, int],
                                    to_coord: Tuple[int, int, int], piece: Piece,
                                    captured_piece: Optional[Piece]) -> None:
        self._refresh_counts()
        affected_squares = {from_coord, to_coord}
        for col in (Color.WHITE, Color.BLACK):
            king_pos = self._find_king(col)
            if king_pos:
                affected_squares.add(king_pos)
        self._batch_update_pieces(affected_squares, color.opposite())

    def _batch_update_pieces(self, affected_squares: Set[Tuple[int, int, int]], color: Color) -> None:
        for coord in affected_squares:
            self._legal_per_piece.pop(coord, None)

        pieces_to_update = [
            coord for coord in affected_squares
            if (p := self._board.piece_at(coord)) and p.color == color
        ]

        if len(pieces_to_update) > 6:
            moves = self._parallel_generator.generate_moves_parallel(
                self._board, color, pieces_to_update, self._cache
            )
            for move in moves:
                self._legal_per_piece.setdefault(move.from_coord, []).append(move)
        else:
            for coord in pieces_to_update:
                self._legal_per_piece[coord] = self._generate_piece_moves(coord)

        self._rebuild_color_lists()

    def _generate_piece_moves(self, coord: Tuple[int, int, int]) -> List[Move]:
        piece = self._board.piece_at(coord)
        if not piece:
            return []
        dispatcher = get_dispatcher(piece.ptype)
        if not dispatcher:
            return []

        # Use symmetry cache key if available
        if hasattr(self.symmetry_manager, 'create_movement_symmetry_key'):
            sym_key = self.symmetry_manager.create_movement_symmetry_key(piece.ptype, coord, piece.color)
            cached_moves = self._simple_move_cache.get(sym_key)
            if cached_moves is not None:
                return cached_moves

        from game3d.game.gamestate import GameState
        tmp_state = GameState.__new__(GameState)
        tmp_state.board = self._board
        tmp_state.color = piece.color
        tmp_state.cache = self._cache

        pseudo_moves = dispatcher(tmp_state, *coord)  # ✅ FIXED

        if self._has_priest[piece.color]:
            if hasattr(self.symmetry_manager, 'create_movement_symmetry_key'):
                sym_key = self.symmetry_manager.create_movement_symmetry_key(piece.ptype, coord, piece.color)
                self._simple_move_cache[sym_key] = pseudo_moves
                if len(self._simple_move_cache) > 3000:
                    self._simple_move_cache.clear()
            return pseudo_moves

        legal_moves = self._filter_king_safe_moves(pseudo_moves, piece)
        if hasattr(self.symmetry_manager, 'create_movement_symmetry_key'):
            sym_key = self.symmetry_manager.create_movement_symmetry_key(piece.ptype, coord, piece.color)
            self._simple_move_cache[sym_key] = legal_moves
        return legal_moves

    def _filter_king_safe_moves(self, moves: List[Move], piece: Piece) -> List[Move]:
        legal_moves = []
        for mv in moves:
            moving = self._board.piece_at(mv.from_coord)
            victim = self._board.piece_at(mv.to_coord)
            self._board.set_piece(mv.from_coord, None)
            self._board.set_piece(mv.to_coord, moving)
            try:
                if not king_in_check(self._board, piece.color, piece.color.opposite(), self._cache):
                    legal_moves.append(mv)
            finally:
                self._board.set_piece(mv.from_coord, moving)
                self._board.set_piece(mv.to_coord, victim)
        return legal_moves

    def _rebuild_color_lists(self) -> None:
        white_moves = []
        black_moves = []
        for coord, moves in self._legal_per_piece.items():
            if not moves:
                continue
            piece = self._board.piece_at(coord)
            if not piece:
                continue
            (white_moves if piece.color == Color.WHITE else black_moves).extend(moves)
        self._legal_by_color[Color.WHITE] = white_moves
        self._legal_by_color[Color.BLACK] = black_moves

    def _full_rebuild(self) -> None:
        self._refresh_counts()
        self._legal_per_piece.clear()
        pieces_to_generate = [
            coord for coord, piece in self._board.list_occupied() if piece.color == self._current
        ]
        if len(pieces_to_generate) > 8:
            all_moves = self._parallel_generator.generate_moves_parallel(
                self._board, self._current, pieces_to_generate, self._cache
            )
            for move in all_moves:
                self._legal_per_piece.setdefault(move.from_coord, []).append(move)
        else:
            for coord in pieces_to_generate:
                self._legal_per_piece[coord] = self._generate_piece_moves(coord)
        self._rebuild_color_lists()

    def _find_king(self, color: Color) -> Optional[Tuple[int, int, int]]:
        if self._king_pos[color]:
            king = self._board.piece_at(self._king_pos[color])
            if king and king.color == color and king.ptype == PieceType.KING:
                return self._king_pos[color]
        for coord, piece in self._board.list_occupied():
            if piece.color == color and piece.ptype == PieceType.KING:
                self._king_pos[color] = coord
                return coord
        self._king_pos[color] = None
        return None

    def _refresh_counts(self) -> None:
        for color in Color:
            self._priest_count[color] = 0
            self._king_pos[color] = None
            self._has_priest[color] = False
        for _, piece in self._board.list_occupied():
            if piece.ptype == PieceType.PRIEST:
                self._priest_count[piece.color] += 1
            elif piece.ptype == PieceType.KING:
                self._king_pos[piece.color] = _
        for color in Color:
            self._has_priest[color] = self._priest_count[color] > 0

    def _optimized_incremental_undo(self, mv: Move, color: Color) -> None:
        affected = {mv.from_coord, mv.to_coord}
        for col in (Color.WHITE, Color.BLACK):
            k = self._find_king(col)
            if k:
                affected.add(k)
        self._batch_update_pieces(affected, color)


# ==============================================================================
# FACTORY
# ==============================================================================
# game3d/cache/movecache.py
def create_optimized_move_cache(
    board: "Board",
    current: Color,
    cache_manager: "CacheManager",
) -> OptimizedMoveCache:
    return OptimizedMoveCache(board, current, cache_manager)
