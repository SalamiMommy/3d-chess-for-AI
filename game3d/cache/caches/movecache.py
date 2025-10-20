# movecache.py
from __future__ import annotations
import random
import time
from typing import Dict, List, Optional, Tuple, Iterable, Set, Any, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np

if TYPE_CHECKING:
    from game3d.cache.manager import get_cache_manager
    from game3d.board.board import Board

from game3d.common.enums import Color, PieceType
from game3d.movement.movepiece import Move
from game3d.pieces.piece import Piece
from game3d.common.common import in_bounds, infer_piece_from_cache, fallback_mode # Updated imports
from game3d.game.zobrist import compute_zobrist, ZobristHash
from game3d.board.symmetry import SymmetryManager
from game3d.cache.caches.symmetry_tt import SymmetryAwareTranspositionTable
from game3d.cache.caches.transposition import TranspositionTable
from game3d.common.common import get_player_pieces, filter_none_moves

# ------------------------------------------------------------------
# Compact move representation for TT entries
# ------------------------------------------------------------------
class CompactMove:
    """Simple move representation for TT entries."""
    __slots__ = ('from_coord', 'to_coord', 'piece_type', 'is_capture', 'captured_type', 'is_promotion')

    def __init__(self, from_coord, to_coord, piece_type, is_capture, captured_type, is_promotion):
        self.from_coord = from_coord
        self.to_coord = to_coord
        self.piece_type = piece_type
        self.is_capture = is_capture
        self.captured_type = captured_type
        self.is_promotion = is_promotion

# ------------------------------------------------------------------
# Bit-packing helpers
# ------------------------------------------------------------------
_SQUARE_MASK = 0x3FF          # 10 bit
_CAPTURE_BIT = 1 << 20
_PROMO_BIT   = 1 << 21

def _coord_to_idx(x: int, y: int, z: int) -> int:
    if not (0 <= x < 9 and 0 <= y < 9 and 0 <= z < 9):
        raise ValueError(f"Invalid coordinate: ({x}, {y}, {z})")
    return x * 81 + y * 9 + z

def move_to_key(m: Move) -> int:
    fx, fy, fz = m.from_coord
    tx, ty, tz = m.to_coord
    if not all(0 <= c < 9 for c in (fx, fy, fz, tx, ty, tz)):
        raise ValueError(f"Move contains out-of-bounds coordinate: {m}")
    f_idx = _coord_to_idx(fx, fy, fz)
    t_idx = _coord_to_idx(tx, ty, tz)
    key = f_idx | (t_idx << 10)
    if getattr(m, 'is_capture', False):
        key |= _CAPTURE_BIT
    if getattr(m, 'is_promotion', False):
        key |= _PROMO_BIT
    return key

# ==============================================================================
# OPTIMIZED MOVE CACHE — FULLY TUNED  (NO KING / PRIEST BOOK-KEEPING)
# ==============================================================================
class OptimizedMoveCache:
    __slots__ = (
        "_current", "_cache", "_legal_per_piece", "_legal_by_color",
        "_zobrist", "_transposition_table",
        "_zobrist_hash", "_age_counter",
        "_simple_move_cache", "symmetry_manager", "symmetry_tt",
        "_main_tt_size_mb", "_sym_tt_size_mb",
        "_needs_rebuild", "_attacked_squares", "_attacked_squares_valid",
        "_cache_manager", "_dirty_flags", "_invalid_squares", "_invalid_attacks",
        "_gen", "_frozen_bitmap", "_debuffed_bitmap"  # NEW: Precomputed bitmaps
    )

    def __init__(
        self,
        board: "Board",
        current: Color,
        cache_manager: "CacheManager",
    ) -> None:
        self._current = current
        self._cache = cache_manager
        self._cache_manager = cache_manager
        self._legal_per_piece: Dict[Tuple[int, int, int], List[Move]] = {}
        self._legal_by_color: Dict[Color, List[Move]] = {
            Color.WHITE: [], Color.BLACK: []
        }

        self._zobrist = ZobristHash()
        self._zobrist_hash = compute_zobrist(board, current)
        main_mb = cache_manager.config.main_tt_size_mb
        sym_mb = cache_manager.config.sym_tt_size_mb

        # 1. create the manager first
        self.symmetry_manager = SymmetryManager()

        # 2. now we can safely pass it to the symmetry TT
        self.symmetry_tt = SymmetryAwareTranspositionTable(
            self.symmetry_manager, size_mb=sym_mb
        )

        self._transposition_table = TranspositionTable(size_mb=main_mb)
        self._main_tt_size_mb = cache_manager.config.main_tt_size_mb
        self._sym_tt_size_mb = cache_manager.config.sym_tt_size_mb

        self._zobrist_hash = compute_zobrist(board, current)
        self._age_counter = 0
        self._simple_move_cache: Dict[int, List[Move]] = {}

        self._invalid_squares: set[Tuple[int,int,int]] = set()
        self._invalid_attacks: set[Color] = set()
        self._needs_rebuild = False
        self._attacked_squares: Dict[Color, Set[Tuple[int, int, int]]] = {
            Color.WHITE: set(),
            Color.BLACK: set()
        }
        self._attacked_squares_valid: Dict[Color, bool] = {
            Color.WHITE: False,
            Color.BLACK: False
        }
        self._dirty_flags = {
            'targets': False,
            'attacks': False,
        }
        self._gen = -1  # NEW: Initialize generation
        self._frozen_bitmap: np.ndarray = np.zeros((9,9,9), dtype=bool)  # NEW
        self._debuffed_bitmap: np.ndarray = np.zeros((9,9,9), dtype=bool)  # NEW

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def _board(self) -> "Board":
        return self._cache.board

    @property
    def main_tt_size_mb(self) -> int:
        return self._main_tt_size_mb

    @property
    def sym_tt_size_mb(self) -> int:
        return self._sym_tt_size_mb

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def legal_moves(self, color: Color, *, parallel: bool = True, max_workers: int = 8) -> list[Move]:
        """Public API – returns cached list unless something is dirty."""
        self._lazy_revalidate()
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

    # ------------------------------------------------------------------
    # Move application / undo
    # ------------------------------------------------------------------
    def apply_move(self, mv: Move, color: Color) -> None:
        """
        Incrementally apply a move to the cache and regenerate only the
        squares that actually changed.  The board tensor and occupancy
        cache are updated **before** any move generation is triggered.
        """
        piece = self._cache_manager.occupancy.get(mv.from_coord)
        if piece is None:
            self._needs_rebuild = True
            return

        captured_piece = (
            self._cache_manager.occupancy.get(mv.to_coord) if mv.is_capture else None
        )

        # Update Zobrist hash **before** we mutate the position
        self._zobrist_hash = self._zobrist.update_hash_move(
            self._zobrist_hash, mv, piece, captured_piece
        )

        # Mutate the **authoritative** board tensor first
        self._cache_manager.board.apply_move(mv)

        # Update the occupancy cache to stay in sync
        self._cache_manager.set_piece(mv.from_coord, None)
        promoted = (
            Piece(color, PieceType(mv.promotion_ptype)) if getattr(mv, "is_promotion", False) else piece
        )
        self._cache_manager.set_piece(mv.to_coord, promoted)

        # Mark the changed squares (and attacked maps) as dirty
        self.invalidate_square(mv.from_coord)
        self.invalidate_square(mv.to_coord)
        self.invalidate_attacked_squares(color)
        self.invalidate_attacked_squares(color.opposite())

        # House-keeping
        self._current = color.opposite()
        self._age_counter += 1
        self._needs_rebuild = False

    def undo_move(self, mv: Move, color: Color) -> None:
        """
        Undo a move and incrementally regenerate only the squares that
        changed.  Board tensor is rolled back **before** any cache update
        or move generation.
        """
        piece_now = self._cache_manager.occupancy.get(mv.to_coord)
        captured_type = getattr(mv, "captured_ptype", None)
        captured_piece = (
            Piece(color.opposite(), PieceType(captured_type)) if captured_type else None
        )
        self._zobrist_hash = self._zobrist.update_hash_move(
            self._zobrist_hash, mv, piece_now, captured_piece
        )

        # Restore the **authoritative** board tensor first
        self._cache_manager.board.undo_move(mv)

        # Restore occupancy cache to match the new tensor
        mover_piece = (
            Piece(color, PieceType.PAWN) if getattr(mv, "is_promotion", False) else piece_now
        )
        self._cache_manager.set_piece(mv.from_coord, mover_piece)
        self._cache_manager.set_piece(mv.to_coord, captured_piece)

        # Mark squares (and attacked maps) as dirty
        self.invalidate_square(mv.from_coord)
        self.invalidate_square(mv.to_coord)
        self.invalidate_attacked_squares(color)
        self.invalidate_attacked_squares(color.opposite())

        # House-keeping
        self._current = color
        self._age_counter += 1

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
            'tt_size_mb': self.main_tt_size_mb,
            'symmetry_tt_size_mb': self._sym_tt_size_mb
        }
        if hasattr(self.symmetry_tt, 'get_symmetry_stats'):
            base_stats['symmetry_stats'] = self.symmetry_tt.get_symmetry_stats()
        return base_stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _optimized_incremental_update(self, mv: Move, color: Color, from_coord: Tuple[int, int, int],
                                    to_coord: Tuple[int, int, int], piece: Piece,
                                    captured_piece: Optional[Piece]) -> None:
        affected_squares = {from_coord, to_coord}
        # Update only affected pieces
        self._batch_update_pieces(affected_squares, color.opposite())
        self._attacked_squares_valid[Color.WHITE] = False
        self._attacked_squares_valid[Color.BLACK] = False

    def _batch_update_pieces(self, affected_squares: Set[Tuple[int,int,int]], color: Color) -> None:
        print("_batch_update_pieces: needs_rebuild =", self._needs_rebuild)
        print('[DEBUG] gen board', self._board.generation,
            'gen occ', occupancy._gen,
            'piece@from', self._cache_manager.occupancy.get(coord),
            'needs_rebuild', self._needs_rebuild)

        if self._needs_rebuild:
            return

        # Clear affected pieces from cache
        for coord in affected_squares:
            self._legal_per_piece.pop(coord, None)

        # Identify pieces that need updating
        pieces_to_update = [
            coord for coord in affected_squares
            if (p := self._cache_manager.occupancy.get(coord)) and p.color == color
        ]

        # Use generator.py for updates
        from game3d.game.gamestate import GameState
        from game3d.movement.generator import generate_legal_moves, generate_legal_moves_for_piece
        tmp_state = GameState(board=self._board, color=color, cache=self._cache)

        for coord in pieces_to_update:
            try:
                moves = generate_legal_moves_for_piece(tmp_state, coord)
                if moves:
                    self._legal_per_piece[coord] = moves
            except Exception as e:
                # If move generation fails, mark for full rebuild
                print(f"[WARNING] Move generation failed for {coord}: {e}")
                self._needs_rebuild = True
                return

        # Rebuild color lists
        self._rebuild_color_lists()

    def _generate_piece_moves(self, coord: Tuple[int, int, int]) -> List[Move]:
        from game3d.game.gamestate import GameState
        from game3d.movement.generator import generate_legal_moves, generate_legal_moves_for_piece
        piece = self._cache_manager.occupancy.get(coord)
        if not piece:
            return []

        tmp_state = GameState(board=self._board, color=piece.color, cache=self._cache)
        return generate_legal_moves_for_piece(tmp_state, coord)

    def _rebuild_color_lists(self) -> None:
        """Rebuild color-indexed move lists from per-piece cache."""
        white_moves = []
        black_moves = []

        for coord, moves in self._legal_per_piece.items():
            if not moves:
                continue

            # DEFENSIVE: Filter out None moves
            moves = filter_none_moves(moves)
            if not moves:
                continue

            piece = self._cache_manager.occupancy.get(coord)
            if not piece:
                continue

            if piece.color == Color.WHITE:
                white_moves.extend(moves)
            else:
                black_moves.extend(moves)

        self._legal_by_color[Color.WHITE] = white_moves
        self._legal_by_color[Color.BLACK] = black_moves

    def _full_rebuild(self) -> None:
        """Full rebuild of legal moves from scratch."""
        try:
            # Get occupancy reference
            occupancy = self._cache_manager.occupancy

            # Create temporary state for move generation
            from game3d.game.gamestate import GameState
            tmp_state = GameState(
                board=self._cache_manager.board,
                color=self._current,
                cache=self._cache_manager
            )

            # Check if board has pieces
            if occupancy.count == 0:
                print("[REBUILD] WARNING: Board has no pieces!")
                self._legal_by_color[Color.WHITE] = []
                self._legal_by_color[Color.BLACK] = []
                self._legal_per_piece.clear()
                return

            # Generate pseudo-legal moves
            from game3d.movement.pseudo_legal import generate_pseudo_legal_moves
            pseudo_moves = generate_pseudo_legal_moves(tmp_state)

            # DEFENSIVE: Filter out None values immediately
            pseudo_moves = filter_none_moves(pseudo_moves)

            # Filter out moves from empty squares (paranoid check)
            valid_moves = []
            for move in pseudo_moves:
                piece = occupancy.get(move.from_coord)
                if piece is None:
                    print(f"[REBUILD] WARNING: Skipping move from empty square {move.from_coord}")
                    continue
                if piece.color != self._current:
                    print(f"[REBUILD] WARNING: Skipping opponent's piece at {move.from_coord}")
                    continue
                valid_moves.append(move)

            # Clear existing per-piece cache
            self._legal_per_piece.clear()

            # Split by piece
            for move in valid_moves:
                self._legal_per_piece.setdefault(move.from_coord, []).append(move)

            # Rebuild color lists (which will also filter None)
            self._rebuild_color_lists()

        except AttributeError as e:
            # Specific handling for missing attributes like _board
            raise RuntimeError(f"Cache misconfiguration during rebuild: {e}")
        except Exception as e:
            print(f"[ERROR] Legal move generation failed during rebuild: {e}")
            import traceback
            traceback.print_exc()
            # Clear caches on failure
            self._legal_by_color[Color.WHITE] = []
            self._legal_by_color[Color.BLACK] = []
            self._legal_per_piece.clear()
            # Raise to surface the error (no silent failure)
            raise

    def _optimized_incremental_undo(self, mv: Move, color: Color) -> None:
        affected = {mv.from_coord, mv.to_coord}
        self._batch_update_pieces(affected, color)
        self._attacked_squares_valid[Color.WHITE] = False
        self._attacked_squares_valid[Color.BLACK] = False

    def _infer_piece(self, coord: Tuple[int, int, int], color: Color) -> Piece:
        """Infer piece type from legal move cache when board is stale."""
        # UPDATED: Use common.py
        return infer_piece_from_cache(self._cache_manager, coord)

    def get_attacked_squares(self, color: Color) -> Set[Tuple[int, int, int]]:
        if not self._attacked_squares_valid[color]:
            self._update_attacked_squares(color)
        return self._attacked_squares[color].copy()

    def _update_attacked_squares(self, color: Color) -> None:
        # CORRECTION: Validate against current occupancy before adding
        self._attacked_squares[color].clear()
        if self._needs_rebuild:
            self._full_rebuild()
            return
        for mv in self._legal_by_color[color]:
            if self._cache.occupancy.get(mv.from_coord):  # Skip if piece gone
                self._attacked_squares[color].add(mv.to_coord)
        self._attacked_squares_valid[color] = True

    def invalidate_square(self, coord: Tuple[int,int,int]) -> None:
        """Cheap O(1) mark."""
        self._invalid_squares.add(coord)
        # remove immediately so we do not rely on the big rebuild
        self._legal_per_piece.pop(coord, None)

    def invalidate_squares(self, squares: Set[Tuple[int, int, int]]) -> None:
        for sq in squares:
            self.invalidate_square(sq)

    def invalidate_attacked_squares(self, color: Color) -> None:
        self._invalid_attacks.add(color)
        self._attacked_squares_valid[color] = False

    def _lazy_revalidate(self) -> None:
        """Regenerate only what is strictly needed; clear flags when done."""
        if not self._invalid_squares and not self._invalid_attacks:
            return  # fast-path – nothing to do

        try:
            from game3d.game.gamestate import GameState
            from game3d.movement.generator import generate_legal_moves_for_piece
            tmp_state = GameState(
                board=self._cache_manager.board,
                color=self._current,
                cache=self._cache_manager
            )

            # ---------- 1.  rebuild frozen / debuffed bitmaps ----------
            player_pieces = list(get_player_pieces(tmp_state, self._current))
            self._frozen_bitmap.fill(False)
            self._debuffed_bitmap.fill(False)
            for coord, piece in player_pieces:
                x, y, z = coord
                # guard against bad coordinates (paranoid)
                if not (0 <= x < 9 and 0 <= y < 9 and 0 <= z < 9):
                    continue
                if self._cache_manager.is_frozen(coord, self._current):
                    self._frozen_bitmap[z, y, x] = True
                if self._cache_manager.is_movement_debuffed(coord, self._current):
                    self._debuffed_bitmap[z, y, x] = True

            # ---------- 2.  regenerate moves for dirty squares ----------
            for coord in list(self._invalid_squares):
                x, y, z = coord
                if not (0 <= x < 9 and 0 <= y < 9 and 0 <= z < 9):  # extra safety
                    self._legal_per_piece.pop(coord, None)
                    continue

                piece = self._cache_manager.occupancy.get(coord)
                if piece and piece.color == self._current and not self._frozen_bitmap[z, y, x]:
                    self._legal_per_piece[coord] = generate_legal_moves_for_piece(tmp_state, coord)
                else:
                    self._legal_per_piece.pop(coord, None)

            self._invalid_squares.clear()

            # ---------- 3.  refresh attacked-squares maps ----------
            for color in list(self._invalid_attacks):
                self._update_attacked_squares(color)
            self._invalid_attacks.clear()

            # ---------- 4.  rebuild colour-level lists ----------
            self._rebuild_color_lists()

            # ---------- 5.  keep generation counter in sync ----------
            if hasattr(self._cache_manager.board, 'generation'):
                self._gen = self._cache_manager.board.generation

        except Exception as e:
            print(f"[ERROR] Lazy revalidation failed: {e}")
            import traceback
            traceback.print_exc()
            self._invalid_squares.clear()
            self._invalid_attacks.clear()
            raise

    def is_frozen(self, coord: Coord) -> bool:
        """Check if a coordinate is frozen."""
        x, y, z = coord
        return self._frozen_bitmap[z, y, x]

    def is_debuffed(self, coord: Coord) -> bool:
        """Check if a coordinate is debuffed."""
        x, y, z = coord
        return self._debuffed_bitmap[z, y, x]
# ==============================================================================
# FACTORY
# ==============================================================================
def create_optimized_move_cache(
    board: "Board",
    current: Color,
    cache_manager: "CacheManager",
) -> OptimizedMoveCache:
    return OptimizedMoveCache(board, current, cache_manager)
