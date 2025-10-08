# movecache.py
from __future__ import annotations
import os
import random
import time
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Set, Any
from dataclasses import dataclass
import gzip
import pickle
import msgpack

if TYPE_CHECKING:
    from game3d.cache.manager import CacheManager
    from game3d.board.board import Board

from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move
from game3d.pieces.piece import Piece
from game3d.common.common import in_bounds
from game3d.game.zobrist import compute_zobrist, ZobristHash
from game3d.board.symmetry import SymmetryManager
from game3d.cache.caches.symmetry_tt import SymmetryAwareTranspositionTable
from game3d.cache.caches.transposition import TranspositionTable
from game3d.movement.generator import generate_legal_moves, generate_legal_moves_for_piece
# Add this to avoid circular import
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
# OPTIMIZED MOVE CACHE — FULLY TUNED
# ==============================================================================
class OptimizedMoveCache:
    __slots__ = (
        "_current", "_cache", "_legal_per_piece", "_legal_by_color",
        "_king_pos", "_priest_count", "_has_priest", "_zobrist",
        "_transposition_table",
        "_zobrist_hash", "_age_counter",
        "_simple_move_cache", "symmetry_manager", "symmetry_tt",
        "_save_dir", "_main_tt_size_mb", "_sym_tt_size_mb",
        "_needs_rebuild", "_attacked_squares", "_attacked_squares_valid",
        "_cache_manager", "_dirty_flags", "_invalid_squares", "_invalid_attacks"
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
        # Disk cache setup
        self._save_dir = "/home/salamimommy/Documents/code/3d/game3d/cache/caches/movescachedisk"  # Directory for save files
        os.makedirs(self._save_dir, exist_ok=True)
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

    @property
    def _board(self) -> "Board":
        return self._cache.board

    @property
    def main_tt_size_mb(self) -> int:
        return self._main_tt_size_mb  # Return stored value

    @property
    def sym_tt_size_mb(self) -> int:
        return self._sym_tt_size_mb  # Return stored value

    # --------------------------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------------------------
    def legal_moves(self, color: Color, *, parallel: bool = True, max_workers: int = 8) -> list[Move]:
        self._lazy_revalidate()          # <-- new
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

    def apply_move(self, mv: Move, color: Color) -> None:
        """Mark as dirty instead of rebuilding."""
        # Just mark dirty - rebuild on next query
        self._dirty_flags['targets'] = True
        self._dirty_flags['attacks'] = True

        from_coord = mv.from_coord
        to_coord = mv.to_coord

        # Update Zobrist
        piece = self._cache_manager.occupancy.get(from_coord)
        if not piece:
            self._needs_rebuild = True
            return

        captured_piece = self._cache_manager.occupancy.get(to_coord) if mv.is_capture else None

        self._zobrist_hash = self._zobrist.update_hash_move(
            self._zobrist_hash, mv, piece, captured_piece
        )

        # Mark affected pieces as dirty
        self._legal_per_piece.pop(from_coord, None)
        self._legal_per_piece.pop(to_coord, None)

        # Mark for incremental rebuild on next query
        self._dirty_flags['targets'] = True

        # Switch turn
        self._current = color.opposite()
        self._age_counter += 1

    def undo_move(self, mv: Move, color: Color) -> None:
        """
        FIXED: Undo a move and roll the Zobrist hash back.
        Order is critical: hash update uses the *current* board state.
        """
        # 1. Read the state BEFORE we touch the board
        piece = self._board.piece_at(mv.to_coord)  # piece that just arrived here
        captured_piece = None
        if getattr(mv, "is_capture", False):
            captured_type = getattr(mv, "captured_ptype", None)
            if captured_type is not None:
                from game3d.pieces.piece import Piece  # ADDED: Missing import
                captured_piece = Piece(color.opposite(), captured_type)

        # 2. Roll the hash back while the position is still intact
        if piece is None:
            raise ValueError(f"No piece found at move target coordinate {mv.to_coord} during undo")

        self._zobrist_hash = self._zobrist.update_hash_move(
            self._zobrist_hash, mv, piece, captured_piece
        )

        # 3. Now mutate the board
        self._undo_move_optimized(mv, color)
        self._current = color
        self._age_counter += 1

        # 4. Incremental cache update
        self._optimized_incremental_undo(mv, color)

    def _undo_move_optimized(self, mv: Move, color: Color) -> None:
        """FIXED: Properly restore board state during undo."""
        from game3d.pieces.enums import PieceType  # ADDED: Missing import
        from game3d.pieces.piece import Piece  # ADDED: Missing import

        # Restore captured piece if needed
        if getattr(mv, "is_capture", False):
            captured_type = getattr(mv, "captured_ptype", None)
            if captured_type is not None:
                self._board.set_piece(mv.to_coord, Piece(color.opposite(), captured_type))

        # Get piece at destination (what was moved)
        piece = self._board.piece_at(mv.to_coord)
        if piece:
            # Move it back
            self._board.set_piece(mv.from_coord, piece)
            self._board.set_piece(mv.to_coord, None)

        # Handle promotion undo
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
            'tt_size_mb': self.main_tt_size_mb,
            'symmetry_tt_size_mb': self._sym_tt_size_mb
        }
        if hasattr(self.symmetry_tt, 'get_symmetry_stats'):
            base_stats['symmetry_stats'] = self.symmetry_tt.get_symmetry_stats()
        return base_stats

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

    def _save_to_disk(self) -> None:
        timestamp = int(time.time())

        for tt, prefix in [(self._transposition_table, "main"), (self.symmetry_tt, "sym")]:
            path = os.path.join(self._save_dir, f"{prefix}_tt_{timestamp}.msgpack")
            try:
                import msgpack
                # Use msgpack for faster serialization
                entries = []
                for idx in range(tt.size):
                    entry = tt.table[idx]
                    if entry is not None:
                        entries.append((idx, entry))

                with open(path, 'wb') as f:
                    msgpack.dump(entries, f)
            except ImportError:
                # Fallback to pickle/gzip
                path = os.path.join(self._save_dir, f"{prefix}_tt_{timestamp}.pkl.gz")
                with gzip.open(path, 'wb') as f:
                    for idx in range(tt.size):
                        entry = tt.table[idx]
                        if entry is not None:
                            pickle.dump((idx, entry), f, protocol=4)

        print(f"Saved TTs to {self._save_dir} with timestamp {timestamp}")

    # --------------------------------------------------------------------------
    # INTERNAL OPTIMIZATIONS
    # --------------------------------------------------------------------------
    def _optimized_incremental_update(self, mv: Move, color: Color, from_coord: Tuple[int, int, int],
                                    to_coord: Tuple[int, int, int], piece: Piece,
                                    captured_piece: Optional[Piece]) -> None:
        self._refresh_counts()
        affected_squares = {from_coord, to_coord}

        # Add king positions to affected squares
        for col in (Color.WHITE, Color.BLACK):
            king_pos = self._find_king(col)
            if king_pos:
                affected_squares.add(king_pos)

        # Update only affected pieces
        self._batch_update_pieces(affected_squares, color.opposite())
        self._attacked_squares_valid[Color.WHITE] = False
        self._attacked_squares_valid[Color.BLACK] = False

    def _batch_update_pieces(self, affected_squares: Set[Tuple[int, int, int]], color: Color) -> None:
        """Only update if cache is valid - otherwise mark for rebuild."""
        if self._needs_rebuild:
            return  # Don't waste time on incremental updates if full rebuild is pending

        # Clear affected pieces from cache
        for coord in affected_squares:
            self._legal_per_piece.pop(coord, None)

        # Identify pieces that need updating
        pieces_to_update = [
            coord for coord in affected_squares
            if (p := self._board.piece_at(coord)) and p.color == color
        ]

        # NEW: Use generator.py for updates
        from game3d.game.gamestate import GameState
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
        piece = self._board.piece_at(coord)
        if not piece:
            return []

        tmp_state = GameState(board=self._board, color=piece.color, cache=self._cache)
        return generate_legal_moves_for_piece(tmp_state, coord)

    def _rebuild_color_lists(self) -> None:
        white_moves = []
        black_moves = []
        for coord, moves in self._legal_per_piece.items():
            if not moves:
                continue
            piece = self._board.piece_at(coord)
            if not piece:
                continue
            if piece.color == Color.WHITE:
                white_moves.extend(moves)
            else:
                black_moves.extend(moves)
        self._legal_by_color[Color.WHITE] = white_moves
        self._legal_by_color[Color.BLACK] = black_moves

    def _full_rebuild(self) -> None:
        """Full rebuild - simplified."""
        # Clear caches
        self._legal_per_piece.clear()

        # Generate all legal moves
        from game3d.game.gamestate import GameState
        tmp_state = GameState(board=self._cache_manager.board,
                             color=self._current,
                             cache=self._cache_manager)

        try:
            all_moves = generate_legal_moves(tmp_state)
        except Exception as e:
            print(f"[ERROR] Legal move generation failed: {e}")
            self._legal_by_color[Color.WHITE] = []
            self._legal_by_color[Color.BLACK] = []
            return

        # Split by piece
        for move in all_moves:
            self._legal_per_piece.setdefault(move.from_coord, []).append(move)

        # Rebuild color lists
        self._rebuild_color_lists()

        # Refresh counts
        self._refresh_counts()

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
        for coord, piece in self._board.list_occupied():
            if piece.ptype == PieceType.PRIEST:
                self._priest_count[piece.color] += 1
            elif piece.ptype == PieceType.KING:
                self._king_pos[piece.color] = coord
        for color in Color:
            self._has_priest[color] = self._priest_count[color] > 0

    def _optimized_incremental_undo(self, mv: Move, color: Color) -> None:
        affected = {mv.from_coord, mv.to_coord}
        for col in (Color.WHITE, Color.BLACK):
            k = self._find_king(col)
            if k:
                affected.add(k)
        self._batch_update_pieces(affected, color)
        self._attacked_squares_valid[Color.WHITE] = False
        self._attacked_squares_valid[Color.BLACK] = False


    def _infer_piece(self, coord: Tuple[int, int, int], color: Color) -> Piece:
        """Infer piece type from legal move cache when board is stale."""
        if coord in self._legal_per_piece:
            # Any move from this coord tells us the piece type
            moves = self._legal_per_piece[coord]
            if moves:
                # Reconstruct piece from move metadata
                return Piece(color, PieceType.PAWN)  # Default fallback
        return Piece(color, PieceType.PAWN)

    def get_attacked_squares(self, color: Color) -> Set[Tuple[int, int, int]]:
        """Get all squares attacked by pieces of the given color."""
        # Trigger rebuild if needed before computing attacks
        if self._needs_rebuild:
            self._full_rebuild()
            self._needs_rebuild = False

        if not self._attacked_squares_valid[color]:
            self._update_attacked_squares(color)
        return self._attacked_squares[color].copy()

    def _update_attacked_squares(self, color: Color) -> None:
        """Update attacked squares for a color using legal moves."""
        self._attacked_squares[color].clear()

        # Get all legal moves for this color
        legal_moves = self._legal_by_color[color]

        # Extract all destination coordinates (attacked squares)
        for move in legal_moves:
            self._attacked_squares[color].add(move.to_coord)

        self._attacked_squares_valid[color] = True



    def invalidate_square(self, coord: Tuple[int,int,int]) -> None:
        """Cheap O(1) mark."""
        self._invalid_squares.add(coord)
        # remove immediately so we do not rely on the big rebuild
        self._legal_per_piece.pop(coord, None)

    def invalidate_attacked_squares(self, color: Color) -> None:
        self._invalid_attacks.add(color)
        self._attacked_squares_valid[color] = False

    def _lazy_revalidate(self) -> None:
        """Regenerate only what is strictly needed."""
        if not self._invalid_squares and not self._invalid_attacks:
            return  # hot-path exit – most moves do not touch auras

        from game3d.game.gamestate import GameState
        tmp_state = GameState(board=self._board, color=self._current, cache=self._cache_manager)

        # 1. fix individual pieces
        for coord in list(self._invalid_squares):
            piece = self._board.piece_at(coord)
            if piece and piece.color == self._current:
                self._legal_per_piece[coord] = generate_legal_moves_for_piece(tmp_state, coord)
            else:
                self._legal_per_piece.pop(coord, None)
        self._invalid_squares.clear()

        # 2. fix attacked-squares bitmaps
        for color in list(self._invalid_attacks):
            self._update_attacked_squares(color)
        self._invalid_attacks.clear()

        # 3. rebuild color lists **only** from what is still there
        self._rebuild_color_lists()


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
