# movecache.py - UPDATED TO USE create_batch
# movecache.py - CONSOLIDATED VERSION WITH AUTO-DETECTION
from __future__ import annotations
import random
import time
from typing import Dict, List, Optional, Tuple, Iterable, Set, Any, TYPE_CHECKING, Union
from dataclasses import dataclass
import numpy as np

if TYPE_CHECKING:
    from game3d.board.board import Board
    from game3d.game.gamestate import GameState
    from game3d.cache.manager import OptimizedCacheManager

from game3d.common.enums import Color, PieceType
from game3d.movement.movepiece import Move
from game3d.pieces.piece import Piece

# Import common modules
from game3d.common.coord_utils import in_bounds, in_bounds_vectorised, validate_and_sanitize_coord, Coord, add_coords
from game3d.common.piece_utils import get_player_pieces, infer_piece_from_cache, get_piece_effect_type, is_effect_piece
from game3d.common.debug_utils import fallback_mode, CacheStatsMixin
from game3d.cache.caches.zobrist import compute_zobrist, ZobristHash
from game3d.board.symmetry import SymmetryManager
from game3d.cache.caches.symmetry_tt import SymmetryAwareTranspositionTable
from game3d.cache.caches.transposition import TranspositionTable
from game3d.common.move_utils import filter_none_moves
from game3d.common.validation import validate_move_comprehensive  # UPDATED: Use new validation module
from game3d.game.gamestate import GameState
from game3d.movement.generator import generate_legal_moves_for_piece
from game3d.common.cache_utils import get_piece, is_occupied

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
# OPTIMIZED MOVE CACHE — CONSOLIDATED VERSION
# ==============================================================================
class OptimizedMoveCache(CacheStatsMixin):
    __slots__ = (
        "_current", "_cache", "_legal_per_piece", "_legal_by_color",
        "_zobrist", "_transposition_table",
        "_zobrist_hash", "_age_counter",
        "_simple_move_cache", "symmetry_manager", "symmetry_tt",
        "_main_tt_size_mb", "_sym_tt_size_mb",
        "_needs_rebuild", "_attacked_squares", "_attacked_squares_valid",
        "_cache_manager", "_dirty_flags", "_invalid_squares", "_invalid_attacks",
        "_gen", "_board"
    )

    def __init__(
        self,
        board: "Board",
        current: Color,
        cache_manager: "OptimizedCacheManager",
    ) -> None:
        super().__init__()
        self._current = current
        self._cache_manager = cache_manager
        self._legal_per_piece: Dict[Tuple[int, int, int], List[Move]] = {}
        self._legal_by_color: Dict[Color, List[Move]] = {
            Color.WHITE: [], Color.BLACK: []
        }

        self._zobrist = ZobristHash()
        self._zobrist_hash = cache_manager._current_zobrist_hash

        # Use cache manager's config
        main_mb = cache_manager.config.main_tt_size_mb
        sym_mb = cache_manager.config.sym_tt_size_mb

        self.symmetry_manager = SymmetryManager()
        self.symmetry_tt = SymmetryAwareTranspositionTable(
            self.symmetry_manager, size_mb=sym_mb
        )

        self._transposition_table = TranspositionTable(size_mb=main_mb)
        self._main_tt_size_mb = cache_manager.config.main_tt_size_mb
        self._sym_tt_size_mb = cache_manager.config.sym_tt_size_mb

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
        self._board = cache_manager.board
        self.attacks_cache = cache_manager.attacks_cache
        self._gen = -1

    @property
    def _cache_manager_ref(self):
        return self._cache_manager

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
        """Incremental move application with proper cache invalidation."""
        # Get current board generation
        board_gen = getattr(self._cache_manager.board, 'generation', 0)

        # Check if we need rebuild due to generation mismatch
        if self._gen != board_gen:
            self._needs_rebuild = True
            self._gen = board_gen

        if self._needs_rebuild:
            self._full_rebuild()
            return

        # Use common validation
        if not validate_move_comprehensive(self._cache_manager, mv, color):
            self._needs_rebuild = True
            return

        # Use common piece access
        piece = get_piece(self._cache_manager, mv.from_coord)
        if piece is None:
            self._needs_rebuild = True
            return

        # Incremental update - only invalidate affected squares
        self.invalidate_affected_squares(mv)
        self.invalidate_attacked_squares(color)
        self.invalidate_attacked_squares(color.opposite())

        # Update state
        self._current = color.opposite()
        self._age_counter += 1
        self._gen = board_gen  # Use the board's generation
        self._needs_rebuild = False

    def undo_move(self, mv: Move, color: Color) -> None:
        """
        Undo a move and incrementally regenerate only the squares that
        changed.  Board tensor is rolled back **before** any cache update
        or move generation.
        """
        # Use common piece access
        piece_now = get_piece(self._cache_manager, mv.to_coord)
        captured_type = getattr(mv, "captured_ptype", None)
        captured_piece = (
            Piece(color.opposite(), PieceType(captured_type)) if captured_type else None
        )

        # Use cache manager's ZobristHash instance - FIXED
        if hasattr(self._cache_manager, '_zobrist'):
            self._cache_manager._current_zobrist_hash = self._cache_manager._zobrist.update_hash_move(
                self._cache_manager._current_zobrist_hash, mv, piece_now, captured_piece
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
        self.invalidate_squares(mv.from_coord)
        self.invalidate_squares(mv.to_coord)
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
    # CONSOLIDATED FUNCTIONS WITH AUTO-DETECTION
    # ------------------------------------------------------------------
    def invalidate_squares(self, coords: Union[Coord, Iterable[Coord], np.ndarray]) -> None:
        """Consolidated invalidation - auto-detects input type"""
        if coords is None:
            return

        # Auto-detect input type and convert to appropriate format
        if isinstance(coords, np.ndarray):
            # Vectorized input
            if coords.ndim == 1 and len(coords) == 3:
                # Single coordinate as array
                coords = [tuple(coords)]
            elif coords.ndim == 2:
                # Batch coordinates
                valid_mask = in_bounds_vectorised(coords)
                coords_set = set(tuple(coord) for coord in coords[valid_mask])
            else:
                return
        elif isinstance(coords, (tuple, list)) and len(coords) == 3 and all(isinstance(c, int) for c in coords):
            # Single coordinate
            coords_set = {validate_and_sanitize_coord(coords)}
        else:
            # Iterable of coordinates
            coords_set = set()
            for coord in coords:
                sanitized = validate_and_sanitize_coord(coord)
                if sanitized:
                    coords_set.add(sanitized)

        # Process all coordinates
        for coord in coords_set:
            if coord is None:
                continue
            self._invalid_squares.add(coord)
            self._legal_per_piece.pop(coord, None)

    def update_pieces(self, coords: Union[Coord, Iterable[Coord], np.ndarray],
                     color: Optional[Color] = None) -> None:
        """Consolidated piece update - auto-detects input type"""
        if color is None:
            color = self._current

        if coords is None or (hasattr(coords, '__len__') and len(coords) == 0):
            return

        # Auto-detect and convert input
        if isinstance(coords, np.ndarray):
            if coords.ndim == 1 and len(coords) == 3:
                # Single coordinate as array
                coords_array = coords.reshape(1, -1)
            elif coords.ndim == 2:
                # Already batch format
                coords_array = coords
            else:
                return
        else:
            if isinstance(coords, (tuple, list)) and len(coords) == 3 and all(isinstance(c, int) for c in coords):
                # Single coordinate
                coords_array = np.array([coords])
            else:
                # Iterable of coordinates
                coords_array = np.array(list(coords))

        if len(coords_array) == 0:
            return

        # Batch processing for all cases - FIXED: Use batch_get_pieces instead of batch_get_pieces_vectorized
        pieces_data = self._cache_manager.occupancy.get_batch(coords_array)
        frozen_mask = self._cache_manager.batch_get_frozen_status(coords_array, color)

        # Filter pieces that belong to the color and need updating
        active_coords = []
        for i, coord in enumerate(coords_array):
            piece = pieces_data[i]
            if piece and piece.color == color and not frozen_mask[i]:
                active_coords.append(tuple(coord))

        if not active_coords:
            return

        # Batch move generation
        from game3d.movement.pseudo_legal import generate_pseudo_legal_moves_batch
        from game3d.game.gamestate import GameState

        tmp_state = GameState(
            board=self._board,
            color=color,
            cache_manager=self._cache_manager  # FIXED: changed 'cache' to 'cache_manager'
        )
        batch_moves = generate_pseudo_legal_moves_batch(tmp_state, np.array(active_coords))

        # Update cache
        for coord, moves in zip(active_coords, batch_moves):
            if moves:
                self._legal_per_piece[coord] = moves
            else:
                self._legal_per_piece.pop(coord, None)

        self._rebuild_color_lists()

    def _generate_piece_moves(self, coord: Union[Coord, np.ndarray]) -> List[Move]:
        """Consolidated move generation - handles single or batch input"""
        if isinstance(coord, np.ndarray) and coord.ndim > 1:
            # Batch processing - FIXED: use cache_manager parameter
            from game3d.game.gamestate import GameState
            from game3d.movement.pseudo_legal import generate_pseudo_legal_moves_batch

            tmp_state = GameState(
                board=self._board,
                color=self._current,
                cache_manager=self._cache_manager  # FIXED: changed 'cache' to 'cache_manager'
            )
            batch_moves = generate_pseudo_legal_moves_batch(tmp_state, coord)

            # Flatten results for single return (caller should handle appropriately)
            all_moves = []
            for moves in batch_moves:
                all_moves.extend(moves)
            return all_moves
        else:
            # Single coordinate processing - FIXED: use cache_manager parameter
            if isinstance(coord, np.ndarray) and coord.ndim == 1:
                coord = tuple(coord)

            from game3d.game.gamestate import GameState
            from game3d.movement.generator import generate_legal_moves_for_piece

            piece = get_piece(self._cache_manager, coord)
            if not piece:
                return []

            tmp_state = GameState(
                board=self._board,
                color=piece.color,
                cache_manager=self._cache_manager  # FIXED: changed 'cache' to 'cache_manager'
            )
            return generate_legal_moves_for_piece(tmp_state, coord)

    # ------------------------------------------------------------------
    # Internal helpers (updated to use consolidated functions)
    # ------------------------------------------------------------------
    def _optimized_incremental_update(self, mv: Move, color: Color, from_coord: Tuple[int, int, int],
                                    to_coord: Tuple[int, int, int], piece: Piece,
                                    captured_piece: Optional[Piece]) -> None:
        affected_squares = {from_coord, to_coord}
        # Use consolidated update function
        self.update_pieces(affected_squares, color.opposite())
        self._attacked_squares_valid[Color.WHITE] = False
        self._attacked_squares_valid[Color.BLACK] = False

    def _rebuild_color_lists(self) -> None:
        """Rebuild color lists using batch operations"""
        white_moves = []
        black_moves = []

        if not self._legal_per_piece:
            self._legal_by_color[Color.WHITE] = white_moves
            self._legal_by_color[Color.BLACK] = black_moves
            return

        # Convert keys to numpy array safely
        coords_list = list(self._legal_per_piece.keys())
        if not coords_list:
            self._legal_by_color[Color.WHITE] = []
            self._legal_by_color[Color.BLACK] = []
            return

        coords = np.array(coords_list)
        pieces = self._cache_manager.occupancy.get_batch(coords)

        for i, coord in enumerate(coords_list):
            piece = pieces[i]
            if not piece:
                continue

            moves = self._legal_per_piece[coord]
            moves = filter_none_moves(moves)

            if not moves:
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

            # Create temporary state for move generation - FIXED: use cache_manager parameter
            from game3d.game.gamestate import GameState
            tmp_state = GameState(
                board=self._cache_manager.board,
                color=self._current,
                cache_manager=self._cache_manager  # FIXED: changed 'cache' to 'cache_manager'
            )

            # Check if board has pieces
            if occupancy.count == 0:
                print("[REBUILD] WARNING: Board has no pieces!")
                self._legal_by_color[Color.WHITE] = []
                self._legal_by_color[Color.BLACK] = []
                self._legal_per_piece.clear()
                return

            # Generate pseudo-legal moves using batch methods
            from game3d.movement.pseudo_legal import generate_pseudo_legal_moves
            pseudo_moves = generate_pseudo_legal_moves(tmp_state)

            # DEFENSIVE: Filter out None values immediately using common utility
            pseudo_moves = filter_none_moves(pseudo_moves)

            # Filter out moves from empty squares (paranoid check)
            valid_moves = []
            for move in pseudo_moves:
                # Use common piece access
                piece = get_piece(self._cache_manager, move.from_coord)
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
        # Use consolidated update function
        self.update_pieces(affected, color)
        self._attacked_squares_valid[Color.WHITE] = False
        self._attacked_squares_valid[Color.BLACK] = False

    def _infer_piece(self, coord: Tuple[int, int, int], color: Color) -> Piece:
        """Infer piece type from legal move cache when board is stale."""
        # Use common piece inference
        return infer_piece_from_cache(self._cache_manager, coord)

    def get_attacked_squares(self, color: Color) -> Set[Tuple[int, int, int]]:
        if not self._attacked_squares_valid[color]:
            self._update_attacked_squares(color)
        return self._attacked_squares[color].copy()

    def _update_attacked_squares(self, color: Color) -> None:
        """Use the attacks cache instead of recalculating"""
        if self.attacks_cache:
            attacked_squares = self.attacks_cache.get_for_color(color)
            if attacked_squares is not None:
                self._attacked_squares[color] = attacked_squares
                self._attacked_squares_valid[color] = True
                return

        # Fallback to manual calculation if attacks cache is invalid
        self._attacked_squares[color].clear()
        if self._needs_rebuild:
            self._full_rebuild()
            return

        for mv in self._legal_by_color[color]:
            if get_piece(self._cache_manager, mv.from_coord):
                self._attacked_squares[color].add(mv.to_coord)
        self._attacked_squares_valid[color] = True

    def invalidate_attacked_squares(self, color: Color) -> None:
        self._invalid_attacks.add(color)
        self._attacked_squares_valid[color] = False

    def _lazy_revalidate(self) -> None:
        """Regenerate only what is strictly needed using consolidated functions"""
        if not self._invalid_squares and not self._invalid_attacks:
            return

        try:
            from game3d.game.gamestate import GameState

            tmp_state = GameState(
                board=self._cache_manager.board,
                color=self._current,
                cache_manager=self._cache_manager  # FIXED: changed 'cache' to 'cache_manager'
            )

            # Use consolidated update function for all invalid squares
            if self._invalid_squares:
                self.update_pieces(list(self._invalid_squares), self._current)
                self._invalid_squares.clear()

            # Update attacked squares
            for color in list(self._invalid_attacks):
                self._update_attacked_squares(color)
            self._invalid_attacks.clear()

            # Keep generation counter in sync
            if hasattr(self._cache_manager.board, 'generation'):
                self._gen = self._cache_manager.board.generation

        except Exception as e:
            print(f"[ERROR] Lazy revalidation failed: {e}")
            import traceback
            traceback.print_exc()
            self._invalid_squares.clear()
            self._invalid_attacks.clear()
            raise

    def invalidate_affected_squares(self, mv: Move) -> None:
        """Fine-grained invalidation of only affected squares."""
        affected = {mv.from_coord, mv.to_coord}

        # Add attacked squares in move direction for better cache efficiency
        from game3d.common.coord_utils import generate_ray
        direction = tuple(b - a for a, b in zip(mv.from_coord, mv.to_coord))
        if any(d != 0 for d in direction):
            # Normalize direction
            max_d = max(abs(d) for d in direction)
            unit_dir = tuple(d // max_d for d in direction)

            # Invalidate squares along attack ray (limited range)
            ray = generate_ray(mv.to_coord, unit_dir, max_steps=3)
            affected.update(ray)

        # Use consolidated invalidation function
        self.invalidate_squares(affected)

# ==============================================================================
# FACTORY
# ==============================================================================
def create_optimized_move_cache(
    board: "Board",
    current: Color,
    cache_manager: "OptimizedCacheManager",
) -> OptimizedMoveCache:
    return OptimizedMoveCache(board, current, cache_manager)
