from __future__ import annotations
"""Optimized incremental legal-move cache for 9×9×9 3D chess with advanced techniques."""
# game3d/cache/movecache.py

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Set, Any
from dataclasses import dataclass
import random
import struct
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np

if TYPE_CHECKING:
    from game3d.cache.manager import CacheManager

from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move
from game3d.movement.registry import get_dispatcher
from game3d.attacks.check import king_in_check
from game3d.board.board import Board
from game3d.pieces.piece import Piece

# ==============================================================================
# ADVANCED CACHING DATA STRUCTURES
# ==============================================================================

@dataclass
class CompactMove:
    """Ultra-compact move representation using bit packing."""
    data: int = 0

    def __init__(self, from_coord: Tuple[int, int, int], to_coord: Tuple[int, int, int],
                 piece_type: PieceType, is_capture: bool = False,
                 captured_type: Optional[PieceType] = None,
                 is_promotion: bool = False):

        from_index = from_coord[0] * 81 + from_coord[1] * 9 + from_coord[2]
        to_index = to_coord[0] * 81 + to_coord[1] * 9 + to_coord[2]

        self.data = (from_index & 0x1FFFFF) | \
                   ((to_index & 0x1FFFFF) << 21) | \
                   ((piece_type.value & 0x3F) << 42) | \
                   ((captured_type.value & 0x3F) << 48 if captured_type else 0) | \
                   (1 << 54 if is_capture else 0) | \
                   (1 << 55 if is_promotion else 0)

    def unpack(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], bool, bool]:
        """Unpack move data."""
        from_index = self.data & 0x1FFFFF
        to_index = (self.data >> 21) & 0x1FFFFF
        is_capture = bool(self.data & (1 << 54))
        is_promotion = bool(self.data & (1 << 55))

        from_coord = (from_index // 81, (from_index % 81) // 9, from_index % 9)
        to_coord = (to_index // 81, (to_index % 81) // 9, to_index % 9)

        return from_coord, to_coord, is_capture, is_promotion

class TTEntry:
    """Transposition table entry with optimized memory layout."""
    __slots__ = ('hash_key', 'depth', 'score', 'node_type', 'best_move', 'age')

    def __init__(self, hash_key: int, depth: int, score: int, node_type: int,
                 best_move: Optional[CompactMove], age: int):
        self.hash_key = hash_key
        self.depth = depth
        self.score = score
        self.node_type = node_type  # 0=exact, 1=lower, 2=upper
        self.best_move = best_move
        self.age = age

# ==============================================================================
# ZOBRIST HASHING SYSTEM
# ==============================================================================

class ZobristHashing:
    """High-performance Zobrist hashing for 3D chess positions."""

    def __init__(self):
        # Generate high-quality random numbers for each piece-square combination
        self.piece_keys = self._generate_piece_keys()
        self.side_to_move_key = random.getrandbits(64)
        self.castling_keys = [random.getrandbits(64) for _ in range(16)]  # 4 castling rights
        self.en_passant_keys = [random.getrandbits(64) for _ in range(9)]  # 9 files

    def _generate_piece_keys(self) -> Dict[Tuple[Color, PieceType, Tuple[int, int, int]], int]:
        """Generate random keys for all piece-position combinations."""
        keys = {}
        for color in [Color.WHITE, Color.BLACK]:
            for piece_type in PieceType:
                for x in range(9):
                    for y in range(9):
                        for z in range(9):
                            key = (color, piece_type, (x, y, z))
                            keys[key] = random.getrandbits(64)
        return keys

    def compute_hash(self, board: Board, current_color: Color) -> int:
        """Compute Zobrist hash for current board state."""
        hash_value = 0

        # XOR all pieces on board
        for coord, piece in board.list_occupied():
            key = (piece.color, piece.ptype, coord)
            hash_value ^= self.piece_keys[key]

        # XOR side to move
        if current_color == Color.WHITE:
            hash_value ^= self.side_to_move_key

        return hash_value

    def update_hash_move(self, hash_value: int, move: Move, piece: Piece,
                        captured_piece: Optional[Piece]) -> int:
        """Incrementally update hash after move using XOR properties."""
        # Remove piece from source square
        hash_value ^= self.piece_keys[(piece.color, piece.ptype, move.from_coord)]

        # Add piece to target square
        hash_value ^= self.piece_keys[(piece.color, piece.ptype, move.to_coord)]

        # Remove captured piece if any
        if captured_piece:
            hash_value ^= self.piece_keys[(captured_piece.color, captured_piece.ptype, move.to_coord)]

        # XOR side to move
        hash_value ^= self.side_to_move_key

        return hash_value

# ==============================================================================
# TRANSPOSITION TABLE
# ==============================================================================

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
              best_move: Optional[CompactMove], age: int) -> None:
        """Store evaluation with advanced replacement strategy."""
        index = hash_value & self.mask
        existing_entry = self.table[index]

        # Replacement strategy: always replace if deeper, or if old entry is too old
        should_replace = (existing_entry is None or
                         existing_entry.depth <= depth or
                         existing_entry.age < age - 4)

        if should_replace:
            self.table[index] = TTEntry(hash_value, depth, score, node_type, best_move, age)

# ==============================================================================
# BITBOARD CACHE KEYS
# ==============================================================================

class BitboardCacheKey:
    """Ultra-fast cache keys using bitboard representation for 9x9x9 board."""

    def __init__(self):
        # Pre-compute bitboard masks for 9x9x9 board (729 squares)
        # Need 12 64-bit words to represent all squares
        self.position_masks = {}
        for x in range(9):
            for y in range(9):
                for z in range(9):
                    pos = (x, y, z)
                    square_index = x * 81 + y * 9 + z
                    word_index = square_index // 64
                    bit_index = square_index % 64
                    self.position_masks[pos] = (word_index, 1 << bit_index)

    def create_cache_key(self, board: Board, current_color: Color) -> int:
        """Create compact bitboard-based cache key."""
        # Use 12 64-bit words for 9x9x9 board
        white_bitboards = [0] * 12
        black_bitboards = [0] * 12

        for coord, piece in board.list_occupied():
            word_index, bit_mask = self.position_masks[coord]
            if piece.color == Color.WHITE:
                white_bitboards[word_index] |= bit_mask
            else:
                black_bitboards[word_index] |= bit_mask

        # Create hash from bitboards
        key_hash = 0
        for i, (w_bb, b_bb) in enumerate(zip(white_bitboards, black_bitboards)):
            key_hash ^= w_bb ^ (b_bb << (i % 64))

        return key_hash ^ (current_color.value << 63)

# ==============================================================================
# PARALLEL MOVE GENERATION
# ==============================================================================

class ParallelMoveGenerator:
    """Parallel move generation for multi-core systems."""

    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or (os.cpu_count() or 1)

    def generate_moves_parallel(self, board: Board, color: Color,
                               piece_positions: List[Tuple[int, int, int]],
                               cache_manager: CacheManager) -> List[Move]:
        """Generate moves for multiple pieces in parallel."""

        if len(piece_positions) < 4:  # Not worth parallelizing for small sets
            return self._generate_moves_sequential(board, color, piece_positions, cache_manager)

        # Split piece positions among workers
        chunk_size = max(1, len(piece_positions) // self.max_workers)
        chunks = [piece_positions[i:i + chunk_size]
                   for i in range(0, len(piece_positions), chunk_size)]

        all_moves = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks for each chunk
            futures = []
            for chunk in chunks:
                future = executor.submit(self._generate_chunk_moves, board, color, chunk, cache_manager)
                futures.append(future)

            # Collect results
            for future in futures:
                all_moves.extend(future.result())

        return all_moves

    def _generate_moves_sequential(self, board: Board, color: Color,
                                  positions: List[Tuple[int, int, int]],
                                  cache_manager: CacheManager) -> List[Move]:
        """Sequential move generation for small sets."""
        moves = []
        for pos in positions:
            piece = board.piece_at(pos)
            if piece and piece.color == color:
                dispatcher = get_dispatcher(piece.ptype)
                if dispatcher:
                    tmp_state = GameState.__new__(GameState)
                    tmp_state.board = board
                    tmp_state.color = color
                    tmp_state.cache = cache_manager
                    piece_moves = dispatcher(tmp_state, *pos)
                    moves.extend([m for m in piece_moves if m.from_coord == pos])
        return moves

    def _generate_chunk_moves(self, board: Board, color: Color,
                             positions: List[Tuple[int, int, int]],
                             cache_manager: CacheManager) -> List[Move]:
        """Generate moves for a chunk of piece positions."""
        chunk_moves = []
        for pos in positions:
            piece = board.piece_at(pos)
            if piece and piece.color == color:
                dispatcher = get_dispatcher(piece.ptype)
                if dispatcher:
                    tmp_state = GameState.__new__(GameState)
                    tmp_state.board = board
                    tmp_state.color = color
                    tmp_state.cache = cache_manager
                    piece_moves = dispatcher(tmp_state, *pos)
                    chunk_moves.extend([m for m in piece_moves if m.from_coord == pos])
        return chunk_moves

# ==============================================================================
# OPTIMIZED MOVE CACHE
# ==============================================================================

class OptimizedMoveCache:
    """High-performance move cache with all advanced optimizations integrated."""

    __slots__ = (
        "_current", "_cache", "_legal_per_piece", "_legal_by_color",
        "_king_pos", "_priest_count", "_has_priest", "_zobrist",
        "_transposition_table", "_bitboard_keys", "_parallel_generator",
        "_dependency_graph", "_zobrist_hash", "_move_cache", "_age_counter",
        "_simple_move_cache", "_vectorized_cache"
    )

    def __init__(self, board: Board, current: Color, cache_manager: CacheManager):
        self._current = current
        self._cache = cache_manager
        self._has_priest = {Color.WHITE: False, Color.BLACK: False}
        self._legal_per_piece: Dict[Tuple[int, int, int], List[Move]] = {}
        self._legal_by_color: Dict[Color, List[Move]] = {
            Color.WHITE: [],
            Color.BLACK: []
        }
        self._king_pos: Dict[Color, Optional[Tuple[int, int, int]]] = {
            Color.WHITE: None,
            Color.BLACK: None
        }
        self._priest_count: Dict[Color, int] = {
            Color.WHITE: 0,
            Color.BLACK: 0
        }

        # Advanced caching systems
        self._zobrist = ZobristHashing()
        self._transposition_table = TranspositionTable(size_mb=512)  # 512MB for serious play
        self._bitboard_keys = BitboardCacheKey()
        self._parallel_generator = ParallelMoveGenerator()
        self._dependency_graph = self._build_dependency_graph()
        self._zobrist_hash = self._zobrist.compute_hash(board, current)
        self._age_counter = 0

        # Performance caches
        self._simple_move_cache: Dict[Tuple[Tuple[int, int, int], Color], List[Move]] = {}
        self._vectorized_cache: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], bool] = {}
        from game3d.board.symmetry import SymmetryManager
        self.symmetry_manager = SymmetryManager()
        self.symmetry_tt = SymmetryAwareTranspositionTable(self.symmetry_manager)
        self._full_rebuild()

    @property
    def _board(self) -> Board:
        """Always get the current board from the cache manager."""
        return self._cache.board

    def _build_dependency_graph(self) -> Dict[PieceType, Set[PieceType]]:
        """Build dependency graph for piece interactions."""
        return {
            PieceType.FREEZE_AURA: {PieceType.PAWN, PieceType.KNIGHT, PieceType.BISHOP,
                                   PieceType.ROOK, PieceType.QUEEN, PieceType.KING},
            PieceType.BLACK_HOLE: {PieceType.PAWN, PieceType.KNIGHT, PieceType.BISHOP,
                                  PieceType.ROOK, PieceType.QUEEN},
            PieceType.WHITE_HOLE: {PieceType.PAWN, PieceType.KNIGHT, PieceType.BISHOP,
                                  PieceType.ROOK, PieceType.QUEEN},
            PieceType.GEOMANCER: {PieceType.PAWN, PieceType.KNIGHT, PieceType.BISHOP,
                                 PieceType.ROOK, PieceType.QUEEN},
            PieceType.ARCHER: {PieceType.PAWN, PieceType.KNIGHT, PieceType.BISHOP,
                              PieceType.ROOK, PieceType.QUEEN, PieceType.KING},
            PieceType.WALL: {PieceType.PAWN, PieceType.KNIGHT, PieceType.BISHOP,
                            PieceType.ROOK, PieceType.QUEEN},
            PieceType.TRAILBLAZER: {PieceType.PAWN, PieceType.KNIGHT, PieceType.BISHOP,
                                   PieceType.ROOK, PieceType.QUEEN},
            PieceType.SPEEDER: {PieceType.PAWN, PieceType.KNIGHT, PieceType.BISHOP,
                               PieceType.ROOK, PieceType.QUEEN},
            PieceType.SLOWER: {PieceType.PAWN, PieceType.KNIGHT, PieceType.BISHOP,
                              PieceType.ROOK, PieceType.QUEEN},
            PieceType.SHARE_SQUARE: {PieceType.KNIGHT}
        }

    # --------------------------------------------------------------------------
    # PUBLIC API - Enhanced with advanced features
    # --------------------------------------------------------------------------
    def legal_moves(self, color: Color) -> List[Move]:
        """Return cached legal moves for side `color` with transposition table lookup."""
        # Check transposition table first
        tt_entry = self._transposition_table.probe(self._zobrist_hash)
        if tt_entry and tt_entry.node_type == 0:  # Exact score
            # Convert compact moves back to full moves if available
            if tt_entry.best_move:
                from_coord, to_coord, is_capture, is_promotion = tt_entry.best_move.unpack()
                # Need to reconstruct full move - this is simplified
                pass

        return self._legal_by_color[color]

    def get_cached_evaluation(self, hash_value: int) -> Optional[Tuple[int, int, Optional[CompactMove]]]:
        """Get cached evaluation from transposition table."""
        tt_entry = self._transposition_table.probe(hash_value)
        if tt_entry:
            return tt_entry.score, tt_entry.depth, tt_entry.best_move
        return None

    def store_evaluation(self, hash_value: int, depth: int, score: int,
                        node_type: int, best_move: Optional[CompactMove] = None) -> None:
        """Store evaluation in transposition table."""
        self._transposition_table.store(hash_value, depth, score, node_type, best_move, self._age_counter)

    def apply_move(self, mv: Move, color: Color) -> None:
        """Ultra-optimized move application with incremental updates."""
        from_coord = mv.from_coord
        to_coord = mv.to_coord

        # Fast path for simple moves
        if self._is_simple_move(mv, color):
            self._apply_simple_move(mv, color)
            return

        # Update Zobrist hash incrementally
        piece = self._board.piece_at(from_coord)
        captured_piece = self._board.piece_at(to_coord) if getattr(mv, 'is_capture', False) else None
        self._zobrist_hash = self._zobrist.update_hash_move(self._zobrist_hash, mv, piece, captured_piece)

        # Apply move to board
        self._board.apply_move(mv)
        self._current = color.opposite()
        self._age_counter += 1

        # Optimized incremental update
        self._optimized_incremental_update(mv, color, from_coord, to_coord, piece, captured_piece)

    def undo_move(self, mv: Move, color: Color) -> None:
        """Optimized undo with hash restoration."""
        # Restore Zobrist hash (XOR is its own inverse)
        piece = self._board.piece_at(mv.to_coord)
        captured_piece = None
        if getattr(mv, 'is_capture', False):
            captured_type = getattr(mv, 'captured_ptype', None)
            if captured_type is not None:
                captured_piece = Piece(color.opposite(), captured_type)

        self._zobrist_hash = self._zobrist.update_hash_move(self._zobrist_hash, mv, piece, captured_piece)

        # Apply undo to board
        self._undo_move_optimized(mv, color)
        self._current = color
        self._age_counter += 1

        # Optimized incremental undo
        self._optimized_incremental_undo(mv, color)

    # --------------------------------------------------------------------------
    # OPTIMIZED INTERNAL METHODS
    # --------------------------------------------------------------------------
    def _is_simple_move(self, mv: Move, color: Color) -> bool:
        """Ultra-fast check for simple moves that don't affect other pieces."""
        cache_key = (mv.from_coord, color)

        # Check simple move cache
        if cache_key in self._simple_move_cache:
            return True

        to_piece = self._board.piece_at(mv.to_coord)

        # Simple move conditions: no capture, no special effects, no king safety issues
        is_simple = (to_piece is None and
                    not getattr(mv, 'is_capture', False) and
                    not self._affects_other_pieces(mv, color) and
                    not self._cache.is_frozen(mv.from_coord, color))

        if is_simple:
            self._simple_move_cache[cache_key] = []

        return is_simple

    def _affects_other_pieces(self, mv: Move, color: Color) -> bool:
        """Check if move affects other pieces through effects."""
        # Quick check for effect pieces near the move
        affected_squares = {mv.from_coord, mv.to_coord}

        # Check if any effect pieces are nearby
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue

                    check_pos = (mv.to_coord[0] + dx, mv.to_coord[1] + dy, mv.to_coord[2] + dz)
                    if 0 <= check_pos[0] < 9 and 0 <= check_pos[1] < 9 and 0 <= check_pos[2] < 9:
                        piece = self._board.piece_at(check_pos)
                        if piece and piece.ptype in [PieceType.FREEZE_AURA, PieceType.BLACK_HOLE,
                                                   PieceType.WHITE_HOLE, PieceType.GEOMANCER]:
                            return True
        return False

    def _apply_simple_move(self, mv: Move, color: Color) -> None:
        """Ultra-fast path for simple moves."""
        # Direct board update
        piece = self._board.piece_at(mv.from_coord)
        self._board.set_piece(mv.from_coord, None)
        self._board.set_piece(mv.to_coord, piece)

        # Update only the moved piece's legal moves
        self._legal_per_piece.pop(mv.from_coord, None)
        self._legal_per_piece[mv.to_coord] = self._generate_piece_moves_optimized(mv.to_coord)

        # Fast color list update
        self._fast_update_color_list(color, mv)

        # Clear simple move cache if it gets too large
        if len(self._simple_move_cache) > 1000:
            self._simple_move_cache.clear()

    def _generate_piece_moves_optimized(self, coord: Tuple[int, int, int]) -> List[Move]:
        """Enhanced with symmetry checking."""
        # Check symmetry-aware transposition table first
        symmetry_hit = self.symmetry_tt.probe_with_symmetry(
            self._zobrist_hash, self._board
        )

        if symmetry_hit:
            # Convert stored moves back to current coordinate system
            if symmetry_hit.best_move:
                from_coord, to_coord, is_capture, is_promotion = symmetry_hit.best_move.unpack()
                # Apply inverse transformation if needed
                return [Move(from_coord=from_coord, to_coord=to_coord, is_capture=is_capture)]

        # Fall back to normal generation
        return self._generate_piece_moves_standard(coord)

    def _generate_piece_moves_standard(self, coord: Tuple[int, int, int]) -> List[Move]:
        """Optimized move generation with transposition table lookup."""
        from game3d.game.gamestate import GameState

        # Check transposition table first
        tt_entry = self._transposition_table.probe(self._zobrist_hash)
        if tt_entry and tt_entry.best_move:
            from_coord, to_coord, is_capture, is_promotion = tt_entry.best_move.unpack()
            if from_coord == coord:
                # Reconstruct full move from compact representation
                move = Move(from_coord=from_coord, to_coord=to_coord, is_capture=is_capture)
                return [move]

        # Standard generation with optimizations
        piece = self._board.piece_at(coord)
        if piece is None:
            return []

        dispatcher = get_dispatcher(piece.ptype)
        if dispatcher is None:
            return []

        # Use vectorized operations where possible
        if self._can_vectorize(piece.ptype):
            return self._generate_vectorized_moves(coord, piece)

        tmp_state = GameState.__new__(GameState)
        tmp_state.board = self._board
        tmp_state.color = piece.color
        tmp_state.cache = self._cache
        pseudo = dispatcher(tmp_state, *coord)

        # Filter valid moves
        pseudo = [m for m in pseudo if m.from_coord == coord]

        # Fast path for priest-present scenarios
        if self._has_priest[piece.color]:
            return pseudo  # Skip expensive king safety checks

        # King-safety check with branchless operations
        return self._filter_king_safe_moves(pseudo, piece)

    def _can_vectorize(self, piece_type: PieceType) -> bool:
        """Check if piece type supports vectorized move generation."""
        return piece_type in {PieceType.ROOK, PieceType.BISHOP, PieceType.QUEEN,
                             PieceType.XZQUEEN, PieceType.YZQUEEN, PieceType.XYQUEEN}

    def _generate_vectorized_moves(self, coord: Tuple[int, int, int], piece: Piece) -> List[Move]:
        """Vectorized move generation for sliding pieces."""
        moves = []

        # Define directions for different piece types
        directions = self._get_directions_for_piece(piece.ptype)

        # Use numpy for vectorized operations
        start_array = np.array(coord)

        for direction in directions:
            # Generate ray in direction using vectorized operations
            steps = np.arange(1, 10)  # Max steps in 9x9x9 board
            direction_array = np.array(direction)

            # Vectorized position calculation
            targets = start_array + direction_array[:, np.newaxis] * steps[np.newaxis, :]

            # Check bounds vectorized
            valid_mask = np.all((targets >= 0) & (targets < 9), axis=1)

            # Process valid targets
            for i, step_targets in enumerate(targets):
                if not valid_mask[i]:
                    continue

                for target in step_targets:
                    target_tuple = tuple(target)

                    # Check blocking
                    stop, can_land = self._should_stop_at_fast(target_tuple, piece.color)

                    if can_land:
                        target_piece = self._board.piece_at(target_tuple)
                        is_capture = target_piece is not None and target_piece.color != piece.color
                        moves.append(Move(
                            from_coord=coord,
                            to_coord=target_tuple,
                            is_capture=is_capture
                        ))

                    if stop:
                        break

        return moves

    def _get_directions_for_piece(self, piece_type: PieceType) -> List[Tuple[int, int, int]]:
        """Get movement directions for piece type."""
        if piece_type == PieceType.ROOK:
            return [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        elif piece_type == PieceType.BISHOP:
            return [(1,1,0), (1,-1,0), (-1,1,0), (-1,-1,0), (1,0,1), (1,0,-1),
                   (-1,0,1), (-1,0,-1), (0,1,1), (0,1,-1), (0,-1,1), (0,-1,-1)]
        elif piece_type in {PieceType.QUEEN, PieceType.XZQUEEN, PieceType.YZQUEEN, PieceType.XYQUEEN}:
            # All directions for queen
            directions = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        directions.append((dx, dy, dz))
            return directions
        return []

    def _should_stop_at_fast(self, target: Tuple[int, int, int], current_color: Color) -> Tuple[bool, bool]:
        """Fast branchless version of should_stop_at."""
        piece = self._board.piece_at(target)

        has_piece = piece is not None
        is_friendly = has_piece and (piece.color == current_color)
        is_enemy = has_piece and (piece.color != current_color)

        stop_friendly = is_friendly
        stop_enemy = is_enemy

        stop = has_piece
        can_land = (not has_piece) or is_enemy

        return stop, can_land

    def _filter_king_safe_moves(self, moves: List[Move], piece: Piece) -> List[Move]:
        """Branchless king safety checking."""
        # Restore Zobrist hash
        piece = self._board.piece_at(mv.to_coord)
        captured_piece = None
        if getattr(mv, 'is_capture', False):
            captured_type = getattr(mv, 'captured_ptype', None)
            if captured_type is not None:
                captured_piece = Piece(color.opposite(), captured_type)

        self._zobrist_hash = self._zobrist.update_hash_move(self._zobrist_hash, mv, piece, captured_piece)

        # Apply undo to board
        self._board.apply_move(mv)  # This should handle undo internally
        self._current = color

        # Fast incremental update for undo
        self._optimized_incremental_undo(mv, color)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        return {
            'zobrist_hash': self._zobrist_hash,
            'tt_hits': self._transposition_table.hits,
            'tt_misses': self._transposition_table.misses,
            'tt_collisions': self._transposition_table.collisions,
            'simple_move_cache_size': len(self._simple_move_cache),
            'vectorized_cache_size': len(self._vectorized_cache),
            'age_counter': self._age_counter
        }

    # --------------------------------------------------------------------------
    # INTERNAL OPTIMIZATIONS
    # --------------------------------------------------------------------------
    def _is_simple_move(self, mv: Move, color: Color) -> bool:
        """Ultra-fast simple move detection."""
        to_piece = self._board.piece_at(mv.to_coord)

        # Check cache key for simple moves
        cache_key = (mv.from_coord, color)
        if cache_key in self._simple_move_cache:
            return True

        # Simple move: no capture, no special effects, no king safety issues
        is_simple = (to_piece is None and
                      not getattr(mv, 'is_capture', False) and
                      not self._affects_other_pieces(mv, color))

        if is_simple:
            self._simple_move_cache[cache_key] = []

        return is_simple

    def _apply_simple_move(self, mv: Move, color: Color) -> None:
        """Ultra-fast path for simple moves."""
        # Direct board update
        piece = self._board.piece_at(mv.from_coord)
        self._board.set_piece(mv.from_coord, None)
        self._board.set_piece(mv.to_coord, piece)

        # Minimal cache updates
        self._legal_per_piece.pop(mv.from_coord, None)
        self._legal_per_piece[mv.to_coord] = self._generate_piece_moves(mv.to_coord)

        # Fast color list update
        self._fast_update_color_list(color, mv)

    def _optimized_incremental_update(self, mv: Move, color: Color, from_coord: Tuple[int, int, int],
                                    to_coord: Tuple[int, int, int], piece: Piece,
                                    captured_piece: Optional[Piece]) -> None:
        """Optimized incremental update with dependency tracking."""
        self._refresh_counts()

        # Use dependency tracking for better incremental updates
        affected_squares = self._get_affected_squares_fast(mv, color)

        # Batch update affected pieces
        self._batch_update_pieces(affected_squares, color)

        # Update occupancy cache
        self._cache.occupancy.rebuild(self._board)

        # Update effect caches based on piece type
        self._update_effect_caches(mv, color, piece, captured_piece)

    def _get_affected_squares_fast(self, mv: Move, color: Color) -> Set[Tuple[int, int, int]]:
        """Fast affected squares calculation using bitboard keys."""
        affected = {mv.from_coord, mv.to_coord}

        # Add king positions for both colors
        for col in (Color.WHITE, Color.BLACK):
            king_pos = self._find_king(col)
            if king_pos:
                affected.add(king_pos)

        # Add squares affected by special pieces
        if piece := self._board.piece_at(mv.to_coord):
            if piece.ptype in self._dependency_graph:
                # Add squares that might be affected by this piece type
                affected.update(self._get_influence_squares(piece.ptype, mv.to_coord))

        return affected

    def _batch_update_pieces(self, affected_squares: Set[Tuple[int, int, int]], color: Color) -> None:
        """Batch update affected pieces using parallel generation."""
        # Clear old entries
        for coord in affected_squares:
            self._legal_per_piece.pop(coord, None)

        # Get pieces that need updating
        pieces_to_update = []
        for coord in affected_squares:
            piece = self._board.piece_at(coord)
            if piece and piece.color == color:
                pieces_to_update.append(coord)

        # Use parallel generation for large sets
        if len(pieces_to_update) > 4:
            moves = self._parallel_generator.generate_moves_parallel(
                self._board, color, pieces_to_update, self._cache)
            # Distribute moves back to pieces
            for move in moves:
                if move.from_coord not in self._legal_per_piece:
                    self._legal_per_piece[move.from_coord] = []
                self._legal_per_piece[move.from_coord].append(move)
        else:
            # Sequential for small sets
            for coord in pieces_to_update:
                self._legal_per_piece[coord] = self._generate_piece_moves(coord)

        self._rebuild_color_lists()

    def _fast_update_color_list(self, color: Color, mv: Move) -> None:
        """Fast color list update for simple moves."""
        # Remove moves from old position
        old_moves = [m for m in self._legal_by_color[color] if m.from_coord != mv.from_coord]

        # Add moves from new position
        new_moves = self._legal_per_piece.get(mv.to_coord, [])

        self._legal_by_color[color] = old_moves + new_moves

    def _generate_piece_moves(self, coord: Tuple[int, int, int]) -> List[Move]:
        """Ultra-optimized piece move generation."""
        from game3d.game.gamestate import GameState

        piece = self._board.piece_at(coord)
        if not piece:
            return []

        # Check simple move cache first
        cache_key = (coord, piece.color)
        if cache_key in self._simple_move_cache:
            return self._simple_move_cache[cache_key]

        dispatcher = get_dispatcher(piece.ptype)
        if not dispatcher:
            return []

        # Create minimal game state
        tmp_state = GameState.__new__(GameState)
        tmp_state.board = self._board
        tmp_state.color = piece.color
        tmp_state.cache = self._cache

        # Generate moves
        pseudo = dispatcher(tmp_state, *coord)
        pseudo = [m for m in pseudo if m.from_coord == coord]  # Paranoid filter

        # Apply effect filters
        if self._cache.is_frozen(coord, piece.color):
            pseudo = []

        # King safety check with priest optimization
        if not self._has_priest[piece.color]:
            pseudo = self._filter_legal_moves(pseudo, piece.color)

        # Cache result for simple moves
        if len(pseudo) > 0 and self._is_simple_position():
            self._simple_move_cache[cache_key] = pseudo

        return pseudo

    def _filter_legal_moves(self, moves: List[Move], color: Color) -> List[Move]:
        """Filter moves for king safety without full board cloning."""
        legal_moves = []

        for mv in moves:
            # Quick check for friendly fire
            victim = self._board.piece_at(mv.to_coord)
            if victim and victim.color == color:
                continue

            # Fast king safety check
            if self._is_king_safe_after_move(mv, color):
                legal_moves.append(mv)

        return legal_moves

    def _is_king_safe_after_move(self, mv: Move, color: Color) -> bool:
        """Ultra-fast king safety check."""
        moving_piece = self._board.piece_at(mv.from_coord)
        victim_piece = self._board.piece_at(mv.to_coord)

        # Make move temporarily
        self._board.set_piece(mv.from_coord, None)
        self._board.set_piece(mv.to_coord, moving_piece)

        try:
            # Fast check detection
            king_pos = self._king_pos[color]
            if not king_pos:
                return True

            return not king_in_check(self._board, color, color)
        finally:
            # Restore board
            self._board.set_piece(mv.from_coord, moving_piece)
            self._board.set_piece(mv.to_coord, victim_piece)

    def _is_simple_position(self) -> bool:
        """Check if current position is simple (no complex effects)."""
        return (self._age_counter < 10 and
                len(self._cache._effect) < 5 and
                all(count == 0 for count in self._priest_count.values()))

    def _get_influence_squares(self, piece_type: PieceType, pos: Tuple[int, int, int]) -> Set[Tuple[int, int, int]]:
        """Get squares influenced by a piece type."""
        # This is a simplified version - you'd implement actual influence patterns
        influence = set()
        x, y, z = pos

        # Add squares in a 3x3x3 cube around the piece
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == dy == dz == 0:
                        continue
                    new_pos = (x + dx, y + dy, z + dz)
                    if in_bounds(new_pos):
                        influence.add(new_pos)

        return influence

    def _affects_other_pieces(self, mv: Move, color: Color) -> bool:
        """Check if move affects other pieces."""
        # Simplified check - you'd implement full dependency analysis
        piece = self._board.piece_at(mv.from_coord)
        if not piece:
            return True  # Safe default

        return piece.ptype in self._dependency_graph

    def _update_effect_caches(self, mv: Move, color: Color, piece: Piece, captured_piece: Optional[Piece]) -> None:
        """Update effect caches based on piece interactions."""
        # This would call the appropriate effect cache update methods
        # Implementation depends on your specific effect cache API
        pass

    def _optimized_incremental_undo(self, mv: Move, color: Color) -> None:
        """Optimized undo with smart cache restoration."""
        # Similar to apply but in reverse
        affected_squares = {mv.from_coord, mv.to_coord}
        for col in (Color.WHITE, Color.BLACK):
            king_pos = self._find_king(col)
            if king_pos:
                affected_squares.add(king_pos)

        # Batch update
        self._batch_update_pieces(affected_squares, color)

    def _find_king(self, color: Color) -> Optional[Tuple[int, int, int]]:
        """Ultra-fast king finding with caching."""
        if self._king_pos[color]:
            king_piece = self._board.piece_at(self._king_pos[color])
            if (king_piece and king_piece.color == color and
                king_piece.ptype == PieceType.KING):
                return self._king_pos[color]

        # Fallback scan
        for coord, piece in self._board.list_occupied():
            if piece.color == color and piece.ptype == PieceType.KING:
                self._king_pos[color] = coord
                return coord

        self._king_pos[color] = None
        return None

    def _refresh_counts(self) -> None:
        """Ultra-fast count refresh with single board scan."""
        # Reset all counts
        for color in Color:
            self._priest_count[color] = 0
            self._king_pos[color] = None
            self._has_priest[color] = False

        # Single scan through board
        for coord, piece in self._board.list_occupied():
            if piece.ptype == PieceType.PRIEST:
                self._priest_count[piece.color] += 1
            elif piece.ptype == PieceType.KING:
                self._king_pos[piece.color] = coord

        # Update has_priest flags
        for color in Color:
            self._has_priest[color] = self._priest_count[color] > 0

    def _rebuild_color_lists(self) -> None:
        """Ultra-fast color list rebuild using list comprehension."""
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
        """Ultra-fast full rebuild with parallel generation."""
        self._refresh_counts()
        self._legal_per_piece.clear()

        # Get all pieces for current color
        pieces_to_generate = []
        for coord, piece in self._board.list_occupied():
            if piece.color == self._current:
                pieces_to_generate.append(coord)

        # Use parallel generation for large sets
        if len(pieces_to_generate) > 8:
            all_moves = self._parallel_generator.generate_moves_parallel(
                self._board, self._current, pieces_to_generate, self._cache)

            # Group moves by piece
            for move in all_moves:
                if move.from_coord not in self._legal_per_piece:
                    self._legal_per_piece[move.from_coord] = []
                self._legal_per_piece[move.from_coord].append(move)
        else:
            # Sequential for small sets
            for coord in pieces_to_generate:
                self._legal_per_piece[coord] = self._generate_piece_moves(coord)

        self._rebuild_color_lists()

# ==============================================================================
# FACTORY FUNCTION FOR BACKWARD COMPATIBILITY
# ==============================================================================

def create_optimized_move_cache(board: Board, current: Color, cache_manager: CacheManager) -> OptimizedMoveCache:
    """Factory function to create optimized move cache."""
    return OptimizedMoveCache(board, current, cache_manager)

# ==============================================================================
# PERFORMANCE MONITORING
# ==============================================================================

class CachePerformanceMonitor:
    """Monitor cache performance and provide optimization suggestions."""

    def __init__(self, move_cache: OptimizedMoveCache):
        self.move_cache = move_cache
        self.start_time = time.time()
        self.move_counts = []
        self.cache_hits = []
        self.cache_misses = []

    def record_search(self, move_count: int):
        """Record statistics from a search iteration."""
        self.move_counts.append(move_count)
        stats = self.move_cache.get_stats()
        self.cache_hits.append(stats['tt_hits'])
        self.cache_misses.append(stats['tt_misses'])

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report with optimization suggestions."""
        stats = self.move_cache.get_stats()

        report = {
            'total_moves_generated': sum(self.move_counts),
            'average_moves_per_search': np.mean(self.move_counts) if self.move_counts else 0,
            'tt_hit_rate': stats['tt_hits'] / max(1, stats['tt_hits'] + stats['tt_misses']),
            'cache_efficiency': len(self.move_cache._simple_move_cache) / max(1, sum(len(moves) for moves in self.move_cache._legal_per_piece.values())),
            'optimization_suggestions': []
        }

        # Add optimization suggestions
        if report['tt_hit_rate'] < 0.3:
            report['optimization_suggestions'].append("Consider increasing transposition table size")

        if len(self.move_cache._simple_move_cache) < 100:
            report['optimization_suggestions'].append("Simple move cache underutilized - check simple move detection")

        if stats['tt_collisions'] > stats['tt_hits'] * 0.01:
            report['optimization_suggestions'].append("High collision rate - consider better hash function or larger table")

        return report
