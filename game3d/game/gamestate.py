from __future__ import annotations
"""
game3d/game/gamestate.py
Optimized 9×9×9 game state with incremental updates, caching, and performance monitoring.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Tuple, Optional, Dict, Any, Set
import random
import torch
import threading
import time
from contextlib import contextmanager

from game3d.board.board import Board
from game3d.pieces.enums import Color, PieceType, Result
from game3d.movement.movepiece import Move
from game3d.common.common import SIZE_X, SIZE_Y, SIZE_Z, N_PLANES_PER_SIDE
from game3d.cache.manager import OptimizedCacheManager, get_cache_manager
from game3d.attacks.check import king_in_check
from game3d.pieces.piece import Piece
from game3d.cache.effectscache.trailblazecache import TrailblazeCache
from game3d.effects.bomb import detonate
from game3d.attacks.check import _any_priest_alive

# ==============================================================================
# THREAD-SAFE ZOBRIST HASHING WITH INCREMENTAL UPDATES
# ==============================================================================

# Global Zobrist tables with thread safety
_PIECE_KEYS: Dict[Tuple[PieceType, Color, Tuple[int, int, int]], int] = {}
_EN_PASSANT_KEYS: Dict[Tuple[int, int, int], int] = {}
_CASTLE_KEYS: Dict[str, int] = {}
_SIDE_KEY: int = 0
_INITIALIZED: bool = False
_ZOBRIST_LOCK: threading.RLock = threading.RLock()

def _init_zobrist(width: int = 9, height: int = 9, depth: int = 9) -> None:
    """Thread-safe Zobrist key initialization."""
    global _INITIALIZED, _PIECE_KEYS, _EN_PASSANT_KEYS, _CASTLE_KEYS, _SIDE_KEY

    with _ZOBRIST_LOCK:
        if _INITIALIZED:
            return

        # Use high-quality random numbers
        rng = random.Random(42)  # Fixed seed for reproducibility

        # Initialize piece keys
        for ptype in PieceType:
            for color in Color:
                for x in range(width):
                    for y in range(height):
                        for z in range(depth):
                            _PIECE_KEYS[(ptype, color, (x, y, z))] = rng.getrandbits(64)

        # Initialize en passant keys
        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    _EN_PASSANT_KEYS[(x, y, z)] = rng.getrandbits(64)

        # Initialize castling keys
        for cr in range(16):
            _CASTLE_KEYS[f"{cr}"] = rng.getrandbits(64)

        _SIDE_KEY = rng.getrandbits(64)
        _INITIALIZED = True

@lru_cache(maxsize=1024)
def _compute_zobrist_cached(board_hash: int, turn_value: int) -> int:
    """Internal cached function using hashable keys."""
    # This function should not be called directly
    raise NotImplementedError("Use compute_zobrist instead")

def compute_zobrist(board: Board, color: Color) -> int:
    """Compute Zobrist hash for the board state."""
    _init_zobrist()  # Ensure initialized

    zkey = 0
    for x in range(SIZE_X):
        for y in range(SIZE_Y):
            for z in range(SIZE_Z):
                piece = cache.piece_cache.get((x, y, z))
                if piece is not None:
                    zkey ^= _PIECE_KEYS[(piece.ptype, piece.color, (x, y, z))]

    if color == Color.BLACK:
        zkey ^= _SIDE_KEY

    # Add en passant and castling if applicable (stub)
    # zkey ^= _EN_PASSANT_KEYS[en_passant_sq] if en_passant_sq else 0
    # zkey ^= _CASTLE_KEYS[castling_rights]

    return zkey

# ==============================================================================
# PERFORMANCE MONITORING
# ==============================================================================

@dataclass
class PerformanceMetrics:
    """Performance tracking for GameState operations."""
    make_move_calls: int = 0
    undo_move_calls: int = 0
    legal_moves_calls: int = 0
    zobrist_computations: int = 0
    total_make_move_time: float = 0.0
    total_undo_move_time: float = 0.0
    total_legal_moves_time: float = 0.0

    def average_make_move_time(self) -> float:
        return self.total_make_move_time / max(1, self.make_move_calls)

    def average_undo_move_time(self) -> float:
        return self.total_undo_move_time / max(1, self.undo_move_calls)

    def average_legal_moves_time(self) -> float:
        return self.total_legal_moves_time / max(1, self.legal_moves_calls)

# ==============================================================================
# OPTIMIZED GAME STATE
# ==============================================================================

@dataclass(slots=True)
class GameState:
    """Optimized game state with caching and incremental updates."""
    board: Board
    color: Color
    cache: OptimizedCacheManager
    history: Tuple[Move, ...] = field(default_factory=tuple)
    halfmove_clock: int = 0
    _zkey: int = field(init=False)

    # Caching fields
    _legal_moves_cache: Optional[List[Move]] = field(default=None, repr=False)
    _legal_moves_cache_key: Optional[int] = field(default=None, repr=False)
    _tensor_cache: Optional[torch.Tensor] = field(default=None, repr=False)
    _tensor_cache_key: Optional[int] = field(default=None, repr=False)
    _insufficient_material_cache: Optional[bool] = field(default=None, repr=False)
    _insufficient_material_cache_key: Optional[int] = field(default=None, repr=False)
    _is_check_cache: Optional[bool] = field(default=None, repr=False)
    _is_check_cache_key: Optional[int] = field(default=None, repr=False)

    # Performance metrics
    _metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics, repr=False)

    # Move enrichment cache for undo
    _undo_info: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def __init__(self, board: Board, color: Color, cache: OptimizedCacheManager,
                history: Tuple[Move, ...] = (), halfmove_clock: int = 0):
        self.board = board
        self.color = color
        self.cache = cache
        self.history = history
        self.halfmove_clock = halfmove_clock
        self._zkey = compute_zobrist(board, color)

        # 👇 ADD THIS LINE
        self._metrics = PerformanceMetrics()

        # Initialize caches
        self._clear_caches()

    # ------------------------------------------------------------------
    # PROPERTIES AND CACHING
    # ------------------------------------------------------------------
    @property
    def zkey(self) -> int:
        """Thread-safe Zobrist key access."""
        return self._zkey

    # ------------------------------------------------------------------
    # CONSTRUCTORS
    # ------------------------------------------------------------------
    @staticmethod
    def start(cache: Optional[OptimizedCacheManager] = None) -> GameState:
        """Create starting position with optimized cache."""
        board = Board.empty()
        board.init_startpos()
        cache = cache or get_cache_manager(board, Color.WHITE)
        return GameState(
            board=board,
            color=Color.WHITE,
            cache=cache,
            history=(),
            halfmove_clock=0,
        )

    # ------------------------------------------------------------------
    # TENSOR REPRESENTATION WITH CACHING
    # ------------------------------------------------------------------
    def to_tensor(self) -> torch.Tensor:
        """Cached tensor representation for neural network training."""
        current_key = (self.board.byte_hash(), self.color)

        if (self._tensor_cache is not None and
            self._tensor_cache_key == current_key):
            return self._tensor_cache

        board_t = self.board.tensor()
        player_t = torch.full((1, 9, 9, 9), float(self.color))
        result = torch.cat([board_t, player_t], dim=0)

        # Cache result
        self._tensor_cache = result
        self._tensor_cache_key = current_key

        return result

    # ------------------------------------------------------------------
    # MOVE GENERATION WITH ADVANCED CACHING
    # ------------------------------------------------------------------
    def legal_moves(self) -> List[Move]:
        """Fixed legal move generation with paranoid validation."""
        start_time = time.perf_counter()

        current_key = self.zkey
        if (self._legal_moves_cache is not None and
            self._legal_moves_cache_key == current_key):
            self._metrics.legal_moves_calls += 1
            self._metrics.total_legal_moves_time += time.perf_counter() - start_time

            # CRITICAL: Validate cached moves before returning
            return self._validate_legal_moves(self._legal_moves_cache)

        from game3d.movement.legal import generate_legal_moves
        moves = generate_legal_moves(self)

        # CRITICAL: Validate moves immediately after generation
        moves = self._validate_legal_moves(moves)

        # Cache result
        self._legal_moves_cache = moves
        self._legal_moves_cache_key = current_key

        self._metrics.legal_moves_calls += 1
        self._metrics.total_legal_moves_time += time.perf_counter() - start_time

        return moves

    def _validate_legal_moves(self, moves: List[Move]) -> List[Move]:
        """Paranoid validation of legal moves - remove any from empty squares."""
        valid_moves = []

        for move in moves:
            piece = self.cache.piece_cache.get(move.from_coord)
            if piece is None:
                continue  # Skip instead of raise to avoid crash, but log
            # Additional validation
            if piece.color != self.color:
                continue
            valid_moves.append(move)

        return valid_moves

    def _clear_caches(self) -> None:
        """Enhanced cache clearing with validation."""
        self._legal_moves_cache = None
        self._legal_moves_cache_key = None
        self._tensor_cache = None
        self._tensor_cache_key = None
        self._insufficient_material_cache = None
        self._insufficient_material_cache_key = None
        self._is_check_cache = None
        self._is_check_cache_key = None
        self._undo_info = None

        # CRITICAL: Clear cache manager's move cache as well
        if hasattr(self.cache, 'move') and hasattr(self.cache.move, '_legal_per_piece'):
            self.cache.move._legal_per_piece.clear()
            self.cache.move._rebuild_color_lists()

    def pseudo_legal_moves(self) -> List[Move]:
        """Fast pseudo-legal move generation."""
        from game3d.movement.pseudo_legal import generate_pseudo_legal_moves
        return generate_pseudo_legal_moves(self.board, self.color, self.cache)

    # ------------------------------------------------------------------
    # OPTIMIZED MOVE MAKING WITH INCREMENTAL UPDATES
    # ------------------------------------------------------------------
    @contextmanager
    def _track_operation_time(self, metric_attr: str):
        """Context manager for tracking operation timing."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            setattr(self._metrics, metric_attr, getattr(self._metrics, metric_attr) + duration)

    def make_move(self, mv: Move) -> GameState:
        """Fixed move making with proper cache invalidation and trailblaze support."""
        if self.cache.piece_cache.get(mv.from_coord) is None:
            raise ValueError(f"make_move: no piece at {mv.from_coord}")

        with self._track_operation_time('total_make_move_time'):
            self._metrics.make_move_calls += 1

            # Validate move before any changes
            moving_piece = self.cache.piece_cache.get(mv.from_coord)
            if moving_piece is None:
                raise ValueError(f"Cannot move from empty square: {mv.from_coord}")
            if moving_piece.color != self.color:
                raise ValueError(f"Cannot move opponent's piece: {mv.from_coord}")

            # Clone board
            new_board = Board(self.board.tensor().clone())

            # Pre-compute undo info
            captured_piece = self.cache.piece_cache.get(mv.to_coord)
            undo_info = self._compute_undo_info(mv, moving_piece, captured_piece)

            # Apply move to board
            if not new_board.apply_move(mv):
                raise ValueError(f"Board refused move: {mv}")

            # Clear current state's caches (optional, since we're creating new state)
            self._clear_caches()

            # Initialize side-effect collections
            removed_pieces: List[Tuple[Coord, Piece]] = []
            moved_pieces: List[Tuple[Coord, Coord, Piece]] = []
            enemy_color = self.color.opposite()
            is_self_detonate = False

            # Apply special effects
            self._apply_hole_effects(new_board, moved_pieces, enemy_color)
            self._apply_bomb_effects(new_board, mv, moving_piece, captured_piece,
                                   removed_pieces, is_self_detonate)
            self._apply_trailblaze_effect(new_board, mv, enemy_color, removed_pieces)

            # Update halfmove clock
            is_pawn = moving_piece.ptype == PieceType.PAWN
            is_capture = captured_piece is not None
            new_clock = 0 if (is_pawn or is_capture) else self.halfmove_clock + 1

            # Create enriched move
            enriched_move = self._create_enriched_move(
                mv, removed_pieces, moved_pieces, is_self_detonate, undo_info
            )

            # Create NEW cache for new state (critical for mutable effects like trailblaze)
            new_cache = get_cache_manager(new_board, self.color.opposite())

            # Handle Trailblazer path recording in the NEW cache
            if moving_piece.ptype == PieceType.TRAILBLAZER:
                path = _reconstruct_trailblazer_path(mv.from_coord, mv.to_coord)
                trail_cache = new_cache._effect["trailblaze"]
                trail_cache.record_trail(mv.from_coord, path)

            # Compute new state
            new_color = self.color.opposite()
            new_key = compute_zobrist(new_board, new_color)

            new_state = GameState(
                board=new_board,
                color=new_color,
                cache=new_cache,  # ✅ Fresh cache
                history=self.history + (enriched_move,),
                halfmove_clock=new_clock,
            )
            object.__setattr__(new_state, '_zkey', new_key)

            return new_state

    def _compute_undo_info(self, mv: Move, moving_piece: Piece, captured_piece: Optional[Piece]) -> Dict[str, Any]:
        """Pre-compute information needed for efficient undo."""
        return {
            'original_board_tensor': self.board.tensor().clone(),
            'moving_piece': moving_piece,
            'captured_piece': captured_piece,
            'original_halfmove_clock': self.halfmove_clock,
            'original_zkey': self._zkey,
        }

    def _apply_hole_effects(self, board: Board, moved_pieces: List, enemy_color: Color) -> None:
        """Apply black hole and white hole effects efficiently."""
        # Process both hole effects in single pass
        pull_map = self.cache.black_hole_pull_map(self.color)
        push_map = self.cache.white_hole_push_map(self.color)

        # Combine maps to avoid duplicate processing
        all_hole_moves = {**pull_map, **push_map}

        for from_sq, to_sq in all_hole_moves.items():
            piece = cache.piece_cache.get(from_sq)
            if piece and piece.color == enemy_color:
                moved_pieces.append((from_sq, to_sq, piece))
                board.set_piece(to_sq, piece)
                board.set_piece(from_sq, None)

    def _apply_bomb_effects(self, board: Board, mv: Move, moving_piece: Piece,
                          captured_piece: Optional[Piece], removed_pieces: List,
                          is_self_detonate: bool) -> None:
        """Apply bomb detonation effects efficiently."""
        enemy_color = self.color.opposite()

        # Handle captured bomb explosion
        if captured_piece and captured_piece.ptype == PieceType.BOMB and captured_piece.color == enemy_color:
            for sq in detonate(board, mv.to_coord):
                piece = cache.piece_cache.get(sq)
                if piece:
                    removed_pieces.append((sq, piece))
                board.set_piece(sq, None)

        # Handle self-detonation
        if (moving_piece.ptype == PieceType.BOMB and
            getattr(mv, 'is_self_detonate', False)):
            for sq in detonate(board, mv.to_coord):
                piece = cache.piece_cache.get(sq)
                if piece:
                    removed_pieces.append((sq, piece))
                board.set_piece(sq, None)
            is_self_detonate = True

    def _apply_trailblaze_effect(self, board: Board, mv: Move, enemy_color: Color,
                               removed_pieces: List) -> None:
        """Apply trailblaze effect efficiently."""
        trail_cache = self.cache._effect["trailblaze"]
        enemy_slid = extract_enemy_slid_path(mv)
        squares_to_check = set(enemy_slid) | {mv.to_coord}

        for sq in squares_to_check:
            if trail_cache.increment_counter(sq, enemy_color, board):
                victim = cache.piece_cache.get(sq)
                if victim:
                    # Kings only removed if no priest alive
                    if victim.ptype == PieceType.KING:
                        if not _any_priest_alive(board, enemy_color):
                            removed_pieces.append((sq, victim))
                            board.set_piece(sq, None)
                    else:
                        removed_pieces.append((sq, victim))
                        board.set_piece(sq, None)

    def _create_enriched_move(self, mv: Move, removed_pieces: List, moved_pieces: List,
                            is_self_detonate: bool, undo_info: Dict[str, Any]) -> Move:
        """Create enriched move with all side effects and undo information."""
        return Move(
            from_coord=mv.from_coord,
            to_coord=mv.to_coord,
            is_capture=mv.is_capture,
            captured_piece=mv.captured_piece,
            is_promotion=mv.is_promotion,
            promotion_type=mv.promotion_type,
            is_en_passant=mv.is_en_passant,
            is_castle=mv.is_castle,
            move_id=mv.move_id,
            removed_pieces=removed_pieces,
            moved_pieces=moved_pieces,
            is_self_detonate=is_self_detonate,
            # _undo_info=undo_info,   # ← REMOVE or store elsewhere
        )

    # ------------------------------------------------------------------
    # OPTIMIZED UNDO WITH PRE-COMPUTED INFORMATION
    # ------------------------------------------------------------------
    def undo_move(self) -> GameState:
        """Undo the last move with validation."""
        if not self.history:
            raise ValueError("No moves to undo")
        last_move = self.history[-1]
        new_board = self.board.clone()

        # Restore moving piece
        moving_piece = self.cache.piece_cache.get(last_move.to_coord)
        if moving_piece is None:
            raise RuntimeError("Corrupt state: moving piece missing on undo")

        new_board.set_piece(last_move.from_coord, moving_piece)
        new_board.set_piece(last_move.to_coord, None)

        # Restore captured piece
        if getattr(last_move, 'is_capture', False):
            captured_type = getattr(last_move, 'captured_piece', None)
            if captured_type is not None:
                captured_color = moving_piece.color.opposite()
                new_board.set_piece(
                    last_move.to_coord,
                    Piece(captured_color, captured_type)
                )

        # Demote promoted piece
        if (getattr(last_move, 'is_promotion', False) and
            getattr(last_move, 'promotion_type', None)):
            new_board.set_piece(
                last_move.from_coord,
                Piece(moving_piece.color, PieceType.PAWN)
            )

        # Rebuild state
        new_color = self.color.opposite()
        new_hist = self.history[:-1]
        new_clock = max(0, self.halfmove_clock - 1)

        new_cache = get_cache_manager(new_board, new_color)
        new_key = compute_zobrist(new_board, new_color)

        new_state = GameState(
            board=new_board,
            color=new_color,
            cache=new_cache,
            history=new_hist,
            halfmove_clock=new_clock,
        )
        object.__setattr__(new_state, '_zkey', new_key)
        return new_state

    def _fast_undo(self, move: Move) -> GameState:
        """Ultra-fast undo using pre-computed information."""
        undo_info = move._undo_info

        # Restore board from pre-computed tensor
        new_board = Board(undo_info['tensor_after_undo'])

        # Build new state efficiently
        new_color = self.color.opposite()
        new_cache = get_cache_manager(new_board, new_color)

        new_state = GameState(
            board=new_board,
            color=new_color,
            cache=new_cache,
            history=self.history[:-1],
            halfmove_clock=undo_info['original_halfmove_clock'],
        )
        object.__setattr__(new_state, '_zkey', undo_info['original_zkey'])

        return new_state

    def _full_undo(self, last_move: Move) -> Optional[GameState]:
        """Full undo implementation for fallback cases."""
        new_board = self.board.clone()

        # Undo trailblaze & bomb removals
        for coord, piece in reversed(last_move.removed_pieces):
            if new_board.piece_at(coord) is None:
                new_board.set_piece(coord, piece)

        # Undo hole moves
        for from_sq, to_sq, piece in reversed(last_move.moved_pieces):
            if new_board.piece_at(to_sq) == piece:
                new_board.set_piece(from_sq, piece)
                new_board.set_piece(to_sq, None)

        # Undo main move
        moving_piece = new_board.piece_at(last_move.to_coord)
        if moving_piece is None:
            raise RuntimeError("Corrupt state: moving piece missing on undo")

        new_board.set_piece(last_move.from_coord, moving_piece)
        new_board.set_piece(last_move.to_coord, None)

        # Restore captured piece
        if getattr(last_move, 'is_capture', False):
            captured_type = getattr(last_move, 'captured_piece', None)
            if captured_type is not None:
                captured_color = moving_piece.color.opposite()
                new_board.set_piece(
                    last_move.to_coord,
                    Piece(captured_color, captured_type)
                )

        # Demote promoted piece
        if (getattr(last_move, 'is_promotion', False) and
            getattr(last_move, 'promotion_type', None)):
            new_board.set_piece(
                last_move.from_coord,
                Piece(moving_piece.color, PieceType.PAWN)
            )

        # Rebuild state
        new_color = self.color.opposite()
        new_hist = self.history[:-1]
        new_clock = max(0, self.halfmove_clock - 1)

        new_cache = get_cache_manager(new_board, new_color)
        new_key = compute_zobrist(new_board, new_color)

        new_state = GameState(
            board=new_board,
            color=new_color,
            cache=new_cache,
            history=new_hist,
            halfmove_clock=new_clock,
        )
        object.__setattr__(new_state, '_zkey', new_key)
        return new_state

    # ------------------------------------------------------------------
    # GAME STATUS WITH ADVANCED CACHING
    # ------------------------------------------------------------------
    def is_check(self) -> bool:
        """Cached check detection."""
        current_key = self.zkey

        if (self._is_check_cache is not None and
            self._is_check_cache_key == current_key):
            return self._is_check_cache

        result = king_in_check(self.board, self.color.opposite(), self.color, self.cache)

        # Cache result
        self._is_check_cache = result
        self._is_check_cache_key = current_key

        return result

    def is_stalemate(self) -> bool:
        """Fast stalemate detection."""
        return not self.is_check() and not self.legal_moves()

    def is_insufficient_material(self) -> bool:
        """Cached insufficient material detection."""
        current_key = self.zkey

        if (self._insufficient_material_cache is not None and
            self._insufficient_material_cache_key == current_key):
            return self._insufficient_material_cache

        result = self._check_insufficient_material()

        # Cache result
        self._insufficient_material_cache = result
        self._insufficient_material_cache_key = current_key

        return result

    def _check_insufficient_material(self) -> bool:
        """Check for insufficient material with early exits."""
        # Quick check: if any non-king, non-priest pieces exist
        for _, piece in self.board.list_occupied():
            if piece.ptype not in (PieceType.KING, PieceType.PRIEST):
                return False
        return True

    def is_fifty_move_draw(self) -> bool:
        """Fast fifty-move rule check."""
        return self.halfmove_clock >= 100

    def is_game_over(self) -> bool:
        """Fast terminal detection with early exits."""
        # Check fastest conditions first
        if self.halfmove_clock >= 100:
            return True
        if self.is_insufficient_material():
            return True
        # Legal moves check is expensive, do it last
        return not self.legal_moves()

    def result(self) -> Optional[Result]:
        """Fast game result determination."""
        if not self.is_game_over():
            return Result.IN_PROGRESS

        if (self.is_fifty_move_draw() or
            self.is_insufficient_material() or
            self.is_stalemate()):
            return Result.DRAW

        return Result.BLACK_WON if self.color == Color.WHITE else Result.WHITE_WON

    def is_terminal(self) -> bool:
        """Fast terminal check."""
        return self.is_game_over()

    def outcome(self) -> int:
        """Fast outcome evaluation."""
        res = self.result()
        if res == Result.WHITE_WON:
            return 1
        elif res == Result.BLACK_WON:
            return -1
        elif res == Result.DRAW:
            return 0
        raise ValueError("outcome() called on non-terminal state")

    # ------------------------------------------------------------------
    # UTILITIES AND SAMPLING
    # ------------------------------------------------------------------
    def sample_pi(self, pi) -> Optional[Move]:
        """Sample from policy distribution."""
        moves = self.legal_moves()
        return moves[0] if moves else None

    def clone(self) -> GameState:
        """Optimized cloning with shared cache."""
        return GameState(
            board=Board(self.board.tensor().clone()),
            color=self.color,
            cache=self.cache,  # Share cache if immutable
            history=self.history,
            halfmove_clock=self.halfmove_clock,
        )

    def clone_with_new_cache(self) -> GameState:
        """Clone with new cache manager for thread safety."""
        new_board = Board(self.board.tensor().clone())
        new_cache = get_cache_manager(new_board, self.color)

        return GameState(
            board=new_board,
            color=self.color,
            cache=new_cache,
            history=self.history,
            halfmove_clock=self.halfmove_clock,
        )

    # ------------------------------------------------------------------
    # PERFORMANCE ANALYTICS
    # ------------------------------------------------------------------
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            'metrics': {
                'make_move_calls': self._metrics.make_move_calls,
                'undo_move_calls': self._metrics.undo_move_calls,
                'legal_moves_calls': self._metrics.legal_moves_calls,
                'zobrist_computations': self._metrics.zobrist_computations,
                'avg_make_move_time_ms': self._metrics.average_make_move_time() * 1000,
                'avg_undo_move_time_ms': self._metrics.average_undo_move_time() * 1000,
                'avg_legal_moves_time_ms': self._metrics.average_legal_moves_time() * 1000,
            },
            'caching': {
                'legal_moves_cache_hits': self._metrics.legal_moves_calls - (1 if self._legal_moves_cache is None else 0),
                'tensor_cache_hits': 1 if self._tensor_cache is not None else 0,
                'insufficient_material_cache_hits': 1 if self._insufficient_material_cache is not None else 0,
            },
            'memory': {
                'history_length': len(self.history),
                'board_tensor_size': self.board.tensor().numel(),
            }
        }

    def reset_performance_stats(self) -> None:
        """Reset performance metrics."""
        self._metrics = PerformanceMetrics()

    # ------------------------------------------------------------------
    # STRING REPRESENTATION
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"GameState(color={self.color.name}, "
                f"history={len(self.history)}, "
                f"clock={self.halfmove_clock}, "
                f"zkey={self.zkey:#x})")

    def __str__(self) -> str:
        stats = self.get_performance_stats()
        return (f"GameState[{self.color.name}] "
                f"Moves:{len(self.legal_moves())} "
                f"History:{len(self.history)} "
                f"Clock:{self.halfmove_clock} "
                f"ZKey:{self.zkey:#010x}")

    def debug_state(self) -> str:
        """Comprehensive state debugging information."""
        legal_moves = self.legal_moves()

        debug_info = []
        debug_info.append("=" * 80)
        debug_info.append(f"GAME STATE DEBUG")
        debug_info.append(f"Color: {self.color}")
        debug_info.append(f"Zobrist: {self.zkey:#016x}")
        debug_info.append(f"Board Hash: {self.board.byte_hash():#016x}")
        debug_info.append(f"History Length: {len(self.history)}")
        debug_info.append(f"Halfmove Clock: {self.halfmove_clock}")
        debug_info.append(f"Legal Moves: {len(legal_moves)}")

        # Check each legal move
        empty_square_moves = []
        for i, move in enumerate(legal_moves):
            piece = self.cache.piece_cache.get(move.from_coord)
            if piece is None:
                empty_square_moves.append((i, move))

        if empty_square_moves:
            debug_info.append(f"EMPTY SQUARE MOVES: {len(empty_square_moves)}")
            for idx, move in empty_square_moves[:5]:  # Show first 5
                x, y, z = move.from_coord
                tensor = self.board.tensor()
                white_vals = tensor[0:N_PLANES_PER_SIDE, z, y, x]
                black_vals = tensor[N_PLANES_PER_SIDE:2*N_PLANES_PER_SIDE, z, y, x]

                debug_info.append(f"  Move {idx}: {move}")
                debug_info.append(f"    Coord: {move.from_coord}")
                debug_info.append(f"    White sum: {white_vals.sum().item()}")
                debug_info.append(f"    Black sum: {black_vals.sum().item()}")
                debug_info.append(f"    Total occupancy: {(white_vals.sum() + black_vals.sum()).item()}")

        debug_info.append("=" * 80)
        return "\n".join(debug_info)

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def _reconstruct_trailblazer_path(from_coord: Tuple[int, int, int], to_coord: Tuple[int, int, int]) -> Set[Tuple[int, int, int]]:
    """
    Reconstruct the set of coordinates traversed by a trailblazer moving in a straight line
    from `from_coord` to `to_coord` in 3D space.

    Assumes movement is along a straight line (orthogonal, diagonal, or 3D-diagonal).
    Returns all intermediate squares, including the destination but excluding the origin.
    """
    fx, fy, fz = from_coord
    tx, ty, tz = to_coord

    # If same square, return empty path
    if from_coord == to_coord:
        return set()

    # Compute deltas
    dx = tx - fx
    dy = ty - fy
    dz = tz - fz

    # Determine step directions (signs)
    step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
    step_z = 0 if dz == 0 else (1 if dz > 0 else -1)

    # Compute max number of steps (for non-orthogonal moves)
    max_steps = max(abs(dx), abs(dy), abs(dz))

    # Validate that it's a straight line
    # All non-zero deltas must have the same absolute value (for diagonal) or one non-zero (orthogonal)
    non_zero_deltas = [d for d in (dx, dy, dz) if d != 0]
    if non_zero_deltas:
        abs_vals = [abs(d) for d in non_zero_deltas]
        if len(set(abs_vals)) != 1:
            # Not a valid straight-line move (e.g., knight-like). Trailblazer shouldn't do this.
            # For safety, just return the destination only.
            return {to_coord}

    path = set()
    x, y, z = fx, fy, fz

    for _ in range(max_steps):
        x += step_x
        y += step_y
        z += step_z
        path.add((x, y, z))
        if (x, y, z) == to_coord:
            break

    return path


def extract_enemy_slid_path(mv: Move) -> List[Tuple[int, int, int]]:
    """Extract enemy sliding path for trailblaze effect."""
    # Implementation depends on your specific trailblaze logic
    return []

def _insufficient_material(board: Board) -> bool:
    """Fast insufficient material check."""
    for _, piece in board.list_occupied():
        if piece.ptype not in (PieceType.KING, PieceType.PRIEST):
            return False
    return True

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

def create_game_state_from_tensor(tensor: torch.Tensor, color: Color,
                                cache: Optional[OptimizedCacheManager] = None) -> GameState:
    """Create GameState from tensor representation."""
    board = Board(tensor)
    cache = cache or get_cache_manager(board, color)
    return GameState(board, color, cache)

def clone_game_state_for_search(state: GameState) -> GameState:
    """Create a clone optimized for search operations."""
    return state.clone_with_new_cache()

# ==============================================================================
# PERFORMANCE MONITORING DECORATORS
# ==============================================================================

def track_performance(func):
    """Decorator for tracking function performance."""
    def wrapper(self, *args, **kwargs):
        start_time = time.perf_counter()
        try:
            return func(self, *args, **kwargs)
        finally:
            duration = time.perf_counter() - start_time
            # You can add custom performance tracking here
            pass
    return wrapper
