# gamestate.py
from __future__ import annotations
"""
game3d/game/gamestate.py
Optimized 9×9×9 game state with incremental updates, caching, and performance monitoring.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING
from enum import Enum

import torch
import numpy as np
from game3d.board.board import Board
from game3d.pieces.enums import Color, PieceType, Result
from game3d.movement.movepiece import Move
from game3d.common.common import SIZE_X, SIZE_Y, SIZE_Z, N_TOTAL_PLANES, N_PIECE_TYPES
from game3d.cache.manager import OptimizedCacheManager, get_cache_manager
from game3d.pieces.piece import Piece
from game3d.cache.effects_cache import EffectsCache  # Import EffectsCache

from .zobrist import compute_zobrist
from .performance import PerformanceMetrics

# Add this GameMode enum definition
class GameMode(Enum):
    """Game mode enumeration."""
    STANDARD = 0

# Avoid circular imports - functions will be bound later
if TYPE_CHECKING:
    from typing import Callable


@dataclass(slots=True)
class GameState:
    """Optimized game state with caching and incremental updates."""
    board: Board
    color: Color
    cache: OptimizedCacheManager
    history: Tuple[Move, ...] = field(default_factory=tuple)
    halfmove_clock: int = 0
    game_mode: GameMode = GameMode.STANDARD
    turn_number: int = 1  # Current turn number (1-indexed)

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

    # Effects cache
    effects: EffectsCache = field(init=False, repr=False)

    def __post_init__(self):
        # Initialize effects cache with this game state
        self.effects = EffectsCache(self.board, self.cache)
        self._zkey = compute_zobrist(self.board, self.color)
        self._metrics = PerformanceMetrics()
        self._clear_caches()

    # ------------------------------------------------------------------
    # PROPERTIES AND CACHING
    # ------------------------------------------------------------------
    @property
    def zkey(self) -> int:
        """Thread-safe Zobrist key access."""
        return self._zkey

    # ------------------------------------------------------------------
    # CACHE ACCESS PROPERTIES
    # ------------------------------------------------------------------
    @property
    def piece_cache(self):
        """Access to the piece cache through the cache manager."""
        return self.cache.piece_cache

    @property
    def occupancy_cache(self):
        """Access to the occupancy cache through the cache manager."""
        return self.cache.occupancy

    # ------------------------------------------------------------------
    # EFFECT ACCESS METHODS
    # ------------------------------------------------------------------
    def is_frozen(self, sq: Tuple[int, int, int], victim: Color) -> bool:
        """Check if a piece is frozen."""
        return self.cache.is_frozen(sq, victim)

    def is_movement_buffed(self, sq: Tuple[int, int, int], friendly: Color) -> bool:
        """Check if a piece has movement buff."""
        return self.cache.is_movement_buffed(sq, friendly)

    def is_movement_debuffed(self, sq: Tuple[int, int, int], victim: Color) -> bool:
        """Check if a piece has movement debuff."""
        return self.cache.is_movement_debuffed(sq, victim)

    def black_hole_pull_map(self, controller: Color) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
        """Get black hole pull targets."""
        return self.cache.black_hole_pull_map(controller)

    def white_hole_push_map(self, controller: Color) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
        """Get white hole push targets."""
        return self.cache.white_hole_push_map(controller)

    def current_trail_squares(self, controller: Color) -> Set[Tuple[int, int, int]]:
        """Get current trailblaze trail squares."""
        return self.cache.current_trail_squares(controller)

    def is_geomancy_blocked(self, sq: Tuple[int, int, int], current_ply: int) -> bool:
        """Check if a square is blocked by geomancy."""
        return self.cache.is_geomancy_blocked(sq, current_ply)

    def archery_targets(self, controller: Color) -> List[Tuple[int, int, int]]:
        """Get archery attack targets."""
        return self.cache.archery_targets(controller)

    def is_valid_archery_attack(self, sq: Tuple[int, int, int], controller: Color) -> bool:
        """Validate archery attack."""
        return self.cache.is_valid_archery_attack(sq, controller)

    def can_capture_wall(self, attacker_sq: Tuple[int, int, int], wall_sq: Tuple[int, int, int], controller: Color) -> bool:
        """Check if a wall can be captured."""
        return self.cache.can_capture_wall(attacker_sq, wall_sq, controller)

    def pieces_at(self, sq: Tuple[int, int, int]) -> List[Piece]:
        """Get pieces at a square (for share square effect)."""
        return self.cache.pieces_at(sq)

    def top_piece(self, sq: Tuple[int, int, int]) -> Optional[Piece]:
        """Get top piece at a square."""
        return self.cache.top_piece(sq)

    def get_attacked_squares(self, color: Color) -> Set[Tuple[int, int, int]]:
        """Get squares attacked by a color."""
        return self.cache.get_attacked_squares(color)

    # ------------------------------------------------------------------
    # TENSOR REPRESENTATION WITH CACHING
    # ------------------------------------------------------------------
    def to_tensor(self, device: Optional[torch.device | str] = None) -> torch.Tensor:
        """Return the 3-D board as a (C, D, H, W) tensor with caching."""
        # Create a cache key based on board state and current player
        cache_key = (self.board.byte_hash(), self.color)

        # Return cached tensor if available and valid
        if (self._tensor_cache is not None and
            self._tensor_cache_key == cache_key and
            (device is None or self._tensor_cache.device == torch.device(device))):
            if device is not None:
                return self._tensor_cache.to(device)
            return self._tensor_cache

        # Generate new tensor
        tensor = torch.zeros(
            (N_TOTAL_PLANES, SIZE_Z, SIZE_Y, SIZE_X),
            dtype=torch.float32,
            device=device,
        )

        # Use vectorized operations instead of loops
        # Get all occupied coordinates and pieces at once
        occupied_data = list(self.board.list_occupied())
        if not occupied_data:
            # No pieces on board
            tensor[-1, :, :, :] = 1.0 if self.color == Color.WHITE else 0.0
            self._tensor_cache = tensor
            self._tensor_cache_key = cache_key
            return tensor

        # Extract coordinates and pieces separately
        coords, pieces = zip(*occupied_data)
        x_coords, y_coords, z_coords = zip(*coords)

        # Convert to numpy arrays for vectorized operations
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        z_coords = np.array(z_coords)

        # Process pieces in batches by color
        for piece_color in [Color.WHITE, Color.BLACK]:
            color_mask = np.array([p.color == piece_color for p in pieces])
            if not np.any(color_mask):
                continue

            # Get coordinates for this color
            color_x = x_coords[color_mask]
            color_y = y_coords[color_mask]
            color_z = z_coords[color_mask]

            # Get piece types for this color
            color_pieces = [p for p, mask in zip(pieces, color_mask) if mask]
            ptype_values = np.array([p.ptype.value for p in color_pieces])

            # Calculate offset based on current player
            offset = 0 if piece_color == self.color else N_TOTAL_PLANES // 2

            # Set tensor values using vectorized indexing
            for i, (x, y, z, ptype) in enumerate(zip(color_x, color_y, color_z, ptype_values)):
                if ptype < N_TOTAL_PLANES - 1:
                    tensor[ptype + offset, z, y, x] = 1.0

        # Set color-to-move plane
        tensor[-1, :, :, :] = 1.0 if self.color == Color.WHITE else 0.0

        # Cache the result
        if device is None:
            self._tensor_cache = tensor
            self._tensor_cache_key = cache_key
        else:
            # If device is specified, we'll cache on CPU and move to device when needed
            self._tensor_cache = tensor.cpu()
            self._tensor_cache_key = cache_key

        return tensor

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

        # Use clear method instead of directly clearing internal structures
        if hasattr(self.cache, 'move') and hasattr(self.cache.move, 'clear'):
            self.cache.move.clear()

    # ------------------------------------------------------------------
    # UTILITIES AND SAMPLING
    # ------------------------------------------------------------------
    def sample_pi(self, pi) -> Optional[Move]:
        """Sample from policy distribution."""
        moves = self.legal_moves()
        return moves[0] if moves else None

    def clone(self) -> 'GameState':
        new_board = Board(self.board.tensor().clone())
        new_cache = get_cache_manager(new_board, self.color)
        return GameState(
            board=new_board,
            color=self.color,
            cache=new_cache,
            history=self.history,
            halfmove_clock=self.halfmove_clock,
            game_mode=self.game_mode,
            turn_number=self.turn_number,
        )

    def clone_with_new_cache(self) -> 'GameState':
        """Clone with new cache manager for thread safety."""
        new_board = Board(self.board.tensor().clone())
        new_cache = get_cache_manager(new_board, self.color)

        return GameState(
            board=new_board,
            color=self.color,
            cache=new_cache,
            history=self.history,
            halfmove_clock=self.halfmove_clock,
            game_mode=self.game_mode,
            turn_number=self.turn_number,
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
                f"turn={self.turn_number}, "
                f"zkey={self.zkey:#x})")

    def __str__(self) -> str:
        stats = self.get_performance_stats()
        return (f"GameState[{self.color.name}] "
                f"Moves:{len(self.legal_moves())} "
                f"History:{len(self.history)} "
                f"Clock:{self.halfmove_clock} "
                f"Turn:{self.turn_number} "
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
        debug_info.append(f"Turn Number: {self.turn_number}")
        debug_info.append(f"Legal Moves: {len(legal_moves)}")

        # Check each legal move
        empty_square_moves = []
        for i, move in enumerate(legal_moves):
            piece = self.piece_cache.get(move.from_coord)
            if piece is None:
                empty_square_moves.append((i, move))

        if empty_square_moves:
            debug_info.append(f"EMPTY SQUARE MOVES: {len(empty_square_moves)}")
            for idx, move in empty_square_moves[:5]:  # Show first 5
                x, y, z = move.from_coord
                tensor = self.board.tensor()
                white_vals = tensor[0:N_PIECE_TYPES, z, y, x]
                black_vals = tensor[N_PIECE_TYPES:2*N_PIECE_TYPES, z, y, x]

                debug_info.append(f"  Move {idx}: {move}")
                debug_info.append(f"    Coord: {move.from_coord}")
                debug_info.append(f"    White sum: {white_vals.sum().item()}")
                debug_info.append(f"    Black sum: {black_vals.sum().item()}")
                debug_info.append(f"    Total occupancy: {(white_vals.sum() + black_vals.sum()).item()}")

        debug_info.append("=" * 80)
        return "\n".join(debug_info)

    def legal_moves(self) -> List[Move]:
        """Get legal moves for current player."""
        from game3d.game.turnmove import legal_moves
        return legal_moves(self)

    def make_move(self, mv: Move) -> 'GameState':
        """Apply move and return new game state."""
        from game3d.game.turnmove import make_move
        return make_move(self, mv)

    def is_check(self) -> bool:
        """Check if current player is in check."""
        from game3d.game.terminal import is_check
        return is_check(self)

    def is_game_over(self) -> bool:
        """Check if game is over."""
        from game3d.game.terminal import is_game_over
        return is_game_over(self)

    def result(self) -> Optional[Result]:
        """Get game result if game is over."""
        from game3d.game.terminal import result
        return result(self)
