# game3d/game/gamestate.py - CLEANED
from __future__ import annotations
"""
Optimized 9×9×9 game state with incremental updates, caching, and performance monitoring.
"""

from dataclasses import dataclass, field, replace
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING, Set
from enum import Enum

import torch
import numpy as np

from game3d.board.board import Board
from game3d.common.enums import Color, PieceType, Result
from game3d.movement.movepiece import Move
from game3d.common.constants import SIZE_X, SIZE_Y, SIZE_Z, N_TOTAL_PLANES, N_PIECE_TYPES
from game3d.game.performance import PerformanceMetrics
from game3d.pieces.piece import Piece

# Common imports for standardized access
from game3d.common.cache_utils import get_cache_manager
from game3d.common.piece_utils import find_king, get_player_pieces
from game3d.common.state_utils import create_new_state
from game3d.common.validation import validate_cache_integrity

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager

class GameMode(Enum):
    """Game mode enumeration."""
    STANDARD = 0

@dataclass(slots=True)
class GameState:
    """Optimized game state with caching and incremental updates."""
    board: Board
    color: Color
    cache_manager: 'OptimizedCacheManager'
    history: Tuple[Move, ...] = field(default_factory=tuple)
    halfmove_clock: int = 0
    game_mode: GameMode = GameMode.STANDARD
    turn_number: int = 1

    _zkey: int = field(init=False)
    _legal_moves_cache: Optional[List[Move]] = field(default=None, repr=False)
    _legal_moves_cache_key: Optional[int] = field(default=None, repr=False)
    _tensor_cache: Optional[torch.Tensor] = field(default=None, repr=False)
    _tensor_cache_key: Optional[int] = field(default=None, repr=False)
    _insufficient_material_cache: Optional[bool] = field(default=None, repr=False)
    _insufficient_material_cache_key: Optional[int] = field(default=None, repr=False)
    _is_check_cache: Optional[bool] = field(default=None, repr=False)
    _is_check_cache_key: Optional[int] = field(default=None, repr=False)
    _metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics, repr=False)
    _undo_stack: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    _undo_info: Optional[Any] = field(default=None, repr=False)

    def __post_init__(self):
        if self.cache_manager is None:
            raise RuntimeError("GameState must be given an external cache_manager")
        self.board.cache_manager = self.cache_manager
        # Use cache manager's Zobrist hash
        self._zkey = self.cache_manager._current_zobrist_hash
        self._clear_caches()

    @property
    def cache_manager(self):
        """Primary cache manager access."""
        return self._cache_manager

    @property
    def zkey(self) -> int:
        return self._zkey

    # CACHE ACCESS PROPERTIES - STANDARDIZED NAMES
    @property
    def piece_cache(self):
        return self.cache_manager.occupancy

    @property
    def occupancy_cache(self):
        return self.cache_manager.occupancy

    # EFFECT ACCESS METHODS - STANDARDIZED (delegate to cache manager)
    def is_frozen(self, sq: Tuple[int, int, int], victim: Color) -> bool:
        return self.cache_manager.is_frozen(sq, victim)

    def is_movement_buffed(self, sq: Tuple[int, int, int], friendly: Color) -> bool:
        return self.cache_manager.is_movement_buffed(sq, friendly)

    def is_movement_debuffed(self, sq: Tuple[int, int, int], victim: Color) -> bool:
        return self.cache_manager.is_movement_debuffed(sq, victim)

    def black_hole_pull_map(self, controller: Color) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
        return self.cache_manager.black_hole_pull_map(controller)

    def white_hole_push_map(self, controller: Color) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
        return self.cache_manager.white_hole_push_map(controller)

    def current_trail_squares(self, controller: Color) -> Set[Tuple[int, int, int]]:
        return self.cache_manager.current_trail_squares(controller)

    def is_geomancy_blocked(self, sq: Tuple[int, int, int]) -> bool:
        return self.cache_manager.is_geomancy_blocked(sq, self.ply)

    def _with_metrics(self, **kw) -> "GameState":
        return replace(self, _metrics=replace(self._metrics, **kw))

    # PIECE UTILITIES (delegate to common modules)
    def has_priest(self, color: Color) -> bool:
        """Check if player has any priests alive."""
        return self.cache_manager.has_priest(color)

    def any_priest_alive(self) -> bool:
        """Check if any priests are alive on the board."""
        return self.cache_manager.any_priest_alive()

    def find_king(self, color: Color) -> Optional[Tuple[int, int, int]]:
        """Find king position for given color."""
        return find_king(self, color)

    def get_attacked_squares(self, color: Color) -> Set[Tuple[int, int, int]]:
        """Get squares attacked by given color."""
        return self.cache_manager.get_attacked_squares(color)

    @property
    def ply(self) -> int:
        """Get current ply (half-move count)."""
        return self.halfmove_clock

    # TENSOR REPRESENTATION - OPTIMIZED
    def to_tensor(self, device: Optional[torch.device | str] = None) -> torch.Tensor:
        from game3d.common.tensor_utils import create_occupancy_mask_tensor, get_current_player

        cache_key = (self.board.byte_hash(), self.color)

        if (self._tensor_cache is not None and
            self._tensor_cache_key == cache_key and
            (device is None or self._tensor_cache.device == torch.device(device))):
            if device is not None:
                return self._tensor_cache.to(device)
            return self._tensor_cache

        tensor = torch.zeros(
            (N_TOTAL_PLANES, SIZE_Z, SIZE_Y, SIZE_X),
            dtype=torch.float32,
            device=device,
        )

        # Use cache_manager for efficient piece iteration
        for color in [Color.WHITE, Color.BLACK]:
            for coord, piece in self.cache_manager.occupancy.iter_color(color):
                x, y, z = coord
                offset = 0 if piece.color == Color.WHITE else N_PIECE_TYPES
                tensor[offset + piece.ptype.value, z, y, x] = 1.0

        tensor[-1, :, :, :] = 1.0 if self.color == Color.WHITE else 0.0

        if device is None:
            self._tensor_cache = tensor
            self._tensor_cache_key = cache_key
        else:
            self._tensor_cache = tensor.cpu()
            self._tensor_cache_key = cache_key

        return tensor

    def _clear_caches(self) -> None:
        self._legal_moves_cache = None
        self._legal_moves_cache_key = None
        self._tensor_cache = None
        self._tensor_cache_key = None
        self._insufficient_material_cache = None
        self._insufficient_material_cache_key = None
        self._is_check_cache = None
        self._is_check_cache_key = None

    def clone(self, deep_cache: bool = False) -> GameState:
        """Clone game state - ALWAYS reuse cache manager."""
        # CRITICAL: ALWAYS reuse existing cache manager
        # Just update the board reference
        new_board = self.board.clone()

        self.cache_manager.board = new_board
        new_board.cache_manager = self.cache_manager

        # Update cache manager state for new board
        self.cache_manager.rebuild(new_board, self.color)

        return GameState(
            board=new_board,
            color=self.color,
            cache_manager=self.cache_manager,  # REUSE same manager
            history=self.history,
            halfmove_clock=self.halfmove_clock,
            game_mode=self.game_mode,
            turn_number=self.turn_number,
        )

    def clone_with_new_cache(self) -> 'GameState':
        """Clone with new cache manager - only for thread safety or search."""
        from game3d.cache.manager import get_cache_manager

        new_board = Board(self.board.tensor().clone())
        new_cache_manager = get_cache_manager(new_board, self.color)

        return GameState(
            board=new_board,
            color=self.color,
            cache_manager=new_cache_manager,
            history=self.history,
            halfmove_clock=self.halfmove_clock,
            game_mode=self.game_mode,
            turn_number=self.turn_number,
        )

    def push_move(self, mv: Move) -> None:
        """Apply move in-place using incremental updates to the single cache manager."""
        from game3d.common.move_validation import validate_move_basic
        from game3d.game.turnmove import _compute_undo_info

        if not validate_move_basic(self, mv, self.color):
            raise ValueError(f"Invalid move: {mv}")

        moving_piece = self.cache_manager.get_piece(mv.from_coord)
        captured_piece = self.cache_manager.get_piece(mv.to_coord) if mv.is_capture else None

        undo_info = _compute_undo_info(self, mv, moving_piece, captured_piece)
        self._undo_stack.append(undo_info)

        if not self.cache_manager.apply_move(mv, self.color):
            self._undo_stack.pop()
            raise ValueError(f"Failed to apply move: {mv}")

        self.history += (mv,)
        self.color = self.color.opposite()
        self.halfmove_clock += 1
        self.turn_number += 1
        self._zkey = self.cache_manager._current_zobrist_hash
        self._clear_caches()

        # Update halfmove clock logic
        is_pawn = moving_piece.ptype == PieceType.PAWN if moving_piece else False
        is_capture = mv.is_capture or captured_piece is not None
        if is_pawn or is_capture:
            self.halfmove_clock = 0

    def pop_move(self) -> None:
        """Undo last move in-place using incremental undo on the single cache manager."""
        if not self.history or not self._undo_stack:
            raise ValueError("No move to undo")

        mv = self.history[-1]
        undo_info = self._undo_stack.pop()

        if not self.cache_manager.undo_move(mv, self.color.opposite()):
            raise RuntimeError("Failed to undo move")

        self.history = self.history[:-1]
        self.color = self.color.opposite()
        self.halfmove_clock = undo_info['original_halfmove_clock']
        self.turn_number = undo_info['original_turn_number']
        self._zkey = undo_info['original_zkey']
        self._clear_caches()

    # GAME LOGIC (delegate to other modules)
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

    def pass_turn(self) -> "GameState":
        """Pass turn without moving - reuse existing cache manager."""
        # Update cache manager for new color
        self.cache_manager._current = self.color.opposite()

        return GameState(
            board=self.board,
            color=self.color.opposite(),
            cache_manager=self.cache_manager,
            history=self.history,
            halfmove_clock=self.halfmove_clock,
            game_mode=self.game_mode,
            turn_number=self.turn_number + 1,
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        return self._metrics.get_stats()

    def reset_performance_stats(self) -> None:
        self._metrics.reset()

    def __repr__(self) -> str:
        return (f"GameState(color={self.color.name}, "
                f"history={len(self.history)}, "
                f"clock={self.halfmove_clock}, "
                f"turn={self.turn_number}, "
                f"zkey={self.zkey:#x})")

    def debug_state(self) -> str:
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
        debug_info.append("=" * 80)
        return "\n".join(debug_info)
