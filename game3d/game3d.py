"""Optimized Game3D controller with enhanced archery, symmetry, and performance features."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Set
from enum import Enum
import time
import weakref

from game3d.pieces.enums import Color, Result, PieceType
from game3d.movement.movepiece import Move
from game3d.cache.transposition import CompactMove
from game3d.game.gamestate import GameState
from game3d.cache.manager import CacheManager, get_cache_manager
from game3d.board.board import Board
from game3d.cache.symmetry_tt import SymmetryAwareTranspositionTable
from game3d.board.symmetry import SymmetryManager
# ==============================================================================
# OPTIMIZATION CONSTANTS
# ==============================================================================

@dataclass(slots=True, frozen=True)
class MoveReceipt:
    """Optimized return value from submit_move with enhanced metadata."""
    new_state: GameState
    is_legal: bool
    is_game_over: bool
    result: Optional[Result] = None
    message: str = ""
    move_time_ms: float = 0.0
    cache_stats: Optional[Dict[str, Any]] = None

class GameMode(Enum):
    """Enhanced game modes for different play styles."""
    STANDARD = "standard"
    ARCHERY_ONLY = "archery_only"
    HIVE_PRIORITY = "hive_priority"
    SYMMETRY_AWARE = "symmetry_aware"

# ==============================================================================
# OPTIMIZED GAME3D CONTROLLER
# ==============================================================================

class OptimizedGame3D:
    # … existing __slots__ …
    __slots__ = (
        # … keep every existing slot …
        "_turn_counter",          # ← NEW
        "_debug_turn_info",       # ← NEW
    )

    def __init__(self,
                game_mode: GameMode = GameMode.STANDARD,
                enable_symmetry: bool = True,
                transposition_size_mb: int = 512,
                *, debug_turn_info: bool = True) -> None:

        # Initialize board and cache
        board = Board.startpos()
        current_color = Color.WHITE

        # Initialize cache manager
        self._cache = get_cache_manager(board, current_color)

        # Initialize transposition table with symmetry support
        if enable_symmetry:
            symmetry_manager = SymmetryManager()
            self._transposition_table = SymmetryAwareTranspositionTable(symmetry_manager, transposition_size_mb)
        else:
            self._transposition_table = None

        # Game configuration
        self._game_mode = game_mode
        self._symmetry_enabled = enable_symmetry
        self._batch_processing = False

        # Performance tracking
        self._performance_stats = {
            'total_moves': 0,
            'archery_attacks': 0,
            'hive_turns': 0,
            'illegal_moves': 0,
            'average_move_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
        }

        # Initialize game state
        self._state = GameState(board, current_color, self._cache)
        self._move_history: List[Tuple[Move, float]] = []  # Move and processing time
        # NEW: turn-tracking
        self._turn_counter: int = 1                 # first turn is 1
        self._debug_turn_info: bool = debug_turn_info

    # ---------- PUBLIC API ----------
    @property
    def state(self) -> GameState:
        """Current position (immutable)."""
        return self._state

    @property
    def current_player(self) -> Color:
        return self._state.current

    def is_game_over(self) -> bool:
        return self._state.is_game_over()

    def result(self) -> Optional[Result]:
        return self._state.result()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        stats = self._performance_stats.copy()
        if self._transposition_table and hasattr(self._transposition_table, 'get_symmetry_stats'):
            stats['symmetry'] = self._transposition_table.get_symmetry_stats()
        return stats

    def toggle_debug_turn_info(self, value: bool | None = None) -> bool:
        """Toggle (or set) turn debug print-outs.  Returns new state."""
        if value is None:
            self._debug_turn_info = not self._debug_turn_info
        else:
            self._debug_turn_info = bool(value)
        return self._debug_turn_info

    # ---------- ENHANCED MOVE SUBMISSION ----------
    def submit_move(self, move: Move) -> MoveReceipt:
        start_time = time.perf_counter()

        if self._debug_turn_info:
            print(f"[Turn {self._turn_counter}] {self.current_player.name} submits {move}")

        # Fast validation pipeline
        validation_result = self._validate_move_fast(move)
        if not validation_result['valid']:
            return self._create_error_receipt(validation_result['message'], start_time)

        # Process move
        try:
            new_state = self._state.make_move(move)
            self._update_performance_stats('move', time.perf_counter() - start_time)

            # Update history
            self._move_history.append((move, time.perf_counter() - start_time))
            self._state = new_state

            # Increment turn counter after successful move
            self._turn_counter += 1

            return MoveReceipt(
                new_state=new_state,
                is_legal=True,
                is_game_over=new_state.is_game_over(),
                result=new_state.result(),
                message="",
                move_time_ms=(time.perf_counter() - start_time) * 1000,
                cache_stats=self._get_cache_stats()
            )

        except ValueError as e:
            self._update_performance_stats('illegal', time.perf_counter() - start_time)
            return self._create_error_receipt(str(e), start_time)

    def submit_archery_attack(self, target_sq: Tuple[int, int, int]) -> MoveReceipt:
        if self._debug_turn_info:
            print(f"[Turn {self._turn_counter}] {self.current_player.name} archery attack → {target_sq}")
        start_time = time.perf_counter()

        # Fast archer validation
        if not self._current_player_has_archer():
            return self._create_error_receipt("No archer controlled.", start_time)

        # Validate target on 2-radius sphere surface
        if not self._is_valid_archery_target(target_sq):
            return self._create_error_receipt("Invalid archery target - must be on 2-radius sphere surface.", start_time)

        # Check line of sight
        if not self._has_archery_line_of_sight(target_sq):
            return self._create_error_receipt("No clear line of sight to target.", start_time)

        try:
            # Create and apply archery attack
            archery_move = self._create_archery_move(target_sq)
            new_state = self._apply_archery_attack(archery_move, target_sq)

            self._update_performance_stats('archery', time.perf_counter() - start_time)
            self._move_history.append((archery_move, time.perf_counter() - start_time))
            self._state = new_state

            # Increment turn counter after successful move
            self._turn_counter += 1

            return MoveReceipt(
                new_state=new_state,
                is_legal=True,
                is_game_over=new_state.is_game_over(),
                result=new_state.result(),
                message="Archery strike successful!",
                move_time_ms=(time.perf_counter() - start_time) * 1000,
                cache_stats=self._get_cache_stats()
            )

        except Exception as e:
            self._update_performance_stats('illegal', time.perf_counter() - start_time)
            return self._create_error_receipt(f"Archery attack failed: {str(e)}", start_time)

    def submit_hive_turn(self, moves: List[Move]) -> MoveReceipt:
        if self._debug_turn_info:
            print(f"[Turn {self._turn_counter}] {self.current_player.name} hive turn ({len(moves)} moves)")
        start_time = time.perf_counter()

        if not moves:
            return self._create_error_receipt("No moves submitted.", start_time)

        # Batch validation
        validation_result = self._validate_hive_moves_batch(moves)
        if not validation_result['valid']:
            return self._create_error_receipt(validation_result['message'], start_time)

        try:
            # Apply moves atomically
            new_state = self._apply_hive_moves_atomically(moves)

            self._update_performance_stats('hive', time.perf_counter() - start_time)

            # Add all moves to history
            for move in moves:
                self._move_history.append((move, time.perf_counter() - start_time))

            self._state = new_state

            # Increment turn counter after successful move
            self._turn_counter += 1

            return MoveReceipt(
                new_state=new_state,
                is_legal=True,
                is_game_over=new_state.is_game_over(),
                result=new_state.result(),
                message="Hive turn completed successfully.",
                move_time_ms=(time.perf_counter() - start_time) * 1000,
                cache_stats=self._get_cache_stats()
            )

        except Exception as e:
            self._update_performance_stats('illegal', time.perf_counter() - start_time)
            return self._create_error_receipt(f"Hive turn failed: {str(e)}", start_time)

    # ---------- ENHANCED VALIDATION ----------
    def _validate_move_fast(self, move: Move) -> Dict[str, Any]:
        """Fast move validation pipeline."""
        # Check game over
        if self.is_game_over():
            return {'valid': False, 'message': "Game already finished."}

        # Check piece exists
        piece = self._state.cache.piece_cache.get(move.from_coord)
        if piece is None:
            return {'valid': False, 'message': f"No piece at {move.from_coord}"}

        # Check color
        if piece.color != self.current_player:
            return {'valid': False, 'message': "Not your turn."}

        # Check legality (cached)
        legal_moves = self._state.legal_moves()
        if move not in legal_moves:
            return {'valid': False, 'message': "Illegal move."}

        return {'valid': True, 'message': ""}

    def _validate_hive_moves_batch(self, moves: List[Move]) -> Dict[str, Any]:
        """Batch validation for hive moves."""
        # Check all pieces are hive pieces
        for move in moves:
            piece = self._state.cache.piece_cache.get(move.from_coord)
            if not piece or piece.ptype != PieceType.HIVE or piece.color != self.current_player:
                return {'valid': False, 'message': "Only Hive pieces may move."}

        # Check no non-hive alternatives exist
        all_legal = self._state.legal_moves()
        non_hive_moves = [m for m in all_legal if self._state.cache.piece_cache.get(m.from_coord).ptype != PieceType.HIVE]
        if non_hive_moves:
            return {'valid': False, 'message': "Must move only Hive pieces this turn."}

        # Validate each move
        for move in moves:
            if move not in all_legal:
                return {'valid': False, 'message': f"Illegal move: {move}"}

        return {'valid': True, 'message': ""}

    def _current_player_has_archer(self) -> bool:
        """Check if current player controls at least one archer."""
        return self._cache.get_archery_cache().get_archer_count(self.current_player) > 0

    def _is_valid_archery_target(self, target_sq: Tuple[int, int, int]) -> bool:
        """Check if target is on 2-radius sphere surface from any archer."""
        archery_cache = self._cache.get_archery_cache()
        return archery_cache.is_valid_attack(target_sq, self.current_player)

    def _has_archery_line_of_sight(self, target_sq: Tuple[int, int, int]) -> bool:
        """Check if any archer has line of sight to target."""
        archery_cache = self._cache.get_archery_cache()
        return any(
            archery_cache.has_line_of_sight(archer_sq, target_sq, self.current_player)
            for archer_sq in archery_cache._archer_positions[self.current_player]
        )

    def _create_archery_move(self, target_sq: Tuple[int, int, int]) -> Move:
        """Create archery attack move."""
        return Move(
            from_coord=target_sq,
            to_coord=target_sq,
            is_capture=True,
            metadata={
                "is_archery": True,
                "archer_player": self.current_player,
                "target_square": target_sq,
                "timestamp": time.time()
            }
        )

    def _apply_archery_attack(self, archery_move: Move, target_sq: Tuple[int, int, int]) -> GameState:
        """Apply archery attack to board state."""
        new_board = self._state.board.clone()

        # ✅ Fast piece lookup via cache
        if self._cache.piece.get(target_sq) is not None:
            new_board.set_piece(target_sq, None)

        self._cache.apply_move(archery_move, self.current_player)

        return GameState(
            board=new_board,
            color=self.current_player.opposite(),
            cache=self._cache,
            history=self._state.history + (archery_move,),
            halfmove_clock=self._state.halfmove_clock + 1,
        )

    def _apply_hive_moves_atomically(self, moves: List[Move]) -> GameState:
        """Apply all hive moves atomically."""
        new_board = self._state.board.clone()

        for move in moves:
            new_board.apply_move(move)
            self._cache.apply_move(move, self.current_player)

        return GameState(
            board=new_board,
            color=self.current_player.opposite(),
            cache=self._cache,
            history=self._state.history + tuple(moves),
            halfmove_clock=0,  # Reset halfmove clock for hive moves
        )

    # ---------- UTILITY METHODS ----------
    def _create_error_receipt(self, message: str, start_time: float) -> MoveReceipt:
        """Create error move receipt with performance tracking."""
        self._performance_stats['illegal_moves'] += 1
        return MoveReceipt(
            new_state=self._state,
            is_legal=False,
            is_game_over=self.is_game_over(),
            result=self.result(),
            message=message,
            move_time_ms=(time.perf_counter() - start_time) * 1000,
            cache_stats=self._get_cache_stats()
        )

    def _update_performance_stats(self, move_type: str, processing_time: float) -> None:
        """Update performance statistics."""
        self._performance_stats['total_moves'] += 1
        self._performance_stats['average_move_time'] = (
            (self._performance_stats['average_move_time'] * (self._performance_stats['total_moves'] - 1) + processing_time) /
            self._performance_stats['total_moves']
        )

        if move_type == 'archery':
            self._performance_stats['archery_attacks'] += 1
        elif move_type == 'hive':
            self._performance_stats['hive_turns'] += 1
        elif move_type == 'illegal':
            self._performance_stats['illegal_moves'] += 1

    def _get_cache_stats(self) -> Dict[str, Any]:
        """Get current cache statistics."""
        if hasattr(self._cache, 'get_stats'):
            return self._cache.get_stats()
        return {}

    def reset(self, start_state: Optional[GameState] = None) -> None:
        """Reset game with optional starting state."""
        board = start_state.board if start_state else Board.startpos()
        current_color = start_state.color if start_state else Color.WHITE  # Fixed to state.color

        # Reinitialize cache
        self._cache = get_cache_manager(board, current_color)

        # Reset state
        self._state = start_state or GameState(board, current_color, self._cache)
        self._move_history.clear()

        # Reset stats
        self._turn_counter = 1   # ← NEW
        for key in self._performance_stats:
            self._performance_stats[key] = 0

    def get_move_history(self) -> List[Tuple[Move, float]]:
        """Get move history with processing times."""
        return self._move_history.copy()

    def __repr__(self) -> str:
        return f"OptimizedGame3D({self._state}, mode={self._game_mode.value})"

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

def create_standard_game() -> OptimizedGame3D:
    """Create standard game."""
    return OptimizedGame3D(game_mode=GameMode.STANDARD)

def create_archery_game() -> OptimizedGame3D:
    """Create archery-focused game."""
    return OptimizedGame3D(game_mode=GameMode.ARCHERY_ONLY)

def create_symmetry_aware_game(transposition_size_mb: int = 512) -> OptimizedGame3D:
    """Create game with full symmetry optimization."""
    return OptimizedGame3D(
        game_mode=GameMode.SYMMETRY_AWARE,
        enable_symmetry=True,
        transposition_size_mb=transposition_size_mb
    )

# ==============================================================================
# BACKWARD COMPATIBILITY
# ==============================================================================

class Game3D(OptimizedGame3D):
    """Backward compatibility wrapper."""

    def __init__(self) -> None:
        super().__init__(game_mode=GameMode.STANDARD, enable_symmetry=False)
