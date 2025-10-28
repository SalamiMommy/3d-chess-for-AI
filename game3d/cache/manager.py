# manager.py - FIXED Zobrist usage
from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Set, Any, TYPE_CHECKING, Iterator, Callable, Iterable, Union
import weakref
import numpy as np

from game3d.common.constants import N_TOTAL_PLANES, SIZE
from game3d.common.enums import Color, PieceType
from game3d.pieces.piece import Piece
from game3d.common.coord_utils import Coord, in_bounds
from game3d.common.piece_utils import get_piece_effect_type, iterate_occupied, find_king
from game3d.common.move_utils import filter_none_moves
from game3d.common.debug_utils import CacheStatsMixin

if TYPE_CHECKING:
    from game3d.board.board import Board
    from game3d.movement.movepiece import Move

from game3d.cache.caches.movecache import (
    OptimizedMoveCache,
    CompactMove,
    create_optimized_move_cache,
)
from game3d.cache.caches.occupancycache import OccupancyCache
from game3d.cache.caches.transposition import TTEntry
from game3d.cache.caches.zobrist import ZobristHash, compute_zobrist
from .managerconfig import ManagerConfig
from .managerperformance import CachePerformanceMonitor, CacheEventType, MemoryManager
from .parallelmanager import ParallelManager
from .export import (
    export_state_for_ai,
    export_tensor_for_ai,
    get_legal_move_indices,
    get_legal_moves_as_policy_target,
    validate_export_integrity,
)
from .diagnostics import record_cache_creation
from game3d.cache.effectscache.auracache import UnifiedAuraCache
from game3d.cache.effectscache.trailblazecache import TrailblazeCache
from game3d.cache.effectscache.geomancycache import GeomancyCache
from game3d.cache.caches.attackscache import AttacksCache
from game3d.cache.cache_protocols import CacheManagerProtocol, MovementCacheProtocol, EffectCacheProtocol

# Global cache manager registry to ensure singleton behavior
_ACTIVE_CACHE_MANAGERS: Dict[int, 'OptimizedCacheManager'] = {}

class CacheDesyncError(Exception):
    pass

class OptimizedCacheManager(CacheManagerProtocol, CacheStatsMixin):
    __slots__ = (
        "config", "board", "occupancy", "performance_monitor",
        "_zobrist", "_current_zobrist_hash", "parallel", "_move_cache",
        "_move_counter", "_age_counter", "_current", "_needs_rebuild",
        "_skip_effect_updates", "_effect_update_counter",
        "_effect_update_interval", "_check_summary_cache", "_check_summary_age",
        "_network_teleport_dirty", "_swap_targets_dirty", "_memory_manager",
        "_integrated_jump_gen", "_swap_targets",
        "_swap_targets_dirty",
        "_network_teleport_targets",
        "_network_teleport_dirty",
        "_reflecting_bishop_gen", "_board",
        "aura_cache", "trailblaze_cache", "geomancy_cache", "attacks_cache",
        "_manager_id", "_is_singleton", "_initialized", "_board_ref"
    )

    def __new__(cls, board: Board, current: Color = Color.WHITE):
        """Singleton pattern - one manager per board."""
        manager_id = id(board)

        # Return existing manager if found
        if manager_id in _ACTIVE_CACHE_MANAGERS:
            existing = _ACTIVE_CACHE_MANAGERS[manager_id]
            # Update current color if needed
            if existing._current != current:
                existing._current = current
            return existing

        # Create new instance
        instance = super().__new__(cls)
        _ACTIVE_CACHE_MANAGERS[manager_id] = instance
        return instance

    def __init__(self, board: Board, current: Color = Color.WHITE) -> None:
        if hasattr(self, '_initialized') and self._initialized:
            # Update only what's necessary
            self._current = current
            self.board = board
            # Update weak reference
            self._board_ref = weakref.ref(board)
            return

        self._initialized = True
        self._manager_id = id(board)
        self._is_singleton = True
        _ACTIVE_CACHE_MANAGERS[self._manager_id] = self

        self.config = ManagerConfig()
        self.board = board
        self._board_ref = weakref.ref(board)  # Add weak reference
        self._current = current

        self.performance_monitor = CachePerformanceMonitor()
        self._memory_manager = MemoryManager(self.config, lambda: self._move_cache)
        self.parallel = ParallelManager(self.config)

        # Initialize Zobrist hash instance
        self._zobrist = ZobristHash()
        self._current_zobrist_hash = self._zobrist.compute_from_scratch(self.board, current)

        self.occupancy = OccupancyCache(self)
        self._move_cache: Optional[OptimizedMoveCache] = None
        self.attacks_cache = AttacksCache(board)
        self._initialize_effect_caches()
        self.attacks_cache._manager = self

        self._move_counter = 0
        self._age_counter = 0

        self._skip_effect_updates = False
        self._effect_update_counter = 0
        self._effect_update_interval = 10
        self._check_summary_cache: Optional[Dict[str, Any]] = None
        self._check_summary_age = -1

        self._integrated_jump_gen: Optional[Any] = None
        self._swap_targets = {Color.WHITE: set(), Color.BLACK: set()}
        self._network_teleport_targets = {Color.WHITE: set(), Color.BLACK: set()}
        self._reflecting_bishop_gen: Optional[Any] = None

        record_cache_creation(self, board)

        # CRITICAL: Set board's cache manager reference
        if board is not None:
            board.cache_manager = self

        self._initialize_effect_caches()

    def __del__(self):
        """Clean up global registry on deletion"""
        if hasattr(self, '_manager_id'):
            _ACTIVE_CACHE_MANAGERS.pop(self._manager_id, None)

    def initialise(self, current: Color) -> None:
        """Initialize the move cache and other components."""
        self._current = current
        self._move_cache = create_optimized_move_cache(self.board, current, self)
        self._current_zobrist_hash = self._zobrist.compute_from_scratch(self.board, current)
        self._needs_rebuild = False

    def _initialize_effect_caches(self) -> None:
        # create caches WITHOUT board first
        self.aura_cache     = UnifiedAuraCache(None, self)
        self.trailblaze_cache = TrailblazeCache(self)
        self.geomancy_cache   = GeomancyCache()

        # set manager references
        self.aura_cache.set_cache_manager(self)
        self.trailblaze_cache.set_cache_manager(self)
        self.geomancy_cache.set_cache_manager(self)

        # NOW rebuild with the real board
        board = self._get_board()
        if board:
            self.aura_cache._full_rebuild(board)


    def apply_move(self, mv: Move, mover: Color) -> bool:
        from_piece = self.occupancy.get(mv.from_coord)
        captured_piece = self.occupancy.get(mv.to_coord) if mv.is_capture else None

        if from_piece is None:
            return False

        # Use ZobristHash instance for incremental update - FIXED
        new_hash = self._zobrist.update_hash_move(
            self._current_zobrist_hash, mv, from_piece, captured_piece
        )

        # Apply move to board FIRST
        success = self.board.apply_move(mv)
        if not success:
            return False

        # Then update hash
        self.sync_zobrist(new_hash)

        # Update ALL caches in correct order
        self.occupancy.incremental_update([(mv.from_coord, mv.to_coord, None)])

        # Update attacks cache
        self.attacks_cache.apply_move(mv, mover, self.board)

        # Update effect caches
        self.aura_cache.apply_move(mv, mover, self.halfmove_clock, self.board)
        self.trailblaze_cache.apply_move(mv, mover, self.halfmove_clock, self.board)
        self.geomancy_cache.apply_move(mv, mover, self.halfmove_clock, self.board)

        # Update move cache last (depends on other caches)
        if self._move_cache:
            self._move_cache.apply_move(mv, mover)

        self._move_counter += 1
        self._effect_update_counter += 1
        if self._effect_update_counter >= self._effect_update_interval:
            self._effect_update_counter = 0

        self._needs_rebuild = False

        board_gen = getattr(self.board, 'generation', 0)
        if self._move_cache:
            self._move_cache._gen = board_gen + 1

        return True

    def undo_move(self, mv: Move, mover: Color) -> bool:
        original_tensor = self.board.tensor().clone()

        try:
            moving_piece = self.occupancy.get(mv.to_coord)
            if moving_piece is None:
                return False

            # Use ZobristHash instance for undo - FIXED
            captured_type = getattr(mv, "captured_ptype", None)
            captured_piece = (
                Piece(mover.opposite(), PieceType(captured_type)) if captured_type else None
            )

            new_hash = self._zobrist.update_hash_move(
                self._current_zobrist_hash, mv, moving_piece, captured_piece
            )

            # Restore the board state
            self.board.set_piece(mv.from_coord, moving_piece)
            self.board.set_piece(mv.to_coord, None)

            if getattr(mv, 'is_capture', False) and hasattr(mv, 'captured_piece'):
                self.board.set_piece(mv.to_coord, mv.captured_piece)

            # Update hash
            self.sync_zobrist(new_hash)

            # Update ALL caches in reverse order
            if self._move_cache:
                self._move_cache.undo_move(mv, mover)

            self.attacks_cache.undo_move(mv, mover, self.board)
            self.geomancy_cache.undo_move(mv, mover, self.halfmove_clock - 1, self.board)
            self.trailblaze_cache.undo_move(mv, mover, self.halfmove_clock - 1, self.board)
            self.aura_cache.undo_move(mv, mover, self.board)

            # Update occupancy
            self.occupancy.incremental_update([
                (mv.to_coord, mv.from_coord, None),
                (mv.to_coord, mv.to_coord, getattr(mv, 'captured_piece', None))
            ])

            self._move_counter -= 1
            self._effect_update_counter -= 1

            return True

        except Exception as e:
            self.board._tensor = original_tensor
            print(f"[ERROR] Undo move failed: {e}")
            return False

    def _get_cache_by_name(self, name: str):
        if name == "aura":
            return self.aura_cache
        elif name == "trailblaze":
            return self.trailblaze_cache
        elif name == "geomancy":
            return self.geomancy_cache
        elif name == "attacks":
            return self.attacks_cache
        elif name == "occupancy":
            return self.occupancy
        elif name == "moves":
            return self.move_cache
        return None

    def get_affected_caches(
        self,
        mv: 'Move',
        mover: Color,
        from_piece: Optional[Piece],
        to_piece: Optional[Piece],
        captured_piece: Optional[Piece],
        is_undo: bool = False
    ) -> Set[str]:
        affected = set()

        effect_map = {
            PieceType.FREEZER:   "aura",
            PieceType.SPEEDER:   "aura",
            PieceType.SLOWER:    "aura",
            PieceType.BLACKHOLE: "aura",
            PieceType.WHITEHOLE: "aura",
            PieceType.TRAILBLAZER: "trailblaze",
            PieceType.GEOMANCER:   "geomancy",
        }

        if from_piece and from_piece.ptype in effect_map:
            affected.add(effect_map[from_piece.ptype])

        if captured_piece and captured_piece.ptype in effect_map:
            affected.add(effect_map[captured_piece.ptype])

        self._add_affected_effects_from_pos(mv.from_coord, affected)
        self._add_affected_effects_from_pos(mv.to_coord, affected)
        if is_undo:
            self._add_affected_effects_from_pos(mv.to_coord, affected)

        return affected

    def get_affected_caches_for_undo(self, mv: 'Move', mover: Color) -> Set[str]:
        return self.get_affected_caches(mv, mover, None, None, None, is_undo=True)

    def _add_affected_effects_from_pos(self, pos: Coord, affected: Set[str]) -> None:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    check_pos = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
                    if all(0 <= c < SIZE for c in check_pos):
                        piece = self.occupancy.get(check_pos)
                        if piece and piece.ptype in {PieceType.FREEZER, PieceType.BLACKHOLE,
                                                     PieceType.WHITEHOLE, PieceType.GEOMANCER}:
                            effect_map = {
                                PieceType.FREEZER: "aura",
                                PieceType.BLACKHOLE: "aura",
                                PieceType.WHITEHOLE: "aura",
                                PieceType.GEOMANCER: "geomancy"
                            }
                            affected.add(effect_map[piece.ptype])

    def update_effect_caches(
            self,
            mv: "Move",
            mover: Color,
            affected_caches: set[str],
            current_ply: int,
    ) -> None:
        for cache_name in affected_caches:
            cache = self._get_cache_by_name(cache_name)
            if cache and hasattr(cache, 'apply_move'):
                cache.apply_move(mv, mover, current_ply, self.board)

    def is_movement_buffed(self, sq: Coord, color: Color) -> bool:
        return self.aura_cache.is_buffed(sq, color)

    def is_movement_debuffed(self, sq: Coord, color: Color) -> bool:
        return self.aura_cache.is_debuffed(sq, color)

    def is_frozen(self, sq: Coord, color: Color) -> bool:
        return self.aura_cache.is_frozen(sq, color)

    def black_hole_pull_map(self, controller: Color) -> Dict[Coord, Coord]:
        return self.aura_cache.pull_map(controller)

    def white_hole_push_map(self, controller: Color) -> Dict[Coord, Coord]:
        return self.aura_cache.push_map(controller)

    def mark_trail(self, trailblazer_sq: Coord, slid_squares: Set[Coord]) -> None:
        self.trailblaze_cache.mark_trail(trailblazer_sq, slid_squares)

    def current_trail_squares(self, controller: Color) -> Set[Coord]:
        return self.trailblaze_cache.current_trail_squares(controller, self.board)

    def is_geomancy_blocked(self, sq: Coord, current_ply: int) -> bool:
        return self.geomancy_cache.is_blocked(sq, current_ply)

    def block_square(self, sq: Coord, current_ply: int) -> bool:
        return self.geomancy_cache.block_square(sq, current_ply, self.board)

    def get_attacked_squares(self, color: Color) -> Set[Coord]:
        cached = self.attacks_cache.get_for_color(color)
        return cached if cached is not None else set()

    def store_attacked_squares(self, color: Color, attacked: Set[Coord]) -> None:
        self.attacks_cache.store_for_color(color, attacked)

    def clear_all_effects(self) -> None:
        for cache in [self.aura_cache, self.trailblaze_cache, self.geomancy_cache, self.attacks_cache]:
            if hasattr(cache, 'clear'):
                cache.clear()
            elif hasattr(cache, 'invalidate'):
                cache.invalidate()

    def apply_blackhole_pulls(self, controller: Color) -> None:
        dirty_squares = self.aura_cache.apply_pull_effects(controller, self.board)
        if dirty_squares:
            self.occupancy.batch_set_positions(
                [(sq, self.occupancy.get(sq)) for sq in dirty_squares]
            )

        self.aura_cache.apply_move(None, controller, self.halfmove_clock, self.board)

        self._move_cache.invalidate_attacked_squares(controller)
        self._move_cache.invalidate_attacked_squares(controller.opposite())
        self._move_cache._lazy_revalidate()

    def apply_whitehole_pushes(self, controller: Color) -> None:
        dirty_squares = self.aura_cache.apply_push_effects(controller, self.board)
        if dirty_squares:
            self.occupancy.batch_set_positions(
                [(sq, self.occupancy.get(sq)) for sq in dirty_squares]
            )

        self.aura_cache.apply_move(None, controller, self.halfmove_clock, self.board)

        self._move_cache.invalidate_attacked_squares(controller)
        self._move_cache.invalidate_attacked_squares(controller.opposite())
        self._move_cache._lazy_revalidate()

    def apply_freeze_effects(self, controller: Color) -> None:
        frozen_squares = self.aura_cache.apply_freeze_effects(controller, self.board)
        if frozen_squares:
            self.occupancy.batch_set_positions(
                [(sq, self.occupancy.get(sq)) for sq in frozen_squares]
            )
        self.aura_cache.apply_move(None, controller, self.halfmove_clock, self.board)

    def get_check_summary(self, has_priest: Callable[[Color], bool]) -> Dict[str, Any]:
        w_k = find_king(self, Color.WHITE)
        b_k = find_king(self, Color.BLACK)
        w_at = self.get_attacked_squares(Color.WHITE)
        b_at = self.get_attacked_squares(Color.BLACK)

        w_checkers = [] if has_priest(Color.WHITE) else [sq for sq in b_at if sq == w_k] if w_k else []
        b_checkers = [] if has_priest(Color.BLACK) else [sq for sq in w_at if sq == b_k] if b_k else []

        return {
            "white_check": bool(w_checkers),
            "black_check": bool(b_checkers),
            "white_king_position": w_k,
            "black_king_position": b_k,
            "white_checkers": w_checkers,
            "black_checkers": b_checkers,
            "attacked_squares_white": w_at,
            "attacked_squares_black": b_at,
            "white_priests_alive": self.has_priest(Color.WHITE),
            "black_priests_alive": self.has_priest(Color.BLACK),
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        base = self.performance_monitor.get_performance_stats()
        if self._move_cache:
            base.update({
                "move_cache_stats": self._move_cache.get_stats(),
                "zobrist_hash": self._current_zobrist_hash,
                "main_tt_size_mb": self.config.main_tt_size_mb,
                "sym_tt_size_mb": self.config.sym_tt_size_mb,
                "main_tt_capacity_estimate": (self.config.main_tt_size_mb * 1024 * 1024) // 32,
                "sym_tt_capacity_estimate": (self.config.sym_tt_size_mb * 1024 * 1024) // 32,
                "enable_parallel": self.config.enable_parallel,
                "max_workers": self.config.max_workers,
                "enable_vectorization": self.config.enable_vectorization,
            })
        base["effect_caches"] = {
            "aura": {"type": type(self.aura_cache).__name__},
            "trailblaze": {"type": type(self.trailblaze_cache).__name__},
            "geomancy": {"type": type(self.geomancy_cache).__name__},
            "attacks": {"type": type(self.attacks_cache).__name__},
        }
        return base

    def get_optimization_suggestions(self) -> List[str]:
        return self.performance_monitor.get_optimization_suggestions()

    def _log_cache_stats(self, context: str) -> None:
        stats = self.get_cache_stats()
        suggestions = self.get_optimization_suggestions()

    @property
    def move(self) -> OptimizedMoveCache:
        if self._move_cache is None:
            raise RuntimeError("MoveCache not initialised. Call initialise() first.")
        return self._move_cache

    def has_priest(self, color: Color) -> bool:
        return self.occupancy.has_priest(color)

    def any_priest_alive(self) -> bool:
        return self.occupancy.any_priest_alive()

    def priest_status(self) -> Dict[str, bool]:
        return {
            "white_priests_alive": self.has_priest(Color.WHITE),
            "black_priests_alive": self.has_priest(Color.BLACK),
        }

    def rebuild(self, board: Board, color: Color) -> None:
        """FULL reset of cache manager for new game."""
        if self.board is board and self._current == color and not self._needs_rebuild:
            return

        self.board = board
        self._current = color

        # FULL RESET of all caches
        self.occupancy.rebuild(board)

        # REBUILD effect caches using appropriate methods
        # Aura cache has _full_rebuild method
        if hasattr(self.aura_cache, '_full_rebuild'):
            self.aura_cache._full_rebuild(board)
        else:
            self.aura_cache.clear()

        # Trailblaze cache only has clear
        self.trailblaze_cache.clear()

        # Geomancy cache only has clear
        self.geomancy_cache.clear()

        # Attacks cache has force_rebuild method
        if hasattr(self.attacks_cache, 'force_rebuild'):
            self.attacks_cache.force_rebuild()
        else:
            self.attacks_cache.clear()

        # Reset move cache
        if self._move_cache:
            if hasattr(self._move_cache, '_full_rebuild'):
                self._move_cache._full_rebuild()
            else:
                self._move_cache.clear()

        # Reset Zobrist hash
        self._current_zobrist_hash = self._zobrist.compute_from_scratch(board, color)
        self._needs_rebuild = False

    def sync_zobrist(self, new_hash: int) -> None:
        self._current_zobrist_hash = new_hash

    def is_movement_blocked_for_hive(self, sq: Tuple[int, int, int], color: Color) -> bool:
        ply = getattr(self.board, "halfmove_clock", 0)
        return (
            self.is_frozen(sq, color) or
            self.is_movement_debuffed(sq, color) or
            self.is_geomancy_blocked(sq, ply)
        )

    @property
    def halfmove_clock(self) -> int:
        return getattr(self.board, 'halfmove_clock', 0)

    def _attach_board(self, b: Board) -> None:
        self.board = b
        b.cache_manager = self

    def update_occupancy_incrementally(
        self,
        board: "Board",
        moved_sq: Coord,
        captured_sq: Optional[Coord] = None
    ) -> None:
        updates = []

        updates.append((moved_sq, None))

        if captured_sq is not None:
            updates.append((captured_sq, None))

        to_piece = self.occupancy.get(moved_sq)
        if to_piece:
            updates.append((moved_sq, to_piece))

        self.batch_set_pieces(updates)

    def _fast_occupancy_update(self, mv: Move, piece: Piece, mover: Color) -> None:
        captured_sq = mv.to_coord if mv.is_capture else None
        self.occupancy.set_position(mv.from_coord, None)
        if captured_sq:
            self.occupancy.set_position(captured_sq, None)
        self.occupancy.set_position(mv.to_coord, piece)

    def get_piece(self, coord: Coord) -> Optional[Piece]:
        return self.occupancy.get(coord)

    def set_piece(self, coord: Coord, piece: Optional[Piece]) -> None:
        self.occupancy.set_position(coord, piece)

    def get_pieces_of_color(self, color: Color) -> Iterable[Tuple[Coord, Piece]]:
        return self.occupancy.iter_color(color)

    def find_king(self, color: Color) -> Optional[Coord]:
        return self.occupancy.find_king(color)

    def get_occupancy_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.occupancy._occ.copy(), self.occupancy._ptype.copy()

    def batch_set_pieces(self, updates: List[Tuple[Coord, Optional[Piece]]]) -> None:
        self.occupancy.batch_set_positions(updates)

    def get_occupancy_array(self) -> np.ndarray:
        return self.occupancy._occ.copy()

    def get_occupancy_array_readonly(self) -> np.ndarray:
        return self.occupancy._occ

    def get_piece_type_array(self) -> np.ndarray:
        return self.occupancy._ptype.copy()

    @property
    def occupancy_cache(self):
        return self.occupancy

    @property
    def move_cache(self):
        return self._move_cache

    def _detect_input_type(self, coords: Union[Coord, np.ndarray, List[Coord]]) -> str:
        """Auto-detect if input is single coordinate or batch."""
        if isinstance(coords, np.ndarray):
            if coords.ndim == 1 and len(coords) == 3:  # Single coordinate as array
                return "single"
            elif coords.ndim == 2 and coords.shape[1] == 3:  # Batch of coordinates
                return "batch"
            else:
                raise ValueError(f"Unexpected coordinate array shape: {coords.shape}")
        elif isinstance(coords, (tuple, list)):
            if len(coords) == 3 and all(isinstance(c, int) for c in coords):  # Single coordinate
                return "single"
            elif all(isinstance(coord, (tuple, list)) and len(coord) == 3 for coord in coords):  # Batch
                return "batch"
            else:
                raise ValueError(f"Unexpected coordinate list structure")
        else:
            raise TypeError(f"Unsupported coordinate type: {type(coords)}")

    def get_frozen_status(self, coords: Union[Coord, np.ndarray, List[Coord]], color: Color) -> Union[bool, np.ndarray]:
        """Consolidated function for frozen status - auto-detects input type."""
        input_type = self._detect_input_type(coords)

        if input_type == "single":
            # Single coordinate - use normal version
            coord_tuple = tuple(coords) if isinstance(coords, (list, np.ndarray)) else coords
            return self.aura_cache.is_frozen(coord_tuple, color)
        else:
            # Batch processing - use vectorized version
            if isinstance(coords, np.ndarray):
                coord_tuples = [tuple(coord) for coord in coords]
            else:
                coord_tuples = coords

            return np.array([self.aura_cache.is_frozen(coord, color) for coord in coord_tuples])

    def get_debuffed_status(self, coords: Union[Coord, np.ndarray, List[Coord]], color: Color) -> Union[bool, np.ndarray]:
        """Consolidated function for debuffed status - auto-detects input type."""
        input_type = self._detect_input_type(coords)

        if input_type == "single":
            # Single coordinate - use normal version
            coord_tuple = tuple(coords) if isinstance(coords, (list, np.ndarray)) else coords
            return self.aura_cache.is_debuffed(coord_tuple, color)
        else:
            # Batch processing - use vectorized version
            if isinstance(coords, np.ndarray):
                coord_tuples = [tuple(coord) for coord in coords]
            else:
                coord_tuples = coords

            return np.array([self.aura_cache.is_debuffed(coord, color) for coord in coord_tuples])

    def get_geomancy_blocked_status(self, coords: Union[Coord, np.ndarray, List[Coord]], current_ply: int) -> Union[bool, np.ndarray]:
        """Consolidated function for geomancy blocked status - auto-detects input type."""
        input_type = self._detect_input_type(coords)

        if input_type == "single":
            # Single coordinate - use normal version
            coord_tuple = tuple(coords) if isinstance(coords, (list, np.ndarray)) else coords
            return self.geomancy_cache.is_blocked(coord_tuple, current_ply)
        else:
            # Batch processing - use vectorized version
            if isinstance(coords, np.ndarray):
                coord_tuples = [tuple(coord) for coord in coords]
            else:
                coord_tuples = coords

            return np.array([self.geomancy_cache.is_blocked(coord, current_ply) for coord in coord_tuples])

    def apply_effects_to_moves(
        self,
        moves: Union[Move, List[Move]],
        mover: Color,
        current_ply: int,
        apply_debuff: bool = True
    ) -> Union[Optional[Move], List[Move]]:
        """Consolidated function for applying effects to moves - auto-detects input type."""
        # Handle single move input
        if not isinstance(moves, list):
            single_input = True
            moves_list = [moves]
        else:
            single_input = False
            moves_list = moves

        if not moves_list:
            return moves_list[0] if single_input else moves_list

        # Extract coordinates for processing
        from_coords = np.array([m.from_coord for m in moves_list])
        to_coords = np.array([m.to_coord for m in moves_list])

        # Check effects
        frozen_mask = self.get_frozen_status(from_coords, mover)
        geomancy_mask = self.get_geomancy_blocked_status(to_coords, current_ply)

        # Filter invalid moves
        valid_mask = ~(frozen_mask | geomancy_mask)
        valid_moves = []

        if apply_debuff:
            # Apply debuff effects to remaining moves
            debuffed_mask = self.get_debuffed_status(from_coords[valid_mask], mover)
            valid_indices = np.where(valid_mask)[0]

            for i, move_idx in enumerate(valid_indices):
                move = moves_list[move_idx]
                if debuffed_mask[i]:
                    move = self._apply_debuff_to_move(move)
                valid_moves.append(move)
        else:
            # Just filter without debuff application
            valid_moves = [m for i, m in enumerate(moves_list) if valid_mask[i]]

        # Return appropriate type based on input
        if single_input:
            return valid_moves[0] if valid_moves else None
        else:
            return valid_moves

    def _apply_debuff_to_move(self, move: "Move") -> "Move":
        """Apply debuff effect to a single move"""
        if hasattr(move, 'max_steps') and move.max_steps > 1:
            move = move._replace(max_steps=max(1, move.max_steps - 1))
        return move

    def get_all_pieces_data(self, color: Color) -> Tuple[np.ndarray, np.ndarray]:
        """Consolidated function to get all pieces for a color"""
        coords = []
        types = []
        for coord, piece in self.occupancy.iter_color(color):
            coords.append(coord)
            types.append(piece.ptype.value)
        return np.array(coords), np.array(types)

    def get_attacked_squares_batch(self, colors: Union[Color, List[Color]]) -> Union[Set[Coord], Dict[Color, Set[Coord]]]:
        """Consolidated function for attacked squares - auto-detects input type."""
        if isinstance(colors, Color):
            # Single color - return set directly
            return self.get_attacked_squares(colors)
        else:
            # Multiple colors - return dictionary
            result = {}
            for color in colors:
                result[color] = self.get_attacked_squares(color)
            return result

    def get_pieces_vectorized(self, coords: Union[Coord, np.ndarray, List[Coord]]) -> Union[Optional[Piece], List[Optional[Piece]]]:
        """Consolidated function for getting pieces - auto-detects input type."""
        input_type = self._detect_input_type(coords)

        if input_type == "single":
            # Single coordinate
            coord_tuple = tuple(coords) if isinstance(coords, (list, np.ndarray)) else coords
            return self.occupancy.get(coord_tuple)
        else:
            # Batch processing
            return self.occupancy.batch_get_pieces_vectorized(coords)

    # Keep aliases for backward compatibility
    def batch_get_frozen_status(self, coords: np.ndarray, color: Color) -> np.ndarray:
        """Alias for batch processing - maintained for backward compatibility."""
        return self.get_frozen_status(coords, color)

    def batch_get_debuffed_status(self, coords: np.ndarray, color: Color) -> np.ndarray:
        """Alias for batch processing - maintained for backward compatibility."""
        return self.get_debuffed_status(coords, color)

    def batch_get_geomancy_blocked(self, coords: np.ndarray, current_ply: int) -> np.ndarray:
        """Alias for batch processing - maintained for backward compatibility."""
        return self.get_geomancy_blocked_status(coords, current_ply)

    def batch_apply_effects_to_moves(self, moves: List[Move], mover: Color, current_ply: int) -> List[Move]:
        """Alias for batch processing - maintained for backward compatibility."""
        return self.apply_effects_to_moves(moves, mover, current_ply, apply_debuff=False)

    def batch_apply_effects(self, moves: List[Move], mover: Color, current_ply: int) -> List[Move]:
        """Alias for batch processing - maintained for backward compatibility."""
        return self.apply_effects_to_moves(moves, mover, current_ply, apply_debuff=True)

    def _get_board(self) -> Optional["Board"]:
        """Safely get board reference"""
        if hasattr(self, '_board_ref'):
            board = self._board_ref()
            if board is not None:
                return board
        return self.board if hasattr(self, 'board') else None

    def get_piece_attributes_vectorized(self, coords: Union[Coord, np.ndarray, List[Coord]]) -> Union[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Get piece colors and types - autodetects single vs batch. Returns (color_code, ptype_value) or arrays thereof.
        color_code: 0=empty, 1=white, 2=black
        For batch, returns colors, types, valid_mask"""
        input_type = self._detect_input_type(coords)
        if input_type == "single":
            piece = self.occupancy.get(tuple(coords))
            if piece is None:
                return 0, 0
            color_code = 1 if piece.color == Color.WHITE else 2
            return color_code, piece.ptype.value
        else:
            return self.occupancy.batch_get_colors_and_types(coords)

    @property
    def zobrist_hash(self) -> int:
        """Get the current Zobrist hash - single source of truth."""
        return self._current_zobrist_hash

    def sync_zobrist(self, new_hash: int) -> None:
        """Update the Zobrist hash - only method that should modify it."""
        self._current_zobrist_hash = new_hash

def get_cache_manager(board: Board, current: Color) -> OptimizedCacheManager:
    """Global factory function - ENSURES single cache manager per board"""
    if board is None:
        raise ValueError("get_cache_manager requires a real Board instance")

    board_id = id(board)

    # Check if board already has a cache manager attached
    if hasattr(board, 'cache_manager') and board.cache_manager is not None:
        existing = board.cache_manager
        # Verify it's the correct instance
        if existing._manager_id == board_id:
            # Update current color if needed
            if existing._current != current:
                existing._current = current
            return existing

    # Check global registry
    if board_id in _ACTIVE_CACHE_MANAGERS:
        existing_manager = _ACTIVE_CACHE_MANAGERS[board_id]
        # Update current color if needed
        if existing_manager._current != current:
            existing_manager._current = current
        board.cache_manager = existing_manager
        return existing_manager

    # Only create new manager if none exists
    cm = OptimizedCacheManager(board, current)
    cm.initialise(current)
    board.cache_manager = cm

    return cm

CacheManager = OptimizedCacheManager
