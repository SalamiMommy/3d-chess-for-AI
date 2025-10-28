# auracache.py (reworked for optimized scalar queries and unnecessary dirty flag sets removed)
from __future__ import annotations
from typing import Set, Tuple, Optional, Dict, Any, List, Union, Sequence
from dataclasses import dataclass
from enum import Enum
import weakref
import numpy as np

from game3d.common.enums import Color, PieceType
from game3d.common.coord_utils import get_aura_squares, Coord, in_bounds, filter_valid_coords
from game3d.common.piece_utils import get_piece_effect_type
from game3d.common.cache_utils import get_piece, is_occupied
from game3d.common.debug_utils import CacheStatsMixin
from game3d.pieces.pieces.speeder import buffed_squares
from game3d.pieces.pieces.slower import debuffed_squares
from game3d.pieces.pieces.whitehole import push_candidates
from game3d.pieces.pieces.blackhole import suck_candidates
from game3d.movement.movepiece import Move
from game3d.pieces.piece import Piece

# Type alias for clarity
CoordLike = Union[Coord, Sequence[Coord], np.ndarray]

@dataclass
class AuraEffect:
    source_square: Tuple[int, int, int]
    affected_squares: Set[Tuple[int, int, int]]
    priority: int

class AuraPriority(Enum):
    HIGH = 3
    MEDIUM = 2
    LOW = 1

class AuraType(Enum):
    BUFF = "buff"
    DEBUFF = "debuff"
    FREEZE = "freeze"
    PUSH = "push"
    PULL = "pull"

AURA_PIECE_MAP = {
    AuraType.BUFF: PieceType.SPEEDER,
    AuraType.DEBUFF: PieceType.SLOWER,
    AuraType.FREEZE: PieceType.FREEZER,
    AuraType.PUSH: PieceType.WHITEHOLE,
    AuraType.PULL: PieceType.BLACKHOLE,
}

AURA_AFFECT_MAP = {
    AuraType.BUFF: "friendly",
    AuraType.DEBUFF: "enemy",
    AuraType.FREEZE: "enemy",
    AuraType.PUSH: "map",
    AuraType.PULL: "map",
}

class UnifiedAuraCache(CacheStatsMixin):
    __slots__ = (
        "_sources", "_coverage", "_affected_sets", "_maps", "_chains", "_undo_stack",
        "_board_ref", "_last_board_hash", "_dirty_flags", "_cache_manager",
        "_frozen_bitmap", "_source_tracking", "_buff_types"
    )

    _AURA_SQUARES_CACHE: Dict[Coord, List[Coord]] = {}  # Class-level precomputed cache

    def __init__(self, board: Optional["Board"] = None, cache_manager=None) -> None:
        super().__init__()
        if not self._AURA_SQUARES_CACHE:  # Populate once on first instance
            for x in range(9):
                for y in range(9):
                    for z in range(9):
                        sq = (x, y, z)
                        self._AURA_SQUARES_CACHE[sq] = [coord for coord in get_aura_squares(sq) if in_bounds(coord)]

        self._sources: Dict[AuraType, Dict[Color, Set[Coord]]] = {
            aura: {Color.WHITE: set(), Color.BLACK: set()} for aura in AuraType
        }

        self._coverage: Dict[AuraType, Dict[Color, np.ndarray]] = {
            aura: {
                Color.WHITE: np.zeros((9, 9, 9), dtype=np.uint8),
                Color.BLACK: np.zeros((9, 9, 9), dtype=np.uint8),
            } for aura in [AuraType.BUFF, AuraType.DEBUFF, AuraType.FREEZE]
        }

        self._affected_sets: Dict[AuraType, Dict[Color, Set[Coord]]] = {
            aura: {Color.WHITE: set(), Color.BLACK: set()} for aura in [AuraType.BUFF, AuraType.DEBUFF, AuraType.FREEZE]
        }

        self._maps: Dict[AuraType, Dict[Color, Dict[Coord, Coord]]] = {
            aura: {Color.WHITE: {}, Color.BLACK: {}} for aura in [AuraType.PUSH, AuraType.PULL]
        }

        self._chains: Dict[AuraType, Dict[Color, List[AuraEffect]]] = {
            aura: {Color.WHITE: [], Color.BLACK: []} for aura in [AuraType.PUSH, AuraType.PULL]
        }

        self._undo_stack: List[Tuple[Color, List[Tuple[Coord, Optional[Piece]]]]] = []

        self._board_ref: Optional[weakref.ref] = weakref.ref(board) if board else None
        self._last_board_hash: int = 0
        self._dirty_flags: Dict[str, bool] = {
            'sources': True, 'coverage': True, 'maps': True, 'chains': True
        }
        self._cache_manager = cache_manager

        self._frozen_bitmap = np.zeros((9, 9, 9), dtype=bool)

        self._source_tracking: Dict[Color, Dict[Coord, Set[Coord]]] = {
            Color.WHITE: {}, Color.BLACK: {}
        }

        self._buff_types: Dict[Color, Dict[Coord, str]] = {
            Color.WHITE: {}, Color.BLACK: {}
        }

        if board:
            self._full_rebuild(board)

    def _normalize_coords(self, sq: CoordLike) -> Tuple[np.ndarray, bool]:
        """Returns (array of shape (N,3), is_scalar)."""
        arr = np.asarray(sq, dtype=int)
        if arr.ndim == 1:
            if arr.shape[0] != 3:
                raise ValueError("Single coordinate must be length 3.")
            return arr[np.newaxis, :], True
        elif arr.ndim == 2:
            if arr.shape[1] != 3:
                raise ValueError("Each coordinate must have 3 elements (x,y,z).")
            return arr, False
        else:
            raise ValueError("Input must be a single Coord or sequence of Coords.")

    def is_buffed(self, sq: CoordLike, friendly_color: Color) -> Union[bool, np.ndarray]:
        if self._dirty_flags['coverage']:
            self._incremental_rebuild()
        if isinstance(sq, tuple) and len(sq) == 3:
            x, y, z = sq
            if not (0 <= x < 9 and 0 <= y < 9 and 0 <= z < 9):
                return False
            return self._coverage[AuraType.BUFF][friendly_color][z, y, x] > 0
        else:
            coords, is_scalar = self._normalize_coords(sq)
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            valid = (x >= 0) & (x < 9) & (y >= 0) & (y < 9) & (z >= 0) & (z < 9)
            result = np.zeros(coords.shape[0], dtype=bool)
            if np.any(valid):
                result[valid] = self._coverage[AuraType.BUFF][friendly_color][z[valid], y[valid], x[valid]] > 0
            return bool(result[0]) if is_scalar else result

    def get_buffed_squares(self, controller: Color) -> Set[Coord]:
        if self._dirty_flags['coverage']:
            self._incremental_rebuild()
        return self._affected_sets[AuraType.BUFF][controller].copy()

    def get_buff_type(self, sq: CoordLike, friendly_color: Color) -> Union[Optional[str], List[Optional[str]]]:
        if self._dirty_flags['coverage']:
            self._incremental_rebuild()
        if isinstance(sq, tuple) and len(sq) == 3:
            return self._buff_types[friendly_color].get(sq)
        else:
            coords, is_scalar = self._normalize_coords(sq)
            results = [self._buff_types[friendly_color].get(tuple(c)) for c in coords]
            return results[0] if is_scalar else results

    def get_buff_sources(self, sq: CoordLike, friendly_color: Color) -> Union[Set[Coord], List[Set[Coord]]]:
        if self._dirty_flags['coverage']:
            self._incremental_rebuild()
        if isinstance(sq, tuple) and len(sq) == 3:
            return self._source_tracking[friendly_color].get(sq, set())
        else:
            coords, is_scalar = self._normalize_coords(sq)
            results = [self._source_tracking[friendly_color].get(tuple(c), set()) for c in coords]
            return results[0] if is_scalar else results

    def is_debuffed(self, sq: CoordLike, victim_color: Color) -> Union[bool, np.ndarray]:
        if self._dirty_flags['coverage']:
            self._incremental_rebuild()
        if isinstance(sq, tuple) and len(sq) == 3:
            x, y, z = sq
            if not (0 <= x < 9 and 0 <= y < 9 and 0 <= z < 9):
                return False
            return self._coverage[AuraType.DEBUFF][victim_color][z, y, x] > 0
        else:
            coords, is_scalar = self._normalize_coords(sq)
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            valid = (x >= 0) & (x < 9) & (y >= 0) & (y < 9) & (z >= 0) & (z < 9)
            result = np.zeros(coords.shape[0], dtype=bool)
            if np.any(valid):
                result[valid] = self._coverage[AuraType.DEBUFF][victim_color][z[valid], y[valid], x[valid]] > 0
            return bool(result[0]) if is_scalar else result

    def get_debuffed_squares(self, controller: Color) -> Set[Coord]:
        if self._dirty_flags['coverage']:
            self._incremental_rebuild()
        return self._affected_sets[AuraType.DEBUFF][controller].copy()

    def is_frozen(self, sq: CoordLike, victim: Color) -> Union[bool, np.ndarray]:
        if self._dirty_flags['coverage']:
            self._incremental_rebuild()
        if isinstance(sq, tuple) and len(sq) == 3:
            x, y, z = sq
            if not (0 <= x < 9 and 0 <= y < 9 and 0 <= z < 9):
                return False
            return self._coverage[AuraType.FREEZE][victim][z, y, x] > 0
        else:
            coords, is_scalar = self._normalize_coords(sq)
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            valid = (x >= 0) & (x < 9) & (y >= 0) & (y < 9) & (z >= 0) & (z < 9)
            result = np.zeros(coords.shape[0], dtype=bool)
            if np.any(valid):
                result[valid] = self._coverage[AuraType.FREEZE][victim][z[valid], y[valid], x[valid]] > 0
            return bool(result[0]) if is_scalar else result

    def get_frozen_squares(self, victim: Color) -> Set[Coord]:
        if self._dirty_flags['coverage']:
            self._incremental_rebuild()
        return self._affected_sets[AuraType.FREEZE][victim].copy()

    @property
    def frozen_bitmap(self) -> np.ndarray:
        if self._dirty_flags['coverage']:
            self._incremental_rebuild()
        return self._frozen_bitmap

    def push_map(self, controller: Color) -> Dict[Coord, Coord]:
        if self._dirty_flags["maps"]:
            self._incremental_rebuild()
        return self._maps[AuraType.PUSH][controller].copy()

    def get_push_targets(self, controller: Color, from_square: Coord) -> List[Coord]:
        if self._dirty_flags["chains"]:
            self._incremental_rebuild()
        targets = []
        if from_square in self._maps[AuraType.PUSH][controller]:
            targets.append(self._maps[AuraType.PUSH][controller][from_square])
        for eff in self._chains[AuraType.PUSH][controller]:
            if eff.source_square == from_square:
                targets.extend(eff.affected_squares)
        return targets

    def is_affected_by_white_hole(self, square: Coord, controller: Color) -> bool:
        if self._dirty_flags["maps"]:
            self._incremental_rebuild()
        affected = self._affected_sets.get(AuraType.PUSH, {controller: set()})[controller]
        return square in affected

    def pull_map(self, controller: Color) -> Dict[Coord, Coord]:
        if self._dirty_flags["maps"]:
            self._incremental_rebuild()
        return self._maps[AuraType.PULL][controller].copy()

    def get_pull_targets(self, controller: Color, from_square: Coord) -> List[Coord]:
        if self._dirty_flags["chains"]:
            self._incremental_rebuild()
        targets = []
        if from_square in self._maps[AuraType.PULL][controller]:
            targets.append(self._maps[AuraType.PULL][controller][from_square])
        for eff in self._chains[AuraType.PULL][controller]:
            if eff.source_square == from_square:
                targets.extend(eff.affected_squares)
        return targets

    def is_affected_by_black_hole(self, square: Coord, controller: Color) -> bool:
        if self._dirty_flags["maps"]:
            self._incremental_rebuild()
        affected = self._affected_sets.get(AuraType.PULL, {controller: set()})[controller]
        return square in affected

    def apply_move(self, mv: Move, mover: Color, current_ply: int, board: "Board") -> None:
        if mv is None:
            self._dirty_flags['coverage'] = True
            self._dirty_flags['maps'] = True
            return

        updated_coverage = False
        updated_maps = False
        from_piece = get_piece(self._cache_manager, mv.from_coord)
        captured_piece = get_piece(self._cache_manager, mv.to_coord) if mv.is_capture else None

        for aura, ptype in AURA_PIECE_MAP.items():
            affect = AURA_AFFECT_MAP[aura]

            if from_piece and from_piece.ptype == ptype and from_piece.color == mover:
                if affect == "map":
                    if self._move_affects_aura(aura, mv, board):
                        self._record_undo_snapshot(mover, board, aura)
                        self._apply_map_effects(aura, mover, board)
                        self._incremental_update_map(aura, mv, mover, board)
                        updated_maps = True
                else:
                    victim = mover if affect == "friendly" else mover.opposite()
                    self._incremental_update_coverage(aura, victim, old_sq=mv.from_coord, new_sq=mv.to_coord)
                    updated_coverage = True
                    self._sources[aura][mover].discard(mv.from_coord)
                    self._sources[aura][mover].add(mv.to_coord)

            if captured_piece and captured_piece.ptype == ptype:
                captured_color = captured_piece.color
                if affect == "map":
                    if self._move_affects_aura(aura, mv, board):
                        self._record_undo_snapshot(mover, board, aura)
                        self._apply_map_effects(aura, mover, board)
                        self._incremental_update_map(aura, mv, mover, board)
                        updated_maps = True
                else:
                    victim = captured_color if affect == "friendly" else captured_color.opposite()
                    self._incremental_update_coverage(aura, victim, old_sq=mv.to_coord)
                    updated_coverage = True
                    self._sources[aura][captured_color].discard(mv.to_coord)

        if updated_coverage or updated_maps:
            self._update_frozen_bitmap()

    def undo_move(self, mv: Move, mover: Color, board: "Board") -> None:
        for aura in [AuraType.PUSH, AuraType.PULL]:
            if self._move_affects_aura(aura, mv, board):
                self._restore_undo_snapshot(mover, board, aura)
                self._minimal_rebuild_map(aura, mv, mover, board)

    def _update_coverage(self, aura: AuraType, color: Color, old_sq: Optional[Coord] = None, new_sq: Optional[Coord] = None) -> None:
        coverage = self._coverage[aura][color]
        if old_sq:
            aura_coords = self._AURA_SQUARES_CACHE.get(old_sq, [])
            if aura_coords:
                aura_array = np.array(aura_coords, dtype=int)
                ax, ay, az = aura_array.T
                valid_mask = (ax >= 0) & (ax < 9) & (ay >= 0) & (ay < 9) & (az >= 0) & (az < 9)
                ax, ay, az = ax[valid_mask], ay[valid_mask], az[valid_mask]
                coverage[az, ay, ax] = np.maximum(0, coverage[az, ay, ax] - 1)
        if new_sq:
            aura_coords = self._AURA_SQUARES_CACHE.get(new_sq, [])
            if aura_coords:
                aura_array = np.array(aura_coords, dtype=int)
                ax, ay, az = aura_array.T
                valid_mask = (ax >= 0) & (ax < 9) & (ay >= 0) & (ay < 9) & (az >= 0) & (az < 9)
                ax, ay, az = ax[valid_mask], ay[valid_mask], az[valid_mask]
                coverage[az, ay, ax] += 1

    def _update_affected_set(self, aura: AuraType, color: Color, old_aura: Set[Coord] = set(), new_aura: Set[Coord] = set()) -> None:
        affected = self._affected_sets[aura][color]
        coverage = self._coverage[aura][color]
        changed = old_aura | new_aura
        is_freeze = aura == AuraType.FREEZE
        for sq in changed:
            if 0 <= sq[0] < 9 and 0 <= sq[1] < 9 and 0 <= sq[2] < 9:
                x, y, z = sq
                count = coverage[z, y, x]
                if count > 0:
                    if is_freeze:
                        target = get_piece(self._cache_manager, sq)
                        if target and target.color == color:
                            affected.add(sq)
                    else:
                        affected.add(sq)
                else:
                    affected.discard(sq)

    def _update_frozen_bitmap(self) -> None:
        self._frozen_bitmap = np.logical_or(
            self._coverage[AuraType.FREEZE][Color.WHITE] > 0,
            self._coverage[AuraType.FREEZE][Color.BLACK] > 0
        )

    def _move_affects_aura(self, aura: AuraType, mv: Move, board: "Board") -> bool:
        moved_piece = get_piece(self._cache_manager, mv.from_coord)
        dest_piece = get_piece(self._cache_manager, mv.to_coord)
        if moved_piece and moved_piece.ptype == AURA_PIECE_MAP[aura]:
            return True
        if dest_piece and dest_piece.ptype == AURA_PIECE_MAP[aura]:
            return True
        affected = self._affected_sets.get(aura, {Color.WHITE: set(), Color.BLACK: set()})
        for color in (Color.WHITE, Color.BLACK):
            if mv.from_coord in affected[color] or mv.to_coord in affected[color]:
                return True
        return False

    def _incremental_update_map(self, aura: AuraType, mv: Move, mover: Color, board: "Board") -> None:
        self._update_sources(aura, board)
        self._dirty_flags['maps'] = True

    def _minimal_rebuild_map(self, aura: AuraType, mv: Move, mover: Color, board: "Board") -> None:
        old_sources = {color: self._sources[aura][color].copy() for color in (Color.WHITE, Color.BLACK)}
        self._update_sources(aura, board)
        for color in (Color.WHITE, Color.BLACK):
            if old_sources[color] != self._sources[aura][color]:
                self._rebuild_map_for_color(aura, board, color)

    def _update_sources(self, aura: AuraType, board: "Board") -> None:
        ptype = AURA_PIECE_MAP[aura]
        for color in (Color.WHITE, Color.BLACK):
            self._sources[aura][color].clear()
            positions = self._cache_manager.occupancy.get_positions_by_type(color, ptype)
            for sq in positions:
                self._sources[aura][color].add(tuple(sq))  # Ensure Coord is tuple

    def _rebuild_map_for_color(self, aura: AuraType, board: "Board", color: Color) -> None:
        self._maps[aura][color].clear()
        self._affected_sets.setdefault(aura, {Color.WHITE: set(), Color.BLACK: set()})[color].clear()
        self._chains.setdefault(aura, {Color.WHITE: [], Color.BLACK: []})[color].clear()

        if aura == AuraType.PUSH:
            candidates = push_candidates(self._cache_manager, color)
        else:
            candidates = suck_candidates(self._cache_manager, color)

        for fr, to in candidates.items():
            self._maps[aura][color][fr] = to
            self._affected_sets[aura][color].update((fr, to))
            self._chains[aura][color].append(AuraEffect(fr, {fr}, AuraPriority.HIGH))

    def _full_rebuild(self, board: "Board") -> None:
        for aura in AuraType:
            affect = AURA_AFFECT_MAP[aura]
            if affect == "map":
                for color in (Color.WHITE, Color.BLACK):
                    self._rebuild_map_for_color(aura, board, color)
            else:
                for victim in (Color.WHITE, Color.BLACK):
                    self._coverage[aura][victim].fill(0)
                    self._affected_sets[aura][victim].clear()
                    controller = victim if affect == "friendly" else victim.opposite()
                    positions = [
                        coord for coord, piece in self._cache_manager.occupancy.iter_color(controller)
                        if piece.ptype == AURA_PIECE_MAP[aura]
                    ]
                    for sq in positions:
                        self._update_coverage(aura, victim, new_sq=tuple(sq))  # Ensure Coord is tuple
        self._update_frozen_bitmap()
        for flag in self._dirty_flags:
            self._dirty_flags[flag] = False

    def _incremental_rebuild(self) -> None:
        board = self._get_board()
        if board is None:
            return
        if self._dirty_flags['sources']:
            for aura in AuraType:
                self._update_sources(aura, board)
        if self._dirty_flags['coverage']:
            for aura in [AuraType.BUFF, AuraType.DEBUFF, AuraType.FREEZE]:
                for color in (Color.WHITE, Color.BLACK):
                    if len(self._affected_sets[aura][color]) == 0:
                        affect = AURA_AFFECT_MAP[aura]
                        controller = color if affect == "friendly" else color.opposite()
                        self._rebuild_coverage_for_color(aura, board, controller, color)
        if self._dirty_flags['maps']:
            for aura in [AuraType.PUSH, AuraType.PULL]:
                for color in (Color.WHITE, Color.BLACK):
                    self._rebuild_map_for_color(aura, board, color)
        for flag in self._dirty_flags:
            self._dirty_flags[flag] = False

    def _rebuild_coverage_for_color(self, aura: AuraType, board: "Board", controller: Color, victim: Color) -> None:
        self._coverage[aura][victim].fill(0)
        self._affected_sets[aura][victim].clear()
        ptype = AURA_PIECE_MAP[aura]
        sources = self._cache_manager.occupancy.get_positions_by_type(controller, ptype)

        if not sources:
            return

        # Precompute the aura squares for all sources and flatten
        all_aura_coords = []
        for src in sources:
            aura_squares = self._AURA_SQUARES_CACHE.get(tuple(src), [])
            all_aura_coords.extend(aura_squares)

        if not all_aura_coords:
            return

        # Convert to numpy array and filter valid coords
        all_aura_coords = np.array(all_aura_coords, dtype=int)
        valid_mask = np.all((all_aura_coords >= 0) & (all_aura_coords < 9), axis=1)
        valid_coords = all_aura_coords[valid_mask]

        if valid_coords.size == 0:
            return

        # Now we have a (N, 3) array of coordinates (x, y, z)
        x, y, z = valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]
        # Calculate linear indices in the coverage array (which is shape (9,9,9) indexed (z,y,x))
        linear_indices = z * 81 + y * 9 + x

        np.add.at(self._coverage[aura][victim].flat, linear_indices, 1)

        # Now update the affected set
        positive_mask = self._coverage[aura][victim] > 0
        if aura == AuraType.FREEZE:
            # We need to check which of these squares are occupied by the victim color
            # Get all positive coordinates
            az, ay, ax = np.where(positive_mask)
            # Convert to list of coordinates (x, y, z)
            positive_coords = list(zip(ax, ay, az))
            # Check occupancy for each coordinate
            for coord in positive_coords:
                piece = self._cache_manager.occupancy.get(coord)
                if piece and piece.color == victim:
                    self._affected_sets[aura][victim].add(coord)
        else:
            az, ay, ax = np.where(positive_mask)
            self._affected_sets[aura][victim].update(zip(ax, ay, az))

    def _get_board(self) -> Optional["Board"]:
        if self._board_ref is None:
            return None
        board = self._board_ref()
        if board is None:
            self._board_ref = None
        return board

    def get_stats(self) -> Dict[str, Any]:
        board = self._get_board()
        stats = {}
        for aura in AuraType:
            for color in (Color.WHITE, Color.BLACK):
                stats[f"{aura.value}_sources_{color.name.lower()}"] = len(self._sources[aura][color])
        stats["dirty_flags"] = self._dirty_flags.copy()
        stats["board_hash"] = board.byte_hash() if board else 0
        return stats

    def clear(self) -> None:
        for aura in AuraType:
            for color in (Color.WHITE, Color.BLACK):
                self._sources[aura][color].clear()
                if aura in self._coverage:
                    self._coverage[aura][color].fill(0)
                if aura in self._affected_sets:
                    self._affected_sets[aura][color].clear()
                if aura in self._maps:
                    self._maps[aura][color].clear()
                if aura in self._chains:
                    self._chains[aura][color].clear()
        self._undo_stack.clear()
        self._last_board_hash = 0
        for flag in self._dirty_flags:
            self._dirty_flags[flag] = True
        self._frozen_bitmap.fill(False)

    def _incremental_update_coverage(self, aura: AuraType, victim: Color, old_sq: Optional[Coord], new_sq: Optional[Coord]):
        coverage = self._coverage[aura][victim]
        affected = self._affected_sets[aura][victim]

        def adjust_aura(sq: Coord, delta: int):
            if sq is None or not (0 <= sq[0] < 9 and 0 <= sq[1] < 9 and 0 <= sq[2] < 9):
                return set()
            aura_coords = self._AURA_SQUARES_CACHE.get(sq, [])
            if not aura_coords:
                return set()
            aura_array = np.array(aura_coords, dtype=int)  # Shape: (N, 3)
            valid_mask = np.all((aura_array >= 0) & (aura_array < 9), axis=1)
            if not np.any(valid_mask):
                return set()
            valid_coords = aura_array[valid_mask]  # Nx3
            az, ay, ax = valid_coords.T  # Transpose for indexing
            old_values = coverage[az, ay, ax].copy()
            coverage[az, ay, ax] += delta  # Vectorized add
            changed_mask = old_values != coverage[az, ay, ax]
            changed_coords = set(zip(ax[changed_mask], ay[changed_mask], az[changed_mask]))
            return changed_coords

        changed = set()
        if old_sq:
            changed.update(adjust_aura(old_sq, -1))
        if new_sq:
            changed.update(adjust_aura(new_sq, 1))

        for sq in changed:
            if 0 <= sq[0] < 9 and 0 <= sq[1] < 9 and 0 <= sq[2] < 9:
                x, y, z = sq
                count = coverage[z, y, x]
                if count > 0:
                    if aura == AuraType.FREEZE:
                        target = get_piece(self._cache_manager, sq)
                        if target and target.color == victim:
                            affected.add(sq)
                    else:
                        affected.add(sq)
                else:
                    affected.discard(sq)

    def _ensure_built(self) -> None:
        if any(self._dirty_flags.values()):
            self._incremental_rebuild()

    def apply_freeze_effects(self, controller: Color, board: "Board") -> Set[Coord]:
        self._ensure_built()
        victim = controller.opposite()
        frozen_now = set()
        for sq in self._affected_sets[AuraType.FREEZE][victim]:
            piece = get_piece(self._cache_manager, sq)
            if piece and piece.color == victim:
                frozen_now.add(sq)
        return frozen_now

    def apply_push_effects(self, controller: Color, board: "Board") -> Set[Coord]:
        return self._apply_map_effects(AuraType.PUSH, controller, board)

    def apply_pull_effects(self, controller: Color, board: "Board") -> Set[Coord]:
        return self._apply_map_effects(AuraType.PULL, controller, board)

    def _apply_map_effects(self, aura: AuraType, controller: Color, board: "Board") -> Set[Coord]:
        self._ensure_built()
        changed: Set[Coord] = set()

        if aura not in self._maps:
            return changed

        cmap = self._maps[aura][controller]

        for fr, to in cmap.items():
            piece = get_piece(self._cache_manager, fr)
            if piece is None:
                continue
            if piece.color == controller:
                continue
            if not in_bounds(to):
                continue

            board.set_piece(to, piece)
            board.set_piece(fr, None)
            self._cache_manager.occupancy.set_position(fr, None)
            self._cache_manager.occupancy.set_position(to, piece)
            changed.update({fr, to})

        if changed:
            self._record_undo_snapshot(controller, board, aura)

        return changed

    def _record_undo_snapshot(self, controller: Color, board: "Board", aura: AuraType) -> None:
        snapshot = []
        for coord in self._affected_sets[aura][controller]:
            piece = get_piece(self._cache_manager, coord)
            snapshot.append((coord, piece))
        self._undo_stack.append((controller, snapshot))

    def _restore_undo_snapshot(self, controller: Color, board: "Board", aura: AuraType) -> None:
        if not self._undo_stack:
            return

        _, snapshot = self._undo_stack.pop()
        for coord, piece in snapshot:
            if piece:
                board.set_piece(coord, piece)
                self._cache_manager.occupancy.set_position(coord, piece)
            else:
                board.set_piece(coord, None)
                self._cache_manager.occupancy.set_position(coord, None)

    def set_cache_manager(self, cache_manager: 'OptimizedCacheManager') -> None:
        """Set the cache manager reference - ensures single instance usage"""
        self._cache_manager = cache_manager

def create_unified_aura_cache(board: Optional["Board"] = None, cache_manager=None) -> UnifiedAuraCache:
    return UnifiedAuraCache(board, cache_manager)

def init_aura_cache(cache_manager=None) -> None:
    global _aura_cache
    _aura_cache = UnifiedAuraCache(None, cache_manager)

def get_aura_cache() -> UnifiedAuraCache:
    if _aura_cache is None:
        raise RuntimeError("UnifiedAuraCache not initialised")
    return _aura_cache
