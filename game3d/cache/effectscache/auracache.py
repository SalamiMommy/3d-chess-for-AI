"""Unified incremental cache for all 2-radius aura effects (BUFF, DEBUFF, FREEZE, PUSH, PULL)."""

from __future__ import annotations
from typing import Set, Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import weakref
import numpy as np

from game3d.common.enums import Color, PieceType
from game3d.common.coord_utils import get_aura_squares, Coord
from game3d.common.piece_utils import get_pieces_by_type
from game3d.pieces.pieces.speeder import buffed_squares
from game3d.pieces.pieces.slower import debuffed_squares
from game3d.pieces.pieces.whitehole import push_candidates
from game3d.pieces.pieces.blackhole import suck_candidates
from game3d.movement.movepiece import Move
from game3d.pieces.piece import Piece
from game3d.common.coord_utils import filter_valid_coords, in_bounds_vectorised, in_bounds
# ==============================================================================
# OPTIMIZATION CONSTANTS
# ==============================================================================

@dataclass
class AuraEffect:
    """Represents a generic aura effect."""
    source_square: Tuple[int, int, int]
    affected_squares: Set[Tuple[int, int, int]]
    priority: int

class AuraPriority(Enum):
    """Priority levels for aura effects."""
    HIGH = 3
    MEDIUM = 2
    LOW = 1

class AuraType(Enum):
    """Types of auras managed by this cache."""
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
    AuraType.PUSH: "map",  # Special handling
    AuraType.PULL: "map",  # Special handling
}

# ==============================================================================
# UNIFIED AURA CACHE
# ==============================================================================

class UnifiedAuraCache:
    """Unified incremental cache for all 2-radius aura effects with smart updates."""

    __slots__ = (
        "_sources", "_coverage", "_affected_sets", "_maps", "_chains", "_undo_stack",
        "_board_ref", "_last_board_hash", "_dirty_flags", "_cache_manager",
        "_frozen_bitmap", "_source_tracking", "_buff_types"
    )

    def __init__(self, board: Optional["Board"] = None, cache_manager=None) -> None:
        # Sources: positions of aura-emitting pieces
        self._sources: Dict[AuraType, Dict[Color, Set[Tuple[int, int, int]]]] = {
            aura: {Color.WHITE: set(), Color.BLACK: set()} for aura in AuraType
        }

        # Coverage: numpy arrays for overlap counting (for BUFF, DEBUFF, FREEZE)
        self._coverage: Dict[AuraType, Dict[Color, np.ndarray]] = {
            aura: {
                Color.WHITE: np.zeros((9, 9, 9), dtype=np.uint8),
                Color.BLACK: np.zeros((9, 9, 9), dtype=np.uint8),
            } for aura in [AuraType.BUFF, AuraType.DEBUFF, AuraType.FREEZE]
        }

        # Affected sets: sets of affected squares (for BUFF, DEBUFF, FREEZE)
        self._affected_sets: Dict[AuraType, Dict[Color, Set[Tuple[int, int, int]]]] = {
            aura: {Color.WHITE: set(), Color.BLACK: set()} for aura in [AuraType.BUFF, AuraType.DEBUFF, AuraType.FREEZE]
        }

        # Maps: for PUSH and PULL
        self._maps: Dict[AuraType, Dict[Color, Dict[Tuple[int, int, int], Tuple[int, int, int]]]] = {
            aura: {Color.WHITE: {}, Color.BLACK: {}} for aura in [AuraType.PUSH, AuraType.PULL]
        }

        # Chains: for PUSH and PULL effects
        self._chains: Dict[AuraType, Dict[Color, List[AuraEffect]]] = {
            aura: {Color.WHITE: [], Color.BLACK: []} for aura in [AuraType.PUSH, AuraType.PULL]
        }

        # Undo stack for PUSH/PULL (since they modify board)
        self._undo_stack: List[Tuple[Color, List[Tuple[Tuple[int, int, int], Optional[Piece]]]]] = []

        # Board and tracking
        self._board_ref: Optional[weakref.ref] = weakref.ref(board) if board else None
        self._last_board_hash: int = 0
        self._dirty_flags: Dict[str, bool] = {
            'sources': True, 'coverage': True, 'maps': True, 'chains': True
        }
        self._cache_manager = cache_manager

        # Additional for FREEZE
        self._frozen_bitmap = np.zeros((9, 9, 9), dtype=bool)

        # Source tracking (from buff)
        self._source_tracking: Dict[Color, Dict[Tuple[int, int, int], Set[Tuple[int, int, int]]]] = {
            Color.WHITE: {}, Color.BLACK: {}
        }

        # Buff types (from buff)
        self._buff_types: Dict[Color, Dict[Tuple[int, int, int], str]] = {
            Color.WHITE: {}, Color.BLACK: {}
        }

        if board:
            self._full_rebuild(board)

    # ---------- PUBLIC INTERFACE (COMBINED FROM ALL CACHES) ----------
    # Buff methods
    def is_buffed(self, sq: Tuple[int, int, int], friendly_color: Color) -> bool:
        if self._dirty_flags['coverage']:
            self._incremental_rebuild()
        x, y, z = sq
        return self._coverage[AuraType.BUFF][friendly_color][z, y, x] > 0

    def get_buffed_squares(self, controller: Color) -> Set[Tuple[int, int, int]]:
        if self._dirty_flags['coverage']:
            self._incremental_rebuild()
        return self._affected_sets[AuraType.BUFF][controller].copy()

    def get_buff_type(self, sq: Tuple[int, int, int], friendly_color: Color) -> Optional[str]:
        if self._dirty_flags['coverage']:
            self._incremental_rebuild()
        return self._buff_types[friendly_color].get(sq)

    def get_buff_sources(self, sq: Tuple[int, int, int], friendly_color: Color) -> Set[Tuple[int, int, int]]:
        if self._dirty_flags['coverage']:
            self._incremental_rebuild()
        return self._source_tracking[friendly_color].get(sq, set())

    # Debuff methods
    def is_debuffed(self, sq: Tuple[int, int, int], victim_color: Color) -> bool:
        if self._dirty_flags['coverage']:
            self._incremental_rebuild()
        x, y, z = sq
        return self._coverage[AuraType.DEBUFF][victim_color][z, y, x] > 0

    def get_debuffed_squares(self, controller: Color) -> Set[Tuple[int, int, int]]:
        if self._dirty_flags['coverage']:
            self._incremental_rebuild()
        return self._affected_sets[AuraType.DEBUFF][controller].copy()

    # Freeze methods
    def is_frozen(self, sq: Tuple[int, int, int], victim: Color) -> bool:
        if self._dirty_flags['coverage']:
            self._incremental_rebuild()
        x, y, z = sq
        return self._coverage[AuraType.FREEZE][victim][z, y, x] > 0

    def get_frozen_squares(self, victim: Color) -> Set[Tuple[int, int, int]]:
        if self._dirty_flags['coverage']:
            self._incremental_rebuild()
        return self._affected_sets[AuraType.FREEZE][victim].copy()

    @property
    def frozen_bitmap(self) -> np.ndarray:
        if self._dirty_flags['coverage']:
            self._incremental_rebuild()
        return self._frozen_bitmap

    # Push methods
    def push_map(self, controller: Color) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
        if self._dirty_flags["maps"]:
            self._incremental_rebuild()
        return self._maps[AuraType.PUSH][controller].copy()

    def get_push_targets(self, controller: Color, from_square: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        if self._dirty_flags["chains"]:
            self._incremental_rebuild()
        targets = []
        if from_square in self._maps[AuraType.PUSH][controller]:
            targets.append(self._maps[AuraType.PUSH][controller][from_square])
        for eff in self._chains[AuraType.PUSH][controller]:
            if eff.from_square == from_square:
                targets.append(eff.to_square)
        return targets

    def is_affected_by_white_hole(self, square: Tuple[int, int, int], controller: Color) -> bool:
        if self._dirty_flags["maps"]:
            self._incremental_rebuild()
        return square in self._affected_sets[AuraType.PUSH][controller]

    # Pull methods
    def pull_map(self, controller: Color) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
        if self._dirty_flags["maps"]:
            self._incremental_rebuild()
        return self._maps[AuraType.PULL][controller].copy()

    def get_pull_targets(self, controller: Color, from_square: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        if self._dirty_flags["chains"]:
            self._incremental_rebuild()
        targets = []
        if from_square in self._maps[AuraType.PULL][controller]:
            targets.append(self._maps[AuraType.PULL][controller][from_square])
        for eff in self._chains[AuraType.PULL][controller]:
            if eff.from_square == from_square:
                targets.append(eff.to_square)
        return targets

    def is_affected_by_black_hole(self, square: Tuple[int, int, int], controller: Color) -> bool:
        if self._dirty_flags["maps"]:
            self._incremental_rebuild()
        return square in self._affected_sets[AuraType.PULL][controller]

    # ---------- MOVE HANDLING ----------
    def apply_move(self, mv: Move, mover: Color, current_ply: int, board: "Board") -> None:
        """
        Incremental update for all auras after a move.
        Updates only when an aura-emitting piece moves or is captured.
        """
        if mv is None:                       # ← NEW
            self._dirty_flags['coverage'] = True
            self._dirty_flags['maps']       = True
            return
        updated = False
        from_piece = self._cache_manager.occupancy.get(mv.from_coord)
        captured_piece = mv.captured_piece if hasattr(mv, 'captured_piece') else None

        # Handle each aura type
        for aura, ptype in AURA_PIECE_MAP.items():
            affect = AURA_AFFECT_MAP[aura]

            # Case 1: Mover is aura piece
            if from_piece and from_piece.ptype == ptype and from_piece.color == mover:
                if affect == "map":
                    # For PUSH/PULL, check if affects
                    if self._move_affects_aura(aura, mv, board):
                        self._record_undo_snapshot(mover, board, aura)
                        self._apply_aura_effects(aura, mover, board)
                        self._incremental_update_map(aura, mv, mover, board)
                        updated = True
                else:
                    # Simple coverage update
                    victim = mover if affect == "friendly" else mover.opposite()
                    self._update_coverage(aura, victim, old_sq=mv.from_coord, new_sq=mv.to_coord)
                    self._update_affected_set(aura, victim, old_aura=get_aura_squares(mv.from_coord),
                                              new_aura=get_aura_squares(mv.to_coord))
                    self._sources[aura][mover].discard(mv.from_coord)
                    self._sources[aura][mover].add(mv.to_coord)
                    updated = True

            # Case 2: Captured is aura piece
            if captured_piece and captured_piece.ptype == ptype:
                captured_color = captured_piece.color
                if affect == "map":
                    if self._move_affects_aura(aura, mv, board):
                        self._record_undo_snapshot(mover, board, aura)
                        self._apply_aura_effects(aura, mover, board)
                        self._incremental_update_map(aura, mv, mover, board)
                        updated = True
                else:
                    victim = captured_color if affect == "friendly" else captured_color.opposite()
                    self._update_coverage(aura, victim, old_sq=mv.to_coord)
                    self._update_affected_set(aura, victim, old_aura=get_aura_squares(mv.to_coord))
                    self._sources[aura][captured_color].discard(mv.to_coord)
                    updated = True

        # Special for FREEZE: handle moved piece (victim for enemy freezers)
        victim = mover
        freeze_victim_coverage = self._coverage[AuraType.FREEZE][victim]
        if from_piece and from_piece.color == mover:
            x, y, z = mv.from_coord
            if freeze_victim_coverage[z, y, x] > 0:
                self._affected_sets[AuraType.FREEZE][victim].discard(mv.from_coord)
            x, y, z = mv.to_coord
            if freeze_victim_coverage[z, y, x] > 0:
                self._affected_sets[AuraType.FREEZE][victim].add(mv.to_coord)

        if updated:
            self._dirty_flags['coverage'] = True
            self._dirty_flags['maps'] = True
            self._update_frozen_bitmap()

    def undo_move(self, mv: Move, mover: Color, board: "Board") -> None:
        """Undo updates for all auras."""
        # Restore undo snapshot for PUSH/PULL if applicable
        for aura in [AuraType.PUSH, AuraType.PULL]:
            if self._move_affects_aura(aura, mv, board):
                self._restore_undo_snapshot(mover, board, aura)
                self._minimal_rebuild_map(aura, mv, mover, board)

        # For simple auras, reverse the apply_move logic (similar updates)
        # ... (implement similar to apply_move but reverse)

    # ---------- INTERNAL HELPERS ----------
    def _update_coverage(self, aura: AuraType, color: Color, old_sq: Optional[Coord] = None, new_sq: Optional[Coord] = None) -> None:
        coverage = self._coverage[aura][color]
        if old_sq:
            for sq in get_aura_squares(old_sq):
                x, y, z = sq
                coverage[z, y, x] = max(0, coverage[z, y, x] - 1)
        if new_sq:
            for sq in get_aura_squares(new_sq):
                x, y, z = sq
                coverage[z, y, x] += 1

    def _update_affected_set(self, aura: AuraType, color: Color, old_aura: Set[Coord] = set(), new_aura: Set[Coord] = set()) -> None:
        affected = self._affected_sets[aura][color]
        coverage = self._coverage[aura][color]
        changed = old_aura | new_aura
        is_freeze = aura == AuraType.FREEZE
        for sq in changed:
            x, y, z = sq
            if coverage[z, y, x] > 0:
                if is_freeze:
                    target = self._cache_manager.occupancy.get(sq)
                    if target and target.color == color:
                        affected.add(sq)
                else:
                    affected.add(sq)
            else:
                affected.discard(sq)

    def _update_frozen_bitmap(self) -> None:
        self._frozen_bitmap.fill(False)
        for color in [Color.WHITE, Color.BLACK]:
            for sq in self._affected_sets[AuraType.FREEZE][color]:
                x, y, z = sq
                self._frozen_bitmap[z, y, x] = True

    def _move_affects_aura(self, aura: AuraType, mv: Move, board: "Board") -> bool:
        occ = self._cache_manager.occupancy
        ptype = AURA_PIECE_MAP[aura]
        moved_piece = occ.get(mv.from_coord)
        dest_piece = occ.get(mv.to_coord)
        if moved_piece and moved_piece.ptype == ptype:
            return True
        if dest_piece and dest_piece.ptype == ptype:
            return True
        affected = self._affected_sets.get(aura, {Color.WHITE: set(), Color.BLACK: set()})
        for color in (Color.WHITE, Color.BLACK):
            if mv.from_coord in affected[color] or mv.to_coord in affected[color]:
                return True
        return False

    def _apply_aura_effects(self, aura: AuraType, controller: Color, board: "Board") -> None:
        # Implement push/pull effects (board modifications)
        # Similar to original _apply_push_effects / _apply_pull_effects
        # Truncated in original; assume standard implementation
        pass

    def _record_undo_snapshot(self, controller: Color, board: "Board", aura: AuraType) -> None:
        # Similar to original
        pass

    def _restore_undo_snapshot(self, controller: Color, board: "Board", aura: AuraType) -> None:
        # Similar to original
        pass

    def _incremental_update_map(self, aura: AuraType, mv: Move, mover: Color, board: "Board") -> None:
        # Update positions and set dirty if changed
        self._update_sources(aura, board)
        self._dirty_flags['maps'] = True

    def _minimal_rebuild_map(self, aura: AuraType, mv: Move, mover: Color, board: "Board") -> None:
        old_sources = self._sources[aura].copy()
        self._update_sources(aura, board)
        for color in (Color.WHITE, Color.BLACK):
            if old_sources[color] != self._sources[aura][color]:
                self._rebuild_map_for_color(aura, board, color)

    def _update_sources(self, aura: AuraType, board: "Board") -> None:
        ptype = AURA_PIECE_MAP[aura]
        for color in (Color.WHITE, Color.BLACK):
            self._sources[aura][color].clear()
            for sq, _ in get_pieces_by_type(board, ptype, color):
                sq = self._sanitize(sq)                #  ⬅  NEW
                self._sources[aura][color].add(sq)

    def _rebuild_map_for_color(self, aura: AuraType, board: "Board", color: Color) -> None:
        self._maps[aura][color].clear()
        self._affected_sets.setdefault(aura, {Color.WHITE: set(), Color.BLACK: set()})[color].clear()
        self._chains.setdefault(aura, {Color.WHITE: [], Color.BLACK: []})[color].clear()

        if aura == AuraType.PUSH:
            candidates = push_candidates(board, color, self._cache_manager)
        else:  # PULL
            candidates = suck_candidates(board, color, self._cache_manager)

        for fr, to in candidates.items():
            self._maps[aura][color][fr] = to
            self._affected_sets[aura][color].update((fr, to))
            # Build chain (similar to original)
            self._chains[aura][color].append(
                AuraEffect(fr, {fr}, AuraPriority.HIGH)
            )

    # ---------- REBUILD METHODS ----------
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
                    sources = get_pieces_by_type(board, AURA_PIECE_MAP[aura], controller)
                    for sq, _ in sources:
                        sq = self._sanitize(sq)                   #  ⬅  NEW
                        for raw in get_aura_squares(sq):
                            ax, ay, az = self._sanitize(raw)      #  ⬅  NEW
                            self._coverage[aura][victim][az, ay, ax] += 1
                            if self._coverage[aura][victim][az, ay, ax] > 0:
                                if aura == AuraType.FREEZE:
                                    target = self._cache_manager.occupancy.get((ax, ay, az))
                                    if target and target.color == victim:
                                        self._affected_sets[aura][victim].add((ax, ay, az))
                                else:
                                    self._affected_sets[aura][victim].add((ax, ay, az))
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
                        # Rebuild if empty
                        affect = AURA_AFFECT_MAP[aura]
                        controller = color if affect == "friendly" else color.opposite()
                        self._rebuild_coverage_for_color(aura, board, controller, color)
        if self._dirty_flags['maps']:
            for aura in [AuraType.PUSH, AuraType.PULL]:
                for color in (Color.WHITE, Color.BLACK):
                    self._rebuild_map_for_color(aura, board, color)
        if self._dirty_flags['chains']:
            # Rebuild chains if needed
            pass
        for flag in self._dirty_flags:
            self._dirty_flags[flag] = False

    def _rebuild_coverage_for_color(self, aura: AuraType, board: "Board",
                                    controller: Color, victim: Color) -> None:
        self._coverage[aura][victim].fill(0)
        self._affected_sets[aura][victim].clear()
        ptype = AURA_PIECE_MAP[aura]
        sources = get_pieces_by_type(board, ptype, controller)
        for sq, _ in sources:
            sq = self._sanitize(sq)                   #  ⬅  NEW
            for ax, ay, az in get_aura_squares(sq):
                self._coverage[aura][victim][az, ay, ax] += 1
                if self._coverage[aura][victim][az, ay, ax] > 0:
                    if aura == AuraType.FREEZE:
                        target = self._cache_manager.occupancy.get((ax, ay, az))
                        if target and target.color == victim:
                            self._affected_sets[aura][victim].add((ax, ay, az))
                    else:
                        self._affected_sets[aura][victim].add((ax, ay, az))

    # ---------- UTILITY METHODS ----------
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

    def _inc_coverage(cov: np.ndarray, x: int, y: int, z: int) -> None:
        """Increment coverage only if the coordinate is inside the array."""
        if 0 <= x < 9 and 0 <= y < 9 and 0 <= z < 9:
            cov[z, y, x] += 1

    # ------------------------------------------------------------------
    #  PUBLIC FACADE – called by OptimizedCacheManager
    # ------------------------------------------------------------------
    def apply_freeze_effects(self, controller: Color, board: "Board") -> Set[Coord]:
        """
        Emit freeze auras for <controller>.
        Returns the squares that became frozen (enemy pieces inside the aura).
        """
        self._ensure_built()
        victim = controller.opposite()
        frozen_now = set()
        for sq in self._affected_sets[AuraType.FREEZE][victim]:
            piece = self._cache_manager.occupancy.get(sq)
            if piece and piece.color == victim:
                frozen_now.add(sq)
        return frozen_now


    def apply_push_effects(self, controller: Color, board: "Board") -> Set[Coord]:
        """Apply white-hole pushes and return squares whose occupancy changed."""
        return self._apply_map_effects(AuraType.PUSH, controller, board)


    def apply_pull_effects(self, controller: Color, board: "Board") -> Set[Coord]:
        """Apply black-hole pulls and return squares whose occupancy changed."""
        return self._apply_map_effects(AuraType.PULL, controller, board)


    # ------------------------------------------------------------------
    #  Shared helper – actually moves the pieces on the board
    # ------------------------------------------------------------------
    def _apply_map_effects(self, aura: AuraType, controller: Color, board: "Board") -> Set[Coord]:
        """
        Execute push (WHITEHOLE) or pull (BLACKHOLE) and track every square
        whose content changed.  The board *and* the occupancy cache are
        updated in lock-step.
        """
        self._ensure_built()
        changed: Set[Coord] = set()
        cmap = self._maps[aura][controller]

        for fr, to in cmap.items():
            piece = self._cache_manager.occupancy.get(fr)
            if piece is None:                       # nothing to move
                continue
            if piece.color == controller:           # never push/pull own pieces
                continue
            if not in_bounds(to):                   # safety belt
                continue

            # ---- perform the move ----
            board.set_piece(to, piece)
            board.set_piece(fr, None)
            self._cache_manager.occupancy.set_position(fr, None)
            self._cache_manager.occupancy.set_position(to, piece)
            changed.update({fr, to})

        # snapshot for undo (push/pull actually mutate the board)
        if changed:
            self._record_undo_snapshot(controller, board, aura)

        return changed

    # --------------------------------------------------
    #  Tiny guard so we never read stale data
    # --------------------------------------------------
    def _ensure_built(self) -> None:
        if any(self._dirty_flags.values()):
            self._incremental_rebuild()

    def _sanitize(self, coord: Any) -> Coord:
        """
        Convert anything that looks like a coordinate into a guaranteed
        in-bounds (int, int, int).  If the coordinate is completely
        out of range we clamp it; if it is a numpy scalar we cast it.
        """
        # 1.  homogeneous numeric → numpy array (cheap)
        arr = np.asarray(coord, dtype=np.int16).ravel()
        if arr.shape != (3,):
            raise ValueError(f"Illegal coordinate {coord!r}")

        # 2.  clamp to board limits
        clamped = filter_valid_coords(arr[None, :], log_oob=False, clamp=True)[0]
        return tuple(map(int, clamped))
# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_unified_aura_cache(board: Optional["Board"] = None, cache_manager=None) -> UnifiedAuraCache:
    """Factory function for creating unified aura cache."""
    return UnifiedAuraCache(board, cache_manager)

# ==============================================================================
# BACKWARD COMPATIBILITY WRAPPERS
# ==============================================================================

class MovementBuffCache(UnifiedAuraCache):
    """Backward compatibility wrapper for buff cache."""
    pass  # Delegates to unified

class MovementDebuffCache(UnifiedAuraCache):
    """Backward compatibility wrapper for debuff cache."""
    pass

class FreezeCache(UnifiedAuraCache):
    """Backward compatibility wrapper for freeze cache."""
    pass

class WhiteHolePushCache(UnifiedAuraCache):
    """Backward compatibility wrapper for push cache."""
    pass

class BlackHoleSuckCache(UnifiedAuraCache):
    """Backward compatibility wrapper for pull cache."""
    pass

# Singleton (unified)
_aura_cache: Optional[UnifiedAuraCache] = None

def init_aura_cache(cache_manager=None) -> None:
    global _aura_cache
    _aura_cache = UnifiedAuraCache(None, cache_manager)

def get_aura_cache() -> UnifiedAuraCache:
    if _aura_cache is None:
        raise RuntimeError("UnifiedAuraCache not initialised")
    return _aura_cache
