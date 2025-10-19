# game3d/cache/effects_cache.py

"""Central management for all effect caches."""

from typing import Dict, Set, Any, Optional
from game3d.common.enums import Color, PieceType
from game3d.pieces.piece import Piece
from game3d.cache.effectscache.auracache import UnifiedAuraCache
from game3d.cache.effectscache.trailblazecache import TrailblazeCache
from game3d.cache.effectscache.geomancycache import GeomancyCache
from game3d.cache.caches.attackscache import AttacksCache
from game3d.common.common import Coord  # Added for type hints

class EffectsCache:
    """Manages all effect-specific caches."""

    def __init__(self, board, cache_manager):
        self.board = board
        self.cache_manager = cache_manager
        self._effect_caches: Dict[str, Any] = {}
        self._init_effects()

    def _init_effects(self) -> None:
        """Initialize all effect caches."""
        self._effect_caches = {
            "aura": UnifiedAuraCache(self.board, self.cache_manager),
            "trailblaze": TrailblazeCache(self.cache_manager),
            "geomancy": GeomancyCache(self.cache_manager),
            "attacks": AttacksCache(self.board),
        }

    def apply_freeze_effects(self, controller: Color, board: "Board") -> None:
        """
        Apply freeze effects from all friendly freezers.
        Called whenever ANY friendly piece moves.
        """
        aura_cache = self._effect_caches["aura"]
        aura_cache.apply_freeze_effects(controller, board)

    def get_affected_caches(self, mv: 'Move', mover: Color, from_piece: Piece,
                            to_piece: Optional[Piece], captured_piece: Optional[Piece], is_undo: bool = False) -> Set[str]:
        """Unified method for affected caches, with is_undo flag."""
        affected = set()

        aura_types = {PieceType.FREEZER, PieceType.SPEEDER, PieceType.SLOWER,
                      PieceType.BLACK_HOLE, PieceType.WHITE_HOLE}
        if from_piece and from_piece.ptype in aura_types | {PieceType.TRAILBLAZER, PieceType.GEOMANCER}:
            effect_map = {
                **{pt: "aura" for pt in aura_types},
                PieceType.TRAILBLAZER: "trailblaze",
                PieceType.GEOMANCER: "geomancy",
            }
            affected.add(effect_map[from_piece.ptype])

        if captured_piece:
            self._add_affected_effects_from_pos(mv.to_coord, affected)

        self._add_affected_effects_from_pos(mv.from_coord, affected)

        if is_undo:
            self._add_affected_effects_from_pos(mv.to_coord, affected)

        return affected

    def get_affected_caches_for_undo(self, mv: 'Move', mover: Color) -> Set[str]:
        return self.get_affected_caches(mv, mover, None, None, None, is_undo=True)

    def _add_affected_effects_from_pos(self, pos: Coord, affected: Set[str]) -> None:
        from game3d.common.common import BOARD_SIZE  # Assume constant for board size (9)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    check_pos = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
                    if all(0 <= c < BOARD_SIZE for c in check_pos):  # Use constant
                        piece = self.cache_manager.piece_cache.get(check_pos)
                        if piece and piece.ptype in {PieceType.FREEZER, PieceType.BLACK_HOLE,
                                                     PieceType.WHITE_HOLE, PieceType.GEOMANCER}:
                            effect_map = {
                                PieceType.FREEZER: "aura",
                                PieceType.BLACK_HOLE: "aura",
                                PieceType.WHITE_HOLE: "aura",
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
        """
        Incremental update: tell the move-cache *precisely* which squares
        need to be regenerated instead of rebuilding the whole thing
        """
        for cache_name in affected_caches:
            cache = self._effect_caches[cache_name]
            if hasattr(cache, 'apply_move'):
                cache.apply_move(mv, mover, self.board)

    def is_movement_buffed(self, sq: Coord, color: Color) -> bool:
        return self._effect_caches["aura"].is_buffed(sq, color)

    def is_movement_debuffed(self, sq: Coord, color: Color) -> bool:
        return self._effect_caches["aura"].is_debuffed(sq, color)

    def is_frozen(self, sq: Coord, color: Color) -> bool:
        return self._effect_caches["aura"].is_frozen(sq, color)

    def black_hole_pull_map(self, controller: Color) -> Dict[Coord, Coord]:
        return self._effect_caches["aura"].pull_map(controller)

    def white_hole_push_map(self, controller: Color) -> Dict[Coord, Coord]:
        return self._effect_caches["aura"].push_map(controller)

    def mark_trail(self, trailblazer_sq: Coord, slid_squares: Set[Coord]) -> None:
        self._effect_caches["trailblaze"].mark_trail(trailblazer_sq, slid_squares)

    def current_trail_squares(self, controller: Color) -> Set[Coord]:
        return self._effect_caches["trailblaze"].current_trail_squares(controller)

    def is_geomancy_blocked(self, sq: Coord, current_ply: int) -> bool:
        return self._effect_caches["geomancy"].is_blocked(sq, current_ply)

    def block_square(self, sq: Coord, current_ply: int) -> bool:
        return self._effect_caches["geomancy"].block_square(sq, current_ply)

    def get_attacked_squares(self, color: Color) -> Set[Coord]:
        cached = self._effect_caches["attacks"].get_for_color(color)
        return cached if cached is not None else set()

    def store_attacked_squares(self, color: Color, attacked: Set[Coord]) -> None:
        self._effect_caches["attacks"].store_for_color(color, attacked)

    def clear_all_effects(self) -> None:
        for cache in self._effect_caches.values():
            if hasattr(cache, 'clear'):
                cache.clear()
            elif hasattr(cache, 'invalidate'):
                cache.invalidate()

    def __getitem__(self, key: str) -> Any:
        return self._effect_caches[key]

    def _apply_hole_effects(self, aura_type: str, controller: Color, board) -> None:
        """Refactored common logic for pulls and pushes."""
        aura = self._effect_caches["aura"]
        if aura_type == "pull":
            dirty_squares = aura.apply_pull_effects(controller, board)
        elif aura_type == "push":
            dirty_squares = aura.apply_push_effects(controller, board)
        else:
            raise ValueError("Invalid aura_type")

        if dirty_squares:
            self.cache_manager.occupancy.batch_set_positions(
                [(sq, board.get(sq)) for sq in dirty_squares]
            )

        aura.apply_move(None, controller, board)

        self.cache_manager.move.invalidate_attacked_squares(controller)
        self.cache_manager.move.invalidate_attacked_squares(controller.opposite())
        self.cache_manager.move._lazy_revalidate()

    def apply_blackhole_pulls(self, controller: Color, board) -> None:
        self._apply_hole_effects("pull", controller, board)

    def apply_whitehole_pushes(self, controller: Color, board) -> None:
        self._apply_hole_effects("push", controller, board)
