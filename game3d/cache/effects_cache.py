# game3d/cache/effects_cache.py

"""Central management for all effect caches."""

from typing import Dict, Set, Any, Optional
from game3d.pieces.enums import Color
from game3d.pieces.piece import Piece
from game3d.cache.effectscache.freezecache import FreezeCache
from game3d.cache.effectscache.blackholesuckcache import BlackHoleSuckCache
from game3d.cache.effectscache.movementdebuffcache import MovementDebuffCache
from game3d.cache.effectscache.movementbuffcache import MovementBuffCache
from game3d.cache.effectscache.whiteholepushcache import WhiteHolePushCache
from game3d.cache.effectscache.trailblazecache import TrailblazeCache
from game3d.cache.effectscache.capturefrombehindcache import BehindCache
from game3d.cache.effectscache.geomancycache import GeomancyCache
from game3d.cache.effectscache.archerycache import ArcheryCache
from game3d.cache.effectscache.sharesquarecache import ShareSquareCache
from game3d.cache.effectscache.armourcache import ArmourCache
from game3d.cache.caches.attackscache import AttacksCache

class EffectsCache:
    """Manages all effect-specific caches."""

    def __init__(self, board, cache_manager):
        self.board = board
        self.cache_manager = cache_manager
        self._effect_caches: Dict[str, Any] = {}
        self._init_effects()

    def _init_effects(self) -> None:
        self._effect_caches = {
            "freeze": FreezeCache(self.cache_manager),
            "movement_buff": MovementBuffCache(self.cache_manager),
            "movement_debuff": MovementDebuffCache(self.cache_manager),
            "black_hole_suck": BlackHoleSuckCache(self.cache_manager),
            "white_hole_push": WhiteHolePushCache(self.cache_manager),
            "trailblaze": TrailblazeCache(self.cache_manager),  # FIXED: Pass self.cache_manager instead of self
            "behind": BehindCache(self.cache_manager),
            "armour": ArmourCache(self.cache_manager),
            "geomancy": GeomancyCache(self.cache_manager),
            "archery": ArcheryCache(self.cache_manager),
            "share_square": ShareSquareCache(self.cache_manager),
            "attacks": AttacksCache(self.board),
        }

    def get_affected_caches(self, mv: 'Move', mover: Color, from_piece: Piece,
                            to_piece: Optional[Piece], captured_piece: Optional[Piece]) -> Set[str]:
        affected = set()
        from game3d.pieces.enums import PieceType
        if from_piece is not None and from_piece.ptype in {PieceType.FREEZER, PieceType.SPEEDER, PieceType.SLOWER,
                                PieceType.BLACKHOLE, PieceType.WHITEHOLE, PieceType.TRAILBLAZER,
                                PieceType.WALL, PieceType.ARMOUR, PieceType.GEOMANCER,
                                PieceType.ARCHER, PieceType.KNIGHT}:
            effect_map = {
                PieceType.FREEZER: "freeze",
                PieceType.SPEEDER: "movement_buff",
                PieceType.SLOWER: "movement_debuff",
                PieceType.BLACKHOLE: "black_hole_suck",
                PieceType.WHITEHOLE: "white_hole_push",
                PieceType.TRAILBLAZER: "trailblaze",
                PieceType.WALL: "behind",
                PieceType.ARMOUR: "armour",
                PieceType.GEOMANCER: "geomancy",
                PieceType.ARCHER: "archery",
                PieceType.KNIGHT: "share_square"
            }
            affected.add(effect_map[from_piece.ptype])

        if captured_piece:
            self._add_affected_effects_from_pos(mv.to_coord, affected)
        self._add_affected_effects_from_pos(mv.from_coord, affected)

        return affected

    def _add_affected_effects_from_pos(self, pos: tuple[int, int, int], affected: Set[str]) -> None:
        from game3d.pieces.enums import PieceType
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    check_pos = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
                    if all(0 <= c < 9 for c in check_pos):
                        # Use cache manager instead of board
                        piece = self.cache_manager.piece_cache.get(check_pos)
                        if piece and piece.ptype in {PieceType.FREEZER, PieceType.BLACKHOLE,
                                                    PieceType.WHITEHOLE, PieceType.GEOMANCER}:
                            effect_map = {
                                PieceType.FREEZER: "freeze",
                                PieceType.BLACKHOLE: "black_hole_suck",
                                PieceType.WHITEHOLE: "white_hole_push",
                                PieceType.GEOMANCER: "geomancy"
                            }
                            affected.add(effect_map[piece.ptype])

    def get_affected_caches_for_undo(self, mv: 'Move', mover: Color) -> Set[str]:
        affected = self.get_affected_caches(mv, mover, None, None, None)
        self._add_affected_effects_from_pos(mv.to_coord, affected)
        return affected

    def update_effect_caches(
            self,
            mv: "Move",
            mover: Color,
            affected_caches: set[str],
            current_ply: int,
    ) -> None:
        """
        Incremental update: tell the move-cache *precisely* which squares
        need to be regenerated instead of rebuilding everything.
        """
        move_cache = self.cache_manager.move          # type: OptimizedMoveCache

        # ----------------------------------------------------------
        # 1.  Collect dirty squares from every affected aura
        # ----------------------------------------------------------
        dirty: set[tuple[int, int, int]] = set()

        for name in affected_caches:
            cache = self._effect_caches[name]
            # Every aura cache now exposes a tiny helper:
            #   dirty_squares(mv, mover) -> set[coord]
            if hasattr(cache, "dirty_squares"):
                dirty.update(cache.dirty_squares(mv, mover))
            else:
                # Fallback for caches that have not been refactored yet
                # (treat as “global” until they get their own helper)
                move_cache._needs_rebuild = True
                return          # give up and let the old rebuild run

        # ----------------------------------------------------------
        # 2.  Invalidate only those squares (and attacked bitmaps)
        # ----------------------------------------------------------
        for sq in dirty:
            move_cache.invalidate_square(sq)

        move_cache.invalidate_attacked_squares(mover)
        move_cache.invalidate_attacked_squares(mover.opposite())

        # ----------------------------------------------------------
        # 3.  Now let the aura caches update their *own* state
        # ----------------------------------------------------------
        for name in affected_caches:
            cache = self._effect_caches[name]
            try:
                if name == "geomancy":
                    cache.apply_move(mv, mover, current_ply, self.board)
                elif name in (
                    "archery", "black_hole_suck", "armour", "freeze",
                    "movement_buff", "movement_debuff", "share_square",
                    "trailblaze", "white_hole_push", "attacks",
                ):
                    cache.apply_move(mv, mover, self.board)
                else:
                    cache.apply_move(mv, mover)          # legacy signature
            except Exception as exc:
                # Never crash the game because an aura failed
                print(f"[EffectsCache] '{name}' update failed: {exc}")

    def update_effect_caches_for_undo(self, mv: 'Move', mover: Color,
                                      affected_caches: Set[str], current_ply: int) -> None:
        for name in affected_caches:
            try:
                cache = self._effect_caches[name]
                if hasattr(cache, 'undo_move'):
                    if name == "geomancy":
                        cache.undo_move(mv, mover, current_ply, self.board)
                    elif hasattr(cache, '_board') or name == "attacks":
                        cache.undo_move(mv, mover, self.board)
                    else:
                        cache.undo_move(mv, mover)
            except Exception as e:
                print(f"Effect cache {name} undo failed: {str(e)}")

    # Updated methods to use cache_manager
    def is_frozen(self, sq: tuple[int, int, int], victim: Color) -> bool:
        return self._effect_caches["freeze"].is_frozen(sq, victim)

    def is_movement_buffed(self, sq: tuple[int, int, int], friendly: Color) -> bool:
        return self._effect_caches["movement_buff"].is_buffed(sq, friendly)

    def is_movement_debuffed(self, sq: tuple[int, int, int], victim: Color) -> bool:
        return self._effect_caches["movement_debuff"].is_debuffed(sq, victim)

    def black_hole_pull_map(self, controller: Color) -> Dict[tuple[int, int, int], tuple[int, int, int]]:
        return self._effect_caches["black_hole_suck"].pull_map(controller)

    def white_hole_push_map(self, controller: Color) -> Dict[tuple[int, int, int], tuple[int, int, int]]:
        return self._effect_caches["white_hole_push"].push_map(controller)

    def mark_trail(self, trailblazer_sq: tuple[int, int, int], slid_squares: Set[tuple[int, int, int]]) -> None:
        self._effect_caches["trailblaze"].mark_trail(trailblazer_sq, slid_squares)

    def current_trail_squares(self, controller: Color) -> Set[tuple[int, int, int]]:
        return self._effect_caches["trailblaze"].current_trail_squares(controller)

    def is_geomancy_blocked(self, sq: tuple[int, int, int], current_ply: int) -> bool:
        return self._effect_caches["geomancy"].is_blocked(sq, current_ply)

    def block_square(self, sq: tuple[int, int, int], current_ply: int) -> bool:
        return self._effect_caches["geomancy"].block_square(sq, current_ply)

    def archery_targets(self, controller: Color) -> list[tuple[int, int, int]]:
        return self._effect_caches["archery"].attack_targets(controller)

    def is_valid_archery_attack(self, sq: tuple[int, int, int], controller: Color) -> bool:
        return self._effect_caches["archery"].is_valid_attack(sq, controller)

    def can_capture_wall(self, attacker_sq: tuple[int, int, int], wall_sq: tuple[int, int, int], controller: Color) -> bool:
        return self._effect_caches["armour"].can_capture(attacker_sq, wall_sq, controller)

    def pieces_at(self, sq: tuple[int, int, int]) -> list['Piece']:
        return self._effect_caches["share_square"].pieces_at(sq)

    def top_piece(self, sq: tuple[int, int, int]) -> Optional['Piece']:
        return self._effect_caches["share_square"].top_piece(sq)

    def get_attacked_squares(self, color: Color) -> Set[tuple[int, int, int]]:
        cached = self._effect_caches["attacks"].get_for_color(color)
        return cached if cached is not None else set()

    def store_attacked_squares(self, color: Color, attacked: Set[tuple[int, int, int]]) -> None:
        self._effect_caches["attacks"].store_for_color(color, attacked)

    def clear_all_effects(self) -> None:
        for cache in self._effect_caches.values():
            if hasattr(cache, 'clear'):
                cache.clear()
            elif hasattr(cache, 'invalidate'):
                cache.invalidate()

    def __getitem__(self, key: str) -> Any:
        return self._effect_caches[key]
