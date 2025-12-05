# generator.py
"""
Legal Move Generator Orchestrator.

Coordinates the move generation pipeline:
1. Raw Generation (geometric, ignore occupancy)
2. Pseudolegal Generation (geometric + occupancy)
3. Validation (board boundaries, occupancy)
4. Filtering (Game rules: Check, Freeze, Hive, Priest logic)
5. Caching (Transposition Table & Incremental updates)
"""

from __future__ import annotations
import numpy as np
import logging
from typing import Optional, Union, Any, List, Dict, Tuple, TYPE_CHECKING
from collections import defaultdict

from game3d.common.shared_types import (
    COORD_DTYPE, PIECE_TYPE_DTYPE, SIZE, VOLUME,
    Color, PieceType, VECTORIZATION_THRESHOLD, DEFAULT_BATCH_SIZE, MAX_BATCH_SIZE,
    MOVE_DTYPE, INDEX_DTYPE, BOOL_DTYPE, MOVE_FLAGS
)
from game3d.common.coord_utils import in_bounds_vectorized, ensure_coords
from game3d.common.validation import validate_coords_batch, validate_coord, validate_moves, validate_move
from game3d.movement.slider_engine import SliderMovementEngine
from game3d.movement.jump_engine import JumpMovementEngine
from game3d.movement.movepiece import Move
from game3d.movement.movementmodifiers import (
    apply_buff_effects_vectorized,
    apply_debuff_effects_vectorized,
    filter_valid_moves
)
from game3d.movement.pseudolegal import (
    generate_pseudolegal_moves_batch,
    refresh_pseudolegal_cache,
    coord_to_key,
    MoveContractViolation
)
from game3d.attacks.check import move_would_leave_king_in_check, king_in_check
from game3d.attacks.pin import get_pinned_pieces, get_legal_pin_squares
from numba import njit, prange

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

logger = logging.getLogger(__name__)

class MoveGenerationError(RuntimeError):
    pass

@njit(cache=True)
def _check_on_segment(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> bool:
    """Check if point is on the line segment between start and end (inclusive)."""
    # Vectors relative to start
    v = end - start
    w = point - start
    
    # 1. Check collinearity using cross product (must be 0)
    # cx = v[1]*w[2] - v[2]*w[1]
    if (v[1]*w[2] - v[2]*w[1]) != 0: return False
    # cy = v[2]*w[0] - v[0]*w[2]
    if (v[2]*w[0] - v[0]*w[2]) != 0: return False
    # cz = v[0]*w[1] - v[1]*w[0]
    if (v[0]*w[1] - v[1]*w[0]) != 0: return False
    
    # 2. Check direction (dot product >= 0)
    dot = v[0]*w[0] + v[1]*w[1] + v[2]*w[2]
    if dot < 0: return False
    
    # 3. Check length (w must not be longer than v)
    # Since they are collinear and same direction, we can compare squared lengths via dot product
    # dot(v, w) = |v|*|w|
    # dot(v, v) = |v|*|v|
    # We want |w| <= |v|, so |v|*|w| <= |v|*|v|
    v_sq = v[0]*v[0] + v[1]*v[1] + v[2]*v[2]
    if dot > v_sq: return False
    
    # 4. Exclude King square (start)
    # The pin ray excludes the King itself (move to friendly piece square is invalid anyway, but good to be precise)
    if w[0] == 0 and w[1] == 0 and w[2] == 0: return False
    
    return True

@njit(cache=True, parallel=True)
def _apply_pin_filter_kernel(
    moves_sources: np.ndarray, # (N, 3)
    moves_dests: np.ndarray,   # (N, 3)
    mask: np.ndarray,          # (N,)
    pinned_keys: np.ndarray,   # (K,)
    attacker_keys: np.ndarray, # (K,)
    king_pos: np.ndarray       # (3,)
) -> None:
    """Vectorized application of pin constraints."""
    n_moves = moves_sources.shape[0]
    n_pinned = pinned_keys.shape[0]
    
    # Pre-unpack attacker keys to coordinates
    attacker_coords = np.empty((n_pinned, 3), dtype=moves_sources.dtype)
    for k in range(n_pinned):
        key = attacker_keys[k]
        attacker_coords[k, 0] = key & 0x1FF
        attacker_coords[k, 1] = (key >> 9) & 0x1FF
        attacker_coords[k, 2] = (key >> 18) & 0x1FF
        
    for i in prange(n_moves):
        if not mask[i]:
            continue
            
        # Recalculate key for source
        sx, sy, sz = moves_sources[i]
        s_key = sx + (sy << 9) + (sz << 18)
        
        # Check against pinned list
        pinned_idx = -1
        for k in range(n_pinned):
            if pinned_keys[k] == s_key:
                pinned_idx = k
                break
                
        if pinned_idx != -1:
            # Pinned! Check validity
            is_valid = _check_on_segment(king_pos, attacker_coords[pinned_idx], moves_dests[i])
            if not is_valid:
                mask[i] = False

# =============================================================================
# MOVE FILTERING HELPERS
# =============================================================================
def filter_safe_moves_optimized(game_state: 'GameState', moves: np.ndarray) -> np.ndarray:
    """
    Optimized move filtration.

    ✅ OPTIMIZED: Uses batch_moves_leave_king_in_check for king moves instead of
    per-move simulation. This is 5-10x faster for positions with many king moves.
    
    ✅ OPTIMIZED: Vectorized pinned piece filtering using pre-built allowed keys lookup.

    Algorithm:
    1. Check if King is currently in check.
    2. If NOT in check:
       - King moves use batch attack mask check
       - PINNED pieces are restricted to their pin rays (geometric check)
       - All other moves are automatically safe.
    3. If IN check:
       - Use batch check for all moves.
    """
    if moves.size == 0:
        return moves

    color = game_state.color
    occ_cache = game_state.cache_manager.occupancy_cache
    king_pos = occ_cache.find_king(color)

    # If no king (captured/error), all moves are theoretically unsafe or game is over
    if king_pos is None:
        return np.empty((0, 6), dtype=moves.dtype)

    # 1. Ensure opponent's moves are cached for efficient check detection
    opponent_color = Color(color).opposite().value
    if game_state.cache_manager.move_cache.get_pseudolegal_moves(opponent_color) is None:
        # ✅ OPTIMIZED: Use lightweight proxy instead of full GameState creation
        opp_coords = occ_cache.get_positions(opponent_color)
        if opp_coords.size > 0:
            class _MinimalStateProxy:
                """Lightweight proxy with only required attributes for move generation."""
                __slots__ = ('board', 'color', 'cache_manager')
                def __init__(self, board, color, cache_manager):
                    self.board = board
                    self.color = color
                    self.cache_manager = cache_manager
            
            opp_state = _MinimalStateProxy(game_state.board, opponent_color, game_state.cache_manager)
            debuffed = game_state.cache_manager.consolidated_aura_cache.get_debuffed_squares(opponent_color)
            
            # 1. Generate and cache RAW moves (Required for pin detection)
            opp_raw_moves = generate_pseudolegal_moves_batch(
                opp_state, opp_coords, debuffed, ignore_occupancy=True
            )
            game_state.cache_manager.move_cache.store_raw_moves(opponent_color, opp_raw_moves)

            # 2. Generate and store PSEUDOLEGAL moves
            opp_moves = generate_pseudolegal_moves_batch(
                opp_state, opp_coords, debuffed, ignore_occupancy=False
            )
            game_state.cache_manager.move_cache.store_pseudolegal_moves(opponent_color, opp_moves)
            
            global _generator
            if _generator is None:
                _generator = LegalMoveGenerator()
            _generator._cache_piece_moves(game_state.cache_manager, opp_coords, opp_moves, opponent_color)
        else:
             game_state.cache_manager.move_cache.store_pseudolegal_moves(opponent_color, np.empty((0, 6), dtype=MOVE_DTYPE))

    is_in_check = king_in_check(game_state.board, color, color, cache=game_state.cache_manager)

    # Pre-calculate keys and masks
    mask = np.ones(len(moves), dtype=bool)
    sources = moves[:, :3]
    dests = moves[:, 3:]

    # Identify King moves
    is_king_move = (sources[:, 0] == king_pos[0]) & \
                   (sources[:, 1] == king_pos[1]) & \
                   (sources[:, 2] == king_pos[2])

    if not is_in_check:
        # --- FAST PATH: NOT IN CHECK ---

        # A. ✅ OPTIMIZED: Batch filter King moves using attack mask
        if np.any(is_king_move):
            from game3d.attacks.check import batch_moves_leave_king_in_check
            king_moves = moves[is_king_move]
            unsafe_mask = batch_moves_leave_king_in_check(game_state, king_moves)
            
            # Apply results to global mask
            king_indices = np.where(is_king_move)[0]
            mask[king_indices] = ~unsafe_mask

        # B. ✅ OPTIMIZED: Vectorized Pinned Pieces Filtering
        pinned_info = get_pinned_pieces(game_state, color)

        if pinned_info:
            # Prepare arrays for vectorized kernel
            pinned_keys = np.array(list(pinned_info.keys()), dtype=np.int64)
            attacker_keys = np.array(list(pinned_info.values()), dtype=np.int64)
            
            # Run kernel
            _apply_pin_filter_kernel(
                sources,
                dests,
                mask,
                pinned_keys,
                attacker_keys,
                king_pos.astype(COORD_DTYPE)
            )

        # C. Unpinned, Non-King moves are automatically safe when not in check.

    else:
        # --- IN CHECK PATH: Use batch check ---
        from game3d.attacks.check import batch_moves_leave_king_in_check
        unsafe_mask = batch_moves_leave_king_in_check(game_state, moves)
        mask = ~unsafe_mask

    return moves[mask]
# =============================================================================
# OPTIMIZED MOVE GENERATOR
# =============================================================================

class LegalMoveGenerator:
    """
    Primary Move Generator.
    Handles generation, validation, filtering, and caching.
    """

    def __init__(self):
        self._batch_size = min(DEFAULT_BATCH_SIZE, MAX_BATCH_SIZE)
        self._stats = {'tt_hits': 0, 'tt_misses': 0, 'cache_hits': 0, 'cache_misses': 0,
                       'pieces_processed': 0, 'moves_generated': 0}

    def generate(self, state: "GameState", mode: Optional[str] = None) -> np.ndarray:
        """Entry point for move generation."""
        return self.generate_fused(state)

    def generate_fused(self, state: 'GameState') -> np.ndarray:
        """
        Fused generation pipeline:
        1. Probe TT.
        2. Probe Legal Move Cache.
        3. Refresh Pseudolegal Cache (Incremental).
        4. Filter (Check, Pin, Hive, etc.).
        5. Cache Legal Moves.
        """
        # 1. Check Transposition Table
        cache_key = state.zkey
        tt_entry = state.cache_manager.transposition_table.probe_with_symmetry(cache_key, state.board)

        if tt_entry and tt_entry.best_move:
            self._stats['tt_hits'] += 1
            from_c, to_c = tt_entry.best_move.from_coord, tt_entry.best_move.to_coord
            moves = np.array([[*from_c, *to_c]], dtype=COORD_DTYPE)
            state.cache_manager.move_cache.store_legal_moves(state.color, moves)
            return moves

        self._stats['tt_misses'] += 1

        # 2. Check Legal Move Cache
        cached = state.cache_manager.move_cache.get_legal_moves(state.color)
        affected_pieces = state.cache_manager.move_cache.get_affected_pieces(state.color)

        if cached is not None and affected_pieces.size == 0:
            self._stats['cache_hits'] += 1
            return cached

        self._stats['cache_misses'] += 1

        # 3. Refresh Pseudolegal Cache (Delegated to pseudolegal.py)
        refresh_pseudolegal_cache(state)
        
        # Retrieve cached pseudolegal moves
        pseudolegal_moves = state.cache_manager.move_cache.get_pseudolegal_moves(state.color)
        
        if pseudolegal_moves is None:
            # Should not happen after refresh, but safety fallback
            pseudolegal_moves = np.empty((0, 6), dtype=MOVE_DTYPE)

        # 4. Apply Game Rule Filters
        final_moves = self._apply_all_filters(state, pseudolegal_moves)

        # 5. Final Cache Store
        state.cache_manager.move_cache.store_legal_moves(state.color, final_moves)
        
        # Note: affected pieces are cleared in refresh_pseudolegal_cache, 
        # but store_legal_moves might need to know if it's fresh. 
        # Actually refresh_pseudolegal_cache clears affected pieces for pseudolegal layer.
        # Legal layer is now fresh too.

        return final_moves

    def _keys_to_coords(self, keys: np.ndarray) -> np.ndarray:
        """Convert cache keys back to coordinates."""
        if keys.size == 0:
            return np.empty((0, 3), dtype=COORD_DTYPE)
        
        x = keys & 0x1FF
        y = (keys >> 9) & 0x1FF
        z = (keys >> 18) & 0x1FF
        
        return np.column_stack((x, y, z)).astype(COORD_DTYPE)

    def _cache_piece_moves(self, cache_manager, batch_coords: np.ndarray, batch_moves: np.ndarray, color: int) -> None:
        """
        Groups batch moves by source coordinate and stores them in piece cache.
        Sorts moves by key for efficient grouping. Complexity: O(M log M).
        """
        if batch_moves.size == 0:
            return

        move_sources = batch_moves[:, :3]
        move_keys = coord_to_key(move_sources)

        # Group moves by sorting keys
        sort_idx = np.argsort(move_keys)
        sorted_moves = batch_moves[sort_idx]
        sorted_keys = move_keys[sort_idx]

        unique_keys, start_indices = np.unique(sorted_keys, return_index=True)
        end_indices = np.append(start_indices[1:], sorted_keys.size)

        for i in range(unique_keys.size):
            cache_manager.move_cache.store_piece_moves(
                color, unique_keys[i], sorted_moves[start_indices[i]:end_indices[i]]
            )

    def validate_move(self, state: "GameState", move: Union[Move, np.ndarray]) -> bool:
        """Validates a single move via validation.py."""
        if isinstance(move, np.ndarray) and move.shape == (6,):
            return validate_moves(state, move.reshape(1, 6))[0]
        return validate_move(state, move)

    def validate_moves(self, state: "GameState", moves: Union[List[Move], np.ndarray]) -> np.ndarray:
        """Validates multiple moves via validation.py."""
        if isinstance(moves, np.ndarray) and moves.ndim == 2:
            return validate_moves(state, moves)
        raise TypeError(f"Moves must be numpy array, got {type(moves)}.")

    def _validate_moves_array(self, state: "GameState", moves: np.ndarray) -> np.ndarray:
        """Internal helper for array validation."""
        return validate_moves(state, moves) if moves.size > 0 else np.zeros(0, dtype=bool)

    def clear_cache(self, cache_manager):
        """Invalidates the move cache."""
        cache_manager.move_cache.invalidate()

    def get_cache_stats(self, cache_manager) -> Dict[str, int]:
        """Merges internal stats with cache manager stats."""
        stats = cache_manager.move_cache.get_statistics()
        stats.update(self._stats)
        return stats


    def _apply_all_filters(self, state: 'GameState', moves: np.ndarray) -> np.ndarray:
        """Apply all move filters (check, pin, wall capture, etc.)."""
        if moves.size == 0:
            return moves

        occ_cache = state.cache_manager.occupancy_cache

        # 1. Filter: Frozen Pieces (Cheap - Vectorized Aura Check)
        is_frozen = state.cache_manager.consolidated_aura_cache.batch_is_frozen(
            moves[:, :3], state.turn_number, state.color
        )
        if np.any(is_frozen):
            moves = moves[~is_frozen]
            if moves.size == 0: return moves

        # 2. Filter: Hive Movement History (Cheap - Set Lookup)
        if hasattr(state, '_moved_hive_positions') and state._moved_hive_positions:
            _, piece_types = occ_cache.batch_get_attributes_unsafe(moves[:, :3])
            hive_mask = piece_types == PieceType.HIVE

            if np.any(hive_mask):
                hive_indices = np.where(hive_mask)[0]
                
                # ✅ OPTIMIZED: Vectorized check using keys
                moved_keys = np.array([
                    x | (y << 9) | (z << 18) 
                    for (x, y, z) in state._moved_hive_positions
                ], dtype=np.int64)
                
                move_coords = moves[hive_indices, :3]
                source_keys = (move_coords[:, 0] | 
                               (move_coords[:, 1] << 9) | 
                               (move_coords[:, 2] << 18)).astype(np.int64)
                
                is_already_moved = np.isin(source_keys, moved_keys)
                
                if np.any(is_already_moved):
                     keep_mask = np.ones(len(moves), dtype=bool)
                     keep_mask[hive_indices[is_already_moved]] = False
                     moves = moves[keep_mask]
                     if moves.size == 0: return moves

        # 3. Filter: Wall Capture Restrictions (Cheap - Type Check)
        # Identify moves that capture a WALL
        _, dest_types = occ_cache.batch_get_attributes_unsafe(moves[:, 3:])
        wall_capture_mask = dest_types == PieceType.WALL
        
        if np.any(wall_capture_mask):
            # Get indices of moves capturing walls
            capture_indices = np.where(wall_capture_mask)[0]
            
            # Get wall colors to determine "front" direction
            # Note: All parts of a wall share the same Z, so we can check dest Z directly
            dest_colors, _ = occ_cache.batch_get_attributes_unsafe(moves[capture_indices, 3:])
            
            attacker_z = moves[capture_indices, 2]
            wall_z = moves[capture_indices, 5]
            
            is_white_wall = (dest_colors == Color.WHITE)
            is_black_wall = (dest_colors == Color.BLACK)
            
            # Invalid if: (White Wall AND Attacker Z > Wall Z) OR (Black Wall AND Attacker Z < Wall Z)
            invalid_capture = (is_white_wall & (attacker_z > wall_z)) | \
                              (is_black_wall & (attacker_z < wall_z))
            
            if np.any(invalid_capture):
                # Remove invalid captures
                # We need to map back to original moves array
                indices_to_remove = capture_indices[invalid_capture]
                
                keep_mask = np.ones(len(moves), dtype=bool)
                keep_mask[indices_to_remove] = False
                
                moves = moves[keep_mask]
                if moves.size == 0: return moves

        # 4. Filter: Trailblazer Counter Avoidance (Cheap - Cache Lookup)
        _, piece_types = occ_cache.batch_get_attributes_unsafe(moves[:, :3])
        king_mask = piece_types == PieceType.KING
        
        if np.any(king_mask):
            king_indices = np.where(king_mask)[0]
            king_positions = moves[king_mask, :3]
            king_destinations = moves[king_mask, 3:]
            
            king_counters = state.cache_manager.trailblaze_cache.batch_get_counters(king_positions)
            danger_mask = king_counters >= 2
            
            if np.any(danger_mask):
                dangerous_dests = king_destinations[danger_mask]
                
                # Vectorized trail check
                hits_trail = state.cache_manager.trailblaze_cache.batch_check_trail_intersection(
                    dangerous_dests, avoider_color=state.color
                )
                
                if np.any(hits_trail):
                    keep_mask = np.ones(len(moves), dtype=bool)
                    dangerous_king_indices = king_indices[danger_mask]
                    
                    for local_idx, global_idx in enumerate(dangerous_king_indices):
                        if hits_trail[local_idx]:
                            keep_mask[global_idx] = False
                    
                    moves = moves[keep_mask]
                    if moves.size == 0: return moves

        # 5. ✅ OPTIMIZED: Check & Pin Filtering (Most Expensive - Run Last)
        # Only run if Priests are gone (otherwise King is immune)
        if not occ_cache.has_priest(state.color):
            # This replaces the old separate filter_safe_moves AND pinned logic
            moves = filter_safe_moves_optimized(state, moves)

        return moves

    def update_legal_moves_incremental(self, state: 'GameState') -> np.ndarray:
        """
        Incrementally update legal moves.
        Now delegates to generate_fused which handles incremental updates via refresh_pseudolegal_cache.
        """
        return self.generate_fused(state)


# =============================================================================
# GLOBAL GENERATOR INSTANCE & PUBLIC API
# =============================================================================

_generator = None

def initialize_generator():
    """Initializes the singleton generator instance."""
    global _generator
    _generator = LegalMoveGenerator()

def generate_legal_moves(state: "GameState") -> np.ndarray:
    """
    Generates all legal moves for the current state.
    Returns: (N, 6) numpy array [from_x, from_y, from_z, to_x, to_y, to_z].
    """
    global _generator
    if _generator is None:
        _generator = LegalMoveGenerator()
    
    return _generator.generate(state)

def generate_legal_moves_for_piece(game_state: 'GameState', coord: np.ndarray) -> np.ndarray:
    """
    Generates legal moves for a specific piece at `coord`.
    Verifies piece ownership before generation.
    """
    coord_arr = ensure_coords(coord)
    if coord_arr.shape == (1, 3): coord_arr = coord_arr.flatten()

    if coord_arr.size == 0 or coord_arr.ndim != 1:
        raise ValueError(f"Invalid coordinate: {coord_arr}")

    piece_info = game_state.cache_manager.occupancy_cache.get(coord_arr)
    if not piece_info or piece_info["color"] != game_state.color:
        raise ValueError(f"Invalid or opponent piece at {coord_arr}")

    all_moves = generate_legal_moves(game_state)
    if all_moves.size == 0: return np.empty((0, 6), dtype=COORD_DTYPE)

    piece_moves_mask = np.all(all_moves[:, :3] == coord_arr, axis=1)
    return all_moves[piece_moves_mask]

def validate_move_via_generator(state: "GameState", move: Union[Move, np.ndarray]) -> bool:
    """Public wrapper for single move validation."""
    return validate_move(state, move)

def validate_moves_via_generator(state: "GameState", moves: Union[List[Move], np.ndarray]) -> np.ndarray:
    """Public wrapper for batch move validation."""
    return validate_moves(state, moves)

__all__ = [
    'generate_legal_moves',
    'generate_legal_moves_for_piece',
    'validate_move_via_generator',
    'validate_moves_via_generator',
    'LegalMoveGenerator',
    'MoveGenerationError',
    'MoveContractViolation',
]
