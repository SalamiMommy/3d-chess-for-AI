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
    MOVE_DTYPE, INDEX_DTYPE, BOOL_DTYPE, MOVE_FLAGS, MinimalStateProxy
)
from game3d.common.coord_utils import in_bounds_vectorized, ensure_coords, coords_to_keys
from game3d.common.validation import validate_coords_batch, validate_coord, validate_moves, validate_move

from game3d.movement.movepiece import Move
from game3d.movement.movementmodifiers import (
    apply_buff_effects_vectorized,
    apply_debuff_effects_vectorized,
    filter_valid_moves
)

from game3d.attacks.check import move_would_leave_king_in_check, king_in_check, batch_moves_leave_king_in_check

from numba import njit, prange

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

logger = logging.getLogger(__name__)

class MoveGenerationError(RuntimeError):
    pass


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
        # (Removed incorrect early return optimization that treated TT best_move as the ONLY legal move)
        # cache_key = state.zkey
        # tt_entry = state.cache_manager.transposition_table.probe_with_symmetry(cache_key, state.board)
        # if tt_entry and tt_entry.best_move: ...


        # 2. Check Legal Move Cache (Identity Fix)
        # 2. Check Legal Move Cache (Identity Fix)
        cached_moves = state.cache_manager.move_cache.get_legal_moves(state.color)
        if cached_moves is not None:
            self._stats['cache_hits'] += 1
            return cached_moves
            
        self._stats['cache_misses'] += 1

        # 3. Incremental Move Generation
        final_moves = self._generate_raw_pseudolegal(state)
        
        # 4. Filter Moves (CRITICAL STEP)
        chk_moves = self._apply_all_filters(state, final_moves, check_king_safety=True)
        
        # 5. Cache Legal Moves
        state.cache_manager.move_cache.store_legal_moves(state.color, chk_moves)
        
        return chk_moves

    def _generate_raw_pseudolegal(self, state: 'GameState') -> np.ndarray:
        """Helper to generate raw pseudolegal moves incrementally."""
        cache_manager = state.cache_manager
        move_cache = cache_manager.move_cache
        occ_cache = cache_manager.occupancy_cache
        
        # Get all pieces for current color
        all_coords = occ_cache.get_positions(state.color)
        
        if all_coords.size == 0:
            return np.empty((0, 6), dtype=COORD_DTYPE)

        all_keys = coords_to_keys(all_coords)

        # Retrieve valid cached moves and identify dirty pieces
        clean_moves_list, dirty_indices = move_cache.get_incremental_state(state.color, all_keys)
        
        new_moves_list = []
        
        # If we have dirty pieces, regenerate them
        if dirty_indices:
            from game3d.core.buffer import state_to_buffer
            from game3d.core.api import generate_pseudolegal_moves_subset
            
            # Get FULL state buffer (read only, zero copy)
            buffer = state_to_buffer(state, readonly=True)
            
            # Get all pieces to map indices
            all_occ_coords, all_occ_types, _, _, all_occ_colors = occ_cache.export_buffer_data()
            
            # Find indices where color == state.color
            color_mask = (all_occ_colors == state.color)
            global_indices = np.where(color_mask)[0]
            
            # ✅ FIX: Validate indices before access to prevent crash
            if isinstance(dirty_indices, list):
                dirty_indices = np.array(dirty_indices, dtype=np.int32)
            
            valid_dirty = dirty_indices[dirty_indices < len(global_indices)]
            if len(valid_dirty) < len(dirty_indices):
                pass
            
            if len(valid_dirty) == 0:
                 batch_new_moves = np.empty((0, 6), dtype=COORD_DTYPE)
            else:
                target_global_indices = global_indices[valid_dirty].astype(np.int32)
                
                # Generate
                batch_new_moves = generate_pseudolegal_moves_subset(buffer, target_global_indices)
            
            if batch_new_moves.size > 0:
                new_moves_list.append(batch_new_moves)
                # Update piece cache
                self._cache_piece_moves(cache_manager, batch_new_moves[:, :3], batch_new_moves, state.color)
        
        # Merge clean and new moves
        if not clean_moves_list and not new_moves_list:
            return np.empty((0, 6), dtype=COORD_DTYPE)
        else:
            return np.concatenate(clean_moves_list + new_moves_list)

    def refresh_pseudolegal_moves(self, state: 'GameState') -> None:
        """
        Refreshes only the pseudolegal cache (for opponent/attack generation).
        Skips expensive king safety checks.
        """
        # 1. Generate Raw Pseudolegal Moves (Incremental)
        raw_moves = self._generate_raw_pseudolegal(state)
        
        # 2. Filter basic constraints (Frozen, Hive, Wall) but SKIP King Safety
        # ✅ FIX: is_attack_generation=True allows frozen/king-capture moves (for check detection)
        filtered_moves = self._apply_all_filters(state, raw_moves, check_king_safety=False, is_attack_generation=True)
        
        # 3. Cache as Pseudolegal Moves (updates attack mask)
        # 3. Cache as Pseudolegal Moves (updates attack mask)
        state.cache_manager.move_cache.store_pseudolegal_moves(state.color, filtered_moves)
        
        # ✅ FIX: Clear affected pieces tracking since we just regenerated
        state.cache_manager.move_cache.clear_affected_pieces(state.color)

    # ✅ OPT 1.3: Lazy Pseudolegal Cache Refresh
    def refresh_pseudolegal_moves_incremental(self, state: 'GameState', dirty_piece_keys: np.ndarray) -> None:
        """
        Incrementally refresh pseudolegal moves ONLY for dirty pieces.
        
        ✅ OPTIMIZED: Instead of regenerating all pieces' moves, this only
        regenerates for explicitly dirty pieces (typically 1-5 vs 20+).
        
        Use case: After a move, only pieces affected by that move need refresh.
        
        Args:
            state: Current game state
            dirty_piece_keys: Packed coordinate keys of pieces that need refresh
        """
        if dirty_piece_keys.size == 0:
            return  # Nothing to refresh
        
        cache_manager = state.cache_manager
        move_cache = cache_manager.move_cache
        occ_cache = cache_manager.occupancy_cache
        
        # Convert keys back to coordinates
        dirty_coords = self._keys_to_coords(dirty_piece_keys)
        
        # Filter to only pieces that still exist (might have been captured)
        valid_mask = np.zeros(dirty_coords.shape[0], dtype=np.bool_)
        for i, coord in enumerate(dirty_coords):
            ptype, _ = occ_cache.get_fast(coord) if hasattr(occ_cache, 'get_fast') else (0, 0)
            valid_mask[i] = (ptype != 0)
        
        dirty_coords = dirty_coords[valid_mask]
        
        if dirty_coords.size == 0:
            return
        
        from game3d.core.buffer import state_to_buffer
        from game3d.core.api import generate_pseudolegal_moves_subset
        
        # Get state buffer
        buffer = state_to_buffer(state, readonly=True)
        
        # Get all occupied coords to find indices
        all_occ_coords, all_occ_types, _, _, all_occ_colors = occ_cache.export_buffer_data()
        
        # Create lookup for coordinates -> global index
        # We need to find which indices in the buffer correspond to dirty_coords
        dirty_new_moves = []
        
        for coord in dirty_coords:
            # Find global index of this coord
            cx, cy, cz = int(coord[0]), int(coord[1]), int(coord[2])
            match_mask = (all_occ_coords[:, 0] == cx) & (all_occ_coords[:, 1] == cy) & (all_occ_coords[:, 2] == cz)
            matched_indices = np.where(match_mask)[0]
            
            if matched_indices.size > 0:
                target_idx = matched_indices[0].astype(np.int32).reshape(1)
                piece_moves = generate_pseudolegal_moves_subset(buffer, target_idx)
                if piece_moves.size > 0:
                    dirty_new_moves.append(piece_moves)
                    # Update piece cache
                    coord_key = cx | (cy << 9) | (cz << 18)
                    move_cache.store_piece_moves(state.color, coord_key, piece_moves)
        
        # Merge new moves with existing clean moves
        existing_moves = move_cache.get_pseudolegal_moves(state.color)
        
        if existing_moves is None:
            # No existing cache - need full refresh
            self.refresh_pseudolegal_moves(state)
            return
        
        # Remove old moves for dirty pieces and add new ones
        if dirty_new_moves:
            # Filter out moves from dirty pieces in existing cache
            dirty_key_set = set(int(k) for k in dirty_piece_keys)
            
            from_coords = existing_moves[:, :3]
            from_keys = coords_to_keys(from_coords)
            keep_mask = ~np.isin(from_keys, np.array(list(dirty_key_set), dtype=np.int64))
            
            clean_moves = existing_moves[keep_mask]
            all_new_moves = np.concatenate(dirty_new_moves)
            
            # Filter new_moves: is_attack_generation=True ensures consistent attack mask
            filtered_new = self._apply_all_filters(state, all_new_moves, check_king_safety=False, is_attack_generation=True)
            
            # Combine and store
            combined = np.vstack([clean_moves, filtered_new]) if clean_moves.size > 0 and filtered_new.size > 0 else (clean_moves if clean_moves.size > 0 else filtered_new)
            move_cache.store_pseudolegal_moves(state.color, combined)



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
        move_keys = coords_to_keys(move_sources)

        # Group moves by sorting keys
        sort_idx = np.argsort(move_keys)
        sorted_moves = batch_moves[sort_idx]
        sorted_keys = move_keys[sort_idx]

        unique_keys, start_indices = np.unique(sorted_keys, return_index=True)
        end_indices = np.append(start_indices[1:], sorted_keys.size)

        # logger.info(f"Caching moves for {unique_keys.size} pieces (batch size {batch_moves.shape[0]})")

        for i in range(unique_keys.size):
            key = unique_keys[i]
            moves = sorted_moves[start_indices[i]:end_indices[i]]
            cache_manager.move_cache.store_piece_moves(color, key, moves)

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


    def _apply_all_filters(self, state: 'GameState', moves: np.ndarray, check_king_safety: bool = True, is_attack_generation: bool = False) -> np.ndarray:
        """
        Apply all move filters using single-pass combined masking.
        
        ✅ OPTIMIZED: Computes all filter conditions upfront and combines them
        into a single mask before array slicing. Reduces 7 potential array 
        allocations to 1.
        """
        if moves.size == 0:
            return moves
            
        n_moves = moves.shape[0]
        occ_cache = state.cache_manager.occupancy_cache
        
        # ✅ OPTIMIZATION: Early return if Priest protects King
        has_priest = occ_cache.has_priest(state.color) if check_king_safety else False

        # =========================================================================
        # PHASE 1: Compute all independent filter masks upfront (no array slicing)
        # =========================================================================
        
        # Start with all moves valid
        keep_mask = np.ones(n_moves, dtype=bool)
        
        # --- Filter 1: Frozen Pieces ---
        if not is_attack_generation:
            is_frozen = state.cache_manager.consolidated_aura_cache.batch_is_frozen(
                moves[:, :3], state.turn_number, state.color
            )
            keep_mask &= ~is_frozen
        
        # --- Pre-fetch attributes (used by multiple filters) ---
        # Get source attributes once
        _, source_types = occ_cache.batch_get_attributes_unsafe(moves[:, :3])
        # Get destination attributes once  
        dest_colors, dest_types = occ_cache.batch_get_attributes_unsafe(moves[:, 3:])
        
        # --- Filter 2: King Capture Prevention ---
        if not is_attack_generation:
            king_capture_mask = dest_types == PieceType.KING
            keep_mask &= ~king_capture_mask

        # --- Filter 3: Hive Movement History ---
        if hasattr(state, '_moved_hive_positions') and state._moved_hive_positions:
            hive_mask = source_types == PieceType.HIVE
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
                    keep_mask[hive_indices[is_already_moved]] = False

        # --- Filter 4: Wall Capture Restrictions ---
        wall_capture_mask = dest_types == PieceType.WALL
        if np.any(wall_capture_mask):
            capture_indices = np.where(wall_capture_mask)[0]
            
            # Use pre-fetched dest_colors for wall color check
            attacker_z = moves[capture_indices, 2]
            wall_z = moves[capture_indices, 5]
            
            is_white_wall = (dest_colors[capture_indices] == Color.WHITE)
            is_black_wall = (dest_colors[capture_indices] == Color.BLACK)
            
            invalid_capture = (is_white_wall & (attacker_z > wall_z)) | \
                              (is_black_wall & (attacker_z < wall_z))
            
            if np.any(invalid_capture):
                keep_mask[capture_indices[invalid_capture]] = False

        # --- Filter 5: Trailblazer Counter Avoidance (King-specific) ---
        king_mask = source_types == PieceType.KING
        if np.any(king_mask):
            king_indices = np.where(king_mask)[0]
            king_positions = moves[king_mask, :3]
            king_destinations = moves[king_mask, 3:]
            
            king_counters = state.cache_manager.trailblaze_cache.batch_get_counters(king_positions)
            danger_mask = king_counters >= 2
            
            if np.any(danger_mask):
                dangerous_dests = king_destinations[danger_mask]
                
                hits_trail = state.cache_manager.trailblaze_cache.batch_check_trail_intersection(
                    dangerous_dests, avoider_color=state.color
                )
                
                if np.any(hits_trail):
                    dangerous_king_indices = king_indices[danger_mask]
                    for local_idx, global_idx in enumerate(dangerous_king_indices):
                        if hits_trail[local_idx]:
                            keep_mask[global_idx] = False

        # --- Filter 6: Swapper-Wall Swap Prevention ---
        swapper_mask = source_types == PieceType.SWAPPER
        if np.any(swapper_mask):
            swapper_indices = np.where(swapper_mask)[0]
            # Use pre-fetched dest_types
            wall_dest_mask = dest_types[swapper_indices] == PieceType.WALL
            
            if np.any(wall_dest_mask):
                keep_mask[swapper_indices[wall_dest_mask]] = False

        # --- Filter 7: Wall Edge Bounds Prevention ---
        wall_source_mask = source_types == PieceType.WALL
        if np.any(wall_source_mask):
            wall_indices = np.where(wall_source_mask)[0]
            wall_dest_x = moves[wall_indices, 3]
            wall_dest_y = moves[wall_indices, 4]
            
            invalid_wall_dest_mask = (wall_dest_x >= SIZE - 1) | (wall_dest_y >= SIZE - 1)
            
            if np.any(invalid_wall_dest_mask):
                keep_mask[wall_indices[invalid_wall_dest_mask]] = False

        # =========================================================================
        # PHASE 2: Apply combined mask (SINGLE array allocation)
        # =========================================================================
        if not np.all(keep_mask):
            moves = moves[keep_mask]
            if moves.size == 0:
                return moves

        # =========================================================================
        # PHASE 3: King Safety Filter (expensive, done last, after reduction)
        # =========================================================================
        if check_king_safety and not has_priest and moves.size > 0:
            from game3d.attacks.check import batch_moves_leave_king_in_check_fused
            
            leaves_check = batch_moves_leave_king_in_check_fused(state, moves, state.cache_manager)
            
            if np.any(leaves_check):
                moves = moves[~leaves_check]

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
