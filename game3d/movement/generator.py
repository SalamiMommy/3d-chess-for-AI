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
if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

logger = logging.getLogger(__name__)

class MoveGenerationError(RuntimeError):
    pass

# =============================================================================
# MOVE FILTERING HELPERS
# =============================================================================
def filter_safe_moves_optimized(game_state: 'GameState', moves: np.ndarray) -> np.ndarray:
    """
    Optimized move filtration.

    Instead of simulating every move to check for king safety, we:
    1. Check if King is currently in check.
    2. If NOT in check:
       - Only KING moves need full simulation safety checks.
       - PINNED pieces are restricted to their pin rays (geometric check).
       - All other moves are automatically safe.
    3. If IN check:
       - All moves must be verified to resolve the check (simulation).
    """
    if moves.size == 0:
        return moves

    color = game_state.color
    occ_cache = game_state.cache_manager.occupancy_cache
    king_pos = occ_cache.find_king(color)

    # If no king (captured/error), all moves are theoretically unsafe or game is over
    if king_pos is None:
        return np.empty((0, 6), dtype=moves.dtype)

    # 1. Determine Global Check State
    # ✅ OPTIMIZATION: Ensure opponent's moves are cached for efficient check detection
    # If we don't do this, check detection falls back to the slow path (generating moves on the fly)
    opponent_color = Color(color).opposite().value
    if game_state.cache_manager.move_cache.get_pseudolegal_moves(opponent_color) is None:
        # Generate and cache opponent's pseudolegal moves
        opp_coords = occ_cache.get_positions(opponent_color)
        if opp_coords.size > 0:
            # Create temporary state for opponent
            from game3d.game.gamestate import GameState
            opp_state = GameState(game_state.board, opponent_color, game_state.cache_manager)
            
            # Get debuffed squares for opponent
            debuffed = game_state.cache_manager.consolidated_aura_cache.get_debuffed_squares(opponent_color)
            
            # Generate all pseudolegal moves
            opp_moves = generate_pseudolegal_moves_batch(
                opp_state, opp_coords, debuffed, ignore_occupancy=False
            )
            
            # Store in cache
            game_state.cache_manager.move_cache.store_pseudolegal_moves(opponent_color, opp_moves)
            
            # Also cache piece moves for incremental updates
            # We access the generator instance to use its helper
            global _generator
            if _generator is None:
                _generator = LegalMoveGenerator()
            _generator._cache_piece_moves(game_state.cache_manager, opp_coords, opp_moves, opponent_color)
        else:
             # No opponent pieces, store empty moves
             game_state.cache_manager.move_cache.store_pseudolegal_moves(opponent_color, np.empty((0, 6), dtype=MOVE_DTYPE))

    is_in_check = king_in_check(game_state.board, color, color, cache=game_state.cache_manager)

    # Pre-calculate keys and masks
    mask = np.ones(len(moves), dtype=bool)
    sources = moves[:, :3]
    dests = moves[:, 3:]

    # Identify King moves (King moves ALWAYS require safety checks)
    is_king_move = (sources[:, 0] == king_pos[0]) & \
                   (sources[:, 1] == king_pos[1]) & \
                   (sources[:, 2] == king_pos[2])

    if not is_in_check:
        # --- FAST PATH: NOT IN CHECK ---

        # A. Filter King Moves (Must simulate to ensure not stepping into check)
        # We only simulate moves where is_king_move is True
        if np.any(is_king_move):
            king_indices = np.where(is_king_move)[0]
            for i in king_indices:
                if move_would_leave_king_in_check(game_state, moves[i]):
                    mask[i] = False

        # B. Filter Pinned Pieces (Geometric constraint, no simulation)
        # Identify pinned pieces once
        pinned_info = get_pinned_pieces(game_state, color)

        if pinned_info:
            source_keys = coord_to_key(sources)
            
            # Get keys of all pinned pieces
            pinned_keys = np.array(list(pinned_info.keys()), dtype=np.int32)
            
            # Identify moves made by pinned pieces
            # Use isin for vectorized check
            is_pinned_move = np.isin(source_keys, pinned_keys)
            
            if np.any(is_pinned_move):
                dest_keys = coord_to_key(dests)
                
                # Iterate only over unique pinned pieces involved in moves (usually 1 or 2)
                unique_pinned_keys = np.unique(source_keys[is_pinned_move])
                
                for s_key in unique_pinned_keys:
                    s_key_int = int(s_key)
                    if s_key_int not in pinned_info:
                        continue
                        
                    attacker_key = pinned_info[s_key_int]
                    
                    # Resolve attacker coordinate
                    ax = attacker_key & 0x1FF
                    ay = (attacker_key >> 9) & 0x1FF
                    az = (attacker_key >> 18) & 0x1FF
                    attacker_pos = np.array([ax, ay, az], dtype=COORD_DTYPE)

                    allowed_keys = get_legal_pin_squares(king_pos, attacker_pos)
                    allowed_keys_arr = np.array(list(allowed_keys), dtype=np.int32)
                    
                    # Apply filter to all moves by this pinned piece
                    # Mask: (This piece) AND (Not King - redundant but safe) AND (Already valid)
                    piece_mask = (source_keys == s_key) & mask & (~is_king_move)
                    
                    if np.any(piece_mask):
                        # Check if destinations are allowed
                        valid_dest = np.isin(dest_keys[piece_mask], allowed_keys_arr)
                        
                        # Update global mask
                        # We need to map back to original indices
                        piece_indices = np.where(piece_mask)[0]
                        mask[piece_indices] = valid_dest

        # C. Unpinned, Non-King moves are automatically safe when not in check.

    else:
        # --- SLOW PATH: IN CHECK ---
        # When in check, ANY move must resolve the check.
        # The number of legal moves in check is usually small, so simulation is acceptable here.
        for i in range(len(moves)):
            if move_would_leave_king_in_check(game_state, moves[i]):
                mask[i] = False

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

        # Filter: Frozen Pieces
        is_frozen = state.cache_manager.consolidated_aura_cache.batch_is_frozen(
            moves[:, :3], state.turn_number, state.color
        )
        if np.any(is_frozen):
            moves = moves[~is_frozen]

        # Filter: Hive Movement (One hive per turn)
        if moves.size > 0 and hasattr(state, '_moved_hive_positions') and len(state._moved_hive_positions) > 0:
            _, piece_types = occ_cache.batch_get_attributes_unsafe(moves[:, :3])
            hive_mask = piece_types == PieceType.HIVE

            if np.any(hive_mask):
                can_move = ~hive_mask
                for i in np.where(hive_mask)[0]:
                    if tuple(moves[i, :3]) not in state._moved_hive_positions:
                        can_move[i] = True
                moves = moves[can_move]

        # ✅ OPTIMIZED: Combined Check & Pin Filtering
        # Only run if Priests are gone (otherwise King is immune)
        if moves.size > 0:
            if not occ_cache.has_priest(state.color):
                # This replaces the old separate filter_safe_moves AND pinned logic
                moves = filter_safe_moves_optimized(state, moves)

        # Filter: Wall Capture Restrictions (Can only capture from behind/side)
        if moves.size > 0:
            # Identify moves that capture a WALL
            # We need to check the piece type at the destination square
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

        # Filter: Trailblazer Counter Avoidance (King with 2 counters cannot hit trail)
        if moves.size > 0:
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
                    
                    keep_mask = np.ones(len(moves), dtype=bool)
                    dangerous_king_indices = king_indices[danger_mask]
                    
                    for local_idx, global_idx in enumerate(dangerous_king_indices):
                        if hits_trail[local_idx]:
                            keep_mask[global_idx] = False
                    
                    moves = moves[keep_mask]

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
