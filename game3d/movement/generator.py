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
            dest_keys = None # Lazy compute

            for i in range(len(moves)):
                # Skip if already filtered (e.g. invalid king move) or is a king move
                if not mask[i] or is_king_move[i]:
                    continue

                s_key = int(source_keys[i])
                if s_key in pinned_info:
                    # Piece is pinned! Check strict movement rules
                    attacker_key = pinned_info[s_key]

                    # Resolve attacker coordinate
                    # (We reconstruct coord from key - bitwise ops matching coord_to_key)
                    ax = attacker_key & 0x1FF
                    ay = (attacker_key >> 9) & 0x1FF
                    az = (attacker_key >> 18) & 0x1FF
                    attacker_pos = np.array([ax, ay, az], dtype=COORD_DTYPE)

                    allowed_keys = get_legal_pin_squares(king_pos, attacker_pos)

                    if dest_keys is None:
                        dest_keys = coord_to_key(dests)

                    if int(dest_keys[i]) not in allowed_keys:
                        mask[i] = False

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
        1. Pseudolegal generation (batch)
        2. Validation (check/pin/etc)
        3. Filtering (safe moves)
        """
        # 0. Check for checkmate/stalemate early exit?
        # 1. Probe TT.
        # 2. Probe Move Cache (fast path).
        # 3. Identify cache misses or affected pieces.
        # 4. Regenerate raw moves (ignore occupancy) AND pseudolegal moves (respect occupancy).
        # 5. Validate and Filter (Freeze, Hive, Priest, Check).
        # 6. Update Cache.
        # """
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

        # 2. Check Move Cache & Incremental Logic
        cached = state.cache_manager.move_cache.get_legal_moves(state.color)
        affected_pieces = state.cache_manager.move_cache.get_affected_pieces(state.color)

        if cached is not None and affected_pieces.size == 0:
            self._stats['cache_hits'] += 1
            return cached

        self._stats['cache_misses'] += 1

        # 3. Rebuild Moves
        coords = state.cache_manager.occupancy_cache.get_positions(state.color)
        if coords.size == 0:
            return np.empty((0, 6), dtype=COORD_DTYPE)

        coord_keys = coord_to_key(coords)
        moves_list = []
        pieces_to_regenerate = []
        debuffed_coords = state.cache_manager.consolidated_aura_cache.get_debuffed_squares(state.color)

        # Classify pieces: reuse cache vs regenerate
        for i in range(len(coords)):
            key = coord_keys[i]
            is_affected = np.any(affected_pieces == key) if affected_pieces.size > 0 else False

            if not is_affected and state.cache_manager.move_cache.has_piece_moves(state.color, key):
                p_moves = state.cache_manager.move_cache.get_piece_moves(state.color, key)
                if p_moves.size > 0:
                    moves_list.append(p_moves)
            else:
                pieces_to_regenerate.append(coords[i])

        # Regenerate raw moves for invalid/affected pieces
        if pieces_to_regenerate:
            regenerate_coords = np.array(pieces_to_regenerate, dtype=COORD_DTYPE)
            
            # 1. Generate Raw Moves (Ignore Occupancy) - for Pin Detection
            raw_moves = generate_pseudolegal_moves_batch(state, regenerate_coords, debuffed_coords, ignore_occupancy=True)
            state.cache_manager.move_cache.store_raw_moves(state.color, raw_moves)

            # 2. Generate Pseudolegal Moves (Respect Occupancy) - for Legal Move Generation
            new_moves = generate_pseudolegal_moves_batch(state, regenerate_coords, debuffed_coords, ignore_occupancy=False)
            
            self._cache_piece_moves(state.cache_manager, regenerate_coords, new_moves, state.color)
            if new_moves.size > 0:
                moves_list.append(new_moves)

        final_moves = np.concatenate(moves_list, axis=0) if moves_list else np.empty((0, 6), dtype=MOVE_DTYPE)

        # ✅ Store PSEUDOLEGAL moves for check detection
        state.cache_manager.move_cache.store_pseudolegal_moves(state.color, final_moves)
        
        if final_moves.size == 0 and len(coords) > 0:
             logger.debug(f"Generator: 0 moves after pseudolegal generation for {len(coords)} pieces")
        # else:
        #      logger.debug(f"Generator: {len(final_moves)} pseudolegal moves generated")

        # 4. Validation
        if final_moves.size > 0:
            valid_mask = self._validate_moves_array(state, final_moves)
            final_moves = final_moves[valid_mask]
            if final_moves.size == 0:
                 logger.debug("Generator: All moves filtered by validation")
            # else:
            #      logger.debug(f"Generator: {len(final_moves)} moves passed validation")

        # 5. Apply Game Rule Filters
        final_moves = self._apply_all_filters(state, final_moves)
        # logger.debug(f"Generator: {len(final_moves)} moves passed all filters (Final Legal Moves)")

        # 6. Final Cache Store
        state.cache_manager.move_cache.store_legal_moves(state.color, final_moves)
        state.cache_manager.move_cache.clear_affected_pieces(state.color)

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
                    hits_trail = np.array([
                        state.cache_manager.trailblaze_cache.check_trail_intersection(
                            dest.reshape(1, 3), avoider_color=state.color
                        )
                        for dest in dangerous_dests
                    ], dtype=bool)
                    
                    keep_mask = np.ones(len(moves), dtype=bool)
                    dangerous_king_indices = king_indices[danger_mask]
                    
                    for local_idx, global_idx in enumerate(dangerous_king_indices):
                        if hits_trail[local_idx]:
                            keep_mask[global_idx] = False
                    
                    moves = moves[keep_mask]

        return moves

    def update_legal_moves_incremental(self, state: 'GameState') -> np.ndarray:
        """
        Incrementally update legal moves - only regenerate affected pieces.
        """
        color = state.color

        # Get affected pieces that need regeneration
        affected_pieces = state.cache_manager.move_cache.get_affected_pieces(color)

        if affected_pieces.size == 0:
            # Nothing affected, return cached moves
            return state.cache_manager.move_cache.get_legal_moves(color)

        # Get debuffed squares for modifier application
        debuffed_coords = state.cache_manager.consolidated_aura_cache.get_debuffed_squares(color)

        # Get all piece coordinates
        all_coords = state.cache_manager.occupancy_cache.get_positions(color)
        if all_coords.size == 0:
            state.cache_manager.move_cache.store_legal_moves(color, np.empty((0, 6), dtype=MOVE_DTYPE))
            return np.empty((0, 6), dtype=MOVE_DTYPE)

        # Identify pieces that need regeneration:
        # 1. Affected pieces (explicitly invalidated)
        # 2. Missing pieces (not in cache yet)
        
        pieces_to_regenerate = []
        coord_keys = coord_to_key(all_coords)
        
        for i in range(len(all_coords)):
            key = coord_keys[i]
            is_affected = np.any(affected_pieces == key) if affected_pieces.size > 0 else False
            is_missing = not state.cache_manager.move_cache.has_piece_moves(color, key)
            
            if is_affected or is_missing:
                pieces_to_regenerate.append(all_coords[i])

        # Regenerate moves for all identified pieces
        if pieces_to_regenerate:
            regenerate_coords = np.array(pieces_to_regenerate, dtype=COORD_DTYPE)
            
            # 1. Generate Raw Moves (Ignore Occupancy)
            raw_moves = generate_pseudolegal_moves_batch(state, regenerate_coords, debuffed_coords, ignore_occupancy=True)
            state.cache_manager.move_cache.store_raw_moves(color, raw_moves)
            
            # 2. Generate Pseudolegal Moves (Respect Occupancy)
            new_moves = generate_pseudolegal_moves_batch(state, regenerate_coords, debuffed_coords, ignore_occupancy=False)
            
            self._cache_piece_moves(state.cache_manager, regenerate_coords, new_moves, color)

        # Reconstruct full move list from piece cache
        all_moves = self._reconstruct_from_piece_cache(state, color)
        
        # Store full pseudolegal moves
        state.cache_manager.move_cache.store_pseudolegal_moves(color, all_moves)

        # Apply validation and filtering
        all_moves = self._apply_all_filters(state, all_moves)

        # Store in legal moves cache
        state.cache_manager.move_cache.store_legal_moves(color, all_moves)
        state.cache_manager.move_cache.clear_affected_pieces(color)

        return all_moves

    def _reconstruct_from_piece_cache(self, state: 'GameState', color: int) -> np.ndarray:
        """Rebuild full move list from piece-level cache."""
        all_coords = state.cache_manager.occupancy_cache.get_positions(color)
        if all_coords.size == 0:
            return np.empty((0, 6), dtype=COORD_DTYPE)

        moves_list = []
        coord_keys = coord_to_key(all_coords)

        for key in coord_keys:
            piece_moves = state.cache_manager.move_cache.get_piece_moves(color, key)
            if piece_moves.size > 0:
                moves_list.append(piece_moves)

        return np.concatenate(moves_list, axis=0) if moves_list else np.empty((0, 6), dtype=COORD_DTYPE)
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
