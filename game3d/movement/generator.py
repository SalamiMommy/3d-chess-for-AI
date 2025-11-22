# generator.py
"""Fully optimized legal move generator - PRIMARY MOVE GENERATION ORCHESTRATOR.

This is the ONLY module that generates moves. After generation, it applies modifiers
and delegates ALL validation to validation.py.

ARCHITECTURAL CONTRACT:
- Dispatchers MUST return numpy arrays of shape (N,6) with dtype=COORD_DTYPE
- No Move objects allowed in hot paths
- All operations must be vectorized or Numba-compiled
- Validation is strictly delegated to validation.py
"""

from __future__ import annotations
import numpy as np
import logging
from typing import Optional, Union, Any, List, Dict, Tuple, TYPE_CHECKING
from collections import defaultdict
from numba import njit, prange

from game3d.common.shared_types import (
    COORD_DTYPE, PIECE_TYPE_DTYPE, SIZE, VOLUME,
    Color, PieceType, VECTORIZATION_THRESHOLD, DEFAULT_BATCH_SIZE, MAX_BATCH_SIZE,
    MOVE_DTYPE, INDEX_DTYPE, BOOL_DTYPE, MOVE_FLAGS
)
from game3d.common.coord_utils import in_bounds_vectorized, ensure_coords
from game3d.common.validation import validate_coords_batch, validate_coord, validate_moves, validate_move
from game3d.common.move_utils import create_move_array_from_objects_vectorized

# Import movement engines (NO validation in these - just raw generation)
from game3d.movement.slider_engine import SliderMovementEngine
from game3d.movement.jump_engine import JumpMovementEngine
from game3d.movement.movepiece import Move

# Import modifiers to apply effects post-generation
from game3d.movement.movementmodifiers import (
    apply_buff_effects_vectorized,
    apply_debuff_effects_vectorized,
    filter_valid_moves
)

# Import registry to access piece dispatchers
from game3d.common.registry import get_piece_dispatcher

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

from game3d.pieces.pieces.pawn import generate_pawn_moves

logger = logging.getLogger(__name__)

class MoveGenerationError(RuntimeError):
    """Raised when move generation fails unrecoverably."""
    pass

class MoveContractViolation(TypeError):
    """Raised when dispatcher returns non-native arrays."""
    pass

# =============================================================================
# CONTRACT ENFORCEMENT - CRITICAL FOR NUMPY-NATIVE OPERATION
# =============================================================================

def _enforce_numpy_native(move_array: np.ndarray, context: str = "") -> np.ndarray:
    """
    ENFORCE CONTRACT: Move arrays must be numpy-native, no Python objects.

    Args:
        move_array: Array to validate
        context: Context string for error messages

    Returns:
        Validated array

    Raises:
        MoveContractViolation: If array contains objects or has wrong shape
    """
    if move_array is None or move_array.size == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # CRITICAL: Reject object arrays
    if move_array.dtype == np.object_:
        raise MoveContractViolation(
            f"🚨 CONTRACT VIOLATION {context}: Move array contains Python objects. "
            f"Dispatchers must return native numpy arrays, not Move objects."
        )

    # Must be 2D with 6 columns
    if move_array.ndim != 2 or move_array.shape[1] != 6:
        raise MoveContractViolation(
            f"🚨 CONTRACT VIOLATION {context}: Invalid shape {move_array.shape}. "
            f"Expected (N, 6) array."
        )

    # Must be integer dtype
    if not np.issubdtype(move_array.dtype, np.integer):
        move_array = move_array.astype(COORD_DTYPE, copy=False)

    return move_array

@njit(cache=True, fastmath=True, parallel=True)
def _extract_piece_moves_from_batch(
    batch_moves: np.ndarray,
    piece_coord: np.ndarray
) -> np.ndarray:
    """
    VECTORIZED: Extract moves for a specific piece from batch.

    Args:
        batch_moves: (N, 6) array of all moves
        piece_coord: (3,) array of piece coordinate

    Returns:
        (M, 6) array of moves for this piece
    """
    # Create mask for this piece's moves
    mask = (batch_moves[:, 0] == piece_coord[0]) & \
           (batch_moves[:, 1] == piece_coord[1]) & \
           (batch_moves[:, 2] == piece_coord[2])

    # Count matches
    n_matches = np.sum(mask)

    if n_matches == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Extract matching moves
    result = np.empty((n_matches, 6), dtype=COORD_DTYPE)
    write_idx = 0

    for i in range(batch_moves.shape[0]):
        if mask[i]:
            result[write_idx] = batch_moves[i]
            write_idx += 1

    return result

# @njit(cache=True, parallel=True)
# def _generate_piece_moves_batch(
#     state_ptr: int,  # Numba-compatible pointer
#     coords: np.ndarray,
#     piece_types: np.ndarray
# ) -> np.ndarray:
#     """Generate all moves for batch of pieces in parallel."""
#     n_pieces = len(coords)
#     # Pre-allocate max possible moves
#     max_moves_per_piece = 64
#     total_buffer = np.zeros((n_pieces * max_moves_per_piece, 6), dtype=COORD_DTYPE)
#     move_counts = np.zeros(n_pieces, dtype=INDEX_DTYPE)
# 
#     for i in prange(n_pieces):
#         # Call piece-specific dispatcher (inlined for performance)
#         moves = _dispatch_piece_move(state_ptr, piece_types[i], coords[i])
#         n_moves = moves.shape[0]
# 
#         # Copy to buffer
#         start_idx = i * max_moves_per_piece
#         total_buffer[start_idx:start_idx + n_moves] = moves
#         move_counts[i] = n_moves
# 
#     # Compact result
#     total_moves = move_counts.sum()
#     result = np.empty((total_moves, 6), dtype=COORD_DTYPE)
#     write_idx = 0
# 
#     for i in range(n_pieces):
#         start_idx = i * max_moves_per_piece
#         n_moves = move_counts[i]
#         result[write_idx:write_idx + n_moves] = total_buffer[start_idx:start_idx + n_moves]
#         write_idx += n_moves
# 
#     return result
# =============================================================================
# OPTIMIZED MOVE GENERATOR
# =============================================================================

class LegalMoveGenerator:
    """
    PRIMARY MOVE GENERATOR - Single source of truth for legal moves.

    Responsibilities:
    1. Call piece-specific dispatchers to generate candidate moves
    2. Apply modifiers from movementmodifiers.py
    3. Return validated and modified legal moves via validation.py
    4. Cache generated moves with incremental updates

    CONTRACT: All dispatchers MUST return numpy arrays, never Move objects.
    """

    def __init__(self, cache_manager):
        """Initialize with cache manager."""
        self._cache_manager = cache_manager
        self._batch_size = min(DEFAULT_BATCH_SIZE, MAX_BATCH_SIZE)

        # Initialize movement engines for piece files to use
        self._slider_engine = SliderMovementEngine(cache_manager)
        self._jump_engine = JumpMovementEngine(cache_manager)

        # Statistics
        self._stats = {
            'tt_hits': 0,
            'tt_misses': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'pieces_processed': 0,
            'moves_generated': 0,
        }

    def generate(self, state: "GameState", mode: Optional[str] = None) -> np.ndarray:
        """
        PRIMARY ENTRY POINT for move generation - delegates to fused implementation.

        Orchestrates the workflow: generate → apply modifiers → validate.
        Returns fully validated and modified legal moves.
        Includes caching and incremental update support.

        ARCHITECTURAL NOTE: This is the ONLY function that should generate moves.
        All move requests must go through this function.
        """
        return self.generate_fused(state)

    def generate_fused(self, state: "GameState") -> np.ndarray:
        """Generate moves with transposition table and incremental update support."""

        # Check transposition table first (fast path)
        cache_key = state.zkey
        tt_entry = self._cache_manager.transposition_table.probe_with_symmetry(cache_key, state.board)

        if tt_entry and tt_entry.best_move:
            # Return cached moves from transposition table
            self._stats['tt_hits'] += 1
            logger.debug(f"TT hit: {tt_entry.best_move}")

            # Convert CompactMove to coordinate array
            from_coord = tt_entry.best_move.from_coord
            to_coord = tt_entry.best_move.to_coord
            moves = np.array([[
                from_coord[0], from_coord[1], from_coord[2],
                to_coord[0], to_coord[1], to_coord[2]
            ]], dtype=COORD_DTYPE)

            self._cache_manager.move_cache.store_moves(state.color, moves)
            return moves

        self._stats['tt_misses'] += 1

        # Check move cache
        cached = self._cache_manager.move_cache.get_cached_moves(state.color)
        affected_pieces = self._cache_manager.move_cache.get_affected_pieces(state.color)

        # Fast path: full cache hit
        if cached is not None and affected_pieces.size == 0:
            self._stats['cache_hits'] += 1
            return cached

        # INCREMENTAL UPDATE LOGIC
        # If we have a cache miss (generation mismatch) or affected pieces, we rebuild
        # but we try to reuse piece moves that are NOT affected.
        
        self._stats['cache_misses'] += 1

        # Get all pieces for this color
        coords = self._cache_manager.occupancy_cache.get_positions(state.color)
        if coords.size == 0:
            return np.empty(0, dtype=MOVE_DTYPE)

        # Convert coords to keys for fast lookup
        # Ensure 2D shape for _coord_to_key
        coord_keys = _coord_to_key(coords)
        
        # Identify which pieces need regeneration
        # 1. Pieces explicitly marked as affected
        # 2. Pieces not found in cache
        
        # Convert affected_pieces to set for O(1) lookup if small, or use isin
        # affected_pieces is array of int keys
        
        moves_list = []
        pieces_to_regenerate = []
        
        # Get debuffed squares from AuraCache
        # Note: get_debuffed_squares returns squares where pieces are debuffed.
        # We need to check if our pieces are on these squares.
        debuffed_coords = self._cache_manager.consolidated_aura_cache.get_debuffed_squares(state.color)
        
        # Iterate and classify
        # We can optimize this with vectorization if needed, but loop is fine for < 50 pieces
        for i in range(len(coords)):
            key = coord_keys[i]
            
            # Check if affected
            is_affected = False
            if affected_pieces.size > 0:
                # Simple linear scan is fast for small arrays, or use isin before loop
                # For now, let's assume affected_pieces is small
                is_affected = np.any(affected_pieces == key)
            
            if not is_affected and self._cache_manager.move_cache.has_piece_moves(state.color, key):
                # Reuse cached moves
                p_moves = self._cache_manager.move_cache.get_piece_moves(state.color, key)
                if p_moves.size > 0:
                    moves_list.append(p_moves)
            else:
                # Needs regeneration
                pieces_to_regenerate.append(coords[i])

        # Regenerate moves for invalid pieces
        if pieces_to_regenerate:
            regenerate_coords = np.array(pieces_to_regenerate, dtype=COORD_DTYPE)
            new_moves = self._generate_batch_moves_from_dispatchers(state, regenerate_coords, debuffed_coords)
            
            # Cache the newly generated piece moves
            if new_moves.size > 0:
                moves_list.append(new_moves)
                self._cache_piece_moves(regenerate_coords, new_moves, state.color)
            else:
                # Even if empty, we should cache the empty result for the piece
                # to avoid regenerating it next time? 
                # _cache_piece_moves handles extraction. If piece has no moves, it might not store anything.
                # We should explicitly store empty moves for pieces with no moves?
                # _cache_piece_moves implementation iterates batch_coords.
                self._cache_piece_moves(regenerate_coords, new_moves, state.color)

        # Combine all moves
        if moves_list:
            final_moves = np.concatenate(moves_list, axis=0)
        else:
            final_moves = np.empty((0, 6), dtype=MOVE_DTYPE)

        # Vectorized validation and filtering (global validation still needed for safety?)
        # Ideally piece moves are already valid. But "affected" logic might miss some interactions?
        # If we trust "affected" logic, we don't need to re-validate reused moves.
        # But let's be safe: validate ONLY the new moves?
        # Or validate everything? Validation is expensive.
        # The contract is: if piece is not affected, its moves are valid.
        
        # However, we must filter valid moves if we want to be 100% sure.
        # Let's assume reused moves are valid.
        # We only validate new moves? 
        # _generate_batch_moves_from_dispatchers returns raw moves.
        # We need to validate them.
        
        # Wait! _generate_batch_moves_from_dispatchers does NOT validate.
        # So we MUST validate the new moves.
        # What about reused moves? If they are from previous turn, are they valid?
        # If they were not affected, they should be.
        
        # Let's validate ALL moves for now to be safe, but this might be slow.
        # Optimization: Validate only new moves, assume reused are valid.
        
        # Actually, `generate_fused` previously did:
        # valid_mask = self._validate_moves_array(state, all_moves)
        # final_moves = all_moves[valid_mask]
        
        # If we reuse moves, they were valid in PREVIOUS state.
        # Are they valid in CURRENT state?
        # If dependencies are correct, yes.
        
        # Let's validate everything for correctness first.
        if final_moves.size > 0:
            valid_mask = self._validate_moves_array(state, final_moves)
            final_moves = final_moves[valid_mask]

        # Store in cache (updates generation)
        self._cache_manager.move_cache.store_moves(state.color, final_moves)
        
        # Clear affected pieces since we handled them
        self._cache_manager.move_cache.clear_affected_pieces(state.color)

        return final_moves

    def _cache_piece_moves(self, batch_coords: np.ndarray, batch_moves: np.ndarray, color: int) -> None:
        """
        Cache moves for each piece using optimized sorting and grouping.
        Complexity: O(M log M) where M is number of moves (vs O(N*M) previously).
        """
        if batch_moves.size == 0:
            return

        # 1. Compute keys for all moves' source coordinates
        # batch_moves is (M, 6), source is columns 0-3
        # We need to reshape to (M, 3) for _coord_to_key
        move_sources = batch_moves[:, :3]
        move_keys = _coord_to_key(move_sources)

        # 2. Sort moves by key
        # This groups all moves for the same piece together
        sort_idx = np.argsort(move_keys)
        sorted_moves = batch_moves[sort_idx]
        sorted_keys = move_keys[sort_idx]

        # 3. Find unique keys and their split indices
        unique_keys, start_indices = np.unique(sorted_keys, return_index=True)
        
        # 4. Iterate and cache
        # We can also compute end_indices
        end_indices = np.append(start_indices[1:], sorted_keys.size)
        
        for i in range(unique_keys.size):
            key = unique_keys[i]
            start = start_indices[i]
            end = end_indices[i]
            
            piece_moves = sorted_moves[start:end]
            
            # Store in cache
            self._cache_manager.move_cache.store_piece_moves(
                color, key, piece_moves
            )

    def _store_in_tt(self, cache_key: int, board, moves: np.ndarray) -> None:
        """Store best move in transposition table."""
        from game3d.cache.caches.transposition import CompactMove

        # Use first move as best move (placeholder - real engine would score)
        best_move = CompactMove(
            moves[0, :3], moves[0, 3:6],
            piece_type=PieceType.PAWN  # Placeholder
        )

        self._cache_manager.transposition_table.store_with_symmetry(
            cache_key, board, depth=1, score=0,
            node_type=0, best_move=best_move
        )

    def _generate_batch_moves_from_dispatchers(self, state: "GameState", batch_coords: np.ndarray, debuffed_coords: np.ndarray = None) -> np.ndarray:
        """Generate moves by calling piece dispatchers - ENFORCED NUMPY-ONLY."""
        moves_list = []

        # Convert debuffed coords to keys for fast lookup
        debuffed_keys = set()
        if debuffed_coords is not None and debuffed_coords.size > 0:
            keys = _coord_to_key(debuffed_coords)
            debuffed_keys = set(keys)

        for i, coord in enumerate(batch_coords):
            piece_info = state.cache_manager.occupancy_cache.get(coord.reshape(1, 3))
            if piece_info is None:
                logger.warning(f"⚠️ No piece at {coord}")
                continue

            piece_type = piece_info["piece_type"]
            
            # Check if piece is frozen
            # Note: We use the piece's color (state.color) as the victim color
            is_frozen = state.cache_manager.consolidated_aura_cache.batch_is_frozen(
                coord.reshape(1, 3), state.turn_number, state.color
            )[0]
            
            if is_frozen:
                # Frozen pieces cannot move
                continue

            # Check if piece is debuffed
            coord_key = _coord_to_key(coord.reshape(1, -1))[0]
            is_debuffed = coord_key in debuffed_keys
            
            if is_debuffed:
                # Use Pawn movement for debuffed pieces
                # Note: We use the piece's actual color, but Pawn movement logic
                dispatcher = lambda s, p: generate_pawn_moves(s.cache_manager, s.color, p)
            else:
                dispatcher = get_piece_dispatcher(piece_type)

            if not dispatcher:
                raise RuntimeError(f"No dispatcher registered for piece type {piece_type}")

            # CRITICAL: DISPATCHER MUST RETURN NUMPY ARRAY OR FAIL FAST
            raw_moves = dispatcher(state, coord)

            # ENFORCE CONTRACT: Crash immediately if dispatcher violates contract
            if not isinstance(raw_moves, np.ndarray):
                raise MoveContractViolation(
                    f"Dispatcher for piece type {piece_type} returned {type(raw_moves)}. "
                    f"Must return numpy array of shape (N, 6) with integer dtype."
                )

            # Ensure correct dtype without copying if already correct
            if raw_moves.dtype != COORD_DTYPE:
                raw_moves = raw_moves.astype(COORD_DTYPE, copy=False)

            # Ensure correct shape
            if raw_moves.ndim != 2 or raw_moves.shape[1] != 6:
                raise MoveContractViolation(
                    f"Dispatcher returned shape {raw_moves.shape}. Expected (N, 6)."
                )

            moves_list.append(raw_moves)

        return np.concatenate(moves_list, axis=0) if moves_list else np.empty((0, 6), dtype=COORD_DTYPE)

    def validate_move(self, state: "GameState", move: Union[Move, np.ndarray]) -> bool:
        """PUBLIC API: Validate a single move - delegates to validation.py."""
        # AVOID CONVERSION: Work directly with numpy arrays
        if isinstance(move, np.ndarray):
            if move.shape == (6,):  # Move array [from_x, from_y, from_z, to_x, to_y, to_z]
                return validate_moves(state, move.reshape(1, 6))[0]
            else:
                raise ValueError(f"Invalid move array shape: {move.shape}")

        # Only create Move object if absolutely necessary
        return validate_move(state, move)

    def validate_moves(self, state: "GameState", moves: Union[List[Move], np.ndarray]) -> np.ndarray:
        """PUBLIC API: Validate multiple moves - delegates to validation.py."""
        # ENFORCE: Only numpy arrays allowed
        if isinstance(moves, np.ndarray):
            if moves.ndim == 2 and moves.shape[1] == 6:
                return validate_moves(state, moves)
            else:
                raise ValueError(f"Invalid moves array shape: {moves.shape}")
        
        raise TypeError(f"Moves must be numpy array, got {type(moves)}. Legacy Move objects not supported.")


    def _validate_moves_array(self, state: "GameState", moves: np.ndarray) -> np.ndarray:
        """Direct array validation via validation.py."""
        if moves.size == 0:
            return np.zeros(0, dtype=bool)

        # Use the already-imported validate_moves function
        # This returns a boolean mask of valid moves
        return validate_moves(state, moves)

    def clear_cache(self):
        """Clear the move cache - call when board state changes."""
        self._cache_manager.move_cache.invalidate()
        logger.debug("Move cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics for debugging."""
        stats = self._cache_manager.move_cache.get_statistics()
        stats.update(self._stats)
        return stats
# =============================================================================
# COORDINATE KEY UTILITIES - NUMBA COMPILED
# =============================================================================

@njit(cache=True, fastmath=True)
def _coord_to_key(coords: np.ndarray) -> np.ndarray:
    """
    Convert (N,3) coordinates to integer keys using bit packing.
    PACKING: 9 bits for x + 9 bits for y + 9 bits for z = 27 bits total
    """
    n = coords.shape[0]
    keys = np.empty(n, dtype=np.int32)

    for i in prange(n):
        # Pack coordinates: x in bits 0-8, y in bits 9-17, z in bits 18-26
        keys[i] = (coords[i, 0]) | (coords[i, 1] << 9) | (coords[i, 2] << 18)

    return keys


# =============================================================================
# GLOBAL GENERATOR INSTANCE
# =============================================================================

_generator = None

def initialize_generator(cache_manager):
    """Initialize global generator with cache manager."""
    global _generator
    _generator = LegalMoveGenerator(cache_manager)

def generate_legal_moves(state: "GameState") -> np.ndarray:
    """
    PUBLIC API: Generate all legal moves.

    CONTRACT: Returns numpy array of shape (N,6) with dtype=COORD_DTYPE.
    """
    global _generator
    if _generator is None:
        initialize_generator(state.cache_manager)

    result = _generator.generate(state)

    # FINAL CONTRACT ENFORCEMENT
    result = _enforce_numpy_native(result, context="final_output")

    return result

def generate_legal_moves_for_piece(game_state: 'GameState', coord: np.ndarray) -> np.ndarray:
    """
    Get legal moves for piece at coordinate.

    Raises:
        ValueError: On invalid input
        MoveGenerationError: On generation failure
        MoveContractViolation: If dispatcher violates contract

    Returns:
        (N,6) numpy array of moves
    """
    coord_arr = ensure_coords(coord)
    if coord_arr.size == 0:
        raise ValueError("Empty coordinate provided")

    # ✅ FIX: Handle both (3,) and (1, 3) shapes
    if coord_arr.shape == (1, 3):
        coord_arr = coord_arr.flatten()

    if coord_arr.ndim != 1 or coord_arr.shape[0] != 3:
        raise ValueError(f"Invalid coordinate shape: {coord_arr.shape}")

    # Validate piece exists and belongs to current player
    piece_info = game_state.cache_manager.occupancy_cache.get(coord_arr)
    if piece_info is None:
        raise ValueError(f"No piece at {coord_arr}")

    if piece_info["color"] != game_state.color:
        raise ValueError(f"Piece at {coord_arr} belongs to opponent")

    # Generate all moves and filter for this piece
    all_moves = generate_legal_moves(game_state)
    if all_moves.size == 0:
        return np.empty(0, dtype=MOVE_DTYPE)

    # Vectorized filter for this piece's moves
    from_coord = coord_arr[0]
    piece_moves_mask = np.all(all_moves[:, :3] == from_coord, axis=1)
    return all_moves[piece_moves_mask]

def validate_move_via_generator(state: "GameState", move: Union[Move, np.ndarray]) -> bool:
    """
    PUBLIC API: Validate a single move - delegates to validation.py.
    """
    return validate_move(state, move)

def validate_moves_via_generator(state: "GameState", moves: Union[List[Move], np.ndarray]) -> np.ndarray:
    """
    PUBLIC API: Validate multiple moves - delegates to validation.py.
    """
    return validate_moves(state, moves)

# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'generate_legal_moves',
    'generate_legal_moves_for_piece',
    'validate_move_via_generator',
    'validate_moves_via_generator',
    'LegalMoveGenerator',
    'MoveGenerationError',
    'MoveContractViolation',
]
