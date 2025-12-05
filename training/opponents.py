import numpy as np
from numba import njit, prange
import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

from game3d.common.shared_types import Color, PieceType, COLOR_EMPTY, COORD_DTYPE, INDEX_DTYPE, FLOAT_DTYPE, HASH_DTYPE, N_PIECE_TYPES, BOOL_DTYPE
import game3d.cache.caches.zobrist as zobrist_module

# Initialize Zobrist keys if not already done and get the table
zobrist_module._init_zobrist()
ZOBRIST_TABLE = zobrist_module._PIECE_KEYS
from game3d.movement.movepiece import Move
import game3d.pieces.pieces as all_pieces

# Initialize Attack Tables
MAX_VECTORS = 512 # Support complex pieces like FaceCone
ATTACK_VECTORS = np.zeros((2, N_PIECE_TYPES + 1, MAX_VECTORS, 3), dtype=COORD_DTYPE)
VECTOR_COUNTS = np.zeros((2, N_PIECE_TYPES + 1), dtype=INDEX_DTYPE)
IS_SLIDER = np.zeros((N_PIECE_TYPES + 1,), dtype=BOOL_DTYPE)

def _init_attack_tables():
    # Known slider types
    SLIDER_TYPES = {
        PieceType.BISHOP, PieceType.ROOK, PieceType.QUEEN,
        PieceType.TRIGONALBISHOP, PieceType.EDGEROOK,
        PieceType.XYQUEEN, PieceType.XZQUEEN, PieceType.YZQUEEN,
        PieceType.VECTORSLIDER, PieceType.CONESLIDER,
        PieceType.XZZIGZAG, PieceType.YZZIGZAG, PieceType.REFLECTOR,
        PieceType.TRAILBLAZER, PieceType.SPIRAL
    }

    # Helper to set vectors
    def set_vecs(pt_val, vecs, slider=False):
        if vecs is None or len(vecs) == 0:
            return
        n = min(len(vecs), MAX_VECTORS)
        # Set for both colors
        for c in range(2):
            ATTACK_VECTORS[c, pt_val, :n] = vecs[:n]
            VECTOR_COUNTS[c, pt_val] = n
        IS_SLIDER[pt_val] = slider

    # Iterate over all piece types
    for pt in PieceType:
        pt_val = pt.value
        if pt_val > N_PIECE_TYPES: continue
        
        # Determine vector name
        # Convention: [NAME]_MOVEMENT_VECTORS
        # Special cases handled below
        name = pt.name
        vec_name = f"{name}_MOVEMENT_VECTORS"
        
        # Special cases
        if pt == PieceType.PAWN:
            # Pawns have color-specific attack directions
            if hasattr(all_pieces, "PAWN_ATTACK_DIRECTIONS"):
                pawn_attacks = getattr(all_pieces, "PAWN_ATTACK_DIRECTIONS")
                # White (Color=1 -> index 0) - First 4
                ATTACK_VECTORS[0, pt_val, :4] = pawn_attacks[:4]
                VECTOR_COUNTS[0, pt_val] = 4
                # Black (Color=2 -> index 1) - Last 4
                ATTACK_VECTORS[1, pt_val, :4] = pawn_attacks[4:]
                VECTOR_COUNTS[1, pt_val] = 4
            continue
            
        elif pt == PieceType.PRIEST:
            vec_name = "KING_MOVEMENT_VECTORS" # Priest shares King vectors
        
        # Try to find vectors in all_pieces
        if hasattr(all_pieces, vec_name):
            vecs = getattr(all_pieces, vec_name)
            is_slider = pt in SLIDER_TYPES
            set_vecs(pt_val, vecs, slider=is_slider)
        else:
            # Fallback: check for [NAME]_ATTACK_VECTORS
            vec_name_alt = f"{name}_ATTACK_VECTORS"
            if hasattr(all_pieces, vec_name_alt):
                vecs = getattr(all_pieces, vec_name_alt)
                is_slider = pt in SLIDER_TYPES
                set_vecs(pt_val, vecs, slider=is_slider)

_init_attack_tables()

logger = logging.getLogger(__name__)

# =============================================================================
# NUMBA-COMPATIBLE DISTANCE FUNCTION (FIX)
# =============================================================================

@njit(cache=True, fastmath=True, inline='always')
def _manhattan_distance(coord1: np.ndarray, coord2: np.ndarray) -> int:
    """Manhattan distance for 3D coordinates - Numba compatible."""
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1]) + abs(coord1[2] - coord2[2])

# =============================================================================
# VECTORIZED UTILITY FUNCTIONS (Compiled)
# =============================================================================

# REMOVED: Old broken _compute_repetition_penalties_vectorized
# The old function compared position_hashes[j] == move[0] which compared
# a hash value to a move coordinate - completely wrong!
# Repetition penalty is now applied via batch_reward using game state data


@njit(cache=True, fastmath=True, parallel=True)
def _compute_capture_rewards_vectorized(
    to_coords: np.ndarray,
    captured_colors: np.ndarray,
    captured_types: np.ndarray,
    player_color: int,
    priest_bonus: float,
    freezer_bonus: float
) -> np.ndarray:
    """Vectorized capture reward calculation."""
    n_moves = len(to_coords)
    rewards = np.zeros(n_moves, dtype=FLOAT_DTYPE)

    enemy_color = 2 if player_color == 1 else 1

    for i in prange(n_moves):
        if captured_colors[i] == enemy_color:
            rewards[i] = 0.1  # Base capture reward

            if captured_types[i] == PieceType.PRIEST.value:
                rewards[i] += priest_bonus
            elif captured_types[i] == PieceType.FREEZER.value:
                rewards[i] += freezer_bonus

    return rewards


@njit(cache=True, fastmath=True)
def _compute_distance_rewards_serial(
    from_coords: np.ndarray,
    to_coords: np.ndarray,
    target_positions: np.ndarray,
    piece_types: np.ndarray,
    reward_per_step: float
) -> np.ndarray:
    """Serial distance-based rewards (optimized for small batches or few targets)."""
    n_moves = len(from_coords)
    rewards = np.zeros(n_moves, dtype=FLOAT_DTYPE)

    if target_positions.shape[0] == 0:
        return rewards

    for i in range(n_moves):
        # Only compute for relevant piece types
        if piece_types[i] == 0:  # Skip empty
            continue

        min_old_dist = 1e9
        min_new_dist = 1e9

        for j in range(target_positions.shape[0]):
            # Use the Numba-compatible distance function
            old_dist = _manhattan_distance(from_coords[i], target_positions[j])
            new_dist = _manhattan_distance(to_coords[i], target_positions[j])

            if old_dist < min_old_dist:
                min_old_dist = old_dist
            if new_dist < min_new_dist:
                min_new_dist = new_dist

        if min_new_dist < min_old_dist:
            rewards[i] = reward_per_step

    return rewards


@njit(cache=True, fastmath=True, parallel=True)
def _compute_distance_rewards_vectorized(
    from_coords: np.ndarray,
    to_coords: np.ndarray,
    target_positions: np.ndarray,
    piece_types: np.ndarray,
    reward_per_step: float
) -> np.ndarray:
    """Vectorized distance-based rewards (optimized for large batches with many targets)."""
    n_moves = len(from_coords)
    rewards = np.zeros(n_moves, dtype=FLOAT_DTYPE)

    if target_positions.shape[0] == 0:
        return rewards

    for i in prange(n_moves):
        # Only compute for relevant piece types
        if piece_types[i] == 0:  # Skip empty
            continue

        min_old_dist = 1e9
        min_new_dist = 1e9

        for j in range(target_positions.shape[0]):
            # Use the Numba-compatible distance function
            old_dist = _manhattan_distance(from_coords[i], target_positions[j])
            new_dist = _manhattan_distance(to_coords[i], target_positions[j])

            if old_dist < min_old_dist:
                min_old_dist = old_dist
            if new_dist < min_new_dist:
                min_new_dist = new_dist

        if min_new_dist < min_old_dist:
            rewards[i] = reward_per_step

    return rewards


def _compute_distance_rewards_adaptive(
    from_coords: np.ndarray,
    to_coords: np.ndarray,
    target_positions: np.ndarray,
    piece_types: np.ndarray,
    reward_per_step: float
) -> np.ndarray:
    """Adaptive dispatch for distance calculation based on work size.
    
    Uses serial version for small work sets, parallel for large work sets.
    Threshold based on total work: n_moves * n_targets.
    """
    n_moves = len(from_coords)
    n_targets = target_positions.shape[0]
    
    # Adaptive dispatch: use serial for small work, parallel for large
    total_work = n_moves * n_targets
    if total_work < 100:
        return _compute_distance_rewards_serial(
            from_coords, to_coords, target_positions, piece_types, reward_per_step
        )
    else:
        return _compute_distance_rewards_vectorized(
            from_coords, to_coords, target_positions, piece_types, reward_per_step
        )


@njit(cache=True, fastmath=True, parallel=True)
def _compute_center_control_rewards_vectorized(
    to_coords: np.ndarray
) -> np.ndarray:
    """Vectorized center control rewards."""
    n_moves = len(to_coords)
    rewards = np.zeros(n_moves, dtype=FLOAT_DTYPE)

    for i in prange(n_moves):
        x, y, z = to_coords[i]
        if (3 <= x <= 4 and 3 <= y <= 4 and 3 <= z <= 4):
            rewards[i] = 0.3

    return rewards


@njit(cache=True, fastmath=True, parallel=True)
def _compute_attack_defense_rewards_vectorized(
    from_coords: np.ndarray,
    to_coords: np.ndarray,
    our_attack_coords: np.ndarray,
    enemy_attack_coords: np.ndarray
) -> np.ndarray:
    """Vectorized attack/defense rewards."""
    n_moves = len(from_coords)
    rewards = np.zeros(n_moves, dtype=FLOAT_DTYPE)

    for i in prange(n_moves):
        from_attacked = False
        to_attacked = False

        # Check if from_coord is under attack
        for j in range(len(enemy_attack_coords)):
            if (enemy_attack_coords[j, 0] == from_coords[i, 0] and
                enemy_attack_coords[j, 1] == from_coords[i, 1] and
                enemy_attack_coords[j, 2] == from_coords[i, 2]):
                from_attacked = True
                break

        # Check if to_coord is under attack
        for j in range(len(enemy_attack_coords)):
            if (enemy_attack_coords[j, 0] == to_coords[i, 0] and
                enemy_attack_coords[j, 1] == to_coords[i, 1] and
                enemy_attack_coords[j, 2] == to_coords[i, 2]):
                to_attacked = True
                break

        if from_attacked and not to_attacked:
            rewards[i] = -0.1

    return rewards


@njit(cache=True, fastmath=True, parallel=True)
def _compute_check_potential_vectorized(
    to_coords: np.ndarray,
    piece_types: np.ndarray,
    enemy_king_pos: np.ndarray,
    check_reward: float,
    occupancy_grid: np.ndarray,
    attack_vectors: np.ndarray,
    vector_counts: np.ndarray,
    is_slider: np.ndarray,
    attacker_color_idx: int
) -> np.ndarray:
    """Vectorized check potential reward using piece definitions and blocking checks."""
    n_moves = len(to_coords)
    rewards = np.zeros(n_moves, dtype=FLOAT_DTYPE)
    
    if enemy_king_pos is None:
        return rewards
        
    kx, ky, kz = enemy_king_pos[0], enemy_king_pos[1], enemy_king_pos[2]

    for i in prange(n_moves):
        # Skip if piece is empty
        if piece_types[i] == 0:
            continue
            
        tx, ty, tz = to_coords[i]
        pt = piece_types[i]
        
        is_check = False
        
        # Calculate difference vector
        dx = kx - tx
        dy = ky - ty
        dz = kz - tz
        
        # Check if piece is slider
        if is_slider[pt]:
            # For sliders, we need to check if aligned with any vector
            # and if path is clear
            
            # Iterate through vectors for this piece
            n_vecs = vector_counts[attacker_color_idx, pt]
            for v in range(n_vecs):
                vx = attack_vectors[attacker_color_idx, pt, v, 0]
                vy = attack_vectors[attacker_color_idx, pt, v, 1]
                vz = attack_vectors[attacker_color_idx, pt, v, 2]
                
                # Check alignment: cross product should be zero, or simpler:
                # diff must be multiple of vector.
                # Since vectors are unit-ish (e.g. 1,0,0 or 1,1,0), we can check steps.
                
                steps = 0
                if vx != 0:
                    if dx % vx != 0: continue
                    steps = dx // vx
                elif dx != 0: continue
                
                if vy != 0:
                    if dy % vy != 0: continue
                    s = dy // vy
                    if steps == 0: steps = s
                    elif steps != s: continue
                elif dy != 0: continue
                
                if vz != 0:
                    if dz % vz != 0: continue
                    s = dz // vz
                    if steps == 0: steps = s
                    elif steps != s: continue
                elif dz != 0: continue
                
                if steps <= 0: continue # Wrong direction or same square
                
                # Aligned! Now check blocking.
                blocked = False
                cx, cy, cz = tx, ty, tz
                for _ in range(steps - 1):
                    cx += vx
                    cy += vy
                    cz += vz
                    if occupancy_grid[cx, cy, cz] != 0:
                        blocked = True
                        break
                
                if not blocked:
                    is_check = True
                    break
                    
        else:
            # Leaper (Knight, Pawn, King, Priest)
            # Check if diff matches any vector exactly
            n_vecs = vector_counts[attacker_color_idx, pt]
            for v in range(n_vecs):
                vx = attack_vectors[attacker_color_idx, pt, v, 0]
                vy = attack_vectors[attacker_color_idx, pt, v, 1]
                vz = attack_vectors[attacker_color_idx, pt, v, 2]
                
                if dx == vx and dy == vy and dz == vz:
                    is_check = True
                    break

        if is_check:
            rewards[i] = check_reward

    return rewards


@njit(cache=True, fastmath=True, parallel=True)
def _compute_king_proximity_rewards_vectorized(
    to_coords: np.ndarray,
    enemy_king_pos: np.ndarray,
    proximity_reward: float
) -> np.ndarray:
    """Vectorized king proximity rewards - reward for moving within 1 space of enemy king.
    
    Args:
        to_coords: Destination coordinates for moves (n_moves, 3)
        enemy_king_pos: Enemy king position (3,)
        proximity_reward: Reward for being adjacent to enemy king
    
    Returns:
        Array of proximity rewards for each move
    """
    n_moves = len(to_coords)
    rewards = np.zeros(n_moves, dtype=FLOAT_DTYPE)
    
    if enemy_king_pos is None or enemy_king_pos.size == 0:
        return rewards
    
    kx, ky, kz = enemy_king_pos[0], enemy_king_pos[1], enemy_king_pos[2]
    
    for i in prange(n_moves):
        tx, ty, tz = to_coords[i]
        
        # Calculate Manhattan distance to king
        distance = abs(tx - kx) + abs(ty - ky) + abs(tz - kz)
        
        # Reward if within 1 space (but not on the king itself)
        if distance == 1:
            rewards[i] = proximity_reward
    
    return rewards


@njit(cache=True, fastmath=True, parallel=True)
def _compute_next_state_repetition_penalty(
    from_coords: np.ndarray,
    to_coords: np.ndarray,
    from_types: np.ndarray,
    from_colors: np.ndarray,
    captured_types: np.ndarray,
    captured_colors: np.ndarray,
    current_zkey: int,
    position_keys: np.ndarray,
    position_counts: np.ndarray,
    zobrist_table: np.ndarray,
    penalty_weight: float
) -> np.ndarray:
    """Vectorized check for moves that lead to repeated positions."""
    n_moves = len(from_coords)
    penalties = np.zeros(n_moves, dtype=FLOAT_DTYPE)
    
    for i in prange(n_moves):
        # 1. Calculate next Zkey incrementally
        next_zkey = current_zkey
        
        # Remove moving piece from source
        p_type = from_types[i]
        p_color = from_colors[i]
        
        if p_type == 0: continue 
        
        # Zobrist table from cache.caches.zobrist is 5D: (piece_type, color_idx, x, y, z)
        # piece_type is 0-indexed (p_type - 1)
        # color_idx is 0-indexed (p_color - 1)
        
        fx, fy, fz = from_coords[i]
        next_zkey ^= zobrist_table[p_type - 1, p_color - 1, fx, fy, fz]
        
        # Add moving piece to dest
        tx, ty, tz = to_coords[i]
        next_zkey ^= zobrist_table[p_type - 1, p_color - 1, tx, ty, tz]
        
        # Remove captured piece if any
        c_type = captured_types[i]
        if c_type != 0:
            c_color = captured_colors[i]
            next_zkey ^= zobrist_table[c_type - 1, c_color - 1, tx, ty, tz]
            
        # 2. Check repetition in game state history
        idx = np.searchsorted(position_keys, next_zkey)
        if idx < position_keys.size and position_keys[idx] == next_zkey:
            count = position_counts[idx]
            # Apply penalty
            if count >= 2: # 3rd repetition imminent (Draw)
                penalties[i] = penalty_weight * 5.0 # Heavy penalty
            elif count >= 1: # 2nd repetition
                penalties[i] = penalty_weight
                
    return penalties


@njit(cache=True, fastmath=True, parallel=True)
def _compute_piece_diversity_rewards_vectorized(
    from_coords: np.ndarray,
    history_from_coords: np.ndarray,
    diversity_reward: float,
    repetition_penalty: float
) -> np.ndarray:
    """Vectorized piece diversity rewards - favor moving different pieces.
    
    Args:
        from_coords: array of from coordinates for candidate moves (n_moves, 3)
        history_from_coords: array of from coordinates from recent move history (n_history, 3)
        diversity_reward: reward for moving a piece not recently moved
        repetition_penalty: penalty for moving a piece that was moved multiple times recently
    
    Returns:
        Array of diversity rewards/penalties for each move
    """
    n_moves = len(from_coords)
    rewards = np.zeros(n_moves, dtype=FLOAT_DTYPE)
    
    if history_from_coords.shape[0] == 0:
        # No history, all moves get diversity bonus
        rewards[:] = diversity_reward
        return rewards
    
    for i in prange(n_moves):
        fx, fy, fz = from_coords[i]
        
        # Count how many times this piece appears in recent history
        move_count = 0
        for j in range(history_from_coords.shape[0]):
            hx, hy, hz = history_from_coords[j]
            if hx == fx and hy == fy and hz == fz:
                move_count += 1
        
        # Apply rewards/penalties based on move count
        if move_count == 0:
            # Not moved recently - diversity bonus
            rewards[i] = diversity_reward
        elif move_count == 1:
            # Moved once - small bonus
            rewards[i] = diversity_reward * 0.25
        else:
            # Moved 2+ times - penalty for overuse
            rewards[i] = repetition_penalty * min(move_count - 1, 3)  # Cap at 3x penalty
    
    return rewards


def _get_recent_history_coords(state: 'GameState', n_recent: int = 8) -> np.ndarray:
    """Extract from_coords from recent move history.
    
    Args:
        state: GameState object
        n_recent: Number of recent moves to consider
    
    Returns:
        Array of from coordinates (n_history, 3)
    """
    if not hasattr(state, 'history') or len(state.history) == 0:
        return np.empty((0, 3), dtype=COORD_DTYPE)
    
    # Get the last n_recent moves
    # history is a deque, so we need to convert to list for negative indexing
    history_len = min(n_recent, len(state.history))
    
    # Convert deque to list and get last n items
    history_list = list(state.history)
    recent_history = history_list[-history_len:]
    
    # Extract from_coords from structured array
    from_coords = np.zeros((history_len, 3), dtype=COORD_DTYPE)
    for i, move in enumerate(recent_history):
        from_coords[i, 0] = move['from_x']
        from_coords[i, 1] = move['from_y']
        from_coords[i, 2] = move['from_z']
    
    return from_coords



class OpponentBase:
    def __init__(self, color: Color):
        self.color = color
        self.repetition_penalty = -1.0
    
    def reward(self, state: 'GameState', move: Move) -> float:
        """Single move reward - delegates to batch version."""
        moves_array = np.array([[move.from_coord[0], move.from_coord[1], move.from_coord[2],
                                move.to_coord[0], move.to_coord[1], move.to_coord[2]]],
                               dtype=COORD_DTYPE)

        rewards = self.batch_reward(state, moves_array)
        return float(rewards[0])

    def batch_reward(self, state: 'GameState', moves: np.ndarray) -> np.ndarray:
        """Batch reward computation - MUST be overridden."""
        raise NotImplementedError
    
    @staticmethod
    def get_halfmove_penalty(halfmove_clock: int) -> float:
        """Progressive penalty as game approaches move rule draw - PRIORITY 2."""
        if halfmove_clock < 50:
            return 0.0
        elif halfmove_clock < 70:
            return (halfmove_clock - 50) * 0.5  # Increased from 0.3
        elif halfmove_clock < 90:
            return (50 - 50) * 0.5 + (halfmove_clock - 70) * 1.0  # Increased from 0.5
        elif halfmove_clock < 120:
            return ((50 - 50) * 0.5 + (90 - 70) * 1.0 + (halfmove_clock - 90) * 2.0)  # Increased from 1.0
        else:
            # Extreme penalty near draw limit (150 halfmoves or 300)
            base = ((50 - 50) * 0.5 + (90 - 70) * 1.0 + (120 - 90) * 2.0)
            return base + (halfmove_clock - 120) * 3.0  # Increased from 2.0
    
    @staticmethod  
    def get_position_repetition_penalty(state: 'GameState') -> float:
        """Get penalty based on how many times current position has occurred."""
        current_zkey = state.zkey
        idx = np.searchsorted(state._position_keys, current_zkey)
        
        if idx < state._position_keys.size and state._position_keys[idx] == current_zkey:
            count = state._position_counts[idx]
            if count >= 4:
                return -10.0  # Very close to 5-fold repetition draw!
            elif count >= 3:
                return -5.0   # 4-fold repetition
            elif count >= 2:
                return -2.0   # 3-fold repetition
            elif count >= 1:
                return -0.5   # 2-fold
        return 0.0

    def observe(self, state: 'GameState', move: Move):
        """Observe a move - now a no-op since we use game state tracking."""
        pass
    
    def _compute_base_rewards(self, state: 'GameState', moves: np.ndarray, 
                               from_coords: np.ndarray, to_coords: np.ndarray,
                               from_types: np.ndarray, from_colors: np.ndarray,
                               captured_types: np.ndarray, captured_colors: np.ndarray) -> tuple:
        """Compute common rewards shared by all opponents.
        
        Returns:
            (rewards, halfmove_clock) - Base rewards array and halfmove clock value
        """
        rewards = np.zeros(len(moves), dtype=FLOAT_DTYPE)
        
        # 1. Repetition penalty (PRIORITY 2)
        repetition_penalties = _compute_next_state_repetition_penalty(
            from_coords, to_coords, from_types, from_colors,
            captured_types, captured_colors,
            state.zkey, state._position_keys, state._position_counts,
            ZOBRIST_TABLE, self.repetition_penalty
        )
        rewards += repetition_penalties
        
        # 2. Halfmove clock penalty (PRIORITY 2)
        halfmove_clock = getattr(state, 'halfmove_clock', 0)
        clock_penalty = self.get_halfmove_penalty(halfmove_clock)
        rewards -= clock_penalty
        
        # 3. Pawn move bonus (resets halfmove clock) - PRIORITY 2
        pawn_mask = (from_types == PieceType.PAWN.value)
        if halfmove_clock > 70 and np.any(pawn_mask):
            rewards[pawn_mask] += 1.5
        if halfmove_clock > 90 and np.any(pawn_mask):
            rewards[pawn_mask] += 1.5  # Total +3.0 at 90+
            
        return rewards, halfmove_clock
    
    def _apply_capture_rewards(self, rewards: np.ndarray, to_coords: np.ndarray,
                                captured_colors: np.ndarray, captured_types: np.ndarray,
                                halfmove_clock: int, priest_bonus: float = 5.0, 
                                freezer_bonus: float = 3.0) -> None:
        """Apply capture rewards with priest/freezer priority and halfmove clock scaling.
        
        Args:
            rewards: Rewards array to modify in-place
            to_coords: Destination coordinates
            captured_colors: Colors of captured pieces
            captured_types: Types of captured pieces
            halfmove_clock: Current halfmove clock value
            priest_bonus: Bonus for capturing priests
            freezer_bonus: Bonus for capturing freezers
        """
        capture_rewards = _compute_capture_rewards_vectorized(
            to_coords, captured_colors, captured_types, self.color.value,
            priest_bonus=priest_bonus, freezer_bonus=freezer_bonus
        )
        
        # Boost capture rewards when clock is dangerously high (PRIORITY 2)
        if halfmove_clock > 70:
            capture_rewards *= 1.5
        if halfmove_clock > 90:
            capture_rewards *= (2.5 / 1.5)  # Total 2.5x at 90+
            
        rewards += capture_rewards
    
    def _apply_diversity_rewards(self, rewards: np.ndarray, state: 'GameState',
                                  from_coords: np.ndarray, n_recent: int = 8) -> None:
        """Apply piece diversity rewards (PRIORITY 3).
        
        Args:
            rewards: Rewards array to modify in-place
            state: Game state
            from_coords: Source coordinates for moves
            n_recent: Number of recent moves to consider
        """
        history_coords = _get_recent_history_coords(state, n_recent=n_recent)
        diversity_rewards = _compute_piece_diversity_rewards_vectorized(
            from_coords, history_coords,
            diversity_reward=1.0,
            repetition_penalty=-1.0
        )
        rewards += diversity_rewards
    
    def _apply_geomancer_penalty(self, rewards: np.ndarray, from_types: np.ndarray,
                                  penalty: float = -0.1) -> None:
        """Apply geomancer overuse penalty (PRIORITY 4).
        
        Args:
            rewards: Rewards array to modify in-place
            from_types: Piece types being moved
            penalty: Penalty value (negative)
        """
        geomancer_mask = (from_types == PieceType.GEOMANCER.value)
        if np.any(geomancer_mask):
            rewards[geomancer_mask] += penalty
    
    def _apply_check_rewards(self, rewards: np.ndarray, cache_manager,
                              to_coords: np.ndarray, from_types: np.ndarray,
                              check_reward: float = 12.0, 
                              proximity_reward: float = 0.08) -> None:
        """Apply check potential and king proximity rewards (PRIORITY 1).
        
        Only applies if enemy has no priests.
        
        Args:
            rewards: Rewards array to modify in-place
            cache_manager: Cache manager from game state
            to_coords: Destination coordinates
            from_types: Piece types being moved
            check_reward: Reward for putting king in check
            proximity_reward: Reward for being adjacent to king
        """
        enemy_color = self.color.opposite()
        if not cache_manager.occupancy_cache.has_priest(enemy_color):
            enemy_king_pos = cache_manager.occupancy_cache.find_king(enemy_color)
            if enemy_king_pos is not None:
                check_rewards = _compute_check_potential_vectorized(
                    to_coords, from_types, enemy_king_pos, check_reward,
                    cache_manager.occupancy_cache._occ,
                    ATTACK_VECTORS, VECTOR_COUNTS, IS_SLIDER,
                    0 if self.color == Color.WHITE else 1
                )
                rewards += check_rewards
                
                # King proximity reward
                proximity_rewards = _compute_king_proximity_rewards_vectorized(
                    to_coords, enemy_king_pos, proximity_reward
                )
                rewards += proximity_rewards

    def _get_enemy_priest_positions(self, cache_manager) -> np.ndarray:
        """Get all enemy priest positions as a single array (vectorized).
        
        This is a shared helper method used by multiple opponent types.
        """
        enemy_color = self.color.opposite()
        all_positions = cache_manager.occupancy_cache.get_positions(enemy_color.value)

        if len(all_positions) == 0:
            return np.empty((0, 3), dtype=COORD_DTYPE)

        # Filter for priests using vectorized batch lookup (10-50x faster than iteration)
        _, types = cache_manager.occupancy_cache.batch_get_attributes(all_positions)
        priest_mask = (types == PieceType.PRIEST.value)
        
        return all_positions[priest_mask]

    def select_move(self, state: 'GameState', from_logits: np.ndarray, to_logits: np.ndarray,
                    legal_moves, epsilon: float = 0.1):
        import random
        from game3d.common.coord_utils import coord_to_idx

        if random.random() < epsilon:
            if isinstance(legal_moves, np.ndarray):
                random_idx = random.randint(0, len(legal_moves) - 1)
                move_struct = legal_moves[random_idx]
                from_coord = np.array([move_struct['from_x'], move_struct['from_y'], move_struct['from_z']],
                                     dtype=COORD_DTYPE)
                to_coord = np.array([move_struct['to_x'], move_struct['to_y'], move_struct['to_z']],
                                   dtype=COORD_DTYPE)
                return Move(from_coord=from_coord, to_coord=to_coord, flags=move_struct['move_type'])

        from_indices = np.array([coord_to_idx(mv.from_coord) for mv in legal_moves], dtype=INDEX_DTYPE)
        to_indices = np.array([coord_to_idx(mv.to_coord) for mv in legal_moves], dtype=INDEX_DTYPE)

        move_probs = from_logits[from_indices] + to_logits[to_indices]

        max_logit = np.max(move_probs)
        exp_logit = np.exp(move_probs - max_logit)
        sum_exp = np.sum(exp_logit)

        if sum_exp == 0 or np.isnan(sum_exp):
            move_probs_soft = np.ones_like(move_probs) / len(move_probs)
        else:
            move_probs_soft = exp_logit / sum_exp

        rewards_np = self.batch_reward(state, legal_moves)

        alpha = 0.5
        scores = move_probs_soft + alpha * rewards_np

        best_idx = int(np.argmax(scores))
        return legal_moves[best_idx]


# =============================================================================
# ADAPTIVE OPPONENT (formerly AdversarialOpponent)
# =============================================================================

class AdaptiveOpponent(OpponentBase):
    def batch_reward(self, state: 'GameState', moves: np.ndarray) -> np.ndarray:
        """Vectorized reward computation for ALL moves."""
        cache_manager = state.cache_manager
        from_coords = moves[:, :3]
        to_coords = moves[:, 3:6]

        # Get attributes (using unsafe for performance - coordinates pre-validated by generator)
        captured_colors, captured_types = cache_manager.occupancy_cache.batch_get_attributes_unsafe(to_coords)
        from_colors, from_types = cache_manager.occupancy_cache.batch_get_attributes_unsafe(from_coords)


        # Base rewards (PRIORITY 2: repetition, halfmove clock, pawn bonus)
        rewards, halfmove_clock = self._compute_base_rewards(
            state, moves, from_coords, to_coords, from_types, from_colors, 
            captured_types, captured_colors
        )

        # Capture rewards (PRIORITY 1: priest/freezer capture)
        self._apply_capture_rewards(rewards, to_coords, captured_colors, captured_types, halfmove_clock)
        
        # Piece diversity (PRIORITY 3)
        self._apply_diversity_rewards(rewards, state, from_coords)
        
        # Attack/Defense rewards (PRIORITY 4)
        enemy_moves = cache_manager.move_cache.get_cached_moves(self.color.opposite())
        if enemy_moves is not None and len(enemy_moves) > 0:
            enemy_to_coords = enemy_moves[:, 3:6] if enemy_moves.ndim == 2 else enemy_moves['to_x', 'to_y', 'to_z']
            attack_rewards = _compute_attack_defense_rewards_vectorized(
                from_coords, to_coords,
                np.empty((0, 3), dtype=COORD_DTYPE),
                enemy_to_coords
            )
            rewards += attack_rewards

        # Geomancer penalty (PRIORITY 4)
        self._apply_geomancer_penalty(rewards, from_types)

        # Priest hunting (PRIORITY 1)
        enemy_color = self.color.opposite()
        if cache_manager.occupancy_cache.has_priest(enemy_color):
            enemy_priest_positions = self._get_enemy_priest_positions(cache_manager)
            if len(enemy_priest_positions) > 0:
                approach_rewards = _compute_distance_rewards_adaptive(
                    from_coords, to_coords, enemy_priest_positions,
                    from_types, 0.5
                )
                rewards += approach_rewards
        else:
            # No priests - check/proximity rewards (PRIORITY 1/4)
            self._apply_check_rewards(rewards, cache_manager, to_coords, from_types)
            # King approach (low priority)
            enemy_king_pos = cache_manager.occupancy_cache.find_king(enemy_color)
            if enemy_king_pos is not None:
                relevant_mask = (from_types != PieceType.KING.value)
                if np.any(relevant_mask):
                    king_targets = np.array([enemy_king_pos], dtype=COORD_DTYPE)
                    distance_rewards = _compute_distance_rewards_adaptive(
                        from_coords[relevant_mask], to_coords[relevant_mask], 
                        king_targets, from_types[relevant_mask], 0.05
                    )

                    rewards[relevant_mask] += distance_rewards

        return rewards


# =============================================================================
# CENTER CONTROL OPPONENT
# =============================================================================

class CenterControlOpponent(OpponentBase):
    def batch_reward(self, state: 'GameState', moves: np.ndarray) -> np.ndarray:
        """Vectorized center control rewards."""
        cache_manager = state.cache_manager
        to_coords = moves[:, 3:6]
        from_coords = moves[:, :3]

        # Get attributes (using unsafe for performance - coordinates pre-validated by generator)
        captured_colors, captured_types = cache_manager.occupancy_cache.batch_get_attributes_unsafe(to_coords)
        from_colors, from_types = cache_manager.occupancy_cache.batch_get_attributes_unsafe(from_coords)

        # Base rewards (PRIORITY 2)
        rewards, halfmove_clock = self._compute_base_rewards(
            state, moves, from_coords, to_coords, from_types, from_colors, 
            captured_types, captured_colors
        )

        # Capture rewards (PRIORITY 1)
        self._apply_capture_rewards(rewards, to_coords, captured_colors, captured_types, halfmove_clock)

        # Piece diversity (PRIORITY 3)
        self._apply_diversity_rewards(rewards, state, from_coords)
        
        # Center control (vectorized) - PRIORITY 4
        center_rewards = _compute_center_control_rewards_vectorized(to_coords)
        rewards += center_rewards * 0.67

        # Bonus for moving specific pieces TO center (reuse already-computed center_rewards)
        relevant_pieces = (PieceType.KNIGHT.value, PieceType.BISHOP.value, PieceType.QUEEN.value)
        piece_mask = np.isin(from_types, relevant_pieces)
        if np.any(piece_mask):
            rewards[piece_mask] += center_rewards[piece_mask] * 0.33


        # Geomancer penalty (PRIORITY 4)
        self._apply_geomancer_penalty(rewards, from_types)

        # Check rewards (PRIORITY 1)
        self._apply_check_rewards(rewards, cache_manager, to_coords, from_types)

        return rewards


# =============================================================================
# PIECE CAPTURE OPPONENT
# =============================================================================

class PieceCaptureOpponent(OpponentBase):
    def batch_reward(self, state: 'GameState', moves: np.ndarray) -> np.ndarray:
        """Pure capture-focused rewards."""
        to_coords = moves[:, 3:6]
        from_coords = moves[:, :3]

        # Get attributes (using unsafe for performance - coordinates pre-validated by generator)
        from_colors, from_types = state.cache_manager.occupancy_cache.batch_get_attributes_unsafe(from_coords)
        captured_colors, captured_types = state.cache_manager.occupancy_cache.batch_get_attributes_unsafe(to_coords)
        
        # Base rewards (PRIORITY 2)
        rewards, halfmove_clock = self._compute_base_rewards(
            state, moves, from_coords, to_coords, from_types, from_colors, 
            captured_types, captured_colors
        )

        # Capture rewards (PRIORITY 1) - primary focus
        self._apply_capture_rewards(rewards, to_coords, captured_colors, captured_types, halfmove_clock)

        # Piece diversity (PRIORITY 3)
        self._apply_diversity_rewards(rewards, state, from_coords)
        
        # Geomancer penalty (PRIORITY 4)
        self._apply_geomancer_penalty(rewards, from_types)

        # Check rewards (PRIORITY 1)
        self._apply_check_rewards(rewards, state.cache_manager, to_coords, from_types)

        return rewards


# =============================================================================
# PRIEST HUNTER OPPONENT
# =============================================================================

class PriestHunterOpponent(OpponentBase):
    def batch_reward(self, state: 'GameState', moves: np.ndarray) -> np.ndarray:
        """Prioritize priest captures above all else."""
        cache_manager = state.cache_manager
        to_coords = moves[:, 3:6]
        from_coords = moves[:, :3]

        # Get attributes (using unsafe for performance - coordinates pre-validated by generator)
        from_colors, from_types = cache_manager.occupancy_cache.batch_get_attributes_unsafe(from_coords)
        captured_colors, captured_types = cache_manager.occupancy_cache.batch_get_attributes_unsafe(to_coords)
        
        # Base rewards (PRIORITY 2)
        rewards, halfmove_clock = self._compute_base_rewards(
            state, moves, from_coords, to_coords, from_types, from_colors, 
            captured_types, captured_colors
        )

        # Capture rewards (priest-focused) with high-clock multiplier (PRIORITY 1)
        # Use vectorized helper with custom bonuses scaled by clock multiplier
        clock_multiplier = 1.0
        if halfmove_clock > 70:
            clock_multiplier = 1.5
        if halfmove_clock > 90:
            clock_multiplier = 2.5
        
        # Apply capture rewards with priest-focused scaling
        self._apply_capture_rewards(
            rewards, to_coords, captured_colors, captured_types, 
            halfmove_clock=0,  # Don't apply additional clock scaling (we handle it manually)
            priest_bonus=0.8 * clock_multiplier,
            freezer_bonus=0.1 * clock_multiplier
        )


        # Approach enemy priests (PRIORITY 1) - compute once, reuse
        enemy_priest_positions = self._get_enemy_priest_positions(cache_manager)
        if len(enemy_priest_positions) > 0:
            approach_rewards = _compute_distance_rewards_adaptive(
                from_coords, to_coords, enemy_priest_positions,
                from_types, 0.5
            )
            rewards += approach_rewards

        # Piece diversity (PRIORITY 3)
        self._apply_diversity_rewards(rewards, state, from_coords)
        
        # Geomancer penalty (PRIORITY 4)
        self._apply_geomancer_penalty(rewards, from_types)

        # Focus on king if no enemy priests (reuse computed priest positions)
        enemy_color = self.color.opposite()
        if len(enemy_priest_positions) == 0 and not cache_manager.occupancy_cache.has_priest(enemy_color):
            self._apply_check_rewards(rewards, cache_manager, to_coords, from_types)
            # King approach (low priority)
            enemy_king_pos = cache_manager.occupancy_cache.find_king(enemy_color)
            if enemy_king_pos is not None:
                relevant_mask = (from_types != PieceType.KING.value)
                if np.any(relevant_mask):
                    king_targets = np.array([enemy_king_pos], dtype=COORD_DTYPE)
                    distance_rewards = _compute_distance_rewards_adaptive(
                        from_coords[relevant_mask], to_coords[relevant_mask], 
                        king_targets, from_types[relevant_mask], 0.05
                    )
                    rewards[relevant_mask] += distance_rewards

        return rewards


# =============================================================================
# GRAPH AWARE OPPONENT
# =============================================================================

class GraphAwareOpponent(OpponentBase):
    def batch_reward(self, state: 'GameState', moves: np.ndarray) -> np.ndarray:
        """Considers piece coordination and local piece density."""
        cache_manager = state.cache_manager
        to_coords = moves[:, 3:6]
        from_coords = moves[:, :3]

        # Get attributes (using unsafe for performance - coordinates pre-validated by generator)
        from_colors, from_types = cache_manager.occupancy_cache.batch_get_attributes_unsafe(from_coords)
        captured_colors, captured_types = cache_manager.occupancy_cache.batch_get_attributes_unsafe(to_coords)
        
        # Base rewards (PRIORITY 2)
        rewards, halfmove_clock = self._compute_base_rewards(
            state, moves, from_coords, to_coords, from_types, from_colors, 
            captured_types, captured_colors
        )

        # Capture rewards (PRIORITY 1)
        self._apply_capture_rewards(rewards, to_coords, captured_colors, captured_types, halfmove_clock)

        # Piece diversity (PRIORITY 3)
        self._apply_diversity_rewards(rewards, state, from_coords)
        
        # Piece coordination bonus (vectorized - 20-100x faster) - PRIORITY 4
        n_moves = len(to_coords)
        neighbor_offsets = np.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]], dtype=COORD_DTYPE)
        all_neighbors = to_coords[:, None, :] + neighbor_offsets[None, :, :]
        neighbors_flat = all_neighbors.reshape(-1, 3)
        valid_mask = ((neighbors_flat >= 0) & (neighbors_flat < 9)).all(axis=1)
        
        # Optimized: use unsafe batch operations since we pre-filter with valid_mask
        neighbor_colors_full = np.zeros(len(neighbors_flat), dtype=np.int8)
        
        if np.any(valid_mask):
            valid_neighbors = neighbors_flat[valid_mask]
            # Use unsafe for performance - coordinates already validated by valid_mask
            valid_colors, _ = cache_manager.occupancy_cache.batch_get_attributes_unsafe(valid_neighbors)
            neighbor_colors_full[valid_mask] = valid_colors
        
        neighbor_colors_grid = neighbor_colors_full.reshape(n_moves, 6)
        ally_mask = (neighbor_colors_grid == self.color.value)
        allies_near = ally_mask.sum(axis=1)
        coordination_bonus = np.where(allies_near > 1, 0.1 * (allies_near - 1), 0.0)
        rewards += coordination_bonus

        # Geomancer penalty (PRIORITY 4)
        self._apply_geomancer_penalty(rewards, from_types)

        # Check rewards (PRIORITY 1)
        self._apply_check_rewards(rewards, cache_manager, to_coords, from_types)

        return rewards


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_opponent(opponent_type: str, color: Color) -> OpponentBase:
    types = {
        'adversarial': AdaptiveOpponent,
        'center_control': CenterControlOpponent,
        'piece_capture': PieceCaptureOpponent,
        'priest_hunter': PriestHunterOpponent,
        'graph_aware': GraphAwareOpponent,
    }
    if opponent_type not in types:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    return types[opponent_type](color)

# Explicitly list available opponents - matches the keys above
AVAILABLE_OPPONENTS = ['priest_hunter', 'adversarial', 'center_control', 'piece_capture', 'graph_aware']

__all__ = [
    "OpponentBase",
    "AdaptiveOpponent",
    "CenterControlOpponent",
    "PieceCaptureOpponent",
    "PriestHunterOpponent",
    "GraphAwareOpponent",
    "create_opponent",
    "AVAILABLE_OPPONENTS",
]
