import numpy as np
from numba import njit, prange
import logging
from typing import Optional

from game3d.common.shared_types import Color, PieceType, COLOR_EMPTY, COORD_DTYPE, INDEX_DTYPE, FLOAT_DTYPE, HASH_DTYPE
from game3d.movement.movepiece import Move

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

@njit(cache=True, fastmath=True, parallel=True)
def _compute_repetition_penalties_vectorized(
    moves: np.ndarray,
    position_hashes: np.ndarray,
    recent_moves: np.ndarray,
    move_count: int,
    repetition_penalty: float
) -> np.ndarray:
    """Vectorized repetition penalty for ALL moves at once."""
    n_moves = moves.shape[0]
    penalties = np.zeros(n_moves, dtype=FLOAT_DTYPE)

    for i in prange(n_moves):
        move = moves[i]

        # Check position repetition in position_hashes
        count = 0
        for j in range(len(position_hashes)):
            if position_hashes[j] == move[0]:  # Simplified hash check
                count += 1

        if count >= 3:
            penalties[i] = repetition_penalty * 4
        elif count >= 2:
            penalties[i] = repetition_penalty * 2
        elif count >= 1:
            penalties[i] = repetition_penalty

        # Check move repetition pattern (last 4 moves)
        if move_count >= 4:
            # Compare move patterns in recent_moves array
            if (recent_moves[0, 0] == recent_moves[2, 0] and
                recent_moves[1, 0] == recent_moves[3, 0]):
                penalties[i] += repetition_penalty * 3

    return penalties


@njit(cache=True, fastmath=True, parallel=True)
def _compute_capture_rewards_vectorized(
    to_coords: np.ndarray,
    captured_colors: np.ndarray,
    captured_types: np.ndarray,
    player_color: int,
    priest_bonus: float,
    queen_rook_bonus: float
) -> np.ndarray:
    """Vectorized capture reward calculation."""
    n_moves = len(to_coords)
    rewards = np.zeros(n_moves, dtype=FLOAT_DTYPE)

    enemy_color = 2 if player_color == 1 else 1

    for i in prange(n_moves):
        if captured_colors[i] == enemy_color:
            rewards[i] = 0.5  # Base capture reward

            if captured_types[i] == PieceType.PRIEST.value:
                rewards[i] += priest_bonus
            elif captured_types[i] in (PieceType.QUEEN.value, PieceType.ROOK.value):
                rewards[i] += queen_rook_bonus

    return rewards


@njit(cache=True, fastmath=True, parallel=True)
def _compute_distance_rewards_vectorized(
    from_coords: np.ndarray,
    to_coords: np.ndarray,
    target_positions: np.ndarray,
    piece_types: np.ndarray,
    reward_per_step: float
) -> np.ndarray:
    """Vectorized distance-based rewards (e.g., king approach, priest hunting)."""
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
    check_reward: float
) -> np.ndarray:
    """Vectorized check potential reward (approximate)."""
    n_moves = len(to_coords)
    rewards = np.zeros(n_moves, dtype=FLOAT_DTYPE)
    
    if enemy_king_pos is None:
        return rewards
        
    kx, ky, kz = enemy_king_pos[0], enemy_king_pos[1], enemy_king_pos[2]

    for i in prange(n_moves):
        # Skip if piece is empty (shouldn't happen in valid moves)
        if piece_types[i] == 0:
            continue
            
        tx, ty, tz = to_coords[i]
        pt = piece_types[i]
        
        is_check = False
        
        # Approximate check detection based on piece type and destination
        # This does NOT account for blocking pieces (ray casting is expensive here)
        # But for a reward signal, it's often sufficient
        
        dx = abs(tx - kx)
        dy = abs(ty - ky)
        dz = abs(tz - kz)
        
        if pt == PieceType.PAWN.value:
            # Pawn attacks diagonally forward (simplified for 3D: any diagonal step?)
            # Actually pawns capture differently. Let's assume standard 3D pawn capture:
            # Forward diagonal. But direction depends on color. 
            # Simplified: if dist is small and not same file
            if dx <= 1 and dy <= 1 and dz <= 1 and (dx + dy + dz) >= 2:
                 is_check = True
                 
        elif pt == PieceType.KNIGHT.value:
            # 1, 2 jump
            dist_sq = dx*dx + dy*dy + dz*dz
            if dist_sq == 5: # 1^2 + 2^2 + 0^2
                is_check = True
                
        elif pt == PieceType.BISHOP.value:
            # Diagonal
            if (dx == dy and dz == 0) or (dx == dz and dy == 0) or (dy == dz and dx == 0):
                is_check = True
            elif dx == dy and dy == dz: # Tri-diagonal
                is_check = True
                
        elif pt == PieceType.ROOK.value:
            # Orthogonal
            if (dx == 0 and dy == 0) or (dx == 0 and dz == 0) or (dy == 0 and dz == 0):
                is_check = True
                
        elif pt == PieceType.QUEEN.value:
            # Bishop + Rook
            if (dx == 0 and dy == 0) or (dx == 0 and dz == 0) or (dy == 0 and dz == 0):
                is_check = True
            elif (dx == dy and dz == 0) or (dx == dz and dy == 0) or (dy == dz and dx == 0):
                is_check = True
            elif dx == dy and dy == dz:
                is_check = True
                
        elif pt == PieceType.PRIEST.value:
             # Priest moves like King (1 step) but captures? 
             # Priest usually doesn't capture or check in some variants, but let's assume it can for now if it moves.
             if dx <= 1 and dy <= 1 and dz <= 1:
                 is_check = True

        if is_check:
            rewards[i] = check_reward

    return rewards

# =============================================================================
# OPTIMIZED OPPONENT BASE CLASS
# =============================================================================

class OpponentBase:
    def __init__(self, color: Color):
        self.color = color
        # ✅ OPTIMIZED: Pre-allocated circular buffer instead of append
        self._position_hashes_buffer = np.zeros(1000, dtype=HASH_DTYPE)  # 1000-move buffer
        self._position_hashes_count = 0
        self.repetition_penalty = -5.0
        self.recent_moves = np.zeros((10, 2, 3), dtype=COORD_DTYPE)
        self.move_count = 0
    
    @property
    def position_hashes(self) -> np.ndarray:
        """Get active portion of position hashes buffer."""
        return self._position_hashes_buffer[:self._position_hashes_count]

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

    def get_repetition_penalty(self, state: 'GameState', move: Move) -> float:
        """Single move penalty - delegates to vectorized version."""
        move_array = np.array([[move.from_coord[0], move.from_coord[1], move.from_coord[2],
                               move.to_coord[0], move.to_coord[1], move.to_coord[2]]],
                              dtype=COORD_DTYPE)

        penalties = _compute_repetition_penalties_vectorized(
            move_array, self.position_hashes, self.recent_moves,
            self.move_count, self.repetition_penalty
        )
        return float(penalties[0])

    def get_repetition_penalties_batch(self, moves: np.ndarray) -> np.ndarray:
        """Vectorized repetition penalty for batch."""
        return _compute_repetition_penalties_vectorized(
            moves, self.position_hashes, self.recent_moves,
            self.move_count, self.repetition_penalty
        )

    def observe(self, state: 'GameState', move: Move):
        """Observe a move - optimized to avoid coordinate mutation."""
        board_hash = state.board.byte_hash()

        move_from = np.array([move.from_coord[0], move.from_coord[1], move.from_coord[2]],
                            dtype=COORD_DTYPE)
        move_to = np.array([move.to_coord[0], move.to_coord[1], move.to_coord[2]],
                          dtype=COORD_DTYPE)

        position_hash = hash((board_hash, tuple(move_from), tuple(move_to), state.color))

        # ✅ OPTIMIZED: Circular buffer instead of np.append
        if self._position_hashes_count < len(self._position_hashes_buffer):
            # Still have space in buffer
            self._position_hashes_buffer[self._position_hashes_count] = position_hash
            self._position_hashes_count += 1
        else:
            # Buffer full - shift left and add at end (FIFO)
            self._position_hashes_buffer[:-1] = self._position_hashes_buffer[1:]
            self._position_hashes_buffer[-1] = position_hash

        idx = self.move_count % 10
        self.recent_moves[idx, 0] = move_from
        self.recent_moves[idx, 1] = move_to
        self.move_count += 1

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

        # Extract coordinates
        from_coords = moves[:, :3]
        to_coords = moves[:, 3:6]

        # SINGLE vectorized lookup for all moves
        captured_colors, captured_types = cache_manager.occupancy_cache.batch_get_attributes(to_coords)
        from_colors, from_types = cache_manager.occupancy_cache.batch_get_attributes(from_coords)

        # Pre-allocate rewards array
        rewards = np.zeros(len(moves), dtype=FLOAT_DTYPE)

        # 1. Repetition penalty (vectorized)
        repetition_penalties = self.get_repetition_penalties_batch(moves)
        rewards += repetition_penalties

        # 2. Improved halfmove clock penalty (exponential scaling to avoid 50-move draw)
        halfmove_clock = getattr(state, 'halfmove_clock', 0)
        if halfmove_clock > 50:
            # Progressive penalties:
            # 50-70: -0.3 per move
            # 70-90: -0.5 per move 
            # 90+: -1.0 per move with exponential growth
            base_penalty = (halfmove_clock - 50) * 0.3
            if halfmove_clock > 70:
                base_penalty += (halfmove_clock - 70) * 0.2  # Total -0.5 per move
            if halfmove_clock > 90:
                base_penalty *= 1.5  # Exponential growth
            rewards -= base_penalty

        # 3. Capture rewards (vectorized) with bonus multiplier for high halfmove clock
        capture_rewards = _compute_capture_rewards_vectorized(
            to_coords, captured_colors, captured_types, self.color.value,
            priest_bonus=1.0, queen_rook_bonus=0.3
        )
        # Boost capture rewards when clock is dangerously high
        if halfmove_clock > 70:
            capture_rewards *= 1.5
        if halfmove_clock > 90:
            capture_rewards *= (2.5 / 1.5)  # Total 2.5x at 90+
        rewards += capture_rewards
        
        # 3b. Pawn move bonus (resets halfmove clock)
        pawn_mask = (from_types == PieceType.PAWN.value)
        if halfmove_clock > 70 and np.any(pawn_mask):
            rewards[pawn_mask] += 0.4
        if halfmove_clock > 90 and np.any(pawn_mask):
            rewards[pawn_mask] += 0.4  # Total +0.8 at 90+

        # 4. Attack/Defense rewards (only if enemy has moves)
        enemy_moves = cache_manager.move_cache.get_cached_moves(self.color.opposite())
        if enemy_moves is not None and len(enemy_moves) > 0:
            enemy_to_coords = enemy_moves[:, 3:6] if enemy_moves.ndim == 2 else enemy_moves['to_x', 'to_y', 'to_z']

            attack_rewards = _compute_attack_defense_rewards_vectorized(
                from_coords, to_coords,
                np.empty((0, 3), dtype=COORD_DTYPE),  # our attacks (skip for perf)
                enemy_to_coords
            )
            rewards += attack_rewards


        # 5. King approach (only if no enemy priests)
        enemy_priest_count = 0
        enemy_color = self.color.opposite()
        if cache_manager.occupancy_cache.has_priest(enemy_color):
             enemy_priest_count = 1 # Simplified check

        if enemy_priest_count == 0:
            enemy_king_pos = cache_manager.occupancy_cache.find_king(enemy_color)
            if enemy_king_pos is not None:
                # Filter relevant pieces
                relevant_mask = (from_types != PieceType.KING.value)
                if np.any(relevant_mask):
                    filtered_from = from_coords[relevant_mask]
                    filtered_to = to_coords[relevant_mask]
                    filtered_types = from_types[relevant_mask]

                    king_targets = np.array([enemy_king_pos], dtype=COORD_DTYPE)
                    distance_rewards = _compute_distance_rewards_vectorized(
                        filtered_from, filtered_to, king_targets,
                        filtered_types, 0.05
                    )
                    rewards[relevant_mask] += distance_rewards
                    
                    # NEW: Check reward
                    check_rewards = _compute_check_potential_vectorized(
                        filtered_to, filtered_types, enemy_king_pos, 1.5
                    )
                    rewards[relevant_mask] += check_rewards

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

        # Pre-compute everything in vectorized form
        captured_colors, captured_types = cache_manager.occupancy_cache.batch_get_attributes(to_coords)
        from_colors, from_types = cache_manager.occupancy_cache.batch_get_attributes(from_coords)

        rewards = np.zeros(len(moves), dtype=FLOAT_DTYPE)

        # 1. Repetition penalty
        rewards += self.get_repetition_penalties_batch(moves)

        # 2. Improved halfmove clock penalty
        halfmove_clock = getattr(state, 'halfmove_clock', 0)
        if halfmove_clock > 50:
            base_penalty = (halfmove_clock - 50) * 0.3
            if halfmove_clock > 70:
                base_penalty += (halfmove_clock - 70) * 0.2
            if halfmove_clock > 90:
                base_penalty *= 1.5
            rewards -= base_penalty

        # 3. Center control (vectorized)
        center_rewards = _compute_center_control_rewards_vectorized(to_coords)
        rewards += center_rewards

        # 4. Capture rewards with bonus for high clock
        capture_rewards = _compute_capture_rewards_vectorized(
            to_coords, captured_colors, captured_types, self.color.value,
            priest_bonus=1.0, queen_rook_bonus=0.0
        )
        if halfmove_clock > 70:
            capture_rewards *= 1.5
        if halfmove_clock > 90:
            capture_rewards *= (2.5 / 1.5)
        rewards += capture_rewards
        
        # 4b. Pawn move bonus
        pawn_mask = (from_types == PieceType.PAWN.value)
        if halfmove_clock > 70 and np.any(pawn_mask):
            rewards[pawn_mask] += 0.4
        if halfmove_clock > 90 and np.any(pawn_mask):
            rewards[pawn_mask] += 0.4

        # 5. Bonus for moving pieces TO center
        relevant_pieces = (PieceType.KNIGHT.value, PieceType.BISHOP.value, PieceType.QUEEN.value)
        piece_mask = np.isin(from_types, relevant_pieces)

        if np.any(piece_mask):
            center_rewards_pieces = _compute_center_control_rewards_vectorized(to_coords[piece_mask])
            rewards[piece_mask] += center_rewards_pieces * 0.33  # 0.1 / 0.3 ratio

        # 6. Conditional Check Reward
        enemy_color = self.color.opposite()
        if not cache_manager.occupancy_cache.has_priest(enemy_color):
            enemy_king_pos = cache_manager.occupancy_cache.find_king(enemy_color)
            if enemy_king_pos is not None:
                 check_rewards = _compute_check_potential_vectorized(
                    to_coords, from_types, enemy_king_pos, 1.5
                )
                 rewards += check_rewards

        return rewards


# =============================================================================
# PIECE CAPTURE OPPONENT
# =============================================================================

class PieceCaptureOpponent(OpponentBase):
    def batch_reward(self, state: 'GameState', moves: np.ndarray) -> np.ndarray:
        """Pure capture-focused rewards."""
        to_coords = moves[:, 3:6]

        rewards = np.zeros(len(moves), dtype=FLOAT_DTYPE)

        # 1. Repetition penalty
        rewards += self.get_repetition_penalties_batch(moves)

        # 2. Improved halfmove clock penalty
        halfmove_clock = getattr(state, 'halfmove_clock', 0)
        if halfmove_clock > 50:
            base_penalty = (halfmove_clock - 50) * 0.3
            if halfmove_clock > 70:
                base_penalty += (halfmove_clock - 70) * 0.2
            if halfmove_clock > 90:
                base_penalty *= 1.5
            rewards -= base_penalty

        # 3. Capture rewards (primary) with high-clock bonus
        captured_colors, captured_types = state.cache_manager.occupancy_cache.batch_get_attributes(to_coords)

        capture_rewards = _compute_capture_rewards_vectorized(
            to_coords, captured_colors, captured_types, self.color.value,
            priest_bonus=1.0, queen_rook_bonus=0.3
        )
        if halfmove_clock > 70:
            capture_rewards *= 1.5
        if halfmove_clock > 90:
            capture_rewards *= (2.5 / 1.5)
        rewards += capture_rewards

        # 4. Conditional Check Reward
        enemy_color = self.color.opposite()
        if not cache_manager.occupancy_cache.has_priest(enemy_color):
            enemy_king_pos = cache_manager.occupancy_cache.find_king(enemy_color)
            if enemy_king_pos is not None:
                 # Need from_types for check calculation
                 from_types = state.cache_manager.occupancy_cache.batch_get_attributes(moves[:, :3])[1]
                 check_rewards = _compute_check_potential_vectorized(
                    to_coords, from_types, enemy_king_pos, 1.5
                )
                 rewards += check_rewards

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

        rewards = np.zeros(len(moves), dtype=FLOAT_DTYPE)

        # 1. Repetition penalty
        rewards += self.get_repetition_penalties_batch(moves)

        # 2. Improved halfmove clock penalty
        halfmove_clock = getattr(state, 'halfmove_clock', 0)
        if halfmove_clock > 50:
            base_penalty = (halfmove_clock - 50) * 0.3
            if halfmove_clock > 70:
                base_penalty += (halfmove_clock - 70) * 0.2
            if halfmove_clock > 90:
                base_penalty *= 1.5
            rewards -= base_penalty

        # 3. Capture rewards (priest-focused) with high-clock multiplier
        captured_colors, captured_types = cache_manager.occupancy_cache.batch_get_attributes(to_coords)

        clock_multiplier = 1.0
        if halfmove_clock > 70:
            clock_multiplier = 1.5
        if halfmove_clock > 90:
            clock_multiplier = 2.5
        

        for i in range(len(moves)):
            if captured_colors[i] == self.color.opposite().value:
                if captured_types[i] == PieceType.PRIEST.value:
                    rewards[i] += 3.0 * clock_multiplier # INCREASED REWARD
                else:
                    rewards[i] += 0.2 * clock_multiplier

        # 4. Approach enemy priests
        enemy_priest_positions = self._get_enemy_priest_positions(cache_manager)
        if len(enemy_priest_positions) > 0:
            piece_types = cache_manager.occupancy_cache.batch_get_attributes(from_coords)[1]

            approach_rewards = _compute_distance_rewards_vectorized(
                from_coords, to_coords, enemy_priest_positions,
                piece_types, 0.1
            )
            rewards += approach_rewards

        # 5. Conditional Check Reward
        enemy_color = self.color.opposite()
        if not cache_manager.occupancy_cache.has_priest(enemy_color):
            enemy_king_pos = cache_manager.occupancy_cache.find_king(enemy_color)
            if enemy_king_pos is not None:
                 check_rewards = _compute_check_potential_vectorized(
                    to_coords, from_types, enemy_king_pos, 1.5
                )
                 rewards += check_rewards

        return rewards

    def _get_enemy_priest_positions(self, cache_manager) -> np.ndarray:
        """Get all enemy priest positions as a single array."""
        enemy_color = self.color.opposite()
        all_positions = cache_manager.occupancy_cache.get_positions(enemy_color.value)

        if len(all_positions) == 0:
            return np.empty((0, 3), dtype=COORD_DTYPE)

        # Filter for priests
        priest_positions = []
        for coord in all_positions:
            piece = cache_manager.occupancy_cache.get(coord.reshape(1, 3))
            if piece and piece['piece_type'] == PieceType.PRIEST.value:
                priest_positions.append(coord)

        return np.array(priest_positions, dtype=COORD_DTYPE) if priest_positions else np.empty((0, 3), dtype=COORD_DTYPE)


# =============================================================================
# GRAPH AWARE OPPONENT
# =============================================================================

class GraphAwareOpponent(OpponentBase):
    def batch_reward(self, state: 'GameState', moves: np.ndarray) -> np.ndarray:
        """Considers piece coordination and local piece density."""
        cache_manager = state.cache_manager
        to_coords = moves[:, 3:6]
        from_coords = moves[:, :3]

        rewards = np.zeros(len(moves), dtype=FLOAT_DTYPE)

        # 1. Repetition penalty
        rewards += self.get_repetition_penalties_batch(moves)

        # 2. Improved halfmove clock penalty
        halfmove_clock = getattr(state, 'halfmove_clock', 0)
        if halfmove_clock > 50:
            base_penalty = (halfmove_clock - 50) * 0.3
            if halfmove_clock > 70:
                base_penalty += (halfmove_clock - 70) * 0.2
            if halfmove_clock > 90:
                base_penalty *= 1.5
            rewards -= base_penalty

        # 3. Piece coordination bonus
        from_types = cache_manager.occupancy_cache.batch_get_attributes(from_coords)[1]

        for i in range(len(moves)):
            if from_types[i] == 0:
                continue

            # Count allies near destination
            allies_near_new = 0
            to_coord = to_coords[i]

            # Check 6 neighboring squares
            for dx, dy, dz in ((-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)):
                neighbor = to_coord + np.array([dx, dy, dz], dtype=COORD_DTYPE)
                if (0 <= neighbor[0] < 9 and 0 <= neighbor[1] < 9 and 0 <= neighbor[2] < 9):
                    neighbor_piece = cache_manager.occupancy_cache.get(neighbor.reshape(1, 3))
                    if neighbor_piece and neighbor_piece['color'] == self.color.value:
                        allies_near_new += 1

            if allies_near_new > 1:
                rewards[i] += 0.1 * (allies_near_new - 1)

        # 4. Capture rewards with high-clock multiplier
        captured_colors, captured_types = cache_manager.occupancy_cache.batch_get_attributes(to_coords)

        capture_rewards = _compute_capture_rewards_vectorized(
            to_coords, captured_colors, captured_types, self.color.value,
            priest_bonus=1.0, queen_rook_bonus=0.0
        )
        if halfmove_clock > 70:
            capture_rewards *= 1.5
        if halfmove_clock > 90:
            capture_rewards *= (2.5 / 1.5)
        rewards += capture_rewards

        # 5. Conditional Check Reward
        enemy_color = self.color.opposite()
        if not cache_manager.occupancy_cache.has_priest(enemy_color):
            enemy_king_pos = cache_manager.occupancy_cache.find_king(enemy_color)
            if enemy_king_pos is not None:
                 check_rewards = _compute_check_potential_vectorized(
                    to_coords, from_types, enemy_king_pos, 1.5
                )
                 rewards += check_rewards

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
AVAILABLE_OPPONENTS = ['adversarial', 'center_control', 'piece_capture', 'priest_hunter', 'graph_aware']

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
