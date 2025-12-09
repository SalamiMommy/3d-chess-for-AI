
import numpy as np
import threading
from game3d.common.shared_types import SIZE, PieceType, Color

# Aura Providers
# SPEEDER=29, SLOWER=30, FREEZER=23, REFLECTOR=35
SPEEDER = 29
SLOWER = 30
FREEZER = 23
REFLECTOR = 35

def _apply_aura_delta_python(counts: np.ndarray, pos: np.ndarray, offsets: np.ndarray, value: int) -> None:
    """
    Apply delta (value) to counts map at pos + offsets.
    Python implementation (numpy vectorized) to avoid Numba call overhead for simple ops
    or if strict Numba compilation is tricky with 5D arrays context.
    Using simple loop with bounds check is fast enough for ~26-124 updates.
    """
    px, py, pz = pos
    
    # Broadcast addition
    targets = pos + offsets
    
    # Filter bounds
    # x: 0..SIZE-1
    mask = (targets[:, 0] >= 0) & (targets[:, 0] < SIZE) & \
           (targets[:, 1] >= 0) & (targets[:, 1] < SIZE) & \
           (targets[:, 2] >= 0) & (targets[:, 2] < SIZE)
           
    valid_targets = targets[mask]
    
    # Advanced indexing to update
    counts[valid_targets[:, 0], valid_targets[:, 1], valid_targets[:, 2]] += value

class AuraCache:
    """
    Maintains Aura maps (Buff, Debuff, Freeze) using increment/decrement reference counting.
    Allows O(N_offsets) updates when aura-providing pieces move.
    
    Maps:
    - Buff Map (Color 0/1): My pieces are buffed. (Source: Friendly Speeders/Reflectors)
    - Debuff Map (Color 0/1): My pieces are debuffed. (Source: Enemy Slowers)
    - Freeze Map (Color 0/1): My pieces are frozen. (Source: Enemy Freezers)
    """
    def __init__(self, board):
        self.board = board
        self._lock = threading.RLock()
        
        # Maps are int8 counts. >0 means active.
        # Structure: [Color 0/1][MapType 0=Buff, 1=Debuff, 2=Freeze]
        # Dimensions: (2, 3, SIZE, SIZE, SIZE)
        self.encroachment_maps = np.zeros((2, 3, SIZE, SIZE, SIZE), dtype=np.int8)
        
        self.initialized = False
        
        # Pre-load offsets
        from game3d.pieces.pieces.kinglike import KING_MOVEMENT_VECTORS
        from game3d.common.shared_types import RADIUS_2_OFFSETS
        self.radius_1_offsets = KING_MOVEMENT_VECTORS
        self.radius_2_offsets = RADIUS_2_OFFSETS

    def trigger_freeze(self, color, turn_number):
        """
        Legacy method for compatibility with turnmove.py.
        The new AuraCache handles freeze status purely incrementally via reference counting maps.
        Temporal/Duration logic (if any) is currently not handled by this cache, 
        or is implicit in the presence of the freezer.
        """
        pass

    def get_maps(self, active_color: int):
        """
        Get boolean (Buff, Debuff, Freeze) maps for the active color.
        Returns tuple of 3 boolean arrays each (SIZE, SIZE, SIZE).
        
        Meaning:
        - Buff: "I am buffed" (by friendly units)
        - Debuff: "I am debuffed" (by enemy units)
        - Freeze: "I am frozen" (by enemy units)
        """
        with self._lock:
            c_idx = 0 if active_color == Color.WHITE else 1
            
            # 0: Buff (My Friendly Speeders)
            buffs = self.encroachment_maps[c_idx, 0] > 0
            
            # 1: Debuff (Enemy Slowers)
            # encroachment_maps[c_idx, 1] tracks "My Debuff Status"
            # It is populated by Enemy Slowers projecting onto ME.
            debuffs = self.encroachment_maps[c_idx, 1] > 0
            
            # 2: Freeze (Enemy Freezers)
            # encroachment_maps[c_idx, 2] tracks "My Freeze Status"
            freezes = self.encroachment_maps[c_idx, 2] > 0
            
            return buffs, debuffs, freezes

    def update_piece(self, pos: np.ndarray, piece_type: int, color: int, remove: bool = False):
        """
        Incremental update for a single piece change.
        If remove=True, subtracts aura. If remove=False, adds aura.
        """
        if piece_type not in (SPEEDER, SLOWER, FREEZER, REFLECTOR):
            return
            
        val = -1 if remove else 1
        
        # Determine which maps to update based on piece type and color.
        # c_idx is the OWNER of the map (who is affected).
        
        my_c_idx = 0 if color == Color.WHITE else 1
        opp_c_idx = 1 - my_c_idx
        
        # 1. SPEEDER (Buff Friendly Radius 1)
        if piece_type == SPEEDER:
            # Affects MY Buff Map
            _apply_aura_delta_python(self.encroachment_maps[my_c_idx, 0], pos, self.radius_1_offsets, val)

        # 2. REFLECTOR (Buff Friendly Neighbors - Radius 1)
        elif piece_type == REFLECTOR:
            # Affects MY Buff Map
            _apply_aura_delta_python(self.encroachment_maps[my_c_idx, 0], pos, self.radius_1_offsets, val)

        # 3. SLOWER (Debuff Enemy Radius 2)
        elif piece_type == SLOWER:
            # Affects OPPONENT'S Debuff Map
            _apply_aura_delta_python(self.encroachment_maps[opp_c_idx, 1], pos, self.radius_2_offsets, val)
            
        # 4. FREEZER (Freeze Enemy Radius 2)
        elif piece_type == FREEZER:
            # Affects OPPONENT'S Freeze Map
            _apply_aura_delta_python(self.encroachment_maps[opp_c_idx, 2], pos, self.radius_2_offsets, val)

    def on_move(self, from_coord: np.ndarray, to_coord: np.ndarray, 
                piece_type: int, color: int, 
                captured_type: int = 0, captured_color: int = 0):
        """Handle piece move with optional capture."""
        with self._lock:
            # 1. Remove aura of moving piece from old pos
            self.update_piece(from_coord, piece_type, color, remove=True)
            
            # 2. Remove aura of captured piece (if any)
            if captured_type != 0:
                self.update_piece(to_coord, captured_type, captured_color, remove=True)
                
            # 3. Add aura of moving piece at new pos
            self.update_piece(to_coord, piece_type, color, remove=False)

    def rebuild_from_board(self, occupancy_cache):
        """Rebuild all maps from scratch using occupancy cache."""
        with self._lock:
            self.encroachment_maps.fill(0)
            
            # Use export_buffer_data to get all pieces efficiently
            # Returns: (occ_grid, ptype_grid, valid_coords, valid_types, valid_colors)
            _, _, coords, types, colors = occupancy_cache.export_buffer_data()
            
            n = coords.shape[0]
            if n == 0:
                self.initialized = True
                return

            # Initial build: Just add effects for all relevant pieces
            for i in range(n):
                ptype = types[i]
                if ptype in (SPEEDER, SLOWER, FREEZER, REFLECTOR):
                    self.update_piece(coords[i], ptype, colors[i], remove=False)
            
            self.initialized = True

    def batch_is_frozen(self, coords: np.ndarray, turn_number: int, color: int) -> np.ndarray:
        """
        Check if pieces at coords are frozen.
        Compatibility method for generator.py.
        """
        if coords.size == 0:
            return np.zeros(0, dtype=bool)
            
        # Helper to ensure we access the correct map
        # If 'color' is active, we check if THEY are frozen.
        color_idx = 0 if color == Color.WHITE else 1
        
        # Map 2 is Freeze
        # Dimensions: (2, 3, SIZE, SIZE, SIZE)
        freeze_counts = self.encroachment_maps[color_idx, 2]
        
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        
        # Direct lookup (counts > 0 means frozen)
        return freeze_counts[x, y, z] > 0
