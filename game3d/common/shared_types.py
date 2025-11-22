"""
Centralized constants and types for 3D chess engine.
Single source of truth for all type definitions and constants.
"""

import numpy as np
from typing import Union, Tuple, Optional, Any, List
from enum import IntEnum
from dataclasses import dataclass

# Define enums directly to avoid circular imports
class Color(IntEnum):
    """Color enum for game pieces."""
    EMPTY = 0
    WHITE = 1
    BLACK = 2

    def opposite(self) -> 'Color':
        """Return the opposite color."""
        if self == Color.WHITE:
            return Color.BLACK
        elif self == Color.BLACK:
            return Color.WHITE
        return Color.EMPTY  # EMPTY has no opposite

    def is_white(self) -> bool:
        """Check if this color is white."""
        return self == Color.WHITE

    def is_black(self) -> bool:
        """Check if this color is black."""
        return self == Color.BLACK

    def is_empty(self) -> bool:
        """Check if this color is empty."""
        return self == Color.EMPTY

class PieceType(IntEnum):
    """Piece type enum for all 40 game pieces (0-40, with 0 as EMPTY placeholder)."""
    # Standard chess pieces (1-6)
    PAWN = 1
    KNIGHT = 2
    BISHOP = 3
    ROOK = 4
    QUEEN = 5
    KING = 6
    
    # Special 3D pieces (7-15)
    PRIEST = 7
    KNIGHT32 = 8  # 3,2 leaper
    KNIGHT31 = 9  # 3,1 leaper
    TRIGONALBISHOP = 10  # diagonal in XY, XZ, YZ planes
    HIVE = 11  # move every friendly Hive once per turn
    ORBITER = 12  # orbital movement
    NEBULA = 13  # area cloud effect
    ECHO = 14  # echo / copy
    PANEL = 15  # panel / wall segment
    
    # Plane-specific pieces (16-19)
    EDGEROOK = 16  # rook bounded to board edges
    XYQUEEN = 17  # queen in XY plane only
    XZQUEEN = 18  # queen in XZ plane only
    YZQUEEN = 19  # queen in YZ plane only
    
    # Advanced movement types (20-21)
    VECTORSLIDER = 20  # vector-slider
    CONESLIDER = 21  # cone-slider
    
    # Special effect pieces (22-26)
    MIRROR = 22  # mirror teleport
    FREEZER = 23  # freeze aura
    WALL = 24  # wall piece
    ARCHER = 25  # archery attack within 2-sphere
    BOMB = 26  # detonates on capture / self-move
    
    # Utility pieces (27-30)
    FRIENDLYTELEPORTER = 27  # teleports to friendly-adjacent empty
    ARMOUR = 28  # immune to pawn captures
    SPEEDER = 29  # +1 step buff
    SLOWER = 30  # -1 step debuff
    
    # Special terrain/movement (31-35)
    GEOMANCER = 31  # blocks squares for 5 plies
    SWAPPER = 32  # swaps with any friendly
    XZZIGZAG = 33  # zig-zag in XZ planes
    YZZIGZAG = 34  # zig-zag in YZ planes
    REFLECTOR = 35  # reflects off walls
    
    # Physics-based pieces (36-37)
    BLACKHOLE = 36  # sucks enemies 1 step closer
    WHITEHOLE = 37  # pushes enemies 1 step away
    
    # Advanced strategic pieces (38-40)
    INFILTRATOR = 38  # infiltration
    TRAILBLAZER = 39  # marks trail, counter removal
    SPIRAL = 40  # counter-clockwise spiral (radius 2)
    
    # Helper methods
    @classmethod
    def get_standard_pieces(cls) -> list['PieceType']:
        """Get list of standard chess pieces (1-6)."""
        return [cls.PAWN, cls.KNIGHT, cls.BISHOP, cls.ROOK, cls.QUEEN, cls.KING]
    
    @classmethod
    def get_special_3d_pieces(cls) -> list['PieceType']:
        """Get list of special 3D pieces (7-15)."""
        return [cls.PRIEST, cls.KNIGHT32, cls.KNIGHT31, cls.TRIGONALBISHOP, cls.HIVE, 
                cls.ORBITER, cls.NEBULA, cls.ECHO, cls.PANEL]
    
    @classmethod
    def get_plane_specific_pieces(cls) -> list['PieceType']:
        """Get list of plane-specific pieces (16-19)."""
        return [cls.EDGEROOK, cls.XYQUEEN, cls.XZQUEEN, cls.YZQUEEN]
    
    @classmethod
    def get_advanced_movement_pieces(cls) -> list['PieceType']:
        """Get list of advanced movement pieces (20-21)."""
        return [cls.VECTORSLIDER, cls.CONESLIDER]
    
    @classmethod
    def get_special_effect_pieces(cls) -> list['PieceType']:
        """Get list of special effect pieces (22-26)."""
        return [cls.MIRROR, cls.FREEZER, cls.WALL, cls.ARCHER, cls.BOMB]
    
    @classmethod
    def get_utility_pieces(cls) -> list['PieceType']:
        """Get list of utility pieces (27-30)."""
        return [cls.FRIENDLYTELEPORTER, cls.ARMOUR, cls.SPEEDER, cls.SLOWER]
    
    @classmethod
    def get_terrain_movement_pieces(cls) -> list['PieceType']:
        """Get list of terrain/movement pieces (31-35)."""
        return [cls.GEOMANCER, cls.SWAPPER, cls.XZZIGZAG, cls.YZZIGZAG, cls.REFLECTOR]
    
    @classmethod
    def get_physics_pieces(cls) -> list['PieceType']:
        """Get list of physics-based pieces (36-37)."""
        return [cls.BLACKHOLE, cls.WHITEHOLE]
    
    @classmethod
    def get_advanced_strategic_pieces(cls) -> list['PieceType']:
        """Get list of advanced strategic pieces (38-40)."""
        return [cls.INFILTRATOR, cls.TRAILBLAZER, cls.SPIRAL]
    
    @classmethod
    def get_all_piece_types(cls) -> list['PieceType']:
        """Get list of all piece types (1-40, excluding 0)."""
        return np.array([pt for pt in cls], dtype=np.int8)
    
    def is_piece_type(self, piece_type: 'PieceType') -> bool:
        """Check if this is a specific piece type."""
        return self == piece_type
    
    def is_standard_piece(self) -> bool:
        """Check if this is a standard chess piece (1-6)."""
        return 1 <= self <= 6
    
    def is_special_3d_piece(self) -> bool:
        """Check if this is a special 3D piece (7-15)."""
        return 7 <= self <= 15
    
    def is_plane_specific_piece(self) -> bool:
        """Check if this is a plane-specific piece (16-19)."""
        return 16 <= self <= 19
    
    def is_advanced_movement_piece(self) -> bool:
        """Check if this is an advanced movement piece (20-21)."""
        return 20 <= self <= 21
    
    def is_special_effect_piece(self) -> bool:
        """Check if this is a special effect piece (22-26)."""
        return 22 <= self <= 26
    
    def is_utility_piece(self) -> bool:
        """Check if this is a utility piece (27-30)."""
        return 27 <= self <= 30
    
    def is_terrain_movement_piece(self) -> bool:
        """Check if this is a terrain/movement piece (31-35)."""
        return 31 <= self <= 35
    
    def is_physics_piece(self) -> bool:
        """Check if this is a physics-based piece (36-37)."""
        return 36 <= self <= 37
    
    def is_advanced_strategic_piece(self) -> bool:
        """Check if this is an advanced strategic piece (38-40)."""
        return 38 <= self <= 40

class Result(IntEnum):
    """Game result enum."""
    ONGOING = 0
    WHITE_WIN = 1
    BLACK_WIN = 2
    DRAW = 3
    
    # Legacy aliases removed - use WHITE_WIN, BLACK_WIN directly

# Core data types
COORD_DTYPE = np.int16
BATCH_COORD_DTYPE = np.int16
COORD_OFFSET_DTYPE = np.int16
INDEX_DTYPE = np.int32
BOOL_DTYPE = np.bool_
FLOAT_DTYPE = np.float32
INT8_DTYPE = np.int8
NODE_TYPE_DTYPE = np.uint8

# Color and piece definitions
COLOR_DTYPE = np.uint8
PIECE_TYPE_DTYPE = np.int8
HASH_DTYPE = np.int64
POINTER_DTYPE = np.int64

# Type aliases
Coord = np.ndarray  # Shape (3,) with dtype COORD_DTYPE
CoordBatch = np.ndarray  # Shape (N, 3) with dtype BATCH_COORD_DTYPE
IndexArray = np.ndarray  # Shape (N,) with dtype INDEX_DTYPE
BoolArray = np.ndarray  # Shape (...) with dtype BOOL_DTYPE

# Input type hints
CoordLike = Union[Coord, List[int], Tuple[int, int, int]]

@dataclass
class Piece:
    """Simple piece container for structured arrays."""
    piece_type: int
    color: int
    has_moved: bool = False
    special_flags: int = 0

# Pawn-specific rank constants (standard chess values for 9x9x9)
PAWN_START_RANK_WHITE = 2
PAWN_START_RANK_BLACK = 6
PAWN_PROMOTION_RANK_WHITE = 8
PAWN_PROMOTION_RANK_BLACK = 0
PAWN_TWO_STEP_START_ROW_WHITE = 1
PAWN_TWO_STEP_START_ROW_BLACK = 7

# Legacy type aliases removed - use Coord, Color, PieceType directly

# Structured dtypes
PieceArrayDtype = np.dtype([
    ('piece_type', PIECE_TYPE_DTYPE),
    ('color', COLOR_DTYPE),
    ('has_moved', BOOL_DTYPE),
    ('special_flags', PIECE_TYPE_DTYPE)
])

BoardStateDtype = np.dtype([
    ('piece_type', PIECE_TYPE_DTYPE),
    ('color', COLOR_DTYPE),
    ('is_white_to_move', BOOL_DTYPE),
    ('castling_rights', PIECE_TYPE_DTYPE),
    ('en_passant', COORD_DTYPE)
])

# Board constants
SIZE = 9
BOARD_SIZE = SIZE  # Alias for SIZE
VOLUME = SIZE ** 3
SIZE_SQUARED = SIZE * SIZE
SIZE_MINUS_1 = SIZE - 1
MAX_COORD_VALUE = SIZE - 1
MIN_COORD_VALUE = 0
MAX_INDEX_VALUE = VOLUME - 1
MIN_INDEX_VALUE = 0
POLICY_DIM = VOLUME
# Move step constants
MOVE_STEPS_MIN = 1
MOVE_STEPS_MAX = 9

# Plane constants - calculated from enum
N_PIECE_TYPES = max(piece_type.value for piece_type in PieceType if piece_type.value > 0)  # Max piece type value (40)
N_COLOR_PLANES = 2
N_CURRENT_PLAYER_PLANES = 1  # Plane indicating current player
N_EFFECT_PLANES = 6  # Planes for game effects
N_TOTAL_PLANES = 2 * N_PIECE_TYPES + N_COLOR_PLANES + N_CURRENT_PLAYER_PLANES + N_EFFECT_PLANES  # All planes for training
N_PLANES_PER_SIDE = N_PIECE_TYPES + N_COLOR_PLANES

# Board slices
PIECE_SLICE = slice(0, 2 * N_PIECE_TYPES)
COLOR_SLICE = slice(2 * N_PIECE_TYPES, 2 * N_PIECE_TYPES + N_COLOR_PLANES)
CURRENT_SLICE = slice(2 * N_PIECE_TYPES + N_COLOR_PLANES, 2 * N_PIECE_TYPES + N_COLOR_PLANES + 1)
EFFECT_SLICE = slice(2 * N_PIECE_TYPES + N_COLOR_PLANES + 1, 2 * N_PIECE_TYPES + N_COLOR_PLANES + 1 + 6)
N_CHANNELS = N_TOTAL_PLANES

# Vectorization threshold
VECTORIZATION_THRESHOLD = 100

# Optimization constants
OCCUPANCY_THRESHOLD = 0.5
COLOR_INDEX_OFFSET = 1
EXPANSION_FACTOR = 1.5

# Precomputed offsets
def _generate_radius_offsets_vectorized(radius: int) -> np.ndarray:
    """Generate Chebyshev distance offsets for movement calculations."""
    coords = np.mgrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
    coords = coords.reshape(3, -1).T
    
    mask = np.max(np.abs(coords), axis=1) <= radius
    center_idx = coords.shape[0] // 2
    mask[center_idx] = False
    
    return coords[mask].astype(COORD_DTYPE, order='C')

# Precomputed movement offsets for common radii
RADIUS_1_OFFSETS = _generate_radius_offsets_vectorized(1)
RADIUS_2_OFFSETS = _generate_radius_offsets_vectorized(2)
RADIUS_3_OFFSETS = _generate_radius_offsets_vectorized(3)



# Color constants
COLOR_EMPTY = np.array([0], dtype=COLOR_DTYPE)
COLOR_WHITE = np.array([1], dtype=COLOR_DTYPE)
COLOR_BLACK = np.array([2], dtype=COLOR_DTYPE)

# Color mappings - consolidated (identity mapping for Color enum)
COLOR_VALUE_TO_CODE_MAP = {0: 0, 1: 1, 2: 2}
CODE_TO_COLOR_VALUE_MAP = COLOR_VALUE_TO_CODE_MAP  # Same mapping, just alias

# Array shapes
BOARD_SHAPE_3D = (SIZE, SIZE, SIZE)
BOARD_SHAPE_FLAT = (VOLUME,)
BOARD_SHAPE_4D = (N_TOTAL_PLANES, SIZE, SIZE, SIZE)
VOLUME_SQUARED = VOLUME * VOLUME

# Common value constants
ONE_VALUE = 1.0
ZERO_VALUE = 0
NEGATIVE_ONE_VALUE = -1.0

# Cache and size constants
EMPTY_CACHE_SIZE = 0
DEFAULT_CACHE_SIZE = 1
MIN_CACHE_SIZE = 1
MAX_CACHE_SIZE = 10000

# Batch constants
MAX_BATCH_SIZE = 10000
DEFAULT_BATCH_SIZE = 1000
BATCH_SIZE_SMALL = 100
BATCH_SIZE_MEDIUM = 500
BATCH_THRESHOLD = 50

# Legacy type aliases removed - use direct numpy types and Union annotations
CoordBatchLike = Union[CoordBatch, List[CoordLike], np.ndarray]
IndexLike = Union[IndexArray, list, int]

# Structured dtypes
COORD_STRUCT_DTYPE = np.dtype([
    ('x', COORD_DTYPE),
    ('y', COORD_DTYPE),
    ('z', COORD_DTYPE)
])

PieceDtype = np.dtype([('piece_type', PIECE_TYPE_DTYPE), ('color', COLOR_DTYPE)])

# Bit manipulation constants for piece encoding
PIECE_COLOR_MASK = 0xF
PIECE_TYPE_MASK = 0xF0
COLOR_SHIFT_BITS = 2
PIECE_TYPE_SHIFT_BITS = 4

# Move bitfield constants
TO_IDX_SHIFT = 10
FLAGS_SHIFT = 20
TO_IDX_MASK = 0x3FF
FLAGS_MASK = 0xFFF
BYTE_MASK = 0xFF
CAPTURE_SHIFT = 20

# Array dimension constants  
N_DIMENSIONS = 3
COORD_AXIS = np.arange(3)  # [0, 1, 2] for x, y, z axes
DEFAULT_PLY = 0

# Game rule and performance constants
# Performance optimization constants
MIN_CALLS = 1  # Minimum calls for division protection
MS_TO_S = 1000  # Milliseconds to seconds conversion
CPU_COUNT_FALLBACK = 6  # Default CPU count for parallel processing
MIN_WORKERS = 1  # Minimum number of workers for parallel processing

# Game rule constants
FIFTY_MOVE_RULE = 150  # Half-moves for 75-move rule
REPETITION_LIMIT = 5  # Maximum position repetitions before draw
INSUFFICIENT_MATERIAL_THRESHOLD = 5  # Threshold for insufficient material detection

# Test and benchmark constants
TEST_SIZE_LARGE = 1000  # Large test dataset size
TEST_SIZE_MEDIUM = 500  # Medium test dataset size
TEST_SIZE_SMALL = 100  # Small test dataset size
LOOP_ITERATIONS = 100  # Standard loop iteration count

# Mathematical and array constants
DAMAGE_THRESHOLD = 10  # Threshold for damage calculations
COORDINATE_DIMS = (10, 3)  # Standard coordinate array dimensions
BATCH_DIMS = (50, 3)  # Standard batch processing dimensions
MAX_MOVE_EFFECT_BATCH_SIZE = 10  # Maximum batch size for move effects
ARRAY_COPY_THRESHOLD = 1  # Threshold for array copying operations
MEMORY_ALIGNMENT = 0  # Memory alignment for array operations

# Movement and piece-specific constants
MAX_STEPS_SLIDER = 8  # Maximum steps for sliding pieces (bishop, rook, queen)
MAX_STEPS_KING = 1    # Maximum steps for king-like movement
BLACKHOLE_PULL_RADIUS = 2  # Black hole pull radius
WHITEHOLE_PUSH_RADIUS = 2  # White hole push radius
ORBITER_MANHATTAN_DISTANCE = 4  # Orbiter movement distance
MAX_SPIRAL_DISTANCE = 8  # Maximum distance for spiral movement
MAX_TRAILBLAZER_DISTANCE = 3  # Maximum distance for trailblazer
FREEZER_RANGE = 8  # Freezer effect range (matches SIZE-1)
SPEEDER_BUFF_STEPS = 1  # Additional steps provided by speeder
SLOWER_DEBUFF_STEPS = -1  # Steps reduced by slower
VECTORSLIDER_MAX_DISTANCE = 8  # Maximum distance for vector slider
XYQUEEN_SLIDER_DISTANCE = 8  # XY queen slider distance
XYQUEEN_HOP_DISTANCE = 1     # XY queen hop distance
XZQUEEN_SLIDER_DISTANCE = 8  # XZ queen slider distance  
XZQUEEN_HOP_DISTANCE = 1     # XZ queen hop distance
YZ_SLIDER_MAX_STEPS = 8      # YZ queen slider steps
ZIGZAG_MAX_DISTANCE = 16     # Maximum distance for zigzag pieces

# Aura and effect constants  
GEOMANCER_BLOCK_DURATION = 5  # Geomancer blocks squares for 5 plies
AREA_EFFECT_RADIUS_1 = 1     # Standard area effect radius
AREA_EFFECT_RADIUS_2 = 2     # Extended area effect radius
MIN_3D = 0                   # Minimum 3D coordinate
MAX_3D = 8                   # Maximum 3D coordinate (SIZE - 1)
MAX_HISTORY_SIZE = 100       # Maximum moves to keep in history for 50-move rule (circular buffer)

# Note: in_bounds_vectorized has been moved to coord_utils.py as the single source
# Use: from .coord_utils import in_bounds_vectorized

# Type aliases for backward compatibility
EMPTY = Color.EMPTY
WHITE = Color.WHITE  
BLACK = Color.BLACK

# Piece type aliases
PAWN = PieceType.PAWN
KNIGHT = PieceType.KNIGHT
BISHOP = PieceType.BISHOP
ROOK = PieceType.ROOK
QUEEN = PieceType.QUEEN
KING = PieceType.KING
PRIEST = PieceType.PRIEST
KNIGHT32 = PieceType.KNIGHT32
KNIGHT31 = PieceType.KNIGHT31
TRIGONALBISHOP = PieceType.TRIGONALBISHOP
HIVE = PieceType.HIVE
ORBITER = PieceType.ORBITER
NEBULA = PieceType.NEBULA
ECHO = PieceType.ECHO
PANEL = PieceType.PANEL
EDGEROOK = PieceType.EDGEROOK
XYQUEEN = PieceType.XYQUEEN
XZQUEEN = PieceType.XZQUEEN
YZQUEEN = PieceType.YZQUEEN
VECTORSLIDER = PieceType.VECTORSLIDER
CONESLIDER = PieceType.CONESLIDER
MIRROR = PieceType.MIRROR
FREEZER = PieceType.FREEZER
WALL = PieceType.WALL
ARCHER = PieceType.ARCHER
BOMB = PieceType.BOMB
FRIENDLYTELEPORTER = PieceType.FRIENDLYTELEPORTER
ARMOUR = PieceType.ARMOUR
SPEEDER = PieceType.SPEEDER
SLOWER = PieceType.SLOWER
GEOMANCER = PieceType.GEOMANCER
SWAPPER = PieceType.SWAPPER
XZZIGZAG = PieceType.XZZIGZAG
YZZIGZAG = PieceType.YZZIGZAG
REFLECTOR = PieceType.REFLECTOR
BLACKHOLE = PieceType.BLACKHOLE
WHITEHOLE = PieceType.WHITEHOLE
INFILTRATOR = PieceType.INFILTRATOR
TRAILBLAZER = PieceType.TRAILBLAZER
SPIRAL = PieceType.SPIRAL

# Additional constants
MOVE_STEPS_MIN = 1
MOVE_STEPS_MAX = 9

# Legacy color array aliases - use COLOR_EMPTY, COLOR_WHITE, COLOR_BLACK directly
COLOR_ARRAY_EMPTY = COLOR_EMPTY
COLOR_ARRAY_WHITE = COLOR_WHITE  
COLOR_ARRAY_BLACK = COLOR_BLACK

# Move dtype for consolidated operations
MOVE_DTYPE = np.dtype([
    ('from_x', COORD_DTYPE),
    ('from_y', COORD_DTYPE), 
    ('from_z', COORD_DTYPE),
    ('to_x', COORD_DTYPE),
    ('to_y', COORD_DTYPE),
    ('to_z', COORD_DTYPE),
    ('is_capture', BOOL_DTYPE),
    ('move_flags', INDEX_DTYPE)
])

# Batch operation flags for consolidated operations
BATCH_OPERATION_FLAGS = {
    'DEFAULT': 0,
    'CAPTURE': 1,
    'CHECK': 2,
    'MATE': 4,
    'SPECIAL': 8,
    'FROZEN': 16,
    'GEOMANCY': 32,    # Geomancy effect move
    'ARCHERY': 64      # Archery attack move
}

def compute_board_index(x: int, y: int, z: int) -> int:
    """Convert 3D coordinates to flat board index using SIZE and SIZE_SQUARED."""
    return x + SIZE * y + SIZE_SQUARED * z

# Legacy Piece structure
Piece = np.dtype([('piece_type', PIECE_TYPE_DTYPE), ('color', COLOR_DTYPE)])

# Utility functions
def get_empty_coord() -> Coord:
    """Return an empty 3D coordinate array."""
    return np.array([0, 0, 0], dtype=COORD_DTYPE)

def get_empty_coord_batch(n_coords: int = 0) -> CoordBatch:
    """Return an empty coordinate batch array."""
    return np.zeros((n_coords, 3), dtype=BATCH_COORD_DTYPE)

def get_empty_coord_2d_batch(n_coords: int = 0) -> np.ndarray:
    """Return an empty 2D coordinate batch array (for export functions)."""
    return np.zeros((n_coords, 2), dtype=INDEX_DTYPE)

def get_empty_2d_array(n_rows: int = 0, n_cols: int = 2) -> np.ndarray:
    """Return an empty 2D array with specified dimensions."""
    return np.zeros((n_rows, n_cols), dtype=INDEX_DTYPE)

def get_empty_move_batch(n_moves: int = 0) -> np.ndarray:
    """Return an empty move batch array (6D: from_x, from_y, from_z, to_x, to_y, to_z)."""
    return np.zeros((n_moves, 6), dtype=COORD_DTYPE)

def get_empty_bool_array(shape: Union[int, Tuple[int, ...]] = ()) -> BoolArray:
    """Return an empty boolean array."""
    if isinstance(shape, int):
        shape = (shape,)
    return np.zeros(shape, dtype=BOOL_DTYPE)

def get_empty_index_array(length: int = 0) -> IndexArray:
    """Return an empty index array."""
    return np.zeros(length, dtype=INDEX_DTYPE)

def create_coord_struct(coords: CoordBatch) -> np.ndarray:
    """Create a structured array from coordinate batch."""
    if coords.shape[1] != 3:
        raise ValueError(f"Invalid coordinate dimension: {coords.shape[1]}, expected 3")

    return np.core.records.fromarrays(
        [coords[:, 0], coords[:, 1], coords[:, 2]],
        dtype=COORD_STRUCT_DTYPE
    )

def extract_coords_struct(struct_array: np.ndarray) -> CoordBatch:
    """Extract coordinates from structured array."""
    if struct_array.dtype == COORD_STRUCT_DTYPE:
        return np.column_stack((struct_array['x'], struct_array['y'], struct_array['z']))
    else:
        raise ValueError(f"Invalid struct dtype: {struct_array.dtype}")

# Error formatters
def format_coord_error(coord: Any, expected_format: str = "3D coordinate") -> str:
    """Format coordinate error messages consistently."""
    return f"Invalid {expected_format}: {coord}. Expected array-like with 3 integer values"

def format_batch_error(coords: Any, expected_shape: str = "(N, 3)") -> str:
    """Format coordinate batch error messages consistently."""
    return f"Invalid coordinate batch: {coords}. Expected shape {expected_shape} with integer values"

def format_bounds_error(coords: CoordBatch) -> str:
    """Format bounds error messages consistently."""
    return f"3D coordinates {coords} are outside bounds [0, {SIZE-1}]"

# Move flags for bit manipulation
MOVE_FLAGS = {
    'CAPTURE': 1,
    'BUFFED': 2,
    'DEBUFFED': 4,
    'GEOMANCY': 8,
    'ARCHERY': 16,
    'FROZEN' : 32,
    'PROMOTION': 64,
    'SELF_DETONATE': 128,
}

# Module exports
__all__ = [
    # Core types
    'Coord', 'CoordBatch', 'BoolArray', 'IndexArray',
    'CoordLike', 'CoordBatchLike', 'IndexLike',

    # Data types
    'COORD_DTYPE', 'BATCH_COORD_DTYPE', 'COORD_OFFSET_DTYPE', 'INDEX_DTYPE',
    'BOOL_DTYPE', 'FLOAT_DTYPE', 'COLOR_DTYPE', 'PIECE_TYPE_DTYPE',
    'HASH_DTYPE', 'POINTER_DTYPE', 'COORD_STRUCT_DTYPE', 'PieceDtype', 'NODE_TYPE_DTYPE',

    # Board constants
    'SIZE', 'VOLUME', 'SIZE_SQUARED', 'SIZE_MINUS_1', 'VOLUME_SQUARED',
    'BOARD_SHAPE_3D', 'BOARD_SHAPE_FLAT',
    'MAX_COORD_VALUE', 'MIN_COORD_VALUE', 'MAX_INDEX_VALUE', 'MIN_INDEX_VALUE',
    'MAX_BATCH_SIZE', 'DEFAULT_BATCH_SIZE',
    
    # Plane constants
    'N_PIECE_TYPES', 'N_COLOR_PLANES', 'N_CURRENT_PLAYER_PLANES', 'N_EFFECT_PLANES', 
    'N_TOTAL_PLANES', 'N_PLANES_PER_SIDE',
    'PIECE_SLICE', 'COLOR_SLICE', 'CURRENT_SLICE', 'EFFECT_SLICE',
    'N_CHANNELS',
    
    # Precomputed constants
    'RADIUS_1_OFFSETS', 'RADIUS_2_OFFSETS', 'RADIUS_3_OFFSETS',
    'VECTORIZATION_THRESHOLD',
    
    # Color constants
    'COLOR_EMPTY', 'COLOR_WHITE', 'COLOR_BLACK',
    'COLOR_VALUE_TO_CODE_MAP', 'CODE_TO_COLOR_VALUE_MAP',

    # Empty array handlers
    'get_empty_coord', 'get_empty_coord_batch', 'get_empty_coord_2d_batch', 'get_empty_move_batch',
    'get_empty_bool_array', 'get_empty_index_array', 'get_empty_2d_array',

    # Utility functions
    'create_coord_struct', 'extract_coords_struct',

    # Error formatters
    'format_coord_error', 'format_batch_error', 'format_bounds_error',

    # Enums
    'Color', 'PieceType', 'Result',
    
    # Bit manipulation constants
    'PIECE_COLOR_MASK', 'PIECE_TYPE_MASK', 'COLOR_SHIFT_BITS', 'PIECE_TYPE_SHIFT_BITS',
    'TO_IDX_SHIFT', 'FLAGS_SHIFT', 'TO_IDX_MASK', 'FLAGS_MASK', 'BYTE_MASK', 'CAPTURE_SHIFT',
    
    # Array dimension constants
    'N_DIMENSIONS', 'COORD_AXIS', 'DEFAULT_PLY',
    
    # Game rule and performance constants
    'MIN_CALLS', 'MS_TO_S', 'CPU_COUNT_FALLBACK', 'MIN_WORKERS',
    'FIFTY_MOVE_RULE', 'REPETITION_LIMIT', 'INSUFFICIENT_MATERIAL_THRESHOLD',
    'TEST_SIZE_LARGE', 'TEST_SIZE_MEDIUM', 'TEST_SIZE_SMALL', 'LOOP_ITERATIONS',
    'DAMAGE_THRESHOLD', 'COORDINATE_DIMS', 'BATCH_DIMS', 'MAX_MOVE_EFFECT_BATCH_SIZE',
    'ARRAY_COPY_THRESHOLD', 'MEMORY_ALIGNMENT',
    
    # Value and cache constants
    'ONE_VALUE', 'ZERO_VALUE', 'NEGATIVE_ONE_VALUE', 'EMPTY_CACHE_SIZE', 
    'DEFAULT_CACHE_SIZE', 'MIN_CACHE_SIZE', 'MAX_CACHE_SIZE', 'BOARD_SHAPE_4D',
    
    # Pawn constants
    'PAWN_START_RANK_WHITE', 'PAWN_START_RANK_BLACK', 'PAWN_PROMOTION_RANK_WHITE', 
    'PAWN_PROMOTION_RANK_BLACK', 'PAWN_TWO_STEP_START_ROW_WHITE', 'PAWN_TWO_STEP_START_ROW_BLACK',
    
    # Movement and piece-specific constants
    'MAX_STEPS_SLIDER', 'MAX_STEPS_KING', 'BLACKHOLE_PULL_RADIUS', 'WHITEHOLE_PUSH_RADIUS',
    'ORBITER_MANHATTAN_DISTANCE', 'MAX_SPIRAL_DISTANCE', 'MAX_TRAILBLAZER_DISTANCE',
    'FREEZER_RANGE', 'SPEEDER_BUFF_STEPS', 'SLOWER_DEBUFF_STEPS', 'VECTORSLIDER_MAX_DISTANCE',
    'XYQUEEN_SLIDER_DISTANCE', 'XYQUEEN_HOP_DISTANCE', 'XZQUEEN_SLIDER_DISTANCE', 'XZQUEEN_HOP_DISTANCE',
    'YZ_SLIDER_MAX_STEPS', 'ZIGZAG_MAX_DISTANCE', 'GEOMANCER_BLOCK_DURATION',
    'AREA_EFFECT_RADIUS_1', 'AREA_EFFECT_RADIUS_2', 'MIN_3D', 'MAX_3D', 'MAX_HISTORY_SIZE',
    
    # Structured dtypes
    'PieceArrayDtype', 'BoardStateDtype', 'Piece', 'EMPTY', 'WHITE', 'BLACK',
    'COLOR_ARRAY_EMPTY', 'COLOR_ARRAY_WHITE', 'COLOR_ARRAY_BLACK',
    
    # Legacy aliases for backward compatibility
    'PAWN', 'KNIGHT', 'BISHOP', 'ROOK', 'QUEEN', 'KING',
    'PRIEST', 'KNIGHT32', 'KNIGHT31', 'TRIGONALBISHOP', 'HIVE', 'ORBITER', 'NEBULA',
    'ECHO', 'PANEL', 'EDGEROOK', 'XYQUEEN', 'XZQUEEN', 'YZQUEEN', 'VECTORSLIDER',
    'CONESLIDER', 'MIRROR', 'FREEZER', 'WALL', 'ARCHER', 'BOMB', 'FRIENDLYTELEPORTER',
    'ARMOUR', 'SPEEDER', 'SLOWER', 'GEOMANCER', 'SWAPPER', 'XZZIGZAG', 'YZZIGZAG',
    'REFLECTOR', 'BLACKHOLE', 'WHITEHOLE', 'INFILTRATOR', 'TRAILBLAZER', 'SPIRAL',
    
    # Additional constants and utilities
    'MOVE_STEPS_MIN', 'MOVE_STEPS_MAX', 'MOVE_DTYPE', 'BATCH_OPERATION_FLAGS', 'MOVE_FLAGS',
    'compute_board_index',
    
    # Optimization constants
    'OCCUPANCY_THRESHOLD', 'COLOR_INDEX_OFFSET', 'EXPANSION_FACTOR',
    
    # Vectorized utilities
    # Note: in_bounds_vectorized has been moved to coord_utils.py
]
