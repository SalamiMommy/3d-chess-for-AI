# Import all piece modules to trigger @register decorator execution
# This ensures all piece dispatchers are registered with the movement generator

# Standard chess pieces
from game3d.pieces.pieces.pawn import *
from game3d.pieces.pieces.knight import *
from game3d.pieces.pieces.bishop import *
from game3d.pieces.pieces.rook import *
from game3d.pieces.pieces.queen import *
from game3d.pieces.pieces.king import *

# Special 3D pieces
from game3d.pieces.pieces.kinglike import *
from game3d.pieces.pieces.bigknights import *
from game3d.pieces.pieces.trigonalbishop import *
from game3d.pieces.pieces.hive import *
from game3d.pieces.pieces.orbiter import *
from game3d.pieces.pieces.nebula import *
from game3d.pieces.pieces.echo import *
from game3d.pieces.pieces.panel import *

# Plane-specific pieces
from game3d.pieces.pieces.edgerook import *
from game3d.pieces.pieces.xyqueen import *
from game3d.pieces.pieces.xzqueen import *
from game3d.pieces.pieces.yzqueen import *

# Advanced movement types
from game3d.pieces.pieces.vectorslider import *

# Special effect pieces
from game3d.pieces.pieces.mirror import *
from game3d.pieces.pieces.freezer import *
from game3d.pieces.pieces.wall import *
from game3d.pieces.pieces.archer import *
from game3d.pieces.pieces.bomb import *

# Utility pieces
from game3d.pieces.pieces.friendlytp import *
from game3d.pieces.pieces.armour import *
from game3d.pieces.pieces.speeder import *
from game3d.pieces.pieces.slower import *

# Special terrain/movement
from game3d.pieces.pieces.geomancer import *
from game3d.pieces.pieces.swapper import *
from game3d.pieces.pieces.xzzigzag import *
from game3d.pieces.pieces.yzzigzag import *
from game3d.pieces.pieces.reflector import *

# Physics-based pieces
from game3d.pieces.pieces.blackhole import *
from game3d.pieces.pieces.whitehole import *

# Advanced strategic pieces
from game3d.pieces.pieces.infiltrator import *
from game3d.pieces.pieces.trailblazer import *
from game3d.pieces.pieces.spiral import *

# Additional pieces
from game3d.pieces.pieces.facecone import *
