"""Import all move generators to register them with the dispatcher."""
#game3d/movement/__init__.py
# Standard pieces
from game3d.movement.pieces.pawn import pawn_move_dispatcher
from game3d.movement.pieces.knight import knight_move_dispatcher
from game3d.movement.pieces.bishop import bishop_move_dispatcher
from game3d.movement.pieces.rook import rook_move_dispatcher, trailblazer_move_dispatcher
from game3d.movement.pieces.queen import queen_move_dispatcher
from game3d.movement.pieces.kinglike import king_move_dispatcher, whitehole_move_dispatcher, speeder_move_dispatcher, slower_move_dispatcher, priest_move_dispatcher, king_move_dispatcher, freezer_move_dispatcher, blackhole_move_dispatcher

# Special pieces
from game3d.movement.pieces.bigknights import knight32_move_dispatcher, knight31_move_dispatcher
from game3d.movement.pieces.trigonalbishop import trigonal_bishop_move_dispatcher
from game3d.movement.pieces.hive import hive_move_dispatcher
from game3d.movement.pieces.orbiter import orbital_move_dispatcher
from game3d.movement.pieces.nebula import nebula_move_dispatcher
from game3d.movement.pieces.echo import echo_move_dispatcher
from game3d.movement.pieces.panel import panel_move_dispatcher
from game3d.movement.pieces.edgerook import edgerook_move_dispatcher
from game3d.movement.pieces.xyqueen import xy_queen_move_dispatcher
from game3d.movement.pieces.xzqueen import xz_queen_move_dispatcher
from game3d.movement.pieces.yzqueen import yz_queen_move_dispatcher
from game3d.movement.pieces.vectorslider import vectorslider_move_dispatcher

from game3d.movement.pieces.spiral import spiral_move_dispatcher

from game3d.movement.pieces.xzzigzag import xz_zigzag_move_dispatcher
from game3d.movement.pieces.yzzigzag import yz_zigzag_move_dispatcher
from game3d.movement.pieces.facecone import face_cone_move_dispatcher
from game3d.movement.pieces.wall import wall_move_dispatcher
from game3d.movement.pieces.archer import archer_move_dispatcher
from game3d.movement.pieces.bomb import bomb_move_dispatcher
from game3d.movement.pieces.geomancer import geomancer_move_dispatcher
from game3d.movement.pieces.swapper import swapper_move_dispatcher
from game3d.movement.pieces.armour import armour_move_dispatcher
from game3d.movement.pieces.reflector import reflector_move_dispatcher
from game3d.movement.pieces.mirror import mirror_move_dispatcher
from game3d.movement.pieces.friendlytp import friendlytp_move_dispatcher
from game3d.movement.pieces.infiltrator import infiltrator_move_dispatcher
