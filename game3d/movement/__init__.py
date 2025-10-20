"""Import all move generators to register them with the dispatcher."""
#game3d/movement/__init__.py
# Standard pieces
from game3d.pieces.pieces.pawn import pawn_move_dispatcher
from game3d.pieces.pieces.knight import knight_move_dispatcher
from game3d.pieces.pieces.bishop import bishop_move_dispatcher
from game3d.pieces.pieces.rook import rook_move_dispatcher
from game3d.pieces.pieces.queen import queen_move_dispatcher
from game3d.pieces.pieces.kinglike import king_move_dispatcher, priest_move_dispatcher
from game3d.pieces.pieces.trailblazer import trailblazer_move_dispatcher
from game3d.pieces.pieces.blackhole import blackhole_move_dispatcher
from game3d.pieces.pieces.whitehole import whitehole_move_dispatcher
from game3d.pieces.pieces.freezer import freezer_move_dispatcher
from game3d.pieces.pieces.speeder import speeder_move_dispatcher
from game3d.pieces.pieces.slower import slower_move_dispatcher
# Special pieces
from game3d.pieces.pieces.bigknights import knight32_move_dispatcher, knight31_move_dispatcher
from game3d.pieces.pieces.trigonalbishop import trigonal_bishop_move_dispatcher
from game3d.pieces.pieces.hive import hive_move_dispatcher
from game3d.pieces.pieces.orbiter import orbital_move_dispatcher
from game3d.pieces.pieces.nebula import nebula_move_dispatcher
from game3d.pieces.pieces.echo import echo_move_dispatcher
from game3d.pieces.pieces.panel import panel_move_dispatcher
from game3d.pieces.pieces.edgerook import edgerook_move_dispatcher
from game3d.pieces.pieces.xyqueen import xy_queen_move_dispatcher
from game3d.pieces.pieces.xzqueen import xz_queen_move_dispatcher
from game3d.pieces.pieces.yzqueen import yz_queen_move_dispatcher
from game3d.pieces.pieces.vectorslider import vectorslider_move_dispatcher

from game3d.pieces.pieces.spiral import spiral_move_dispatcher

from game3d.pieces.pieces.xzzigzag import xz_zigzag_move_dispatcher
from game3d.pieces.pieces.yzzigzag import yz_zigzag_move_dispatcher
from game3d.pieces.pieces.facecone import face_cone_move_dispatcher
from game3d.pieces.pieces.wall import wall_move_dispatcher
from game3d.pieces.pieces.archer import archer_move_dispatcher
from game3d.pieces.pieces.bomb import bomb_move_dispatcher
from game3d.pieces.pieces.geomancer import geomancer_move_dispatcher
from game3d.pieces.pieces.swapper import swapper_move_dispatcher
from game3d.pieces.pieces.armour import armour_move_dispatcher
from game3d.pieces.pieces.reflector import reflector_move_dispatcher
from game3d.pieces.pieces.mirror import mirror_move_dispatcher
from game3d.pieces.pieces.friendlytp import friendlytp_move_dispatcher
from game3d.pieces.pieces.infiltrator import infiltrator_move_dispatcher
