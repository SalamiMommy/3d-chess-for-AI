
import sys
import os

sys.path.append(os.getcwd())

print("Importing pieces...")
try:
    import game3d.pieces.pieces
    # Manually import all files to trigger registration and check imports
    from game3d.pieces.pieces import archer
    from game3d.pieces.pieces import armour
    from game3d.pieces.pieces import bigknights
    from game3d.pieces.pieces import bishop
    from game3d.pieces.pieces import blackhole
    from game3d.pieces.pieces import bomb
    from game3d.pieces.pieces import echo
    from game3d.pieces.pieces import edgerook
    from game3d.pieces.pieces import facecone
    from game3d.pieces.pieces import freezer
    from game3d.pieces.pieces import friendlytp
    from game3d.pieces.pieces import geomancer
    from game3d.pieces.pieces import hive
    from game3d.pieces.pieces import infiltrator
    from game3d.pieces.pieces import king
    from game3d.pieces.pieces import kinglike
    from game3d.pieces.pieces import knight
    from game3d.pieces.pieces import mirror
    from game3d.pieces.pieces import nebula
    from game3d.pieces.pieces import orbiter
    from game3d.pieces.pieces import panel
    from game3d.pieces.pieces import pawn
    from game3d.pieces.pieces import queen
    from game3d.pieces.pieces import reflector
    from game3d.pieces.pieces import rook
    from game3d.pieces.pieces import slower
    from game3d.pieces.pieces import speeder
    from game3d.pieces.pieces import spiral
    from game3d.pieces.pieces import swapper
    from game3d.pieces.pieces import trailblazer
    from game3d.pieces.pieces import trigonalbishop
    from game3d.pieces.pieces import vectorslider
    from game3d.pieces.pieces import wall
    from game3d.pieces.pieces import whitehole
    from game3d.pieces.pieces import xyqueen
    from game3d.pieces.pieces import xzqueen
    from game3d.pieces.pieces import xzzigzag
    from game3d.pieces.pieces import yzqueen
    from game3d.pieces.pieces import yzzigzag
    
    print("All pieces imported successfully.")
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)
