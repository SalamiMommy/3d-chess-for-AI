# debug_moves.py
from game3d.pieces.enums import PieceType
from game3d.movement.registry import get_dispatcher
from game3d.game.gamestate import GameState   # adjust import if your path differs

print('--- dispatcher lookup ---')
for pt in (PieceType.REFLECTOR, PieceType.FRIENDLYTELEPORTER):
    print(pt.name, ':', get_dispatcher(pt))

print('--- live generation test ---')
# build the smallest 9×9×9 board that owns the two pieces
state = GameState.starting_position()   # or however you obtain a GameState

for x, y, z, pt in [
    (0, 0, 8, PieceType.REFLECTOR),
    (1, 2, 8, PieceType.FRIENDLYTELEPORTER),
]:
    print(f'\nGenerating {pt.name} at {(x,y,z)}')
    disp = get_dispatcher(pt)
    if disp is None:
        print('  ❌ dispatcher is None')
        continue
    try:
        moves = disp(state, x, y, z)
        print('  ✅ returned', type(moves), 'with', len(moves), 'items')
        if moves:
            print('     first move:', moves[0])
    except Exception as e:
        import traceback
        print('  ❌ exception:', e)
        traceback.print_exc()
