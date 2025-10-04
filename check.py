from game3d.pieces.enums import PieceType

max_ptype = max(p.value for p in PieceType)          # 39
max_offset = (max_ptype + 1) // 2                    # 20  (or 21 if you keep the current split)
max_index  = max_ptype + max_offset                  # 39 + 21 = 60
N_TOTAL_PLANES = max_index + 2                       # 61 for pieces + 1 for side-to-move

print("max_ptype       =", max_ptype)
print("max_offset      =", max_offset)
print("max_index       =", max_index)
print("N_TOTAL_PLANES  =", N_TOTAL_PLANES)
