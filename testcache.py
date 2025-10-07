from game3d.cache.manager import get_cache_manager
cm = get_cache_manager(board, Color.WHITE)
occ, pt = cm.piece_cache.export_arrays()
print(occ.dtype, occ.max(), pt.max())
