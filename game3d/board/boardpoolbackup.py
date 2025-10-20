# game3d/board/boardpool.py
"""
Thread-safe Board object pool (LRU) with automatic cache-manager eviction.
"""

from __future__ import annotations
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING
from functools import lru_cache

if TYPE_CHECKING:
    from game3d.board.board import Board

# ----------------------------------------------------------
# Pool configuration
# ----------------------------------------------------------
_MAX_POOL_SIZE: int = 4

# ----------------------------------------------------------
# Shared pool state
# ----------------------------------------------------------
_BOARD_POOL: OrderedDict[int, Board] = OrderedDict()
_POOL_LOCK = threading.Lock()

# ----------------------------------------------------------
# Cache-manager registry eviction helper
# ----------------------------------------------------------
def _evict_cache_manager(board: Board) -> None:
    from game3d.cache.manager import _cache_instance_registry
    _cache_instance_registry.pop(id(board), None)

# ----------------------------------------------------------
# Public API
# ----------------------------------------------------------
def get_pooled_board() -> Board:
    with _POOL_LOCK:
        if _BOARD_POOL:
            board = _BOARD_POOL.popitem(last=False)[1]
            board.init_startpos()  # Reset to start
            if board.cache_manager:
                board.cache_manager.rebuild(board)
            return board
    b = Board.startpos()
    return b

def return_board_to_pool(board: Board) -> None:
    if board is None:
        return

    _evict_cache_manager(board)

    with _POOL_LOCK:
        if len(_BOARD_POOL) < _MAX_POOL_SIZE:
            _BOARD_POOL[id(board)] = board
        if board.cache_manager is not None:
            board.cache_manager.rebuild(board)
