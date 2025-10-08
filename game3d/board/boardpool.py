# game3d/board/boardpool.py
"""
Thread-safe Board object pool (LRU) with automatic cache-manager eviction.
"""

from __future__ import annotations
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING

from game3d.board.board import Board

if TYPE_CHECKING:
    from game3d.board.board import Board

# ----------------------------------------------------------
# Pool configuration
# ----------------------------------------------------------
_MAX_POOL_SIZE: int = 4          # tune if you run many workers

# ----------------------------------------------------------
# Shared pool state
# ----------------------------------------------------------
_BOARD_POOL: OrderedDict[int, Board] = OrderedDict()
_POOL_LOCK  = threading.Lock()

# ----------------------------------------------------------
# Cache-manager registry eviction helper
# ----------------------------------------------------------
def _evict_cache_manager(board: Board) -> None:
    """
    Remove the board's cache manager from the global weak-registry so it can
    be garbage-collected once no other references exist.
    """
    # Import here to avoid circular imports on start-up
    from game3d.cache.manager import _cache_instance_registry
    _cache_instance_registry.pop(id(board), None)   # ignore if absent


# ----------------------------------------------------------
# Public API
# ----------------------------------------------------------
def get_pooled_board() -> Board:
    """
    Return a board that is already allocated and reset to start position.
    Thread-safe, LRU eviction when pool is full.
    """
    with _POOL_LOCK:
        if _BOARD_POOL:                       # pop oldest
            board = _BOARD_POOL.popitem(last=False)[1]
            board.reset_to_start_position()
            return board

    # Pool empty → create new
    return Board.startpos()


def return_board_to_pool(board: Board) -> None:
    """
    Give a board back to the pool when the game is **fully finished**.
    Also evicts its cache manager so a future game does not accidentally
    resurrect stale objects.
    """
    if board is None:
        return

    _evict_cache_manager(board)          # allow GC of huge cache graphs

    with _POOL_LOCK:
        if len(_BOARD_POOL) < _MAX_POOL_SIZE:
            _BOARD_POOL[id(board)] = board
        # else: let it die naturally
