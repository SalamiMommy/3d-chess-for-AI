# game3d/cache/diagnostics.py
"""
Cache diagnostics and monitoring tools.
Use this to verify cache usage is correct.
"""

import weakref
import gc
from typing import Dict, List, Tuple
from dataclasses import dataclass
import threading

@dataclass
class CacheCreationEvent:
    """Record of a cache creation event."""
    instance_id: int
    thread_name: str
    stack_trace: List[str]
    timestamp: float
    board_id: int

class CacheDiagnostics:
    """Track and diagnose cache creation patterns."""

    def __init__(self):
        self.creation_events: List[CacheCreationEvent] = []
        self.active_caches: Dict[int, weakref.ref] = {}
        self.lock = threading.Lock()

    def record_creation(self, cache_instance, board, stack_depth=5):
        """Record a cache creation event."""
        import time
        import traceback

        with self.lock:
            event = CacheCreationEvent(
                instance_id=id(cache_instance),
                thread_name=threading.current_thread().name,
                stack_trace=traceback.format_stack()[-stack_depth:],
                timestamp=time.time(),
                board_id=id(board)
            )

            self.creation_events.append(event)
            self.active_caches[id(cache_instance)] = weakref.ref(cache_instance)

    def get_active_count(self) -> int:
        """Get number of currently active caches."""
        # Clean up dead references
        dead_refs = []
        for cache_id, ref in self.active_caches.items():
            if ref() is None:
                dead_refs.append(cache_id)

        for cache_id in dead_refs:
            del self.active_caches[cache_id]

        return len(self.active_caches)

    def get_creation_count(self) -> int:
        """Get total number of caches created."""
        return len(self.creation_events)

    def analyze_patterns(self) -> Dict[str, any]:
        """Analyze cache creation patterns."""
        # Group by board
        by_board = {}
        for event in self.creation_events:
            board_id = event.board_id
            if board_id not in by_board:
                by_board[board_id] = []
            by_board[board_id].append(event)

        # Find boards with multiple caches (BAD)
        problematic_boards = {
            board_id: events
            for board_id, events in by_board.items()
            if len(events) > 1
        }

        # Group by call stack (find common creation paths)
        by_stack = {}
        for event in self.creation_events:
            # Use last 2 frames as key
            key = tuple(event.stack_trace[-2:])
            if key not in by_stack:
                by_stack[key] = []
            by_stack[key].append(event)

        return {
            'total_created': len(self.creation_events),
            'currently_active': self.get_active_count(),
            'unique_boards': len(by_board),
            'problematic_boards': len(problematic_boards),
            'creation_paths': len(by_stack),
            'boards_with_multiple_caches': problematic_boards,
        }

    def print_report(self):
        """Print diagnostic report."""
        analysis = self.analyze_patterns()

        print("\n" + "="*80)
        print("CACHE DIAGNOSTICS REPORT")
        print("="*80)

        print(f"\nTotal caches created: {analysis['total_created']}")
        print(f"Currently active: {analysis['currently_active']}")
        print(f"Unique boards: {analysis['unique_boards']}")
        print(f"Boards with multiple caches: {analysis['problematic_boards']}")

        if analysis['problematic_boards'] > 0:
            print("\n⚠️  WARNING: Found boards with multiple caches!")
            print("This indicates cache creation is not being reused properly.\n")

            for board_id, events in analysis['boards_with_multiple_caches'].items():
                print(f"\nBoard {board_id} has {len(events)} caches:")
                for i, event in enumerate(events, 1):
                    print(f"\n  Cache #{i}:")
                    print(f"    Thread: {event.thread_name}")
                    print(f"    Created at:")
                    for line in event.stack_trace:
                        print(f"      {line.strip()}")
        else:
            print("\n✅ All boards have single cache instances (GOOD!)")

        print("\n" + "="*80)

    def reset(self):
        """Reset all tracking."""
        with self.lock:
            self.creation_events.clear()
            self.active_caches.clear()

# Global diagnostics instance
_diagnostics = CacheDiagnostics()

def record_cache_creation(cache_instance, board):
    """Record a cache creation (call from OptimizedCacheManager.__init__)."""
    _diagnostics.record_creation(cache_instance, board)

def get_diagnostics() -> CacheDiagnostics:
    """Get the global diagnostics instance."""
    return _diagnostics

def print_cache_report():
    """Print cache diagnostics report."""
    _diagnostics.print_report()

def get_cache_stats() -> Dict[str, int]:
    """Get cache statistics."""
    return {
        'total_created': _diagnostics.get_creation_count(),
        'currently_active': _diagnostics.get_active_count(),
    }


# ============================================================================
# INTEGRATION: Add to manager.py __init__
# ============================================================================

"""
Add to OptimizedCacheManager.__init__():

    def __init__(self, board: Board) -> None:
        # ... existing init code ...

        # Record cache creation for diagnostics
        from game3d.cache.diagnostics import record_cache_creation
        record_cache_creation(self, board)
"""


# ============================================================================
# USAGE IN TRAINING
# ============================================================================

"""
Usage in your training script:

from game3d.cache.diagnostics import print_cache_report, get_cache_stats

# At start of training
print("Starting training with cache diagnostics...")

# Generate games
for i in range(num_games):
    game = OptimizedGame3D()
    # ... play game ...

# After all games
print_cache_report()

# Expected output for 10 games:
# Total caches created: 10 (one per game)
# Currently active: 0-10 (depends on garbage collection)
# Unique boards: 10
# Boards with multiple caches: 0 ✅

# BAD output (indicates problem):
# Total caches created: 181 (too many!)
# Boards with multiple caches: 10 ⚠️
"""


# ============================================================================
# AUTOMATED TEST
# ============================================================================

def test_cache_usage():
    """
    Automated test to verify cache usage is correct.

    Returns:
        bool: True if cache usage is correct
    """
    from game3d.cache.diagnostics import get_diagnostics, _diagnostics
    from game3d.game3d import OptimizedGame3D

    # Reset diagnostics
    _diagnostics.reset()

    # Create games
    num_games = 5
    games = []

    for i in range(num_games):
        game = OptimizedGame3D()
        games.append(game)

        # Make some moves
        for _ in range(3):
            moves = game.state.legal_moves()
            if moves:
                game.state = game.state.make_move(moves[0])

    # Analyze
    analysis = _diagnostics.analyze_patterns()

    # Verify
    success = True

    # Check 1: Should create exactly num_games caches
    if analysis['total_created'] != num_games:
        print(f"❌ Created {analysis['total_created']} caches, expected {num_games}")
        success = False
    else:
        print(f"✅ Created exactly {num_games} caches")

    # Check 2: No board should have multiple caches
    if analysis['problematic_boards'] > 0:
        print(f"❌ {analysis['problematic_boards']} boards have multiple caches")
        success = False
    else:
        print(f"✅ No boards have multiple caches")

    if success:
        print("\n🎉 Cache usage is CORRECT!")
    else:
        print("\n⚠️  Cache usage has PROBLEMS!")
        _diagnostics.print_report()

    return success


if __name__ == "__main__":
    # Run test
    test_cache_usage()
