#!/usr/bin/env python3
"""
Standalone precomputation script for 3D chess movement patterns.
Run this file directly to generate and save precomputed movement patterns.
"""

import pickle
import os
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
from math import gcd
from ..cache.manager import OptimizedCacheManager
# Constants
SIZE = 9
CACHE_DIR = Path(__file__).parent / "precomputed_cache"
CACHE_DIR.mkdir(exist_ok=True)

def in_bounds(coord: Tuple[int, int, int]) -> bool:
    """Check if coordinate is within 9x9x9 board."""
    x, y, z = coord
    return 0 <= x < SIZE and 0 <= y < SIZE and 0 <= z < SIZE

def add_coords(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Add two coordinates."""
    return (c1[0] + c2[0], c1[1] + c2[1], c1[2] + c2[2])

class StandaloneMovementPrecomputer:
    """Precomputes movement patterns without any game dependencies."""

    def __init__(self):
        self.cache: Dict[str, Any] = {}

    def save_cache(self):
        """Save cache to disk."""
        cache_file = CACHE_DIR / "movement_patterns.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
        print(f"Cache saved to: {cache_file}")

    def precompute_all(self) -> Dict[str, Any]:
        """Precompute all movement patterns."""
        print("Precomputing all movement patterns...")

        # Sliding pieces
        self.cache['rook_directions'] = self._precompute_rook_directions()
        self.cache['bishop_directions'] = self._precompute_bishop_directions()
        self.cache['queen_directions'] = self._precompute_queen_directions()
        self.cache['vectorslider_directions'] = self._precompute_vectorslider_directions()
        self.cache['trigonalbishop_directions'] = self._precompute_trigonalbishop_directions()
        self.cache['edgerook_directions'] = self._precompute_edgerook_directions()
        self.cache['xyqueen_directions'] = self._precompute_xyqueen_directions()
        self.cache['xzqueen_directions'] = self._precompute_xzqueen_directions()
        self.cache['yzqueen_directions'] = self._precompute_yzqueen_directions()
        self.cache['cone_directions'] = self._precompute_cone_directions()

        # Jumping pieces
        self.cache['knight_offsets'] = self._precompute_knight_offsets()
        self.cache['knight31_offsets'] = self._precompute_knight31_offsets()
        self.cache['knight32_offsets'] = self._precompute_knight32_offsets()
        self.cache['king_offsets'] = self._precompute_king_offsets()

        # Zig-zag pieces
        self.cache['xzzigzag_patterns'] = self._precompute_xzzigzag_patterns()
        self.cache['yzzigzag_patterns'] = self._precompute_yzzigzag_patterns()

        # Special movement
        self.cache['mirror_targets'] = self._precompute_mirror_targets()
        self.cache['neighbor_directions'] = self._precompute_neighbor_directions()

        # Precomputed rays for all positions
        self.cache['sliding_rays'] = self._precompute_all_sliding_rays()

        self.save_cache()
        return self.cache

    # ============================================================================
    # BASIC DIRECTION SETS
    # ============================================================================

    def _precompute_rook_directions(self) -> List[Tuple[int, int, int]]:
        """6 orthogonal directions."""
        return [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1)
        ]

    def _precompute_bishop_directions(self) -> List[Tuple[int, int, int]]:
        """20 diagonal directions."""
        return [
            # XY plane
            (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
            # XZ plane
            (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
            # YZ plane
            (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
            # 3D diagonals
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
            (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
        ]

    def _precompute_queen_directions(self) -> List[Tuple[int, int, int]]:
        """26 directions (rook + bishop)."""
        return self._precompute_rook_directions() + self._precompute_bishop_directions()

    def _precompute_vectorslider_directions(self) -> List[Tuple[int, int, int]]:
        """All possible primitive directions."""
        directions = set()
        MAX_D = SIZE - 1

        for dx in range(-MAX_D, MAX_D + 1):
            for dy in range(-MAX_D, MAX_D + 1):
                for dz in range(-MAX_D, MAX_D + 1):
                    if dx == dy == dz == 0:
                        continue
                    g = gcd(gcd(abs(dx), abs(dy)), abs(dz))
                    if g > 0:
                        prim = (dx // g, dy // g, dz // g)
                        directions.add(prim)

        return list(directions)

    def _precompute_trigonalbishop_directions(self) -> List[Tuple[int, int, int]]:
        """Bishop-like but only in coordinate planes (no 3D diagonals)."""
        return [
            # XY plane
            (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
            # XZ plane
            (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
            # YZ plane
            (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
        ]

    def _precompute_edgerook_directions(self) -> List[Tuple[int, int, int]]:
        """Same as rook - edge logic handled at runtime."""
        return self._precompute_rook_directions()

    def _precompute_xyqueen_directions(self) -> List[Tuple[int, int, int]]:
        """Queen movement restricted to XY planes (Z fixed)."""
        return [
            # Orthogonal in XY
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
            # Diagonal in XY
            (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)
        ]

    def _precompute_xzqueen_directions(self) -> List[Tuple[int, int, int]]:
        """Queen movement restricted to XZ planes (Y fixed)."""
        return [
            # Orthogonal in XZ
            (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1),
            # Diagonal in XZ
            (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1)
        ]

    def _precompute_yzqueen_directions(self) -> List[Tuple[int, int, int]]:
        """Queen movement restricted to YZ planes (X fixed)."""
        return [
            # Orthogonal in YZ
            (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
            # Diagonal in YZ
            (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)
        ]

    def _precompute_cone_directions(self) -> List[Tuple[int, int, int]]:
        """Face cone slider directions."""
        directions: Set[Tuple[int, int, int]] = set()
        MAX_D = SIZE - 1

        cones = [
            lambda dx, dy, dz: dx > 0 and abs(dy) <= dx and abs(dz) <= dx,  # +X
            lambda dx, dy, dz: dx < 0 and abs(dy) <= -dx and abs(dz) <= -dx, # -X
            lambda dx, dy, dz: dy > 0 and abs(dx) <= dy and abs(dz) <= dy,  # +Y
            lambda dx, dy, dz: dy < 0 and abs(dx) <= -dy and abs(dz) <= -dy, # -Y
            lambda dx, dy, dz: dz > 0 and abs(dx) <= dz and abs(dy) <= dz,  # +Z
            lambda dx, dy, dz: dz < 0 and abs(dx) <= -dz and abs(dy) <= -dz, # -Z
        ]

        for cone in cones:
            for dx in range(-MAX_D, MAX_D + 1):
                for dy in range(-MAX_D, MAX_D + 1):
                    for dz in range(-MAX_D, MAX_D + 1):
                        if dx == dy == dz == 0:
                            continue
                        if not cone(dx, dy, dz):
                            continue
                        g = gcd(gcd(abs(dx), abs(dy)), abs(dz))
                        if g > 0:
                            prim = (dx // g, dy // g, dz // g)
                            directions.add(prim)

        return list(directions)

    # ============================================================================
    # KNIGHT VARIANTS
    # ============================================================================

    def _precompute_knight_offsets(self) -> List[Tuple[int, int, int]]:
        """Standard 3D knight (2,1,0) combinations."""
        offsets = []
        for a, b in [(2, 1), (1, 2)]:
            for dx in [-a, a]:
                for dy in [-b, b]:
                    offsets.append((dx, dy, 0))
                for dz in [-b, b]:
                    offsets.append((dx, 0, dz))
            for dy in [-a, a]:
                for dx in [-b, b]:
                    offsets.append((dx, dy, 0))
                for dz in [-b, b]:
                    offsets.append((0, dy, dz))
            for dz in [-a, a]:
                for dx in [-b, b]:
                    offsets.append((dx, 0, dz))
                for dy in [-b, b]:
                    offsets.append((0, dy, dz))
        return list(set(offsets))

    def _precompute_knight31_offsets(self) -> List[Tuple[int, int, int]]:
        """Knight with (3,1,0) movement pattern."""
        offsets = []
        for a, b in [(3, 1), (1, 3)]:
            for dx in [-a, a]:
                for dy in [-b, b]:
                    offsets.append((dx, dy, 0))
                for dz in [-b, b]:
                    offsets.append((dx, 0, dz))
            for dy in [-a, a]:
                for dx in [-b, b]:
                    offsets.append((dx, dy, 0))
                for dz in [-b, b]:
                    offsets.append((0, dy, dz))
            for dz in [-a, a]:
                for dx in [-b, b]:
                    offsets.append((dx, 0, dz))
                for dy in [-b, b]:
                    offsets.append((0, dy, dz))
        return list(set(offsets))

    def _precompute_knight32_offsets(self) -> List[Tuple[int, int, int]]:
        """Knight with (3,2,0) movement pattern."""
        offsets = []
        for a, b in [(3, 2), (2, 3)]:
            for dx in [-a, a]:
                for dy in [-b, b]:
                    offsets.append((dx, dy, 0))
                for dz in [-b, b]:
                    offsets.append((dx, 0, dz))
            for dy in [-a, a]:
                for dx in [-b, b]:
                    offsets.append((dx, dy, 0))
                for dz in [-b, b]:
                    offsets.append((0, dy, dz))
            for dz in [-a, a]:
                for dx in [-b, b]:
                    offsets.append((dx, 0, dz))
                for dy in [-b, b]:
                    offsets.append((0, dy, dz))
        return list(set(offsets))

    def _precompute_king_offsets(self) -> List[Tuple[int, int, int]]:
        """All 26 adjacent squares."""
        return [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == dy == dz == 0)
        ]

    # ============================================================================
    # ZIG-ZAG PATTERNS
    # ============================================================================

    def _precompute_xzzigzag_patterns(self) -> Dict[str, List[List[Tuple[int, int, int]]]]:
        """Precompute XZ zig-zag ray patterns."""
        return self._precompute_zigzag_patterns(['XZ', 'XY', 'YZ'])

    def _precompute_yzzigzag_patterns(self) -> Dict[str, List[List[Tuple[int, int, int]]]]:
        """Precompute YZ zig-zag ray patterns."""
        return self._precompute_zigzag_patterns(['YZ', 'XZ', 'XY'])

    def _precompute_zigzag_patterns(self, planes: List[str]) -> Dict[str, List[List[Tuple[int, int, int]]]]:
        """Generate zig-zag patterns for specified planes."""
        SEGMENT_LENGTH = 3
        directions = [(1, -1), (-1, 1)]
        patterns = {}

        for plane in planes:
            plane_patterns = []
            for pri, sec in directions:
                ray = self._generate_single_zigzag_ray(plane, pri, sec, SEGMENT_LENGTH)
                plane_patterns.append(ray)
            patterns[plane] = plane_patterns

        return patterns

    def _generate_single_zigzag_ray(self, plane: str, primary_dir: int, secondary_dir: int, segment_length: int) -> List[Tuple[int, int, int]]:
        """Generate a single zig-zag ray pattern as relative offsets."""
        ray = []
        curr1, curr2 = 0, 0
        move_axis_1 = True

        for segment in range(4):
            direction = primary_dir if move_axis_1 else secondary_dir
            for step in range(segment_length):
                if move_axis_1:
                    curr1 += direction
                else:
                    curr2 += direction

                if plane == 'XZ':
                    coord = (curr1, 0, curr2)
                elif plane == 'XY':
                    coord = (curr1, curr2, 0)
                else:  # YZ
                    coord = (0, curr1, curr2)

                ray.append(coord)

            move_axis_1 = not move_axis_1

        return ray

    # ============================================================================
    # SPECIAL PIECE TARGETS
    # ============================================================================

    def _precompute_mirror_targets(self) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
        """Mirror teleport targets."""
        mirror_map = {}
        for x in range(SIZE):
            for y in range(SIZE):
                for z in range(SIZE):
                    mirror_map[(x, y, z)] = (SIZE - 1 - x, SIZE - 1 - y, SIZE - 1 - z)
        return mirror_map

    def _precompute_neighbor_directions(self) -> List[Tuple[int, int, int]]:
        """26 neighbor directions for Network Teleporter."""
        return self._precompute_king_offsets()

    # ============================================================================
    # SLIDING RAYS FOR ALL POSITIONS
    # ============================================================================

    def _precompute_all_sliding_rays(self) -> Dict[str, Dict[Tuple[int, int, int], List[List[Tuple[int, int, int]]]]]:
        """Precompute sliding rays for all positions and piece types."""
        piece_types = {
            'rook': self._precompute_rook_directions(),
            'bishop': self._precompute_bishop_directions(),
            'queen': self._precompute_queen_directions(),
            'vectorslider': self._precompute_vectorslider_directions(),
            'trigonalbishop': self._precompute_trigonalbishop_directions(),
            'xyqueen': self._precompute_xyqueen_directions(),
            'xzqueen': self._precompute_xzqueen_directions(),
            'yzqueen': self._precompute_yzqueen_directions(),
            'cone': self._precompute_cone_directions(),
            'edgerook': self._precompute_edgerook_directions()
        }

        all_rays = {}

        for piece_name, directions in piece_types.items():
            piece_rays = {}
            for x in range(SIZE):
                for y in range(SIZE):
                    for z in range(SIZE):
                        position_rays = []
                        for direction in directions:
                            ray = self._generate_ray_from((x, y, z), direction)
                            if ray:
                                position_rays.append(ray)
                        piece_rays[(x, y, z)] = position_rays
            all_rays[piece_name] = piece_rays

        return all_rays

    def _generate_ray_from(self, start: Tuple[int, int, int], direction: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Generate a ray from start position in given direction."""
        ray = []
        current = start
        while True:
            current = add_coords(current, direction)
            if not in_bounds(current):
                break
            ray.append(current)
        return ray

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run precomputation."""
    precomputer = StandaloneMovementPrecomputer()
    patterns = precomputer.precompute_all()

    print(f"Precomputation completed!")
    print(f"Generated patterns for {len(patterns)} movement types")
    print(f"Cache directory: {CACHE_DIR.absolute()}")

    # Show some stats
    total_rays = sum(len(rays) for rays_dict in patterns.get('sliding_rays', {}).values() for rays in rays_dict.values())
    print(f"Total sliding rays precomputed: {total_rays:,}")

if __name__ == "__main__":
    main()
