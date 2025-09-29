# game3d/policy_encoder.py
from game3d.game.gamestate import GameState
from game3d.movement.movepiece import Move  # or wherever Move is

class MoveEncoder:
    def __init__(self, max_actions: int = 600_000):
        self.max_actions = max_actions
        self.board_size = 9
        self.squares = self.board_size ** 3  # 729

    def move_to_index(self, move: Move) -> int:
        """
        Map a Move to a unique global index.
        Uses: from_coord → to_coord encoding.
        Total possible moves: 729 * 729 = 531,441
        """
        # ✅ Use .from_coord and .to_coord (properties that return (x,y,z))
        fx, fy, fz = move.from_coord
        tx, ty, tz = move.to_coord

        # Validate coordinates (optional, but safe)
        assert 0 <= fx < 9 and 0 <= fy < 9 and 0 <= fz < 9, f"Invalid from_coord: {move.from_coord}"
        assert 0 <= tx < 9 and 0 <= ty < 9 and 0 <= tz < 9, f"Invalid to_coord: {move.to_coord}"

        # Flatten 3D → 1D index (0 to 728)
        from_idx = fz * 81 + fy * 9 + fx  # z*81 + y*9 + x
        to_idx = tz * 81 + ty * 9 + tx

        # Global move index
        global_index = from_idx * self.squares + to_idx  # 0 to 531,440

        if global_index >= self.max_actions:
            raise ValueError(
                f"Move index {global_index} >= max_actions {self.max_actions}. "
                f"Increase max_actions to at least {self.squares * self.squares}"
            )

        return global_index

    # Optional: reverse mapping (useful for inference)
    def index_to_coords(self, index: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Convert global index back to (from_coord, to_coord)."""
        to_idx = index % self.squares
        from_idx = index // self.squares

        fx = from_idx % 9
        fy = (from_idx // 9) % 9
        fz = from_idx // 81

        tx = to_idx % 9
        ty = (to_idx // 9) % 9
        tz = to_idx // 81

        return (fx, fy, fz), (tx, ty, tz)

    def coord_to_index(self, coord: tuple[int, int, int]) -> int:
        x, y, z = coord
        return z * 81 + y * 9 + x  # 0..728
