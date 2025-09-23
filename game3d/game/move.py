"""Minimal immutable move value object."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

from pieces.enums import Color


@dataclass(slots=True, frozen=True)
class Move:
    """A 3-D move."""
    from_coord: tuple[int, int, int]  # (x, y, z)
    to_coord: tuple[int, int, int]
    is_capture: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    # optional helpers
    @property
    def dx(self) -> int:
        return self.to_coord[0] - self.from_coord[0]

    @property
    def dy(self) -> int:
        return self.to_coord[1] - self.from_coord[1]

    @property
    def dz(self) -> int:
        return self.to_coord[2] - self.from_coord[2]

    def __str__(self) -> str:
        fx, fy, fz = self.from_coord
        tx, ty, tz = self.to_coord
        cap = "x" if self.is_capture else "-"
        return f"{fx}{fy}{fz}{cap}{tx}{ty}{tz}"


# ------------------------------------------------------------------
# helper factories (optional)
# ------------------------------------------------------------------
def make_move(
    from_coord: tuple[int, int, int],
    to_coord: tuple[int, int, int],
    is_capture: bool = False,
    **meta: Any,
) -> Move:
    """Convenience factory."""
    return Move(
        from_coord=from_coord,
        to_coord=to_coord,
        is_capture=is_capture,
        metadata=meta,
    )
