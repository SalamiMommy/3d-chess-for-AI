import torch
from .enums import Color, PieceType

class Piece:
    """Immutable piece descriptor."""
    __slots__ = ("color", "ptype")

    def __init__(self, color: Color, ptype: PieceType):
        self.color = color
        self.ptype = ptype

    def to_tensor(self) -> torch.Tensor:
        """Compressed 2-int tensor (color, type)."""
        return torch.tensor([self.color, self.ptype], dtype=torch.int8)

    def __hash__(self):
        return hash((self.color, self.ptype))

    def __eq__(self, other):
        return isinstance(other, Piece) and self.color == other.color and self.ptype == other.ptype
