import torch
from .enums import Color, PieceType

class Piece:
    """Immutable piece descriptor."""
    __slots__ = ("color", "ptype", "armoured")

    def __init__(self, color: Color, ptype: PieceType, armoured: bool = False):
        self.color = color
        self.ptype = ptype
        self.armoured = armoured

    def to_tensor(self) -> torch.Tensor:
        """Returns shape (2,) int8 tensor: [color_value, ptype_value]."""
        return torch.tensor([self.color.value, self.ptype.value], dtype=torch.int8)

    def __hash__(self):
        return hash((self.color, self.ptype))

    def __eq__(self, other):
        return isinstance(other, Piece) and self.color == other.color and self.ptype == other.ptype

    def __repr__(self) -> str:
        return f"Piece({self.color.name}, {self.ptype.name})"
