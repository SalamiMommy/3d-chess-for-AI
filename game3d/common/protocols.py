#game3d/common/protocols.py
from typing import Protocol, runtime_checkable
from typing import Iterable, Tuple, Optional
from game3d.common.common import Coord
from game3d.pieces.piece import Piece

@runtime_checkable
class BoardProto(Protocol):
    def list_occupied(self) -> Iterable[Tuple[Coord, Piece]]: ...
    def piece_at(self, c: Coord) -> Optional[Piece]: ...
    # Add other methods as needed
