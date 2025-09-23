"""Move class for 9×9×9 3D chess — lightweight, hashable, training-ready."""

from __future__ import annotations
from typing import Optional, Tuple
from pieces.enums import PieceType, Color
from common import Coord

class Move:
    """
    Represents a single atomic move on the 9x9x9 board.

    Designed for:
    - Move generation
    - Legal move lists
    - Engine search
    - Dataset serialization (zero-copy ready)
    - GUI/UCI output
    """
    __slots__ = (
        'from_coord',      # (x, y, z) source
        'to_coord',        # (x, y, z) destination
        'is_capture',      # bool
        'captured_piece',  # Optional[PieceType] — for recaptures, Zobrist, training
        'is_promotion',    # bool
        'promotion_type',  # Optional[PieceType] — what pawn becomes
        'is_en_passant',   # bool
        'is_castle',       # bool (for 3D castling if implemented)
        'move_id'          # Optional[int] — for move ordering or dataset indexing
    )

    def __init__(
        self,
        from_coord: Coord,
        to_coord: Coord,
        is_capture: bool = False,
        captured_piece: Optional[PieceType] = None,
        is_promotion: bool = False,
        promotion_type: Optional[PieceType] = None,
        is_en_passant: bool = False,
        is_castle: bool = False,
        move_id: Optional[int] = None
    ):
        self.from_coord: Coord = from_coord
        self.to_coord: Coord = to_coord
        self.is_capture: bool = is_capture
        self.captured_piece: Optional[PieceType] = captured_piece
        self.is_promotion: bool = is_promotion
        self.promotion_type: Optional[PieceType] = promotion_type or (PieceType.QUEEN if is_promotion else None)
        self.is_en_passant: bool = is_en_passant
        self.is_castle: bool = is_castle
        self.move_id: Optional[int] = move_id

    def __eq__(self, other) -> bool:
        if not isinstance(other, Move):
            return False
        return (self.from_coord == other.from_coord and
                self.to_coord == other.to_coord and
                self.is_capture == other.is_capture and
                self.is_promotion == other.is_promotion and
                self.promotion_type == other.promotion_type and
                self.is_en_passant == other.is_en_passant and
                self.is_castle == other.is_castle)
                # captured_piece ignored for equality — optional metadata

    def __hash__(self) -> int:
        # Hash based on core immutable move data
        return hash((
            self.from_coord,
            self.to_coord,
            self.is_capture,
            self.is_promotion,
            self.promotion_type,
            self.is_en_passant,
            self.is_castle
        ))

    def __repr__(self) -> str:
        fx, fy, fz = self.from_coord
        tx, ty, tz = self.to_coord
        capture = "x" if self.is_capture else "-"
        promo = f"={self.promotion_type.name[0]}" if self.is_promotion and self.promotion_type else ""
        ep = " e.p." if self.is_en_passant else ""
        castle = " O-O" if self.is_castle else ""
        return f"({fx},{fy},{fz}){capture}({tx},{ty},{tz}){promo}{ep}{castle}"

    def to_tuple(self) -> Tuple[int, int, int, int, int, int, int, int, int, int]:
        """
        Flatten move to tuple of ints for serialization or tensor conversion.
        Format: (fx, fy, fz, tx, ty, tz, is_cap, is_promo, promo_type_id, is_ep)
        """
        fx, fy, fz = self.from_coord
        tx, ty, tz = self.to_coord
        promo_id = self.promotion_type.value if self.promotion_type else 0
        return (
            fx, fy, fz,
            tx, ty, tz,
            int(self.is_capture),
            int(self.is_promotion),
            promo_id,
            int(self.is_en_passant)
        )

    @classmethod
    def from_tuple(cls, data: Tuple[int, ...]) -> 'Move':
        """
        Reconstruct Move from serialized tuple (inverse of to_tuple).
        """
        fx, fy, fz, tx, ty, tz, is_cap, is_promo, promo_id, is_ep = data[:10]
        from_coord = (fx, fy, fz)
        to_coord = (tx, ty, tz)
        promotion_type = PieceType(promo_id) if is_promo and promo_id != 0 else None

        return cls(
            from_coord=from_coord,
            to_coord=to_coord,
            is_capture=bool(is_cap),
            is_promotion=bool(is_promo),
            promotion_type=promotion_type,
            is_en_passant=bool(is_ep)
        )

    def is_forward_pawn_push(self, color: Color) -> bool:
        """Helper: useful for heuristics or policy masking."""
        if not (self.from_coord[1] + (1 if color == Color.WHITE else -1) == self.to_coord[1]):
            return False
        return self.from_coord[0] == self.to_coord[0] and self.from_coord[2] == self.to_coord[2]

    def manhattan_distance(self) -> int:
        """Distance moved — useful for sorting or heuristics."""
        fx, fy, fz = self.from_coord
        tx, ty, tz = self.to_coord
        return abs(tx-fx) + abs(ty-fy) + abs(tz-fz)
