"""Full GameState with 50-move rule, insufficient-material, stalemate."""

from __future__ import annotations
import torch
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Tuple, Optional

from pieces.enums import Color, PieceType, Result
from game.board import Board
from game.move import Move
from game3d.cache.movecache import init_cache, get_cache, MoveCache
from game3d.movement.check import king_in_check


# ------------------------------------------------------------------
# insufficient-material evaluators  (extend as needed)
# ------------------------------------------------------------------
def _insufficient_material(board: Board) -> bool:
    """Placeholder – returns True if neither side can mate."""
    # TODO: plug in real logic (e.g. K vs K, K+B vs K, etc.)
    # for now we just count total pieces
    total = 0
    for _, p in board.list_occupied():
        if p.ptype not in (PieceType.KING, PieceType.PRIEST):
            total += 1
    return total == 0          # K+priest vs K+priest  →  insufficient


# ------------------------------------------------------------------
# GameState
# ------------------------------------------------------------------
@dataclass(slots=True)
class GameState:
    board: Board
    current: Color
    history: List[Move] = field(default_factory=list)
    halfmove_clock: int = 0          # plies since last pawn move / capture

    # ----------------------------------------------------------
    # life-cycle
    # ----------------------------------------------------------
    from game3d.cache.manager import init_cache_manager, get_cache_manager

    def __post_init__(self) -> None:
        init_cache_manager(self.board)
    # ----------------------------------------------------------
    # tensor for NN – unchanged
    # ----------------------------------------------------------
    def to_tensor(self) -> torch.Tensor:
        board_t = self.board.tensor()          # (C,9,9,9)
        player_t = torch.full((1, 9, 9, 9), float(self.current))
        return torch.cat([board_t, player_t], dim=0)

    # ----------------------------------------------------------
    # move generation
    # ----------------------------------------------------------
    @lru_cache(maxsize=32_768)
    def legal_moves(self) -> Tuple[Move, ...]:
        return tuple(get_cache().legal_moves(self.current))

    def pseudo_legal_moves(self) -> List[Move]:
        from game3d.movement.pseudo_legal import generate_pseudo_legal_moves
        return generate_pseudo_legal_moves(self)

    # ----------------------------------------------------------
    # make / undo – cache-coherent + rule updates
    # ----------------------------------------------------------
    def make_move(self, mv: Move) -> GameState:
        new_board = self.board.clone()
        new_board.apply_move(mv)

        # 50-move rule update
        moved_piece = self.board.piece_at(mv.from_coord)
        target_piece = self.board.piece_at(mv.to_coord)
        is_pawn = moved_piece is not None and moved_piece.ptype == PieceType.PAWN
        is_capture = target_piece is not None
        new_clock = 0 if (is_pawn or is_capture) else self.halfmove_clock + 1

        get_cache().apply_move(mv, self.current)

        new_hist = self.history.copy()
        new_hist.append(mv)

        # === BLACK-HOLE SUCK ===
        pull_map = get_cache_manager().black_hole_pull_map(self.current)
        for from_sq, to_sq in pull_map.items():
            piece = new_board.piece_at(from_sq)
            if piece is None or piece.color == self.current:
                continue  # safety
            new_board.set_piece(to_sq, piece)
            new_board.set_piece(from_sq, None)  # vacate old square
        # === END SUCK ===

        # === WHITE-HOLE PUSH ===
        push_map = get_cache_manager().white_hole_push_map(self.current)
        for from_sq, to_sq in push_map.items():
            piece = new_board.piece_at(from_sq)
            if piece is None or piece.color == self.current:
                continue
            new_board.set_piece(to_sq, piece)
            new_board.set_piece(from_sq, None)
        # === END PUSH ===

        # === BOMB DETONATION ===
        trigger_piece = self.board.piece_at(mv.from_coord)
        target_piece  = self.board.piece_at(mv.to_coord)

        # trigger 1: enemy captured a bomb
        if (target_piece is not None and
            target_piece.ptype == PieceType.BOMB and
            target_piece.color != self.current):
            cleared = detonate(new_board, mv.to_coord)
            for sq in cleared:
                new_board.set_piece(sq, None)

        # trigger 2: bomb moved onto its own square (self-detonation)
        if (trigger_piece is not None and
            trigger_piece.ptype == PieceType.BOMB and
            mv.from_coord == mv.to_coord):          # you can define this as legal for bombs
            cleared = detonate(new_board, mv.to_coord)
            for sq in cleared:
                new_board.set_piece(sq, None)
        # === END BOMB ===

        # === TRAILBLAZE COUNTERS ===
        trail_cache = get_trailblaze_cache()
        enemy_color = self.current.opposite()

        # 1. enemy **moved through** squares (every slid square)
        #    (you must pass the enemy's slid path from their generator)
        #    below shows the idea for the enemy that just moved
        enemy_slid = extract_enemy_slid_path(mv)  # you’ll implement this helper
        for sq in enemy_slid:
            if trail_cache.increment_counter(sq, enemy_color):
                victim = new_board.piece_at(sq)
                if victim is not None and victim.ptype != PieceType.KING:
                    new_board.set_piece(sq, None)  # remove non-king
                elif victim is not None and victim.ptype == PieceType.KING:
                    # king spared only if priests alive
                    if not _any_priest_alive(new_board, enemy_color):
                        new_board.set_piece(sq, None)

        # 2. enemy **ended** move on a trail square
        if trail_cache.increment_counter(mv.to_coord, enemy_color):
            victim = new_board.piece_at(mv.to_coord)
            if victim is not None and victim.ptype != PieceType.KING:
                new_board.set_piece(mv.to_coord, None)
            elif victim is not None and victim.ptype == PieceType.KING:
                if not _any_priest_alive(new_board, enemy_color):
                    new_board.set_piece(mv.to_coord, None)
        # === END TRAILBLAZE ===

        return GameState(
            board=new_board,
            current=self.current.opposite(),
            history=new_hist,
            halfmove_clock=new_clock,
        )

    def undo_move(self) -> Optional[GameState]:
        if not self.history:
            return None
        mv = self.history[-1]

        new_board = self.board.clone()
        new_board.undo_move(mv)           # you’ll add this

        # undo clock (naïve – restore previous; full impl stores prev)
        new_clock = max(0, self.halfmove_clock - 1)

        get_cache().undo_move(mv, self.current.opposite())

        return GameState(
            board=new_board,
            current=self.current.opposite(),
            history=self.history[:-1],
            halfmove_clock=new_clock,
        )
    def submit_block(self, sq: Tuple[int, int, int]) -> bool:
        """Geomancer player requests to block an unoccupied square within 3-sphere."""
        if not self.state.current_player_controls_geomancer():  # you’ll add this helper
            return False
        if sq not in block_candidates(self.state.board, self.state.current):
            return False  # outside 3-sphere
        return get_cache_manager().block_square(sq, len(self.state.history))  # ply = history length
    # ----------------------------------------------------------
    # game outcome
    # ----------------------------------------------------------
    def is_check(self) -> bool:
        return king_in_check(self.board, self.current, self.current)

    def is_stalemate(self) -> bool:
        return not self.is_check() and not self.legal_moves()

    def is_insufficient_material(self) -> bool:
        return _insufficient_material(self.board)

    def is_fifty_move_draw(self) -> bool:
        return self.halfmove_clock >= 100  # 50 moves = 100 plies

    def is_game_over(self) -> bool:
        return (
            self.is_fifty_move_draw()
            or self.is_insufficient_material()
            or not self.legal_moves()  # checkmate or stalemate
        )

    def result(self) -> Optional[Result]:
        """None while ongoing; Result.WHITE/BLACK/DRAW when finished."""
        if not self.is_game_over():
            return None
        if self.is_fifty_move_draw() or self.is_insufficient_material() or self.is_stalemate():
            return Result.DRAW
        # otherwise no legal moves → checkmate
        return Result.WHITE if self.current == Color.BLACK else Result.BLACK

    # ----------------------------------------------------------
    # debug
    # ----------------------------------------------------------
    def __repr__(self) -> str:
        return (f"GameState(current={self.current}, "
                f"history={len(self.history)}, "
                f"clock={self.halfmove_clock}, "
                f"legal={len(self.legal_moves())})")
