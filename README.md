# 3d-chess-for-AI
I thought it would be cool to create a game that's likely too complex/time-intensive for humans, but that AI could play. It would be interesting to see what sorts of strategies it develops.

9x9x9 3d chess game with 40 different pieces

Each side has 3 starting rows: Back row - king and pieces with large movement 2nd row - walls and other pieces 3rd row - pawns

Pieces:

    king
    queen
    knight (can share square)
    3,2 knight
    3, 1 knight
    rook
    bishop
    pawn
    trigonal bishop
    priest (must capture before king)
    twin (up to 2 can move in the same turn)
    orbiter (jumps to a square on an empty sphere surrounding it)
    nebula (jumps to any square in a small sphere surrounding it)
    echo - jumps to any square in a small sphere detached from it
    panel - jumps to a 3x3 wall two spaces in front
    edgerook - moves along the edges
    xy slice queen
    xz slice queen
    yz slice queen
    piece that can jump to any free square it can draw a line to
    piece that can move along a cone
    piece that moves like a king, but can teleport to the mirror coords
    friendly teleporter
    sphere freezer
    2x2 wall you can only capture from behind
    archer
    bomb
    invincible to pawns piece
    aura movement buff in z
    aura debuff (pawn movement)
    blocker magician
    piece that can swap places with a friendly upon capture
    xz zigzag piece
    yz zigzag piece
    bishop that reflects off edges/pieces
    black hole
    white hole
    teleport to spot in front of pawns
    trail of fire
    spiral piece

