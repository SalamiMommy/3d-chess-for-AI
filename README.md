# 3d-chess-for-AI
I thought it would be cool to create a game that's likely too complex/time-intensive for humans, but that AI could play. It would be interesting to see what sorts of strategies it develops.

9x9x9 3d chess game with 40 different pieces

Each side has 3 starting rows: 
Back row - king and pieces with large movement 
2nd row - walls and other pieces 
3rd row - pawns

Pieces:
1. king
2. queen
3. knight (can share square)
4. 3,2 knight
5. 3, 1 knight
6. rook
7. bishop
8. pawn
9. trigonal bishop
10. priest (must capture before king)
11. twin (up to 2 can move in the same turn)
12. orbiter (jumps to a square on an empty sphere surrounding it)
13. nebula (jumps to any square in a small sphere surrounding it)
14. echo - jumps to any square in a small sphere detached from it
15. panel - jumps to a 3x3 wall two spaces in front
16. edgerook - moves along the edges
17. xy slice queen
18. xz slice queen
19. yz slice queen
20. piece that can jump to any free square it can draw a line to
21. piece that can move along a cone
22. piece that moves like a king, but can teleport to the mirror coords
23. friendly teleporter
24. sphere freezer
25. 2x2 wall you can only capture from behind
26. archer
27. bomb
28. invincible to pawns piece
29. aura movement buff in z
30. aura debuff (pawn movement)
31. blocker magician
32. piece that can swap places with a friendly upon capture
33. xz zigzag piece
34. yz zigzag piece
35. bishop that reflects off edges/pieces
36. black hole
37. white hole
38. teleport to spot in front of pawns
39. trail of fire
40. spiral piece

