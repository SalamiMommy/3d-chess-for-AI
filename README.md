# 3d-chess-for-AI
I thought it would be cool to create a game that's likely too complex/time-intensive for humans, but that AI could play. It would be interesting to see what sorts of strategies it develops.

9x9x9 3d chess game with 40 different pieces

Each side has 3 starting rows: 
Back row - king and pieces with large movement 
2nd row - walls and other pieces 
3rd row - pawns

Pieces:
1. pawn
2. knight
3. bishop
4. rook
5. queen
6. king
7. priest - king cannot be captured when alive
8. 3,2 knight
9. 3,1 knight
10. trigonal bishop
11. hive - can move any number like kings in a move
12. orbiter - jumps to space on hollow shell centered on it
13. nebula - jumps to space in small filled shell centered on it
14. echo - jumps to small filled shell 2 spaces away from each face
15. panel - jumps to orthogonal 9x9 wall from each face
16. edgerook
17. xyqueen - plus king moves
18. xz queen - plus king moves
19. yz gueen - plus king moves
20. vector slider - (any combination of vectors)
21. coneslider - slides from a cone projected outward at each face
22. mirror - teleports to opposite coordinates plus king movement
23. armour - invulnerable to pawns, king movement
24. freezer - small freezing aura
25. wall - 2x2 wall can only capture from behind
26. archer - small shell aura that can capture instead of move, plus king movement
27. bomb - small filled capture aura, happens on death or move to same square
28. friendlyteleporter - can teleport to any square touching a friendly, king movement
29. speeder - small aura doubles friendly movement in z
30. slower - small aura forces pawn movement for enemies
31. geomancer - medium aura summons rocks to block off spaces that last 5 turns
32. swapper - swaps with friendly piece or king movement
33. xz zigzag - horizontal zigzag slider from each face
34. xy zigzag - vertical zigzag slider from each face
35. reflector - bishop that can reflect off edges or pieces
36. blackhole - all enemies pulled 1 space towards black hole at end of owner's turn
37. white hole - all enemies pushed 1 space away from white hole at end of owner's turn
38. infiltrator - can teleport in front of enemy pawns, plus king movement
39. trailblazer - 3 space rook moves that leave a trail from the past 3 turns, enemy pieces that get 3 fire counters on them are captured
40. spiral - spiral of radius 2 slider counterclockwise from each face

