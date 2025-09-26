# 3d-chess-for-AI
I thought it would be cool to create a game that's likely too complex/time-intensive for humans, but that AI could play. It would be interesting to see what sorts of strategies it develops.

9x9x9 3d chess game with 40 different pieces

Each side has 3 starting rows: 
Back row - king and pieces with large movement 
2nd row - walls and other pieces 
3rd row - pawns

Pieces:
1. PWN - pawn
2. KNT - knight
3. BIS - bishop
4. ROO - rook
5. QUE - queen
6. KNG - king
7. PRS - priest - king cannot be captured when alive
8. BKN - 3,2 knight
9. MKN - 3,1 knight
10. TRG - trigonal bishop (1,1,1)
11. HIV - hive - can move any number of this piece in a move (king movement)
12. ORB - orbiter - jumps to space on 4 radius hollow shell centered on it
13. NBL - nebula - jumps to space in small filled sphere centered on it
14. ECO - echo - jumps to small filled sphere 2 spaces away from each face
15. PNL - panel - jumps to orthogonal 9x9 wall from each face
16. EDG - edgerook - travels any number of spaces along edges
17. XYQ - xyqueen - plus king moves
18. XZQ - xz queen - plus king moves
19. YZQ - yz gueen - plus king moves
20. VSL - vector slider - (any combination of vectors)
21. CNS - coneslider - slides from a cone projected outward at each face
22. MIR - mirror - teleports to opposite coordinates plus king movement
23. ARM - armour - invulnerable to pawns, king movement
24. FRZ - freezer - small freezing aura
25. WAL - wall - 2x2 wall can only capture from behind
26. ARC - archer - small shell aura that can capture instead of move, plus king movement
27. BMB - bomb - small filled sphere capture aura, happens on death or move to same square
28. FTP - friendlyteleporter - can teleport to any square touching a friendly, king movement
29. SPD - speeder - small aura doubles friendly movement in z
30. SLW - slower - small aura forces pawn movement for enemies
31. GEO - geomancer - medium aura summons rocks to block off spaces that last 5 turns
32. SWP - swapper - swaps with friendly piece or king movement
33. XZZ - xz zigzag - horizontal zigzag slider from each face
34. YZZ - yz zigzag - vertical zigzag slider from each face
35. RFL - reflector - bishop that can reflect off edges or pieces
36. BKH - blackhole - all enemies pulled 1 space towards black hole at end of owner's turn
37. WTH - white hole - all enemies pushed 1 space away from white hole at end of owner's turn
38. INF - infiltrator - can teleport in front of enemy pawns, plus king movement
39. TRL - trailblazer - 3 space rook moves that leave a trail from the past 3 turns, enemy pieces that get 3 fire counters on them are captured
40. SPR - spiral - spiral of radius 2 slider counterclockwise from each face



1st Rank:

RFL  CNS  EDG  ECO  ORB  ECO  EDG  CNS  RFL<br>
SPR  XZZ  XZQ  YZQ  MIR  YZQ  XZQ  XZZ  SPR<br>
YZZ  FTP  PNL  HIV  MKN  HIV  PNL  FTP  YZZ<br>
BMB  SWP  NBL  BKN  TRL  BKN  NBL  SWP  BMB<br>
ORB  MIR  MKN  TRL  KNG  TRL  MKN  MIR  ORB<br>
BMB  SWP  NBL  BKN  TRL  BKN  NBL  SWP  BMB<br>
YZZ  FTP  PNL  HIV  MKN  HIV  PNL  FTP  YZZ<br>
SPR  XZZ  XZQ  YZQ  MIR  YZQ  XZQ  XZZ  SPR<br>
RFL  CNS  EDG  ECO  ORB  ECO  EDG  CNS  RFL<br>


2nd Rank:

FRZ  SLW  BKH  GEO  BIS  GEO  BKH  SLW  FRZ<br>
SPD  WAL  WAL  ARM  TRG  ARM  WAL  WAL  SPD<br>
WTH  WAL  WAL  PRS  KNT  PRS  WAL  WAL  WTH<br>
QUE  ARC  INF  ROO  XYQ  ROO  INF  ARC  QUE<br>
BIS  TRG  KNT  XYQ  VSL  XYQ  KNT  TRG  BIS<br>
QUE  ARC  INF  ROO  XYQ  ROO  INF  ARC  QUE<br>
WTH  WAL  WAL  PRS  KNT  PRS  WAL  WAL  WTH<br>
SPD  WAL  WAL  ARM  TRG  ARM  WAL  WAL  SPD<br>
FRZ  SLW  BKH  GEO  BIS  GEO  BKH  SLW  FRZ<br>


3rd Rank:

Pawns
