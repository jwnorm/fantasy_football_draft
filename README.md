# Fantasy Football Draft

> **Note:** This repo contains the files for my final project for *OR 708: Integer Programming* at NC State University.

## Overview

This project is intended to develop a **B**inary **I**nteger **P**rogramming (BIP) model to determine the optimal fantasy football roster based on total projected points scored for the 2025 NFL season. The model is based on 12-team PPR league with a 14-round draft, eliminating the defense and kicker positions since those decisions are often trivial. The roster consists of the following positions:

-   **Q**uarter**B**ack (QB): 1
-   **R**unning **B**ack (RB): 2
-   **W**ide **R**eceiver (WR): 2
-   **T**ight **E**nd (TE): 1
-   Flex: 2
-   Bench: 6

Data for the projections is obtained from several different sources, including:

-   [**The Athletic**](https://www.nytimes.com/athletic/6432965/2025/06/19/fantasy-football-2025-rankings-projections-cheat-sheet/)
-   [**Pro Football Focus (PFF)**](https://www.pff.com/news/fantasy-football-rankings-builder-2025)
-   [**numberFire**](https://www.fanduel.com/research/fantasy-football-printable-cheat-sheet-top-200-players-for-12-team-ppr-league-2025)
-   [**Rotoballer**](https://www.rotoballer.com/free-fantasy-football-draft-cheat-sheet)

Draft position and related data was obtained from the [**N**ational **F**antasy **C**hampionship (NFC)](https://nfc.shgn.com/adp/football) platform. Specifically, RotoWire Online 12-team leagues with drafts during the week leading up to Week 1 kickoff.

The goal of the project is to:

1)  Successfully model a fantasy football draft as a BIP model;
2)  Compare this against actual data through Week 13 of the 2025 NFL season; and
3)  Conduct a series of scenario analyses by changing various aspects of the draft.

The scenario analyses will include:

-   Projection system utilized
-   Beginning draft position in the first round
-   Proxy for player draft value
-   Reception scoring settings
-   Impact of removing the **T**ight **E**nd (TE) positional requirement
