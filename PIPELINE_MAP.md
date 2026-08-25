# FPL Workbook & Pipeline Map

Mapped 2026-07-21 from `Fantasy Premier League.xlsx` (32 sheets) and the repo scripts.
Purpose: reference for replicating the whole process in Python with observable DataFrames.

> **Status: replicated and retired.** The `fpl_pipeline/` package reproduces the Players
> sheet with full parity (161/161 columns vs the workbook's cached values, 2026-07-21)
> and has since gained model improvements, automated archives, a coefficient-refit tool,
> a consolidated name-mapping table, and a pytest suite (2026-07-25). **See README.md for
> current usage** — this document remains the map of how the original workbook worked.
>
> - Run: `python -m fpl_pipeline.run` — every stage snapshots to `outputs/NN_*.csv`,
>   ending in `outputs/13_players_master.csv` (the Players sheet equivalent).
> - Validate vs Excel: `python -m fpl_pipeline.run --validate` (parity report in outputs/).
> - Manual data now lives in `inputs/*.csv` (exported by `tools/export_workbook_inputs.py`);
>   regression coefficients in `fpl_pipeline/data/coefficients.json` (extracted by
>   `tools/extract_coefficients.py`).
> - Known deliberate deviations from the workbook (all documented in
>   `fpl_pipeline/players.py`): the AW2 leftover-formula cell, exact-match fallback-factor
>   lookups, and NaN instead of VLOOKUP's 0 for blank opponent/venue cells.
> - Single-gameweek odds handling (added 2026-07-25): `sportsbet.py` no longer duplicates
>   F1 odds into the F2 CSVs when only one gameweek is priced; the pipeline additionally
>   ignores F2 odds files that are byte-identical to F1; and when F2 match odds are
>   missing, F2 falls back to the F3-style model (win probabilities predicted from season
>   odds, stats via factor × baseline) — requires GW2 opponents in `inputs/fixtures.csv`.
>   Real F2 odds always take precedence once bookmakers price the second gameweek.
> - Probability clamping (added 2026-07-25): the workbook's linear regressions are
>   unbounded (win prediction unclamped, opponent-win clamped only at 0, factor ×
>   baseline stats can leave [0, 1]); improved mode clips every modelled probability to
>   [0, 1] before the ladders and XP scoring. Odds-derived probabilities are untouched.
>   `--validate` disables all these improvements for workbook-exact parity (still 161/161).
> - Both PuLP optimisers now default to the pipeline outputs
>   (`outputs/13_players_master.csv` + `inputs/gw_teams.csv`) and were verified to produce
>   byte-identical results vs their Excel-fed runs; pass an `.xlsx` path to use the
>   legacy workbook. Run them with the repo virtualenv (`env/Scripts/python`), which has
>   PuLP installed.

## The process in one paragraph

Bookmaker odds (Sportsbet scrape + manual entry) and FPL data (FPL-Core-Insights CSVs) flow into
staging sheets. The **Players** sheet (818 rows × 161 cols, ~131k formulas) converts odds into
per-player probabilities for 6 upcoming fixtures (F1–F6), turns them into expected FPL points (XP)
per fixture using the official scoring rules, and blends them into **Total XP**
(`F1 + 0.85·F2 + 0.7·(F3+F4+F5+F6)`). The two PuLP optimisers read Players + GW Teams and print
optimal squads / transfer plans to the console. Results are manually recorded back into
tracking sheets (GW Teams, Historical Expected Points).

Where real odds exist (F1, partially F2) they are used directly. Where they don't (F3–F6), the
model predicts them: each stat has a **regression baseline** (probability as a function of
win/opponent-win probability, position, venue) and each player has a **factor** = actual F1 odds ÷
F1 baseline. Future fixture probability = factor × baseline at that fixture's predicted win odds.
Win odds for F3–F6 are themselves predicted from title/relegation/top-6 odds (Lasso regression).
**Exception (`config.USE_PROJECTION_MODEL`):** for `clean_sheet`/`concede2`/`saves3` in F3–F8 — and
the F2 fallback where F2 has no real odds (synthetic seeds F1 only, so F2+ derive off F1) — a trained
LightGBM model replaces `factor × baseline` (via `fpl_pipeline/projection_serving.py`). See README
**Forward-projection models**.

## Scripts → sheets

| Script | Reads | Writes (sheets) | Notes |
|---|---|---|---|
| `sportsbet.py` | Sportsbet AU API | Fixture Odds, Score, Score 2+, Assist, Assist 2+, Yellow Card, Clean Sheet, Team Total Goals, Goalkeeper Saves, F2 Clean Sheet, F2 Team Total Goals (+ CSVs in `sportsbet/`) | F2 sheets get `.tail(20)` of the same scrape (next GW's matches), base sheets `.head(20)` |
| `starting_lineups.py` | fantasyfootballscout.co.uk team news | Starting Lineups **cols A–B only** (+ `starting_lineups/data.csv`) | Start probabilities C–H are manual |
| `extract_fpl_data.py` | FPL-Core-Insights `players.csv`, `playerstats.csv`, `teams.csv` + `fpl_data/player_name_changes.csv` | FPL Players (name/position/team/cost) | Roster source for Players sheet |
| `extract_defensive_contributions.py` | FPL-Core-Insights `players.csv`, `playerstats.csv` | Defensive Contribution DEF / MID **cols A–C only** | Cols D–E are formulas; G–H (SD, average) are manual |
| `optimisation.py` | Players, GW Teams | console only | PuLP multi-transfer optimiser (max_transfers, hits, captain) |
| `optimisation_full.py` | Players | console only | PuLP full-squad optimiser (wildcard/free-hit style) |
| `fpl_data/calculate_standard_deviations.py` | FPL-Core-Insights GW folders | console only | Mean/SD of defensive_contribution per 90 → manually typed into DC sheets G/H |
| `modelling/predict_match_win.py` | **hardcoded data** (paste of Historical Fixture Odds) | console only | Lasso for win% from title/releg/top6 odds; prints the Excel formula pasted into Players (F3–F6 "Win Pred") |
| `strength_modelling/match_odds.py` | hardcoded data | console only | Source of Coefficients rows 41–46 ("Match Odds" = opponent win pred) |
| `strength_modelling/strength_modelling*.py` | hardcoded data | console only | Older experiments fitting the per-stat baselines (Score/Assist/YC/CS/Concede/Saves) vs Win/OppWin/Diff/Venue |
| *(deleted 2025-12-25)* `clean_sheet.py`, `goalkeeper_saves.py`, `score_or_assist.py`, `team_total_goals.py`, `yellow_card.py`, `bonus_point_analysis.py` | saved HTML odds pages (`odds_data/`) | produced `odds_data_outputs/*.csv` | Fed **F2 Score & Assist, F2 Yellow Card, F2 Goalkeeper Saves** sheets — that pipeline is gone; those sheets are now manual/stale |

## Manual steps (no script)

- **Title Odds / Relegation Odds / Top 6 Odds**: bookmaker odds pasted into cols B–Y (~24 books); AD = AVERAGE.
- **Fixtures**: team → opponent/venue for upcoming GWs (currently only GW38 — season over).
- **Starting Lineups C–H**: start probability per player per fixture (F1–F6).
- **GW Teams**: squad picked each GW (15 slots × GW1–GW37 columns).
- **Defensive Contribution DEF/MID G–H**: SD and average DC per 90 (from `calculate_standard_deviations.py` console output).
- **Fallback Factors**: paste-values snapshot of each player's factors (used when a player has no F1 odds, e.g. blank GW/no market).
- **Historical Fixture Odds**: append each GW's fixture odds + season odds; feeds `predict_match_win.py` (by copy-paste into the script).
- **Historical Player Data**: paste-values archive of Players F1 columns each GW (training data for baseline regressions).
- **Historical Expected Points**: log of optimiser outputs (current team XP vs transfers/free hit/wildcard) per GW.
- **Coefficients**: regression outputs typed in from the modelling scripts.

## Sheet inventory

| Sheet | Populated by | Read by | Status |
|---|---|---|---|
| Players | formulas (hub) | Starting Lineups I–L; both optimisers | **core** |
| FPL Players | `extract_fpl_data.py` | Players A–D | active |
| Starting Lineups | script (A–B) + manual (C–H) + formulas (I–L) | Players (start probs H, AN, BM, CK, DI, EG) | active |
| Fixture Odds | `sportsbet.py` (WDW) | Team Fixture Odds | active |
| Fixtures | manual | Team Fixture Odds | active |
| Team Fixture Odds | formulas | Players (win/opp-win/venue/opponent per fixture) | active |
| Title / Relegation / Top 6 Odds | manual | Overall Odds | active |
| Overall Odds | formulas (incl. team-name cleanup and Man City/Wolves special cases) | Players E–G, N–P etc.; Historical Fixture Odds | active |
| Score, Score 2+, Assist, Yellow Card, Clean Sheet, Team Total Goals, Goalkeeper Saves | `sportsbet.py` + margin-removal formulas | Players F1 block | active |
| F2 Clean Sheet, F2 Team Total Goals | `sportsbet.py` (tail 20) | Players F2 block | active |
| F2 Yellow Card | orphaned (deleted pipeline) | Players BA (IFERROR → falls back to model) | **stale but referenced** |
| Defensive Contribution DEF / MID | script (A–C) + formulas (D–E) + manual (G–H) | Players AA/AB | active |
| Coefficients | manual (from modelling scripts) | Players (F2 Score 1+, F3–F6 opponent win pred) | active |
| Fallback Factors | manual paste-values | Players AC–AI fallbacks | active |
| GW Teams | manual | `optimisation.py` | active (script input) |
| Historical Fixture Odds | manual + formulas | `modelling/predict_match_win.py` (via paste) | active (offline) |
| Historical Player Data | manual paste-values | baseline regression fitting (offline) | active (offline) |
| Historical Expected Points | manual | nothing | tracking only |
| **Assist 2+** | `sportsbet.py` | **nothing** | unused (scraped but never read) |
| **F2 Score & Assist** | orphaned (deleted pipeline) | **nothing** | unused |
| **F2 Goalkeeper Saves** | orphaned (deleted pipeline) | **nothing** | unused |
| **Teams** | one-off FPL API dump | **nothing** | unused |

## Model logic (the formulas, in DataFrame terms)

### 1. Odds → probability (margin removal)
- Player/team markets: `p = 1/odds/1.05` (5% overround). WDW: `1/odds/1.03`. Season odds: `1/avg_odds/1.08`.
- Two-sided markets (Team Total Goals 2+/4+): `p = (1/over) / (1/over + 1/under)`.
- GK saves default when no odds: 3+ saves → 0.6, 6+ saves → 0.
- Defensive contribution: `P(DC ≥ threshold) = 1 − NORM.DIST(threshold, dc_per_90, SD, TRUE)`; threshold 10 (DEF) / 12 (MID); if <4 games played use position average instead.

### 2. Baseline regressions (hardcoded in Players formulas)
For each stat s ∈ {Score1+, Assist, YellowCard, CleanSheet, Concede2+, Concede4+, 3+Saves, 6+Saves}:
`baseline_s(win, opp_win, pos, venue)` — linear model with terms: win, opp_win, pos dummies,
home dummy, diff, |diff|, diff², win×home, opp_win×home, diff×home, pos×diff (+ win×opp_win for saves/CS).
Same coefficient sets are reused for F1 (to derive factors) and F2–F6 (to project).

### 3. Player factors
`factor_s = F1_actual_prob_s / baseline_s(F1 win, F1 opp_win, …)`; if no F1 odds → Fallback Factors sheet.
Projection for fixture k: `prob_s(k) = factor_s × baseline_s(win_k, opp_win_k, …)`.
F2 uses real odds where available (Clean Sheet, Team Total Goals, Yellow Card via IFERROR; Score 1+ via
Coefficients rows 1–5 model × factor; Score 2+/3+ via threshold ladder on Score 1+).

### 4. Win probability for F3–F6 (no odds yet)
- Team: hardcoded Lasso (from `predict_match_win.py`): features from title/releg/top6 probs of both teams + venue.
- Opponent: Coefficients rows 41–46 (`max(0, intercept + b·title + c·opp_title + d·releg + e·opp_releg + f·away)`).

### 5. Expected points per fixture
`XP_pre = P(start) × Σ position-weighted points`:
appearance 2; goals = P(1+)·g + P(2+)·g + P(3+)·g with g = 10/6/5/4 for GK/DEF/MID/FWD
(tail-sum trick ≈ E[goals]·g); assist ·3; yellow −1; clean sheet +4 (GK/DEF, +1 MID);
concede 2+/4+ −1 each (GK/DEF); saves 3+/6+ +1 each (GK); DC 2 pts at position threshold.
Bonus: `P(bonus) = clamp(−0.021039 + 0.023522·XP_pre, 0, 1)`; `XP = XP_pre + 2·P(bonus)`.
`Total XP = F1 + 0.85·F2 + 0.7·(F3+F4+F5+F6)` (start-probability weighted already).
`F1 Pred XP` (col AM) = pure-regression XP sanity check, not used downstream.

### 6. Optimisation
PuLP ILP: 15-man squad, ≤3 per club, budget, position quotas; `optimisation.py` adds
current-squad constraints, transfer count/hits, captaincy; both print rosters to console.

## Quirks worth preserving/reviewing in the rewrite

- Team-name normalisation is scattered: `sportsbet.py` (Tottenham→Spurs, Nottm Forest→Nott'm Forest),
  Overall Odds formula (Nottingham Forest/Tottenham/Wolverhampton), `player_name_changes.csv`, and a
  large hardcoded map in `starting_lineups.py`.
- Overall Odds D2: Man City's relegation odds are taken from row 6 ($B$6) and Wolves' zero-odds are
  mapped to 1 — special-case handling of missing/suspended markets.
- Zero odds in Title/Top 6 → 5001, Relegation → 2001 (i.e. "no market" sentinel).
- The workbook is currently in end-of-season state (Fixtures only has GW38), so many F2+ lookups
  currently resolve via fallbacks/#N/A paths.
