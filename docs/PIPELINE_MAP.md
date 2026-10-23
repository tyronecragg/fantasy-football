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
>   ending in `outputs/players_master.csv` (the Players sheet equivalent).
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
>   (`outputs/players_master.csv` + `inputs/gw_teams.csv`) and were verified to produce
>   byte-identical results vs their Excel-fed runs; pass an `.xlsx` path to use the
>   legacy workbook. Run them with the repo virtualenv (`env/Scripts/python`), which has
>   PuLP installed.

## The process in one paragraph

Bookmaker odds (Sportsbet scrape + manual entry) and FPL data (FPL-Core-Insights CSVs) flow into
staging sheets. The **Players** sheet (818 rows × 161 cols, ~131k formulas) converts odds into
per-player probabilities for 6 upcoming fixtures (F1–F6), turns them into expected FPL points (XP)
per fixture using the official scoring rules, and blends them into **Total XP**
(`F1 + 0.85·F2 + 0.7·(F3+F4+F5+F6)`). *(That's the legacy workbook; the Python pipeline projects
F1–F`config.HORIZON` and retired the Total XP column — the optimiser now weights the horizon itself.)*
The two PuLP optimisers read Players + GW Teams and print
optimal squads / transfer plans to the console. Results are manually recorded back into
tracking sheets (GW Teams, Historical Expected Points).

The projection horizon is **`config.HORIZON`** (default **17** → a GW3 run projects F1..F17 = GW3–GW19,
for chip-window planning). F1/F2 use real odds; **F3..HORIZON are fully modelled** by the same machinery
(extended from a hard-coded 8 on 2026-09-02 — purely additive, F1–F8 unchanged). Past F8, start-probs are
held flat at F8 and the GBM's `horizon` feature saturates. The **archive stays F1–F8** and the **optimiser
keeps its own `num_fixtures` (default 8)**, so both are unaffected.

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
- Defensive contribution: `P(DC ≥ threshold) = 1 − NORM.DIST(threshold, dc_per_90, SD, TRUE)`; threshold 10 (DEF) / 12 (MID). **Workbook rule:** if <4 games played, use the position average instead. *(The Python pipeline replaced this hard binary with **smooth shrinkage**: a player's own dc90 is blended toward the reliable-population average with weight = nineties ÷ `DC_SHRINK_NINETIES` (=4.0, i.e. @360 min), capped at 1 — so 1 game = 25% own / 75% average, 4+ full games = fully own. Promoted/foreign signings get a face-value prior via `tools/build_dc_prior.py` → `external_dc_prior.csv` (~20 nineties), so they clear the gate regardless. See `config.py` DC_SHRINK_NINETIES.)* *(Since 2026-09-16 the blended rate is also scaled by the player's expected **minutes per start** ÷ 90 before the normal CDF — dc90 is per 90 but 46% of midfield starts end early; see `config.py` DC_MINUTES_PER_START and the `mps` column in `02/03_dc_stats_*.csv`. Same day: F1–F8 DefCon is priced **per fixture** — the per-start rate × a heavy-favourite multiplier from that fixture's win probability (`config.DC_FAVOURITE_CURVE`, defenders only, 1.0 at 0.55 → 0.82 at 0.85) — replacing the flat F1 copy.)*

### 2. Baseline regressions (hardcoded in Players formulas)
For each stat s ∈ {Score1+, Assist, YellowCard, CleanSheet, Concede2+, Concede4+, 3+Saves, 6+Saves}:
`baseline_s(win, opp_win, pos, venue)` — linear model with terms: win, opp_win, pos dummies,
home dummy, diff, |diff|, diff², win×home, opp_win×home, diff×home, pos×diff (+ win×opp_win for saves/CS).
Same coefficient sets are reused for F1 (to derive factors) and F2–F6 (to project).

### 3. Player factors
`factor_s = F1_actual_prob_s / baseline_s(F1 win, F1 opp_win, …)`; if no F1 odds → Fallback Factors sheet.
  *Python (improved mode, 2026-10-08):* the fallback is also used for score/assist when the **match** is priced but the
  player isn't (Betway omitted him) — his F1 score/assist chances are then built from fallback factor × baseline.
  `inputs/fallback_factors.csv` is **rebuilt from the archive** on each `--gw` run (`history.refresh_fallback_factors`):
  each player's median over his last 10 real-odds weeks (all seasons); a value < 0.05 with no history → position median.
  (It used to copy the live master, which let synthetic 1%-price placeholders leak in — e.g. Joelinton assist 0.01.)
Projection for fixture k: `prob_s(k) = factor_s × baseline_s(win_k, opp_win_k, …)`.
F2 uses real odds where available (Clean Sheet, Team Total Goals, Yellow Card via IFERROR; Score 1+ via
Coefficients rows 1–5 model × factor; Score 2+/3+ via threshold ladder on Score 1+).

### 4. Win probability for F3–F6 (no odds yet)
- **Since 2026-09-02 (improved mode):** a rolling odds-implied team rating (`fpl_pipeline/rating.py`,
  weighted least squares on football-data closing prices, 30-day half-life) mapped to three-way
  probabilities per horizon. The Lasso below is the workbook/parity path and the runtime fallback.
- **Personal team views (since 2026-10-17):** `inputs/team_views.csv` nudges a team's rating by `nudge` goals per
  game — 1X2 strength += nudge × 1.3 (log-odds per goal), attack and defence log-rates each += log(1 + nudge/2/1.35).
  Only rows for the current F1 gameweek apply (`rating.load_team_views`), so they go inert after the roll. Applied once
  in `run.py` to the live FixtureRating (never to the training/backtest path), so it moves rating-served weeks and the
  GBM's `rating_win`/`rating_xg` features. Priced weeks (F1, priced F2) are shaded at `TEAM_VIEW_MARKET_WEIGHT`
  (0.5) by `team_views.shade`: win/loss log-odds += 1.3×(n_t−n_o) with the draw kept; conceding rate × exp(s(n_o)−s(n_t))
  → clean sheet, concede 2+/4+, keeper saves; each player's goal/assist rate × exp(s(n_t)−s(n_o)). `--gw` archive
  runs rebuild an unshaded master (raw markets + plain rating) for `update_player_history` / fallback factors. Measured GW7: Leeds +0.12 → Muharemović +0.46 over F3–F8;
  Man Utd −0.10 → Mbeumo −0.54, B.Fernandes −0.59; Brighton −0.08 → De Cuyper −0.59.
- Team: hardcoded Lasso (from `predict_match_win.py`): features from title/releg/top6 probs of both teams + venue.
- Opponent: Coefficients rows 41–46 (`max(0, intercept + b·title + c·opp_title + d·releg + e·opp_releg + f·away)`).

### 5. Expected points per fixture
`XP_pre = P(start) × Σ position-weighted points`:
appearance 2; goals = P(1+)·g + P(2+)·g + P(3+)·g with g = 10/6/5/4 for GK/DEF/MID/FWD
(tail-sum trick ≈ E[goals]·g); assist ·3; yellow −1; clean sheet +4 (GK/DEF, +1 MID);
concede 2+/4+ −2 each in the workbook (GK/DEF) — double the FPL rule; improved mode charges −1 each since 2026-09-01; saves 3+/6+ +1 each (GK); DC 2 pts at position threshold.
Bonus: `P(bonus) = clamp(−0.021039 + 0.023522·XP_pre, 0, 1)`; `XP = XP_pre + 2·P(bonus)`.
`Total XP` (the workbook's `F1 + 0.85·F2 + 0.7·(F3+F4+F5+F6)` blend) was **RETIRED 2026-08-31** — the master no longer carries it; each optimiser now weights the horizon itself (ownership × reliability) into its own `Weighted_Total_XP`.
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


## Model refresh — 2026-10-17
- Projection GBMs retrained on 2025-26 + 2026-27 (`save_projection_models.py`: TRAIN_SEASONS, current-season latest week = early-stopping val, MIN_TREES=20 guard). Unseen-GW7 test: clean_sheet −14%, concede2 −21%, saves3 −25%, score1 −24%, yellow −13% MAE vs the old models.
- Defensive GBMs served as `w·model + (1−w)·raw factor×baseline` (`config.DEFENSIVE_MODEL_WEIGHT`: CS .75, concede2 .65, saves3 .80) in F3–F8 and the unpriced-F2 fallback.
- MID DefCon sd 3.86 (`inputs/dc_params.csv`); `config.YELLOW_CALIBRATION` 0.85 applied in `players.xp_block`.
- Kept after testing: goals/assist longshot curves, bonus model, reliability row, rating maps.
- Coefficient refit WRITTEN (same day): `refit_coefficients.py` holdout is now season-aware (newest season's latest weeks), and `--only baselines` writes just the baseline block. Baselines (gate passed) + `win_pred_f3plus` (own gate passed) written; match-odds and F2-score formulas untouched; prior set kept as `coefficients_workbook.json` (parity). `pred_xp` (display-only `F1 Pred XP`) is excluded from the refit — its target includes non-starters and rescaled it 3.8→1.2. Then: training data rebuilt, GBMs retrained, blend/stack weights re-checked (unchanged). Like-for-like `--gw` comparison: every stat's mean within ~1%, players within ±0.5 XP over F1–F8.
- Comparing outputs: projection GBMs serve only on `--gw` runs; a plain run is factor×baseline only, so compare like with like.
- Raw odds archive: `inputs/raw_odds/<season>/GW<NN>/` (`history.archive_raw_odds`, each `--gw` run) holds the sportsbet files marked real in the provenance manifest, prices exactly as scraped, minus synthetic seed rows. These are the untouched bookmaker prices; the player archive stores adjusted chances. Started GW8 2026-27, for re-fitting the goal/assist longshot curves from scratch.
- Odds-lag test (2026-10-18): recent actual-vs-odds and xG/xA-vs-odds gaps don't predict real goals/assists beyond the odds (cross-season, 3/6-game windows). No form or xG adjustment; re-test ~GW19 (docs/TODO.md).
- Fixture risk (2026-10-23): `inputs/fixture_risk.csv` + `fpl_pipeline/fixture_risk.py` scale a current-GW match's F1 XP by its chance of being played (published master only; archive uses `master_raw`). First use: Liverpool v Brighton x0.95 (Storm Boelo, yellow warning).
- Projection blend OFF (2026-10-23): `config.PROJECTION_BLEND = {}`. Future goals/assists/saves no longer mix in this week's price, which carried this week's fixture forward. Judged on ranking, not average error. Goals stack weight 0.65 re-checked (still best). Re-test ~GW12.
- Probability models scored on SQUARED error (2026-10-23), was absolute: GBM early stopping (`eval_metric="l2"` in save/train_projection_model), the tuning search, both refit gates (`refit_coefficients.py`), and the backtest/training printouts (now RMSE). Squared error is the proper score for a probability and targets the mean, which XP needs. GBMs retrained; DEFENSIVE_MODEL_WEIGHT / PROJECTION_STACK re-checked on squared error + ranking (all within ~1% of best, unchanged). Outcome tests stay on Brier; log loss only as a longshot-curve side check.
