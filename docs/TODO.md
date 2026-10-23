# To-do

Open work for the pipeline. Tick items off (or delete them) as they land, and say where the change is documented.

## Scoring rules not yet modelled (from the FPL rules review, 2026-10-17)
Measured on 33,895 player-weeks (2025-26 + 2026-27, FPL-Core-Insights per-GW stats).

- [ ] **Minutes model: substitute appearances.** Non-starters currently score 0. Active players who don't start come on 28% of the time and average 1.3 pts (≈ +0.6 per non-start for FWD, +0.4 MID, +0.2 DEF; more for genuine rotation players). Per-player P(comes on | not started) from his own history, shrunk to a position average. Undervalues rotation players and bench options today.
- [ ] **Minutes model: starters off before 60.** Every starter gets the 2-pt appearance and the full clean-sheet chance. 6.9% of starts end before 60 min (MID 9.2%, FWD 8.3%, DEF 5.5%) → 1 pt and no clean sheet. Overstates starters ≈ 0.05–0.12 pts. Per-player P(off < 60 | starts); the DefCon minutes data (`mps`, full-90 share) is a starting point.
- [ ] Walk-forward test the minutes model on past weeks before switching it on (score XP vs actual points, as `tools/reliability_check.py` does).
- [ ] **Penalty saves (+5):** keepers save a penalty in 1.5% of starts → ≈ +0.06 pts/game missing.
- [ ] **Discipline / own goals / penalty misses:** red cards (−3), own goals (−2), missed penalties (−2) → ≈ −0.04 DEF, −0.03 FWD, −0.02 MID per start. Fixed position-level adjustments.
- [x] **Vice-captain cover in the optimiser** (done 2026-10-19): armband = captain XP + (chance he doesn't start) × best other starter's XP, in the main LP, hold baseline, gain gate and reported captain (`vice_captain_cover=True`, README › Optimiser tuning). Not yet in the transfer-timing plan or chip_history / league_differentials (they still captain on XP alone).
- [ ] **Formation-aware automatic substitutions in the optimiser:** replace the fixed bench weights (0.25 / 0.05 / 0.01) with each starter's chance of not playing and the real autosub rules (GK for GK; first eligible outfield sub that keeps ≥3 DEF / ≥1 FWD).

## Fixes
- [x] **`tools/refit_coefficients.py` season-aware holdout** (done 2026-10-17): the check now holds out the newest season's latest weeks and fits on everything before. Re-run: refit PASSES (3-week holdout: assist −2.3%, clean_sheet −4.9%, concede2 +1.9%, saves3 −3.7%, score1 −1.3%, yellow +1.7%; 4-week: +3.0 / −3.1 / −1.8 / −5.5 / −0.6 / +0.8%).
- [x] **Refit written** (2026-10-17): baselines + win_pred_f3plus only (`--only baselines`, then `--only win_pred`); training data rebuilt, GBMs retrained, DEFENSIVE_MODEL_WEIGHT / PROJECTION_STACK re-checked (unchanged). `pred_xp` excluded from the refit. Documented in docs/PIPELINE_MAP.md (Model refresh).
- [ ] **Strip the dead `improved=False` branches** (players/markets/team_model). Parity was removed 2026-10-23 and `run.py` always passes improved; the old workbook-exact paths now run only in unit tests.
- [ ] **Before GW9 lineups: review Ajer's F3+ slide.** Andrews called him a "mainstay"; check whether the slide is still justified.
- [ ] **~GW12: re-run the refit** (`tools/refit_coefficients.py`, dry run first); if it writes, repeat the cascade (build_training_data → save_projection_models → weight re-check).

## Scheduled re-checks
- [ ] **~GW9–12: goalscorer longshot curve** (`LONGSHOT_CALIBRATION`, still marked provisional): re-test against actual goals, as for assists. If it changes, add a regime list like `ASSIST_CURVE_REGIMES` first, because archived weeks hold the adjusted chance. Untouched prices are in `inputs/raw_odds/` (scraped from GW8; GW1–7 rebuilt exactly by `tools/backfill_raw_odds.py`).
- [ ] **~GW9–12: assist longshot curve.** The >30% band ran 2.1× over (17 rows); a top trim (>30% ×0.85) tested P = 0.90 better, too thin to act on yet.
- [ ] **~GW10–12: optimiser reliability weights** (`tools/reliability_check.py --season=2026-2027`): longer horizons need more weeks; F7/F8 measurable from ~GW12.
- [ ] **~GW12: bonus model** (`tools/bonus_check.py --season=2026-2027`): keepers look ≈ 25% over-credited this season (flat 0.32/start).
- [ ] **~GW12: DefCon curves** (DEF 0.50→0.65, MID 0.55→0.91), the underdog boost (rejected 2026-10-17, re-test with more data), and the MID spread (raised to 3.86).
- [ ] **~GW12: `YELLOW_CALIBRATION` (0.85).**
- [ ] **~GW12: re-test the projection blend on ranking** (removed 2026-10-23). Compare no blend vs a 30% last-3-week price average vs this week's price, scored on within-position rank match + top-5 pick shortfall (single week and 6-week totals), 2026-27 only. Scratch harness `blend_rank.py` (session scratchpad; port to `tools/` if kept). Also re-check the goals stack weight (0.65).
- [ ] **Every few gameweeks: retrain the projection GBMs** (`tools/build_training_data.py`, then `tools/save_projection_models.py`), then re-check `DEFENSIVE_MODEL_WEIGHT` / `PROJECTION_STACK` with leak-free folds.
- [ ] **Season end: rating maps** (`tools/fit_rating_maps.py`).
- [ ] **~GW19 + season end: do the goal/assist odds lag real performance?** Tested 2026-10-18 (scratch `odds_lag.py`): actual outcome vs logit(odds) + trailing (actual − odds) and (xG/xA − odds) gaps, 3- and 6-game windows, fit on one season and tested on the other (2025-26 GW16–29 + 2026-27 GW1–6, ~1,700 player-games each). No consistent gain; every gap effect straddles zero. Unlike the Aug-2026 real-form test (target = future odds), this targets real outcomes. Re-run with more weeks, and port the script into `tools/` if it's worth keeping.
- [ ] **~GW12: Saka goal/assist odds vs xG/xA.** GW1–6: goal odds about 25% below his xG (typical attacker about 15% above), assist odds about 2× his xA (typical about 1.4×). Net effect ≈ 0; re-check, no hand adjustment for now.

## Weekly habits
- [ ] At each new gameweek: ask whether to carry the team views forward (`inputs/team_views.csv`, current-GW rows only).
- [ ] Before each deadline: any weather/safety threat to a match? Set `inputs/fixture_risk.csv` (rows only apply to their gameweek). PL wind postponements have come under AMBER warnings covering the ground (Everton–Liverpool Dec 2024, Man City–West Ham Feb 2020); none found under yellow only.
