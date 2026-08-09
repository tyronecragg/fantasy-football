# FPL Pipeline

Python replacement for the old `Fantasy Premier League.xlsx` process: bookmaker odds and
FPL data in, per-player expected points (XP) over the next six fixtures out, feeding two
PuLP squad optimisers. Every stage is an inspectable DataFrame snapshotted to `outputs/`.

The workbook itself is retired — kept only as a frozen reference for parity validation.
`PIPELINE_MAP.md` documents how the original workbook worked and how it was replicated.

## Weekly routine (in-season)

The whole loop, in order (N = the upcoming gameweek):

```
git -C fpl_data/FPL-Core-Insights pull        # 1. latest FPL data (players, prices, stats)
python sportsbet.py                           # 2. scrape match + player odds
python starting_lineups.py                    # 3. scrape predicted lineups (keeps your prob edits)
python tools/build_fixtures.py --gw N         # 4. regenerate the F1-F8 fixture window
python -m fpl_pipeline.run --gw N             # 5. build projections + record archives
env\Scripts\python optimisation_gameweek.py   # 6. transfer advice (PuLP, needs the venv)
```

Steps 1–5 use plain Python; only step 6 needs the repo virtualenv. Re-running step 5 for
the same gameweek replaces that week's archive rows, so it's safe to iterate. Omit
`--gw N` for a build that touches nothing (e.g. while player odds are still synthetic
pre-season — never record archives from synthetic odds).

Around the commands:

- **Read the run output**: the `name reconciliation:` block lists any lineup/odds names
  that failed to join (paste verified suggestions into `inputs/name_mappings.csv` and
  re-run); the history lines confirm what was recorded.
- **When there's team news**, before step 5: add injuries/suspensions/sales to
  `inputs/unavailable_players.csv` (and remove recovered players), adjust
  `inputs/lineup_overrides.csv` for start-probability judgement calls, and/or edit the
  F1–F6 probabilities in `inputs/starting_lineups.csv` directly. Editing any input CSV
  in Excel is fine — the loaders repair Excel's ANSI re-saves automatically.
- **After making your transfers**, record the new squad in the `GW{N}` column of
  `inputs/gw_teams.csv` — that's what the next week's transfer optimiser starts from —
  and add each new signing to `inputs/purchase_prices.csv` at the price you paid
  (remove sold players). Owned players are then correctly valued at their FPL sell
  price in the transfer budget.
- **Wildcard / free-hit weeks**: `env\Scripts\python optimisation_full.py` builds a
  squad from scratch instead of from your current team.
- **Occasionally**: refresh the season odds CSVs when title/relegation/top-6 markets
  move materially (they drive the F3–F6 projections), and re-derive `dc_params.csv`
  from `fpl_data/calculate_standard_deviations.py` once a few gameweeks of the new
  season exist.

## File reference

### inputs/ — who updates what

| File | Updated by | Notes |
|---|---|---|
| `season_fixtures.csv` | once per season | full 380-fixture list (long format) |
| `fixtures.csv` | `tools/build_fixtures.py --gw N` | F1–F8 window; hand-edit after generating for postponements/DGWs |
| `title_odds.csv`, `relegation_odds.csv`, `top6_odds.csv` | you, occasionally | one row per team, paste odds into `book_*` columns; blanks ignored, filled columns averaged |
| `gw1_match_odds.csv` | pre-season only | pasted 1X2 + total-goals lines; obsolete once `sportsbet.py` works |
| `starting_lineups.csv` | **curated** (Claude + your feedback), patched by the pre-season tool; in-season `starting_lineups.py` refreshes the roster | XIs are picked with judgement, not the algorithm; teams may carry >11 rows with graded probabilities. `tools/build_preseason_data.py --rebuild-lineups` regenerates algorithmically (discards curation) |
| `lineup_overrides.csv` | you | `Player,start_prob,replaces` — judgement calls applied after XI selection (pre-season tool) |
| `unavailable_players.csv` | you | `Player,reason` — excluded before XI selection (pre-season tool) |
| `gw_teams.csv` | you, weekly | your squad per gameweek; rightmost filled column = current team |
| `name_mappings.csv` | you, from reconciliation suggestions | `type,name,name_cleaned`; single source for all renames |
| `purchase_prices.csv` | you/Claude, on every transfer | what you paid per squad player; drives FPL sell prices (rise banked at half, falls in full) — the transfer optimiser values owned players at sell price, not market |
| `dc_params.csv` | you, once a season | DC threshold SD / average per position |
| `fallback_factors.csv` | pipeline (`--gw` runs / pre-season tool) | never edit; per-player factors on the current coefficient scale |
| `historical_player_data.csv`, `historical_fixture_odds.csv` | pipeline (`--gw` runs) | season-keyed training archives; never edit |
| `historical_expected_points.csv` | you, optional | legacy tracking log, nothing reads it |
| `f2_yellow_card.csv` | vestigial | second-source F2 card odds; empty since that pipeline was retired |

### sportsbet/ — always script-written

`sportsbet.py` (real markets) or `tools/build_preseason_data.py` (synthetic placeholders,
flagged by `SYNTHETIC_NOTE.txt`). Never edit by hand.

### outputs/ — regenerated every run, safe to delete

Numbered snapshots `01`–`12` (roster, DC stats, season probs, team fixture view, market
tables), then **`13_players_master.csv`** — guaranteed the master, it's what the
optimisers read — then `14_name_reconciliation.csv` when issues exist. Parity runs
write to `outputs/parity/` so they never clobber the live master. The backtest writes
`backtest_pairs.csv` here too.

## Tools

| Command | Purpose |
|---|---|
| `python tools/build_fixtures.py --gw N` | regenerate the rolling fixture window from the season list |
| `python tools/build_preseason_data.py` | pre-season bootstrap: estimated lineups, factor rebuild, synthetic odds (real GW1 markets used when `gw1_match_odds.csv` exists) |
| `python tools/backtest_projections.py` | forecast-vs-actual evaluation of the projection machinery against the archives |
| `python tools/refit_coefficients.py [--write]` | refit all regressions from the archives (holdout-gated; see Refitting) |
| `python -m fpl_pipeline.reconcile` | standalone name-reconciliation report |
| `tools/extract_coefficients.py`, `tools/export_workbook_inputs.py` | one-off migration tools (workbook → pipeline); kept for provenance |

## Optimiser tuning

Both optimisers configure via their `__main__` blocks (edit and run):

- **`optimisation_gameweek.py`** (weekly transfers): `max_transfers`, `num_fixtures`,
  `fixture_weights`, `additional_budget` (money in the bank), `bench_weights` /
  `gk_bench_weights` (per-fixture bench value), `force_transfer_out=[names]`,
  `compute_solutions` / `num_solutions_display` (solution pool + frequency analysis
  showing how often each player appears across near-optimal solutions),
  `max_defensive_players_per_team` (GK+DEF cap per club). The module-top `DGW_TEAMS` /
  `DGW_EXTRA_FACTOR` block is the double-gameweek hack: list DGW teams to boost their
  F1 XP (proper DGW support is deliberately deferred).
- **`optimisation_full.py`** (from-scratch squads): `num_fixtures`, `fixture_weights`,
  `bench_weight`, `total_squad_cost`, and `find_multiple=True` with `num_teams` /
  `diversity_method` / `points_tolerance` to generate several distinct candidate squads.

Both accept `excel_file="...xlsx"` to run against the legacy workbook instead.

## Model improvements over the workbook (improved mode, the default)

Documented with rationale in `fpl_pipeline/players.py`; all are disabled in parity mode:

| Improvement | What it fixes |
|---|---|
| F2 model fallback | GW1-style single-gameweek odds: F2 degrades to the F3-style model instead of zeros, upgrading automatically when GW2 odds appear |
| F2 duplicate-odds guard | The old tail(20) bug that copied F1 odds into the F2 files |
| Probability clamping | Unbounded linear regressions — every modelled probability is clipped to [0, 1] |
| Win-pair scaling | Independently predicted win + opponent-win pairs can exceed certainty; scaled to sum ≤ 1 |
| Draw-aware de-margining | Three-way (home/draw/away) margin removal when the scrape has draw odds, instead of 1/odds/1.03 per side |
| Poisson score curves | Smooth P(score 2+/3+) from P(score 1+) instead of step ladders (which had an exact `p == 0.3` branch) |
| Generic F2 score model | The workbook's Coefficients-sheet F2 score formula mixed model families (factor calibrated on one model, applied to another): backtested MAE 0.063 even with perfect odds vs 0.019 for the generic factor × baseline now used |
| Persistence blend | Modelled F2–F6 score/assist/saves probabilities blended with the player's current F1 odds-implied probability at backtested weights (`config.PROJECTION_BLEND`: 0.70/0.85/0.85) — ~10% error reduction on score projections |

### Projection quality (backtested)

`python tools/backtest_projections.py` reconstructs forecast-vs-actual pairs from the
historical archive (what would the model have predicted at GW M for GW M+k, vs the
odds-implied probability that materialised) — ~124k pairs across six stats on GW16–29
of 2025-26. Headlines: the factor × baseline machinery beats both persistence and
position-mean baselines for every stat at every horizon; score projections sit at MAE
≈ 0.019 (F2) to 0.027 (F5) before blending; the oracle-odds decomposition shows the
remaining error splits roughly 60/40 between factor drift and win-prediction error, so
a horizon-aware win-pred refit is the next meaningful gain once more gameweeks are
archived. The backtest measures the pre-blend machinery; pairs land in
`outputs/backtest_pairs.csv` for deeper slicing.

## Season rollover

1. `git -C fpl_data/FPL-Core-Insights pull`, then set `SEASON` in `fpl_pipeline/config.py`.
2. Refresh the season inputs: paste the new fixture list into `season_fixtures.csv` and
   run `tools/build_fixtures.py --gw 1`; rewrite the season-odds CSVs with the new
   20-team list; reset `gw_teams.csv` (record your squad from GW1); prune
   `unavailable_players.csv` and `lineup_overrides.csv` of last season's entries.
3. Before real markets/team news exist: `python tools/build_preseason_data.py` builds
   estimated starting lineups (last-season minutes + launch prices; injured/sold
   players in `inputs/unavailable_players.csv` are excluded before XI selection, then
   judgement calls from `inputs/lineup_overrides.csv` — Player, start_prob, optional
   replaces — are applied)
   and synthetic GW1/GW2 odds from the model, and rebuilds `fallback_factors.csv` on
   the current coefficient scale from the archive. If `inputs/gw1_match_odds.csv`
   exists (pasted 1X2 + per-team total-goals lines), GW1 win/team-goals/clean-sheet
   markets use those real odds (Poisson fits on the goals lines) instead. Player
   markets stay synthetic — never record `--gw` archives until those are real too.
4. Run the scrapers once fixtures/lineups are published (they overwrite the synthetic
   files); the name-reconciliation stage will flag any unmatched bookmaker names for
   promoted teams — add the verified ones to `inputs/name_mappings.csv`.
5. The historical archives are keyed by (Season, Gameweek) so new-season `--gw` runs
   never collide with last season's rows. `dc_params.csv` and `fallback_factors.csv`
   carry over and self-refresh as data arrives.

Parity validation stays pinned to the 2025-26 workbook (`config.PARITY_SEASON`): it
reads its roster/DC data from the workbook's own sheets and its inputs/odds from the
frozen `parity_reference/` snapshot, so it keeps passing regardless of season, weekly
scrapes, or upstream FPL-data rewrites. The test suite uses the same frozen data.

## Validation

`python -m fpl_pipeline.run --validate` runs **parity mode**: all improvements and
input-mutating side effects off, output compared cell-for-cell against the workbook's
cached values (161/161 columns clean at migration). Parity is only meaningful while
`inputs/` and `sportsbet/*.csv` are unchanged since the workbook last calculated —
gameweek runs mutate the archives, after which the frozen workbook is a stale reference.

## Refitting the model

`python tools/refit_coefficients.py` refits every regression from
`inputs/historical_player_data.csv` and `inputs/historical_fixture_odds.csv` (which the
pipeline now maintains automatically). Dry run by default — shows n / R² / changed
coefficients; `--write` regenerates `fpl_pipeline/data/coefficients.json`, backing up the
workbook-extracted original to `coefficients_workbook.json` (parity mode then uses the
backup automatically). Feature construction is shared with serving code, so train and
serve cannot drift. The bonus model is carried over unchanged (refitting it needs actual
bonus-point outcomes, which the archives don't hold).

**A holdout projection check gates every `--write`** and refuses if the candidate
baselines degrade F2–F6 forecast accuracy (override with `--force`, but don't). This is
not theoretical: the baselines are consumed as a *ratio* (factor = odds ÷ baseline at
week M, projection = factor × baseline at week M+k), so a refit that improves same-week
fit can still wreck projections — refitting on the 2025-26 archive (13 gameweeks)
degraded holdout projection MAE by 100–225% across every stat, and the tool correctly
refused it. Revisit once the archive holds substantially more gameweeks, and expect the
refit objective to need regularisation toward ratio stability, not just OLS fit.

## Names

All player/team renames live in `inputs/name_mappings.csv` (`type,name,name_cleaned`) —
the single source used by ingest, the season-odds team cleanup, and both scrapers.
Accent transliteration stays algorithmic in `starting_lineups.py`.

Every improved-mode run includes a **name reconciliation stage** (`fpl_pipeline/reconcile.py`,
also runnable standalone as `python -m fpl_pipeline.reconcile`): it reports every lineup
player, odds player, and odds team that fails to join to the FPL roster — failures that
would otherwise silently zero out start probabilities or drop odds — plus XI starters
with no attacking odds at all, and prints ready-to-verify `name_mappings.csv` suggestions
(accent/case, unique-surname, fuzzy). Suggestions are candidates, never auto-applied:
surname matches can be wrong players. Input CSVs may be edited in Excel — the loaders
tolerate Excel's ANSI re-saves, repair double-encoded accents ('TourÃ©' → 'Touré'), and
heal the files back to UTF-8 automatically.

## Architecture

`fpl_pipeline/` modules, in pipeline order: `config` (paths, constants, blend weights) →
`ingest` (FPL data, odds CSVs, inputs; encoding-tolerant) → `names` (mapping table) →
`markets` (odds → probabilities) → `team_model` (season probs, fixture view, win
predictions) → `model` (baselines, factors machinery, Poisson curves, scoring — all
coefficients from `data/coefficients.json`) → `players` (the master build; every
improvement documented in its docstring) → `reconcile` (name-join audit) → `history`
(season-keyed archive upserts) → `validate` (workbook parity) → `run` (orchestration),
with `io_utils` (snapshots, tolerant CSV reading) underneath. `PIPELINE_MAP.md` maps
each module back to the workbook sheets it replaced.

## Tests

`python -m pytest` — unit tests for markets/model/team model, end-to-end improved-mode
behaviour on a synthetic single-gameweek state, history upsert semantics, refit
round-trip (predictions, not raw coefficients — the feature sets are collinear),
name reconciliation and Excel-encoding repair, the fixture-window builder, and the
full workbook parity check.

## Retired (deleted; recoverable from git history)

- `extract_fpl_data.py`, `extract_defensive_contributions.py` — ported into `fpl_pipeline/ingest.py`
- `fpl_data/player_name_changes.csv` — merged into `inputs/name_mappings.csv`
- All openpyxl workbook writes in `sportsbet.py` / `starting_lineups.py`
- `modelling/` and `strength_modelling/` — `tools/refit_coefficients.py` replaces their
  paste-data-in, copy-coefficients-out workflow
- `odds_data_outputs/`, `starting_lineups/data.csv`, old workbook copies — dead pipeline
  leftovers; `outputs/` and `.idea/` untracked (regenerable / IDE config)
