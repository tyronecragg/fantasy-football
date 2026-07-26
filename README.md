# FPL Pipeline

Python replacement for the old `Fantasy Premier League.xlsx` process: bookmaker odds and
FPL data in, per-player expected points (XP) over the next six fixtures out, feeding two
PuLP squad optimisers. Every stage is an inspectable DataFrame snapshotted to `outputs/`.

The workbook itself is retired — kept only as a frozen reference for parity validation.
`PIPELINE_MAP.md` documents how the original workbook worked and how it was replicated.

## Weekly workflow

1. **Scrape** (repo virtualenv not required):
   - `python sportsbet.py` — match/player odds → `sportsbet/*.csv`
   - `python starting_lineups.py` — predicted lineups → updates `inputs/starting_lineups.csv`
     (preserves your manual F1–F6 start probabilities; prints which new players need them)
2. **Edit manual inputs** in `inputs/` as needed:
   - `starting_lineups.csv` — start probabilities per player per fixture
   - `fixtures.csv` — opponent + venue per team for upcoming gameweeks (F2+ need this)
   - `title_odds.csv` / `relegation_odds.csv` / `top6_odds.csv` — season odds per bookmaker
   - `gw_teams.csv` — your squad per gameweek
   - `dc_params.csv` — defensive-contribution SD / average per position
3. **Build**: `python -m fpl_pipeline.run --gw N`
   - snapshots every stage to `outputs/NN_*.csv`, ending in `13_players_master.csv`
   - `--gw N` also upserts the historical archives and refreshes fallback factors
     (re-running the same gameweek replaces rather than appends). Without `--gw` the
     archives are never touched — the run just prints the gameweek the FPL data suggests
4. **Optimise** (needs the repo virtualenv for PuLP):
   - `env/Scripts/python optimisation_gameweek.py` — transfers for the coming gameweek
   - `env/Scripts/python optimisation_full.py` — full-squad builds (wildcard / initial team)
   - both read `outputs/13_players_master.csv` + `inputs/gw_teams.csv` by default; pass an
     `.xlsx` path for the legacy workbook

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

## Tests

`python -m pytest` — unit tests for markets/model/team model, end-to-end improved-mode
behaviour on a synthetic single-gameweek state, history upsert semantics, refit
round-trip (predictions, not raw coefficients — the feature sets are collinear), and the
full workbook parity check.

## Retired

- `extract_fpl_data.py`, `extract_defensive_contributions.py` — ported into `fpl_pipeline/ingest.py`
- `fpl_data/player_name_changes.csv` — merged into `inputs/name_mappings.csv`
- All openpyxl workbook writes in `sportsbet.py` / `starting_lineups.py`
- The modelling scripts (`modelling/`, `strength_modelling/`) remain as historical
  reference; `tools/refit_coefficients.py` replaces their paste-data-in, copy-coefficients-out workflow
