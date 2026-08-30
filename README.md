# FPL Pipeline

Python replacement for the old `Fantasy Premier League.xlsx` process: bookmaker odds and
FPL data in, per-player expected points (XP) over the next eight fixtures out, feeding two
PuLP squad optimisers. Every stage is an inspectable DataFrame snapshotted to `outputs/`.

The workbook itself is retired — kept only as a frozen reference for parity validation.
`docs/PIPELINE_MAP.md` documents how the original workbook worked and how it was replicated.

## Weekly routine (in-season)

The whole loop, in order (N = the upcoming gameweek):

One-command version (recommended) — `weekly_update.py` runs the steps below with
preflights (Excel file locks, Sportsbet VPN check) and pauses between phases for
the lineup curation:

```
python weekly_update.py                       # phase 1: FPL data + odds + FFS staging
                                              #   ... curate lineups with Claude ...
python weekly_update.py --resume --gw N       # phase 2: fixtures + projections + optimiser + chip log + radar
```

> **Manual re-run of the chip log.** Phase 2 runs `tools/chip_history.py --radar` with the **defaults**
> (1 free transfer, £0 bank), so its wildcard/free-hit deltas — and the radar's Free Hit column — are
> only right when those match. Re-run it by hand each week with your real numbers (it overwrites that
> GW's row and refreshes the radar):
> `env\Scripts\python tools/chip_history.py --gw N --free-transfers <FT> --bank <BANK> --radar`
> The `--radar` block prints a forward F1..F8 chip radar — each fixture valued as a full-weight
> one-off (Bench Boost / Triple Captain / Free Hit) — so a chip spike in the 8-fixture window is
> visible in advance, not only when it becomes the next fixture. The Free Hit column uses an
> **accumulating** free-transfer baseline (`FT` at F1, +1 per week, capped at 5), so it decays
> further out (by then you'd have transferred into those fixtures anyway).

Or step by step:

```
git -C fpl_data/FPL-Core-Insights pull        # 1. latest FPL data (players, prices, stats)
python tools/betway.py                        # 2. scrape odds (preferred — see below)
python starting_lineups.py                    # 3. stage FFS predicted lineups (prints diff vs curated)
python tools/injury_check.py                  # 3a. curated probs vs FPL's own availability flags
                                              # 3b. curate start probabilities with Claude (see below)
python tools/build_fixtures.py --gw N         # 4. regenerate the F1-F8 fixture window
python -m fpl_pipeline.run --gw N             # 5. build projections + record archives
env\Scripts\python optimisation.py   # 6. transfer advice (PuLP, needs the venv)
```

Archive safety: while `sportsbet/SYNTHETIC_NOTE.txt` exists (pre-season placeholder
player odds), `--gw` records **match odds only** — player history and the fallback-factor
refresh are withheld so synthetic player prices can't poison the trailing-median factors.
Match odds are still archived because they're often real even when player markets are
closed (GW1 2026-27 was hand-pasted) and are unbackfillable. Delete the note by hand once
real Betway odds and Ladbrokes cards have landed; `--force-archive` records everything.

### Rolling to a new gameweek (before Betway prices it)

When a gameweek finishes you move the F1–F8 window forward. But **F1, unlike F2, has no model
fallback** — the pipeline expects real odds for it — so if you shift the window before bookmakers
open the new gameweek's markets, every F1 market goes empty. The fix is to **seed synthetic F1**
from the last played week's actuals, then let Betway overwrite it market-by-market as real prices
arrive. One command does the roll:

```
python tools/roll_gameweek.py --gw N            # N = the new gameweek (becomes F1)
```

**Before you run it**

1. **Outright odds** — if you have fresh title/relegation/top-6 numbers, update
   `inputs/title_odds.csv`, `inputs/relegation_odds.csv`, `inputs/top6_odds.csv` (they feed the
   synthetic win probabilities via `season_probs` → `win_pred`). They move slowly week to week, so
   most gameweeks you can skip this. Format is `Team,book_1,book_2,…`; `season_probs` averages the
   `book_` columns (skipping blanks), so a single consensus column per team is fine. **Or scrape them:**
   `python tools/betway_outrights.py` writes all three files from Betway's outright markets (title/
   top-6 raw odds; relegation via the per-team Yes/No markets with a main-market fallback). Betway's
   ~8% outright margin matches `MARGIN_SEASON=1.08`, so raw odds de-margin correctly.
2. **Lineups** — curate `inputs/starting_lineups.csv` for the new gameweek's team news (the synthetic
   player markets are built for whoever is in the XIs there). `python tools/build_preseason_data.py
   --lineups-only` patches `lineup_overrides.csv` onto the lineups without touching odds.
3. **Archive** — GW N-1 must already be archived on real odds (`python -m fpl_pipeline.run --gw N-1`
   was run), because the synthetic factors are read from it. Normal weekly flow guarantees this.

**What `roll_gameweek.py` does** (each step is a subprocess; a failure stops the roll)

1. `tools/build_fixtures.py --gw N` — regenerates `inputs/fixtures.csv` as the GW N…N+7 window
   (F1 = GW N). Columns are labelled by **absolute** gameweek (`GW{N} Opponent`, …). Hand-edit this
   file afterwards for postponements / double-gameweeks before projecting.
2. `tools/build_synthetic_gw.py --gw N` — for each player, reads their **GW N-1 actual market factor**
   from the archive (`factor = archived P(stat) ÷ baseline at GW N-1's win probs` — GW-specific, not
   the all-history median `rebuild_factors` uses) and re-casts it onto GW N: `P = factor × baseline`
   at GW N's win probs (from `win_pred` on the current outright odds). It writes every **F1-only**
   market — win/draw/win, goalscorer, 2+ goals, assist, bookings, saves, clean sheets, team goals — to
   `sportsbet/*.csv`, with the usual margins so de-margining recovers the intended probabilities, and
   drops `sportsbet/SYNTHETIC_NOTE.txt`. Synthetic seeds **F1 only**; the `_f2` files are cleared
   (header-only) so F2+ derive off F1 (projection model / `factor × baseline`), and real Betway F2
   odds repopulate them when priced.
   - `--factor-gw M` overrides which archived gameweek supplies the factors (default `N-1`). Use it
     if GW N-1 wasn't archived on real odds (e.g. it was itself synthetic) — point at the last week
     that was.

**After you run it**

1. `python -m fpl_pipeline.run` — projects GW N on the synthetic markets (no `--gw`, so nothing is
   archived). Sanity-check the top players.
2. `tools/betway.py` — run it once the real markets open. It **upserts each market Betway-authoritative
   per match**: real odds replace synthetic for every match Betway prices, a player Betway omits from a
   priced match goes NA (it didn't rate him), and **wholly-unpriced matches keep synthetic** — so real
   odds override automatically as they arrive, no manual merge. ("Priced team" = a majority team of some
   priced match, so a loan player can't falsely flag his parent club.) For F1 player-prop gaps in
   still-unpriced matches, `tools/fill_synthetic_gaps.py` backfills from the prior GW's archived F2.
   - **Stray earlier-gameweek fixture?** Betway sometimes still lists a not-yet-played match from a
     past gameweek; because the F1/F2 split is by kickoff order (first 10 = F1, next 10 = F2), a stray
     earlier match would sort into F1 and shift everything by one. Drop it with
     `--skip-fixture "TERM"` (a team name or any text in the `Home vs. Away` label, repeatable) —
     e.g. `tools/betway.py --skip-fixture "Wolves"`. The remaining fixtures then split cleanly (first
     10 → F1, next 10 → F2), and the built-in crosscheck against `fixtures.csv`'s GW N / GW N+1
     pairings **warns** if any block still doesn't line up, so a missed or wrong skip is caught.
3. `python -m fpl_pipeline.run --gw N` — project and archive. While `SYNTHETIC_NOTE.txt` exists the
   guard records **match odds only** (player history is withheld so synthetic prices can't poison the
   trailing-median factors). **Delete `sportsbet/SYNTHETIC_NOTE.txt` by hand** once real Betway player
   odds (and Ladbrokes cards) have landed, then re-run `--gw N` to archive player history too;
   `--force-archive` records everything regardless.

Next gameweek it's just `python tools/roll_gameweek.py --gw N+1`. The in-season builder mirrors the
pre-season `build_preseason_data.py`; use that one only before the season starts.

**Transfer-market check, every curation pass.** Before committing to a player, research
whether he is being sold or is close to it — a move away from the league, or to a club where
he is not first choice, destroys his value regardless of what the odds and start
probabilities say. Bookmakers keep pricing players who are about to leave, and FFS predicted
XIs lag transfer news by days, so neither input catches this. **Check the squad you already
own too, not just incoming picks** — a player bought last week can become worthless mid-week,
and the pipeline will keep projecting points for him until his start probability is edited by
hand in `inputs/starting_lineups.csv`.

Step 3b: the FFS scrape only **stages** its output — predicted XIs to
`inputs/ffs_predicted_lineups.csv` (with a printed diff vs the curated lineups, accent-folded
so only real disagreements show) and the per-team write-ups (next match, out/doubts with FFS
percentages, bans, latest-news paragraphs) to `inputs/ffs_team_news.md`. It never overwrites
the curated file. The curated `starting_lineups.csv` (graded probabilities, the pipeline's
actual input) is maintained weekly in conversation with Claude, who folds together the FFS
predictions, the staged write-ups, wider team news, and your feedback files. FFS is a signal,
not an authority.

**Cross-reference beyond FFS.** `inputs/curation_sources.md` is the trusted-source panel:
a fixed base of three predicted-XI aggregators (FFS + RotoWire + All About FPL) used for
every team, an injury/availability layer, and club-specific escalation reserved for the hard
teams (promoted sides, new managers, injury clouds) where the base panel disagrees. Trust in
each source is *provisional* — the **source-history ledger** (`tools/source_history.py`) freezes
each source's predicted XI per gameweek and scores it against the actual lineups once they play,
so trust is a hit rate, not a reputation. It's wired into `weekly_update.py`: phase 2 freezes
FFS + our own curated XI automatically and scores last week's frozen predictions. See the
"Earning trust" section of `inputs/curation_sources.md`.

Steps 1–5 use plain Python; only step 6 needs the repo virtualenv. Omit `--gw N` for a
build that touches nothing.

**Re-running `--gw N` is idempotent, not additive.** Both archives upsert — match odds on
`(Season, fixture pair)`, player history on `(Season, Gameweek)` — so a second run for the
same gameweek *replaces* those rows rather than appending duplicates. Iterate freely: after
a lineup curation pass, after fresher odds, after a name-mapping fix. The values recorded
are whatever the inputs say at that moment, so the last run for a gameweek wins.

That asymmetry is deliberate. Match odds are **unbackfillable** — once a fixture kicks off
its pre-match prices are gone for good, and they train the win-probability model — so record
them early even if the squad picture is unsettled. Player history can always be rebuilt from
inputs that still exist, so it is withheld whenever any player market is synthetic.

Around the commands:

- **Read the run output**: the `name reconciliation:` block lists any lineup/odds names
  that failed to join (paste verified suggestions into `inputs/name_mappings.csv` and
  re-run); the history lines confirm what was recorded.
- **When there's team news**, before step 5: zero injured/suspended/sold players and set
  start-probability judgement calls directly in `inputs/starting_lineups.csv` (or in
  `inputs/lineup_overrides.csv`), then apply the overrides
  with `python tools/build_preseason_data.py --lineups-only` (patches the lineups and
  touches nothing else — safe after real odds are scraped; the tool's full mode would
  overwrite real odds with synthetic ones). Or edit the F1–F8 probabilities in
  `inputs/starting_lineups.csv` directly — all eight fixtures are curated (F7/F8 added
  2026-08-21; a file with only F1–F6 falls back to the F6 value for F7/F8). Editing any input CSV in Excel is
  fine — the loaders repair Excel's ANSI re-saves automatically.
  **After any curation pass, check the team sums** (`groupby('Team')['F1'].sum()`): **F1 must be
  exactly 11** (the GW1 XI is known — 11 players start), and **no column may exceed 11** (>11
  starters is impossible). F2 onward *may* sit below 11 — that is CORRECT when the missing minutes
  belong to a player we can't credit yet: a brand-new signing not in the FPL data (Estupiñán at
  Villa, Enciso at Ipswich), or one who may leave. Never pad the gap onto a different player just
  to reach 11. Normalisation is certainty-preserving, so 1.0 declarations never move and an
  inflated total is taken out of the *graded* players — a believed 0.80 was once consumed as 0.56
  because its team summed to 12.25.
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

## Longshot calibration (PROVISIONAL — active in improved mode)

`config.LONGSHOT_CALIBRATION` shrinks over-stated long prices on the player attacking
markets (goalscorer, 2+ goals, assists). Applied in `markets.py::longshot_calibrate`, right
after `implied()` converts odds to probabilities — **the scraped CSVs stay raw**, so a
different curve can be fitted later without re-scraping. Off in parity mode, since the
workbook applied a flat margin. `LONGSHOT_CALIBRATION = None` disables it.

**Why.** `MARGIN_PLAYER = 1.05` removes the same 5% at every price. Two measurements say
that is wrong in a specific way. From prices: the total load on Betway's goalscorer market
is ~3.5x (`tools/margin_goals.py`, identified by linearity of expectation against the
clean-sheet market, which de-margins exactly). From outcomes: the pipeline's own
probabilities, checked against what happened (`tools/calibration.py`, 2025-26 GW16–29,
minutes ≥ 60) came out —

| projected | actual | ratio |
|---|---|---|
| 25–35% | 27.3% | 1.08× |
| 18–25% | 18.7% | 1.14× |
| 12–18% | 10.3% | 1.46× |
| 8–12% | 4.2% | 2.38× |
| 5–8% | 2.8% | 2.42× |
| 2–5% | 0.8% | 5.23× |

Favourites are calibrated; longshots are overstated 2.4–5×. The two sides triangulate —
prices gave the total, outcomes gave the distribution.

**Control that matters**: filter to `minutes >= 60`. A projected player who never came on
cannot score, so unfiltered the bias reads 1.93× when it is really 1.41× — half of it was
start probability, not margin.

**Gate passed**: fitting on GW<23 and scoring GW≥23 improved Brier by 1.2% and log loss by
2.2%, with mean prediction moving 15.6% → 12.1% against an actual 12.3%.

### Still to do before trusting this

1. **F2–F8 propagation — TESTED, acceptable, one caveat** (`tools/calibration.py`,
   `horizon()`). The concern was that correcting F1 rescales every downstream fixture by the
   same multiplier, since factors are `probability / baseline` and future fixtures are
   `factor × baseline`. Measured, the bias is stable across horizons — overall 1.41 / 1.43 /
   1.36 / 1.38 / 1.31 / 1.38 for F1–F6, same shape in every band — so one correction is
   defensible. **But F1's longshot band is the most biased (2.64×) while F2–F6 sit at
   1.46–1.73×**, so applying F1's curve uniformly slightly OVER-corrects distant fixtures'
   fringe players. Small, partly noise (fewest observations in those cells), but directional.
2. **Refit on deadline-day odds — now UNBLOCKED** (see #4: player-odds archiving is live as of
   2026-08-30). The current curve comes from archive-era projections under a different pipeline
   state and odds sources, and prices carry more margin four days out than at the deadline. Do
   the refit once several gameweeks of deadline-day odds accrue (GW5+).
3. **Only one season, 14 gameweeks, 1,759 rows.** Enough to establish the shape, thin for
   fitting precise knots. The 35–50% bucket already breaks the monotone pattern (1.47×,
   n=81) and is probably noise.
4. **Player odds ARE now archived — RESOLVED 2026-08-30** (was task #25). Cards went real via
   Ladbrokes, so every scraped player market reports `is_real=True` and `guard_synthetic_archive()`
   returns `all` — a `--gw` run records player history, so the refit raw material now accumulates
   (`historical_player_data.csv`, ~22 MB). The old `SYNTHETIC_NOTE.txt` gate is gone; the guard is
   driven purely by the provenance manifest and self-clears. (Was: history withheld while booking
   odds were synthetic, "which they permanently are" — no longer true.)
5. **`LONGSHOT_FLOOR = 0.15`** exists because the raw fit produced ×0.00 on the lowest
   bucket — no one in that training cell scored, so the curve concluded a 4% player is
   impossible. The floor is a guard, not a measurement; Bayesian shrinkage toward the
   neighbouring bucket would be better.

## Open questions

Things known to be provisional, with what would settle them:

- **DefCon vs fixture difficulty** — *closed 2026-08-17, no change made.* Defensive
  contributions do scale with possession relative to a team's own norm (−0.006/pt, t = −5.5;
  15 points above norm → 0.91×), but **not** with bookmaker odds under any specification.
  Out-of-sample (`python tools/defcon_vs_odds.py --forecast`) the adjustment makes DC
  forecasts *worse* (−0.39% MAE), and even with perfect possession knowledge the ceiling is
  +0.85%. The effect is ~1% of match-to-match variance — significant, not useful. `dc90`
  stays flat.

- **The DefCon prior weight** — `DC_PRIOR_CAP_MINUTES = 1710` caps last season's *weight* at
  19 matches (the rate itself uses all 38): pure prior at GW1, ~50/50 after 19 current
  matches. Whether that is the right relevance weight is empirical — backtest once 2026-27
  minutes accumulate.

- **DefCon shrinkage for thin samples** (improved mode, 2026-08-21) — a player's own DC
  hit-probability is blended with the reliable-population average in proportion to his
  evidence: weight = nineties / `DC_SHRINK_NINETIES` (4), capped at 1, and **zero below
  `DC_SHRINK_MIN_NINETIES` (0.65 nineties ≈ 59 min)** — a brief cameo is not evidence. So
  0.65 nineties = 0.65/4 own + 3.35/4 average, one full match = 0.25 own, four or more = own,
  under ~59 min = the average. Replaces the old hard
  cliff (own rate at ≥4 nineties, otherwise the average). Blended in probability space so
  the zero-evidence fallback stays the population's expected probability. Parity keeps
  the cliff. Moved ~60 partial-sample players on the live data (e.g. Pinnock 3.5 nineties at
  11.9/90: 0.32 → 0.63), none of the reliable or zero-evidence ones.

- **Bonus points — ODDS-ANCHORED model, LIVE** (`config.BONUS_ODDS_MODEL=True`, improved mode;
  built 2026-08-22, replacing the old flat `P(bonus)=-0.021+0.0235*XP_pre` which under-allocated
  total bonus **2.24x** and, being a straight line, handed fodder bonus while starving the elite).
  `model.bonus_points_odds`: each team's bonus pot = `P(win)*5.62 + P(draw)*3.24 + P(lose)*0.78`
  (measured 2025-26 per-team means; match mean 6.41, winner's share ~88%), split within the team in
  proportion to **XP above the 2-pt appearance floor** (goalscorers/clean-sheet keepers take most,
  benched ~0). Validated on GW1 2026-27: **total 64.2** (was 28.1; reality ~64), per-team pots track
  win prob (Arsenal 4.93 hosting Coventry → Coventry 1.48 away at Arsenal, a 3.3x spread ≈ the
  measured 4x), Haaland's bonus 0.21→0.53. Parity unaffected (flag is improved-only). Revert with the
  one-line flag. RESIDUAL / to refine, gated through `tools/backtest_projections.py`: (1) the
  within-team split is proportional-to-performance-XP, a reasonable but unvalidated shape — real BPS
  concentrates harder on the single top performer (3/2/1); a concentration power or a direct BPS
  model would be sharper. (2) constants are 2025-26; re-measure as 2026-27 accrues. (3) draw
  home/away asymmetry (2.96/3.51) was averaged to 3.24.

- **The synthetic assist ratio — low materiality now.** Used in `betway.py` only to fill
  fixtures Betway has NOT priced assists for; it recalibrated unstably early (0.836 → 1.215 in a
  day) and now falls back to a 1.132 convention. Betway prices assists directly for almost every
  fixture (e.g. this GW 357/361 real, ~4 synthetic), so the ratio touches only the rare unpriced
  fixture — the instability barely reaches XP. Still worth measuring per market source if the
  synthetic-fill count grows.

- **`SYNTHETIC_NOTE.txt` — RETIRED (2026-08-30).** The file no longer exists and nothing reads it.
  The `--gw` archive guard is now driven entirely by the provenance manifest (`provenance.is_real`
  per player market, in `guard_synthetic_archive()`) and self-clears when the last market goes real
  — which it now has (cards real via Ladbrokes; all eight markets real). `tools/odds_status.py`
  shows the state.

- **The saves calibration factor and `CONVERSION = 0.30`** — see the odds-sources section.
  The 3.80× multiplier is not purely bookmaker overround and is not yet explained.

## File reference

### inputs/ — who updates what

| File | Updated by | Notes |
|---|---|---|
| `season_fixtures.csv` | once per season | full 380-fixture list (long format) |
| `fixtures.csv` | `tools/build_fixtures.py --gw N` | F1–F8 window; hand-edit after generating for postponements/DGWs |
| `title_odds.csv`, `relegation_odds.csv`, `top6_odds.csv` | you, occasionally | one row per team, paste odds into `book_*` columns; blanks ignored, filled columns averaged |
| `gw1_match_odds.csv` | pre-season only | pasted 1X2 + total-goals lines; obsolete once `tools/betway.py` works |
| `starting_lineups.csv` | **curated** (Claude, weekly: FFS staging + news + your feedback) | the pipeline's actual start-prob input; graded probabilities, teams may carry >11 rows; never overwritten by scrapers. `--rebuild-lineups` regenerates algorithmically (discards curation) |
| `ffs_predicted_lineups.csv` | `starting_lineups.py` | staged FFS predicted XIs — curation input only, nothing reads it directly |
| `source_history/predicted_xis.csv` | `tools/source_history.py` (weekly) | each source's frozen predicted XI per GW (`Season,Gameweek,Source,Team,Player`) — scored against actual lineups to earn source trust |
| `ffs_team_news.md` | `starting_lineups.py` | staged FFS write-ups (outs, graded doubts, bans, news paragraphs) — weekly curation reading, nothing reads it directly |
| `lineup_overrides.csv` | you | `Player,start_prob,replaces` — judgement calls applied after XI selection (pre-season tool) |
| `gw_teams.csv` | you, weekly | your squad per gameweek; rightmost filled column = current team |
| `name_mappings.csv` | you, from reconciliation suggestions | `type,name,name_cleaned` = raw spelling → canonical; applied to the roster, the odds files AND `starting_lineups.csv`. Never add a reversed (canonical → raw) row — it un-maps canonical names. |
| `purchase_prices.csv` | you/Claude, on every transfer | what you paid per squad player; drives FPL sell prices (rise banked at half, falls in full) — the transfer optimiser values owned players at sell price, not market |
| `season_odds_corrections.csv` | you/Claude, rarely (optional) | `market,Team,corrected_odds,reason` — overrides market season odds that don't reflect footballing strength (e.g. Man City relegation odds pricing points-deduction legal risk); improved mode only |
| `dc_params.csv` | you, once a season | DC threshold SD / average per position |
| `fallback_factors.csv` | pipeline (`--gw` runs / pre-season tool) | never edit; per-player factors on the current coefficient scale |
| `historical_player_data.csv` | pipeline (`--gw` runs) | season-keyed training archive, full F1–F8 forecast block; never edit |
| `historical_fixture_odds.csv` | pipeline (`--gw` runs) | per-gameweek match + season odds; never edit. Records more than the pipeline consumes (draw odds, `Gameweek` stamp, per-team clean-sheet and over-1.5/over-3.5 goal odds) because odds can't be backfilled — see task #19. Rows before 2026-27 predate those columns and carry NA, and the archive starts ~GW12 2025-26, so it describes the mid-season regime only |
| `historical_expected_points.csv` | you, optional | legacy tracking log, nothing reads it |
| `f2_yellow_card.csv` | vestigial | second-source F2 card odds; empty since that pipeline was retired |

## Odds sources

**`python tools/betway.py` is the preferred scraper.** Betway South Africa serves plain
`requests` with no VPN, no TLS impersonation, no browser and no session token — and it
prices **Player 1+ Assists**, which Sportsbet never did. It writes the same
`sportsbet_*.csv` filenames the pipeline already reads, so `ingest.py` and the
reconciliation stage need no changes.

| flag | effect |
|---|---|
| *(none)* | discover PL fixtures, fetch every market, derive the gaps, write |
| `--limit N` | only the first N fixtures |
| `--dry-run` | report what would be written, change nothing |
| `--no-fill` | write only what Betway priced; derive nothing |
| `--har FILE` | parse a saved capture instead of fetching |

**Two-stage by design.** First it writes whatever Betway actually prices. Then
`fill_gaps()` derives what Betway hasn't published yet *from the real odds it just
fetched* — which beats regenerating synthetically, because each derivation is anchored to
real prices for the same fixtures:

- **2+ goals** — a derivation, not an estimate: P(2+) follows from P(1+) through the same
  `model.poisson_score2` curve the projections already use.
- **Assists** — where Betway prices both assists and goalscorer, the real assist:score
  ratio is measured and applied to fixtures still missing assists. On 2026-08-17 that
  calibrated to **0.841** from 36 real pairs, against the **1.132** convention constant
  measured pre-season from another book — a 35% difference, so prefer the calibration.
- **Goalkeeper saves** — derived by `derive_saves()` from the shots-on-target ladders,
  de-margined against a market anchor. **Betway carries no saves market at all** (confirmed
  2026-08-17 against every market group), but it prices ~43 players per match for 1+…4+
  shots on target. Those are unusable raw: summing P(X≥k) gives Arsenal **23.2** expected
  shots on target against a league average near 5, because ~135 independent yes/no
  selections compound an enormous overround.

  So the sum gets anchored. Clean-sheet prices give expected goals conceded directly
  (P(CS) = e^−λ), and total match shots on target should be total goals ÷ conversion rate.
  The ratio of the naive ladder sum to that anchor **is** the margin — measured at
  **3.80×** for Arsenal v Coventry — and dividing it out leaves Arsenal 6.1 and Coventry
  2.3 shots on target. Then `saves = shots faced − goals conceded`, and 3+/6+ prices come
  off a Poisson.

  The ladders contribute what the anchor alone cannot: **how the shots split between the
  sides**. A team's share of the shots is not its share of the goals — sides differ in
  shots needed per goal — and the market prices that difference.

  **The 3.80× is a multiplier, not a 3.8% margin — and not purely overround.** A yes/no
  prop carries ~5%, nowhere near 280%, so the factor is absorbing something unexplained
  (prices behaving as if conditional on starting, or a ladder extending past 4+).
  `CONVERSION = 0.30` (`--conversion` to override) and the factor are both **provisional**
  — TODO: refine once odds have been collected across several gameweeks and real save
  counts can be regressed against the estimates. Saves are **upserted by team**, so
  fixtures without shots-on-target markets keep their placeholder rather than being wiped.

**Two market shapes, and the trap in the second.** Some markets name the threshold
(`Player 1+ Assists`); the 2+ and 3+ prices instead live in *ladder* markets
(`Player Goals (Incl. Overtime)`, `Player Assists (Incl. Overtime)`) where the threshold is
a **suffix on the selection** — `"Magalhaes, Gabriel 2+"` — and `sbv` is empty. There is no
`Player 2+ Goals` market despite the 1+ one being named that way. `LADDERS` handles these;
players arriving from both shapes are de-duplicated on (player, match).

**Fetch each market group by name.** A blank `marketGroupId` returns a truncated view —
134 markets when `MarketGroupings/group-names` reports ~380, with Player Specials alone
holding 101. `GROUPS = ("Player", "Goals", "Team", "Main")` yields 113 relevant markets and
~1,225 selections per fixture against 1,099 before. Player props were the ones going
missing.

Markets Betway hasn't priced keep their existing file untouched, so placeholders survive
and are replaced one market at a time as bookmakers publish.

**Check before concluding a player is "not in the FPL roster" from the reconciler's silence** —
though `_suggest()` now matches tokens in **any position** (2026-08-19, task #24 done), not just
the last, so it resolves `Jair Cunha` → `Jair Paula da Cunha Filho`, `Julio Soler` →
`Julio Soler Barreto`, and surname-first / reordered names structurally. Two routes, each needing
a unique hit: an **abbrev** match (one name's tokens are a subset of the other's — handles the
long-form and reordered cases) or a shared **surname** token in any position with a compatible
forename. The forename guard still refuses same-surname collisions (`Jair Cunha` ↛ `Matheus
Cunha`). Genuinely-absent names — Betway's fringe/academy/departed players — still fail to join,
correctly, and are the ones to leave unmatched.

**Name handling.** Betway writes players as `"Surname, Firstname"` and lower-cases inside
names (`Mcburnie`, `O'brien`). `player_name()` flips and repairs both, which resolved 117
of 128 initial mismatches structurally; a further 35 verified rows went into
`name_mappings.csv`, taking a full-gameweek scrape from 167 reconciliation issues to 79.

**`betway.py` applies mappings at WRITE time.** Mappings added afterwards do not
retro-fix CSVs already written — either re-run `betway.py`, or remap in place:

```python
d["player_name"] = names.apply_player_names(d["player_name"])
```

**Never auto-apply reconciliation suggestions.** The 2026-08-17 run proposed
`Jair Cunha → Matheus Cunha` on a surname match: a Nott'm Forest **defender** onto a
Man Utd **forward** in the squad at the time. Applying it would have piped one player's
goalscorer and assist odds into another's projection with no error and no visible symptom.
Also rejected: `Nicolas Gonzalez → Nico Gonzalez` and a bare `Souza`. Same-surname
collisions are the failure mode to watch — check the club, not just the name.

Roughly 21 players per market stay permanently unresolved: Betway prices fringe, academy
and departed players FPL never lists (Alan Browne, Max Aarons, Wataru Endo…). They should
keep failing to join rather than be forced onto a lookalike.

`tools/betway.py` is the odds source (plain requests, no VPN). `tools/bet365.py --har FILE` parses a saved
bet365 capture (their live API is gated behind a WebSocket-bound token — see that module's
docstring for everything ruled out).

### sportsbet/ — always script-written

`tools/betway.py` (real markets) or `tools/build_preseason_data.py` (synthetic placeholders,
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
| `python weekly_update.py [--resume --gw N]` | the whole weekly routine in two phases, with preflights and a curation pause |
| `python tools/build_fixtures.py --gw N` | regenerate the rolling fixture window from the season list |
| `python tools/roll_gameweek.py --gw N` | **roll to a new gameweek**: shift the window to GW N, then seed synthetic F1 markets from GW N-1's actual factors (chains the two tools below). Update outright odds + lineups first; run `betway.py` after |
| `python tools/betway_outrights.py [--har F]` | scrape Betway PL **outright** markets → `title_odds.csv` / `top6_odds.csv` / `relegation_odds.csv`. Title & Top-6 = raw team odds (winners=K, de-margined by `MARGIN_SEASON=1.08` which matches Betway's ~8%); relegation = per-team Yes/No markets (more current than the stale main market), main-market fallback for the safe teams with no two-way. `--har` parses a saved capture |
| `python tools/build_synthetic_gw.py --gw N [--factor-gw M]` | seed synthetic **F1-only** (GW N) markets = GW M's (default N-1) archived actual factors × the new fixtures' win probs; **F2+ derive off F1** (synthetic never seeds F2 — the `_f2` files are cleared so F2 projects via the model / factor × baseline); drops `SYNTHETIC_NOTE.txt`. In-season sibling of `build_preseason_data` |
| `python tools/fill_synthetic_gaps.py [--source-gw M]` | backfill the current F1 player-prop markets (goalscorer/2+/assist/booking) from GW M's archived F2 projection, **only for matches Betway hasn't priced** — so attackers in unpriced fixtures aren't stuck on appearance points. Non-destructive; Betway-authoritative per match (a team is "priced" = a majority team of some priced match, robust to loan players) |
| `python tools/build_preseason_data.py` | pre-season bootstrap: estimated lineups, factor rebuild, synthetic odds (real GW1 markets used when `gw1_match_odds.csv` exists) |
| `python tools/injury_check.py` | curated start probs vs FPL's own availability flags (weekly, step 3a) |
| `python tools/price_change_analysis.py [--season Y]` | how prices moved after GW1 by initial ownership — a **tie-breaker** for picks (see below) |
| `python tools/bench_blank_split.py [--season Y] [--nailed N]` | how often a nailed starter blanks next GW, split into late injury (pure autosub), foreseeable (flagged pre-deadline) and rotation — grounds `bench_slot_weights` |
| `env\Scripts\python tools/chip_history.py [--gw N] [--free-transfers 1]` | logs Bench Boost + Triple Captain (single-GW F1), plus **Wildcard** (15-transfer rebuild vs the `--free-transfers` baseline, on the **8-fixture** weighted objective + captaincy) and **Free Hit** (same rebuild-vs-baseline delta but **F1 only** + captaincy) per GW to `inputs/chip_history.csv` (upsert) — spot the week a chip pays. Both deltas are measured against the best team reachable with your free transfers, not the raw current squad. Runs in `weekly_update.py` phase 2; the two deltas need the venv (PuLP) |
| `python tools/source_history.py --seed-ffs --seed-ours --gw N` / `--score --gw N` | source-accuracy ledger: freeze each source's predicted XI per GW, then score it against actual lineups. Both run automatically in `weekly_update.py` phase 2 (freeze this week, score last week) |
| `python tools/source_history.py --capture-csv FILE.csv --gw N` | freeze external sources (RotoWire, All About FPL, sweep consensus) from a `Source,Team,Player` CSV — the deadline-sweep step; surnames/nicknames auto-resolve, unmatched names are printed. GW1 back-fill example: `tools/seed_gw1_sources.py` |
| `python tools/backtest_projections.py` | forecast-vs-actual evaluation of the projection machinery against the archives |
| `python tools/build_training_data.py` | builds the per-component training matrices (`outputs/training/train_<stat>.csv`): 19 objective odds-derived features + `predicted` (baseline) + `actual` (target). See **Forward-projection models** below |
| `python tools/walkforward_projection.py [--embargo N]` | leak-free walk-forward validation of the projection models on 2025-26, picking one common hyperparameter set per component by pooled VAL |
| `python tools/save_projection_models.py` | trains + persists the deployable models to `outputs/models/`. **Rerun after each new gameweek of real archive data** |
| `python tools/refit_coefficients.py [--write] [--only win_pred]` | refit regressions from the archives (holdout-gated; see Refitting) |
| `python -m fpl_pipeline.reconcile` | standalone name-reconciliation report |
| `tools/extract_coefficients.py`, `tools/export_workbook_inputs.py` | one-off migration tools (workbook → pipeline); kept for provenance |

### Forward-projection models (`config.USE_PROJECTION_MODEL`)

Trained LightGBM models replace `factor × baseline` for the **defensive markets** — `clean_sheet`,
`concede2`, `saves3` — in **F3–F8** (F2 keeps its real-odds/fallback logic). These are the only
components that beat the *actual* pipeline value on walk-forward (2025-26): score1's F1-blend
already wins, assist/yellow never do. All three are used **pure** (no F1-blend). Features are the
19 objective, odds-derived ones — real on-pitch stats/minutes add nothing (the odds already price
in form + rotation). Pipeline: `build_training_data` → `walkforward_projection` (validate) →
`save_projection_models` (persist to `outputs/models/`) → `fpl_pipeline/projection_serving.py`
rebuilds the features at runtime (validated skew-free by `tools/validate_serving.py`) and
`players.py` applies them when `--gw N` is set. Toggle off with `config.USE_PROJECTION_MODEL =
False`. **Caveat:** trained on 2025-26, unverified on real 2026-27 until synthetic GW odds are
replaced; the models overshoot corrections slightly. Rebuild after each new week of real archive.

### Injury cross-check (`tools/injury_check.py`)

FPL publishes `status` / `chance_of_playing_next_round` / `news` per player. That answers
"can he play at all?", which is **not** our start probability ("will he be in the XI?"), so
it's used only as a **ceiling**: `a` imposes no constraint (a fully available player can
still be graded 0.15), `d` at 75% caps us at 0.75, and `i`/`u`/`s` forces 0. The tool reports
violations of that ceiling. Nothing auto-applies — a pending sale legitimately reads as
"available", since FPL only reflects completed moves.

### Price changes by ownership (`tools/price_change_analysis.py`)

2025-26 finding: **high initial ownership predicted price falls, not rises** (correlation
−0.335; early points explained almost nothing at −0.039). Players 30%+ owned at GW1 averaged
−0.556 by GW2 *while the tide rose* (447 of 752 players rose GW1→GW2, mean +0.087); 0–1% and
1–5% owned gained. It's front-loaded — most damage by GW2, largely reverted by GW19 — so
there's no sell-before-the-fall window. Mechanism: FPL prices popular players up front,
leaving only downside.

**Use it as a tie-breaker, never a driver.** When two candidates sit within noise on XP,
prefer the less-owned one; don't drop a better player to protect value, since a ~1m value
swing is worth 1–2 XP a season against a good pick's 20+. Note the asymmetry: a rise returns
only half on sale (`fpl_pipeline/prices.py`), a fall costs full price. Caveats: the top
buckets had n=8/9, it's a single season, and FPL has changed price mechanics before — re-run
on the current season as it accumulates.

## Optimiser tuning

Both optimisers configure via their `__main__` blocks (edit and run):

- **`optimisation.py`** (weekly transfers): `max_transfers`, `num_fixtures`,
  `additional_budget` (money in the bank), `force_transfer_out=[names]`,
  `force_transfer_in=[names]` (pin a player into every squad to see its XP cost),
  `compute_solutions` / `num_solutions_display` (solution pool + frequency analysis
  showing how often each player appears across near-optimal solutions),
  `max_defensive_players_per_team` (GK+DEF cap per club). **Optional two-stage tie-break**
  for when many squads score ~the same XP (acute at GW1): `tie_breaker="ownership"` +
  `tie_break_mode="differential"|"template"` + `xp_tolerance` locks XP within tolerance of the
  max, then breaks the tie by ownership (differential = favour low-owned to chase rank; template =
  high-owned to protect), plus a keep-owned term for fungible slots. This shapes the SQUAD /
  transfers only — the XI you field is ALWAYS set afterwards by pure XP (`pure_xp_lineup`:
  highest expected points per fixture, bench ordered by points), never by ownership. Off by default. One pool
  cleanup always runs: interchangeable 0-XP bench fillers collapse to one
  "Any £X.Xm 0-XP {pos}" per position (owned fillers kept). The module-top `DGW_TEAMS` /
  `DGW_EXTRA_FACTOR` block is the double-gameweek hack: list DGW teams to boost their
  F1 XP (proper DGW support is deliberately deferred).

  **Fixture weights are two inputs, not one.** `OWNERSHIP_WEIGHTS × RELIABILITY_WEIGHTS`,
  combined and normalised by `combine_fixture_weights()`. They're kept apart because they
  have different causes and change on different schedules:

  | | what it means | how it's set |
  |---|---|---|
  | ownership | "can I still fix it?" — a bad F6 is recoverable with a transfer, a bad F1 isn't | behavioural: ~1.2 transfers/week over 15 players = 8%/GW, `0.92 ** (k-1)`; only changes if your transfer habits do |
  | reliability | "how much do we trust the projection?" | **measured** from the backtest as skill vs a positional-mean baseline; re-measure as the archive grows |

  Product: `[1.0, 0.78, 0.52, 0.45, 0.36, 0.33, 0.29, 0.26]`. Note the **cliff between F2
  and F3**, not a linear ramp — that's where the input changes from market odds to model
  projection. Pass `ownership_weights=` / `reliability_weights=` to tune one in isolation;
  an explicit `fixture_weights=` still overrides both. Caveats in the code: the reliability
  figures come from a mid-season archive (August is more diffuse and unmeasured), F7/F8 are
  extrapolated from the flat tail, and F2's level rests on the `F2_VS_F1 = 0.85` assumption —
  raise it toward 1.0 if F2 usually has real odds scraped.

  **Bench value is priced by sub order**, not per fixture: `bench_slot_weights=(0.25, 0.05, 0.01)` for outfield subs 1/2/3, plus `gk_bench_weights=[0.01]*8`. Rationale: the
  optimiser already re-picks the XI per fixture at full weight, so rotation value is captured
  there and the bench weight is *pure autosub insurance* — slot 1 is used ~25–30% of weeks,
  slot 2 ~5%, slot 3 almost never. The solver assigns the best bench player to slot 1 by
  itself, and the printed bench is in sub order (set it that way in the FPL app). Per-fixture
  triples work too, which is how you model a Bench Boost:
  `bench_slot_weights=[(1.0, 1.0, 1.0)] + [(0.25, 0.05, 0.01)]*7` with
  `gk_bench_weights=[1.0] + [0.01]*7`. Slot-1 ~0.25-0.30 is grounded by `tools/bench_blank_split.py`. Each fixture summary also prints a **chip watch** line
  (Bench Boost = raw bench XP, Triple Captain = captain XP) so a good week doesn't slip by.
- **`optimisation_full.py`** (from-scratch squads, i.e. wildcard/free-hit): `num_fixtures`,
  `fixture_weights`, `bench_weight`, `total_squad_cost`, and `find_multiple=True` with
  `num_teams` / `diversity_method` / `points_tolerance` to generate several distinct
  candidate squads. **Still on the old scheme** — a single flat `bench_weight` and a
  hand-set `fixture_weights` list; the ownership × reliability split and slot-decayed bench
  above are `optimisation.py` only. Port them if you wildcard, or pass that file's
  `combine_fixture_weights()` output as `fixture_weights`.

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
| F7–F8 horizon | Two extra fully-modelled fixtures (the fixture data always went to F8) so the optimisers can plan two months out with `num_fixtures=8`; decay the tail harder (e.g. …0.5, 0.35, 0.25). Start probabilities come from curated F7/F8 columns in `starting_lineups.csv` (F6 fallback if absent); Total XP stays the 6-fixture blend |

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
   `lineup_overrides.csv` of last season's entries.
3. Before real markets/team news exist: `python tools/build_preseason_data.py` builds
   estimated starting lineups (last-season minutes + launch prices; then
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

## Pending model decision — re-run at GW5

**`python tools/fit_win_blend.py`** (no arguments). Decides whether F3–F8 should move from
the incumbent `win_pred_f3plus` (season outrights only) to a **blend** of outrights with
odds-implied team ratings built from football-data closing prices.

Evidence as of 2026-08-16 — the blend leads in both regimes tested:

| model | mid-season CV (2025-26) | GW1 2026-27 vs real odds |
|---|---|---|
| incumbent `win_pred_f3plus` | — | 0.0236 |
| rating only | 0.0647 | 0.0318 |
| **blend** | **0.0478** | **0.0205** |
| predict the mean | — | 0.1245 |

**Not adopted yet, deliberately.** The August test is 20 team-perspectives — ten matches
seen from both sides — so a 0.003 MAE gap sits inside noise, and the blend's outright
coefficients are fitted on 2025-26 *mid-season* data only, which is the regime-transfer
risk. Adopting on that would break the standard that correctly rejected the coefficient
refit and the xG-blend experiment.

Every gameweek adds 20 more rows of real ground truth: ~100 by GW5, ~200 by GW10. Re-run
then. If the blend still leads, put it through the backtest harness on real F2–F6
projections (not win probabilities alone) before wiring it in behind `improved`.

Supporting tools, both read-only — nothing in `fpl_pipeline/` reads `team_ratings.csv`:

- `python tools/build_team_ratings.py` — cached football-data downloader, team-name
  mapping, de-margined venue-split ratings, promoted-team substitution. Promoted sides
  inherit the mean of the previous season's three lowest-**rated** teams (not the three
  relegated — a well-rated side can go down; using relegated scored 0.0472 vs 0.0449).
  Ordering among them doesn't matter. 2026-27 level: 0.213.
- `python tools/rating_crosscheck.py` — second opinion on F1–F10, reporting unweighted
  per-fixture disagreement (*is it wrong?*) and a weighted summary via the optimiser's
  own fixture weights (*does it matter?*). Reaches F9/F10 by reading `season_fixtures.csv`
  directly, so it sees past the pipeline's 8-fixture window.

**Method warning worth keeping.** Two comparisons in the same session pointed the wrong
way because of shortcuts in *measurement*, not modelling: an ad-hoc script made a bad
coefficient refit look 7.5% better (the proper gate said 9.5% worse), and approximating
the blend's away leg as `1 − home − 0.24` made a good model look like a loser (0.0280 vs
0.0205 once both legs were fitted properly). Always reuse the production feature
construction, and fit every leg the way the live model does.

## Refitting the model

`python tools/refit_coefficients.py` refits every regression from
`inputs/historical_player_data.csv` and `inputs/historical_fixture_odds.csv` (which the
pipeline now maintains automatically). Dry run by default — shows n / R² / changed
coefficients; `--write` regenerates `fpl_pipeline/data/coefficients.json`, backing up the
workbook-extracted original to `coefficients_workbook.json` (parity mode then uses the
backup automatically). Feature construction is shared with serving code, so train and
serve cannot drift. The bonus model is carried over unchanged (refitting it needs actual
bonus-point outcomes, which the archives don't hold).

**A holdout check gates every `--write`** and refuses if the candidate degrades forecast
accuracy (override with `--force`, but don't). Two different gates, matched to what's being
written:

- `projection_holdout_check` — for the **baselines**. Not theoretical: baselines are consumed
  as a *ratio* (factor = odds ÷ baseline at week M, projection = factor × baseline at week
  M+k), so a refit that improves same-week fit can still wreck projections. Refitting on the
  2025-26 archive degraded holdout projection MAE by 100–225% across every stat and was
  correctly refused. Expect the refit objective to need regularisation toward ratio
  stability, not just OLS fit.
- `win_pred_holdout_check` — for **`win_pred_f3plus`**, which the projection check does *not*
  cover (it only swaps `model.BASELINES`). Chronological split of the fixture archive, never
  random: it's a time series and a shuffled split leaks later gameweeks into training.

`--only win_pred` writes that model alone, leaving every other coefficient untouched —
"refit everything at once" is not a safe default given the baseline result above.

**Worked example of why the gate exists.** An ad-hoc script suggested the win-pred refit was
7.5% *better* on holdout; the proper gate found it 9.5% *worse* and refused the write. The
ad-hoc version had used home-perspective rows only, which fed the incumbent's `home` and
`home_x_strength_diff` coefficients a constant and handicapped it. Never treat a quick
comparison as adoption evidence — run it through a gate that reuses the production feature
construction (`_win_features`, shared by fit and check for exactly this reason).

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
with `io_utils` (snapshots, tolerant CSV reading) underneath. `docs/PIPELINE_MAP.md` maps
each module back to the workbook sheets it replaced.

## Tests

`python -m pytest` — unit tests for markets/model/team model, end-to-end improved-mode
behaviour on a synthetic single-gameweek state, history upsert semantics (player and
fixture archives), the synthetic-odds archive guard, FFS staging and team-news parsing,
refit round-trip (predictions, not raw coefficients — the feature sets are collinear),
name reconciliation and Excel-encoding repair, the fixture-window builder, and the
full workbook parity check.

`tests/test_optimiser_bench_slots.py` (bench sub-order pricing, fixture-weight
combination) **is skipped under system Python** — PuLP lives only in the repo
virtualenv, which has no pytest. Run those assertions with `env\Scripts\python`
directly, or install pytest into the venv.

## Retired (deleted; recoverable from git history)

- `extract_fpl_data.py`, `extract_defensive_contributions.py` — ported into `fpl_pipeline/ingest.py`
- `fpl_data/player_name_changes.csv` — merged into `inputs/name_mappings.csv`
- All openpyxl workbook writes in `starting_lineups.py`
- `modelling/` and `strength_modelling/` — `tools/refit_coefficients.py` replaces their
  paste-data-in, copy-coefficients-out workflow
- `odds_data_outputs/`, `starting_lineups/data.csv`, old workbook copies — dead pipeline
  leftovers; `outputs/` and `.idea/` untracked (regenerable / IDE config)
