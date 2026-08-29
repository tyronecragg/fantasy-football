# FPL Challenge optimisers

Tools for the **FPL Challenge** game — a separate competition from the main FPL squad.
Each gameweek is a fresh, standalone pick with its own scoring twist. This folder has one
script per gameweek of the opening "Kick Off" series (GW1–GW5) plus a shared engine.

> This is **not** the main-squad optimiser (`../optimisation.py`). Nothing here
> carries over week to week, there is no budget, and there are no transfers.

---

## The base game (same every week)

- Pick **1 goalkeeper + 5 outfielders** — six players total.
- The five outfielders must form one of six allowed shapes (GK is implicit; the shape is
  DEF-MID-FWD): **1-1-3, 1-2-2, 1-3-1, 2-1-2, 2-2-1, 3-1-1**.
- **Unlimited budget** — any player is affordable.
- **Max 1 player per club**, unless a week's rule raises it (GW2 and GW4 raise it to 3).
- Pick **one captain**, who scores **double**.
- Every week is independent: you rebuild from scratch, and a team/player can be reused in
  a later week (unlike Last Fan Standing).

What changes week to week is the **scoring twist**. Each `gw*.py` expresses that twist and
hands a prepared table to the shared engine, which does the actual optimisation.

---

## The five weeks

| Script | Week | Rule | Club cap | How it's modelled |
|---|---|---|---|---|
| `gw1.py` | Instant Impact | New signings score **×2** | 1 | double players on the signings list |
| `gw2.py` | Welcome Back | Promoted trio (Coventry, Ipswich, Hull) score **×2** | **3** | double players from those clubs |
| `gw3.py` | All Out Attack | Goals **and** assists score **×2** | 1 | add one more copy of goal+assist points |
| `gw4.py` | Derby Day | Man City & Man Utd score **×2** | **3** | double players from those clubs |
| `gw5.py` | The Shield | Defensive contributions score **10** (not 2) | 1 | add `8 × P(DefCon threshold)` |

Two kinds of twist:

- **"Double these players" (GW1, GW2, GW4).** A set of players gets ×2. The captain
  stacks on top, so a captained doubled player scores **×4**. If it turns out a week does
  *not* stack the two doublers, pass `--no-stack` and a captained doubled player counts ×3
  instead.
- **"Change the points" (GW3, GW5).** The scoring itself changes for everyone, so there's
  no player set — the effective points are recomputed and the captain then doubles that
  boosted total.

---

## How the optimisation works

For each of the six formations, `challenge_core.solve_and_report` solves a small integer
program (PuLP, same solver as the main optimiser) that:

- picks exactly the formation's count in each position (1 GK + the DEF/MID/FWD shape),
- respects the per-club cap,
- picks exactly one captain, who must be one of the six picked players,
- maximises **`sum(eff_xp over the XI) + cap_bonus(captain)`**.

It then ranks the six formations and prints the best XI, the captain, and the boosted
players driving the pick.

A per-club cap means the pick is **not** separable by position (one club can be the best
option in two positions at once), which is why it's solved exactly rather than greedily.

### The two columns every week produces

Each `gw*.py` builds these on the player table before calling the engine:

- **`eff_xp`** — effective expected points for the week, already including any doubling.
- **`cap_bonus`** — the extra points if this player is captain (one more copy of their
  score). Normally equal to `eff_xp` (captain = ×2). Under `--no-stack`, a doubled
  player's `cap_bonus` drops to their base `F1 XP`.
- **`boosted`** — flag for display, marking who got the week's bonus.

---

## Inputs

- **`../outputs/13_players_master.csv`** — the pipeline's player projections. The scripts
  read the **`F1 XP`** column (expected points for the upcoming gameweek) and, for GW3/GW5,
  the underlying component columns (`F1 Start`, `F1 Score 1+/2+/3+`, `F1 Assist`,
  `F1 Assist 2+`, `F1 Defensive Contribution - DEF/MID`).
- **`../inputs/fpl_challenge_new_signings.csv`** (GW1 only) — the double-points list, one
  `Player` per row. **The in-game "new signings" filter toggle is the definitive source**;
  this file was built from the Transfermarkt 2026/27 arrivals. Only that summer's arrivals
  qualify — last season's signings score single points. Names are matched
  accent/case-insensitively with a last-name fallback; unmatched names are printed so
  typos and departed players surface.

---

## GW3 & GW5: staying consistent with the model

GW3 and GW5 rebuild points from the projection's components rather than guessing. The
reconstruction was checked against the pipeline's own `F1 XP Pre` and matches to within
~0.001, so these weeks stay consistent with `fpl_pipeline/model.py`:

- **GW3** adds `start × (goal_points + assist_points)`, where goal points use the FPL
  per-position values (GK 10, DEF 6, MID 5, FWD 4) times `P(score 1+/2+/3+)`, and assists
  are `3 × P(assist)`. Appearance, clean sheets, saves, cards, DefCon and bonus are left
  alone.
- **GW5** adds `start × 8 × P(hit DefCon threshold)` — the uplift from 2 points to 10.
  The model carries DefCon for **defenders and midfielders only**, so forwards get no
  boost this week.

---

## Running

From inside this folder:

```bash
cd fpl_challenge
python gw1.py
python gw2.py
python gw3.py
python gw4.py
python gw5.py
```

Flags:

- `--max-per-club N` — override the club cap for the week.
- `--no-stack` — GW1/GW2/GW4 only; captain does not re-double an already-doubled player.
- `--confirmed-not-starting NAME …` — players confirmed benched/out; removed from the pool.
- `--confirmed-starting NAME …` — players confirmed in the XI; start set to 1.0 (see below).

`main()` also takes `confirmed_not_starting=[…]` and `confirmed_starting=[…]` as lists, so in
PyCharm you can just edit the call at the bottom of the script instead of passing CLI args:

```python
if __name__ == "__main__":
    main(
        confirmed_not_starting=["Bobby Thomas"],
        confirmed_starting=["Ephron Mason-Clark"],
    )
```

The list and the CLI flag merge, so you can use either or both. Names are matched
accent/case-insensitively; **a name that matches nothing prints a large `!!!!` banner** and
is otherwise ignored — a typo means the intended drop / force-start silently did not happen,
so the warning is deliberately hard to miss.

---

## Late-swap: confirmed lineups and the "bring in if they start" watchlist

FPL Challenge lets you change any player right up until their match kicks off, so you can react
as team news lands. The tooling supports this two ways.

**Marking confirmed lineups** (`confirmed_starting` / `confirmed_not_starting`):

- **`confirmed_not_starting`** drops the player from the pool entirely — they can't be picked
  and can't appear as a swap candidate.
- **`confirmed_starting`** sets their start probability to **1.0 in place** and lifts their
  projection to the full-match value, so the optimiser values them as nailed. Only the
  *pre-bonus* points scale with start (`F1 XP Pre = start × full-match value`, exact), so it
  divides that part by the old start probability and keeps the bonus term — dividing the whole
  `F1 XP` by start would over-inflate the bonus. A player with essentially no minutes projection
  (start ≈ 0) can't be scaled, so it sets start to 1.0, warns, and leaves XP untouched.

Both are applied **before** selection and the watchlist, so re-running after each lineup drop
updates the pick and the remaining swaps.

**The watchlist** (`BRING IN IF THEY START`, printed automatically every week): for each
non-nailed player (`0 < start < 1`) who isn't already picked, it works out their value **if they
start** and, if slotting them in at that value beats the current XI in *any* formation, lists
them as a candidate swap.

- **Value if they start** = the full-match projection (pre-bonus scaled by `1/start`, bonus
  kept), carrying the week's boost. This is exact for the ×2 weeks and a close approximation for
  the add-on weeks.
- **Grouped by the club whose lineup you'd watch**, most valuable club first, under that club's
  provisional picks. Each club header shows its **kickoff in SAST (UTC+2)**.
- **Only swaps you can actually make in time are shown.** Each is tagged:
  - `same match` — incoming and outgoing kick off together; you see the whole XI at once. Always safe.
  - `actionable` — cross-club, but the incoming player's news (~1h pre-kickoff) lands before the
    outgoing player kicks off.
  - `TOO LATE` swaps are **hidden** (you'd already be locked into the outgoing player), as are
    clubs with no swappable upgrade; both are rolled into a one-line footer count.
  - `timing?` — shown only if there's no kickoff data (see below); the swap can't be checked.
- Tunable in `highlight_if_start`: `min_gain` (default **0.1** — hide upgrades worth less than
  this) and `max_candidates` (default 40).

**Kickoff times** come from `../outputs/fixture_kickoffs.csv`, written by the weekly Betway
scrape (`tools/betway.py`) from Betway's `expectedStartEpoch`. It stores UTC (canonical); the
watchlist displays SAST. Timing comparisons run in UTC, so the display offset never affects which
swaps are judged actionable. Without the file, headers show `kickoff ?` and swaps fall back to
`timing?` (the old, timing-blind behaviour).

---

## Important caveat: which gameweek the data points at

All five scripts read the **`F1`** columns — i.e. whatever the pipeline currently treats as
the *next* gameweek. Run for GW1, that's GW1. To play GW2–GW5 for real, **refresh the
pipeline so `F1` points at that week's fixtures** before running the matching script; the
scripts do not reach into the `F2`/`F3`/… columns.

If you'd prefer them to take a `--gw N` argument and read the matching `Fn` columns
directly (so you could run all five off one data build), that's a small change to
`challenge_core.py` — the goal/assist/DefCon component names are the only per-week parts.

---

## Files

| File | Purpose |
|---|---|
| `challenge_core.py` | shared engine: player loading, name matching, confirmed-lineup handling, formation ILP, reporting, the goal/assist/DefCon helpers, and the "bring in if they start" watchlist |
| `gw1.py` … `gw5.py` | one week each; each just builds `eff_xp` / `cap_bonus` / `boosted` and calls the engine |
| `../inputs/fpl_challenge_new_signings.csv` | GW1 double-points list |
| `../outputs/fixture_kickoffs.csv` | per-fixture UTC kickoffs for the watchlist's timing; written by `tools/betway.py` |
