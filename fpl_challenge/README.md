# FPL Challenge optimisers

Tools for the **FPL Challenge** game — a separate competition from the main FPL squad.
Each gameweek is a fresh, standalone pick with its own scoring twist. This folder has one
script per gameweek — the "Kick Off" series (GW1–GW5) and the "Game Changers" series
(GW6–GW9) — plus a shared engine.

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

## The weeks

| Script | Week | Rule | Club cap | How it's modelled |
|---|---|---|---|---|
| `gw1.py` | Instant Impact | New signings score **×2** | 1 | double players on the signings list |
| `gw2.py` | Welcome Back | Promoted trio (Coventry, Ipswich, Hull) score **×2** | **3** | double players from those clubs |
| `gw3.py` | All Out Attack | Goals **and** assists score **×2** | 1 | add one more copy of goal+assist points |
| `gw4.py` | Derby Day | Man City & Man Utd score **×2** | **3** | double players from those clubs |
| `gw5.py` | The Shield | Defensive contributions score **10** (not 2) | 1 | add `8 × P(DefCon threshold)` |
| `gw6.py` | Matchwinner | Winning goals worth an extra **10** | 1 | add `10 × P(win) × goal-share` |
| `gw7.py` | Away Days | Players in **away** matches score **×2** | 1 | double players whose fixture is away |
| `gw8.py` | Legends | ≥5 years at club score **×2** | 1 | double players on the Legends list |
| `gw9.py` | Trick or Treat | **+3** per successful dribble, **−1** per dispossession | 1 | add `start × (3·dribbles/90 − dispossessed/90)` |

Three kinds of twist:

- **"Double these players" (GW1, GW2, GW4, GW7, GW8).** A set of players gets ×2. The captain
  stacks on top, so a captained doubled player scores **×4**. If a week does *not* stack the
  two doublers, pass `--no-stack` and a captained doubled player counts ×3 instead.
- **"Change the points" (GW3, GW5).** The scoring itself changes for everyone — the effective
  points are recomputed and the captain then doubles that boosted total.
- **"Add a bespoke term" (GW6, GW9).** A per-player expected-points term is added on top of
  normal XP (winning-goal value; dribble/dispossession value), then captain-doubled.

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

- **`../outputs/players_master.csv`** — the pipeline's player projections. The scripts
  read the **`F1 XP`** column (expected points for the upcoming gameweek) plus component
  columns as each week needs: `F1 Start`, `F1 Score 1+/2+/3+`, `F1 Assist`, `F1 Assist 2+`,
  `F1 Defensive Contribution - DEF/MID` (GW3/GW5), `F1 Win` (GW6), and `F1 Venue` (GW7).
- **`../inputs/fpl_challenge_new_signings.csv`** (GW1 only) — the double-points list, one
  `Player` per row. **The in-game "new signings" filter toggle is the definitive source**;
  this file was built from the Transfermarkt 2026/27 arrivals. Only that summer's arrivals
  qualify — last season's signings score single points. Names are matched
  accent/case-insensitively with a last-name fallback; unmatched names are printed so
  typos and departed players surface.
- **`../inputs/fpl_challenge_dribbles.csv`** (GW9 only) — this-season successful-dribbles
  per 90, one `Player` per row, matched to FPL names. **Sourced from FotMob** (the per-90
  dribble leaderboard); **refresh it closer to GW9** by re-pasting an updated FotMob list, as
  the current values are an early-season snapshot. Values are raw per-90 (unshrunk small
  samples early on), so treat outliers with care until minutes build.

---

## Modelling the twists

GW3/GW5 rebuild points from the projection's components; the reconstruction was checked
against the pipeline's own `F1 XP Pre` and matches to within ~0.001, so they stay consistent
with `fpl_pipeline/model.py`:

- **GW3** adds `start × (goal_points + assist_points)`, where goal points use the FPL
  per-position values (GK 10, DEF 6, MID 5, FWD 4) times `P(score 1+/2+/3+)`, and assists
  are `3 × P(assist)`. Appearance, clean sheets, saves, cards, DefCon and bonus are left
  alone.
- **GW5** adds `start × 8 × P(hit DefCon threshold)` — the uplift from 2 points to 10.
  The model carries DefCon for **defenders and midfielders only**, so forwards get no
  boost this week.

GW6/GW9 add a bespoke expected-points term:

- **GW6** adds `10 × P(team wins) × goal_share`, where `goal_share` = the player's expected
  goals ÷ the team's expected goals (from goalscorer odds). Shares within a team sum to 1, so
  exactly one winning goal is spread across the winning side. `P(win)` is `F1 Win`.
- **GW9** adds `start × (3 × dribbles/90 − 1 × dispossessed/90)`. Dribbles/90 come from
  `fpl_challenge_dribbles.csv` (this season, pasted leaderboard); dispossessed/90 from this
  season's FPL-Core-Insights match stats, shrunk by 270 pseudo-minutes. Note: this season's
  feed leaves `successful_dribbles` empty, which is why dribbles come from the pasted file
  rather than being read directly.

---

## Running

From inside this folder:

```bash
cd fpl_challenge
python gw1.py   # … through gw9.py
```

Flags:

- `--max-per-club N` — override the club cap for the week.
- `--no-stack` — GW1/GW2/GW4/GW7/GW8 only; captain does not re-double an already-doubled player.
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

All scripts read the **`F1`** columns — i.e. whatever the pipeline currently treats as the
*next* gameweek. So **refresh the pipeline so `F1` points at that week's fixtures** before
running the matching script; the scripts do not reach into the `F2`/`F3`/… columns.

If you'd prefer them to take a `--gw N` argument and read the matching `Fn` columns directly
(so you could run any week off one data build), that's a small change to `challenge_core.py`.

---

## Files

| File | Purpose |
|---|---|
| `challenge_core.py` | shared engine: player loading, name matching, confirmed-lineup handling, formation ILP, reporting, the goal/assist/DefCon helpers, and the "bring in if they start" watchlist |
| `gw1.py` … `gw9.py` | one week each; each builds `eff_xp` / `cap_bonus` / `boosted` and calls the engine |
| `../inputs/fpl_challenge_new_signings.csv` | GW1 double-points list |
| `../inputs/fpl_challenge_legends.csv` | GW8 Legends list (5+ years at club), from the in-game filter |
| `../inputs/fpl_challenge_dribbles.csv` | GW9 this-season dribbles per 90 |
| `../outputs/fixture_kickoffs.csv` | per-fixture UTC kickoffs for the watchlist's timing; written by `tools/betway.py` |
