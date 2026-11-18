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
| `gw9.py` | Trick or Treat | **+3** per successful dribble, **−1** each time tackled | 1 | add `start × (3·dribbles/90 − tackled/90)` |
| `gw10.py` | Spectacular | **+1** per outside-box shot on target, **+5** per outside-box goal | 1 | add `start × (1·obox SoT/90 + 5·obox xG/90)` |
| `gw11.py` | Make the Call | substitutes score **×3** | 1 | add `3 × P(comes on) × pts per sub appearance` |
| `gw12.py` | Wonderkids | aged ≤23 at the final GW12 deadline score **×2** | 1 | double players not yet 24 on 29 Nov 2026 |
| `gw13.py` | Dribbling: 20 | **+1** per dribble, **+5** once for 3+ | 1 | add `start × (λ + 5·P(3+))`, Poisson λ = dribbles/90 × 85/90 |

Three kinds of twist:

- **"Double these players" (GW1, GW2, GW4, GW7, GW8, GW12).** A set of players gets ×2. The captain
  stacks on top, so a captained doubled player scores **×4**. If a week does *not* stack the
  two doublers, pass `--no-stack` and a captained doubled player counts ×3 instead.
- **"Change the points" (GW3, GW5).** The scoring itself changes for everyone — the effective
  points are recomputed and the captain then doubles that boosted total.
- **"Add a bespoke term" (GW6, GW9, GW10, GW11, GW13).** A per-player expected-points term is added on top of
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
- **`../inputs/fpl_challenge_dribbles.csv`** (GW9 only) — this-season successful dribbles per 90 **and
  success %**, one `Player` per row, matched to FPL names. **Sourced from FotMob**
  (<https://www.fotmob.com/leagues/47/stats/season/36781/players/won_contest> — season id 36781 =
  2026/27); **refresh it closer to GW9**. The success % drives the failed-dribble penalty, so keep it.
  To refresh: save the leaderboard to `raw/fotmob_dribbles.txt` (one `Name;success_pct;per90` per line)
  and run `python build_dribbles.py` — it matches names to the master, with an `ALIASES` table for
  FotMob spellings that don't resolve, and prints anything unmatched loudly.
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
- **GW9** adds `start × (3 × dribbles/90 − tackled/90)`. There is no "was tackled" stat, so
  **tackled = dispossessed + failed dribbles**: a player is tackled either while holding the ball
  or while trying to beat a man. Dribbles/90 and success % come from `fpl_challenge_dribbles.csv`
  (FotMob); failed dribbles/90 = dribbles/90 × (100 / success% − 1). Dispossessed/90 comes from this
  season's FPL-Core-Insights match stats (shrunk by 270 pseudo-minutes), matched on **player_id** —
  the feed's full names often differ from the master's. This season's feed leaves
  `successful_dribbles` empty, which is why dribbles come from FotMob.
  **Two-season blend:** every rate mixes this season with last season at **half weight, by minutes** —
  `(this-season total + ½ × last-season total) ÷ (this-season mins + ½ × last-season mins)` — so
  regular starters stay close to this season while small samples lean on last season. Last season's
  FotMob list lives in `fpl_challenge_dribbles_last.csv`; minutes and dispossessed for both seasons come
  from FPL-Core-Insights (last season linked by full name, as player ids change each season). A season
  only counts toward a player's dribbles if FotMob lists him that season. `LAST_WEIGHT` in `gw9.py`.
- **GW10** adds `start × (1 × outside-box shots on target/90 + 5 × outside-box xG/90)`, from Understat
  shot data (`build_longrange.py`): every shot has a pitch position, so outside-the-box is exact (box edge
  at X ≈ 0.843, confirmed by the step in shot volume there; penalties always inside). Premier League shots
  only — a player's Understat history includes other leagues, which would inflate rates against
  Premier League minutes. Goals use outside-box **xG**, not actual goals (rare and swingy). Same
  two-season blend as GW9 (last season at half weight, by minutes), shrunk by 270 pseudo-minutes.
  Refresh with `python build_longrange.py` (cached; `--refresh` re-fetches everyone).
- **GW11** adds `3 × P(comes on) × points per sub appearance`, where P = available (FPL status flag) ×
  (1 − start) × sub-on rate when not starting. Rates from FPL per-gameweek data (`build_subs.py`: a sub
  appearance = minutes > 0 with no start), two seasons with last at half weight, steadied toward position
  averages. `confirmed_bench` keeps benched players in the pool at 0% start; the watchlist and
  contingency sections are switched off (`show_watchlist` / `show_contingency` in `solve_and_report`),
  since starting can be worth less than coming off the bench.
- **GW12** doubles players not yet 24 on the last GW12 kickoff (strictest reading of "final deadline").
  Birth dates from FPL's API (`build_ages.py` → `fpl_challenge_birthdates.csv`); 24th birthdays near the
  window are printed to check in-game; no birth date = not eligible.
- **GW13** adds `start × (λ + 5 × chance of 3+ dribbles)`, dribbles per match as a Poisson around GW9's
  blended dribbles/90 over 85 minutes. Refresh the FotMob paste and `build_dribbles.py` first.

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
- **One line per swap, in kickoff order** — the same style as "If a pick doesn't start", e.g.
  `Sun 16:00 Amad Diallo (Man Utd, start 80%): OUT Jacob Murphy | captain -> Amad Diallo  [+0.72]`.
  The gain in brackets is the projected improvement if he's confirmed starting.
- **Only swaps you can actually make are listed.** While re-picking, any pick whose match kicks off
  before the candidate's team news (~1h pre-kickoff) is locked in, and so is a captain who is locked —
  so the swap never relies on dropping someone who has already played. A swap with no legal version
  is counted in the footer as "too late to swap"; one with no kickoff data is marked `(timing unknown)`.
- Tunable in `highlight_if_start`: `min_gain` (default **0.1** — hide upgrades worth less than
  this) and `max_candidates` (default 40).

**If a pick doesn't start** (printed after the watchlist): for each selected player, in kickoff
order, the changes to make if their lineup drops and they're out, e.g.
`Sat 16:00 Antonee Robinson (Fulham): OUT Alisson Becker | IN Jordan Pickford, Virgil van Dijk`.
The missing player is always removed, so `OUT` lists only the *other* picks to drop; any armband
move is shown as `captain -> …`. **The armband can only move to a player whose match hasn't kicked
off** (same kickoff time or later) — you can't hand it to someone who has already played. At that moment, picks whose match
kicks off **earlier** are locked in (and keep the armband if they have it); only players whose
match is at that kickoff time **or later** can come in, valued at their normal XP since later
lineups aren't known yet. A tiny preference for keeping current picks stops tied options from
causing needless reshuffles. Players you've listed in **`confirmed_starting` are skipped** — there's
no "if he's out" case to plan for — and named in a one-line footer instead. Needs
`fixture_kickoffs.csv`. Lives in the shared engine, so it runs for every gameweek script.

**Kickoff times** come from `../outputs/fixture_kickoffs.csv`, written by the weekly Betway
scrape (`tools/betway.py`) from Betway's `expectedStartEpoch`. It stores UTC (canonical); the
watchlist displays SAST. Timing comparisons run in UTC, so the display offset never affects which
swaps are judged actionable. Without the file, headers show `kickoff ?` and swaps fall back to
`timing?` (the old, timing-blind behaviour). **In archive mode** (`archive_gw=N`) kickoffs come instead
from the official FPL fixtures for that gameweek (FPL-Core-Insights `fixtures.csv`, Premier League
rows only), because a fresh Betway scrape overwrites the kickoff file with the *current* weeks.

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
| `gw1.py` … `gw13.py` | one week each; each builds `eff_xp` / `cap_bonus` / `boosted` and calls the engine |
| `../inputs/fpl_challenge_new_signings.csv` | GW1 double-points list |
| `../inputs/fpl_challenge_legends.csv` | GW8 Legends list (5+ years at club), from the in-game filter |
| `../inputs/fpl_challenge_dribbles.csv` | GW9 this-season dribbles per 90 |
| `../inputs/fpl_challenge_longrange.csv` | GW10 outside-box shots / goals / xG per player-season, from Understat |
| `../inputs/fpl_challenge_subs.csv` | GW11 starts / sub appearances / sub points per player-season (`build_subs.py`) |
| `../inputs/fpl_challenge_birthdates.csv` | GW12 birth dates from the FPL API (`build_ages.py`) |
| `../outputs/fixture_kickoffs.csv` | per-fixture UTC kickoffs for the watchlist's timing; written by `tools/betway.py` |

---

## Mid-week: snapshots, odds refresh and live locking

**Snapshot (not the archive).** Once the pipeline rolls on, run a Challenge week from a refreshable copy:
`cc.make_snapshot(8)` copies GW8's archive rows to `snapshots/gw8.csv`, and `main(..., snapshot_gw=8)` reads
it — or just let `load_players(gw=8)` find it automatically. The archive stays the untouched decision record; the snapshot is yours to update. Kickoffs then come
from the official FPL fixtures for that gameweek.

**Odds refresh.** `python refresh_snapshot.py 8` fetches fresh Betway prices (read-only — it never runs the
scraper's write step) for the snapshot's still-to-play matches, and scales each goal / assist / clean-sheet
/ win probability by how far its raw price has moved since that gameweek's run (`inputs/raw_odds/…/GWnn/`).
XP changes by the resulting change in points only, so the pipeline's other adjustments carry through.
Cards, saves, concede ladders and DefCon aren't refreshed.

**Live locking.** Pass `current_team=[your six]` and `captain=...`. Using the current time (UTC, shown in
SAST), anyone whose match has kicked off is locked: your played picks stay in, played non-picks can't be
added, and the armband can't move onto a match that's already started (a played captain keeps it).
Players in `confirmed_not_starting` are dropped without a warning; other unmatched names get a banner.

---

## Adding a new week

Copy the closest existing script (a "double these players" week → `gw8.py`; a bespoke term → `gw9.py`) and
change only the twist: how `boosted`, `eff_xp` and `cap_bonus` are built. Keep the standard `main()` so
every week has the same controls:

```python
def main(confirmed_not_starting=None, confirmed_starting=None,
         current_team=None, captain=None, now=None):
    ...
    df = cc.load_players(gw=N)   # snapshots/gwN.csv if it exists, else the live master
    df = cc.confirm_not_starting(df, ...)
    df = cc.confirm_starting(df, ...)
    cc.set_live(df, current_team, captain, now)   # time-locking once matches start
    ... build boosted / eff_xp / cap_bonus ...
    cc.solve_and_report(df, max_per_club, title, boost_tag)
```

The watchlist, "if a pick doesn't start" plan, live locking and SAST kickoffs all come from the shared
engine — nothing extra to wire. Passing `gw=N` is all it takes to pick up a snapshot: if `snapshots/gwN.csv` exists it is used, otherwise the
live master. `archive_gw` remains for reading the frozen archive directly.
