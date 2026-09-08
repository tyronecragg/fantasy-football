# Source-accuracy capture — "compile the teams"

How the predicted-XI accuracy ledger ([`tools/source_history.py`](../tools/source_history.py))
is fed each gameweek. **Trigger phrase: when Tyrone says "compile the
teams", Claude runs this end-to-end** (drives the browser, builds the CSV, captures, reports UNMATCHED).

## When
**At / near the GW deadline** — predicted-lineup sites don't populate their XIs until ~a day out
(3 days before GW3, FPL Edits had only Brentford filled, the other 19 blank). Capturing early freezes
a provisional/empty XI, which rots the ledger. So: compile at the deadline, not before.

## Sources to freeze (all-20 predicted XIs each)
**Main sources (capture EVERY gameweek — `MAIN_SOURCES` in `tools/source_history.py`; `--score` warns if any is missing):**
- **Curated** — ours, via `--seed-ours` (top-11 F1 per team). Never skip; it's the calibration record.
- **RotoWire** — https://www.rotowire.com/soccer/lineups.php — all-20 predicted XIs. **Capture at the
  deadline while it still says "Predicted Lineup"; it flips to *confirmed* near kickoff and scoring a
  confirmed XI as a "prediction" is dishonest.** Uses `Initial. Surname` labels (`E. Haaland`,
  `V. van Dijk`) — the `Resolver` strips the initial, so they resolve fine.
- **FFS** — already scraped to `inputs/ffs_predicted_lineups.csv`; use `--seed-ffs` **only if that
  file is fresh for this GW** (it can be stale — check its date first), else read the live site.

**On trial (capture when populated, promote once they earn it):**
- **Fantasy Football Pundit** — https://www.fantasyfootballpundit.com/fantasy-premier-league-team-news/
  (all 20 XIs with a per-player Start %, own prediction; added 2026-09-03).
- **FPL Edits** — https://fpledits.com/predicted-lineups-pl (added 2026-09-03).
- **All About FPL** — https://allaboutfpl.com/category/predicted-lineups/ (its per-GW page can vanish
  after the GW — capture it live at the deadline, no retro back-fill).

## Steps (Claude)
1. **Read each source's XIs** via the browser (`navigate` + `get_page_text`/`read_page`), 11 starters per team.
2. **Write `inputs/source_history/gw<N>_external.csv`** with columns `Source,Team,Player` — one row per
   starter. Use **canonical team names** (Arsenal, Man City, Nott'm Forest, Man Utd, Spurs …) so the
   scorer's team match works; player names can be as the site shows them (the `Resolver` maps them).
3. **Capture:**
   ```
   env/Scripts/python tools/source_history.py --capture-csv inputs/source_history/gw<N>_external.csv --gw <N>
   ```
   Re-running a (Source, GW) upserts (replaces) its rows, so it's safe to re-capture.
4. **Fix `UNMATCHED`** lines it prints (a name the roster didn't resolve) — add a `name_mappings.csv`
   row or a `source_history.ALIASES` entry — then re-run. An unmatched name silently undercounts a source.
5. **Freeze our own XI too:** `... --seed-ours --gw <N>` (top-11 F1 per team) — this is the one that
   matters most (calibration of our 1.0s / coin-flips), so never skip it.

## After the GW plays
```
env/Scripts/python tools/source_history.py --score --gw <N>
```
Prints each source's hit rate (of the 11 it predicted per team, how many actually started). The scorer
derives actual starters from `player_gameweek_stats.csv` `starts==1` (the current FPL-Core-Insights
dumps ship no `lineups.csv`).

## History
- **GW1** scored (2026-09-03): FFS 89.5%, Curated 88.6%, All About FPL 85.9%.
- **GW2** scored (2026-09-03, FFS + FFP captured retroactively off their still-live GW2 pages):
  FFS 90.0%, Curated 87.7%, Fantasy Football Pundit 85.5%. AAF GW2 page no longer reachable, so not back-filled.
- **GW3** — **RotoWire wired in as a main source (2026-09-03)**: its 20 predicted XIs captured pre-deadline
  (`inputs/source_history/gw3_rotowire.csv`), 220/220 resolved after teaching the `Resolver` to strip
  `Initial. Surname` leading initials. Score after GW3 plays.
- Per-source *ranking* was deprioritised 2026-08-28 (the odds price in form; the panel is hand-curated),
  so this is mainly (a) calibration of our own XI and (b) a one-off validation of a new source like FPL
  Edits before promoting it — not a heavy weekly obligation.
