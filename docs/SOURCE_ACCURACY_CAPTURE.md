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
- **FFS** — compiled by hand into `inputs/ffs_predicted_lineups.csv` at the deadline (the automated
  scrape/staging was retired). Use `--seed-ffs` **only if that file is fresh for this GW** (check its
  date first — a stale file freezes a pre-season XI, as happened at GW4), else refresh it from the live site.
  The page is public (no login) — the full XIs, doubt %s and notes are all visible. A WebFetch summary can drop
  players, so request the 11 names per team verbatim, or paste the page. Check each team's "Last Updated" date.
- **Fantasy Football Pundit** (ledger name `FantasyFootballPundit`) — https://www.fantasyfootballpundit.com/fantasy-premier-league-team-news/
  — all 20 XIs with a per-player Start %. Promoted to main 2026-10-08. Capture its predicted XI (its 11
  listed starters per team, not a top-11-by-% reconstruction).

**On trial (capture when populated, promote once they earn it):**
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
- **GW4** — captured 2026-09-13: **FFS + RotoWire**, 220 rows each (20 teams × 11), 0 UNMATCHED after
  adding `name_mappings` rows for António Silva (BOU) and Angulo (SUN). `ffs_predicted_lineups.csv` was
  refreshed for GW4 first (it had been the Aug-18 pre-season file). All About FPL had no GW4 all-team
  article, so it was not captured (not substituted). Sources were gathered via a panel-restricted
  agent (Tier-1 URLs only, roster-checked) rather than the browser. Score after GW4 plays.
- **GW4 scored (2026-09-18):** FFS **89.1%**, RotoWire **88.2%**. ⚠️ **Curated NOT captured** — `--seed-ours` was skipped at the GW4 deadline and the roll then shifted `starting_lineups.csv`, so our own XI is lost/unscoreable for GW4 (git HEAD was a stale 427-row pre-final state). **Fix: seed-ours at the deadline BEFORE rolling** (now in the rollover memory).
- **GW5 scored (2026-10-08):** **Curated 90.9%**, FFS 90.5%, RotoWire 90.0% (Curated = our full-panel reconciled XI). FFP not captured for GW5. Everyone missed the same late rotations (Hemmings, Cash, Kluivert, Dunk, Livramento, Udogie, Stroud, Hume). Our coin-flip picks split both ways vs FFS: we were right on Man Utd LB (Shaw) and Forest (Ndoye); FFS was right on Everton (Armstrong) and Palace (Canvot).
- Per-source *ranking* was deprioritised 2026-08-28 (the odds price in form; the panel is hand-curated),
  so this is mainly (a) calibration of our own XI and (b) a one-off validation of a new source like FPL
  Edits before promoting it — not a heavy weekly obligation.

## Agent-sweep contract — evidence, not verdicts (added 2026-09-18, tightened 2026-10-08)
**Division of labour:** when team-lineup research is handed to agents, the agents **read both the predicted lineups AND the descriptive text** (previews, injury explanations, presser quotes, rotation notes, Start %s) and **return everything they find** — every XI, every quote, every number, with source + date. **The agents never set percentages or make the start/bench call.** Claude (the caller) reconciles the evidence and decides every F1–F8 value.

Lineup/team-news research (sub-agents or direct) must **return raw per-source evidence, not a synthesized IN/OUT call**. Three agent syntheses overstated a single source and got relayed as fact (Bijol "named starter"; Maddison "shoulder" — FPL said 100%; Porro/Tonali "out" — both 75% FPL doubts, only Sports Mole's cautious XI benched them).
- **Output = per-player × per-source grid, verbatim + dated** (FFS / RotoWire / Sports Mole / SportsGambler each quoted). Agent flags **source SPLITS**; the reconciliation (vs the FPL flag) is done by the caller, not the agent.
- **Cross-check every call vs the FPL `chance_of_playing`/`status` flag.** A 75% flag + one predicted-XI omission = DOUBT leaning start (~0.7), NOT "out".
- **Never cite a source not present in the return.** Label single-source claims "one source only".
- **Squad-critical players: fetch directly (no agent).**
See memory `agent-sweep-evidence-not-verdicts`.
- **GW6 captured (Fri 9 Oct ~23:45 BST, pre-deadline):** Curated + FFS + FFP + RotoWire, 220 rows each, 0 UNMATCHED. RotoWire all 20 still "Predicted Lineup". **FFS as shown:** 11 teams still "Last Updated Thu 8 Oct" (e.g. Forest starts Milenković, now out). **FFP:** 19 teams dated 9 Oct, Coventry 6 Oct; it starts Brobbey (out), Havertz, Koumas, Stroud. Curated top-11 ties broken by slot: Konsa (CB, .5 vs Mosquera .45), Neco Williams (incumbent RB, .6 tie with Aina). Raw file: `inputs/source_history/gw6_external.csv`. Resolver now handles FFP's glued initials (`J.TIMBER`, initial used to split `L.MILEY`/`J.MURPHY`), FFS's `SURNAME (FIRST)`, and `Ø`; aliases are case/accent-insensitive.
- **GW6 scored (2026-10-13):** **Curated 85.9% (189/220)**, FFP 85.9% (189), FFS 84.5% (186), RotoWire 84.1% (185). A rotation-heavy week after the break (GW5 was ~90% for all). **Nobody predicted 17 of the 31 players who started against our XI** (White, Madueke, Cash, Brooks, Diouf, Struijk, Acheampong, Cherif, Keane, Tete, Gelhardt, Maeda, Florentino, O'Reilly, Amad, Yoro, Bergvall, Senesi). 11 of our flat-1.0 players didn't start (Cook, Sangaré, Mfuni, Belloumi, Palacios, Clarke, Szoboszlai, Gvardiol, Dalot, van de Ven, Marmoush). Coin-flips: right on Konsa, João Pedro, Walton, N.Williams; wrong on Bruno G. (Lewis-Skelly), Wan-Bissaka (Cash), Mykolenko, Robinson, Dedić, Jair (Aina), Ngumoha (Gravenberch/Koumas). **Calibration flag:** our 0.8–0.99 band started only 68% (13/19; avg belief 0.89) — Eze, Timber, Gyökeres, Dunk, Ahanor, Onyeka were all too high. Worst teams: Arsenal 7/11 (every source 6–8), Spurs 8/11 (sources 6–8).
