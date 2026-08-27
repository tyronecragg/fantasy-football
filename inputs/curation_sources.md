# Curation sources — trusted panel for predicted lineups

Reference for the weekly start-probability curation (step 3b). The goal is a **stable base
panel** used for every team, with **club-specific escalation only where it matters**. Trust
here is **provisional** — it should be earned by measurement, not reputation (see the
"Earning trust" section at the bottom, tracked as a to-do).

> 📌 **Load [`league_state_2026-27.md`](league_state_2026-27.md) first.** It's the durable
> snapshot of managers, key transfers in/out, promoted sides, and injuries for the season — the
> facts that go stale fastest and cause the most predicted-XI errors. Refresh it, don't re-derive it.

> 🕒 **Golden rule — date-check everything, and re-check late.** Every article and match page must
> be confirmed current for THIS gameweek before you trust it — search results and game-guide URLs
> routinely serve a **prior-fixture or prior-season** preview (the Aug-2026 audit caught May-2026
> and Oct-2025 pages presented as current). And **absence is not disqualifying**: many previews
> only land in the **last day or two before kickoff**, so a quiet source now is worth a second look
> closer to the deadline. Considered and rejected for the predicted-XI panel: **WhoScored** (403-blocked),
> **Football365** (opinion/columns, no per-match XI), **SI/si.com** (inconsistent, frequently stale).

The base panel is the workhorse: for a settled team (Arsenal, Liverpool) three sources agree
and you're done. Escalation is a targeted second look for the ~20% of teams that are hard to
call in a given week — and **which teams those are changes weekly**.

---

## How the numbers work — writing probabilities into `starting_lineups.csv`

The file is `Player, Team, F1 … F8` (F7/F8 curated since 2026-08-21 — the optimiser plans eight fixtures, and its fixture weights already discount the later ones, so the later columns carry real beliefs rather than a copy of F6). Each cell is the probability that
player **starts** that fixture. It is **curated by hand from the sources below** — not generated —
and it stores **raw beliefs**; the pipeline normalises at consumption time.

> ✅ **TWO DISCRETE CROSS-CHECKS EVERY PASS** (added 2026-08-27 — each was skipped once and left the wrong players starting; normalisation will NOT save you, it only rescales existing values and has no idea who's injured or newly-signed):
> 1. **Zero the entire out-list, as its own step.** Take the full SportsGambler + FPL-API availability list and set F1≈0 for EVERY player out/suspended for this GW — do not rely on targeted edits to surface injuries. A stale high F1 on an injured player sails straight through a sum-normalisation. *(Caught starting despite being out: Riad, Kovačić [red], Ajayi, Coyle, Osula.)*
> 2. **Reconcile the file against the LIVE roster, as its own step.** Diff every club's `outputs/01_fpl_players.csv` squad vs the file: **ADD** starter-level new signings the file is missing (they belong in the roster now and carry a real F1), and drop players who left. *(Caught: Sávio — a £50m signing both RotoWire & All About FPL named in the XI — was entirely ABSENT from the file while the youth Mikey Moore sat at 0.42.)*
> Then check `groupby('Team')['F1'].sum() == 11` and that no cell > 1.0.

- **Sum rule.** `F1` must sum to **exactly 11** per team (the GW1 XI is a known 11). `F2`+ may sit
  **below 11 but never above.** A below-11 column is *correct* when the missing minutes belong to a
  player who isn't in the FPL roster yet (a new signing) — **never pad the gap onto another player**
  to force 11. Check `groupby('Team')['F1'].sum()` after every pass.
- **F1 models SELECTION, not survival — declare hard.** If every current source starts a player,
  set `F1 = 1.0`. **Injury *chance* is never a reason to shade F1** — a late knock means reality
  diverged, not that the estimate was wrong. Discount F1 only for genuine *selection* uncertainty:
  a contested slot, an active fitness game-time decision (manager says "we decide tomorrow"), or
  thin sourcing (e.g. only 2 current XIs exist for the team). With most of an XI declared, clear
  bench players must drop to ~0 so the contested slots keep their intended shares.
- **F2-F6 move off F1 ONLY for a NAMED reason — no generic time-decay.** The pipeline already
  discounts the future (Total XP weights fixtures 1.0 / 0.85 / 0.7 …), so tapering a nailed
  starter "because the future is uncertain" double-counts it. A player expected to start
  long-term stays at **1.0 (or close) across the whole horizon**. The only legitimate reasons to
  shade a later week: a **named returning rival** (and then it's the *displaced* player who fades,
  not everyone), a **transfer** in/out, a **suspension**, **documented rotation** (European
  midweeks, load-managed veterans), or a **fitness ramp**. Never injury chance. Make the curves
  *coherent with the threads*: injury returns **ramp up and END at the player's steady-state
  level** (a first-choice player ramps to ~0.8-0.9, not to 0.4), players being sold or losing
  their place **fade down**, new signings **integrate upward**, suspensions **clear** on the right
  week. Caught 2026-08-21: Mateus Fernandes tapered 1.0 → 0.66 with no rival named; Okafor's
  ramp plateaued at 0.39 when he converges to a starter.
- **Injury ramps need a VERIFIED return date — F3 is only ~3 weeks away.** Never give an injured
  player F3+ probability off a "when fit he starts" projection alone: check the return date first
  against the [SportsGambler injury table](https://www.sportsgambler.com/injuries/football/england-premier-league/)
  (fetchable, dated per update — our standing return-date checker; PremierInjuries is 403-blocked).
  "No return date listed" = keep the ramp at ~0 until F4+ tokens. Caught in Aug 2026: Conor Bradley
  got an F3 ramp from a season preview while the table said knee, out to 1 Jan 2027.
- **Coin-flips stay splits.** Where two players genuinely contest one slot and the sources are split,
  keep `F1` as a **calibrated split** (e.g. 0.55 / 0.45) — do **not** force it to 0/1. Confirmed XIs
  never arrive before the deadline (see "Confirmed XIs are NEVER available before the deadline"
  below), so a split is the honest final state; the optimiser weighs the probabilities directly.
- **Normalisation** (`players.normalize_start_probs`) holds declared **1.0 / 0.0 fixed** and scales
  only the uncertain players to hit the target — so a nailed 1.0 survives and only genuine
  uncertainty absorbs the balance. It runs at consumption, which is why the file keeps raw beliefs.
- **F2+ fallback.** Any stat with no future odds is projected as **F1-factor × opponent-baseline**;
  DefCon is a **straight copy of F1**. So a sensible `F1` start prob is what carries the later weeks.
- **Placeholder players.** New signings not yet in the FPL roster are added to the file **to suppress
  the incumbent they displace** (e.g. so Ben White's minutes fall when a new RB arrives). They show
  up in `reconcile` as expected **"not in roster"** notes — that's normal, not an error, and they
  join automatically once FPL adds them (Benítez, Suzuki, Cherif, Ruggeri and Elvedi all have).
  Current placeholder: **Bradley Barcola** only.
- **Name mapping direction.** `inputs/name_mappings.csv` rows are `raw spelling → canonical`
  (e.g. `player,David Raya Martín,David Raya`) and the mapper is applied to the FPL roster, the
  odds files, **and — since 2026-08-21 — the lineups file**. So a reconcile suggestion like
  `player,Walter Benitez,Walter Benítez` works as printed. **Never add a reversed row** (canonical
  → raw): because canonical names now pass through the mapper too, a reversed row silently
  un-maps them (two such rows were found and removed on 2026-08-21 — Khusanov, Murillo).

---

## Tier 1 — Base panel (predicted XIs, all 20 teams, every week)

Cross-reference these three. Where they agree → high confidence. Where they split → that's
your judgement call, and the trigger to escalate to Tier 3.

| Source | Role | Notes |
|---|---|---|
| **Fantasy Football Scout** — [team-news](https://www.fantasyfootballscout.co.uk/team-news) | **Primary, automated** | Already scraped by `starting_lineups.py` into `ffs_predicted_lineups.csv` + `ffs_team_news.md`. FPL-native; grades doubts with a % chance. |
| **RotoWire** — [lineups](https://www.rotowire.com/soccer/lineups.php) | Cross-check | Predicted *and* confirmed XIs, injury feed integrated, all 20 clubs. |
| **All About FPL** — [predicted lineups](https://allaboutfpl.com/category/predicted-lineups/) | Cross-check | All 20 teams, updates after friendlies. Good third consensus voice. |

Alternatives if one is thin: [Never Manage Alone / Yahoo "FPL GW Predicted Lineups: every PL
club"](https://www.nevermanagealone.com/) (all 20 XIs on one page — a fast whole-league
cross-check), [Fantasy Football Hub](https://www.fantasyfootballhub.co.uk/premier-league-predicted-lineups)
(press-conference round-ups), [FPL Edits](https://fpledits.com/predicted-lineups-pl)
(XIs + subs), [Squawka per-match previews](https://www.squawka.com).

**Keep it to three.** More sources add noise and conflicting signals, not clarity.

> ⚠️ **Reliability caveat (learned GW1 2026-27):** automated *fetches of FFS/other per-club
> "season preview" pages* sometimes return **stale prior-season squads** — GW1 2026-27 pulls
> hallucinated Salah, Casemiro, Ederson, Bernardo, Meslier, and other players who had already
> left. The FFS **team-news hub** and the dated **match previews** are trustworthy; a per-club
> *preview* scrape is not, unless every name is checked against the current roster. When a
> source lists a player who isn't in `fpl_data/.../players.csv`, discard that source for the week.

---

## Tier 1b — Non-FPL all-match cross-checks (general football, every fixture)

Not FPL-specific, but each publishes a **predicted XI for every match** — so they triangulate
against the FPL panel above, and a split between the two camps is your escalation trigger. Same
rule applies: run every name past the current roster (stale-squad trap).

| Source | Role | Notes |
|---|---|---|
| **Sports Mole** — [per-match predicted lineups](https://www.sportsmole.co.uk) | Standing all-match cross-check | Predicted XI + team news for every fixture; also our **freshest** source for last-minute outs/suspensions (e.g. Fofana/Andersen bans, Doku's calf). ⚠️ **Check the article date** — its game-guide pages sometimes serve a stale prior-fixture / prior-season preview (caught May-2026 and Oct-2025 pages in the Aug-2026 audit). |
| **FotMob** — [match pages](https://www.fotmob.com) | Fast predicted → confirmed | App-first; flips to confirmed XIs ~1hr before kick-off (the confirmed side is post-deadline — accuracy tracking, not selection). ⚠️ **Hit-or-miss to fetch** — some match pages render the XI, others return a JS-only shell. |
| **Goal.com** — [match previews](https://www.goal.com) | Predicted XI + freshest outs | Full predicted XI per fixture plus up-to-date suspensions/injuries; the **most reliable read for the promoted clubs** (Hull/Ipswich) in the Aug-2026 audit, where aggregators were thin. Date-check the article. |
| **Yahoo Sports** — [PL/FPL lineups](https://sports.yahoo.com) | All-20 aggregator + per-match | Runs the all-20 "FPL GW Predicted Lineups" one-pager (with Never Manage Alone) *and* per-fixture "confirmed team news + predicted XI" articles; verified current in the audit. |

> ❌ **WhoScored — REMOVED (Aug 2026).** It is **Cloudflare-403 blocked on every fetch**, for all 20 clubs — unusable in this workflow. Do not reinstate unless the block clears; use FotMob/Sports Mole for the stats-backed non-FPL read instead.

---

## Tier 2 — Injury / availability ground truth (when there's a fitness question)

Use to confirm a doubt, get a return date, or resolve a disagreement about *why* someone is
out. These feed `inputs/unavailable_players.csv` and gate every F3+ injury ramp (see the
return-date rule in "How the numbers work").

- **[SportsGambler injury table](https://www.sportsgambler.com/injuries/football/england-premier-league/)
  — THE GO-TO.** Fetchable by automated tools, per-club, shows injury type + expected return
  date, and stamps its last-update time (verified current on 21 Aug 2026). Check it before
  writing any injury ramp.
- **Plus the FPL API availability feed** (`chance_of_playing_next_round` + `news`) — authoritative
  outs / doubts / return-dates, pulled every sweep; covers most of what the removed tables did.

_(Removed 2026-08-27 — unfetchable by automated tools, so useless here: Premier Injuries (403),
Official PL injury list & Sky Sports table (both JS-rendered). Use SportsGambler + the FPL API feed.)_

---

## Tier 3 — Club-specific escalation (ONLY for flagged hard teams)

Reach here when Tier 1 **disagrees**, or the team is genuinely hard to predict — **promoted,
new manager, heavy summer turnover, or an injury cloud**. A club beat writer knows that squad
better than a national outlet filling in 20 XIs quickly.

- **Freshest injury/suspension lists** — dated **[Goal.com](https://www.goal.com)** and
  **[Sports Mole](https://www.sportsmole.co.uk)** match previews carried the most up-to-date
  outs (e.g. Fofana/Andersen GW1 suspensions, Doku's Community Shield calf) — better than the
  aggregators for last-minute availability.
- **Promoted / smaller clubs** (2026-27: Coventry, Hull, Ipswich) — local press and fan sites.
  The national aggregators are weakest exactly here.
- **Big clubs** — [ESPN week previews](https://www.espn.com/soccer) go deeper on the top six.
- **Club fan sites & local beat writers** — see the **Tier 3b directory** below for a verified
  two-fan-site + one-local-press-beat set (with the writer to follow) for every club. Best for
  injury depth and "who deputises" on a decimated squad.
- **Manager pre-match pressers** — the ground truth for "who's fit / who rotates". Official
  club channels report these.

Don't escalate for settled teams. Any three sources agree on Arsenal; the effort is wasted.

---

## Tier 3b — Club directory: fan sites + local beat (verified live Aug 2026)

The standing escalation panel — for the ~20% of teams hard to call in a given week. A fan site or a
local beat writer knows a squad's rotation and injury quirks better than a national outlet doing 20
XIs at speed. Each club has up to **two independent fan sites + one regional-press beat** (with the
named writer to follow), all verified active August 2026. **Cloudflare-blocked sites were removed
2026-08-27** (Arseblog, This Is Anfield, We Ain't Got No History, Roker Report, The Fighting Cock,
Bluemoon) and **replaced the same day with fetch-verified alternatives** (Pain in the Arsenal, The
Pride of London, Anfield Watch, Man City Square, Spurs Web; Sunderland → A Love Supreme) — so every
club is back to **two fetchable fan sites + one regional beat**. **Date-check every article** (Golden rule).

| Club | Fan site 1 | Fan site 2 | Local/regional beat (writer) |
|---|---|---|---|
| Arsenal | [Daily Cannon](https://dailycannon.com) | [Pain in the Arsenal](https://paininthearsenal.com) | [football.london/arsenal](https://www.football.london/arsenal-fc/) — Kaya Kaynak |
| Aston Villa | [My Old Man Said](https://myoldmansaid.com) | [Read Aston Villa](https://readastonvilla.com) | [BirminghamLive](https://www.birminghammail.co.uk/all-about/aston-villa-fc) — John Townley |
| Bournemouth | [AFCB Podcast](https://afcbpodcast.com) | [Somerset Cherries](https://somersetcherries.co.uk) | [Bournemouth Echo](https://www.bournemouthecho.co.uk/sport/afcb/) — Alexander Smith |
| Brentford | [Beesotted](https://beesotted.com) | [Griffin Park Grapevine](https://griffinpark.org) | [West London Sport](https://www.westlondonsport.com/brentford) |
| Brighton | [We Are Brighton](https://wearebrighton.com) | [Read Brighton](https://readbrighton.com) | [The Argus](https://www.theargus.co.uk/sport/) — Brian Owen |
| Chelsea | [The Chelsea Chronicle](https://thechelseachronicle.com) | [The Pride of London](https://theprideoflondon.com) | [football.london/chelsea](https://www.football.london/chelsea-fc/) — Bobby Vincent |
| Coventry City | [Sky Blues Blog](https://skybluesblog.co.uk) | [Let's All Sing Together](https://letsallsingtogether.com) | [CoventryLive](https://www.coventrytelegraph.net/all-about/coventry-city-fc) — Andy Turner |
| Crystal Palace | [We Are Palace](https://wearepalace.uk) | [The Holmesdale Online](https://holmesdale.net) | [football.london/crystal-palace](https://www.football.london/crystal-palace-fc/) |
| Everton | [ToffeeWeb](https://toffeeweb.com) | [Goodison News](https://goodisonnews.com) | [Liverpool Echo — Everton](https://www.liverpoolecho.co.uk/all-about/everton-fc) — Joe Thomas |
| Fulham | [HammyEnd](https://hammyend.com) | [Fulhamish](https://fulhamish.co.uk) | [West London Sport — Fulham](https://www.westlondonsport.com/fulham) — Jack Kelly |
| Hull City | [Hull City Forum](https://hullcityforum.co.uk) | [hcafcHub](https://hcafchub.com) | [Hull Live](https://www.hulldailymail.co.uk/all-about/hull-city) — Barry Cooper |
| Ipswich Town | [TWTD](https://twtd.co.uk) | [Blue Monday](https://bluemondayitfc.co.uk) | [East Anglian Daily Times](https://www.eadt.co.uk/sport/) — Stuart Watson |
| Leeds United | [The Square Ball](https://thesquareball.net) | [Leeds, That!](https://leedsthat.com) | [Yorkshire Evening Post](https://www.yorkshireeveningpost.co.uk/sport/football/leeds-united) |
| Liverpool | [Empire of the Kop](https://empireofthekop.com) | [Anfield Watch](https://anfieldwatch.co.uk) | [Liverpool Echo — LFC](https://www.liverpoolecho.co.uk/all-about/liverpool-fc) — Doyle / Gorst |
| Man City | [City Xtra](https://cityxtra.co.uk) | [Man City Square](https://www.mancitysquare.com) | [Man. Evening News — City](https://www.manchestereveningnews.co.uk/all-about/manchester-city-fc) — Simon Bajkowski |
| Man Utd | [Stretty News](https://strettynews.com) | [The Peoples Person](https://thepeoplesperson.com) | [Man. Evening News — Utd](https://www.manchestereveningnews.co.uk/all-about/manchester-united-fc) — Steven Railston |
| Newcastle | [The Mag](https://themag.co.uk) | [True Faith](https://tf1892.substack.com) | [ChronicleLive](https://www.chroniclelive.co.uk/all-about/newcastle-united-fc) — Lee Ryder |
| Nott'm Forest | [Forest Rumours](https://nottinghamforestrumours.co.uk) | [Forza Garibaldi](https://forzagaribaldi.com) | [Nottingham Post](https://www.nottinghampost.com/all-about/nottingham-forest-fc) — Sarah Clapson |
| Sunderland | [Wise Men Say](https://wisemensay.co.uk) | [A Love Supreme](https://alovesupreme.co.uk) | [Sunderland Echo](https://www.sunderlandecho.com/sport/football/sunderland-afc) — Phil Smith |
| Tottenham | [The Boy Hotspur](https://theboyhotspur.com) | [Spurs Web](https://www.spurs-web.com) | [football.london/tottenham](https://www.football.london/tottenham-hotspur-fc/) — Alasdair Gold |

_Maintenance notes (Aug 2026):_
- _Fan sites removed as dead/dormant — **Amber Nectar** (Hull, domain lapsed 2019), **Cherry Chimes** (Bournemouth, 2021), **The Scratching Shed** (Leeds, 2020). Corrected: **FYP Fanzine** (Palace) → fypfanzine.uk; SB Nation Fulham is **Cottagers Confidential** not "Cottagers Corner". Fade-alternates: A Love Supreme (Sunderland), Cartilage Free Captain (Spurs), To Hull and Back (Hull)._
- _Local beat — the London clubs' regional desk is **football.london** (per-club sections), but **Brentford & Fulham** are better served by **West London Sport** (football.london is thin there); **Crystal Palace**'s football.london section has no single named correspondent. Byline churn to watch: **Leeds** — Graham Smyth (YEP) moving to the club ~mid-Sept 2026; **Man Utd** — Steven Railston is now MEN lead (Luckhurst → The Sun)._
- _**Re-verify liveness/bylines each season.**_

---

## Confirmed XIs are NEVER available before the deadline — a hard rule

**The FPL deadline always precedes every official team-sheet.** The deadline is **90 min before the
*first* match** of the gameweek; clubs release official lineups **~60 min before *each* match**. So
even the first game's XI drops ~30 min too late, and every later game's later still. **We never get a
confirmed XI before we pick — not for a single player.**

**Consequence for our start probs (F1) — do not fight this:**
- Predicted XIs are the **ceiling of certainty** for selection. That's as good as F1 ever gets.
- A **genuine coin-flip** (two players contesting one slot, sources split) is **irreducible at the
  deadline**. Keep it a **calibrated split** (e.g. 0.55 / 0.45) and let the optimiser weigh it —
  **never push it to 0/1.** Firm F1 only as far as the *predicted* consensus is genuinely decisive.
- There is **no "confirmed-XI sweep"** that firms F1 before the deadline. Any workflow that says
  "wait for the team news / official XI" is describing information we **cannot act on**. Don't build it.

**Confirmed XIs** (RotoWire, [Sports Gambler](https://www.sportsgambler.com/lineups/football/england-premier-league/))
are still collected — but **only** as the **ground truth for post-hoc source-accuracy scoring**
(see below), never for team selection.

---

## Earning trust — the source-history ledger (`tools/source_history.py`)

Do not *assign* trust to a source — **measure** it. We store the actual lineups for every match
(`By Gameweek/GW*/lineups.csv`, `is_starting`), so each source's predicted XI can be scored
against who actually started, giving a hit rate per source. The reliable sources earn their
Tier-1 place with a number; authoritative-looking-but-inaccurate ones get demoted. Same principle
as the longshot calibration: outcomes decide.

**How it works.** Each source's predicted XI is *frozen* per gameweek in
`inputs/source_history/predicted_xis.csv` (`Season, Gameweek, Source, Team, Player`), then scored
once the gameweek plays. Names are resolved to the FPL roster by a **team-scoped resolver**
(`source_history.Resolver`) that handles full names (FFS) *and* surnames/nicknames (AAF, RotoWire) —
"Palmer" → Cole Palmer, "DCL" → Calvert-Lewin — flagging anything ambiguous rather than guessing.

- **Frozen automatically every week** by phase 2 of `weekly_update.py`:
  - **`FFS`** — the staged `ffs_predicted_lineups.csv` (`--seed-ffs`).
  - **`Curated`** — our own call, the 11 highest-F1 players per team from `starting_lineups.csv`
    (`--seed-ours`). This is the one that actually matters: it validates *our* curation, not just a feed.
- **Frozen during the deadline sweep** (external cross-checks): write a `Source,Team,Player` CSV
  (one row per predicted starter) for **RotoWire**, **All About FPL** and the sweep consensus, then
  `python tools/source_history.py --capture-csv FILE.csv --gw N`. Surnames/nicknames resolve
  automatically; **check the printed UNMATCHED lines** and add an `ALIASES` entry or a
  `name_mappings.csv` row before trusting the row. GW1's back-fill (pre-CLI) is
  `tools/seed_gw1_sources.py`, kept as a worked example.
- **Scored** by `weekly_update.py` (or `python tools/source_history.py --score --gw N`): once GW N-1
  has played, it prints a hit-rate table per source. Fails cleanly if the actuals aren't in yet.

GW1 2026-27 seeded with **Curated + FFS + AllAboutFPL** (20 XIs each; cross-source agreement ~9.9/11,
so the fringe calls are what get graded). RotoWire's live page had already flipped to confirmed XIs
by the time the ledger was built, so its GW1 pre-deadline prediction wasn't back-filled — it joins
from GW2 via the sweep. **Next step:** add per-club-type splits (settled vs promoted) once a few
gameweeks of scored data accrue.
