# Chip Strategy 2026-27 — First Half (up to GW19)

*Living document. Started 2026-08-26 (at GW2). Build on it as fixtures, form and the squad evolve.*

## Framework

- **First-half chips — Wildcard, Free Hit, Bench Boost, Triple Captain — expire at the GW19 deadline (Sat 2 Jan 2027).** Use them or lose them; no carry-over into the second half (a second set unlocks from GW20).
- **No first-half doubles or blanks.** Verified against the pipeline's `season_fixtures.csv`: every GW2–19 is a clean 10-fixture week. So — unlike the second half — these chips are **not** for DGW/BGW. They are for **fixture swings and premium home games**. Do not hoard them waiting for doubles that never come.
- The **Free Hit is not a blank-navigator** here, so it becomes a flexible tool: attack one great slate, or dodge one bad week.
- General meta (from the 2026/27 guides, see Sources): an early Wildcard spends information you don't have — fixtures/form only settle ~GW6+. Bench Boost is strongest right after a Wildcard (all 15 freshly picked for good fixtures). Triple Captain is best on a big premium at home to a promoted/weak side.

## Fixture opportunities (this season's actual schedule)

Team strength proxy = `title + top6 − relegation` from the season odds. Weakest-6 (best chip-target opponents): **Coventry, Hull, Ipswich (promoted), Sunderland, Fulham, Crystal Palace.**

### Triple Captain targets — premium at HOME to a weakest-6 side

Owned premiums:

| Player | Best TC home games (GW2–19) |
|---|---|
| **Bruno** (Man Utd) | Ipswich GW2, Coventry GW14, Sunderland GW18 |
| **Isak** (Liverpool) | Fulham GW4, Sunderland GW13, Coventry GW19 |
| **Palmer** (Chelsea) | Hull GW4, Palace GW13 |

If **Haaland** is brought in, Man City have the best TC run of anyone: home to Coventry (GW3), Sunderland (GW5), Ipswich (GW7), Fulham (GW11), Hull (GW16).

### Easiest 5-GW runs — team-level

- **Aston Villa GW11–15** — softest stretch in the half (Sun, Ips-a, Eve, Cry, Cov)
- **Spurs GW9–13** (Cry, Lee, Ips, Sun, Ful)
- **Man City GW3–7**; several top sides (Arsenal, Chelsea, Liverpool) peak together **GW15–19**

> **Caveat (2026-08-26): are Spurs/Villa rated enough?** No — strength ranks them 6th/7th (Spurs +0.34, Villa +0.26), a clear tier below the big five (Arsenal +1.39, City +1.01, Chelsea +0.92, Liverpool +0.83, Man Utd +0.67) and level with Brighton/Brentford. Their soft runs are a fixture-proof **floor** (reliable minutes, modest returns) — good for cheap enablers and Bench-Boost bench depth, **not** for the captaincy/ceiling core. Don't build a Wildcard *around* them; own the premiums and use these windows to slot value enablers.

### Collective green weeks — the real BB/WC anchor

Better than one team's run: when do the **strong-7** (the five above + Newcastle + Villa) collectively face weak sides, home-weighted?

- **GW14 — 3 strong teams AT HOME to weak sides** (Villa, Man Utd, Newcastle). Best home-heavy **Bench Boost** signal in the half.
- **GW4 — 3 home** (Villa, Chelsea, Liverpool) — strong but early.
- **GW18 — 2 home** (Man City, Newcastle); **GW13 — 2 home** (Chelsea, Liverpool).
- GW10 / GW17 have *volume* (4–5 easy) but mostly **away** — less BB-friendly.

So the Bench-Boost anchor is really **~GW14**, not "the Villa/Spurs window."

### Wildcard timing — three anchors

1. **Window-close (~GW4).** The summer window shuts ~1 Sept (≈ GW3–4). A WC right after trades *fixture info* for *roster certainty* — no late signing displacing a nailed-on pick, no target moving clubs. GW4 also has 3 strong home-soft fixtures. Earliest sensible WC.
2. **Info-settled (~GW8).** Form and fixture reads have firmed up (~6 GWs), the meta's default. Points at the GW9–15 soft runs.
3. **Fixture swing (later).** Only if the squad hasn't needed a rebuild before then.

Trade-off: earlier = more roster certainty, less fixture information; later = the reverse. Window-close (#1) is the strongest early option because it removes transfer-window risk without waiting long.

## Candidate strategies

**A — Patient swing (current lean).** Ride GW2–8 on transfers while form/fixtures settle. **Wildcard ~GW8–9** to load teams pointed at the GW9–15 soft runs (Villa, Spurs). **Bench Boost the week after the WC**, when all 15 are freshly picked for good fixtures. **Triple Captain** on Isak vs Sunderland (GW13) or into a GW15–19 premium home game. **Free Hit held to ~GW18–19** to end the half strong or cover a bad week.

**B — Early ceiling.** **TC at GW4 on Palmer (vs Hull) or Isak (vs Fulham)** — both home to bottom sides, both already owned (no set-up cost). Then WC + BB together around the GW11 Villa/Spurs swing. Banks a chip while promoted sides are still leaky, but a lower ceiling than a Haaland home game.

**C — Haaland pivot.** If tempted by Haaland, his GW3/5/7 home run makes an early TC genuinely strong; Wildcard to bring him in. Higher ceiling, but commits ~£14m of structure to one player.

## Data-driven fixture read (`tools/fixture_horizon.py`, 2026-08-26)

Extends the pipeline's projection across the whole first half (GW2–19): F1/F2 real odds, F3–F8 model-predicted, **F9–F18 extended with the same win_pred/baseline/xp_pre machinery**. Read-only; doesn't touch the pipeline/optimiser. Full matrix → `outputs/fixture_horizon.csv`.

> **Caveat, precisely:** the **win-probability model past F8 is the *same* as F3–F8** (both predicted from team strength + venue — only F1/F2 use real market odds anywhere), so the **fixture-ease matrix is as valid at GW9–19 as at GW4–9**. The genuine degradations past F8 are the defensive GBM models (clean_sheet/concede2/saves3) and the F1-odds blend not being applied, plus start held at the F8 steady-state — these touch *player XP*, not the fixture-ease read. So the fixture/transfer view is on solid ground; the player-XP extension is directional.

**Team fixture ease (win %, GW2–19):**
- **Arsenal is the strongest run by a distance** (avg 68), peaking **GW10 (83)** and **GW16–19 (69/66/68/82)**. We own 3 Arsenal assets — a natural captaincy/hold window late in the half.
- Promoted sides floor out (Hull 21, Ipswich 25, Coventry 26 avg) — the persistent green *opponents*.
- Man City have a sharp **GW12 dip (19, @Arsenal)**; Chelsea a rough **GW3 (18, @Arsenal)** then a soft GW4 (Hull, 75).

**Where our current squad struggles / peaks** (fixture-driven upside, above the appearance floor):
- **Toughest week: GW9 (upside 24).** The **Arsenal–Liverpool** fixture lands that week and hits ~5 of our players at once (Raya, Gabriel, Tzolis + Isak, Frimpong). A genuine squad-wide trough — **Free Hit candidate**, and at minimum don't chip or captain into it.
- Also soft for us: GW5 (Raya @Bri, J.Pedro @Bre, Isak @Bou), GW14 (Isak @Che, Raya @Spu).
- **Best week: GW19 (upside 33)** — Arsenal v Ipswich (H) + Liverpool v Coventry (H). GW4/GW10/GW16/GW17 (≈29) next.

**Reconciling with the generic BB call:** the league-wide "best home-heavy week" was GW14 — but for *our actual 15*, GW14 is only mediocre (Isak/Raya both away to strong sides). Our squad's own upside peaks at **GW19** (and GW10/16/17). Caveat: the Bench Boost is played on the **Wildcard** squad, not today's 15 — so treat GW19/GW10 as the current-squad read and re-run this tool on the post-WC squad before committing.

**Transfer targets ahead of the Wildcard** (`--from-gw 8 --to-gw 15`, best non-owned by projected XP over the WC window):
- **Best team runs GW8–15:** Arsenal (67% avg, miles clear), then Chelsea/Man City/Liverpool (~45–50).
- **MID:** Saka (4.75/gw), Ødegaard (4.36), Wirtz, Ndiaye, Semenyo, Mbeumo.
- **FWD:** **Haaland (4.50/gw)** stands out — doubles as the Strategy-C TC piece; then Calvert-Lewin, Igor Thiago.
- **DEF:** Van Dijk, Calafiori, Konsa, O'Reilly (Arsenal/Liverpool/City defence off the good runs).
- Read: a WC pointed at this window leans **Arsenal-heavy** with Haaland the marquee forward. Re-run with the actual WC-GW window once chosen.

**Implications for the plan:**
- **Concentration risk.** We're heavy on Arsenal (3) + Liverpool (2), so those teams' fixtures move our score *together* — that's exactly why GW9 (Arsenal–Liverpool head-to-head) is our sharpest trough. Worth diversifying slightly on the Wildcard, or at least knowing our variance is Arsenal/Liverpool-driven.
- **The three findings line up into one coherent path:** Arsenal have the best run (esp. GW16–19), Haaland is both the top non-owned buy *and* the best Triple-Captain piece, and our own upside peaks GW19. A Wildcard (~GW8) that goes Arsenal-heavy + Haaland sets up a TC on Haaland (or an Arsenal asset into GW16–19) and a Bench Boost on our best week — the transfer plan and chip plan reinforce each other.
- **GW9 Free Hit** is now a data-backed option (dodge the Arsenal–Liverpool clash), though the trough is shallow (upside 24 vs ~27 avg) — a marginal week, not a blank, so only if nothing better emerges. Default FH stays GW18–19.

## Week-by-week playbooks

Chip weeks are **bold**; everything else is normal transfers. Chip GWs are anchors, not locks — the exact week flexes with form/injuries. Fixture shorthand: lower-case = home to a weak side (juicy), `-A` vs a strong side = hard.

### Strategy A — Patient swing (current lean)

| GW | Chip | What the week looks like |
|----|------|--------------------------|
| 2  | — | Ride. Sangaré settling in; transfer only if forced (injury/price). Bank the FT if nothing urgent. |
| 3–5 | — | Normal transfers. Chelsea (Hull H, GW4) and Liverpool (Fulham H, GW4) have soft homes — captain fodder, but hold the chips. |
| 6–7 | — | Form/fixtures now settling. Shortlist the Wildcard rebuild; build team value and stack a 2nd FT. |
| **~4 or ~8** | **🃏 WILDCARD** | Two anchors: **window-close (~GW4)** for roster certainty (window shuts ~1 Sept), or **info-settled (~GW8)** to point at the GW9–15 soft runs. Own the premiums; use the soft runs for value enablers — don't build around mid-tier Spurs/Villa. |
| 9–13 | — | Ride the fresh squad; navigate on FTs. |
| **13** | **©️ TRIPLE CAPTAIN** | **Isak — Liverpool vs Sunderland (H)**. (If BB takes GW14, TC can also slide to a GW19 premium home.) |
| **14** | **🪑 BENCH BOOST** | Best home-heavy collective week: Villa, Man Utd, Newcastle all home to weak sides. Confirm your actual 15 is greenest here vs GW13/18. |
| 15–17 | — | Navigate on FTs. |
| **18 or 19** | **🎟️ FREE HIT** | End the half strong: attack the best available slate, or cover your squad's worst-fixture week. Must be spent by GW19. |

*Alt TC anchors if GW13 is taken: Liverpool vs Coventry (H, GW19) or Arsenal vs Ipswich (H, GW19).*

### Strategy B — Early ceiling

| GW | Chip | What the week looks like |
|----|------|--------------------------|
| 2–3 | — | Ride; make sure Palmer **or** Isak is captain-ready for GW4. |
| **4** | **©️ TRIPLE CAPTAIN** | **Palmer — Chelsea vs Hull (H)** or **Isak — Liverpool vs Fulham (H)**. Both home to bottom sides, both already owned — zero set-up cost. Banks a chip while promoted sides leak. |
| 5–10 | — | Transfers, build value, plan the Wildcard. |
| **11** | **🃏 WILDCARD** | Into the Villa/Spurs GW11–15 swing. |
| **12–14** | **🪑 BENCH BOOST** | Week after the WC, on the greenest collective GW in the band. |
| 15–17 | — | Ride. |
| **18 or 19** | **🎟️ FREE HIT** | Same as A — finish the half. |

### Strategy C — Haaland pivot

| GW | Chip | What the week looks like |
|----|------|--------------------------|
| 2 | — | Ride; line up the rebuild. |
| **3** | **🃏 WILDCARD** | Bring in **Haaland** and restructure. (Cost: an early WC spends info you don't fully have yet — that's the trade for the Haaland ceiling.) |
| 4 | — | Ride (City @ Man Utd is hard — **don't** BB here). |
| **5** | **©️ TRIPLE CAPTAIN** | **Haaland — Man City vs Sunderland (H)**. Or hold to **GW7 (City vs Ipswich, H)**. |
| 6–10 | — | Ride the City-heavy core. |
| **11–15** | **🪑 BENCH BOOST** | The soft-run band, greenest week for your 15. |
| 16–17 | — | Ride. |
| **18 or 19** | **🎟️ FREE HIT** | Finish the half. |

**Reading across:** A spreads risk and keeps optionality longest (chips clustered GW8–19). B locks a low-cost TC early then mirrors A's back end. C front-loads a big-ceiling TC but commits ~£14m of structure to Haaland from GW3 and burns the WC before information settles.

## Current lean (2026-08-26)

**Strategy A**, with two refinements from the 2026-08-26 review:
- **Wildcard**: lean **window-close (~GW4)** over the later GW8 — roster certainty once the window shuts outweighs the extra fixture info, and GW4 has 3 strong home-soft fixtures. Revisit if the squad doesn't need a rebuild by then.
- **Bench Boost → ~GW14**, the best home-heavy collective week (Villa/Man Utd/Newcastle all home to weak sides), not the mid-tier Villa/Spurs window.
- **Triple Captain**: Isak vs Sunderland (H, GW13), or a GW19 premium home.
- Explicitly **not** TC'ing Bruno vs Ipswich in GW2 — fine fixture, low ceiling for the only first-half TC this early. And **not** building the WC around Spurs/Villa — they're floor/enabler picks, not ceiling.

## To build on

- [x] Pressure-test the Bench Boost against the *actual current 15's* GW2–19 run → `tools/fixture_horizon.py`: current squad peaks **GW19** (also GW10/16/17), troughs **GW9** (Ars–Liv clash). Re-run on the post-WC squad before committing.
- [ ] Re-run `tools/fixture_horizon.py` weekly (F1/F2 real odds roll forward, extension updates) and after any Wildcard.
- [ ] Revisit after ~GW6 once form/fixtures settle — re-rank team strength from real results, not just pre-season odds.
- [ ] Decide the Haaland question before GW3 if going Strategy C.
- [ ] Track which chips are still live and update the lean each week.
- [ ] Note: strength ranks here are pre-season odds-implied; promoted/weak sides may over- or under-perform their price early.

## Sources

- [Premier League — chips in 2026/27](https://www.premierleague.com/en/news/4679879)
- [Fantasy Football Scout — chip strategy guide](https://www.fantasyfootballscout.co.uk/2026/08/04/fpl-2026-27-best-chip-strategy-guide)
- [All About FPL — first-half chip guide](https://allaboutfpl.com/2026/08/2026-27-fpl-chip-strategy-guide-first-half-of-the-season/)
- [RotoWire — chip strategy](https://www.rotowire.com/soccer/article/best-fpl-chip-strategy-2026-27-when-to-play-every-chip-127670)
