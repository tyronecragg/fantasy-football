# Starting lineups — FIRST-RUN sweep (GW4 2026-27)

**Compiled 2026-09-10 (first run, ~3 days pre-deadline); kept current through 2026-09-13.** GW4 deadline **Sat 13 Sep 11:00 BST**; matches Sat 13 – Sun 14 Sep. Just **after the September international break**. **WILDCARD week — the whole pool matters, not just our XV.** `starting_lineups.csv` **is built, exhaustive, and validated** (F1=GW4…F8=GW11). Still refine the contested calls at the Saturday-AM final sweep before locking the wildcard.

> 📍 **CURRENT STATE — committed values as of 2026-09-13 (single source of truth; supersedes all dated entries below where they conflict).** File: **653 rows = the full FPL roster** (`tools/sync_lineups_roster.py`; Mfuni added, 3 in `transferred_out.csv`), names byte-exact, **0 dups / 0 NaN**, bounds [0,1], **all 20 teams × F1–F8 = 11.00**, **0 FPL-ceiling breaches**. Exact per-player numbers live in `inputs/starting_lineups.csv`; the dated sections below are the chronological reasoning/audit trail — earlier same-day values were iterated on, so **trust this block over them.**

> ⛔ **Value = P(NAMED IN THE STARTING XI)** — selection prob, not minutes. Bench = 0.0. Unanimous starter → 1.0. Fractional only for a genuine slot-contest or fitness doubt; injury *chance* never shades F1 (that's the FPL availability ceiling). **F1 is firm (GW4 is imminent) — shade F1 only for a real GW4 reason; future rotation/returns live in F2–F8.** Method this cycle: GW1–3 minutes are the ground truth for *established* players; official-source predicted XIs (FFS/RotoWire panel, roster-checked) decide new signings & injury-returnees, where minutes mislead.

## 🔴 F1 (GW4) contested / watch calls — committed
- **Liverpool:** Gakpo **0.35** (adductor doubt, didn't train; Barcola primed) — biggest single swing; wide pool Munoz/Barcola/Ngumoha ~0.45 each.
- **Leeds CB:** Elvedi **0.55** / Bijol **0.45** — genuine coin-flip (Rodon out the whole window; Elvedi started GW3).
- **Chelsea:** Reece James **0.88** (minutes-managed; plays CM under Alonso though FPL-listed DEF). CB 4-for-3: Fofana 0.85 / Acheampong 0.70 / Colwill 0.45.
- **Newcastle:** Botman **1.00 (F1–F4)** → 0.60 (F5–F8, Burn back ~mid-Oct). RB Dedić 0.65 / Livramento 0.35 (handover). **Shaw (MUN) 0.75** (75% flag).
- **Spurs front:** Tel 0.92, Marmoush 0.80, Sávio 0.70, Gallagher 0.62, Porro 0.72 (75% flag), Udogie 0.52; **Maddison 0.12 (benched — NOT injured, FPL says 100%)**.
- **Everton:** Röhl 0.90, Garner 0.70, B. Johnson 0.55, Grealish 0.30 (loan signing easing in); Hackney/Maitland-Niles low.
- **Man Utd:** Rashford **0.65** (started last 2) / Šeško 0.35 (marquee signing still subbing — re-check at XI drop).
- **Others:** De Cuyper **1.00**; Awoniyi 0.92 (COV); Palacios 0.58 / Berge 0.42 (FUL); Nketiah 0.68 / Strand Larsen 0.32 (CRY).
- **Nailed 1.0 (no watch):** the GKs (Raya/Donnarumma/Verbruggen), Gabriel, Konsa, Guéhi, Gvardiol, Muharemović, Ajer, Palmer, Rogers, Tzolis, Mbeumo, E. Anderson, João Pedro, Isak, Igor Thiago, Wissa, Gibbs-White, Groß, Szoboszlai, etc.

## 🔁 Key F2–F8 handovers (named-reason tapers)
- **Arsenal RB:** White 1.00→0.52 / **Timber** 0→0.48 — imminent return, RB rival; handover crosses ~F6–F7, ends ≈even (the Reddit "Timber clearly first-choice" take was discounted). **Konsa 1.00 all window** (Saliba out to ~21 Nov, beyond F8); Mosquera 0.
- **Brighton:** De Cuyper 1.00→0.58 as **Mitoma** returns (~F3), Mitoma ramping to 0.51.
- **Man Utd:** pivot Mainoo/Tielemans/**Baleba** = 3-way once Baleba's fit (~F2/GW5); Tielemans floors 0.65 (not cratered). CB: Maguire/Yoro/**de Ligt** share as de Ligt returns.
- **Palace/Newcastle:** Nketiah fades as **Mateta** returns ~F5; Burn returns F5 (Botman/Burn/Schär squeeze).

*(The dated ledger below is the historical audit trail. Read it for the reasoning behind a call, but the block above is the current committed state.)*

## PER-CLUB — GW4 (OUT / doubt / returns / new signings / predicted-XI note)

**Sunderland** — OUT: Diarra (hamstring, ~GW8-9), Mundle (knee, ~year-end), Adingra. Meunier nailed RB; Danso in; **Fofana pushing to start**. XI ~ Roefs; Meunier, Mukiele, Ballard, Reinildo; Xhaka, Sadiki; Hume, Le Fée, Angulo; Brobbey.
**Arsenal** — OUT: **Saliba (back, WC-aggravated, ~21 Nov / GW11-12)**. Doubts: **Mosquera major** (muscular), **Timber minor** (may return midweek), **Bruno G. 75%** (our flag). CB area fluid; Calafiori/Gabriel start.
**Aston Villa** — OUT: **Onana (ACL, WC — beyond F8)**, Goretzka (~Oct, GW6-7), **João Gomes SUSPENDED → returns GW5**, Madjo. Doubts: **Manzambi (WC knock, major)**, Cash (minor). **[RW-reconciled: Mings & Cash START; Torres & Wan-Bissaka benched — SM's Torres/WB swap NOT corroborated.]** Mbaye (£47m) integrating. XI ~ Suzuki; Cash, Mings, Lindelöf, Maatsen; Kamara, Barkley; McGinn, Buendía, Hemmings; Jackson.
**Nott'm Forest** — OUT: Savona (~GW5-6). Doubts: **Sangaré (calf, ~0.4 — RW leaves him out)**, Savio (fitness watch). **Delap + McAtee + Schlager start**. Gibbs-White 75%. XI ~ Sels; Williams, Milenković, Murillo, Cunha; Schlager, McAtee, Muñoz; Ndoye, Gibbs-White; Delap.
**Bournemouth** — OUT: Kroupi (~Nov), Adli (calf), Milosavljević (knee, first back), J. Araújo (~Nov). **[RW: Adams starts; Christie bench.]** **Tavernier nailed**. XI ~ Petrović; Smith, Hill, Silva, Truffert; Adams, Scott; Rayan, Kluivert, Tavernier; Evanilson.
**Brentford** — OUT: Jensen, van den Berg, Milambo, Dasilva. Doubt: **Collins (calf, likely out → Schuster debut)**. **M. Sangaré starts** (Janelt–Sangaré pivot). XI ~ Kelleher; Kayode, Ajer, Schuster, Lewis-Potter; Janelt, Sangaré; Ouattara, Damsgaard, Schade; Thiago.
**Coventry** — OUT: Wright, Woolfenden, Kesler-Hayden. Onyeka doubt. Revert to back four. **Amenda starts CB** (our new DC prior). XI ~ Rushworth; Thomas, Amenda, van Ewijk, Dasilva; Grimes, Rudoni; Yirenkyi, Onyeka, Mason-Clark; Awoniyi.
**Brighton** — OUT: **Mitoma (hamstring, post-break delay → ~GW5+)**, Minteh, Hinshelwood, Wieffer, Tzimas, Ferguson. **Rutter likely back**; Azeez near debut. **De Cuyper + Verbruggen start**. XI ~ Verbruggen; Boscagli, Vušković, Kadıoğlu, De Cuyper; Kostoulas…
**Chelsea** — OUT: **Henderson (wrist, WC → ~GW5)**, Emegha. Doubt: **Caicedo (calf, RW benches — Lavia in)**, Palestra. **[RW: Colwill benched too.]** **Rogers recalled (4 assists off bench)**. XI ~ Martínez; Fofana, Lacroix, Acheampong, Hato; Lavia, James, Neto; Rogers, Palmer; João Pedro. **(Palmer/Rogers/João Pedro all start.)**
**Hull City** — OUT: Butland, Matazo (ACL), Gyabi, Hughes, Morita, Ansah. Doubts: Iroegbunam, Norton-Cuffy, McNair. Gelhardt/Crooks back in training. **Egan starts** (new deal). XI ~ Tzolakis; Ajayi, Egan, Mendy; Coyle, Slater, Gourna-Douath, Giles; Belloumi, Stroud; McBurnie.
**Crystal Palace** — OUT: Riad, Mateta, Ahanor, Chilwell. Doubt: **Sarr (back, likely misses → ~GW5)**. Wharton recalled; **Khalaili starts RWB** (our new DC prior). XI ~ Henderson; Canvot, Disasi, Richards; Khalaili, …, Wharton, Mitchell; Kamada, Pino; Strand Larsen.
**Ipswich** — OUT: Philogene, Matusiwa, Taylor, Tuanzebe. **Flemming pushing to start** (may displace Emersonn). Scherpen starts GK (Button bench). XI ~ Scherpen; O'Shea, Diop, Greaves, Davis; Palacios, Lukić; Fatawu, Enciso, Maeda; Emersonn.
**Spurs** — OUT: **Mudryk (ankle, 6-8wk → ~GW11-13)**, Xavi (ACL, out all F2-F8), Odobert, Kulusevski. **[RW: Kudus & Maddison BOTH benched; Udogie & Marmoush START (not Robertson/Solanke).]** XI ~ Kinský; Porro, van Hecke, van de Ven, Udogie; Tonali, Bentancur; Sávio, Fernandes, Tel; Marmoush.
**Everton** — OUT: **Nørgaard (~Oct, GW7-8)**. Garner fitness doubt. **Grealish returning & starts**. **[RW: Garner starts RB; Maitland-Niles benched — SM's M-N-at-RB not corroborated.]** **Mykolenko starts**. XI ~ Pickford; Mykolenko, Branthwaite, Tarkowski, Röhl; Garner, Hackney; Grealish, Dewsbury-Hall, George; Barry.
**Leeds** — OUT: Rodon (~GW10-11), **Gudmundsson was flagged 'i' but is FULLY BACK (per Tyrone) → starts**. **Bijol BENCHED (0.00)** — GW3 sub, XI full (RW confirms). **Muharemović starts** (our DC prior). XI ~ Trafford; Justin, Elvedi, Muharemović, Gudmundsson; Tanaka, Ampadu, Bogle; Okafor, Stach; Calvert-Lewin.
**Newcastle** — OUT: **Burn (ankle, OUT GW4 → back ~GW5)**, Joelinton, Osula. **[RW: Dedic benched; Livramento STARTS at RB.]** Livramento back from calf (45' midweek). **Botman starts.** XI ~ Horníček; Hall, Thiaw, Botman, Livramento; González, Miley; Barnes, Willock, Elanga; Wissa.
**Liverpool** — OUT: **Ekitiké (Achilles, long)**, Leoni (ACL, long), **Bradley (knee, ~Nov)**, Chiesa (~GW7). Doubts: **Gakpo (adductor, MAJOR)**, Gomez (close). **Barcola fit** (cramp only). **RB: Araújo now preferred** (started, impressed), Szoboszlai can fill, **Frimpong out of favour**. Isak starts.
**Fulham** — OUT: **Cairney (knee surgery, ~GW7)**, Muniz, King. Doubt: Tete (hamstring), Sessegnon (GW4 too soon → GW4-6).
**Man Utd** — OUT: **Ugarte (knee, WC — most of season)**, Baleba (ankle, debut ~GW7), de Ligt (back, long), Heaton. Doubts: Amad (ankle). **[RW+GW3: Rashford STARTS; Šeško (~0.36) benched; Maguire/Shaw/Tielemans keep places — SM's Yoro/Mazraoui/Santos/Šeško swaps NOT corroborated. Andrey Santos was a false-positive (matched Cunha's middle name), fixed to 0.36.]** **Bruno starts.** XI ~ Lammens; Dalot, Maguire, Martínez, Shaw; Tielemans, Mainoo; Mbeumo, Fernandes, Cunha; Rashford.
**Man City** — clean bill. Doubts: **Doku (calf, major — Maresca won't rule out derby squad)**, O'Reilly (back, "probably" fit). Enzo available. XI ~ near-unchanged.

## F2–F8 RETURN / INTEGRATION / SUSPENSION TIMELINE
- **GW5:** João Gomes ban ends (AVL); Henderson (CHE ~GW5); Sarr (CRY ~GW5); Burn (NEW ~GW4-5); Doku (MCI ~GW4-5); Mitoma (BHA ~GW5+); Savona (NFO GW5-6).
- **GW6–7:** Goretzka (AVL, Oct); Chiesa (LIV ~GW7); Cairney (FUL ~GW7); Baleba (MUN debut ~GW7); Nørgaard (EVE GW7-8).
- **GW8–9:** Diarra (SUN ~GW8-9).
- **GW10–11:** Rodon (LEE ~GW10); Kroupi & J. Araújo (BOU ~Nov).
- **GW11–13:** Saliba (ARS ~GW11-12); Mudryk (TOT ~GW11-13); Bradley (LIV ~Nov).
- **Beyond F8 / season-long:** Onana (AVL ACL), Ugarte (MUN), Xavi Simons (TOT ACL), Ekitiké & Leoni (LIV), Matazo (HUL ACL), Kulusevski/Odobert (TOT), Mundle (SUN).
- **New-signing / role ramps:** Šeško (MUN, rotation w/ Rashford — benched GW4), Enzo (MCI, available), Delap+McAtee (NFO, starting), Araújo-at-RB (LIV, now preferred), Livramento (NEW, starts GW4, ramp to full), Fofana (SUN, GW5-8), Mbaye (AVL, GW4-6), Flemming (IPS, into XI soon), Azeez (BHA, GW4+), Grealish (EVE, starts), Rutter (BHA, minutes ramp GW4-5).

## INTERNATIONAL-DUTY / WORLD-CUP INJURIES (flagged)
Onana (Belgium, ACL), Saliba (aggravated at WC), Manzambi (Switzerland), Ugarte (WC), Henderson (England, wrist), Livramento (calf, withdrew from England squad), Mitoma (post-break delay).

## 🔎 VERIFY AT THE FINAL PRE-DEADLINE SWEEP (Sat AM)
1. **Gakpo (LIV adductor)** — squad-critical. 2. **Bijol (LEE) start vs sub.** 3. Mosquera/Timber (ARS CB). 4. Doku/O'Reilly (MCI). 5. Collins→Schuster (BRE). 6. Rutter (BHA). 7. Amad/Rashford & Šeško-starts (MUN). 8. Livramento/Burn (NEW). 9. Sarr (CRY), Sangaré (NFO), Manzambi/Cash (AVL), Caicedo (CHE), Flemming (IPS), Maddison/Kudus (TOT).

## ✅ BUILD UPDATE — full starting_lineups.csv written + RotoWire reconciliation (2026-09-10, later same day)
`starting_lineups.csv` was **fully built** for GW4 (supersedes the "untouched" note above).

**Method:** F1 base = **GW3 actual XIs** (FPL data, 220 starters) at 1.0; sweep overrides layered on (injuries→0/ramp, suspensions, returns, new signings, doubts); **F2–F8** carry F1 forward with the return/integration ramps above (resolve to steady-state by F8, which the master holds for F9–F12). **Every F-column normalised to 11 per team.** Robust surname matcher (translit ø/ß/ı/đ) after an accent/abbrev bug; 11 missing players added (Colwill, Maitland-Niles, Pinnock, Q. Timber, Button, + injured Saliba/Onana/Mudryk/Henderson/Cairney).

**RotoWire added as a 2nd MAIN source** and reconciled vs Sports Mole + GW3 actuals. RotoWire OVERTURNED several single-source (Sports Mole) swaps — weighted RotoWire+GW3 (2) over Sports Mole (1):
- **Gakpo → 0.30** (RotoWire benches, adductor). **Man Utd:** Rashford starts / **Šeško 0.36 benched**; Maguire/Shaw/Tielemans keep places (Yoro/Mazraoui/Santos were SM-only; **Andrey Santos false-positive 1.0→0.36 fixed** — matched Cunha's middle name). **Villa:** Mings/Cash start / Torres/WB benched. **Spurs:** Udogie/Marmoush start / Robertson/Solanke benched; Kudus/Maddison both benched. **Everton:** Maitland-Niles benched (Garner in). **Newcastle:** Dedic benched, Livramento in. **Chelsea:** Colwill benched.
- **RotoWire confirmed:** Leeds (Gudmundsson in / Bijol out), Liverpool RB = Araújo, City Doku/O'Reilly out, and all our nailed starters.

**FFScout added as a 3rd MAIN source (10 Sep):** confirms the bulk of the above (Araújo RB, Gakpo out, Rashford>Šeško, Maguire/Shaw/Tielemans retained, Doku/O'Reilly out, Villa Mings/Cash, Spurs Udogie/Marmoush + Kudus/Maddison benched, Leeds Gudmundsson/Bijol, Forest Schlager). FFScout **disagreed with RotoWire on three → reconciled as coin-flips (2-of-3 wins):** **Chelsea (back-5): Acheampong 0.52 / Colwill 0.48** for the swing CB — Acheampong is the GW3 incumbent + RW; Colwill = FFScout+SM. Four defenders locked (Fofana, Lacroix, James, Hato); **Newcastle RB — Dedic 0.46 / Livramento 0.42** (FFScout+GW3 = Dedic; RW+sweep = Livramento — genuine 50/50); **Everton — Maitland-Niles 0.53 back at RB** (FFScout+SM; RW had Garner). Gakpo held ~0.33 (all three lean out; FFScout 75% *chance-of-playing* flag = likely sub, not start).

**Corrections applied this session (vs first-run):** Gudmundsson 1.0 (fully back, per Tyrone) → **Bijol 0.00**; **Burn → 0.15** (out GW4, back GW5) *[later cut to 0.00 in the curation pass below — returns after the deadline]*; Livramento 0.62.

**Two intentional overrides of STALE FPL flags** (deadline-verify when flags refresh): **Gudmundsson** (FPL still 'i', Tyrone says back) and **Livramento** (FPL 'i', sweep+RotoWire say in XI).

**Validation:** 437 unique players, 0 dups, 0 NaN, bounds [0,1], **all 20 teams × F1–F8 sum to 11.00**. Spread teams (MUN/TOT/AVL) eyeballed — contests on the right players. **Back-3/5 sides (Chelsea, Coventry, Palace, Hull, Leeds, Forest) are handled correctly** — F1 = GW3 actual XIs bakes the shape in; sum-to-11 is formation-agnostic. Only Chelsea needed a hand-fix (a returning defender, Colwill, was inserted → paired with Acheampong as the swing CB in a back-5). FPL position *labels* mis-tag some wing-backs/DMs (Q. Timber, Khalaili, Murillo, Dasilva) but the start probs are right.

## ✅ CURATION-CHECK PASS — FPL-flag reconciliation (2026-09-10, later same day)
Ran the pipeline curation checks; fixed every flagged inconsistency. File now **fully consistent with FPL availability data**.

- **Chelsea F1 10.52 → 11.00:** duplicate Colwill row — the hand-added `Levi Colwill` (0.476 vector) didn't match the FPL roster name **`Levi Samuels Colwill`**, so it was dropped at consumption. Merged the vector onto the canonical row, deleted the dup. **Rows 437 → 436.**
- **Collins (BRE) 1.00 → 0.25:** was stale — normalization had water-filled him back to 1.0 because Brentford was a starter short. Now reflects the fresh FPL 25% calf flag; **Sepp van den Berg → 0.75** as deputy CB (ramp 0.4/0.15…). Brentford holds 11.
- **Injured trio F1 → 0.00 (accuracy, not conservatism, per Tyrone):** **Burn** (back 14 Sep, *after* the 13 Sep deadline → can't play GW4), **Amad** (unknown return → unavailable), **Manzambi** (back 12 Sep but just-returned fringe → won't start). Freed share re-normalized to the real deputies (NEW Dedic/Livramento, MUN rotation, AVL back line). F2+ return ramps retained.

**Result:** **0 F1-over-FPL-ceiling breaches** (every injured/suspended at 0, every doubtful at/under its chance-of-playing cap); **all 20 teams × F1–F8 = 11.00**; **436 rows**, 0 dups, 0 NaN. All 436 exact-match the FPL roster (Pascal Groß etc. verified) → every player valid in the optimiser. No further **data-driven** adjustments remain — what's left is weekend team-news judgment (the coin-flips on the verify list).

**NEXT:** re-run `pipeline.run` (refresh master — it still holds pre-edit values) → wildcard optimiser (`max_transfers=15, num_fixtures=12, wildcard_f1=True`). Run the **Saturday-AM final sweep** on the verify list (esp. the coin-flips) as the final gate before committing the wildcard.

## ✅ WILDCARD SQUAD STARTER-VETTING PASS (2026-09-11)
Vetted a 15-man wildcard draft for long-term (GW4–GW11) starter security — web research (FFScout/RotoWire/Sports Mole/club + confirmed XIs) cross-checked against **hard GW1–GW3 minutes** from the FPL data. Reminder: this is a **heavy-transfer season** — Isak→LIV, João Pedro→CHE, Donnarumma & Semenyo & Ndiaye & Enzo & Guéhi→MCI, Tzolis→ARS are all real (confirmed in the FPL roster); the GW3 XIs are NOT scrambled.

**Nailed, no concern:** Verbruggen (BHA #1), Donnarumma (MCI #1, Ederson sold), Palmer (83/90/90), Saka (68/90/89), Isak (90/90/64, fit & scoring), João Pedro (90/90/90), Botman (Schär/Burn now depth), Szoboszlai (locked MID — minor RB-cover watch per Iraola).
**Genuine starters, minor caveat:** **Tzolis** (ARS, £34m, 76/45/90, 4 assists — only Eze rotates); **Semenyo** (MCI, £64m — web flags City depth but he's played **every minute 90/90/90**, watch UCL weeks only).

**Corrections applied to starting_lineups.csv this pass:**
- **Nketiah (CRY) 1.0→[0.55,0.55,0.50,0.25,0.12,0.08,0.08,0.08]:** only starts because **Mateta injured ~4wk**; 3rd-choice behind Mateta + club-record Strand Larsen; never finishes a game (55/68/45). Modelled the single striker slot as a ramp — **Strand Larsen [0.45,0.45,0.45,0.25,0.18,0.12,0.12,0.12]** splits it early, **Mateta [0,0,0.05,0.50,0.70,0.80,0.80,0.80]** reclaims it from F4 (~GW7).
- **De Cuyper (BHA) 1.0→[0.90…0.80]:** minutes fine (78/90/77) but Hürzeler may use him as a winger; Kadıoğlu (fit) + new £9.5m Hadjam squeeze LB. Modest trim (starting now, squeeze risk later).
- **Pinnock (COV) 1.0→[0.80…0.70]:** just arrived (0/12/90 — first full start GW3), 33yo, 3 CBs for 3 slots with Thomas + Amenda → age/rotation trim.
- All three teams rebalanced to 11 across F1–F8.

**Retractions / no-change (agent errors caught):**
- **Gabriel (ARS):** agent claimed a hamstring "on international duty → out weeks" — **FALSE.** Our FPL flag = `a` (available), no news, 90/90/90. Held at 1.0. (No international break affected him.)
- **Fofana (CHE):** deep-CB-competition point stands (Colwill/Chalobah/Badiashile/Acheampong/Lacroix/Hato/Anselmino) but **Tosin has left** — agent listed a departed player. Held at 1.0 (starting −/68/90; risk is injury, not rotation).

**GW3-actual vs current-F1 audit (whole file, start_min==0 = started):**
- **Started GW3 but now <1.0 (20):** all deliberate shades — injury/doubt (Collins 0.25, Gakpo 0.33), coin-flip/new-signing contest (Dedic 0.52, Acheampong 0.52, Angulo 0.54, Emersonn 0.55, Adams 0.72), or heavy-rotation squads where a GW3 start ≠ a nailed slot (MUN Rashford/Maguire/Tielemans/Shaw, TOT Marmoush/Udogie/Gallagher, EVE Garner/B.Johnson, AVL Mings/Cash, NFO Schlager/Gibbs-White).
- **At ≥0.9 now but did NOT start GW3: NONE** (0 unmapped names) — no false promotions; returners (Livramento 0.48, Joao Gomes 0→F2, Mateta 0→ramp) are all held *below* 1.0 as they ramp. Note: **Gudmundsson actually started GW3 (90m)** — his 1.0 is legitimate (earlier "GW3 sub" note was wrong).

## ✅ EXHAUSTIVE-FILE + 2nd WILDCARD-DRAFT VETTING (2026-09-11, later)
**starting_lineups.csv is now exhaustive:** reconciled to the full FPL roster via new **`tools/sync_lineups_roster.py`** (wired in as roll step 4/4, opt-out `--no-roster-sync`). **436 → 655 rows** (whole roster), all beliefs preserved, every name byte-exact to the roster (kills the exact-match-vlookup silent-zero class — e.g. the `Walter Benitez`→`Benítez` drift, fixed here). 0.0 rows are math-inert (absent already `.fillna(0.0)`; normalize's water-fill is multiplicative so a 0.0 is never lifted) — the value is a **holistic view** so no emerging starter is ever out of sight. The weekly roll keeps it synced going forward.

**2nd wildcard draft vetted** (adds Egan/Mbeumo/Cherki/Wissa vs the 1st). Names: all 15 byte-exact ✓. New names by minutes: **Egan 90/90/90, Mbeumo 90/90/90, Wissa 89/82/90 = nailed**; **Cherki 27/82/66 = rotation (never a full 90)**. (NB 2026-09-11: the "Woltemade barely plays" competition note is stale — **Woltemade has LEFT Newcastle**; see the cumulative-roster / transferred_out section below.)
**Three false 1.0s corrected (each with a named deputy so the trim survives normalization):**
- **Cherki (MCI) 1.0→[0.65,0.70…]:** rotates the attacking-mid slot with **Foden** — who was wrongly at 0.0 despite starting GW1–2 (82/90); raised **Foden 0.0→[0.35,0.30…]**. (Cherki's flat-1.0 XP was the most inflated in the draft.)
- **De Cuyper (BHA) 1.0→[0.90…0.80]:** raised **Hadjam** as LB deputy (residual).
- **Pinnock (COV) 1.0→[0.82…0.75]:** raised **Jake Bidwell** as bench-CB deputy (residual).
- ⚠️ **Mechanism note:** the *earlier* De Cuyper/Pinnock trims had **silently reverted to 1.0** — normalize water-fills a lone sub-1.0 player back up when the rest of the team is 11×1.0. A trim only sticks if a teammate is raised at the same time. All three teams now sum to exactly 11 across F1–F8 and the trims hold.

**Re-run `pipeline.run`** before the optimiser so Cherki/De Cuyper/Pinnock XP drop to reality.

## ✅ 3rd WILDCARD-DRAFT VETTING (2026-09-11, later)
New names vs prior drafts: Ben White, Khalaili, Muharemović, Victor Munoz, Igor Thiago. All byte-exact ✓. By minutes:
- **Nailed, 1.0 correct:** **Ben White** (ARS, 90/90/90 — Timber & Saliba both 0 mins all season, White undisputed RB), **Muharemović** (LEE, 90/90/90), **Igor Thiago** (BRE, 83/90/90).
- **Khalaili** (CRY, 35/75/90) — ramped into the XI, now playing 90s; **left at 1.0** (trending nailed).
- **Victor Munoz (LIV) 1.0→[0.62,0.62,0.60…] — corrected:** cheap (£6.5m) rotation piece in a stacked midfield (Szoboszlai/Wirtz/Mac Allister nailed); never locked at 90 (27/90/64). Flat 1.0 inflated his ~24 XP. Rotates the 4th-mid slot with **Gravenberch** (78/19/26, established, now benched) → raised Gravenberch as deputy (residual). Liverpool holds 11 across F1–F8.

**Re-run `pipeline.run`** so Munoz's XP drops to reality before the optimiser.

## ✅ NAILED-STARTER WEB VERIFICATION — 7 players (2026-09-11, pre-GW4)
Web-checked current role / competition / injury / GW4–GW11 security. Sources dated Aug–Sep 2026.
**Three GKs + Guéhi — all confirmed nailed, hold 1.0:**
- **Kinský (TOT) 90/90/90:** #1 because **Vicario loaned to Juventus** (summer, non-mandatory £8.6m option-to-buy). Kinský signed an extension affirming No.1 under De Zerbi. No GK competition, no injury. **Secure through GW11.**
- **Guéhi (MCI) 90/90/90:** confirmed £20m Jan-2026 Palace→City, deal to 2031. Formed the CB partnership during City's injury crisis (Gvardiol broken leg, Dias hamstring, Stones thigh). Other CBs: **Dias, Gvardiol, Khusanov, Aké, Stones**. Playing every minute now = nailed. ⚠️ CORRECTION 2026-09-11: the research agent wrote "Guardiola" — **Guardiola is GONE; City's manager is Enzo Maresca** (league_state doc L119/L208; "Guardiola still at City" is a flagged stale-2025-26 phantom, L95). So the "Pep rotates heavily" premise was doubly wrong — wrong manager, and we have no read on Maresca's City CB rotation. **No taper applied; hold 1.0, watch when the injured CBs actually return.**
- **Henderson (CRY) 90/90/90:** undisputed #1 + club captain, new contract agreed. No cup-keeper split, no injury. **Secure.**

**Emerging-threat scan on 4 "nailed" attackers/defenders:**
- **Tzolis (ARS LW) — mostly secure, mild risk.** First-choice left (Martinelli sold to Al-Hilal). Threat = **Eze** can cover left (subbed on for Tzolis at Villa Park for workload). Likely starter but rotation/hook risk. Hold ~1.0, eye Eze.
- **Semenyo (MCI) — secure when fit, rotation environment.** Marquee £62.5m signing. Competes with **Doku / Cherki** (and Grealish) for wide slots in 4-3-3; Doku had an early-season calf issue which opened the door. Nailed while scoring, but City wing rotation is real. Moderate risk.
- **Muharemović (LEE CB) 90/90/90 — secure.** Club-record £34m, starts left of Farke's back three (replaced the departed Struijk). Partners Rodon/Bijol. No returning-player threat found. Low risk.
- **Khalaili (CRY RWB) — secure & strengthening.** ⚠️ Not a conventional "wing-back" — he's a right WB/winger. **Daniel Muñoz SOLD to Nottingham Forest (~£22m)**, removing the main competitor → helps Khalaili. Residual competition = **Óscar Mingueza** (versatile cover) + Mitchell. Ramped into 90s (35/75/90). Low–moderate risk.

**Net flags for GW4–11:** none outright insecure. Watch-list order: **Semenyo > Guéhi > Tzolis** for rotation; Henderson/Kinský/Muharemović/Khalaili clean.

## ✅ CUMULATIVE-ROSTER DISCOVERY + transferred_out.csv (2026-09-11)
**The FPL roster feed is CUMULATIVE, not lagging** (Tyrone's call, verified): `ingest.load_fpl_players()` keeps every player who has appeared in a PL squad this season, so a player who has since **left the league** still shows with his old club (e.g. Woltemade still listed at Newcastle after his deadline-day loan to Juventus). Intra-PL movers are fine — the roster already has them at their NEW club (Enzo/Bruno G./Konsa/Baleba/Tielemans/Ndiaye/Tosin→Spurs/Grealish→Everton all correct).

**New mechanism: `inputs/transferred_out.csv`** (columns `name,team,note`) — a manual exclusion list the sync subtracts from the roster each run, so league-leavers are neither added nor kept. `tools/sync_lineups_roster.py` now reads it; it reports how many listed names were present vs already-absent (prunable). **Add a name here the moment a player leaves the PL.**

**Seeded from the league_state doc's "departures that still appear in roster data" (L89) + the deadline-day check**, keeping only leavers actually lingering in our roster (all were 0.0, so no team-sum impact):
- **Nick Woltemade** (NEW → Juventus loan, deadline day)
- **Tijjani Reijnders** (MCI → Al Qadsiah)
- **Cristian Romero** (TOT → Atlético)
Checked and NOT added (traps avoided): "Rodrigo" matches only the active **Rodri** (MCI) + **Rodrigo Muniz**/**Bentancur**; **Nico González** is a Newcastle *arrival*, not a leaver. Named leavers already absent from the roster (Salah, Son, Casemiro, Gordon, Trossard, etc.) need no entry. **File: 655 → 652 rows**, all teams still F1–F8 = 11.

**Newcastle striker update (both Isak & Woltemade now gone):** **Matías Fernández-Pardo** (£51m, Lille, deadline day) is the signed replacement. Handover over GW4–11 — **Wissa leads the line early, then shifts to a No.10** as Pardo integrates at CF. Wissa stays a **nailed starter** either way (role changes, not minutes) → F1 held at 1.0; Fernández-Pardo is a F2+ ramp to watch (not yet edited — low-confidence timing).

## ✅ GW4 PRE-DEADLINE PREDICTED-XI SWEEP — Bournemouth/Brentford/Coventry/Brighton (2026-09-12)
Web-checked current GW4 predicted XIs + team news. Sources: FFS predicted line-ups (12 Sep), Sports Mole predicted-XI + team-news pages (11–12 Sep), Premier Fantasy Tools press conferences (11 Sep), aggregated previews. ⚠️ FFS page fetched a STALE last-season squad (Semenyo/Mbeumo/Wissa/Toney/Pinnock@BRE) via WebFetch — discarded; used its search-summary + Sports Mole instead.

- **Bournemouth (4-2-3-1):** Petrovic; Smith, Hill, Silva, Truffert; **Christie + Scott** (Sports Mole) — but pivot CONTESTED: FFS-summary had **Cook + Scott**, and **Tyler Adams** also in the mix (all fit). Med confidence on the pair; Scott looks the most secure of the four. Front: Rayan + Kluivert wide, Tavernier at 10, **Evanilson** ST. OUT: Kroupi(foot), Adli(calf), Milosavljevic(knee), Araujo(thigh). Rose: "everyone [else] fit." Europa League = low rotation THIS week (EL match later); predicted XI is full-strength.
- **Brentford (4-2-3-1):** Kelleher; Kayode, **Ajer + Schuster** (CB), Lewis-Potter; **Janelt + Sangaré** pivot; Ouattara, Damsgaard(10), Schade; **Igor Thiago** ST. HIGH conf. **Nathan Collins OUT — calf, ruled out until int'l break** (came off after 17' vs Sunderland); the 25% flag is now a definite miss. Schuster deputises at CB (NOT van den Berg — groin, out). Also OUT: Jensen, van den Berg, Milambo, Dasilva.
- **Coventry (back THREE, 3-4-2-1):** Rushworth; **Amenda, Bobby Thomas, Ethan Pinnock** (all 3 CBs start — Pinnock NAILED); WBs van Ewijk + Jay Dasilva; Grimes + Onyeka; Rudoni + Mason-Clark (dual 10s); **Awoniyi** ST. Med-HIGH conf on back three (Sports Mole labels it "4-2-3-1" but personnel = 3 CB + 2 WB; a few outlets floated a back-four revert — low prob). Amenda kept his spot despite being "bullied by Haaland" last week. OUT: Haji Wright(thigh), Woolfenden, Eccles(not fit), Kesler-Hayden. IN/returning: Gboho, Mfuni (bench). Tchaouna/Simms bench.
- **Brighton (4-2-3-1):** Verbruggen; **Kadıoğlu (RB)**, Dunk, Vušković, **Boscagli (LB)**; Gross + Ayari; Gomez, Yalcouye, **De Cuyper**; **Kostoulas** ST. KEY: **De Cuyper starts but as a makeshift LEFT WINGER, NOT at LB** — LB is Boscagli, Kadıoğlu at RB (new signing Hadjam not starting). **Rutter = DOUBT (ankle setback), NOT starting** but "close" to squad (75% flag → bench at best). Vušković starts at CB. Conference League = NO rotation, XI unchanged from Leeds draw. OUT: Mitoma, Minteh, Wieffer, Hinshelwood, Ferguson, Tzimas, Yohanna. DOUBT: Rutter, Azeez.

## ✅ GW4 PRE-DEADLINE PREDICTED-XI SWEEP — CHE / HUL / CRY / IPS (2026-09-12)
Deadline Sat 13 Sep 11:00 BST. Sources dated 11–12 Sep 2026: RotoWire lineups, Sports Mole previews (604821 CHE·604830 CRY), Fantasy Football Scout team-news (12 Sep), VAVEL GW4 preview, Last Word on Sports. Fixtures: Chelsea v Hull; Crystal Palace v Ipswich.

**CHELSEA (Xabi Alonso, 3-4-2-1 / RotoWire calls it 3-2-3-2):** Martinez; **Lacroix nailed** at CB; James + Hato wing-backs; Lavia in midfield (**Caicedo OUT**, not ready — Brentford target); front: **Palmer, Rogers** (started over Estêvão after 4 assists v Leeds), **Neto**, **João Pedro** ST. All confirmed. ⚠️ **CONTESTED — 3rd/rotation CB:** any TWO of Fofana/Acheampong/Colwill partner Lacroix. RotoWire = Fofana+Colwill+Lacroix; VAVEL = Fofana+Acheampong+Lacroix; Sports Mole = Acheampong+Colwill+Lacroix. **Colwill = minor injury** (only bench-fit last PL game, captained midweek). **Fofana available** (minor concern). Lean: **Lacroix + Fofana + Acheampong**, Colwill a fitness doubt. Confidence: Lacroix/James/Hato/Palmer/Rogers/Pedro/Neto/Lavia high; 3rd CB low. Out: Caicedo, Henderson (wrist), Emegha (muscle), Palestra.

**HULL CITY (Jakirović, PROMOTED, ~5-2-3 / back-5, injury crisis):** Tzolakis; back five **Ajayi, Egan, Mendy** + wing-backs **Coyle, Giles**; midfield **Gourna-Douath, Slater**; **Belloumi + Stroud** wide/10s; **McBurnie** ST (100th PL app, yet to score). **John Egan CONFIRMED at CB — started all 3, leads squad for minutes.** Tzolakis, Mendy confirmed starters. Only side yet to concede (3 clean sheets). Confidence: high. Out (crisis): Butland (GK, shoulder), Zambrano, Matazo, Gelhardt, Gyabi, C.Hughes, Hjerto-Dahl, Morita, Ömür, Ansah; Sorba Thomas doubtful (road accident Fri).

**CRYSTAL PALACE (Pierre Sage, 3-4-2-1 / 5-3-2):** D. Henderson; back three **Richards, Disasi, Canvot**; wing-backs **Khalaili (RWB) + Mitchell**; pivot **Wharton + Q. Timber**; tens **Kamada + Pino**; ST. ⚠️ **STRIKER (Mateta OUT, thigh):** **Jørgen Strand Larsen** favoured over Nketiah — scored 2 + assist v Middlesbrough (Carabao). RotoWire, Sports Mole, FFS all list Strand Larsen. Nketiah = bench option. Confidence: XI high, striker medium-high (Strand Larsen). Out: Mateta (thigh), Sarr (groin), Riad (knee). [Guéhi & Eze both LEFT per confirmed brief.]

**IPSWICH (Gary O'Neil, PROMOTED, 4-2-3-1):** Scherpen; back four **O'Shea, Diop, Greaves, Davis**; double pivot **Lukić + Palacios**; attacking band **Fatawu, Enciso, Maeda**; ST. ⚠️ **CONTESTED STRIKER:** Sports Mole = **Emersonn**; RotoWire = **Zian Flemming** (new signing may get the start, Emersonn at risk). Genuine coin-flip. Confidence: back-four/pivot/band high; striker low (Emersonn vs Flemming). Out: Philogene (ankle, back training), Matusiwa (muscle, back training), Jack Taylor (knee).

**Net contested slots for optimiser/captaincy caution:** Chelsea 3rd CB (Fofana/Acheampong/Colwill, Colwill injury doubt); Ipswich striker (Emersonn vs Flemming). Everything else consensus-nailed across sources.

## ✅ GW4 PRE-DEADLINE PREDICTED-XI SWEEP — Leeds/Newcastle/Liverpool/Fulham (2026-09-12)
Deadline Sat 13 Sep 11:00 BST. Fixtures: Leeds v Newcastle (Mon 14th, 20:00), Liverpool v Fulham (Sat 3pm). Sources dated 11–12 Sep: RotoWire (11th), FFS team-news (12th), Sports Mole (11th/12th), Premier Fantasy Tools press-conf (11th), SportsGambler/ThisIsAnfield fitness updates.

**LEEDS (3-4-2-1, RotoWire):** Trafford; Elvedi, Muharemović, Justin; Bogle, Ampadu, Tanaka, Gudmundsson; Okafor, Stach; Calvert-Lewin.
- **Gudmundsson STARTS** (back from injury, LWB) — high conf.
- **Bijol BENCHED** — RotoWire back three = Elvedi/Muharemović/Justin. Med conf (Bijol pushing).
- **Tanaka AND Stach both start** (Tanaka CM, Stach adv/SS) — not either/or this week.

**NEWCASTLE (4-2-3-1, RotoWire/Sports Mole):** Horníček; Livramento, Thiaw, Botman, Hall; Nico González, Miley; Elanga, Willock, Barnes; Wissa.
- **RB = Livramento** (IN, back from calf). **Dedic DOUBT** (hamstring, GTD, "not too serious"). Med-high conf Livramento.
- **Striker = Wissa. Fernández-Pardo DOUBT/bench** ("getting closer" to 1st start; eased in as sub). High conf Wissa CF for GW4.
- Burn OUT (ankle, ~19 Sep), Joelinton OUT, Osula OUT. CBs Thiaw+Botman, Hall LB confirmed.

**LIVERPOOL (4-2-3-1 / 4-3-3, Sports Mole+Yahoo+ThisIsAnfield):** Alisson; Araújo (RB), Jacquet, Van Dijk, Kerkez; Mac Allister, Szoboszlai; Ngumoha, Wirtz, Munoz; Isak.
- **Gakpo DOUBT (thigh/adductor, FPL 75%)** — missed midweek CL, "didn't train"; predicted to MISS/bench. If out, slot → **Victor Munoz** (both predicted XIs start Munoz, not Barcola). Med-high conf Gakpo OUT.
- **Barcola FIT** (cramp only, "should be available") but predicted BENCH — notable. Med conf.
- **Kerkez STARTS LB** (cramp late in CL but keeps place; unchanged back four). RB = Araújo (Bradley OUT knee; Frimpong not starting). Isak CF confirmed.
- OUT: Ekitiké (Achilles), Bradley, Leoni, Gomez (chasing return), Chiesa.

**FULHAM (4-2-3-1, Sports Mole/Yahoo):** Leno; Castagne, Andersen, Bassey, Robinson; Berge, Charles; King, Bobb, Iwobi; García.
- **Josh King STARTS** (No.10/right of front three). **Gonzalo García striker** confirmed. Med-high conf.
- Midfield double pivot Berge+Charles; Palacios/Bobb rotate the wide/10 slots (Sports Mole had Palacios, Yahoo had Bobb).
- Only absentee: **Cairney OUT** (knee/back, ~10 Oct). Otherwise clean bill.

**FPL-relevant flags:** Gakpo (bench risk — don't captain/start), Dedic (DC asset, GTD — Livramento in), Fernández-Pardo (not yet nailed — Wissa still the Newcastle CF), Bijol (bench risk vs Elvedi).

## ✅ GW4 PRE-DEADLINE PREDICTED-XI SWEEP — Man Utd/Man City/Spurs/Everton (2026-09-12)
Deadline Sat 13 Sep 11:00 BST. Fixtures: Spurs v Everton (Sat 12th, 17:30), Man Utd v Man City (Sun 13th, 16:30, Old Trafford). Sources 11–12 Sep: Sports Mole (11th/12th), Fantasy Football Scout team-news (12th), Premier Fantasy Tools press-conf (last upd 11th), Yahoo Sports (12th), 3addedminutes/Squawka (11th).

**MAN UTD (4-2-3-1, Sports Mole 12th):** Lammens; Dalot, Maguire, L.Martínez, Shaw; Tielemans, Mainoo; Mbeumo, B.Fernandes, Cunha; Šeško.
- **Front line: Šeško STARTS as the 9, Cunha + Mbeumo wide/attack, Rashford BENCHED** — high conf. Šeško & Cunha both scored starting v Sabah (CL, Thu 4-0); Sports Mole headline "Sesko, Cunha both start". Rashford IN (available, rested midweek — Carrick: kept "fresh and ready for the weekend") but predicted to start on bench.
- **CBs = Maguire + L.Martínez** (Sports Mole). Yoro not in predicted XI; Mazraoui used as cover not starter. Med conf (De Ligt OUT back = fewer CB options, so Maguire/Martínez firm).
- **Pivot = Tielemans + Mainoo** (NOT Andrey Santos). Med-high conf.
- **Shaw LB but was a DOUBT** (cramp, missed CL "wasn't 100%") — cleared to start per manager. Dalot RB. Med conf Shaw; Mazraoui/Dalot cover if Shaw fails late.
- OUT: Amad (ankle/muscle), Baleba (ankle, ~19 Sep), De Ligt (back), Ugarte (knee, 2027), Heaton (hamstring).

**MAN CITY (4-2-3-1, Yahoo/3addedminutes 11th):** Donnarumma; Nunes, Dias, Guéhi, Gvardiol; Enzo Fernández, E.Anderson; Foden, Cherki, Semenyo; Haaland.
- **Doku OUT (calf) — confirmed misses this** (Maresca: "needs some more days"). 
- **Both Cherki AND Foden start:** Cherki at the 10, Foden RW, Semenyo LW. Med-high conf.
- **Back line: Dias + Guéhi CB, Gvardiol at LB.** RB contested — **Nunes vs Khusanov**: Nunes started v Porto midweek and is favoured to keep RB ahead of Khusanov (med conf; Khusanov the alternative). O'Reilly fit and available (bench/LB cover). Note earlier search had Khusanov RB — Nunes is the newer 11th read.
- **Pivot = Enzo Fernández + Elliot Anderson** confirmed. Haaland CF, Donnarumma GK. Semenyo confirmed. Ndiaye NOT in predicted XI (attacking depth).

**SPURS (4-2-3-1) — HEAVILY CONTESTED, sources disagree:**
- Consensus spine: Kinský (GK; Vicario left→Juve); **CBs van de Ven + van Hecke**; double pivot **Bentancur + Tonali**; **Savio + Mateus Fernandes** in the wide/10 band.
- **RB: Porro (Yahoo 12th, FFS) vs Gray (Sports Mole 11th).** Porro was a doubt (deep World Cup run) but back — lean Porro, med conf, Gray the fallback.
- **LB: Udogie (Yahoo 12th) vs Robertson (Sports Mole 11th).** Genuine coin-flip / rotation — low-med conf either.
- **Striker + 3rd attacker: Marmoush, Tel and Solanke all in the mix.** Yahoo (12th) = Marmoush CF with Tel in the band; Sports Mole (11th) = Tel + Solanke. Solanke fitness back but not nailed. Low-med conf — game-time call.
- **Maddison IN** (back in training from shoulder fracture, available for selection) but unlikely to start — return off the bench. 
- OUT: Mudryk (ankle, 6–8 wks), Kulusevski (knee, setback — not in team training), Xavi Simons (OUT), Odobert (OUT).

**EVERTON (4-2-3-1, Sports Mole 11th):** Pickford; Maitland-Niles, Tarkowski, Branthwaite, Mykolenko; Armstrong, Hackney; George, Dewsbury-Hall, Grealish; Barry.
- **RB = Maitland-Niles** (Garner not 100% — "status unclear", 75% flag holds; AM-N stepped in v Man Utd). Med conf AM-N.
- **Grealish STARTS** (Moyes: "Jack is ready to play") — displaces Brennan Johnson from the XI. Med-high conf. B.Johnson bench.
- **Striker = Barry** (over Beto). Attacking band George + Dewsbury-Hall + Grealish; Dibling not in predicted XI. Med conf.
- Back line Tarkowski + Branthwaite, Mykolenko LB, Pickford GK — all confirmed.
- OUT: Nørgaard (out ~October). DOUBT: Gueye (50-50, missed Sunderland). Garner (75% flag — AM-N preferred).

**FPL-relevant flags:** Rashford (benched — fade), Šeško (nailed CF, differential), Doku (OUT), Man City RB unsettled (Nunes/Khusanov/O'Reilly — avoid City RB DC punt), Spurs LB + striker unresolved (avoid Udogie/Robertson & Solanke/Tel/Marmoush captaincy calls), Grealish (starts, Everton attacking asset), B.Johnson (benched at Everton).

## ✅✅ GW4 FINAL PRE-DEADLINE SWEEP — APPLIED F1–F8 (2026-09-12, deadline Sat 13 Sep 11:00)
Full 20-team sweep: 5 research agents (all GW4 fixtures) across RotoWire + Sports Mole + FFScout + Premier Fantasy Tools pressers + SportsGambler, primed with the correct managers/transfers (Palace=Sage, NEW=Jaïssle, MCI=Maresca, CHE=Alonso, LIV=Iraola, TOT=De Zerbi). Reconciled vs GW3 actual XIs + live FPL flags. **Values are per-slot probabilities — source disagreement carried as fractions, unanimous+no-doubt = 1.0.** Validated: **all 20 teams × F1–F8 = 11.00**, 652 rows, 0 dups/NaN, 0 FPL-ceiling breaches.

**No change (XI unanimous across sources):** Arsenal (White/Konsa/Gabriel/Calafiori; Rice/Lewis-Skelly; Saka/Ødegaard/Tzolis; Havertz — Bruno G. rested, Mosquera/Timber not risked), Hull (Egan nailed), Leeds (Gudmundsson starts, Bijol benched, Tanaka+Stach both). Sunderland XI intact.

**F1 coin-flips set (contested slot → probabilities):**
- **CHE** back-3: Lacroix+Fofana nailed, 3rd CB **Acheampong 0.62 / Colwill 0.38** (Colwill fitness doubt). Caicedo out → Lavia starts.
- **NEW** RB: **Livramento 0.65 / Dedic 0.35** (Dedic hamstring doubt). Wissa CF nailed; Fernández-Pardo eased in as sub (held ~0, WATCH F5+).
- **CRY** striker (Mateta out→11 Oct): **Strand Larsen 0.65 / Nketiah 0.35**; Mateta ramps back F5+ (F5 0.55→F8 0.75).
- **MCI**: Doku out. **Cherki 0.75 / Foden 0.70 / Ndiaye 0.55** (2 att slots); RB **Nunes 0.55 / Khusanov 0.45**.
- **MUN**: **Šeško 0.80 / Rashford 0.20** (Rashford benched, fresh for wknd); Maguire 0.90/Yoro 0.10; Shaw 0.80/Mazraoui 0.20; Tielemans 0.85/Santos 0.15.
- **TOT** (least certain — sources split; final RW check advised): RB Porro 0.70/Gray 0.30; LB Udogie 0.52/Robertson 0.48; front Mateus Fernandes 0.70, Marmoush 0.60, Tel 0.55, Kudus 0.50, Solanke 0.35, Maddison 0.20 (bench), Gallagher 0.10.
- **EVE**: RB Maitland-Niles 0.65/Garner 0.35; pivot Hackney 0.60/Röhl 0.40; Grealish 0.72 (starts)/B.Johnson 0.28.
- **AVL**: RB Wan-Bissaka 0.55/Cash 0.45 (Cash 75% doubt); CB Mings 0.50/Torres 0.50 (true 50-50). Gomes out, Manzambi bench.
- **NFO**: Gibbs-White 0.90, Schlager 0.85, Sangaré 0.25 (calf); RWB Ola Aina 0.60/Daniel Muñoz 0.40 (debut).
- **SUN**: Angulo 0.60/Fofana 0.40. **BOU**: Adams 0.55/Christie 0.25/Cook 0.20. **IPS**: Emersonn 0.50/Flemming 0.50. **FUL**: Berge 0.70/Palacios 0.30. **BHA**: De Cuyper 0.92 (starts as makeshift LW).

**Injury-return F2–F8 ramps (full vectors, per-column sum held to 11):**
- **BRE**: Collins definitively OUT (calf) → **Jannik Schuster starts** [1.0→fade]; Collins ramps back F2+; van den Berg 0 (groin). Igor Thiago ST nailed.
- **CRY**: Strand Larsen/Nketiah/Mateta three-way ramp as above.
- **LIV**: **Gakpo benched GW4** (thigh doubt, didn't train) → F1 0.20; wide slots GW4 = **Munoz 0.60 + Ngumoha 0.45** (Barcola fit but benched 0.45). F2+ settles Gakpo 0.60/Barcola 0.60/Munoz 0.45/Ngumoha 0.25.

**Deadline soft-spots to re-check at the final RotoWire XI drop (~1hr pre-KO):** Arsenal CF (Havertz vs Gyökeres — we have Havertz), Spurs LB + front + RB, Man City RB (Nunes/Khusanov) + Ndiaye squeeze, Liverpool Gakpo (bench vs start), Chelsea 3rd CB (Colwill fitness), Villa RB/CB toss-ups.

### F2–F8 refinement pass (2026-09-12, same day) — corrections after review
- **Pinnock (COV) 1.0 → 0.88** (Bidwell 0.12): back-3 with Pinnock is *medium-high* (GW3 actual + agent lean), NOT a lock — league_state warns Lampard can play a back-4, and he's a just-signed 33yo. Overstated it as 1.0 first pass.
- **Baleba (MUN, back 19 Sep/F2):** was flat 0.00 → ramps into the pivot [0,0.30,0.50,0.62,0.65…]; Tielemans/Santos fade to hold the 2nd-pivot-slot sum.
- **Doku (MCI, back 20 Sep/F2) + Foden:** Foden was dropping to 0.30 from F2 (under-rated) — rebalanced the wide group so Foden ~0.55 and Doku ramps to ~0.45, Ndiaye trimmed (0.20→0.15).
- **Fernández-Pardo (NEW):** new-signing integration ramp [0,0.10,0.20,0.30,0.40,0.45…], offset against Willock (Wissa expected to shift to a No.10 as Pardo takes CF).
- **Deliberately left at ~0 (NOT returning to starting roles):** Kroupi, Eccles, Josh Dasilva, Ferguson, Cairney, Mathias Jensen. Low-confidence/low-relevance gap acknowledged: Hull midfield (Morita ~11 Oct return) — promoted injury-crisis side, week-to-week guesswork, no FPL assets ride on it.
- Validated after: all 20 teams × F1–F8 = 11.00, 652 rows, 0 dups/NaN, 0 ceiling breaches.

### Team-by-team final check (2026-09-12) — F2–F8 coherence fixes
Full 20-team read caught F1-only edits that had left **incoherent F2–F8 jumps** (a GW4 starter vanishing / a benched player snapping to 1.0). Fixed with smooth carry-forward curves:
- **Fulham:** Berge was 0.70→0.00 at F2 → now persists (0.70→0.65); Palacios the deputy.
- **Man City RB:** flipped to **Khusanov incumbent** (0.55, league ever-present) over Nunes (0.45) and smoothed F2+ (was Nunes 0.55→0 / Khusanov 0.45→1.0).
- **Everton:** Grealish now persists as a GW4 starter (0.72→0.60, not →0.39); pivot Hackney/Röhl smoothed and made a 3-way from F4 when **Nørgaard** returns; RB AM-N/Garner smoothed.
- **Spurs:** front flattened to a coherent rotation (removed Tel 0.55→1.0 jump); LB Udogie/Robertson fixed to a clean 0.52/0.48 (Robertson's share had collapsed to 0.28); marginal Mudryk return zeroed.
- **Forest:** Muñoz now a proper handover ramp (0.40→0.65, was →0.00); Aina fades.
**Verified:** all 20 teams × F1–F8 = 11.00, 652 rows, 0 dups/NaN, 0 ceiling breaches, all names byte-exact. Structural scan clean (every team exactly 1 GK, sensible DEF/MID/FWD mass).

### Botman + Fofana security check (2026-09-12) — both downgraded from "nailed"
Fan/beat-level research (not just aggregator XIs) on the two cheap differential CBs I'd called nailed:
- **Botman (NEW) 1.0 → 0.80** (Schär 0.20): starts GW4 only ~75-80%. **Three defensive errors in three PL games** (German press "threatened with the bench"; FootballFanCast "new Almirón"); **Schär is fit on the bench as the live replacement** (not a returning-injury question). Jaïssle has backed him + soft fixtures help, but he's one error from the bench → **rotation risk, not set-and-forget**. DefCon (9.98) still elite IF he plays — but discount by the ~0.80 start prob.
- **Fofana (CHE) 1.0 → 0.60** (Acheampong 0.80, Colwill 0.60): only ~60-65% to start — **Josh Acheampong has played every competitive minute** under Alonso and is the stronger starter; Chelsea also recruiting another CB and treat Fofana as "depth" given his injury record. Still a back-3 (Lacroix nailed). **Materially changes the Fofana case**: his whole appeal was a nailed 11.4-dc90 machine at 0.1% owned — at 0.60 start prob + rotation/injury risk, that differential is much softer. Re-check at the XI drop; consider Acheampong (the actual every-minute man) if you want the Chelsea-CB DefCon route.
- **Burn correction:** FPL flag says "back 14 Sep" but RotoWire/beat says **~mid-October** — modelled Burn out through F4, cover from F5. (Flag/source conflict — verify.)
- Validated: all 20 teams × F1–F8 = 11.00, 0 ceiling breaches.

### CORRECTION (2026-09-12, same day) — Fofana/Botman over-downgraded; minutes-grounded revision
The prior entry (Fofana 0.60, Botman 0.80) leaned on agent NARRATIVE (incl. the stale "Chelsea shopping for a CB" — window is CLOSED, irrelevant). Hard GW1-3 minutes correct it:
- **Chelsea CBs are a 3-into-2 rotation** (Lacroix 90/90/90 nailed; Fofana/Acheampong/Colwill rotate 2 spots): GW1 Acheampong+Colwill (Fofana susp), GW2 Fofana+Colwill, **GW3 Fofana+Acheampong (Colwill 0)**. **Fofana: DNP(susp)/68(start)/90(start) — started the last two, in possession.** → **Fofana 0.60→0.75**, Acheampong 0.70, Colwill 0.55 (missed GW3). Real rotation, so not 1.0, but not a fringe 0.60.
- **Botman is EVER-PRESENT: 90/90/90.** The error/bench narrative is media — the manager has not acted on it, and **Schär has played 1 minute all season** (not a live threat). → **Botman 0.80→0.88** (small error-tail + Burn cover from F5), Schär 0.12.
- Lesson: weight the minutes over transfer-window/fan narrative, especially post-deadline. All teams × F1-F8 = 11.00.

### FINAL VALUES (2026-09-12) — supersede the two entries above for Fofana/Botman
After confirming fitness flags (both CBs + Schär are `status=a, chance=100%`) and reading Newcastle fan/beat sentiment, these are the committed numbers (all teams × F1–F8 still = 11.00):
- **Fofana (CHE): F1 0.85**, F2+ ≈0.78. Favoured starter in a genuine **4-for-3** back line (started every game he's been available for; 100% fit). Acheampong **0.70** (every-minute man, started GW3), Colwill **0.45** (dropped GW3, injury niggle). Not 1.0 because one of the three sits each week — but not the 0.60 I briefly had.
- **Botman (NEW): F1–F4 0.90, F5–F8 0.60.** Near-nailed now — **Schär is 100% fit yet has played 1 minute**, so the manager is actively choosing Botman; the "errors/bench" talk is media the manager hasn't acted on. The **F5 taper is a named reason**: Burn returns ~mid-Oct and multiple Newcastle watchers expect a real 3-for-2 squeeze (Botman/Burn/Schär). F5+ split: Botman 0.60 / Burn 0.28 / Schär 0.12. Thiaw stays nailed (the watchers' most-secure CB).
- **Burn:** return modelled ~mid-Oct (F5), cover only (FPL "14 Sep" flag vs RotoWire/beat "mid-Oct" — conflict, verify).
- **Luke Shaw (MUN) 0.80 → 0.75** to respect his FPL 75% (cramp) flag; Mazraoui 0.25. (Cleared the last remaining ceiling breach.)
- Data fix (not a lineup call): added `Macauley Zepa → Macaulay Zepa` to `name_mappings.csv` so the odds feed resolves.

**Net for the wildcard:** Fofana is back to a credible low-owned DefCon differential (11.4 dc90 × 0.85); Botman is a strong GW4–GW7 pick that should be re-assessed once Burn is back. Re-run `pipeline.run` before optimising.

### Flag re-check (2026-09-12, post-compact) — FPL availability cross-check
Re-read FPL `playerstats.csv` flags against the committed file. All contested CBs confirmed: **Fofana** `a/100%/67min` (started GW2+3), **Schär** `a/100%/1min` (the basis for Botman nailed), **Botman** ever-present 180min, **Burn** `i, "back 14 Sep", 0%` → out for GW4 (deadline 13th) regardless. No value changes.
- **Note (no change):** **Luke Shaw's** 75% cramp flag has **cleared** (`st=a`, no chance figure). His 0.75 was originally a flag-cap; it now stands purely as a Shaw(0.75)/Mazraoui(0.25) rotation split, which is still the right read. Justification updated, value unchanged.

### Three-player start-prob sweep (2026-09-12) — James / De Cuyper / Awoniyi
Fan/beat/press-conf sentiment, current (Sep 2026) only, for GW4 (deadline Sat 13 Sep 11:00).
- **Reece James (CHE, DEF £5.4m) — started GW1, benched GW2 (22m sub), started GW3.** POSITION FLAG (model-relevant): under **Xabi Alonso** James is being deployed as a **central MIDFIELDER** (Alonso: "can become one of the world's best midfielders"; GW4 preview has the Lavia–James midfield pairing) though FPL still lists him as DEF. GW4 vs Hull (home) he's predicted to start, but **Valentin Barco is a live rotation threat** after his Carabao Cup cameo, and Alonso is plainly managing the long-term load of a player with James's injury record. Verdict: **high-value starter when fit, but genuine minutes-management + rotation risk — not set-and-forget.** → **GW4 0.88, steady-state 0.82.**
- **Maxim De Cuyper (BHA, DEF £4.7m) — started GW1–3 (243 mins, every minute).** Nailed right now, and the "makeshift LW" note is real but currently *protects* his minutes, not threatens them: **Mitoma is out (hamstring surgery, ~GW6 return)** so De Cuyper covers the left flank, and **Kadioglu** (the preferred LB) only just returned from a minor issue that cost him GW1–3. Hürzeler views De Cuyper as a winger; he plays LB or LW depending on who's fit. Verdict: **nailed for GW4; steady-state softens once Kadioglu AND Mitoma are both fit (~GW6+) and the LB/LW picture gets crowded (Hadjam £9.5m signed as backup LB).** → **GW4 0.95, steady-state 0.85** (position is LB-or-LW, not pure LB — watch CS relevance).
- **Taiwo Awoniyi (COV, FWD £5.5m) — sub GW1 (20m), started GW2 & GW3 (90 each).** Has locked down the lead-CF role under Lampard; previews say he "should retain his place ahead of **Ellis Simms**" and Lampard should "stick with" the Mason-Clark/Rudoni/Awoniyi front unit. GW4 is **Coventry v Brighton (home, Sun 13 Sep)**. Competition is Simms (bench) only. Verdict: **first-choice starting striker, low rotation risk — tempered slightly as a brand-new signing at a struggling promoted side.** → **GW4 0.92, steady-state 0.90.**

### APPLIED — transfer-pool / ownership deep-dive (2026-09-12) — minutes-grounded
Deep-dive on the wildcard transfer pool + most-transferred-in list. Grounded every call in **actual GW1–3 starts/minutes** (FPL `playerstats.csv`), web research only for forward-looking fitness/rotation. Minutes confirmed nailed (3/3 starts, left at 1.0): Raya, Groß, M.Sangaré(BRE), Tzolis(ARS), Verbruggen, Szoboszlai, Donnarumma, **Ben White**(270m), Wissa, Mbeumo, Semenyo, **Guéhi**, Gvardiol, Ajer, Saka, **Justin**(LEE), Muharemović, **Rogers**(CHE — transfer real, nailed), Egan(HUL). Confirmed bench (0.0): Button, Walle Egeli. **Five edits applied (all team sums held at 11.00):**
- **Cody Gakpo (LIV):** 0.20→**0.35** F1, 0.60→**0.68** F2+. Started all 3 GW (250m); the bench risk is a current **adductor** (not thigh) knock — rested vs Atlético 9 Sep, *didn't train*, Iraola "has a chance", aggregators omit him, Barcola primed. Steady ~0.72 (Barcola now a real rotation threat, no longer an auto-lock). Offset: **Victor Munoz** 0.60→0.45 F1 / 0.45→0.37 F2+.
- **Jaka Bijol (LEE):** 0.00→**0.55** F1 / **0.50** F2+ — was wrongly frozen at 0. Started GW1+2, benched GW3, but **Rodon injured** and aggregators flip him back into GW4 XI. Live CB rotation, NOT frozen out. Offset: **Nico Elvedi** 1.00→0.45 F1 / 0.50 F2+ (Elvedi only 1 start — GW3 — so he's the rotation partner, not a lock; Justin & Muharemović stay nailed on 3/3 minutes).
- **Reece James (CHE):** 1.00→**0.88** F1 / **0.82** F2+. Start/bench/start across GW1–3 = active minutes-management; Barco a live alternative. POSITION FLAG: Alonso plays him as a **CM** though FPL lists DEF (shifts returns from CS toward mid). Offset: **Valentín Barco** 0→0.12 F1 / 0.18 F2+.
- **Maxim De Cuyper (BHA):** F1 0.92→**0.95** (Mitoma out ~GW6 *protects* his minutes now). F2–F8 left as-is (0.88→0.80 taper already matches the ~GW6 Mitoma/Kadioglu softening). Offset: **Hadjam** F1 0.08→0.05.
- **Taiwo Awoniyi (COV):** 1.00→**0.92** F1 / **0.90** F2+ — first-choice CF but a brand-new signing (bench GW1, then 2 starts); Simms the only competition. Offset: **Ellis Simms** 0→0.08 F1 / 0.10 F2+.
Validated: 652 rows, 20 teams × F1–F8 = 11.00, 0 dups/NaN, 0 ceiling breaches.

### Late FFS/RotoWire check (2026-09-12, ~eve of GW4 deadline) — last-48h news
Swept FFScout / RotoWire / pressers for news dated 11–12 Sep. Most items **confirm the committed file**; two edits applied.
**No change (already modelled correctly):**
- **Arsenal:** Havertz starts CF, Gyökeres bench (27 min across 4 games). Ben White "fine" after limping off. ✓
- **Man City:** no fresh 11–12 Sep item. Doku still out (~4 games), Nunes back, Khusanov in the four, Ndiaye a doubt but featuring wide. ✓
- **Liverpool Gakpo:** confirmed DOUBT vs Fulham (adductor, missed midweek, needs assessing); predicted XI has Munoz/Ngumoha wide, Barcola bench-leaning (cramp only, available). Our 0.35 F1 already prices the doubt. ✓
- **Chelsea:** Colwill returning after missing GW3 (Alonso "calming worries"); Acheampong was the deputy who started. Already priced (Colwill 0.45 / Acheampong 0.70 / Fofana 0.85 = 2.00 over 2 CB slots). Reece James confirmed in CM, no fresh fitness concern → 0.88 holds. ✓
- **Villa:** Torres/Cash items are 8-Sep-dated, not fresh; our Cash 0.45/AWB 0.55 + Mings/Torres split already reflect it. ✓
- **Newcastle:** Dan Burn NO early return — out until after the Oct int'l break (~4–5 games, misses GW4–7), back ~F5. Matches our Burn-0-through-F4 / return-F5 model and the Botman F5 taper timing. Botman fit, starts. ✓
**Edits applied (sums held at 11.00):**
- **Jaka Bijol (LEE) 0.55→0.62 F1 & 0.50→0.62 F2–F8 (flat):** Joe Rodon ruled out **8–10 weeks** (thigh) = the ENTIRE F1–F8 horizon (Rodon back ~mid-Nov, beyond GW11), and Bijol is the named starting deputy. So he's a favoured starter for the whole window, not a settling coin-flip. Offset: **Nico Elvedi** → 0.38 flat (rotation partner; started GW3 so not zero). Rodon stays 0 across the window.
- **James Maddison (TOT) 0.30→0.12:** left OUT of the matchday squad Fri 11 Sep (shoulder doubt) per De Zerbi's predicted XI. Redistributed +0.18 to the predicted starters: Mateus Fernandes 0.65→0.70, Tel 0.60→0.66, Marmoush 0.55→0.62.
- **WATCH (not actioned):** one predicted Spurs XI listed **Savinho** in the front line — if that transfer/role is real and he's not in our file as a starter, it's a gap. Low confidence from a single aggregator; verify at the XI drop.
Validated: 652 rows, 20 teams × F1–F8 = 11.00, 0 dups/NaN, 0 ceiling breaches.

### CONSOLIDATED — transfer-pool deep-dive, final state (2026-09-12)
This entry supersedes the intermediate notes above for the players listed. Grounded in **GW1–3 FPL minutes** (reliable for established players) + `league_state_2026-27.md` (transfers/injuries). ⚠️ **Process note (learned this pass):** open-web sub-agents produced unreliable claims (Maddison "shoulder doubt" — FPL says 100% available/false; Bijol "named starter" — unverifiable). **Going forward, agents are constrained to the `curation_sources.md` Tier-1/2 panel with mandatory live-roster name-checks.** Also: three accent-match bugs (`'Sav'`≠`'Sávio'`, Groß, Guéhi) — use accent-insensitive matching.

**Backwards splits corrected (established players — minutes were the right signal, file was wrong):**
- **Everton** midfield/attack fully rebuilt to GW1–3 starts: **Röhl [SSS] 0.40→0.90** (was badly under), Garner [sSS] 0.70, B.Johnson [ssS] 0.55; **Grealish [s-s] 0.72→0.30** (loan signing, 0 starts — was badly over, ramps up as he integrates), **Maitland-Niles [barely] 0.65→0.20**, Hackney [Sss, falling] 0.60→0.25; Nørgaard returnee ramp. (col sums held = 3.00 contested.)
- **Fulham:** Palacios [SSS] 0.30→0.58 over Berge [SSs] 0.70→0.42 (Palacios started all 3, Berge benched G3).
- **Spurs:** Tel [SSS] 0.66→0.85 (under-rated), Gallagher [S-S] 0.10→0.55 (started 2/3 incl. G3), Mateus Fernandes/Kudus/Solanke trimmed (few/no starts), Maddison 0.30→0.12 (0/3 starts — a SELECTION call; FPL has him 100% available, NOT injured — agent's "shoulder doubt" was false). Sávio left at 1.0 (started G3, new signing; mild over-rate flagged, not changed).
- **Man Utd:** Rashford [sSS] 0.20→0.65 (returnee from Barça loan, started last 2 — league_state "rising F2–F6" ✓). **Šeško 0.80→0.35** — ⚠️ KEPT per Tyrone, but FLAGGED: downgrade rests on sub-minutes [sss], and Šeško is a marquee new signing being eased in, not dropped; **re-check at the predicted-XI drop** (minutes under-sell new signings).
- **Leeds:** Bijol/Elvedi settled to a **0.50/0.50 coin-flip** (Rodon out 8–10 wks = whole horizon; recency favours Elvedi [DNP/sub/start] over Bijol [start/start/bench], so NOT a Bijol-favoured lean — earlier 0.62 retracted as agent-narrative-driven).

**Other edits (from earlier in the turn, retained):** Gakpo 0.35/0.68 (adductor doubt, Munoz offset), Reece James 0.88/0.82 (rotated + now a CM under Alonso; Barco deputy), De Cuyper 0.95 F1 (Mitoma out protects his minutes), Awoniyi 0.92/0.90 (Simms deputy).

**Roster mirror:** ran `sync_lineups_roster.py` — added **Stephen Mfuni (Coventry)** at 0.0 (only genuine roster gap). File now 653 rows. Confirmed: starting_lineups DOES mirror the full FPL roster; "Sávio missing" was my accent-search error, he was present at 1.0 throughout.
Validated: 653 rows, 20 teams × F1–F8 = 11.00, 0 dups/NaN, 0 ceiling breaches.

### F1-firming pass (2026-09-12) — Tyrone: "F1 is more certain than F2-F8"
Principle (per curation_sources.md): **F1 = the imminent GW4 XI — declare nailed starters at 1.0; shade F1 ONLY for a real GW4 reason (fitness flag or a genuine same-position contest), never for future risk (that lives in F2-F8).** Corrected over-shaded F1s + found more backwards splits. All team F1 sums held at 11.00.
- **De Cuyper (BHA) 0.95→1.00:** [SSS], no flag, no real rival (Hadjam 0→). The 0.95 was future Mitoma/Kadioglu risk leaking into F1; that belongs in the F2-F8 taper (kept).
- **Botman (NEW) 0.90→1.00 for F1-F4:** [SSS], no flag. The 0.90 was the Burn-return risk — an **F5** event. Taper stays at F5 (Botman 0.60 / Burn 0.28 / Schär 0.12). Schär 0.10→0.00 for F1-F4.
- **Spurs front rebalanced to actual starts:** Tel [SSS] 0.85→0.92; Marmoush [-SS] 0.68→0.80; **Sávio 1.00→0.70** (only 1 start — G3 — over-rated at nailed); Gallagher 0.55→0.62; Mateus 0.50→0.38; Gray 0.20→0.28; Porro kept 0.72 (**75% fitness flag** = legit shade); Udogie kept 0.52 (75% flag).
- **Newcastle RB was BACKWARDS:** **Dedić [SSS] started all 3** but sat at 0.35, while **Livramento [--s]** (just back from calf surgery) was at 0.65. Flipped to a handover — Dedić 0.65 now → fades to 0.35 as Livramento (first-choice when fit) ramps to 0.65 by F6.
- **Palace striker was BACKWARDS:** **Nketiah [SSS] started all 3**, Strand Larsen [sSs] only 1 — but we had Strand Larsen 0.65 / Nketiah 0.35. Flipped (Nketiah 0.68 incumbent with Mateta injured to ~11 Oct; Mateta-return taper kept at F5).
- **Leeds:** Bijol/Elvedi leaned to the G3 starter — Elvedi 0.55 / Bijol 0.45 (was dead-even).
- **Left as legitimate contests (SSS but a real live rival, NOT errors):** Khusanov 0.55 (Nunes back), Ndiaye 0.55 (crowded City attack), Palacios 0.58 (Berge), Aina 0.60 (debutant Muñoz pushing).
Validated: 653 rows, 20 teams × F1–F8 = 11.00, 0 dups/NaN, 0 ceiling breaches.

### F2-F8 curve fixes (2026-09-12) — stray/over-aggressive fades
Tyrone review of biggest F2-F8 deltas vs F1.
- **Gibbs-White (NFO) — FIXED:** was 0.90→0.51, a stray fade with no named reason. He's the captain/talisman, [SSS]. Set to **1.00 flat F1-F8** (nailed). Rebalanced Forest so each slot sums right: Schlager/Sangaré (CM) = 1.00; **RWB slot = Aina/Muñoz/Savona** now sums 1.00 (Savona ramps back from knee, taking share from Aina — previously the trio over-summed and pushed Forest >11). Forest now exactly 11 every column.
- **Tielemans (MUN) — FIXED:** was 0.85→0.24, cratered by assuming Baleba (injured new signing, back ~19 Sep/F2) walks straight into his spot. Both are quality; the pivot is a genuine **3-way rotation (Mainoo/Tielemans/Baleba)** for 2 slots. Now: Tielemans floors at **0.65** (not 0.18), Baleba integrates to 0.60, Mainoo eases 1.0→0.75. While fixing, corrected pre-existing MUN slot mis-sums so each = 1.00: CB2 = Maguire/Yoro/**de Ligt** (de Ligt's return now takes share, not stacks), LB = Shaw/Mazraoui, front-4th = Rashford/Šeško/Amad. MUN now exactly 11 every column.
- **Fofana (CHE) −0.07 / Pinnock (COV) −0.12 — NO CHANGE (explained):** Fofana is a genuine **3-CB-for-2** back-3 rotation (Lacroix nailed; Fofana/Acheampong/Colwill rotate 2 spots) — F1 0.85 is his started-last-2 near-term edge, F2+ 0.78 the steady share. Pinnock is a mild **veteran-load/rotation** assumption (just-signed 33yo, Bidwell deputy) — softer-grounded; flatten if we want him firmer.
Validated: 653 rows, 20 teams × F1–F8 = 11.00, 0 dups/NaN, 0 ceiling breaches.

### Source freeze + De Cuyper taper (2026-09-12)
- **Froze GW4 predicted XIs** for the source-accuracy ledger via the constrained panel fetch (panel URLs only, anti-stale checks clean): `source_history.py --gw 4 --seed-ffs` (refreshed `ffs_predicted_lineups.csv` from the Aug-18 stale file; old one backed up to `.aug18.bak.csv`) + `--capture-csv rotowire_gw4.csv`. 20 teams × 11 each from **FFS** and **RotoWire**. AllAboutFPL had no GW4 all-team article — not substituted. 3 cosmetic UNMATCHED to fix with name_mappings before scoring: FFS "António Silva" (BOU), FFS/RotoWire "Angulo" (SUN). No departed/stale players in either XI.
- **De Cuyper (BHA) taper deepened:** was 1.00→0.80; only nailed because he covered BOTH Mitoma's (LW) and Kadioglu's (LB) absences. Mitoma returns ~F3 (GW6), Kadioglu already back → De Cuyper becomes a rotation option, not a 0.80 near-starter. New: 1.00/0.88/0.75/0.68/0.63/0.60/0.58/0.58. Offset by raising **Mitoma** (returning star, was under-rated at 0.29) to ramp to 0.51; DC+Mitoma pair total held so Brighton stays 11 every column.
- **Ben White (ARS) — left at 1.00 (Timber considered):** Timber returns ~F4-F6 but **Saliba is out all of F1-F6**, so Timber most likely fills CB, not White's RB; both FFS+RotoWire start White GW4. Caveat is minor; no shade applied.
Validated: 653 rows, 20 teams × F1-F8 = 11.00, 0 dups/NaN, 0 ceiling breaches.

### Timber return deep-dive (2026-09-12) — RB-vs-CB & Ben White threat
Constrained-source sweep for GW4 (deadline Sat 13 Sep).
- **Return:** imminent. **FFScout (Sep 11):** "Jurrien Timber (groin) is already back in training... a return to action seemingly imminent... you'd imagine he'll be coming off the bench at first after so long out." **Daily Cannon (Sep 8-9):** back in full training, travelled to Napoli. **Premier Fantasy Tools (Sep 11):** Timber = DOUBT. **RotoWire:** QUES. **SportsGambler:** not listed (only Saliba, back, return ~21 Nov). Read: likely a GW4 bench option, not a nailed starter yet — our F1 0.25 is about right near-term, ramping up F2-F6.
- **RB vs CB / Ben White threat:** Low near-term threat to White. White flagged "fine" by Arteta (PFT IN, RotoWire starts him at DR). With **Saliba out through ~F6**, Timber's most valuable slot is CB cover, not displacing an ever-present White at RB. (FFScout's "CB or LB" descriptor looks shaky — Timber's Arsenal history is RB/CB; no source put him at LB before injury. Flag, don't act on it.) Medium-term once Saliba + Mosquera are back, Timber is a genuine RB rival to White, but no source shows that yet.
- **Start-prob (steady-state, once Timber fully fit):** RB slot White **0.72** / Timber **0.28** of a shared ~1.0 while Saliba still out (Timber pulled toward CB). When Saliba returns and CB is full, shift toward White **0.60** / Timber **0.40** at RB (genuine rotation). CB pairing stays Gabriel + (Saliba/Konsa) nailed; Timber/Mosquera share the cover, no change to the CB starters' probs now. No confirmed XI exists pre-deadline.

### Ben White / Timber — resolved with web + fan research (2026-09-12)
Tyrone pushed (twice) on White being a flat 1.0; constrained panel search + r/FantasyPL consensus confirmed he was right. Corrected:
- **Timber's return is IMMINENT, not distant** (FFS Sep 11 "return seemingly imminent", in the Napoli travelling squad; Arteta "can be selected"). He's a GW4 BENCH option (F1 start-prob stays 0.0 — won't start Sat), eased in via cameos GW5, and is the **first-choice RB when fully match-fit** (strong fan consensus; "Timber first choice, not even close"). White's own injury history means Arteta manages his minutes.
- **White is NOT a season-long nail** — it's a handover. New curve: White 1.00/0.90/0.80/0.68/0.58/0.52/0.48/0.45; Timber 0.00/0.10/0.20/0.32/0.42/0.48/0.52/0.55 (crosses to Timber-favoured ~F6-F7). My earlier "flat 1.0, Timber→CB" was wrong — Timber's natural spot is RB and Arsenal are stacked at CB, so the pressure is on White's RB.
- **Saliba out until ~21 Nov** (SportsGambler) = BEYOND our F8/GW11 window → Saliba 0.0 all columns (was wrongly returning at F8). **Konsa nailed CB all window** (plays over Mosquera while Saliba out; was wrongly cratered to 0.24 at F8). Mosquera 0.0 (cover/50% doubt).
- Net: the old file had White over-nailed AND an incoherent F8 CB four-way (Saliba returning early + Konsa cratering). Both fixed. Arsenal = 11.00 every column.

### White/Timber — softened (2026-09-12, same day): Reddit discounted
Tyrone flagged the fan thread is "random Redditors." Re-weighted: RELIABLE sources (FFS, Arteta presser, The Athletic, SportsGambler) establish only a *developing rotation* — Timber imminent-but-eased-in, White "fine"/ever-present, Saliba out-of-window, Konsa nailed. The "Timber is clearly first-choice / overtakes White" claim was a Reddit opinion (and other commenters disagreed), so I pulled the back end from Timber-favoured to ~even. Final: White 1.00/0.90/0.82/0.74/0.66/0.60/0.55/0.52; Timber 0.00/0.10/0.18/0.26/0.34/0.40/0.45/0.48 (crosses toward even by F8, White marginally ahead). Still a clear move off the old flat 1.0, but not over-committed to unverified fan narrative. Konsa 1.0 / Saliba 0 / Mosquera 0 unchanged.
