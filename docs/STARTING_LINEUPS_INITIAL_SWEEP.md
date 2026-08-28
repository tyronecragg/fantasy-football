# Starting lineups — initial sweep (GW2 2026-27)

**Compiled 2026-08-27.** GW2 deadline **Fri 28 Aug 17:30 UTC** (before every kickoff — no confirmed XIs before we pick). Transfer window shuts **Tue 1 Sept 23:00 UK**. **`starting_lineups.csv` not changed** — this is the reference we act on tomorrow.

## Sources — what's fetchable
| Source | Tier | This sweep |
|---|---|---|
| **FPL API** (bootstrap/fixtures/picks) | ground truth | ✅ full — flags, minutes, fixtures |
| **Fantasy Football Scout** team-news | T1 primary | ✅ live 27 Aug — XIs for the 9 big clubs |
| **RotoWire** lineups | T1 cross-check | ✅ live — **all 20** XIs |
| **SportsGambler** injuries | T2 | ✅ full, dated (updated 27 Aug 19:18) |
| All About FPL | T1 | ❌ no GW2 article yet |
| Fantasy Football Hub | T1 alt | ❌ **paywalled** |
| FPL Edits | T1 alt | ❌ **JS shell** (unfetchable) |
| Never Manage Alone / Yahoo all-20 | T1 alt | ❌ **403** |
| Premier Injuries / Official PL / Sky (injuries) | T2 | ❌ removed (403 / JS) |

**Working panel = FFS + RotoWire (XIs) + SportsGambler + FPL API (availability).** Two XI feeds + two injury feeds is enough to triangulate the whole league. `ffs_predicted_lineups.csv` on disk is stale (18 Aug) — ignore; use the live fetch. No GW2 rows in the source-accuracy ledger yet.

---

## OUR SQUAD — F1 all clear ✅
Every starter is predicted by **both** FFS and RotoWire and played GW1:

| Player (team) | FFS | RW | GW1 |
|---|:-:|:-:|---|
| Raya, Gabriel, **Tzolis (C)** (ARS) | ✓ | ✓ | 90/90/75′ |
| Ballard (SUN), Wieffer (BHA) | ✓ | ✓ | 90/77′ |
| Frimpong (LIV), Bijol (LEE) | ✓ | ✓ | 90/90 |
| Anderson (MCI), B.Fernandes (VC) (MUN) | ✓ | ✓ | 62/90 |
| Palmer, João Pedro (CHE) | ✓ | ✓ | 82/90 |
| Sangaré (BRE), Isak (LIV, bench) | ✓ | ✓ | 75/90 |

No F1 change warranted. **Amad (out)** reconfirmed. **Correction:** earlier I said Martinelli's return would pressure Tzolis — **Martinelli is NOT injured** (absent from SportsGambler *and* the API flag list); he's simply behind Saka/Tzolis/Havertz in the predicted XI. So the Tzolis watch is a pure *rotation/selection* question (Arteta's front-line depth), not an injury return.

---

## FULL-LEAGUE PREDICTED XIs + AVAILABILITY (by GW2 fixture)
XI = RotoWire base, **[split: FFS≠RW]** flags genuine coin-flips, **OUT** = SportsGambler/API with return date, **doubt** = API %.

### Fri 28 Aug — Crystal Palace v Manchester City
- **Crystal Palace:** Henderson; Richards, **Tomiyasu**, Canvot, Mitchell; Kamada, Wharton, Muñoz; McNeil, Pino, Strand Larsen. *OUT:* Sarr (groin, ~28 Aug — on the mend), Riad (knee, TBD). *(Tomiyasu & Nketiah confirmed at Palace.)*
- **Man City:** Donnarumma; Khusanov, Dias, Guéhi, Gvardiol/O'Reilly; **Anderson**, [split: FFS pairs Anderson with **Guéhi in midfield**, RW with O'Reilly]; Foden, Cherki, Semenyo; Haaland. *OUT:* **Doku** (calf, 5 Sep), Kovačić (red, TBD). *doubt:* Matheus Nunes 75%. Anderson fit (recovered from cramp).

### Sat 29 Aug — Liverpool v Nottingham Forest
- **Liverpool:** Alisson; **Frimpong**, Jacquet, van Dijk, Kerkez; Szoboszlai, Gravenberch; [split: FFS **Víctor Muñoz**, RW **Mac Allister** — Mac Allister returning from a light pre-season], Wirtz, Gakpo; **Isak**. *OUT:* Gomez (4 Sep), Ekitiké (calf, 12 Oct), Bradley (knee, 1 Jan), Leoni (ACL, TBD), Chiesa (back), Bajčetić (TBD). *Live-in:* **Barcola** (deal agreed — see transfers).
- **Nott'm Forest:** Sels; Milenković, Murillo, N.Williams, Aina; I.Sangaré, McAtee; Ndoye, Gibbs-White, Hudson-Odoi; Igor Jesus/Wood. *OUT:* Savona (knee, 11 Oct), Yates (5 Sep). *doubt:* Gibbs-White 75% (knee). ⚠️ RW listed a phantom "Cunha" at Forest — ignore (Matheus Cunha is Man Utd). **McAtee (from Man City) confirmed at Forest.**

### Sat 29 Aug — Bournemouth v Everton
- **Bournemouth:** Petrović; Truffert, Hill, Silva, Smith; Christie, Scott, Tavernier; Kluivert, Rayan, Evanilson. *OUT:* Christie (red, back ~29 Aug — likely available), Adli (20 Sep), J.Araújo (thigh, 21 Nov), Kroupi (foot, 7 Nov), Milosavljević (knee). *doubt:* Soler 75%.
- **Everton:** Pickford; Mykolenko, Branthwaite, Tarkowski, [RB open]; Garner, Röhl/Armstrong; Dewsbury-Hall, Ndiaye, George; Barry/Beto. *OUT:* Iroegbunam (11 Oct). *doubt:* Nørgaard (~29 Aug), Hackney 75%.

### Sat 29 Aug — Coventry (P) v Hull (P)
- **Coventry:** Rushworth; Dasilva, Thomas, Amenda, van Ewijk; Yirenkyi, Grimes, Hamer, Rudoni; Tchaouna, Awoniyi. *OUT:* Haji Wright (thigh, 21 Nov), Woolfenden (knock, TBD). *(Promoted — thin sourcing, escalation candidate.)*
- **Hull:** Tzolakis; Giles, Egan, Mendy, Coyle; Slater, Crooks, Stroud; Belloumi, Hjertø-Dahl, McBurnie. *OUT (injury crisis, ~9):* Butland (21 Nov), Matazo (20 Feb), Gyabi (21 Nov), Zambrano (12 Sep), Hughes (5 Sep), Gelhardt (~29 Aug), Ajayi/Coyle/Jacob (TBD). *(Promoted — escalation candidate.)*

### Sat 29 Aug — Tottenham v Newcastle
- **Tottenham:** Kinsky; Porro, van Hecke, Senesi, [LB]; Tonali, Bergvall; M.Fernandes, Tel, **Sávio**; Solanke. *OUT (heavy):* Xavi Simons (ACL, 20 Feb), Odobert (28 Nov), Maddison (shoulder, 5 Sep), Kulusevski (knee, TBD), Kudus (thigh, 5 Sep — 50% doubt), P.Sarr (~29 Aug). *(Sávio from City confirmed. Emi Martínez a live GK target.)*
- **Newcastle:** Horníček; Hall, Thiaw, Botman, Dedić; Miley, J.Ramsey; Barnes, Willock, **Elanga**; Wissa. *OUT:* Joelinton (groin, 14 Sep), Burn (ankle, TBD), Osula (TBD), Livramento (~29 Aug). **Elanga, Thiaw, J.Ramsey confirmed at Newcastle.** Osula out → Wissa/Woltemade lead.

### Sun 30 Aug — Chelsea v Brighton
- **Chelsea:** Sánchez; Gusto, Fofana, Colwill, Hato; [split: FFS **Caicedo**+Hato, RW **Lavia**+James — Caicedo 75% doubt], James; **Palmer**, Rogers; **João Pedro**. *OUT/back:* **Fofana** returns from ban (was 6 Sep in feed but eligible GW2), Henderson (wrist, 6 Sep), Emegha (~30 Aug), Badiashile (illness). *doubt:* Caicedo 75%. **Palmer & João Pedro both start** — the "out" headlines earlier were stale (April 2026).
- **Brighton:** Verbruggen; **Wieffer**, Vuskovic, [split: FFS **Boscagli+Kadıoğlu**, RW **Dunk**], De Cuyper; Ayari, Groß; Gómez, Hinshelwood, [FFS **Kostoulas** / RW **Rutter**]. *OUT:* Mitoma (hamstring, TBD), Minteh (calf, 24 Oct), Ferguson (ankle, TBD), Tzimas (ACL, 12 Sep), O'Riley (illness). *doubt:* Hinshelwood 75%, Kadıoğlu.

### Sun 30 Aug — Leeds v Brentford
- **Leeds:** Trafford; Bogle, Rodon, **Bijol**, Muharemović, Justin; Ampadu, Stach; Wilson, Aaronson, Calvert-Lewin. *OUT:* Gnonto (hamstring, 5 Sep), Gudmundsson (~30 Aug), Gruev (knee, TBD), Mateo Joseph (30 Jan). **Bijol nailed.**
- **Brentford:** Kelleher; Kayode, Collins, Ajer, Lewis-Potter; Janelt, **Sangaré**; O.Dango, Jensen, Schade; Igor Thiago. *OUT:* van den Berg (10 Oct), Milambo (knee, 10 Oct). **Sangaré nailed** (75′ GW1). ⚠️ No "Henderson" at Brentford (aggregator error earlier).

### Sun 30 Aug — Sunderland v Fulham
- **Sunderland:** Roefs; Meunier, [split: FFS **O'Nien**, RW **Alderete**], **Ballard**, Reinildo; Xhaka, Sadiki; Hume, Le Fée, Angulo; Brobbey. *OUT:* Adingra (ankle, 12 Sep). **Ballard nailed.**
- **Fulham:** Leno; Castagne, Andersen, Bassey, Robinson; Berge, Iwobi; Bobb, King, Palacios; Gonzalo García. *OUT:* Cairney (knee, 17 Oct). **Andersen** back from ban (~30 Aug).

### Sun 30 Aug — Man Utd v Ipswich (P)
- **Man Utd:** Lammens; Dalot, Maguire, [split: FFS **Heaven**, RW **Martínez**], Shaw; Tielemans, A.Santos; Mbeumo, **B.Fernandes**, [split: FFS **Dorgu**, RW **Rashford**]; [FFS **Cunha** lone / RW Cunha+Rashford]. *OUT:* de Ligt (back, 6 Sep), Ugarte (ACL, TBD), Baleba (ankle, TBD). *doubt:* Mount (~30 Aug), **Amad 75%** (our transfer-out), Heaton. **Šeško benched again** — Utd forward line the week's real coin-flip (Cunha/Rashford/Šeško/Mount).
- **Ipswich (P):** Scherpen; Davis, Diop, Greaves, O'Shea; Lukić, Núñez; Fatawu, Enciso, Maeda; Emersonn. *OUT:* Taylor (4 Sep), Matusiwa (4 Sep). *doubt:* Florentino 75%.

### Mon 31 Aug — Aston Villa v Arsenal
- **Aston Villa:** Suzuki; Cash, Lindelöf, Torres, Maatsen; Kamara, Barkley; McGinn, Buendía, [wide]; Watkins. *OUT:* **João Gomes** (red, ban to 19 Sep), Onana (knee, 1 Jun 2027), Manzambi (knee, 5 Sep), Madjo (ankle, 19 Sep), Bailey (muscle, TBD). *doubt:* **Watkins & Abraham 75%.** (**Emi Martínez leaving** — Suzuki #1.)
- **Arsenal:** **Raya**; White, Mosquera, **Gabriel**, Calafiori; Rice, Lewis-Skelly; Saka, Ødegaard, **Tzolis**; Havertz. *OUT:* Timber (groin, 12 Sep), Saliba (back, TBD). *doubt:* **Bruno Guimarães 75%** (thigh, ret ~31 Aug). Havertz false-9, **Gyökeres & Martinelli benched** (rotation, not injury).

---

## KEY F1 COIN-FLIPS (FFS ≠ RotoWire — genuine splits to keep calibrated, not force to 0/1)
- **Chelsea CM:** Caicedo (75% doubt) vs Lavia.
- **Liverpool CM:** Víctor Muñoz vs Mac Allister (returning).
- **Man Utd:** CB Heaven/Martínez; wide Dorgu/Rashford; the whole Cunha/Rashford/Šeško/Mount forward mix.
- **Brighton:** Dunk vs Boscagli+Kadıoğlu; Kostoulas vs Rutter.
- **Sunderland CB:** O'Nien vs Alderete.
- **Man City pivot:** Guéhi-in-midfield (FFS) vs O'Reilly (RW).

## F2–F8 DATED THREADS (drive the later columns — now firm via SportsGambler)
**Ramp back in:**
- **~F2 (this week):** Bruno Guimarães (ARS ~31 Aug), Nørgaard (EVE), Gudmundsson (LEE), Livramento (NEW), P.Sarr (TOT), Emegha (CHE), Christie (BOU), Andersen (FUL).
- **~F3 (early Sep):** Doku (MCI 5 Sep), Gomez (LIV 4 Sep), Gnonto (LEE 5 Sep), Yates (NFO 5 Sep), Hughes (HUL 5 Sep), Maddison/Kudus (TOT 5 Sep), de Ligt (MUN 6 Sep), Fofana/Henderson (CHE 6 Sep), Taylor/Matusiwa (IPS 4 Sep), Manzambi (AVL 5 Sep).
- **~F3–F4 (mid-Sep):** Timber (ARS 12 Sep), Zambrano (HUL 12 Sep), Tzimas (BHA 12 Sep), Adingra (SUN 12 Sep), Joelinton (NEW 14 Sep); **João Gomes ban clears 19 Sep**, Adli (BOU 20 Sep), Madjo (AVL 19 Sep).
- **~F6+ (Oct–Nov):** Ekitiké (LIV 12 Oct), Iroegbunam (EVE 11 Oct), Savona (NFO 11 Oct), Cairney (FUL 17 Oct), van den Berg/Milambo (BRE 10 Oct), Minteh (BHA 24 Oct), Kroupi (BOU 7 Nov), Araújo (BOU 21 Nov), Haji Wright/Butland/Gyabi (COV/HUL 21 Nov), Odobert (TOT 28 Nov).
- **Season-enders / very long (keep ~0):** Onana (AVL Jun 2027), Xavi Simons (TOT Feb 2027), Matazo (HUL Feb 2027), Mateo Joseph (LEE Jan 2027), Bradley (LIV Jan 2027), Leoni (LIV ACL), Ugarte (MUN ACL), Saliba (ARS back TBD), Mitoma/Ferguson (BHA TBD).

## TRANSFER SWEEP (27 Aug — window shuts Tue 1 Sept 23:00 UK)
- **Barcola → Liverpool — DEAL AGREED** (£100m + ~£20m, medical/sign ~28 Aug). **Keep the `Bradley Barcola` placeholder**; ramp up F2–F8; pressures LFC wide slots (Gakpo/Wirtz/Ngumoha), not Frimpong/Isak.
- **Emi Martínez → leaving Villa** (Suzuki #1); **Chelsea** & **Spurs** the live suitors — reshuffles that club's GK.
- **Gakpo → Spurs** may reactivate now LFC signed an attacker — monitor.
- Dropped as unverified (stale Jan-2026 search hits): Douglas Luiz destination, Enciso→Chelsea.

## PROCESS NOTES
- **Live roster is ground truth**, not the Aug-20 league-state doc. I wrongly flagged Tomiyasu (Palace) & Elanga (Newcastle) as stale — both real. Doc patched with a dated "Roster reconciliation" block; roster-missed arrivals also caught: Thiaw/J.Ramsey→NEW, McAtee→NFO, Aït-Nouri/Bouaddi→MCI, Sávio→TOT, Baleba→MUN.
- **Triangulate before trusting a single feed:** RotoWire's "Martinelli OUT" and a phantom "Cunha" at Forest were both caught by cross-checking SportsGambler + API + roster.
- **`curation_sources.md` cleaned 27 Aug:** removed unfetchable injury tables (Premier Injuries/Official PL/Sky) and Cloudflare-blocked fan sites (Arseblog, This Is Anfield, We Ain't Got No History, Roker Report, The Fighting Cock, Bluemoon).

## NEXT STEPS (tomorrow)
1. Second read of FFS + RotoWire + Sports Mole/Goal closer to the deadline (pressers land late Thu/Fri).
2. Re-pull SportsGambler (updates through the day) + retry All About FPL (should post its GW2 article).
3. For any team still split, escalate to a *fetchable* club source (per the cleaned Tier 3b).
4. Capture FFS + RotoWire + our Curated XI into the source-accuracy ledger (`tools/source_history.py`).
5. Then adjust `starting_lineups.csv` F1–F8 — including **removing the Barcola placeholder only if his deal somehow collapses** (currently: keep) and firming the coin-flips above.
