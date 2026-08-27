# Premier League 2026-27 — state of the league (curation reference)

**Purpose.** A durable snapshot of who manages whom, who moved where, and who's injured — the facts that go stale fastest and cause the most predicted-XI errors. Load this first when refreshing lineups so you don't re-derive (and re-hallucinate) the landscape. Last compiled **2026-08-20** from multi-source research (team-by-team, ≥5 club-focused sources each). Transfer window shuts **~1 Sept 2026**, so the "live threads" below can still change.

> ⚠️ **This body is the 20-Aug snapshot. The 27-Aug reconciliation below OVERRIDES it where they conflict — and the LIVE FPL roster (`players.csv` / bootstrap) always overrides both** for "is player X at club Y." (A source is only "stale-squad" if it names a player **not in the roster** — do NOT reject a real transfer just because it post-dates this doc. Caught 27 Aug: Tomiyasu and Elanga wrongly flagged as hallucinations against this doc when both are in the roster.)

---

## Roster reconciliation update — 2026-08-27 (diff vs live FPL roster)

Confirmed against `players.csv`. The 20-Aug body missed these.

**Arrivals the body omits (all present in the roster now):**
- **Takehiro Tomiyasu → Crystal Palace** (from Arsenal)
- **Anthony Elanga → Newcastle** (from Nott'm Forest)
- **Malick Thiaw → Newcastle** · **Jacob Ramsey → Newcastle**
- **James McAtee → Nott'm Forest** (from Man City)
- **Rayan Aït-Nouri → Man City** · **Ayyoub Bouaddi → Man City** (the doc's "talks intensifying" — DONE)
- **Sávio → Spurs** (from Man City; doc's "pushing to join Spurs" — DONE. Canonical name **Sávio** everywhere; `name_mappings.csv` maps the old "Savinho"/"Savio" spellings to it)
- **Carlos Baleba → Man Utd** (doc's "imminent" — DONE; currently injured, ankle)
- Minor: Eddie Nketiah at Palace; Jamie Gittens at Chelsea; Kepa + Meslier are Arsenal's backup GKs.

**LIVE sagas — window open to Tue 1 Sept 23:00 UK, so roster ≠ final. Transfer sweep 27 Aug:**
- **Bradley Barcola → Liverpool — DEAL AGREED** (£100m + ~£20m add-ons; medical + signing imminent, ~Thu 28 Aug; This Is Anfield "agreed", multi-source). Effectively done. **KEEP the `Bradley Barcola` placeholder in `starting_lineups.csv`** (do NOT remove — my earlier "did not join" call was the roster snapshot, not the live deal). Ramp him UP F2-F8; he pressures the LFC wide slots (Gakpo/Wirtz/Ngumoha).
- **Emi Martínez → LEAVING Villa** (Suzuki is Emery's #1). Live suitors **Chelsea** ("would say yes") and **Tottenham**; expected to move before deadline. Watch which — it reshuffles that club's GK (Chelsea: Sánchez; Spurs: Kinsky). Suzuki nailed at Villa regardless.
- **Cody Gakpo → Spurs** was "on hold unless LFC sign an attacker" — **Barcola arriving reactivates this**; monitor.
- **Omar Marmoush → Spurs** (from Man City) — RUMOUR only (Spurs Web headline 27 Aug); **roster still shows him at MCI**, so NOT done. Monitor to 1 Sept; if it completes, adds a Spurs attacker and thins City's front line.
- Roster-current but window still open (were live sagas): **Sarr** at Palace, **Ben White** at Arsenal. **Enzo Fernández** stayed at Chelsea (confirmed).
- Unverified this sweep (dropped — searches returned stale Jan-2026 content): Douglas Luiz destination, Enciso→Chelsea.

**Departures FPL flags as done (still appear in roster data, but out):** Reijnders (City → Al Qadsiah), Rodrigo (City → Barcelona), Romero (Spurs → Atlético, body had this).

_Not exhaustive — regenerate the full body from the roster when time allows; until then trust this block + the live roster over the 20-Aug prose._

---

> ⚠️ **Known stale-data trap:** national aggregators (Squawka, Goal, OneFootball, FotMob, Sports Mole predicted-XI pages, some FFS *per-club preview* pages) frequently serve **2025-26 squads/managers**. Concrete phantoms caught this window: Guardiola still at City, Rodri/Reijnders at City, Salah/Konaté at Liverpool, Maresca/Cucurella/Delap at Chelsea, Mbeumo/Wissa at Brentford, Senesi at Bournemouth, Romero at Spurs, Bruno Guimarães/Isak/Gordon at Newcastle. If a source names any of these as current, discard it.

---

## Newly promoted for 2026-27
**Hull City, Coventry City, Ipswich Town.** (Their players carry **zero** 2025-26 PL DefCon minutes — confirmed in the data.) **Leeds and Sunderland were already in the PL in 2025-26** — treat them as established, NOT newly promoted (their regulars have full prior-season data). Research agents loosely called Leeds/Sunderland "promoted"; that's about a *recent* promotion, not 2026-27.

## Managers (2026-27)
| Team | Manager | Note |
|---|---|---|
| Arsenal | Mikel Arteta | — |
| Aston Villa | Unai Emery | — |
| Bournemouth | **Marco Rose** | NEW — Iraola left (Athletic Bilbao) |
| Brentford | Keith Andrews | 2nd season (Frank → Spurs, 2025) |
| Brighton | Fabian Hürzeler | extension to 2029 |
| Chelsea | **Xabi Alonso** | NEW — Maresca left → Man City |
| Coventry | Frank Lampard | promoted |
| Crystal Palace | **Pierre Sage** | NEW (ex-Lens, Jun 2026) — replaced Glasner |
| Everton | David Moyes | — |
| Fulham | **Álvaro Arbeloa** | NEW — Marco Silva → Benfica |
| Hull City | Sergej Jakirović | promoted |
| Ipswich Town | **Gary O'Neil** | NEW — McKenna resigned |
| Leeds | Daniel Farke | — |
| Liverpool | **Andoni Iraola** | NEW — Slot out |
| **Man City** | **Enzo Maresca** | NEW — **Guardiola GONE** (came from Chelsea) |
| Man Utd | Michael Carrick | made permanent |
| Newcastle | **Matthias Jaïssle** | ex-Salzburg/Al-Ahli; the "Howe stays" read was a stale source |
| Nott'm Forest | Oliver Glasner | 3-4-2-1 |
| Spurs | Roberto De Zerbi | appointed 31 Mar 2026 |
| Sunderland | Régis Le Bris | — |

---

## Per-team notes (manager · shape · key ins/outs · GW1 injuries · live threads)

### Arsenal — Arteta · 4-2-3-1
- **In:** Bruno Guimarães (£75m, Newcastle), Christos Tzolis (£34m, Club Brugge — the Trossard replacement), Cristhian Mosquera (Valencia), Viktor Gyökeres, Ezri Konsa (£51m, Villa — **missed the noon GW1 registration cutoff, unavailable until F2**).
- **Out:** Trossard (Beşiktaş).
- **Injuries:** Saliba (chronic back — **out months**, all of F1-F6), Timber (groin, ~weeks, ~F4-F6 return), Eze (calf), Bruno (thigh — GW1 doubt), Rice (hamstring-nerve — GW1 doubt/rest), Saka (limited pre-season).
- **Live:** Konsa registers ~F2; **Ben White** exit interest (Everton) — medium-term, drops his F5-F6; Zubimendi (Chelsea interest).

### Aston Villa — Emery · 4-2-3-1
- **In:** Zion Suzuki (£30m, Parma — new #1, signed 19 Aug), Garnacho (loan, Chelsea), Joao Gomes (£38m, Wolves), Johan Manzambi (~£52-59m record, Freiburg), Victor Lindelöf.
- **Out:** Konsa (→ Arsenal), Morgan Rogers (→ Chelsea £117m), Tielemans (→ Man Utd), Guessand (→ loan Palace).
- **Injuries:** Onana (ACL, to spring), Joao Gomes (calf ~5 Sept), Manzambi (knee), Garnacho (head/facial, GW1 doubt).
- **Live:** **Emi Martínez being sold** (Juventus) → Suzuki takes #1; no CB replacement for Konsa signed yet.

### Bournemouth — Marco Rose · 4-2-3-1
- **In:** António Silva (£25.7m, Benfica — Senesi replacement), Rayan (£24.7m, Vasco, Jan 2026), Álvaro Rodríguez (£25.7m, Elche), Juanlu Sánchez, Adrien Truffert.
- **Out:** Semenyo (→ Man City, Jan 2026 — now plays FOR City), Senesi (→ Spurs, free).
- **Injuries:** Kroupi (metatarsal ~3mo), Adli (calf ~1mo), Julián Araújo (thigh surgery, months), Milosavljević (knee), Christie (**suspended GW1**), Adams/Juanlu/Rodríguez (fitness, ~GW2). Also in Europa League (rotation).

### Brentford — Keith Andrews · 4-3-3
- **In:** Mamadou Sangaré (£41m record), Michael Kayode (perm, Fiorentina), Dango Ouattara, Callum Wilson (free, West Ham — striker cover).
- **Out:** Mbeumo (→ Man Utd), Wissa (→ Newcastle £55m), Frank Onyeka (→ Coventry).
- **Injuries:** van den Berg (~3mo, to ~Nov).
- **Live:** Jadon Sancho (Man Utd) exploratory.

### Brighton — Hürzeler · 4-2-3-1
- **In:** Luka Vušković (£45-50m perm, Spurs), Costinha (£10m, Olympiacos — Veltman replacement, RB), Promise David (loan, Union SG), Kostoulas.
- **Out:** Danny Welbeck (→ Chelsea), Veltman.
- **Injuries:** Mitoma (hamstring, ~late Sept/Oct), Minteh (leg surgery), Tzimas (ACL ~Oct), Ferguson (unavailable), **Baleba (ankle + Man Utd move imminent)**, Struijk (just back). Also Conference League play-off (Tromsø) — rotation.
- **Live:** Baleba → Man Utd (imminent); Minteh (Liverpool £50m rejected, staying, injured); Ferguson possible loan (Genoa).

### Chelsea — Xabi Alonso · back-3 / 4-3-3 (back-4 likely early)
- **In:** Morgan Rogers (£117m, Villa), Danny Welbeck (Brighton), Marco Palestra (Atalanta), Jorrel Hato, Estêvão.
- **Out:** **Cucurella (→ Real Madrid — hands Hato the LB job)**, Andrey Santos (→ Man Utd), Tyrique George (→ Everton), Garnacho (loan, Villa), Maresca (→ Man City).
- **Injuries:** Fofana (**suspended GW1**, back GW2), Estêvão (hamstring, ~GW1-3), Anselmino, Henderson (wrist), Emegha.
- **Live:** Enzo Fernández → Man City (Maresca reunion) — tail risk to ~1 Sept.

### Coventry — Lampard · **back-4** 4-2-3-1 · PROMOTED
- **In:** Carl Rushworth (£22.5m record, Brighton), Aurèle Amenda (£15.3m perm), Caleb Yirenkyi (£26m, Nordsjælland), Loum Tchaouna (£20m, Burnley), **Taiwo Awoniyi (£9-17m, Forest)**, Sidiki Cherif (~€24.5m), Miguel Brau (free, Granada), Frank Onyeka (Brentford).
- **Out:** Jahnoah Markelo (→ Shabab Al-Ahli).
- **Injuries:** **Haji Wright (quad, ~12 weeks)**, Woolfenden, Onyeka (knock, GW1 doubt).
- Note: Lampard plays a **back four** (van Ewijk/Dasilva are full-backs) — do not assume a back three.

### Crystal Palace — Pierre Sage · 3-4-2-1
- **In:** Yeremy Pino (£26m, Villarreal — Eze replacement), Jaydee Canvot (Toulouse), Daichi Kamada (free, Lazio), Walter Benítez (free, PSV), Borna Sosa (Ajax), Evann Guessand (loan, Villa), Jørgen Strand Larsen (£43m+5m, Jan 2026).
- **Out:** **Marc Guéhi (→ Man City, Jan 2026 £20m — settled, gone)**, Eze (→ Arsenal).
- **Injuries:** McNeil (fitness doubt), Doucouré (fitness), Muñoz (returning from injury), Wharton (ankle scare, recovered — monitor).
- **Live:** **Sarr → Galatasaray/Fenerbahçe/Man Utd** (live exit; Palace resisting) — collapses his F2-F6 if sold. Striker is a Mateta↔Strand Larsen rotation.

### Everton — Moyes · 4-2-3-1
- **In:** Tyler Dibling (£42m, Southampton), Merlin Röhl (£18m, Freiburg), Tyrique George (£18m, Chelsea), Hayden Hackney (£16.5m, Middlesbrough), Christian Nørgaard (£7m, Arsenal), Brennan Johnson (swap for McNeil, 10 Aug).
- **Out:** Dwight McNeil (→ Palace), Gueye, Coleman (released), Grealish (loan ended → City).
- **Injuries:** Garner (groin surgery), Iroegbunam (long-term + exit talk), Nørgaard (injury, ~29 Aug return). Midfield "triple blow" → Hackney + Armstrong start the pivot.

### Fulham — Arbeloa · 4-2-3-1
- **In:** Gonzalo García (~€40m, Real Madrid — Jiménez replacement), César Palacios (Real Madrid Castilla), Shea Charles (Southampton), Oscar Bobb (£27m, Man City, Jan 2026), Kevin (£34.6m record, Shakhtar, Sept 2025).
- **Out:** Raúl Jiménez (free), Harry Wilson (→ Leeds).
- **Injuries:** Andersen (**suspended GW1**, back GW2 → Cuenca deputises), Cairney (injured).

### Hull City — Jakirović · 4-2-3-1 · PROMOTED · injury crisis
- **In:** Konstantinos Tzolakis (~£20m record GK), Nobel Mendy (~£20m record CB), Jens Hjertø-Dahl (£10m, Tromsø — **a STARTER, not injured; common data error**), Laalaoui.
- **Injuries (~10-14 out GW1):** Butland (elbow, ~Christmas), Gelhardt (ankle), Morita (calf), Matazo (knee, serious/long), Gyabi (groin), Charlie Hughes (groin), Zambrano (thigh), Drameh (thigh doubt), Jacob (hip). McBurnie leads the line; Tzolakis/Mendy/Hjertø-Dahl/Stroud are debut-starters.
- **Live:** Leon Bailey (Villa) linked.

### Ipswich Town — Gary O'Neil · 4-2-3-1 · PROMOTED
- **In (£100m+ overhaul):** Julio Enciso (Brighton), Kjell Scherpen (£8.5m, Union SG), Daizen Maeda (£8.5m, Celtic), Saša Lukić (£9m, Fulham), Issa Diop (£8.5m, Fulham), Abdul Fatawu (£20m, Leicester), Emersonn (~£22-24m record, Toulouse), Florentino Luís (£16m), Abdoul Ouattara (Strasbourg), Marcelino Núñez.
- **Out:** Muric (→ Sassuolo), Wes Burns (→ Leicester). (Delap/Hutchinson left at the 2025 relegation — NOT current.)
- **Injuries:** Matusiwa, Jack Taylor.

### Leeds — Farke · 3-4-3 (established, not newly promoted)
- **In:** James Trafford (£40m GK), Dominic Calvert-Lewin (free, Everton), Harry Wilson (free, Fulham), Anton Stach, Jaka Bijol, Sean Longstaff (Newcastle), Tarik Muharemović.
- **Out:** Pascal Struijk (→ Brighton).
- **Injuries:** Gudmundsson (thigh, ~first 2 GWs), Gruev (knee doubt), Perri (wrist).

### Liverpool — Iraola · 4-2-3-1
- **In:** Alexander Isak (from Newcastle), Florian Wirtz, Milos Kerkez, Jeremie Frimpong, Giorgi Mamardashvili, Ronald Araújo (partners Van Dijk).
- **Out:** **Mohamed Salah (LEFT — 9-yr spell ended)**, Konaté (→ Real Madrid), Curtis Jones (→ Inter Milan).
- **Injuries:** Ekitiké (Achilles ~Jan), Leoni (ACL, long), Bradley (knee), Gomez (muscular ~late Aug), Jacquet (knee doubt), Bajčetić (hamstring). Mac Allister only ~54 pre-season min → **benched GW1, returns ~GW2-3**.
- **Live:** **Gakpo → Spurs** (personal terms, but ON HOLD unless LFC sign an attacker first); **Barcola ← PSG** (advanced, valuation gap — **cannot register for GW1**, possible before Sept 1). 17-yo Ngumoha is a GW1 necessity who fades as those resolve. Elliott/Chiesa exit candidates.

### Man City — Maresca · 4-2-3-1 · (Guardiola GONE)
- **In:** Elliot Anderson (£115m record, Forest), Antoine Semenyo (Bournemouth, Jan 2026), Marc Guéhi (£20m, Palace, Jan 2026), Rayan Cherki.
- **Out:** **Rodri (→ Barcelona £65m)**, Reijnders (sold), Bernardo; Grealish (loan back from Everton).
- **Injuries:** **Doku (calf, Community Shield — out several weeks, misses GW1)**, Nunes (muscle), Sávio (illness), Marmoush (knock). With Doku out, **Cherki starts** wide.
- **Live (targets, to 1 Sept):** Ayyoub Bouaddi (Lille, **DM**, 18) — talks intensifying; Manu Koné (Roma, **CM**); Enzo Fernández (Chelsea, **CM**); a **new winger** (because **Sávio is pushing to join Spurs** — e.g. Mateus Mané, Wolves); a **new right-back**. All in the pivot / wide / RB areas — **none competes for Foden's central No.10 role.**

### Man Utd — Carrick (permanent) · 4-2-3-1
- **In:** Bryan Mbeumo (Brentford), Matheus Cunha, Benjamin Šeško, Youri Tielemans (Villa), Andrey Santos (Chelsea), Senne Lammens (GK).
- **Out:** Casemiro, Antony; **Marcus Rashford RETURNED** (Barcelona declined the buy option → he's back, lacks fitness, **benched GW1** but a real F2-F6 mover).
- **Injuries:** de Ligt (back), Ugarte, Mount (knock, out GW1), Šeško (shin — GW1 start "highly unlikely", bench), Martínez (fitness), Mainoo (fitness/benched). **Mbeumo leads the line** GW1 with Šeško doubtful.
- **Live:** Rashford reintegration (rising F2-F6); Sancho (→ Brentford) exploratory.

### Newcastle — Matthias Jaïssle · 4-3-3
- **In:** Bazoumana Touré (£43m, Hoffenheim), Amar Dedić (£31m, Benfica), Aladji Bamba (£30m, Monaco), Lukáš Horníček (£26m, Braga GK — **confirmed #1**), Sean Steur (£20m, Ajax), Nick Woltemade (£69m record — **now loan-listed / out of favour**), Yoane Wissa (£55m, Brentford — **starts, off the left**).
- **Out:** **Bruno Guimarães (→ Arsenal £75m), Tonali (→ Spurs £92.5m), Gordon (→ Barcelona), Isak (→ Liverpool)**, Longstaff (→ Leeds), Trippier/Targett (free) — ~11 departures; midfield gutted.
- **Injuries:** Joelinton (2+ wk), Livramento (calf surgery — back ~GW2-3), Schär (~7mo, returning), Miley, Burn (World Cup return). Makeshift XI; GK a Horníček/Pope split.
- **Live:** **Woltemade told he can leave on loan** (Liverpool / Man Utd / Villa / Atlético circling) → **Osula leads the line, Wissa off the left** (the earlier "Woltemade leads" read was wrong). Pope transfer-listed; chasing a CM. Manager confirmed **Jaïssle** (Aug-2026 audit; the "Howe stays" reading was a stale source).

### Nott'm Forest — Glasner · 3-4-2-1
- **In:** Ousmane Diomandé (Sporting CP — straight into the back three), Xaver Schlager, Dan Ndoye; Kalimuendo returned from loan.
- **Out:** **Elliot Anderson (→ Man City £115m)**, Taiwo Awoniyi (→ Coventry).
- **Injuries:** Ryan Yates (minor), Nicolò Savona (knee) — both miss MD1.
- **Live:** chasing a CM (failed to replace Anderson).

### Spurs — De Zerbi · 4-2-3-1 · injury-hit
- **In:** Sandro Tonali (£92.5m, Newcastle), Jan Paul van Hecke (Brighton), Marcos Senesi (free, Bournemouth), Andrew Robertson (LB), Mathys Tel, Conor Gallagher, Mateus Fernandes.
- **Out:** **Cristian Romero (→ Atlético £34m)**, Vicario (→ Juventus).
- **Injuries:** Xavi Simons (ACL, long), Kudus (thigh since Jan + possible sale), Kulusevski (knee), Odobert (knee), **van de Ven (no pre-season — major GW1 doubt)**, Udogie (doubt), Maddison (ACL return — bench GW1), Solanke (fitness — out GW1), Porro (injured — Gray deputises at RB). All return over F2-F6.
- **Live:** Bergvall possible exit; Kudus possible sale.

### Sunderland — Le Bris · 4-3-3 (established, not newly promoted)
- **In:** Granit Xhaka (**stayed — resisted Chelsea**), Enzo Le Fée, Noah Sadiki, Habib Diarra (club-record), Chemsdine Talbi, Brian Brobbey, Robin Roefs (GK), Reinildo Mandava, Nordi Mukiele, Omar Alderete, Simon Adingra, Wilson Isidor.
- **Injuries:** Alderete, Mukiele (fitness, GW1 doubt); Meunier confirmed to start (earlier "unavailable" report was wrong).

---

## Live transfer sagas to watch to ~1 Sept (affect F2 onward)
- **Gakpo → Spurs** (Liverpool, on hold) · **Barcola → Liverpool** (can't register GW1) · **Sarr → Galatasaray/Fenerbahçe** (Palace) · **Woltemade → loan out** (Newcastle) · **Baleba → Man Utd** (Brighton, imminent) · **Emi Martínez → Juventus** (Villa) · **Ben White → Everton** (Arsenal) · **Bouaddi/Koné → Man City** · **Sávio → Spurs** (City then buy a winger) · **Sancho → Brentford** (exploratory) · **Enzo Fernández → Man City COLLAPSED** — staying at Chelsea · Newcastle chasing a CM; Forest chasing a CM.

## Items to VERIFY (research conflicted or user-flagged)
1. **Newcastle manager = Jaïssle** (RESOLVED, Aug-2026 audit — "Howe" was a stale-source read).
2. **Man City manager = Maresca** (user-confirmed Guardiola gone) — keep an eye that no stale source reverts this.
3. Any deal above that completes/collapses before the 1 Sept deadline.
