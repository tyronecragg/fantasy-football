# Starting lineups — FINAL sweep (GW3 2026-27)

**Final reconciliation compiled 2026-09-04 (deadline day).** GW3 deadline **Fri 4 Sep ~17:30 UTC** (Ipswich v Liverpool Fri 19:00; rest Sat–Sun 5–6 Sep). Window CLOSED 1 Sept. `starting_lineups.csv` **rewritten** F1=GW3 … F8=GW10 and validated.

> ⛔ **Value = P(NAMED IN THE STARTING XI)** — selection probability, not minutes. Bench = 0.0. Unanimous starter across sources → 1.0. Fractional only for a genuine slot-contest or a fitness doubt. Full rule: `inputs/curation_sources.md` + memory `no-confirmed-xi-before-deadline`.

## SOURCES (all GW3-current, 4 Sep)
Four predicted-XI feeds: **RotoWire**, **FFS**, **Fantasy Football Pundit (per-player Start %)**, **FPL-Edits-syndicated feed** (fpledits.com itself is a JS SPA — unreadable to fetchers; used a syndicated GW3 feed instead). Plus **SportsGambler** injury table (upd 4 Sep 11:18), **Sports Mole** per-club previews (4 Sep), **FFS Friday blog**, **FPL availability flags**, and club/beat sources (This Is Anfield/Paul Joyce, Ready To Go, etc.). Blocked/excluded: Squawka, WhoScored, PhysioRoom, This Is Anfield (403 to our tools), FFHub (paywall).

## RULES APPLIED
1. **Unanimous + high confidence → 1.0.** Fully-unanimous XIs (all 4 sources identical): **Aston Villa, Bournemouth, Fulham, Man Utd, Newcastle, Sunderland**. Their nailed starters + every other unanimous nailed-on player = 1.0.
2. **Coin-flips split** by source agreement + FFP Start %.
3. **Injured / suspended / departed → 0** (F1), respecting the FPL ceiling (status i/u/s → 0; d → chance%).
4. **F2–F8 forward-looking** — return dates + new-signing integration ramped, no generic decay.

## KEY RESOLUTIONS / CORRECTIONS (final data overturned earlier reads)
- **Fofana (CHE) PLAYS** — in all 3 XIs; the earlier "suspended GW3" (SportsGambler) was not borne out → 0.75.
- **Bobb (FUL) starts, not Shea Charles** — all 3 sources start Bobb (0.80); Shea Charles → 0.35.
- **Emersonn (IPS) fit & unanimous** → 0.80 (reverses the morning "knee, out").
- **Disasi (CRY) added** — unanimous in Palace's back three (0.60); Tomiyasu shaded to 0.40.
- **Doku (MCI) back ~GW4**, not January (one source's Jan claim rejected as an outlier) → F1 0, F2 0.55+.
- **Ballard (SUN) unanimous, Danso benched by all three** → Ballard 0.80, Danso 0.20 (the Danso-for-Ballard fear didn't materialise).
- **Frimpong (LIV) genuine RB coin-flip** (Szoboszlai can shift to RB) → 0.65, NOT settled.
- **Iroegbunam (HUL) → 0** (FPL injured, despite one XI listing him).

## COIN-FLIPS (kept split — the live calls)
Arsenal LW **Tzolis 0.55 / Eze 0.45**; Bournemouth DM Cook/Adams; Brentford AMC **Damsgaard 0.60 / Jensen 0.35**; Brighton CF Kostoulas/Rutter (+Ayari 75% doubt); Chelsea RWB Neto/Gusto(not rostered); Coventry LF Thomas-Asante/Mason-Clark; Palace 2nd-fwd Nketiah/McNeil; Everton 3-way Johnson/Hackney/Grealish; Hull DM Gourna-Douath/Crooks; Ipswich DM Núñez/(phantom); Leeds Wilson/Aaronson; **Liverpool RB Frimpong 0.65 / Gravenberch 0.45 (Szoboszlai 0.90 plays either way)**; Man City RB Khusanov/Nunes; Forest CM Sangaré/Schlager; Spurs FB reshuffle Udogie/Robertson/Gray + Sávio 50%.

## F2–F8 RETURN / INTEGRATION THREADS
- **GW4:** Doku, Norgaard, Mitoma, Timber; new signings begin (Enzo MCI, Šeško MUN, Delap NFO, Barcola LIV, Livramento NEW).
- **GW5:** João Gomes ban ends, Joelinton, Osula, Baleba, Adingra.
- **GW7–8:** Mateta (CRY ~11 Oct), Rodon (LEE 8–10wk), van den Berg/Milambo (BRE), Cairney (FUL), Woolfenden/K-Hayden (COV).
- **GW9+:** Diarra (SUN 6–8wk). **Season/very long:** Saliba, Onana, Simons, Ugarte, Riad, Minteh, Bradley/Leoni.

## VALIDATION (post-write)
All values in [0,1]; **every team sums to 11.00 across F1–F8**; **zero FPL-ceiling breaches** (Sarr/Livramento zeroed F1 for injury; Ayari capped so Brighton's water-fill can't push him over 0.75); no team-name mismatches. 3 formation flags = wing-back/false-9 FPL-label artifacts, not errors.

## 🔴 VERIFY AT THE DEADLINE (ours in bold)
1. **Frimpong (LIV RB)** — Szoboszlai-to-RB would bench him. Fri 19:00 KO.
2. **Tzolis (ARS LW)** vs Eze — Arsenal play Sun, latest news.
3. Brighton (Ayari/Rutter fitness), Spurs FB reshuffle, Everton 3-way, City RB, Forest Sangaré — non-squad.

## NEXT
`--seed-ours --gw 3` (freeze our XI) + capture RotoWire/FFS/FFP; `python -m fpl_pipeline.run` then `--gw 3` at deadline on real odds; `--score --gw 3` after it plays.

## DEADLINE SWEEP — 4 Sep ~16:45 UTC (five club-group web sweeps, sources dated 2–4 Sep only)
Applied to `starting_lineups.csv` (F1 unless noted; pipeline re-run, curation checks clean):
- **Manager/press-conference outs:** Jensen (BRE) 0 / F2 0.10 (Andrews: "won't be available"); Morita (HUL) 0 / F2 0.25 (calf); Richarlison (TOT) 0 F1–F8 (left out of PL squad); Shaw (MUN) 0.60 + Mazraoui 0.50 (late fitness test); Mount 0.10.
- **Missing from the file, in the roster:** Gusto (CHE) 0.70 (GW2 90 min, 2/4 XIs); Harwood-Bellis (AVL) 0.15; Exequiel Palacios (IPS) 0.40; Bouaddi (MCI) 0.25; Malick Fofana (SUN) 0.40.
- **Splits re-weighted on ≥3-source evidence:** Mac Allister (LIV) 0.45→0.75; Charles 0.65/Bobb 0.55/C.Palacios 0.65 (FUL); Tomiyasu 0.60/Disasi 0.45, Khalaili 0.80, Pino 0.85 (CRY); Hamer 0.35/Rudoni 0.65, Mason-Clark 0.70/Thomas-Asante 0.30, Awoniyi 0.85 (COV); Hackney 0.20/Grealish 0.55/Garner 0.85/George 0.65/Dibling 0.30 (EVE); Iroegbunam 0.45, Crooks 0.25, Targett/Gelhardt 0.10 (HUL); Brobbey 0.55/Isidor 0.45, Hume 0.60, Alderete/Danso 0.10 (SUN); Garnacho 0.30, Abraham 0.15 (AVL); Damsgaard 0.90, Lewis-Potter 0.85, Ouattara 0.70/Anthony 0.30 (BRE); Yalcouyé 0.85 (BHA); Fofana 0.90, Lavia 0.85, Estêvão 0.20 (CHE); Florentino 0.15, Clarke/Akpom 0.25 (IPS); Enzo 0.60, Nunes 0.10 (MCI); Delap 0.20 (NFO); Sávio 0.50, Gray 0.35, Solanke 0.30, Maddison 0.10 (TOT).
- **Confirmed as-is:** Tzolis/Eze split, Frimpong, Villa/Bournemouth/Newcastle/Leeds/Forest XIs, Kostoulas/Rutter, Willock, Nketiah/McNeil.
- Curated GW3 re-frozen in the source ledger after these edits.
- **Open:** F2–F8 columns sum above 11 for 10 teams (Newcastle F8 12.5, Man Utd 12.3, Villa 12.1) — return ramps were added without fading the displaced players, so nailed starters get squeezed ~15% in F5–F8 at consumption. Not touched at the deadline; fix per team after GW3.
