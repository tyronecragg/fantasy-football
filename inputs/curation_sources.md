# Curation sources — trusted panel for predicted lineups

Reference for the weekly start-probability curation (step 3b). The goal is a **stable base
panel** used for every team, with **club-specific escalation only where it matters**. Trust
here is **provisional** — it should be earned by measurement, not reputation (see the
"Earning trust" section at the bottom, tracked as a to-do).

The base panel is the workhorse: for a settled team (Arsenal, Liverpool) three sources agree
and you're done. Escalation is a targeted second look for the ~20% of teams that are hard to
call in a given week — and **which teams those are changes weekly**.

---

## Tier 1 — Base panel (predicted XIs, all 20 teams, every week)

Cross-reference these three. Where they agree → high confidence. Where they split → that's
your judgement call, and the trigger to escalate to Tier 3.

| Source | Role | Notes |
|---|---|---|
| **Fantasy Football Scout** — [team-news](https://www.fantasyfootballscout.co.uk/team-news) | **Primary, automated** | Already scraped by `starting_lineups.py` into `ffs_predicted_lineups.csv` + `ffs_team_news.md`. FPL-native; grades doubts with a % chance. |
| **RotoWire** — [lineups](https://www.rotowire.com/soccer/lineups.php) | Cross-check | Predicted *and* confirmed XIs, injury feed integrated, all 20 clubs. |
| **All About FPL** — [predicted lineups](https://allaboutfpl.com/category/predicted-lineups/) | Cross-check | All 20 teams, updates after friendlies. Good third consensus voice. |

Alternatives if one is thin: [Fantasy Football Hub](https://www.fantasyfootballhub.co.uk/premier-league-predicted-lineups)
(press-conference round-ups), [FPL Edits](https://fpledits.com/predicted-lineups-pl)
(XIs + subs), [Squawka per-match previews](https://www.squawka.com).

**Keep it to three.** More sources add noise and conflicting signals, not clarity.

---

## Tier 2 — Injury / availability ground truth (when there's a fitness question)

Use to confirm a doubt, get a return date, or resolve a disagreement about *why* someone is
out. These feed `inputs/unavailable_players.csv`.

- [Premier Injuries](https://www.premierinjuries.com) — the underlying data most others cite.
- [Official Premier League injury list](https://www.premierleague.com/en/latest-player-injuries) — authoritative.
- [Sky Sports injury table](https://www.skysports.com/football/news/11661/13567456) — return dates and days-lost, searchable by club.

---

## Tier 3 — Club-specific escalation (ONLY for flagged hard teams)

Reach here when Tier 1 **disagrees**, or the team is genuinely hard to predict — **promoted,
new manager, heavy summer turnover, or an injury cloud**. A club beat writer knows that squad
better than a national outlet filling in 20 XIs quickly.

- **Promoted / smaller clubs** (2026-27: Coventry, Hull, Ipswich) — local press and fan sites.
  The national aggregators are weakest exactly here.
- **Big clubs** — [ESPN week previews](https://www.espn.com/soccer) go deeper on the top six.
- **Manager pre-match pressers** — the ground truth for "who's fit / who rotates". Official
  club channels report these.

Don't escalate for settled teams. Any three sources agree on Arsenal; the effort is wasted.

---

## Not usable for selection, but needed for the tracker

**Confirmed XIs** (RotoWire, [Sports Gambler](https://www.sportsgambler.com/lineups/football/england-premier-league/))
publish ~1 hour before each kick-off — **after the FPL deadline** (90 min before the *first*
match of the gameweek). So they cannot inform team selection. They are the **ground truth for
scoring source accuracy** (see below).

---

## Earning trust (to-do — needs a few gameweeks of data)

Do not *assign* trust to a source — **measure** it. We already store the actual lineups for
every match (`By Gameweek/GW*/lineups.csv`, `is_starting`). So each source's predicted XI can
be scored against who actually started, giving a hit rate per source — and per club-type, so
we can see who is best on established teams versus promoted ones. The reliable sources earn
their Tier-1 place with a number; authoritative-looking-but-inaccurate ones get demoted. Same
principle as the longshot calibration: outcomes decide.
