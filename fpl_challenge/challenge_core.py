"""Shared engine for the FPL Challenge weekly optimisers.

Every week uses the same base game:
  - Pick 1 goalkeeper + 5 outfielders in one of six allowed formations
    (GK implicit; outfield shape is DEF-MID-FWD): 1-1-3, 1-2-2, 1-3-1, 2-1-2, 2-2-1, 3-1-1.
  - No budget. A per-club cap (default 1, some weeks 3).
  - One captain scores DOUBLE.

What differs week to week is the scoring twist, which each gw*.py expresses as two
columns on the player frame before calling solve_and_report:
  eff_xp    effective expected points for the week (already includes any x2 doubling)
  cap_bonus the extra points if this player is captain (one more copy of their score)

With a per-club cap the pick is not separable by position, so each formation is solved
exactly as a small integer program.
"""
import os
import sys
import unicodedata

import numpy as np
import pandas as pd
import pulp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAYERS_CSV = os.path.join(ROOT, "outputs", "13_players_master.csv")
KICKOFFS_CSV = os.path.join(ROOT, "outputs", "fixture_kickoffs.csv")
INPUTS = os.path.join(ROOT, "inputs")
XP_COL = "F1 XP"

# Lineups drop ~1h before kickoff, so a player's start is "known" from this long before KO.
LINEUP_LEAD_MIN = 60

# Kickoffs are stored in UTC; display them in SAST (UTC+2). Timing maths stays in UTC.
DISPLAY_OFFSET = pd.Timedelta(hours=2)


def load_kickoffs(gw="f1"):
    """team -> UTC kickoff Timestamp for the current gameweek, from fixture_kickoffs.csv
    (written by tools/betway.py). Empty dict if the file isn't there — timing then unknown."""
    ko = {}
    if not os.path.exists(KICKOFFS_CSV):
        return ko
    k = pd.read_csv(KICKOFFS_CSV)
    if "gw" in k.columns and (k["gw"] == gw).any():
        k = k[k["gw"] == gw]
    for _, r in k.iterrows():
        t = pd.to_datetime(r.get("kickoff_utc"), errors="coerce")
        if pd.notna(t):
            for tm in (r.get("home_team"), r.get("away_team")):
                if isinstance(tm, str) and tm.strip():
                    ko[tm.strip()] = t
    return ko

FORMATIONS = {
    "1-1-3": (1, 1, 3), "1-2-2": (1, 2, 2), "1-3-1": (1, 3, 1),
    "2-1-2": (2, 1, 2), "2-2-1": (2, 2, 1), "3-1-1": (3, 1, 1),
}

GOAL_POINTS = {"GK": 10, "DEF": 6, "MID": 5, "FWD": 4}  # mirrors fpl_pipeline/model.py


def norm(s):
    """Accent- and case-insensitive key for name matching."""
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()


def load_players():
    df = pd.read_csv(PLAYERS_CSV)
    df.columns = df.columns.str.strip()
    if XP_COL not in df.columns:
        sys.exit(f"'{XP_COL}' column not found in {PLAYERS_CSV}")
    df["Position"] = df["Position"].astype(str).str.upper().replace({"GKP": "GK"})
    df["_key"] = df["Player Name"].map(norm)
    return df.reset_index(drop=True)


def match_names(df, names):
    """Resolve a list of raw names to canonical Player Name values in df.
    Returns (matched set, unmatched list)."""
    by_key = df.set_index("_key")["Player Name"]
    matched, missed = set(), []
    for name in names:
        key = norm(name)
        if not key:
            continue
        if key in by_key.index:
            hit = by_key.loc[key]
            matched.add(hit if isinstance(hit, str) else hit.iloc[0])
        else:
            hit = df[df["_key"].str.contains(key.split()[-1], regex=False)]
            if len(hit):
                matched.add(hit.iloc[0]["Player Name"])
            else:
                missed.append(name)
    return matched, missed


def apply_exclusions(df, raw_names):
    """Drop explicitly-excluded players from the pool before optimising (e.g. a player you
    already know is injured/benched, or one you simply don't want). Names are matched the
    same accent/case-insensitive way as everywhere else; unmatched names are reported."""
    if not raw_names:
        return df
    excl, missed = match_names(df, raw_names)
    if missed:
        print(f"  ! exclude: no match for {', '.join(missed)}")
    keep = ~df["Player Name"].isin(excl)
    if (~keep).any():
        print(f"  excluding {int((~keep).sum())} player(s): {', '.join(sorted(excl))}")
    return df[keep].reset_index(drop=True)


def _series(df, col):
    return df[col].fillna(0.0) if col in df.columns else pd.Series(0.0, index=df.index)


def attacking_points(df):
    """Expected points from goals + assists this week (start-scaled), i.e. the amount
    that a 'goals and assists double' rule adds on top of normal XP."""
    pos = df["Position"]
    start = _series(df, "F1 Start")
    goals = pos.map(GOAL_POINTS).astype(float) * (
        _series(df, "F1 Score 1+") + _series(df, "F1 Score 2+") + _series(df, "F1 Score 3+"))
    assists = 3.0 * (_series(df, "F1 Assist") + _series(df, "F1 Assist 2+"))
    return start * (goals + assists)


def defcon_points(df, per_hit):
    """Expected defensive-contribution points at `per_hit` points per threshold hit
    (start-scaled). DEF uses the DEF probability, MID the MID probability; the model
    carries no forward DefCon, so forwards get 0."""
    pos = df["Position"]
    start = _series(df, "F1 Start")
    dc = np.where(pos == "DEF", _series(df, "F1 Defensive Contribution - DEF"),
                  np.where(pos == "MID", _series(df, "F1 Defensive Contribution - MID"), 0.0))
    return start * per_hit * pd.Series(dc, index=df.index)


def solve_formation(df, shape, max_per_club):
    """Exact ILP for one (DEF, MID, FWD) shape under the club cap.
    Returns (total, xi_dataframe, captain_row) or None if infeasible."""
    need = {"GK": 1, "DEF": shape[0], "MID": shape[1], "FWD": shape[2]}
    idx = list(df.index)
    prob = pulp.LpProblem("challenge", pulp.LpMaximize)
    pick = {i: pulp.LpVariable(f"pick_{i}", cat="Binary") for i in idx}
    capt = {i: pulp.LpVariable(f"capt_{i}", cat="Binary") for i in idx}

    prob += pulp.lpSum(df.loc[i, "eff_xp"] * pick[i] for i in idx) \
        + pulp.lpSum(df.loc[i, "cap_bonus"] * capt[i] for i in idx)

    for pos, k in need.items():
        pos_idx = [i for i in idx if df.loc[i, "Position"] == pos]
        prob += pulp.lpSum(pick[i] for i in pos_idx) == k
    prob += pulp.lpSum(capt[i] for i in idx) == 1
    for i in idx:
        prob += capt[i] <= pick[i]
    for team in df["Team"].unique():
        team_idx = [i for i in idx if df.loc[i, "Team"] == team]
        prob += pulp.lpSum(pick[i] for i in team_idx) <= max_per_club

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if prob.status != pulp.LpStatusOptimal:
        return None
    chosen = [i for i in idx if pick[i].varValue > 0.5]
    cap_i = next(i for i in idx if capt[i].varValue > 0.5)
    xi = df.loc[chosen].copy()
    total = xi["eff_xp"].sum() + df.loc[cap_i, "cap_bonus"]
    return total, xi, df.loc[cap_i]


def best_over_formations(df, max_per_club):
    """Best (total, xi, cap) across all six formations, or None."""
    best = None
    for shape in FORMATIONS.values():
        r = solve_formation(df, shape, max_per_club)
        if r and (best is None or r[0] > best[0]):
            best = r
    return best


def highlight_if_start(df, base_total, base_xi, max_per_club, boost_tag,
                       max_candidates=40, min_gain=0.1):
    """'Bring in if they start' watchlist, grouped by club.

    Because the Challenge lets you change a player until their match kicks off, a rotation
    risk you left out becomes worth having the moment they're confirmed. Every player's
    points scale with start probability, so their value IF they start is eff_xp / start.

    For each non-nailed player (0 < start < 1) not already picked, we raise them to that
    confirmed value and re-solve every formation; if the best team then beats the base pick,
    they're a candidate. Candidates are grouped by their own club (i.e. the lineup you watch),
    listed under that club's provisional picks. Only upgrades worth >= min_gain are shown.

    Timing is NOT wired in yet. Within one match you see the whole XI at once before kickoff,
    so a same-club swap is always actionable; a CROSS-club swap only works if the incoming
    player's news lands before the player they'd replace kicks off — check that by eye until
    fixture kickoff times are available.
    """
    if "F1 Start" not in df.columns:
        return
    base_names = set(base_xi["Player Name"])
    start = df["F1 Start"].fillna(0.0)
    base = df["Player Name"].isin(base_names)
    min_base_eff = df.loc[base, "eff_xp"].min()
    # Value IF they start. Only the PRE-BONUS points scale with start (F1 XP Pre = start x
    # full-match value, exact); the bonus term does not, so dividing the whole XP by start
    # would over-inflate it. Scale the pre-bonus part, keep the bonus, and lift eff/cap by
    # that ratio (exact for the x2 weeks; a close approximation for the add-on weeks).
    if "F1 XP Pre" in df.columns:
        xppre = df["F1 XP Pre"].fillna(0.0)
        full_single = xppre / start.clip(lower=1e-6) + (df[XP_COL] - xppre)
        ratio = full_single / df[XP_COL].where(df[XP_COL].abs() > 1e-9, other=1.0)
    else:
        ratio = 1.0 / start.clip(lower=1e-6)       # fallback: whole-XP scaling
    cond_eff = df["eff_xp"] * ratio
    cond_cap = df["cap_bonus"] * ratio

    # only uncertain starters who, confirmed, could out-score the weakest current pick
    cand = df.index[(start > 0.0) & (start < 1.0) & (~base) & (cond_eff > min_base_eff)]
    cand = sorted(cand, key=lambda i: cond_eff[i], reverse=True)[:max_candidates]

    hits, hidden = [], 0
    for i in cand:
        tmp = df.copy()
        tmp.loc[i, "eff_xp"] = cond_eff[i]
        tmp.loc[i, "cap_bonus"] = cond_cap[i]
        best = best_over_formations(tmp, max_per_club)
        if not best or best[0] <= base_total + 1e-6:
            continue
        new_names = set(best[1]["Player Name"])
        if df.loc[i, "Player Name"] not in new_names:
            continue
        gain = best[0] - base_total
        if gain < min_gain:
            hidden += 1
            continue
        hits.append((i, cond_eff[i], gain, base_names - new_names))

    # timing: a swap works only if the incoming player's lineup news (~LINEUP_LEAD_MIN before
    # their kickoff) lands before the player they'd replace kicks off. Same-match swaps always
    # qualify (you see the whole XI at once). Needs fixture_kickoffs.csv; unknown without it.
    kickoffs = load_kickoffs()
    n2t = dict(zip(base_xi["Player Name"], base_xi["Team"]))
    lead = pd.Timedelta(minutes=LINEUP_LEAD_MIN)

    def swap_status(inc_team, dropped):
        ki = kickoffs.get(inc_team)
        outs = [kickoffs.get(n2t.get(n)) for n in dropped]
        if ki is None or not outs or any(o is None for o in outs):
            return "timing?"
        if all(n2t.get(n) == inc_team for n in dropped):
            return "same match"
        return "actionable" if (ki - lead) < min(outs) else "TOO LATE"

    def fmt_ko(t):  # stored UTC -> displayed SAST (UTC+2); comparisons above stay in UTC
        return (t + DISPLAY_OFFSET).strftime("%a %d %b %H:%M SAST") if t is not None else "kickoff ?"

    print("\nBRING IN IF THEY START — watch each club's lineup; if a listed player starts, swap in")
    print("(only swaps you could actually make in time are shown)")

    # keep only actionable swaps (drop TOO LATE); a club with none left isn't printed
    picks_by_club = base_xi.groupby("Team")["Player Name"].apply(list).to_dict()
    hits_by_club, too_late = {}, 0
    for i, ce, gain, dropped in hits:
        club = df.loc[i, "Team"]
        st = swap_status(club, dropped)
        if st == "TOO LATE":
            too_late += 1
            continue
        hits_by_club.setdefault(club, []).append((i, ce, gain, dropped, st))

    footer = [f"{n} {label}" for n, label in
              ((hidden, f"below +{min_gain:.1f}"), (too_late, "too late to swap")) if n]
    if not hits_by_club:
        print("    none actionable — worthwhile upgrades are all nailed, selected, or too late")
        if footer:
            print(f"    ({', '.join(footer)})")
        return

    # most valuable club first: biggest available upgrade at the top
    clubs = sorted(hits_by_club, key=lambda c: max(h[2] for h in hits_by_club[c]), reverse=True)
    for club in clubs:
        prov = ", ".join(picks_by_club.get(club, [])) or "none"
        print(f"\n  {club}  [{fmt_ko(kickoffs.get(club))}]  (provisional: {prov})")
        for i, ce, gain, dropped, st in sorted(hits_by_club[club], key=lambda h: h[2], reverse=True):
            r = df.loc[i]
            who = ", ".join(sorted(dropped)) if dropped else "(reshuffle)"
            print(f"      [{st:<10}] {r['Position']:<4}{r['Player Name']:<22}"
                  f"start {r['F1 Start']*100:3.0f}%  xp {ce:5.2f}  +{gain:.2f}  (replaces {who})")
    if footer:
        print(f"\n  ({', '.join(footer)} — hidden)")


def solve_and_report(df, max_per_club, title_lines, boost_tag):
    """Rank all six formations on the prepared frame and print the best XI.
    df must carry: eff_xp, cap_bonus, boosted (bool). boost_tag labels boosted rows."""
    print("=" * 72)
    for ln in title_lines:
        print(ln)
    print(f"max {max_per_club} per club   |   boosted players marked ({boost_tag})")
    print("=" * 72)

    results = []
    for name, shape in FORMATIONS.items():
        res = solve_formation(df, shape, max_per_club)
        if res:
            results.append((name, *res))
    if not results:
        sys.exit("No feasible formation (club cap too tight?).")
    results.sort(key=lambda r: r[1], reverse=True)

    print("\nFORMATION RANKING (projected total, captain applied)")
    for name, total, xi, cap in results:
        nb = int(xi["boosted"].sum())
        print(f"  {name:<7} {total:6.2f}   (captain {cap['Player Name']}, {nb} boosted in XI)")

    name, total, xi, cap = results[0]
    print("\n" + "=" * 72)
    print(f"BEST PICK  ->  {name}   projected {total:.2f} pts")
    print("=" * 72)
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    for _, row in xi.sort_values("Position", key=lambda c: c.map(order)).iterrows():
        tag = f" ({boost_tag})" if row["boosted"] else ""
        star = "  (C)" if row["Player Name"] == cap["Player Name"] else ""
        print(f"    {row['Position']:<4}{row['Player Name']:<24}{row['Team']:<15}"
              f"xp {row[XP_COL]:4.2f} -> {row['eff_xp']:5.2f}{tag}{star}")
    counts = cap["eff_xp"] + cap["cap_bonus"]
    print(f"\n  Captain: {cap['Player Name']}  "
          f"(base xp {cap[XP_COL]:.2f}, effective {cap['eff_xp']:.2f} -> counts {counts:.2f})")

    print(f"\nTOP BOOSTED PLAYERS BY EFFECTIVE XP (what drives the pick)")
    top = df[df["boosted"]].sort_values("eff_xp", ascending=False).head(12)
    for _, r in top.iterrows():
        print(f"    {r['Position']:<4}{r['Player Name']:<24}{r['Team']:<15}"
              f"xp {r[XP_COL]:4.2f} -> {r['eff_xp']:5.2f}")

    highlight_if_start(df, total, xi, max_per_club, boost_tag)
    return results
