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
PLAYERS_CSV = os.path.join(ROOT, "outputs", "players_master.csv")
KICKOFFS_CSV = os.path.join(ROOT, "outputs", "fixture_kickoffs.csv")
INPUTS = os.path.join(ROOT, "inputs")
XP_COL = "F1 XP"

# Lineups drop ~1h before kickoff, so a player's start is "known" from this long before KO.
LINEUP_LEAD_MIN = 60

# Kickoffs are stored in UTC; display them in SAST (UTC+2). Timing maths stays in UTC.
DISPLAY_OFFSET = pd.Timedelta(hours=2)


def _official_kickoffs(gameweek):
    """team -> UTC kickoff for one gameweek from FPL-Core-Insights (official FPL fixture times).
    Used in archive mode: the Betway kickoff file only ever holds the CURRENT weeks, so after a
    roll it no longer matches an archived gameweek."""
    import glob as _glob
    seasons = sorted(_glob.glob(os.path.join(ROOT, "fpl_data", "FPL-Core-Insights", "data", "20*")))
    base = os.path.join(seasons[-1], "By Gameweek", f"GW{gameweek}") if seasons else ""
    fx, tm = os.path.join(base, "fixtures.csv"), os.path.join(base, "teams.csv")
    if not (os.path.exists(fx) and os.path.exists(tm)):
        print(f"  ! no official fixtures for GW{gameweek} — kickoff timing unavailable")
        return {}
    f, t = pd.read_csv(fx), pd.read_csv(tm)
    if "tournament" in f.columns:
        f = f[f["tournament"].astype(str).str.contains("prem", case=False)]
    name = dict(zip(t["code"], t["name"]))
    ko = {}
    for _, r in f.iterrows():
        when = pd.to_datetime(r["kickoff_time"], errors="coerce")
        if pd.isna(when):
            continue
        when = when.tz_localize(None) if when.tzinfo else when
        for code in (r["home_team"], r["away_team"]):
            if code in name:
                ko[name[code]] = when
    return ko


def load_kickoffs(gw="f1"):
    """team -> UTC kickoff Timestamp for the current gameweek, from fixture_kickoffs.csv
    (written by tools/betway.py). Empty dict if the file isn't there — timing then unknown."""
    if _ARCHIVE_GW is not None:
        return _official_kickoffs(_ARCHIVE_GW)
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


ARCHIVE_CSV = os.path.join(ROOT, "inputs", "historical_player_data.csv")
_ARCHIVE_NOTE = None   # set by load_players when reading the archive; re-printed in reports
_ARCHIVE_GW = None      # gameweek being read from the archive, if any
_CONFIRMED_STARTERS = set()   # names confirmed via confirm_starting; no contingency needed
_CONFIRMED_OUT = set()        # raw names passed to confirm_not_starting (removed from the pool)
SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
# live mid-week state (set_live): players whose match has started, which of them are locked into
# your team, and a locked captain. Applied inside every solve, so all sections respect it.
_LIVE = {"started": set(), "locked": set(), "cap": None, "now": None}


def make_snapshot(gameweek):
    """Copy that gameweek's archive rows to fpl_challenge/snapshots/gwN.csv — a Challenge-only
    copy you can refresh with late odds without touching the archive."""
    a = pd.read_csv(ARCHIVE_CSV, low_memory=False)
    season = sorted(a["Season"].astype(str).unique())[-1]
    rows = a[(a["Season"].astype(str) == season) & (a["Gameweek"] == int(gameweek))]
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    path = os.path.join(SNAPSHOT_DIR, f"gw{int(gameweek)}.csv")
    rows.to_csv(path, index=False)
    print(f"snapshot written: {path} ({len(rows)} players)")
    return path


def set_live(df, current_team=None, captain=None, now=None):
    """Mid-week mode: lock in your current picks whose match has already kicked off, bar anyone
    else whose match has started, and keep the armband on a captain who has already played
    (otherwise it may only move to someone whose match hasn't started). now defaults to the
    current time; kickoffs are compared in UTC."""
    if not current_team:
        return
    now = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else pd.Timestamp(now)
    ko = load_kickoffs()
    started = {n for n, t in zip(df["Player Name"], df["Team"]) if t in ko and ko[t] <= now}
    # players already confirmed out were removed from the pool — expected, not a typo
    current_team = [n for n in current_team if norm(n) not in _CONFIRMED_OUT]
    if captain and norm(captain) in _CONFIRMED_OUT:
        print(f"  captain {captain} is confirmed out — armband will move")
        captain = None
    team, missed = match_names(df, current_team)
    if missed:
        _big_warning([f"current_team: NO MATCH for {', '.join(missed)} — not locked in."])
    cap = None
    if captain:
        c, cm = match_names(df, [captain])
        if cm:
            _big_warning([f"captain: NO MATCH for {captain}."])
        cap = next(iter(c), None)
    _LIVE.update(started=started, locked=team & started,
                 cap=cap if cap in started else None, now=now)
    print(f"LIVE @ {(now + DISPLAY_OFFSET):%a %d %b %H:%M} SAST — already played, locked in: "
          f"{', '.join(sorted(team & started)) or 'none'}"
          + (f"; captain {cap} locked" if cap in started else
             "; armband free to move to anyone yet to play"))


def _archive_reminder():
    if _ARCHIVE_NOTE:
        _big_warning(_ARCHIVE_NOTE)


def load_players(archive_gw=None, snapshot_gw=None, gw=None):
    """The live master (whatever F1 currently is); or with snapshot_gw, the refreshable Challenge
    snapshot fpl_challenge/snapshots/gwN.csv (preferred once the pipeline has rolled on); or with
    archive_gw, that gameweek's frozen archive rows (the untouchable decision record)."""
    global _ARCHIVE_NOTE, _ARCHIVE_GW
    # gw=N: use snapshots/gwN.csv automatically when it exists, else the live master
    if snapshot_gw is None and archive_gw is None and gw is not None:
        if os.path.exists(os.path.join(SNAPSHOT_DIR, f"gw{int(gw)}.csv")):
            snapshot_gw = int(gw)
        else:
            print(f"  GW{int(gw)}: no snapshot - using the live master (outputs/players_master.csv)")
    if snapshot_gw is not None:
        path = os.path.join(SNAPSHOT_DIR, f"gw{int(snapshot_gw)}.csv")
        if not os.path.exists(path):
            sys.exit(f"No snapshot at {path} — create it with make_snapshot({int(snapshot_gw)})")
        df = pd.read_csv(path, low_memory=False)
        _ARCHIVE_GW = int(snapshot_gw)          # kickoffs then come from official fixtures
        _ARCHIVE_NOTE = [f"Using Challenge SNAPSHOT GW{snapshot_gw} ({len(df)} players) — refreshable copy,",
                         "the archive is untouched."]
        _big_warning(_ARCHIVE_NOTE)
    elif archive_gw is None:
        df = pd.read_csv(PLAYERS_CSV)
    else:
        a = pd.read_csv(ARCHIVE_CSV, low_memory=False)
        season = sorted(a["Season"].astype(str).unique())[-1]
        df = a[(a["Season"].astype(str) == season) & (a["Gameweek"] == int(archive_gw))].copy()
        if df.empty:
            sys.exit(f"No archive rows for {season} GW{archive_gw} in {ARCHIVE_CSV}")
        _ARCHIVE_GW = int(archive_gw)
        _ARCHIVE_NOTE = [f"Using ARCHIVED projections: {season} GW{archive_gw} ({len(df)} players).",
                         "Frozen at that week's pipeline run — later lineup curation is not in it."]
        _big_warning(_ARCHIVE_NOTE)
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


def _big_warning(lines):
    """A hard-to-miss banner — used when a confirmed-lineup name doesn't match any player,
    since that silently means the intended fix (drop / force-start) did NOT happen."""
    bar = "!" * 74
    print("\n" + bar)
    for ln in lines:
        print("!!  " + ln)
    print(bar + "\n")


def confirm_not_starting(df, raw_names):
    """Drop players confirmed out of the XI (benched/injured/rotated), so they can't be
    picked. Names are matched the same accent/case-insensitive way as everywhere else;
    unmatched names are reported."""
    if not raw_names:
        return df
    _CONFIRMED_OUT.update(norm(n) for n in raw_names)
    out, missed = match_names(df, raw_names)
    if missed:
        _big_warning([f"confirmed_not_starting: NO MATCH for {len(missed)} name(s): "
                      + ", ".join(missed),
                      "These players were NOT removed — check the spelling against the data."])
    keep = ~df["Player Name"].isin(out)
    if (~keep).any():
        print(f"  confirmed NOT starting — removed {int((~keep).sum())}: {', '.join(sorted(out))}")
    return df[keep].reset_index(drop=True)


def confirm_starting(df, raw_names):
    """Mark players confirmed IN the XI: set start probability to 1.0 in place and lift their
    projection to the full-match value. Only the pre-bonus points scale with start
    (F1 XP Pre = start x full, exact), so we divide that by the old start and keep the bonus;
    downstream eff_xp then values them as nailed. Matched the usual way; unmatched reported."""
    if not raw_names:
        return df
    conf, missed = match_names(df, raw_names)
    if missed:
        _big_warning([f"confirmed_starting: NO MATCH for {len(missed)} name(s): "
                      + ", ".join(missed),
                      "These players were NOT forced to start — check the spelling against the data."])
    m = df["Player Name"].isin(conf)
    if not m.any():
        return df
    start = df.loc[m, "F1 Start"].fillna(0.0)
    tiny = start < 0.01
    if tiny.any():
        print(f"  ! confirmed_starting: no minutes projection to scale for "
              f"{', '.join(sorted(df.loc[m & (df['F1 Start'].fillna(0) < 0.01), 'Player Name']))}"
              f" — start set to 1.0 but XP left as-is")
    denom = start.clip(lower=1e-6)
    if "F1 XP Pre" in df.columns:
        xppre = df.loc[m, "F1 XP Pre"].fillna(0.0)
        xppre_full = xppre / denom
        df.loc[m, XP_COL] = xppre_full + (df.loc[m, XP_COL] - xppre)
        df.loc[m, "F1 XP Pre"] = xppre_full
    else:
        df.loc[m, XP_COL] = df.loc[m, XP_COL] / denom
    df.loc[m, "F1 Start"] = 1.0
    _CONFIRMED_STARTERS.update(conf)
    print(f"  confirmed starting (start -> 1.0): {', '.join(sorted(conf))}")
    return df


def _series(df, col):
    return df[col].fillna(0.0) if col in df.columns else pd.Series(0.0, index=df.index)


def expected_goals(df):
    """Expected goals per player this week (start-scaled), from the goalscorer-odds tail
    P(1+)+P(2+)+P(3+). Used by GW6 to split each team's goals into per-player shares."""
    return _series(df, "F1 Start") * (
        _series(df, "F1 Score 1+") + _series(df, "F1 Score 2+") + _series(df, "F1 Score 3+"))


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


def solve_formation(df, shape, max_per_club, force_in=(), force_cap=None, cap_allowed=None):
    """Exact ILP for one (DEF, MID, FWD) shape under the club cap. force_in: indices that
    must be picked (players already locked in); force_cap: index that must be captain.
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
    for i in force_in:
        prob += pick[i] == 1
    if _LIVE["now"] is not None:
        for i in idx:
            nm = df.loc[i, "Player Name"]
            if nm in _LIVE["locked"]:
                prob += pick[i] == 1                       # already played, in your team
            elif nm in _LIVE["started"]:
                prob += pick[i] == 0                       # already played, can't be added
            if nm in _LIVE["started"] and nm != _LIVE["cap"]:
                prob += capt[i] == 0                       # armband can't move onto a played match
            if _LIVE["cap"] is not None and nm == _LIVE["cap"]:
                prob += capt[i] == 1
    if force_cap is not None:
        prob += capt[force_cap] == 1
    if cap_allowed is not None:          # armband can only go to players whose match hasn't started
        for i in idx:
            if i not in cap_allowed:
                prob += capt[i] == 0

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if prob.status != pulp.LpStatusOptimal:
        return None
    chosen = [i for i in idx if pick[i].varValue > 0.5]
    cap_i = next(i for i in idx if capt[i].varValue > 0.5)
    xi = df.loc[chosen].copy()
    total = xi["eff_xp"].sum() + df.loc[cap_i, "cap_bonus"]
    return total, xi, df.loc[cap_i]


def best_over_formations(df, max_per_club, force_in=(), force_cap=None, cap_allowed=None):
    """Best (total, xi, cap) across all six formations, or None."""
    best = None
    for shape in FORMATIONS.values():
        r = solve_formation(df, shape, max_per_club, force_in, force_cap, cap_allowed)
        if r and (best is None or r[0] > best[0]):
            best = r
    return best


def highlight_if_start(df, base_total, base_xi, max_per_club, boost_tag, base_cap=None,
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

    # Lock the picks you couldn't still change when this candidate's lineup news lands (their
    # match kicks off before then), so the re-pick only drops players it actually could — and
    # ties can't send it to an earlier, already-locked player.
    kos = load_kickoffs()
    lead_td = pd.Timedelta(minutes=LINEUP_LEAD_MIN)
    picks_idx = list(base_xi.index)
    base_cap_i = next((j for j in picks_idx if df.loc[j, "Player Name"] == base_cap), None)

    hits, hidden = [], 0
    for i in cand:
        tmp = df.copy()
        tmp[["eff_xp", "cap_bonus"]] = tmp[["eff_xp", "cap_bonus"]].astype(float)
        tmp.loc[i, "eff_xp"] = cond_eff[i]
        tmp.loc[i, "cap_bonus"] = cond_cap[i]
        tmp.loc[tmp.index.isin(picks_idx), "eff_xp"] += 1e-3      # prefer keeping current picks on ties
        ki = kos.get(df.loc[i, "Team"])
        locked = ([j for j in picks_idx if kos.get(df.loc[j, "Team"]) is not None
                   and kos[df.loc[j, "Team"]] <= ki - lead_td] if ki is not None else [])
        force_cap = base_cap_i if base_cap_i in locked else None
        best = best_over_formations(tmp, max_per_club, force_in=locked, force_cap=force_cap)
        if not best:
            continue
        true_total = df.loc[best[1].index, "eff_xp"].sum() + df.loc[best[2].name, "cap_bonus"]
        true_total += (cond_eff[i] - df.loc[i, "eff_xp"]) + (
            (cond_cap[i] - df.loc[i, "cap_bonus"]) if best[2].name == i else 0.0)
        if true_total <= base_total + 1e-6:
            continue
        new_names = set(best[1]["Player Name"])
        if df.loc[i, "Player Name"] not in new_names:
            continue
        gain = true_total - base_total
        if gain < min_gain:
            hidden += 1
            continue
        hits.append((i, cond_eff[i], gain, base_names - new_names, best[2]["Player Name"]))

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

    # one line per swap, in kickoff order — same style as IF A PICK DOESN'T START
    rows, too_late = [], 0
    for i, ce, gain, dropped, new_cap in hits:
        club = df.loc[i, "Team"]
        st = swap_status(club, dropped)
        if st == "TOO LATE":
            too_late += 1
            continue
        rows.append((kickoffs.get(club, pd.Timestamp.max), -gain, i, gain, dropped, new_cap, st))
    footer = [f"{n} {label}" for n, label in
              ((hidden, f"below +{min_gain:.1f}"), (too_late, "too late to swap")) if n]
    print("\nBRING IN IF THEY START:")
    if not rows:
        print("none actionable")
    for t, _, i, gain, dropped, new_cap, st in sorted(rows, key=lambda r: (r[0], r[1])):
        r = df.loc[i]
        when = (t + DISPLAY_OFFSET).strftime("%a %H:%M") if t is not pd.Timestamp.max else "?"
        parts = [f"OUT {', '.join(sorted(dropped))}" if dropped else "IN (reshuffle)"]
        if base_cap is not None and new_cap != base_cap:
            parts.append(f"captain -> {new_cap}")
        timing = " (timing unknown)" if st == "timing?" else ""
        print(f"{when} {r['Player Name']} ({r['Team']}, start {r['F1 Start'] * 100:.0f}%): "
              f"{' | '.join(parts)}  [+{gain:.2f}]{timing}")
    if footer:
        print(f"({', '.join(footer)} — hidden)")


def contingency_plan(df, base_total, xi, cap, max_per_club, boost_tag):
    """'IF A PICK DOESN'T START' — for each selected player, in kickoff order, the best change if
    their lineup drops and they're out. At that moment anyone whose match kicks off EARLIER is
    already locked in (and keeps the armband if they had it); only players whose match is at
    that kickoff time or later can come in. Replacements are valued at their normal (weighted)
    XP, since later lineups aren't known yet. Formation may change if the locked players fit.
    """
    kickoffs = load_kickoffs()
    if not kickoffs:
        print("\nIF A PICK DOESN'T START — no kickoff data (fixture_kickoffs.csv), skipped")
        return
    far = pd.Timestamp.max
    ko = df["Team"].map(lambda t: kickoffs.get(t, far))
    picks = list(xi.index)
    order = sorted(picks, key=lambda i: (ko[i], df.loc[i, "Player Name"]))
    cap_i = cap.name

    def fmt(t):
        return (t + DISPLAY_OFFSET).strftime("%a %H:%M") if t is not far else "?"

    print("\nIF A PICK DOESN'T START:")
    skipped = [df.loc[i, "Player Name"] for i in order if df.loc[i, "Player Name"] in _CONFIRMED_STARTERS]
    for p in order:
        if df.loc[p, "Player Name"] in _CONFIRMED_STARTERS:
            continue   # confirmed starting — no "if he's out" case to plan for
        if df.loc[p, "Player Name"] in _LIVE["started"]:
            continue   # already played — nothing left to plan
        t = ko[p]
        locked = [i for i in picks if i != p and ko[i] < t]
        pool = df.index[(df.index != p) & ((ko >= t) | df.index.isin(locked))]
        # tiny bonus for keeping current picks / captain, so ties don't trigger needless churn
        sub = df.loc[pool].copy()
        sub[["eff_xp", "cap_bonus"]] = sub[["eff_xp", "cap_bonus"]].astype(float)
        sub.loc[sub.index.isin(picks), "eff_xp"] += 1e-3
        if cap_i in sub.index:
            sub.loc[cap_i, "cap_bonus"] += 1e-3
        # a locked captain keeps the armband; otherwise it can only move to an unlocked player
        force_cap = cap_i if (cap_i in locked) else None
        cap_allowed = None if force_cap is not None else set(sub.index[ko[sub.index] >= t])
        best = best_over_formations(sub, max_per_club, force_in=locked, force_cap=force_cap,
                                    cap_allowed=cap_allowed)
        name = df.loc[p, "Player Name"]
        head = f"{fmt(t)} {name} ({df.loc[p, 'Team']})"
        if not best:
            print(f"{head}: no legal team — keep him")
            continue
        _, new_xi, new_cap = best
        new_names = set(new_xi["Player Name"])
        out_ = [df.loc[i, "Player Name"] for i in picks
                if i != p and df.loc[i, "Player Name"] not in new_names]
        in_ = [r["Player Name"] for _, r in new_xi.iterrows() if r.name not in picks]
        parts = ([f"OUT {', '.join(out_)}"] if out_ else []) + [f"IN {', '.join(in_) or '-'}"]
        if new_cap["Player Name"] != cap["Player Name"]:
            parts.append(f"captain -> {new_cap['Player Name']}")
        print(f"{head}: {' | '.join(parts)}")
    if skipped:
        print(f"(confirmed starting, no plan needed: {', '.join(skipped)})")



def solve_and_report(df, max_per_club, title_lines, boost_tag, show_watchlist=True, show_contingency=True):
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
        st = row.get("F1 Start", float("nan"))
        st_txt = f"start {st * 100:3.0f}%" if pd.notna(st) else "start   ?"
        flag = "  << NOT NAILED - check lineup" if (pd.isna(st) or st < 0.999) else ""
        print(f"    {row['Position']:<4}{row['Player Name']:<24}{row['Team']:<15}{st_txt}  "
              f"xp {row[XP_COL]:4.2f} -> {row['eff_xp']:5.2f}{tag}{star}{flag}")
    counts = cap["eff_xp"] + cap["cap_bonus"]
    print(f"\n  Captain: {cap['Player Name']}  "
          f"(base xp {cap[XP_COL]:.2f}, effective {cap['eff_xp']:.2f} -> counts {counts:.2f})")

    print(f"\nTOP BOOSTED PLAYERS BY EFFECTIVE XP (what drives the pick)")
    top = df[df["boosted"]].sort_values("eff_xp", ascending=False).head(12)
    for _, r in top.iterrows():
        print(f"    {r['Position']:<4}{r['Player Name']:<24}{r['Team']:<15}"
              f"xp {r[XP_COL]:4.2f} -> {r['eff_xp']:5.2f}")

    _archive_reminder()
    if show_watchlist:
        highlight_if_start(df, total, xi, max_per_club, boost_tag, base_cap=cap["Player Name"])
    if show_contingency:
        contingency_plan(df, total, xi, cap, max_per_club, boost_tag)
    _archive_reminder()
    return results
