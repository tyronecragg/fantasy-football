"""Rolling odds-implied team strength -> F2 (fallback) / F3-F8 win probabilities.

Replaces the season-outrights Lasso (model.win_pred / opp_win_pred) as the fixture-strength source
in improved mode (config.USE_ROLLING_RATING). Tested 2026-09-02 (tools/rolling_rating.py,
tools/rating_backtest.py): match-odds MAE 0.045 pooled over 1-8 weeks ahead vs 0.052 for the
outrights model and 0.066 for a static prior-season rating; 0.025 vs 0.035 on live GW4 prices.

HOW
  1. Closing 1X2 prices for every PL match (football-data.co.uk, cached in fpl_data/football_data,
     the current season refreshed when the cache is older than a day) are de-margined.
  2. Each match is one equation  log(pH/pA) = r_home - r_away + home_advantage.  All 20 strengths
     and the home advantage are solved by weighted least squares over every match played BEFORE the
     cutoff, weights halving every RATING_HALFLIFE_DAYS. Promoted sides get a pseudo-match at the
     prior season's three-lowest level (weight RATING_PROMOTED_WEIGHT). The current league's
     strengths are anchored to sum to zero.
  3. The rating gap (home minus away) maps to P(home, draw, away) through a per-horizon multinomial
     logistic fitted on past seasons' closing prices (fpl_pipeline/data/rating_maps.json, written by
     tools/fit_rating_maps.py) - far horizons get a flatter slope, so distant fixtures regress toward
     even by a fitted amount rather than by assumption.

Serving: `load()` -> FixtureRating; `win_probs(team, opponent, home, weeks_ahead)` returns the
team's win and its opponent's win probabilities for each row, symmetric by construction (both rows
of a fixture see the same match), so no unify/reconcile step is needed.
"""
import datetime
import io
import json
import os

import numpy as np
import pandas as pd

from . import config, names

FOOTBALL_DATA_DIR = os.path.join(config.ROOT, "fpl_data", "football_data")
URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
# football-data's team spellings that differ from ours; everything else matches names.apply_team_names
FOOTBALL_DATA_NAMES = {
    "Man United": "Man Utd", "Tottenham": "Spurs", "Sheffield United": "Sheffield Utd",
    "Ipswich": "Ipswich Town", "Coventry": "Coventry City", "Hull": "Hull City",
    "Nott'm Forest": "Nott'm Forest", "Newcastle": "Newcastle", "Leeds": "Leeds",
}
RIDGE = 1e-3
HORIZONS = range(1, 9)


def season_code(season):
    """'2026-2027' -> '2627' (football-data's season key)."""
    a, b = season.split("-")
    return a[2:] + b[2:]


def season_label(code):
    return f"20{code[:2]}-20{code[2:]}"


def _read_football_data(text):
    """Parse a football-data E0 CSV, repairing stray in-field line breaks first. Their served file
    occasionally injects a \\r\\n inside a cell (seen 2026-09: 'Brentford,To\\r\\nttenham,3,...'),
    splitting one match across two physical lines so a stat lands in the Date column and the whole
    parse fails. Every real data row starts with the division code 'E0,'; any other line is a
    continuation, so glue it back onto the previous one (no separator — the break was mid-field)."""
    lines = text.splitlines()
    if not lines:
        return pd.read_csv(io.StringIO(text))
    out = [lines[0]]                                    # header (Div,Date,Time,...)
    for ln in lines[1:]:
        if ln.startswith("E0,") or not out:
            out.append(ln)
        else:
            out[-1] += ln                              # rejoin a mid-field newline
    return pd.read_csv(io.StringIO("\n".join(out)))


def fetch_season(code, refresh=False, max_age_days=None):
    """Season CSV from the cache; (re)downloaded when missing, when `refresh`, or when the cached
    file is older than `max_age_days`. A failed download falls back to the cache."""
    os.makedirs(FOOTBALL_DATA_DIR, exist_ok=True)
    path = os.path.join(FOOTBALL_DATA_DIR, f"E0_{code}.csv")
    stale = (max_age_days is not None and os.path.exists(path)
             and (datetime.datetime.now() - datetime.datetime.fromtimestamp(os.path.getmtime(path))).days >= max_age_days)
    if os.path.exists(path) and not refresh and not stale:
        with open(path, encoding="utf-8-sig") as fh:
            return _read_football_data(fh.read())
    try:
        import requests
        r = requests.get(URL.format(season=code), timeout=30)
        r.raise_for_status()
        d = _read_football_data(r.content.decode("utf-8-sig"))
        d.to_csv(path, index=False)                     # cache the repaired version
        print(f"  rating: downloaded football-data {season_label(code)} ({len(d)} matches)")
        return d
    except Exception as exc:  # network / parse - keep serving from the cache
        if os.path.exists(path):
            print(f"  rating: football-data {season_label(code)} refresh failed ({exc}); using cache")
            with open(path, encoding="utf-8-sig") as fh:
                return _read_football_data(fh.read())
        raise


def _poisson_pmf(lam, n=11):
    """(rows x n) Poisson pmf for 0..n-1 goals, vectorised over rates."""
    lam = np.asarray(lam, dtype=float)[:, None]
    k = np.arange(n)[None, :]
    from scipy.special import gammaln
    return np.exp(-lam + k * np.log(np.maximum(lam, 1e-9)) - gammaln(k + 1))


def implied_goals(p_home, p_away, p_over):
    """Each side's expected goals implied by the market: total from the over-2.5 price (Poisson
    tail), the split from the home/away win gap (Skellam via the score matrix). Vectorised bisection."""
    p_home, p_away, p_over = (np.asarray(x, dtype=float) for x in (p_home, p_away, p_over))
    lo, hi = np.full(len(p_over), 0.2), np.full(len(p_over), 8.0)
    for _ in range(40):                                        # lambda_total: P(N >= 3) = p_over
        mid = (lo + hi) / 2
        tail = 1 - np.exp(-mid) * (1 + mid + mid ** 2 / 2)
        lo, hi = np.where(tail < p_over, mid, lo), np.where(tail < p_over, hi, mid)
    tot = (lo + hi) / 2
    target = p_home - p_away
    lo, hi = -tot + 0.05, tot - 0.05                            # d = lam_home - lam_away
    for _ in range(40):
        mid = (lo + hi) / 2
        lh, la = (tot + mid) / 2, (tot - mid) / 2
        ph, pa = _poisson_pmf(lh), _poisson_pmf(la)
        cum_a = np.cumsum(pa, axis=1)                          # P(away <= j)
        p_h = (ph[:, 1:] * cum_a[:, :-1]).sum(axis=1)          # home scores more
        cum_h = np.cumsum(ph, axis=1)
        p_a = (pa[:, 1:] * cum_h[:, :-1]).sum(axis=1)          # away scores more
        gap = p_h - p_a
        lo, hi = np.where(gap < target, mid, lo), np.where(gap < target, hi, mid)
    d = (lo + hi) / 2
    return (tot + d) / 2, (tot - d) / 2


def match_frame(raw, season):
    """De-margined closing 1X2 (+ over/under 2.5) with canonical team names, dated, plus the
    log-odds target and each side's market-implied expected goals."""
    cols = ["Date", "HomeTeam", "AwayTeam", "AvgCH", "AvgCD", "AvgCA"]
    d = raw[cols].dropna().copy()
    inv = 1 / d["AvgCH"] + 1 / d["AvgCD"] + 1 / d["AvgCA"]
    out = pd.DataFrame({
        "season": season,
        "date": pd.to_datetime(d["Date"], dayfirst=True),
        "HomeTeam": names.apply_team_names(d["HomeTeam"].replace(FOOTBALL_DATA_NAMES)),
        "AwayTeam": names.apply_team_names(d["AwayTeam"].replace(FOOTBALL_DATA_NAMES)),
        "p_home": (1 / d["AvgCH"]) / inv,
        "p_draw": (1 / d["AvgCD"]) / inv,
        "p_away": (1 / d["AvgCA"]) / inv,
    })
    out["y"] = np.log(out["p_home"] / out["p_away"])
    over = raw.get("AvgC>2.5", raw.get("Avg>2.5"))
    under = raw.get("AvgC<2.5", raw.get("Avg<2.5"))
    if over is not None and under is not None:
        o, u = pd.to_numeric(over.loc[d.index], errors="coerce"), pd.to_numeric(under.loc[d.index], errors="coerce")
        p_over = ((1 / o) / (1 / o + 1 / u)).fillna(0.52).values
    else:
        p_over = np.full(len(d), 0.52)
    lh, la = implied_goals(out["p_home"].values, out["p_away"].values, p_over)
    out["lam_home"], out["lam_away"] = lh, la
    return out


def load_matches(codes, refresh_current=None):
    """All cached seasons (season codes like '2526'), sorted by date. refresh_current: a season
    code whose file is re-downloaded when older than a day (the live season)."""
    frames = []
    for c in codes:
        raw = fetch_season(c, max_age_days=1 if c == refresh_current else None)
        frames.append(match_frame(raw, season_label(c)))
    m = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    m["season_idx"] = m["season"].map({s: i for i, s in enumerate(sorted(m["season"].unique()))})
    return m


def solve(g, teams, w, pseudo=None, anchor=None):
    """Weighted least squares for strengths r (anchored: `anchor` teams sum to zero) and the home
    advantage. pseudo: {team: (level, weight)} pseudo-observations for sides with no prices."""
    pseudo = pseudo or {}
    idx = {t: i for i, t in enumerate(teams)}
    n, T = len(g), len(teams)
    X = np.zeros((n + len(pseudo) + 1, T + 1))
    y = np.zeros(n + len(pseudo) + 1)
    ww = np.zeros(n + len(pseudo) + 1)
    X[np.arange(n), g["HomeTeam"].map(idx).values] = 1.0
    X[np.arange(n), g["AwayTeam"].map(idx).values] = -1.0
    X[:n, T] = 1.0
    y[:n], ww[:n] = g["y"].values, w
    for j, (t, (level, wt)) in enumerate(pseudo.items()):
        X[n + j, idx[t]], y[n + j], ww[n + j] = 1.0, level, wt
    for t in (anchor or teams):
        X[-1, idx[t]] = 1.0
    y[-1], ww[-1] = 0.0, 1e3
    sw = np.sqrt(ww)[:, None]
    A = np.vstack([X * sw, np.sqrt(RIDGE) * np.eye(T + 1)])
    b = np.concatenate([y * sw[:, 0], np.zeros(T + 1)])
    beta = np.linalg.lstsq(A, b, rcond=None)[0]
    return pd.Series(beta[:T], index=teams), float(beta[T])


def solve_goals(g, teams, w, pseudo=None, anchor=None):
    """Attack / defence strengths by weighted least squares on the market-implied goals:
        log lam_home = att_home - def_away + c_home
        log lam_away = att_away - def_home + c_away
    Returns (att, dfn, c_home, c_away); `anchor` teams' att and def each sum to zero.
    pseudo: {team: (att_level, def_level, weight)} for sides with no prices."""
    pseudo = pseudo or {}
    idx = {t: i for i, t in enumerate(teams)}
    n, T = len(g), len(teams)
    P = len(pseudo)
    X = np.zeros((2 * n + 2 * P + 2, 2 * T + 2))
    y = np.zeros(2 * n + 2 * P + 2)
    ww = np.zeros(2 * n + 2 * P + 2)
    h, a = g["HomeTeam"].map(idx).values, g["AwayTeam"].map(idx).values
    r = np.arange(n)
    X[r, h], X[r, T + a], X[r, 2 * T] = 1.0, -1.0, 1.0                      # home goals
    X[n + r, a], X[n + r, T + h], X[n + r, 2 * T + 1] = 1.0, -1.0, 1.0      # away goals
    y[:n], y[n:2 * n] = np.log(g["lam_home"].values), np.log(g["lam_away"].values)
    ww[:n] = ww[n:2 * n] = w
    for j, (t, (al, dl, wt)) in enumerate(pseudo.items()):
        X[2 * n + j, idx[t]], y[2 * n + j], ww[2 * n + j] = 1.0, al, wt
        X[2 * n + P + j, T + idx[t]], y[2 * n + P + j], ww[2 * n + P + j] = 1.0, dl, wt
    for t in (anchor or teams):
        X[-2, idx[t]] = 1.0
        X[-1, T + idx[t]] = 1.0
    ww[-2] = ww[-1] = 1e3
    sw = np.sqrt(ww)[:, None]
    A = np.vstack([X * sw, np.sqrt(RIDGE) * np.eye(2 * T + 2)])
    b = np.concatenate([y * sw[:, 0], np.zeros(2 * T + 2)])
    beta = np.linalg.lstsq(A, b, rcond=None)[0]
    return (pd.Series(beta[:T], index=teams), pd.Series(beta[T:2 * T], index=teams),
            float(beta[2 * T]), float(beta[2 * T + 1]))


def season_teams(m):
    return {s: sorted(set(g["HomeTeam"]) | set(g["AwayTeam"])) for s, g in m.groupby("season")}


def promoted_level(m, prev_season):
    """Level a newly promoted side starts at: mean of the prior season's three lowest full-season
    strengths (validated substitution, tools/build_team_ratings.py)."""
    g = m[m["season"] == prev_season]
    teams = sorted(set(g["HomeTeam"]) | set(g["AwayTeam"]))
    r, _ = solve(g, teams, np.ones(len(g)))
    return float(r.nsmallest(3).mean())


def goals_at(m, cutoff, season, halflife, w0, teams_by_season=None):
    """Attack / defence strengths for `season`'s teams from matches before `cutoff`, same weighting
    as ratings_at. Promoted sides start at the mean att/def of the prior season's three lowest-rated
    teams. Returns (att, dfn, c_home, c_away) or None."""
    teams_by_season = teams_by_season or season_teams(m)
    seasons = sorted(teams_by_season)
    if season not in seasons:
        seasons.append(season)
    si = seasons.index(season)
    prev = seasons[si - 1] if si else None
    g = m[m["date"] < cutoff]
    age = (cutoff - g["date"]).dt.days.values
    w = 0.5 ** (age / halflife)
    keep = w > 1e-3
    g, w = g[keep], w[keep]
    if len(g) < 20:
        return None
    cur_teams = teams_by_season.get(season) or _current_teams()
    teams = sorted(set(cur_teams) | set(g["HomeTeam"]) | set(g["AwayTeam"]))
    pseudo = {}
    if prev and prev in teams_by_season:
        gp = m[m["season"] == prev]
        pt = sorted(set(gp["HomeTeam"]) | set(gp["AwayTeam"]))
        r, _ = solve(gp, pt, np.ones(len(gp)))
        att, dfn, _, _ = solve_goals(gp, pt, np.ones(len(gp)))
        low = r.nsmallest(3).index
        for t in cur_teams:
            if t not in teams_by_season[prev]:
                pseudo[t] = (float(att[low].mean()), float(dfn[low].mean()), w0)
    return solve_goals(g, teams, w, pseudo, anchor=cur_teams)


def ratings_at(m, cutoff, season, halflife, w0, teams_by_season=None):
    """Strengths for `season`'s teams from matches before `cutoff`. Returns (ratings, hfa, weight)
    or None when fewer than 20 matches carry weight."""
    teams_by_season = teams_by_season or season_teams(m)
    seasons = sorted(teams_by_season)
    if season not in seasons:            # a season with no matches yet: rate from the prior one
        seasons.append(season)
    si = seasons.index(season)
    prev = seasons[si - 1] if si else None
    g = m[m["date"] < cutoff]
    age = (cutoff - g["date"]).dt.days.values
    w = 0.5 ** (age / halflife)
    keep = w > 1e-3
    g, w = g[keep], w[keep]
    cur_teams = teams_by_season.get(season) or _current_teams()
    teams = sorted(set(cur_teams) | set(g["HomeTeam"]) | set(g["AwayTeam"]))
    pseudo = {}
    if prev and prev in teams_by_season:
        level = promoted_level(m, prev)
        for t in cur_teams:
            if t not in teams_by_season[prev]:
                pseudo[t] = (level, w0)
    if len(g) < 20:
        return None
    r, hfa = solve(g, teams, w, pseudo, anchor=cur_teams)
    return r, hfa, float(w.sum())


def _current_teams():
    """This season's 20 teams from inputs/season_fixtures.csv (before any match is played)."""
    f = pd.read_csv(os.path.join(config.INPUTS_DIR, "season_fixtures.csv"))
    return sorted(set(f["home_team"]) | set(f["away_team"]))


def features(diff):
    d = np.asarray(diff, dtype=float)
    return np.column_stack([d, np.abs(d), d ** 2])


class RatingMaps:
    """Per-horizon multinomial logistic: rating gap -> (P home, P draw, P away)."""

    def __init__(self, doc):
        self.doc = doc
        self.h = {int(k): (np.array(v["coef"]), np.array(v["intercept"]))
                  for k, v in doc["horizons"].items()}

    @classmethod
    def load(cls, path=None):
        path = path or config.RATING_MAPS_JSON
        with open(path, encoding="utf-8") as fh:
            return cls(json.load(fh))

    def predict(self, diff, weeks_ahead):
        k = min(max(int(weeks_ahead), min(self.h)), max(self.h))
        coef, b = self.h[k]
        z = features(diff) @ coef.T + b
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)          # columns: home, draw, away

    def goals(self, log_lam, weeks_ahead):
        """Per-horizon shrink of a predicted log expected-goals toward the mean (fitted a + b*x);
        identity when the maps carry no goals section."""
        gm = self.doc.get("goals")
        if not gm:
            return np.exp(np.asarray(log_lam, dtype=float))
        k = str(min(max(int(weeks_ahead), 1), max(int(x) for x in gm)))
        a, b = gm[k]["a"], gm[k]["b"]
        return np.exp(a + b * np.asarray(log_lam, dtype=float))


class FixtureRating:
    def __init__(self, ratings, hfa, weight, maps, cutoff, season, goals=None):
        self.ratings, self.hfa, self.weight, self.maps = ratings, hfa, weight, maps
        self.cutoff, self.season = cutoff, season
        self.goals = goals                                    # (att, dfn, c_home, c_away) or None

    def expected_goals(self, team, opponent, home, weeks_ahead):
        """Per row: (expected goals FOR the team, AGAINST it) for a fixture `weeks_ahead` weeks
        out, from the attack/defence split; NaN where either side is unrated or no split exists."""
        idx = team.index if hasattr(team, "index") else None
        if self.goals is None:
            return pd.Series(np.nan, index=idx), pd.Series(np.nan, index=idx)
        att, dfn, c_h, c_a = self.goals
        home = pd.Series(home).astype(bool).values
        t_att, t_def = pd.Series(team).map(att).values, pd.Series(team).map(dfn).values
        o_att, o_def = pd.Series(opponent).map(att).values, pd.Series(opponent).map(dfn).values
        log_for = np.where(home, t_att - o_def + c_h, t_att - o_def + c_a)
        log_against = np.where(home, o_att - t_def + c_a, o_att - t_def + c_h)
        ok = np.isfinite(log_for) & np.isfinite(log_against)
        lf = self.maps.goals(np.nan_to_num(log_for, nan=0.0), weeks_ahead)
        la = self.maps.goals(np.nan_to_num(log_against, nan=0.0), weeks_ahead)
        return (pd.Series(np.where(ok, lf, np.nan), index=idx),
                pd.Series(np.where(ok, la, np.nan), index=idx))

    def win_probs(self, team, opponent, home, weeks_ahead):
        """Per row: (P(team wins), P(opponent wins)) for a fixture `weeks_ahead` weeks out.
        NaN where either side has no rating (caller falls back to the incumbent)."""
        own = pd.Series(team).map(self.ratings)
        opp = pd.Series(opponent).map(self.ratings)
        home = pd.Series(home).astype(bool).values
        diff = np.where(home, own - opp, opp - own)
        ok = np.isfinite(diff)
        p = self.maps.predict(np.nan_to_num(diff, nan=0.0), weeks_ahead)
        win = np.where(home, p[:, 0], p[:, 2])
        lose = np.where(home, p[:, 2], p[:, 0])
        idx = team.index if hasattr(team, "index") else None
        return (pd.Series(np.where(ok, win, np.nan), index=idx),
                pd.Series(np.where(ok, lose, np.nan), index=idx))

    def summary(self):
        top = self.ratings.reindex(_current_teams()).dropna().sort_values(ascending=False)
        head = ", ".join(f"{t} {v:+.2f}" for t, v in top.head(3).items())
        tail = ", ".join(f"{t} {v:+.2f}" for t, v in top.tail(2).items())
        return (f"rolling rating @ {self.cutoff.date()}: {self.weight:.0f} weighted matches, "
                f"home adv {self.hfa:.2f} | {head} ... {tail}")


class RatingSource:
    """Every cached football-data season + the maps, loaded once; a FixtureRating at ANY cutoff.
    Training-time features (tools/build_training_data via tools/rating_backtest.rating_win_fn) and
    live serving (players.build / projection_serving) both go through this class, so the GBM
    features `rating_win` / `rating_opp` are built by one code path and cannot skew."""

    def __init__(self, season=None, refresh=True):
        self.season = season or config.SEASON
        self.maps = RatingMaps.load()
        cur = season_code(self.season)
        codes = sorted({f[3:7] for f in os.listdir(FOOTBALL_DATA_DIR)
                        if f.startswith("E0_") and f.endswith(".csv")} | {cur})
        self.m = load_matches(codes, refresh_current=cur if refresh else None)
        self.teams = season_teams(self.m)
        self._cache = {}

    def at(self, cutoff, season=None):
        season = season or self.season
        cutoff = pd.Timestamp(cutoff)
        key = (cutoff, season)
        if key not in self._cache:
            snap = ratings_at(self.m, cutoff, season, config.RATING_HALFLIFE_DAYS,
                              config.RATING_PROMOTED_WEIGHT, self.teams)
            goals = None if snap is None else goals_at(self.m, cutoff, season, config.RATING_HALFLIFE_DAYS,
                                                       config.RATING_PROMOTED_WEIGHT, self.teams)
            self._cache[key] = (None if snap is None
                                else FixtureRating(snap[0], snap[1], snap[2], self.maps, cutoff, season, goals))
        return self._cache[key]


def current_f1_gw(path=None):
    """The gameweek inputs/fixtures.csv currently starts at ('GW7 Opponent' -> 7), else None."""
    try:
        head = str(pd.read_csv(path or os.path.join(config.INPUTS_DIR, "fixtures.csv"), nrows=0).columns[1])
    except (OSError, IndexError, pd.errors.EmptyDataError):
        return None
    digits = "".join(ch for ch in head.upper().replace("GW", "", 1).split()[0] if ch.isdigit())
    return int(digits) if digits else None


def load_team_views(gameweek, path=None):
    """{team: nudge in goals per game} for `gameweek` only (rows for any other GW are ignored, so a
    stale file has no effect). Missing/empty file or no matching rows -> {}."""
    path = path or config.TEAM_VIEWS_CSV
    if gameweek is None or not os.path.exists(path):
        return {}
    v = pd.read_csv(path, comment="#")
    if v.empty:
        return {}
    v = v[pd.to_numeric(v["Gameweek"], errors="coerce") == gameweek]
    out = {}
    for team, n in zip(names.apply_team_names(v["Team"].astype(str).str.strip()), pd.to_numeric(v["nudge"])):
        if not np.isfinite(n):
            continue
        out[team] = float(np.clip(n, -config.TEAM_VIEW_MAX, config.TEAM_VIEW_MAX))
    return out


def apply_team_views(fr, views):
    """A copy of FixtureRating `fr` with each team nudged by `views` {team: goals per game}:
    the 1X2 strength moves by nudge x TEAM_VIEW_LOGODDS_PER_GOAL; the goals split moves half to
    attack, half to defence (log rate of +nudge/2 goals on a TEAM_VIEW_AVG_GOALS base each way).
    Unknown teams are returned in `skipped`. Returns (new_fr, applied, skipped)."""
    if fr is None or not views:
        return fr, {}, []
    known = set(fr.ratings.index)
    applied = {t: n for t, n in views.items() if t in known}
    skipped = sorted(set(views) - known)
    r = fr.ratings.copy()
    for t, n in applied.items():
        r[t] += n * config.TEAM_VIEW_LOGODDS_PER_GOAL
    goals = fr.goals
    if goals is not None:
        att, dfn, c_h, c_a = goals
        att, dfn = att.copy(), dfn.copy()
        for t, n in applied.items():
            step = np.log1p(abs(n) / 2 / config.TEAM_VIEW_AVG_GOALS) * np.sign(n)
            if t in att.index:
                att[t] += step
            if t in dfn.index:
                dfn[t] += step            # higher def = fewer goals conceded (log_against = o_att - t_def + c)
        goals = (att, dfn, c_h, c_a)
    return FixtureRating(r, fr.hfa, fr.weight, fr.maps, fr.cutoff, fr.season, goals), applied, skipped


def load(season=None, cutoff=None):
    """The live FixtureRating for `season` (default config.SEASON) as of `cutoff` (default now),
    or None (with a message) when the maps or price data are unavailable."""
    cutoff = pd.Timestamp(cutoff or datetime.datetime.now())
    try:
        src = RatingSource(season)
    except OSError as exc:
        print(f"  rating: maps or price data unavailable ({exc}) - falling back to the outrights model")
        return None
    except Exception as exc:
        print(f"  rating: price data unavailable ({exc}) - falling back to the outrights model")
        return None
    fr = src.at(cutoff)
    if fr is None:
        print("  rating: too few priced matches - falling back to the outrights model")
    return fr
