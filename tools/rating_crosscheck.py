"""Cross-check the pipeline's F1-F10 win probabilities against odds-implied team ratings.

    python tools/rating_crosscheck.py [--gw 1]

Read-only. Changes nothing, adopts nothing — this is a second opinion on the projections
the squad decision rests on, from a model built on completely different data (5 seasons of
football-data closing prices rather than this season's outright markets).

Two things it reports, because they answer different questions:

  * UNWEIGHTED per-fixture disagreement — is the projection wrong? Every fixture counts
    equally, because model error does not care how much we act on it.
  * WEIGHTED summary — does it matter? Scaled by the optimiser's own fixture weights
    (ownership x reliability), since a big disagreement at F9 barely moves a squad.

F1 is special and worth reading first: those are REAL market odds, so the gap there is a
direct measure of the ratings model's error in the August regime, this season, for these
squads. F3+ is model versus model, with no ground truth either way.

F9-F10 have no pipeline counterpart (the window stops at F8), so they are reported as
ratings-only — new information rather than a comparison.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import config  # noqa: E402
from tools.build_team_ratings import DEFAULT_SEASONS, fetch, match_probs, rate  # noqa: E402

RATINGS = os.path.join(config.INPUTS_DIR, "team_ratings.csv")
FIXTURES = os.path.join(config.INPUTS_DIR, "season_fixtures.csv")
MASTER = os.path.join(config.OUTPUTS_DIR, "13_players_master.csv")


def fit_rating_model(seasons=DEFAULT_SEASONS):
    """Fit win probability on (own rating, opponent rating) using each season's OPENING
    five rounds, pooled across transitions — the August regime, which is what we are
    checking. Returns (home_beta, away_beta)."""
    rows = []
    prev = None
    for yy in seasons:
        d = match_probs(fetch(yy))
        if prev is not None:
            early = d.head(50).copy()
            level = float(prev.nsmallest(3).mean())
            early["own"] = early["HomeTeam"].map(prev).fillna(level)
            early["opp"] = early["AwayTeam"].map(prev).fillna(level)
            rows.append(early)
        prev = rate(d)
    pool = pd.concat(rows, ignore_index=True)
    X = np.c_[np.ones(len(pool)), pool["own"], pool["opp"]]
    home, *_ = np.linalg.lstsq(X, pool["p_home"].values, rcond=None)
    away, *_ = np.linalg.lstsq(X, pool["p_away"].values, rcond=None)
    return home, away, len(pool)


def rating_projection(start_gw, n_fixtures, ratings, home_beta, away_beta):
    """Rating-implied win / opponent-win for every team's next `n_fixtures` fixtures."""
    fx = pd.read_csv(FIXTURES)
    out = []
    for team in sorted(ratings.index):
        played = 0
        for gw in range(start_gw, 39):
            if played >= n_fixtures:
                break
            match = fx[(fx["gameweek"] == gw) &
                       ((fx["home_team"] == team) | (fx["away_team"] == team))]
            if match.empty:
                continue
            m = match.iloc[0]
            at_home = m["home_team"] == team
            opp = m["away_team"] if at_home else m["home_team"]
            if opp not in ratings.index:
                continue
            played += 1
            x = np.array([1.0, ratings[team if at_home else opp],
                          ratings[opp if at_home else team]])
            win = (home_beta if at_home else away_beta) @ x
            opp_win = (away_beta if at_home else home_beta) @ x
            out.append({"Team": team, "Fixture": played, "Gameweek": gw,
                        "Opponent": opp, "Venue": "H" if at_home else "A",
                        "rating_win": win, "rating_opp_win": opp_win})
    return pd.DataFrame(out)


def pipeline_projection(n_fixtures=8):
    """Team-level win probabilities as the pipeline currently sees them."""
    m = pd.read_csv(MASTER)
    rows = []
    for k in range(1, n_fixtures + 1):
        win = f"F{k} Win" if k <= 2 else f"F{k} Win Pred"
        opp = f"F{k} Opponent Win" if k <= 2 else f"F{k} Opponent Win Pred"
        if win not in m.columns:
            continue
        d = m[["Team", win, opp]].drop_duplicates(subset="Team")
        rows.append(d.rename(columns={win: "pipe_win", opp: "pipe_opp_win"}).assign(Fixture=k))
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, default=1)
    ap.add_argument("--fixtures", type=int, default=10)
    args = ap.parse_args()

    r = pd.read_csv(RATINGS)
    ratings = r[r["Season"] == config.SEASON].set_index("Team")["rating"]
    home_beta, away_beta, n = fit_rating_model()
    print(f"rating model fitted on {n} opening-round matches across "
          f"{len(DEFAULT_SEASONS) - 1} season transitions")
    print(f"  home win = {home_beta[0]:+.3f} {home_beta[1]:+.3f}*own {home_beta[2]:+.3f}*opp")
    print(f"  away win = {away_beta[0]:+.3f} {away_beta[1]:+.3f}*own {away_beta[2]:+.3f}*opp\n")

    proj = rating_projection(args.gw, args.fixtures, ratings, home_beta, away_beta)
    merged = proj.merge(pipeline_projection(), on=["Team", "Fixture"], how="left")
    merged["diff"] = merged["rating_win"] - merged["pipe_win"]

    print("=" * 74)
    print("DISAGREEMENT BY FIXTURE (unweighted — is the projection wrong?)")
    print("=" * 74)
    print(f"{'F':>3}{'n':>5}{'mean |diff|':>13}{'max |diff|':>12}{'bias':>9}   note")
    for k, d in merged.groupby("Fixture"):
        cmp = d.dropna(subset=["pipe_win"])
        if cmp.empty:
            print(f"{k:>3}{len(d):>5}{'—':>13}{'—':>12}{'—':>9}   ratings only (beyond F8)")
            continue
        note = "F1 = REAL odds, so this is the ratings model's own error" if k == 1 else ""
        print(f"{k:>3}{len(cmp):>5}{cmp['diff'].abs().mean():>13.4f}"
              f"{cmp['diff'].abs().max():>12.4f}{cmp['diff'].mean():>+9.4f}   {note}")

    try:
        sys.path.insert(0, config.ROOT)
        from optimisation import combine_fixture_weights
        w = combine_fixture_weights(num_fixtures=args.fixtures) if args.fixtures <= 8 else \
            combine_fixture_weights(num_fixtures=8) + [0.24, 0.22]
    except Exception:
        w = [1.0] * args.fixtures
    merged["weight"] = merged["Fixture"].map(lambda k: w[k - 1])
    merged["wdiff"] = merged["diff"].abs() * merged["weight"]

    print("\n" + "=" * 74)
    print("BIGGEST WEIGHTED DISAGREEMENTS (does it matter for the squad?)")
    print("=" * 74)
    per_team = (merged.dropna(subset=["pipe_win"]).groupby("Team")["wdiff"].sum()
                .sort_values(ascending=False))
    for team, score in per_team.head(8).items():
        d = merged[(merged["Team"] == team) & merged["pipe_win"].notna()]
        worst = d.loc[d["wdiff"].idxmax()]
        print(f"  {team:<16}{score:.3f}   worst F{int(worst['Fixture'])} v {worst['Opponent']} "
              f"({worst['Venue']}): ratings {worst['rating_win']:.2f} vs pipeline "
              f"{worst['pipe_win']:.2f}")

    out = os.path.join(config.OUTPUTS_DIR, "rating_crosscheck.csv")
    merged.to_csv(out, index=False)
    print(f"\nfull detail -> {os.path.relpath(out, config.ROOT)}")
