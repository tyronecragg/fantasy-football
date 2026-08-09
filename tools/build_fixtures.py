"""Generate inputs/fixtures.csv (the rolling F1-F8 window the pipeline consumes) from
inputs/season_fixtures.csv (the full-season fixture list).

Usage:  python tools/build_fixtures.py --gw N     # N = the upcoming gameweek (F1)

Run this each week (or whenever fixtures change), then hand-edit inputs/fixtures.csv
for postponements/double gameweeks — the generator assumes the static one-match-per-
round list, so late-season reschedules are yours to adjust after generating.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import config  # noqa: E402

SEASON_FIXTURES_CSV = os.path.join(config.INPUTS_DIR, "season_fixtures.csv")
FIXTURES_CSV = os.path.join(config.INPUTS_DIR, "fixtures.csv")
WINDOW = 8  # the Team Fixture Odds view reads up to F8


def build_window(season_fixtures, start_gw, window=WINDOW):
    """Wide per-team frame: Team, then (opponent, venue) per gameweek in the window —
    the positional layout team_model.team_fixture_view expects."""
    teams = sorted(set(season_fixtures["home_team"]) | set(season_fixtures["away_team"]))
    out = pd.DataFrame({"Team": teams})

    for gw in range(start_gw, start_gw + window):
        sub = season_fixtures[season_fixtures["gameweek"] == gw]
        if sub.empty:
            continue
        home = sub.set_index("home_team")["away_team"]
        away = sub.set_index("away_team")["home_team"]
        opp = out["Team"].map(home).where(out["Team"].isin(home.index), out["Team"].map(away))
        venue = np.where(out["Team"].isin(home.index), "H",
                         np.where(out["Team"].isin(away.index), "A", None))
        out[f"GW{gw} Opponent"] = opp
        out[f"GW{gw} Venue"] = venue
    return out


def main(start_gw):
    season_fixtures = pd.read_csv(SEASON_FIXTURES_CSV)
    out = build_window(season_fixtures, start_gw)
    out.to_csv(FIXTURES_CSV, index=False)
    n_gws = (out.shape[1] - 1) // 2
    print(f"Wrote {FIXTURES_CSV}: {len(out)} teams, GW{start_gw}-GW{start_gw + n_gws - 1} "
          f"({n_gws} gameweeks in window)")


if __name__ == "__main__":
    if "--gw" not in sys.argv:
        raise SystemExit("Usage: python tools/build_fixtures.py --gw N")
    main(int(sys.argv[sys.argv.index("--gw") + 1]))
