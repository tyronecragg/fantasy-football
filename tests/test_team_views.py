import numpy as np
import pandas as pd

from fpl_pipeline import config, rating


class _Maps:
    def predict(self, diff, weeks_ahead):
        p = 1 / (1 + np.exp(-np.asarray(diff)))
        return np.column_stack([p * 0.8, np.full_like(p, 0.2), (1 - p) * 0.8])

    def goals(self, log_lam, weeks_ahead):
        return np.exp(np.asarray(log_lam))


def _fr():
    teams = ["Leeds", "Man Utd", "Arsenal"]
    r = pd.Series([0.0, 0.2, 0.5], index=teams)
    att = pd.Series([0.0, 0.1, 0.3], index=teams)
    dfn = pd.Series([0.0, 0.1, 0.3], index=teams)
    return rating.FixtureRating(r, 0.3, 100.0, _Maps(), pd.Timestamp("2026-10-17"), "2026-2027",
                                (att, dfn, 0.3, 0.1))


def _csv(tmp_path, body):
    p = tmp_path / "team_views.csv"
    p.write_text("# comment line\nGameweek,Team,nudge,note\n" + body, encoding="utf-8")
    return str(p)


def test_only_current_gameweek_rows_apply(tmp_path):
    p = _csv(tmp_path, "7,Leeds,0.12,x\n6,Arsenal,-0.5,stale\n")
    assert rating.load_team_views(7, p) == {"Leeds": 0.12}
    assert rating.load_team_views(8, p) == {}          # forgotten rows go inert after the roll
    assert rating.load_team_views(None, p) == {}


def test_missing_file_is_a_no_op(tmp_path):
    assert rating.load_team_views(7, str(tmp_path / "nope.csv")) == {}


def test_nudge_is_capped(tmp_path):
    p = _csv(tmp_path, "7,Leeds,3.0,typo\n")
    assert rating.load_team_views(7, p) == {"Leeds": config.TEAM_VIEW_MAX}


def test_apply_shifts_strength_and_goals_split():
    fr = _fr()
    new, applied, skipped = rating.apply_team_views(fr, {"Leeds": 0.1, "Man Utd": -0.1, "Nowhere FC": 0.2})
    assert applied == {"Leeds": 0.1, "Man Utd": -0.1} and skipped == ["Nowhere FC"]
    assert np.isclose(new.ratings["Leeds"], 0.1 * config.TEAM_VIEW_LOGODDS_PER_GOAL)
    assert np.isclose(new.ratings["Man Utd"], 0.2 - 0.1 * config.TEAM_VIEW_LOGODDS_PER_GOAL)
    assert new.ratings["Arsenal"] == fr.ratings["Arsenal"]
    assert fr.ratings["Leeds"] == 0.0                   # original untouched
    # Leeds at home to Man Utd: Leeds more likely to win, scores more, concedes less
    t, o, h = pd.Series(["Leeds"]), pd.Series(["Man Utd"]), pd.Series([True])
    w0, _ = fr.win_probs(t, o, h, 2)
    w1, _ = new.win_probs(t, o, h, 2)
    assert w1.iloc[0] > w0.iloc[0]
    f0, a0 = fr.expected_goals(t, o, h, 2)
    f1, a1 = new.expected_goals(t, o, h, 2)
    assert f1.iloc[0] > f0.iloc[0] and a1.iloc[0] < a0.iloc[0]


def test_no_views_returns_same_object():
    fr = _fr()
    assert rating.apply_team_views(fr, {})[0] is fr


def test_current_f1_gw(tmp_path):
    p = tmp_path / "fixtures.csv"
    p.write_text("Team,GW7 Opponent,GW7 Venue\nLeeds,Man Utd,H\n", encoding="utf-8")
    assert rating.current_f1_gw(str(p)) == 7
