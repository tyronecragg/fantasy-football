import pandas as pd
import pytest

pulp = pytest.importorskip("pulp")  # optimisers run under the PuLP venv (env/Scripts/python)

import optimisation as og
from tests.test_optimiser_bench_slots import _squad_df


def _doubtful_m1():
    """M1 (index 7, 4.5 XP) has only a 50% chance of starting; everyone else is nailed."""
    df = _squad_df()
    df["F1 Start"] = 1.0
    df.loc[7, "F1 Start"] = 0.5
    return df


def test_captain_and_vice_counts_the_vice_when_captain_may_miss():
    df = pd.DataFrame({"Player Name": ["A", "B", "C"], "Position": ["MID"] * 3,
                       "F1 XP": [6.0, 5.8, 4.0], "F1 Start": [1.0, 0.6, 1.0]})
    # XP alone: A. With cover: B is worth 5.8 + 0.4 x 6.0 (A as vice) = 8.2 > A's 6.0 + 0 x 5.8
    assert og.captain_and_vice([0, 1, 2], df, "F1", cover=False)[0] == 0
    cap, vice, pts = og.captain_and_vice([0, 1, 2], df, "F1")
    assert (cap, vice) == (1, 0)
    assert pts == pytest.approx(0.4 * 6.0)


def test_no_start_column_keeps_xp_only_captaincy():
    df = _squad_df()                       # no 'F1 Start' -> nobody can miss -> old rule
    cap, _, pts = og.captain_and_vice(list(range(15)), df, "F1")
    assert cap == 0 and pts == 0.0         # GK A, the top XP


def test_baseline_lp_credits_vice_cover():
    df = _doubtful_m1()
    total, starting_xi, _, _ = og.calculate_optimised_baseline(
        df, list(df.index), ["F1"], [1.0], (0.30, 0.10, 0.05), [0.10])
    # Same XI as the plain case (42.1). Captain M1: 4.5 + 0.5 x 5.0 (GK A as vice) = 7.0, beating
    # GK A's 5.0 - the LP and captain_and_vice agree.
    assert starting_xi == pytest.approx(42.1 + 7.0)
    assert total == pytest.approx(49.1 + 0.8 * 0.30 + 0.5 * 0.10 + 0.4 * 0.05 + 1.0 * 0.10)


def test_baseline_cover_switch_off_restores_old_value():
    df = _doubtful_m1()
    _, starting_xi, _, _ = og.calculate_optimised_baseline(
        df, list(df.index), ["F1"], [1.0], (0.30, 0.10, 0.05), [0.10], vice_captain_cover=False)
    assert starting_xi == pytest.approx(42.1 + 5.0)   # GK A captain, no cover


def test_pure_xp_lineup_and_hold_pick_the_cover_aware_captain():
    df = _doubtful_m1()
    _, _, cap = og.pure_xp_lineup(list(df.index), df, "F1 XP")
    assert cap == 7
    hold = og.build_hold_pseudo_solution(df, list(df.index), ["F1"], 0.0)
    assert hold["captains"]["F1"] == 7


def test_f1_lineup_uses_unadjusted_xp_when_fixture_risk_applied():
    # M1 (index 7) is in a postponement-risk match: adjusted F1 XP 3.0, unadjusted 6.0. He should still
    # start and captain (autosub covers a call-off); other weeks are unaffected.
    df = _squad_df()
    df["F1 XP Raw"] = df["F1 XP"]
    df.loc[7, "F1 XP Raw"] = 6.0
    df.loc[7, "F1 XP"] = 3.0
    assert og.lineup_col(df, "F1") == "F1 XP Raw" and og.lineup_col(df, "F2") == "F2 XP"
    xi, _, cap = og.pure_xp_lineup(list(df.index), df, "F1 XP")
    assert 7 in xi and cap == 7
