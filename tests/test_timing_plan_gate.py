import pandas as pd
import pytest

pulp = pytest.importorskip("pulp")  # optimisers run under the PuLP venv (env/Scripts/python)

import optimisation as og
from tests.test_optimiser_bench_slots import _squad_df


def _two_week_pool():
    """The 15-man squad plus one outside forward, X: useless in F1 (would sit on the bench), huge in F2."""
    df = _squad_df()
    df["F2 XP"] = df["F1 XP"]
    df = pd.concat([df, pd.DataFrame({"Player Name": ["X"], "Position": ["FWD"],
                                      "F1 XP": [0.1], "F2 XP": [10.0]})], ignore_index=True)
    df["Team"] = [f"T{i}" for i in range(len(df))]
    df["Cost"] = 5.0
    df["_obj0"] = df["F1 XP"] + 0.8 * df["F2 XP"]   # the planner's flat pre-filter column for step 0
    return df


def _best(df, gate):
    w, slots, gk = [1.0, 0.8], [(0.25, 0.05, 0.01)] * 2, [0.05] * 2
    return og._tc_best_squad_rich(df, 75.0, ["F1 XP", "F2 XP"], w, 0, list(range(15)), 1, slots, gk, gate=gate)


def test_ungated_plan_buys_a_player_who_only_helps_later():
    sq = _best(_two_week_pool(), None)
    assert 15 in sq and 14 not in sq            # X in for F3 (0.4), the weakest forward


def test_gate_blocks_a_signing_who_would_sit_on_the_bench_this_week():
    # X adds nothing to the F1 XI (he'd be benched), so a +0.1 F1 gate refuses the move
    assert _best(_two_week_pool(), (0.1, 0.0, 0.0)) is None


def test_gate_allows_the_move_once_only_later_weeks_are_gated():
    sq = _best(_two_week_pool(), (0.0, 1.0, 1.0))  # tail and total gains are large
    assert 15 in sq
