import pandas as pd
import pytest

pulp = pytest.importorskip("pulp")  # optimisers run under the PuLP venv (env/Scripts/python)

import optimisation as og


def _squad_df():
    rows = [
        ("GK A", "GK", 5.0), ("GK B", "GK", 1.0),
        ("D1", "DEF", 4.0), ("D2", "DEF", 3.9), ("D3", "DEF", 3.8),
        ("D4", "DEF", 0.9), ("D5", "DEF", 0.8),
        ("M1", "MID", 4.5), ("M2", "MID", 4.4), ("M3", "MID", 4.3),
        ("M4", "MID", 4.2), ("M5", "MID", 4.1),
        ("F1", "FWD", 3.0), ("F2", "FWD", 0.5), ("F3", "FWD", 0.4),
    ]
    return pd.DataFrame({"Player Name": [r[0] for r in rows],
                         "Position": [r[1] for r in rows],
                         "F1 XP": [r[2] for r in rows]})


def test_bench_points_prices_sub_order():
    df = _squad_df()
    bench = [1, 6, 13, 14]  # GK B 1.0, D5 0.8, F2 0.5, F3 0.4
    pts = og.bench_points_for_fixture(df, bench, "F1 XP", (0.30, 0.10, 0.05), 0.10)
    # best outfielder gets the slot-1 weight, then slot 2, slot 3; GK priced separately
    assert pts == pytest.approx(1.0 * 0.10 + 0.8 * 0.30 + 0.5 * 0.10 + 0.4 * 0.05)


def test_normalise_slot_weights():
    # one triple -> repeated per fixture; per-fixture list -> padded with its last triple
    assert og._normalise_slot_weights((0.3, 0.1, 0.05), 3) == [(0.3, 0.1, 0.05)] * 3
    assert og._normalise_slot_weights([(1, 1, 1), (0.3, 0.1, 0.05)], 4) == \
        [(1, 1, 1)] + [(0.3, 0.1, 0.05)] * 3


def test_baseline_lp_per_fixture_bench_boost_weights():
    df = _squad_df()
    df["F2 XP"] = df["F1 XP"]
    result = og.calculate_optimised_baseline(
        df, list(df.index), ["F1", "F2"], [1.0, 1.0],
        [(1.0, 1.0, 1.0), (0.30, 0.10, 0.05)], [0.10, 0.10])
    total = result[0]
    # F1 (boost): all outfielders count fully (38.8) + GK A + captain + 0.1 * GK B = 48.9
    # F2 (normal): XI 47.1 + bench 0.41
    assert total == pytest.approx(48.9 + 47.51)


def test_baseline_lp_matches_hand_computed_slot_pricing():
    df = _squad_df()
    result = og.calculate_optimised_baseline(
        df, list(df.index), ["F1"], [1.0], (0.30, 0.10, 0.05), [0.10])
    assert result is not None
    total, starting_xi, f1_total, f1_starting = result
    # Optimal XI: GK A, D1-D4, M1-M5, F1 (starting D4's 0.9 beats any bench-slot shuffle),
    # captain GK A (5.0). Bench: D5/F2/F3 earn 0.30/0.10/0.05 by XP order, GK B earns 0.10.
    assert starting_xi == pytest.approx(42.1 + 5.0)
    assert total == pytest.approx(47.1 + 0.8 * 0.30 + 0.5 * 0.10 + 0.4 * 0.05 + 1.0 * 0.10)
    assert f1_total == pytest.approx(total)  # single fixture at weight 1.0


def test_fixture_weights_combine_ownership_and_reliability():
    import optimisation as og

    w = og.combine_fixture_weights()
    assert w[0] == 1.0                                  # normalised to F1
    assert all(a >= b for a, b in zip(w, w[1:]))        # monotonically decreasing
    assert w[1] - w[2] > w[2] - w[3]                    # F2->F3 cliff, not a linear ramp

    # each component is usable in isolation, and both default to 8 fixtures
    assert og.combine_fixture_weights(reliability=[1] * 8) == pytest.approx(
        list(og.OWNERSHIP_WEIGHTS))
    assert og.combine_fixture_weights(ownership=[1] * 8) == pytest.approx(
        list(og.RELIABILITY_WEIGHTS))
    assert len(og.combine_fixture_weights(num_fixtures=6)) == 6
    with pytest.raises(ValueError, match="need 8"):
        og.combine_fixture_weights(ownership=[1.0, 0.9])
