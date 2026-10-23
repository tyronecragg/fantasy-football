import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import backfill_raw_odds as b  # noqa: E402
from fpl_pipeline import config, markets  # noqa: E402


@pytest.mark.parametrize("knots", [config.LONGSHOT_CALIBRATION, config.LONGSHOT_CALIBRATION_ASSIST])
def test_invert_recovers_the_quoted_price(knots):
    odds = pd.Series([1.45, 2.96, 3.35, 6.5, 13.0, 34.0, 81.0])
    prob = markets.longshot_calibrate(markets.implied(odds, config.MARGIN_PLAYER), knots=knots)
    back = b.invert(prob, knots)
    assert (back - odds).abs().max() < b.ODDS_TOL


def test_assist_regime_lookup():
    assert b._knots("goalscorer", "2026-2027", 2) == config.LONGSHOT_CALIBRATION
    assert b._knots("assist", "2026-2027", 2) == config.ASSIST_CURVE_REGIMES[0][3]
    assert b._knots("assist", "2026-2027", 9) == config.LONGSHOT_CALIBRATION_ASSIST
