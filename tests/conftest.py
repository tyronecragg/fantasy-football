import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpl_pipeline import ingest  # noqa: E402

# Frozen test data (tests/data/), never the live inputs/ and sportsbet/ files - those change every
# scrape/season and would make tests flaky. It is the 2025-26 snapshot the retired workbook-parity check
# used (moved here 2026-10-23 when parity was removed); the roster and DefCon tables were exported from
# the workbook's FPL Players / Defensive Contribution sheets so the named players stay deterministic.
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@pytest.fixture(scope="session")
def inputs():
    return ingest.load_inputs(os.path.join(DATA, "inputs"))


@pytest.fixture(scope="session")
def sportsbet():
    return ingest.load_sportsbet(os.path.join(DATA, "sportsbet"))


@pytest.fixture(scope="session")
def roster():
    return pd.read_csv(os.path.join(DATA, "fpl_players.csv"))


@pytest.fixture(scope="session")
def dc_stats():
    return {pos: pd.read_csv(os.path.join(DATA, f"dc_stats_{pos.lower()}.csv")) for pos in ("DEF", "MID")}
