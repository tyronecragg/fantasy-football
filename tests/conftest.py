import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpl_pipeline import config, ingest  # noqa: E402


# All heavy fixtures use the frozen parity-reference data, not the live inputs/ and
# sportsbet/ files — those change every scrape/season and would make tests flaky.
@pytest.fixture(scope="session")
def inputs():
    return ingest.load_inputs(config.PARITY_INPUTS_DIR)


@pytest.fixture(scope="session")
def sportsbet():
    return ingest.load_sportsbet(config.PARITY_SPORTSBET_DIR)


# Roster/DC come from the workbook's frozen sheets, not the live FPL data: the upstream
# repo rewrites past seasons and the live roster changes every season, which would make
# these tests (and their named players) non-deterministic.
@pytest.fixture(scope="session")
def roster():
    return ingest.load_fpl_players_workbook()


@pytest.fixture(scope="session")
def dc_stats():
    return ingest.load_defensive_contributions_workbook()
