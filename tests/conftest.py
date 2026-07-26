import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpl_pipeline import ingest  # noqa: E402


@pytest.fixture(scope="session")
def inputs():
    return ingest.load_inputs()


@pytest.fixture(scope="session")
def sportsbet():
    return ingest.load_sportsbet()


@pytest.fixture(scope="session")
def roster():
    return ingest.load_fpl_players()


@pytest.fixture(scope="session")
def dc_stats():
    return ingest.load_defensive_contributions()
