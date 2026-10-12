"""The P1.5 timing guarantee, as a regression test: no text reaches a forecast before it exists."""

import pandas as pd
import pytest

from src.data.check_timing import SELECTED_DOMAINS, check_domain, mentioned_dates
from src.data.loader import DEFAULT_ROOT


def test_mentioned_dates_granularities():
    found = dict(mentioned_dates("Week 18 (April 27, 2020 to May 3, 2020); March 2012; in 2016"))
    assert found["April 27, 2020"] == pd.Timestamp("2020-04-27")
    assert found["March 2012"] == pd.Timestamp("2012-03-01")
    assert found["2016"] == pd.Timestamp("2016-01-01")
    assert "2020" not in found  # already consumed by the full dates


@pytest.mark.skipif(not (DEFAULT_ROOT / "numerical").exists(), reason="Time-MMD not downloaded")
@pytest.mark.parametrize("domain", SELECTED_DOMAINS)
def test_no_leakage_with_default_alignment(domain):
    r = check_domain(domain)
    assert r["metadata_violations"] == 0
    assert r["content_violations"] == 0


@pytest.mark.skipif(not (DEFAULT_ROOT / "numerical").exists(), reason="Time-MMD not downloaded")
def test_energy_lag2_no_leakage():
    """R4 held-out domain: lag 1 leaks (351/354 reports quote the next week's price); lag 2 plus
    the one known leaker is clean — the condition R4's Energy freeze rests on."""
    assert check_domain("Energy", lag=1)["content_violations"] > 300
    r = check_domain("Energy", lag=2)
    assert r["metadata_violations"] == 0
    assert r["content_violations"] == 0
