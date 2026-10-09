"""P6b analysis guards: the selection rule, the 20-test Holm family, TextGuard-L."""

import itertools

import numpy as np
import pandas as pd
import pytest

from src import analyze_p6 as A

DOMS = ["Economy", "SocialGood", "Traffic", "Health_US"]


def _grid(rows):
    return pd.DataFrame(rows, columns=["tau", "fraction", "dss", "fbr"])


def test_selection_takes_lowest_dss_within_budget_then_tie_breaks():
    g = _grid([(0.2, 0.5, 0.10, 0.05), (0.1, 0.5, 0.10, 0.05), (0.0, 1.0, 0.10, 0.08),
               (0.3, 0.25, 0.02, 0.20)])               # lowest DSS but over budget
    pick, ok = A._select(g, ["tau", "fraction"])
    assert ok and pick.tau == 0.1 and pick.fraction == 0.5   # tie on DSS + FBR -> smaller τ


def test_budget_not_met_falls_back_to_lowest_fbr_and_ignores_dss_on_ties():
    """Amendment 2: a tie at the lowest FBR is broken by smaller τ, then smaller w — never DSS."""
    g = _grid([(0.5, 0.25, -0.05, 0.113), (0.0, 0.25, -0.02, 0.113), (0.0, 0.5, -0.07, 0.15)])
    pick, ok = A._select(g, ["tau", "fraction"])
    assert not ok and pick.tau == 0.0 and pick.fraction == 0.25


def test_budget_is_inclusive():
    pick, ok = A._select(_grid([(0.0, 0.25, 0.3, 0.10)]), ["tau", "fraction"])
    assert ok


def _tests_table():
    return pd.DataFrame([dict(block=b, domain=d, p_raw=0.01, sign_ok=True)
                         for b, d in itertools.product(("T1", "T2", "T3", "T4"), DOMS + ["pooled"])])


def test_holm_family_is_the_preregistered_20():
    fam = A.holm_family_p6(_tests_table())
    assert len(fam) == 20 and (fam.p_holm >= fam.p_raw).all()


def test_holm_family_refuses_a_missing_or_extra_test():
    t = _tests_table()
    with pytest.raises(ValueError, match="pre-registered"):
        A.holm_family_p6(t.iloc[1:])
    extra = pd.concat([t, t.iloc[[0]].assign(block="T5")])
    with pytest.raises(ValueError, match="pre-registered"):
        A.holm_family_p6(extra)


def test_extra_groups_stay_outside_the_family():
    t = pd.concat([_tests_table(), _tests_table().assign(domain="change-flip (Eco+SG+Tra)")
                   .query("domain == 'change-flip (Eco+SG+Tra)'").head(4)])
    assert len(A.holm_family_p6(t)) == 20


def test_textguard_l_uses_the_level_gate_on_health_us_only():
    gt = pd.DataFrame({"domain": ["Health_US", "Health_US", "Traffic"],
                       "g": [0.2, 0.3, 0.4], "g_level": [0.9, np.nan, np.nan]})
    assert A.level_only(gt).tolist() == [0.9, 1.0, 0.4]


def test_mcnemar_counts_discordant_windows_only():
    a = np.array([1, 1, 0, 0, np.nan])
    b = np.array([0, 1, 1, 0, 1])
    r = A._mcnemar(a, b, np.ones(5, bool))
    assert (r["only_first"], r["only_second"]) == (1, 1) and r["p"] == 1.0


def test_test_stage_refuses_to_run_without_frozen_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "FROZEN", tmp_path / "missing.json")
    with pytest.raises(SystemExit, match="tune"):
        A.load_frozen()
