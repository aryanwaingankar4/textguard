"""Tests for R3 Amendment 1's cross-sentence contradiction scan (``src.attacks.consistency_r3``).

Each check is tested on a short text modelled on a case found in Aryan's by-eye review
(``texts/r3_review_pairs.md``) — the contradiction must be flagged — and on its real counterpart,
which must not be. A last test (skipped without the generated texts) asserts that no genuine
val/test report trips any single-text check, i.e. the scan blames only the attacks.
"""

import pytest

from src.attacks import common
from src.attacks import consistency_r3 as cr

# pair 3 (A1, Traffic): week 6 = 14.2B, week 7 = 14.5B, but "2.1% lower than the previous week"
WEEKS_REAL = ("Vehicle miles traveled for Week 6 are estimated to be 14.2 billion vehicle miles, "
              "7.4% higher than the previous week.\n"
              "For Week 7, vehicle miles traveled are estimated to be 14.5 billion vehicle miles, "
              "2.1% higher than the previous week.")
WEEKS_FAKE = WEEKS_REAL.replace("higher", "lower")

# pair 15 (A3, Traffic): "+19.0%" vs "a 1.50% change", and +19.0% (+45.0B) with level 233.7B
MONTH_REAL = ("VMT increased by 1.9% (+4.5 billion vehicle miles) in February 2023 compared to "
              "February 2022, with an estimated 233.7 billion vehicle miles traveled. The seasonally "
              "adjusted VMT is 270.5 billion miles, a 1.50% change over February 2022.")
MONTH_FAKE = MONTH_REAL.replace("increased by 1.9% (+4.5", "increased sharply by 19.0% (+45.0")

# pair 11 (A3, Economy): June recomputed to $74.9B, 3-month average still $43.2B
ECON_REAL = ("* The goods and services deficit was $41.9 billion in July, down $3.3 billion from "
             "$45.2 billion in June.\n* Exports were $188.5 billion, $0.8 billion more than June exports.\n"
             "* Imports were $230.4 billion, $2.5 billion less than June imports.\n"
             "* The average goods and services deficit decreased $0.1 billion to $43.2 billion for "
             "the three months ending in July.")
ECON_FAKE = (ECON_REAL.replace("$3.3 billion from $45.2", "$33.0 billion from $74.9")
             .replace("$0.8", "$8.0").replace("$2.5", "$25.0").replace("$0.1", "$1.0"))

# §8 "R3" item 2 (A3, Health_US): "decreased sharply to 5.3%" where the real said "decreased to 2.4%"
HUS_REAL = ("The proportion of outpatient visits for ILI decreased to 2.4%, but remains above the "
            "national baseline of 2.2%.")
HUS_FAKE = ("The proportion of outpatient visits for ILI decreased sharply to 5.3%, but remains well "
            "above the national baseline of 2.2%.")


def test_c1_week_over_week_change_against_stated_levels():
    assert cr.c1_week_sequence(WEEKS_REAL, "Traffic") == {}
    assert list(cr.c1_week_sequence(WEEKS_FAKE, "Traffic")) == ["C1 week 7"]


def test_c2_percent_vs_absolute_change_vs_level():
    assert cr.c2_pct_vs_level(MONTH_REAL, "Traffic") == {}
    assert len(cr.c2_pct_vs_level(MONTH_FAKE, "Traffic")) == 1


def test_c2_needs_the_level_in_the_same_sentence():
    # a "cumulative" sentence with its own change but no level must not borrow another's level
    t = ("Travel changed by 0.8% (2.2 billion vehicle miles) for July 2017. The estimated travel for "
         "the month is 283.6 billion vehicle miles. The cumulative travel for 2017 changed by 1.5% "
         "(26.8 billion vehicle miles).")
    assert cr.c2_pct_vs_level(t, "Traffic") == {}


def test_c3_unadjusted_vs_seasonally_adjusted(monkeypatch):
    monkeypatch.setattr(cr, "nsa_sa_tolerance", lambda: 2.4)
    assert cr.c3_nsa_vs_sa(MONTH_REAL, "Traffic") == {}
    assert list(cr.c3_nsa_vs_sa(MONTH_FAKE, "Traffic")) == ["C3 nsa/sa"]


def test_c4_three_month_average_implies_an_impossible_month(monkeypatch):
    monkeypatch.setattr(cr, "deficit_range", lambda: (18.7, 219.6))
    assert cr.c4_economy_avg3(ECON_REAL, "Economy") == {}           # third month 42.5
    assert list(cr.c4_economy_avg3(ECON_FAKE, "Economy")) == ["C4 avg3 #0"]   # third month 12.8


def test_c5_deficit_change_equals_imports_minus_exports():
    assert cr.c5_economy_components(ECON_FAKE, ECON_REAL) == {}      # A3 scales all three: consistent
    broken = ECON_FAKE.replace("$25.0 billion less", "$9.5 billion less")
    assert len(cr.c5_economy_components(broken, ECON_REAL)) == 1


def test_c5_carries_the_real_reports_rounding_gap_through_the_edit():
    # real: +3.2 vs +1.6 − (−1.5) = +3.1 (a rounding gap); ×10 → +32 vs +31 is the same gap, not new
    real = ("The goods and services deficit was $45.0 billion in May, up $3.2 billion from $41.8 billion "
            "in April. Exports were $180.0 billion, $1.5 billion less than April exports. Imports were "
            "$225.0 billion, $1.6 billion more than April imports.")
    fake = (real.replace("$3.2 billion from $41.8", "$32.0 billion from $13.0")
            .replace("$1.5 billion", "$15.0 billion").replace("$1.6 billion", "$16.0 billion"))
    assert cr.c5_economy_components(fake, real) == {}


def test_c6_level_moved_against_its_own_verb():
    assert list(cr.c6_level_against_direction(HUS_FAKE, HUS_REAL)) == ["C6 sent 0 #0"]
    # a flip that moves the level WITH the new verb is consistent
    ok = HUS_REAL.replace("decreased to 2.4%", "increased to 2.6%")
    assert cr.c6_level_against_direction(ok, HUS_REAL.replace("decreased to 2.4%", "decreased to 2.0%")) == {}
    # the lie itself — verb flipped, genuine level kept — stays attacked (Amendment 1 rule)
    verb_only = HUS_REAL.replace("decreased to 2.4%", "increased to 2.4%")
    assert cr.c6_level_against_direction(verb_only, HUS_REAL) == {}
    # example #5 (A1 Health_US 2019-03-04): the level-mirror edit moved 4.7 → 1.0 under "increased"
    real5 = "ILI decreased slightly to 4.7%, which is above the national baseline of 2.2%."
    fake5 = "ILI increased slightly to 1.0%, which is below the national baseline of 2.2%."
    assert list(cr.c6_level_against_direction(fake5, real5)) == ["C6 sent 0 #0"]


@pytest.mark.parametrize("text", [
    "The U.S. international trade deficit increased sharply by $64.0 billion to $42.6 billion.",
    "The average goods and services deficit increased $66.0 billion to $48.9 billion for the three months.",
])
def test_c7_applies_p2_as_written(text):
    # §10f P2: "by Y … to X" must imply a start > 0 — also with A3's own "sharply" before "by",
    # and without "by"; the generator's P2 implementation missed both forms
    assert len(cr.c7_p2_implied_start(text, "Economy")) == 1
    assert cr.c7_p2_implied_start(text.replace("$64.0", "$6.4").replace("$66.0", "$6.6"), "Economy") == {}


def test_new_problems_ignores_problems_the_real_report_already_has():
    assert cr.new_problems(WEEKS_FAKE, WEEKS_FAKE.replace("Week 6", "Week 6 "), "Traffic") == {}
    assert cr.new_problems(WEEKS_REAL, WEEKS_REAL, "Traffic") == {}


@pytest.mark.skipif(not (common.TEXTS_DIR / "real.csv").exists(), reason="texts not generated")
def test_no_genuine_report_trips_a_single_text_check():
    real = common.load("real")
    assert not [k for k, r in real.iterrows() if cr.problems(r.text, r.domain)]
