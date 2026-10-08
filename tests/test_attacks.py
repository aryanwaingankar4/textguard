"""Tests for the attack module (P3): the four simple conditions on real data, and the
rule engine behind ``flipped`` / ``exaggerated`` on hand-written reports in the style of
each domain (including the traps found while scanning the real texts)."""

import re
from pathlib import Path

import pandas as pd
import pytest

from src.attacks import common, cross_domain, empty, shuffled, time_shifted
from src.attacks.rules import coherence_problems, edit_text, headline

DATA = Path(__file__).resolve().parents[1] / "data" / "Time-MMD"
needs_data = pytest.mark.skipif(not DATA.exists(), reason="Time-MMD not downloaded")


@pytest.fixture(scope="module")
def w():
    return common.windows()


# --- simple conditions ------------------------------------------------------------------------

@needs_data
def test_empty_has_no_text(w):
    df = empty.build(w)
    assert len(df) == len(w) and not df.has_text.any()


@needs_data
def test_shuffled_text_comes_from_other_earlier_dates(w):
    df = shuffled.build(w).merge(w[["domain", "split", "sample_id", "text"]],
                                 on=["domain", "split", "sample_id"], suffixes=("", "_real"))
    assert df.has_text.all()
    assert (df.source_sample_id != df.sample_id).all()                     # another date
    assert (pd.to_datetime(df.source_sample_id) < pd.to_datetime(df.sample_id)).all()  # no future
    own = df.text_real.notna()
    assert (df.text[own] != df.text_real[own]).all()                       # never its own text
    for d in common.DOMAINS:                                               # text really is that date's
        tl = common.timeline(d).set_index("sample_id").text
        sub = df[df.domain == d]
        assert (sub.text.values == tl.reindex(sub.source_sample_id).values).all()
    assert shuffled.build(w).equals(shuffled.build(w))                    # deterministic


@needs_data
def test_time_shift_offset_is_one_year(w):
    df = time_shifted.build(w)
    has = df[df.has_text]
    tgt, src = pd.to_datetime(has.sample_id), pd.to_datetime(has.source_sample_id)
    monthly = has.domain != "Health_US"
    assert (src[monthly] == tgt[monthly] - pd.DateOffset(months=12)).all()
    assert ((tgt[~monthly] - src[~monthly]).dt.days.between(52 * 7 - 7, 52 * 7 + 7)).all()
    for d in common.DOMAINS:                                               # exactly SHIFT rows back
        tl = common.timeline(d).set_index("sample_id").row
        sub = has[has.domain == d]
        shift = time_shifted.SHIFT["weekly" if d == "Health_US" else "monthly"]
        assert ((tl.reindex(sub.sample_id).values - tl.reindex(sub.source_sample_id).values) == shift).all()


@needs_data
def test_cross_domain_uses_another_domain_available_at_the_time(w):
    df = cross_domain.build(w)
    assert (df.source_domain != df.domain).all()
    assert (pd.to_datetime(df.source_sample_id) <= pd.to_datetime(df.sample_id)).all()


# --- rule engine: flip -------------------------------------------------------------------------

ECON = ("The goods and services deficit was $43.1 billion in May, down $3.0 billion from $46.1 "
        "billion in April. Exports of goods for May were $143.6 billion, $2.9 billion more than "
        "April exports. The U.S. deficit decreased by $6.4 billion to $64.2 billion in March 2023.")
TRAFFIC = ("Travel on all roads and streets changed by -18.6% (-50.6 billion vehicle miles) for "
           "March 2020 as compared with March 2019. The estimated travel for the month is 221.0 "
           "billion vehicle miles. The seasonally adjusted travel is 266.7 billion miles, a -0.1% "
           "(-0.4 billion vehicle miles) decline from February 2018.")
SOCIAL = ("The unemployment rate for February 2015 was 5.5 percent, down from 6.7 percent a year "
          "earlier. The unemployment rate declined by 0.2 percentage point to 3.6 percent in March.")
HEALTH = ("During week 35, 1.0% of patient visits reported through ILINet were due to "
          "influenza-like illness (ILI). This percentage is below the national baseline of 2.2%.")


def test_economy_flip_keeps_level_and_recomputes_the_from_value():
    r = edit_text(ECON, "Economy", "flip")
    assert r.attacked and r.real_direction == +1 and r.claimed_direction == -1  # deficit down = balance up
    assert "$43.1 billion in May, up $3.0 billion from $40.1 billion" in r.text
    assert "$2.9 billion less than April exports" in r.text
    assert "deficit increased by $6.4 billion to $64.2 billion in March 2023" in r.text
    assert coherence_problems(r.text) == []


def test_traffic_flip_signs_and_double_negatives():
    r = edit_text(TRAFFIC, "Traffic", "flip")
    assert r.claimed_direction == +1
    assert "changed by +18.6% (+50.6 billion vehicle miles)" in r.text
    assert "a +0.1% (+0.4 billion vehicle miles) rise" in r.text
    assert "221.0 billion" in r.text and "266.7 billion" in r.text           # levels untouched


def test_unsigned_change_becomes_negative():
    t = "Travel on all roads and streets changed by 1.0% (2.9 billion vehicle miles) for October 2019."
    r = edit_text(t, "Traffic", "flip")
    assert "changed by -1.0% (-2.9 billion vehicle miles)" in r.text and "October 2019" in r.text


def test_socialgood_down_from_is_mirrored_around_the_current_level():
    r = edit_text(SOCIAL, "SocialGood", "flip")
    assert "5.5 percent, up from 4.3 percent" in r.text
    assert "rose by 0.2 percentage point to 3.6 percent" in r.text and r.claimed_direction == +1


def test_health_level_flip_moves_the_value_across_the_baseline():
    r = edit_text(HEALTH, "Health_US", "flip")
    assert r.variant == "level" and r.claimed_direction == +1
    v = float(re.search(r"(\d+\.\d)% of patient visits", r.text).group(1))
    assert v > 2.2 and "above the national baseline of 2.2%" in r.text
    assert coherence_problems(r.text) == []


@pytest.mark.parametrize("text, domain, keep", [
    ("The unemployment rate declined to 3.6%. The number of persons unemployed for less than "
     "5 weeks decreased by 393,000.", "SocialGood", "for less than 5 weeks"),
    ("The unemployment rate declined to 3.6%. The rate for Hispanics was 5.8 percent, higher "
     "than the 4.7-percent rate for non-Hispanics.", "SocialGood", "5.8 percent, higher than"),
    ("The unemployment rate edged down to 3.6%. Job growth averaged 158,000, below the average "
     "of 223,000 in 2018.", "SocialGood", "below the average of 223,000"),
    ("The deficit increased by less than $0.1 billion to $65.3 billion.", "Economy",
     "less than $0.1 billion"),
])
def test_context_traps_are_not_flipped(text, domain, keep):
    r = edit_text(text, domain, "flip")
    assert r.attacked and keep in r.text


def test_up_down_before_less_than_is_flipped():
    t = "The goods and services deficit was $89.2 billion in February, down less than $0.1 billion from January."
    assert "up less than $0.1 billion" in edit_text(t, "Economy", "flip").text


def test_record_phrases_swap_highest_and_lowest():
    t = "The unemployment rate declined by 0.2 percentage point to 3.6 percent, the lowest rate since December 1969."
    assert "rose by 0.2 percentage point to 3.6 percent, the highest rate since" in edit_text(t, "SocialGood", "flip").text


@pytest.mark.parametrize("text, domain, reason", [
    ("The unemployment rate was unchanged at 5.3 percent in July.", "SocialGood", "no directional headline"),
    ("The proportion of ILI was 2.1%, which is at the national baseline of 2.1%. Six of 10 regions "
     "reported ILI at or above region-specific baseline levels.", "Health_US", "no directional headline"),
    ("The unemployment rate fell to 4.0%. Mining lost jobs.", "SocialGood", "unhandled directional word"),
    ("", "Economy", "no text"),
    (None, "Economy", "no text"),
])
def test_not_attacked_returns_the_original(text, domain, reason):
    r = edit_text(text, domain, "flip")
    assert not r.attacked and r.reason.startswith(reason) and r.text == text and r.claimed_direction == 0


def test_flip_never_creates_contradictions_on_the_examples():
    for t, d in ((ECON, "Economy"), (TRAFFIC, "Traffic"), (SOCIAL, "SocialGood"), (HEALTH, "Health_US")):
        r = edit_text(t, d, "flip")
        assert set(coherence_problems(r.text)) <= set(coherence_problems(t))
        assert headline(r.text, d).series_direction == -headline(t, d).series_direction


# --- rule engine: exaggerate ------------------------------------------------------------------

def test_exaggerate_scales_change_and_keeps_direction():
    r = edit_text(ECON, "Economy", "exaggerate")
    assert r.claimed_direction == r.real_direction == +1
    assert "down $9.0 billion from $52.1 billion" in r.text                   # 43.1 + 9.0
    assert "$8.7 billion more than" in r.text and "decreased by $19.2 billion" in r.text  # x3, verb not in the headline sentence
    assert coherence_problems(r.text) == []


def test_exaggerate_level_moves_further_from_baseline():
    r = edit_text(HEALTH, "Health_US", "exaggerate")
    v = float(re.search(r"(\d+\.\d)% of patient visits", r.text).group(1))
    assert v < 1.0 and "well below the national baseline of 2.2%" in r.text


def test_exaggerated_decrease_cannot_reach_100_percent():
    t = "Travel on all roads and streets changed by -40.0% (-100.0 billion vehicle miles) for April 2020."
    r = edit_text(t, "Traffic", "exaggerate")
    assert not r.attacked and "100%" in r.reason


# --- generated files -------------------------------------------------------------------------

@needs_data
def test_generated_conditions_have_standard_columns_and_label_fakes(tmp_path, w):
    from src.attacks import flipped
    df = flipped.build(w)
    assert list(df.columns) == common.COLUMNS
    assert (df.synthetic == df.attacked).all()
    common.save(df, "flipped", tmp_path)
    back = common.load("flipped", tmp_path)
    assert len(back) == len(df) and "SYNTHETIC" in (tmp_path / "README.txt").read_text()
    a = df[df.attacked]
    assert (a.claimed_direction == -a.real_direction).all() and (a.claimed_direction != 0).all()


def test_level_mirror_is_linear_with_ratio_fallback():
    from src.attacks.rules import level_mirror
    assert level_mirror(0.9, 2.2) == pytest.approx(3.5)        # same distance, other side
    assert level_mirror(3.2, 2.2) == pytest.approx(1.2)
    assert level_mirror(5.0, 2.2) == pytest.approx(2.2 ** 2 / 5.0)  # 2B - V = -0.6 -> ratio
    t = ("During week 34, 0.9% of patient visits were due to influenza-like illness (ILI). "
         "This percentage is below the national baseline of 2.2%.")
    assert "3.5% of patient visits" in edit_text(t, "Health_US", "flip").text
