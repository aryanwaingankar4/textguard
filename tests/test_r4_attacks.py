"""R4 held-out attack engines (Energy reports, FNSPID headline bundles), their scan, and the
held-out windows' timing guarantees."""

import numpy as np
import pandas as pd
import pytest

from src.attacks import consistency_r4 as c4
from src.attacks import heldout_rules as hr
from src.data.loader import DEFAULT_ROOT

E_DOWN = ("The national average retail regular gasoline price decreased for the tenth week in a row to "
          "$2.679 per gallon on December 8, 2014, $0.099 per gallon less than last week and $0.590 under a year ago.")
E_WAS = ("The national average retail regular gasoline price was $1.812 per gallon on April 20, 2020, "
         "$0.041 lower than last week’s price and $1.029 under a year ago.")
E_UP_SMALL = ("The national average retail regular gasoline price increased to $2.240 per gallon on May 2, 2016, "
              "$0.031 above last week’s price but $0.424 below a year ago.")


# --- Energy -----------------------------------------------------------------------------------

def test_energy_flip_swaps_verb_and_comparator_only():
    r = hr.energy_edit(E_DOWN, "flip")
    assert r.attacked and (r.real_direction, r.claimed_direction) == (-1, 1)
    assert r.text == E_DOWN.replace("decreased", "increased").replace("per gallon less", "per gallon more")
    assert "$2.679" in r.text and "$0.099" in r.text and "$0.590 under a year ago" in r.text


def test_energy_flip_without_verb():
    r = hr.energy_edit(E_WAS, "flip")
    assert r.attacked and r.claimed_direction == 1 and "$0.041 higher than last week’s price" in r.text


@pytest.mark.parametrize("text, reason", [
    ("The national average retail regular gasoline price was $3.704 per gallon on June 30, 2014, same as "
     "last week but $0.208 above a year ago.", "no week-over-week claim"),
    ("The national average retail regular gasoline price increased to $3.0 per gallon on May 1, 2016, "
     "$0.010 below last week’s price.", "real report's verb contradicts its comparator"),
    (None, "no text"),
])
def test_energy_not_attacked(text, reason):
    r = hr.energy_edit(text, "flip")
    assert not r.attacked and r.reason == reason and r.text == text


def test_energy_a3_scales_change_and_checks_plausibility():
    r = hr.energy_edit(E_UP_SMALL, "exaggerate", 10, max_step=0.464)
    assert r.attacked and r.claimed_direction == 1
    assert "increased sharply to $2.240" in r.text and "$0.310 above last week" in r.text
    big = hr.energy_edit(E_DOWN, "exaggerate", 10, max_step=0.464)       # 0.99 > 0.464
    assert not big.attacked and big.reason.startswith("implausible")


def test_energy_forward_sentence():
    assert hr.energy_forward(-1) == ("The national average retail regular gasoline price is expected to "
                                     "fall next week.")


def test_e1_flags_verb_comparator_mismatch_new_only():
    bad = E_DOWN.replace("decreased", "increased")                       # verb flipped, comparator not
    assert c4.new_problems(bad, E_DOWN, "Energy", "Energy")
    assert not c4.new_problems(hr.energy_edit(E_DOWN, "flip").text, E_DOWN, "Energy", "Energy")


# --- FNSPID headlines -------------------------------------------------------------------------

@pytest.mark.parametrize("ticker, h, expected", [
    ("NFLX", "Pivotal Research Maintains Buy on Netflix, Raises Price Target to $425",
     "Pivotal Research Maintains Buy on Netflix, Lowers Price Target to $425"),
    ("NFLX", "Jefferies Upgrades Netflix From Hold To Buy, Raises Price Target From $300 To $400",
     "Jefferies Downgrades Netflix From Buy To Hold, Lowers Price Target From $400 To $300"),
    ("NFLX", "UBS Downgrades Netflix to Neutral", "UBS Upgrades Netflix to Buy"),
    ("NFLX", "Stifel Upgrades Netflix to Strong Buy", "Stifel Downgrades Netflix to Buy"),
    ("NFLX", "Netflix Q3 EPS $1.47 Beats $1.05 Estimate, Sales $5.24B Miss $5.25B Estimate",
     "Netflix Q3 EPS $1.05 Misses $1.47 Estimate, Sales $5.25B Beat $5.24B Estimate"),
    ("NFLX", "Netflix Shares Up 5% After Strong Subscriber Growth",
     "Netflix Shares Down 5% After Weak Subscriber Growth"),
    ("LLY", "Eli Lilly Raises FY18 Sales Outlook From $23.7B-$24.2B To $24.0B-$24.5B vs $23.99B Est.",
     "Eli Lilly Lowers FY18 Sales Outlook From $24.0B-$24.5B To $23.7B-$24.2B vs $23.99B Est."),
    ("NVDA", "NVIDIA shares are trading higher after RBC Capital maintained an Outperform rating on the stock "
             "and raised its price target from $251 to $301.",
     "NVIDIA shares are trading lower after RBC Capital maintained an Outperform rating on the stock "
     "and lowered its price target from $301 to $251."),
])
def test_headline_flips(ticker, h, expected):
    assert not hr._unflippable(h, ticker)
    assert hr._flip_headline(h, ticker) == expected


@pytest.mark.parametrize("ticker, h", [
    ("NFLX", "Roku Falls On Morgan Stanley Downgrade, Analyst Thinks Revenue Growth Will Slow In 2020"),
    ("JNJ", "Geron Plunges As Johnson & Johnson Unit Discontinues Blood Cancer Drug Collaboration"),
    ("PFE", "Merck Or Pfizer: Which Is The Better Large-Cap Pharma Stock?"),
    ("PFE", "Leap Therapeutics Announces Collaboration with Merck and Pfizer to Evaluate TRX518 in Advanced "
            "Solid Tumors"),
    ("PCG", "PG&E Seeks Higher Electric Rates"),
])
def test_headlines_without_a_claim_about_the_stock(ticker, h):
    assert hr.headline_direction(h, ticker) == 0 and hr._flip_headline(h, ticker) == h


def test_roundup_flips_only_the_item_naming_the_stock():
    h = ("The Daily Biotech Pulse: FDA Nod For Immunomedics, Lilly's Solid Q1 Beat, Xeris Jumps On Data")
    assert hr._flip_headline(h, "LLY") == ("The Daily Biotech Pulse: FDA Nod For Immunomedics, Lilly's Weak Q1 "
                                           "Miss, Xeris Jumps On Data")


@pytest.mark.parametrize("ticker, h", [
    ("LLY", "Shares of several healthcare companies are trading lower with the overall market after oil prices "
            "fell to historic lows"),                                    # market clause with a direction
    ("MU", "Micron Shares May Have Fallen Following Comments On A Weaker Outlook In NAND"),   # fallen: handled,
    ("PCG", "PG&E shares are down 10.6% after a release indicating deteriorating finances"),
    ("NFLX", "UPDATE: Morgan Stanley Lowers Netflix Price Target As Firm Factors 'in modestly lower margins'"),
    ("PCG", "Citi Upgrades PG&E, Raises Price Target By 200%"),
])
def test_unflippable_headlines_block(ticker, h):
    if ticker == "MU":                    # "Fallen" is now reversible; the clause flips coherently
        assert not hr._unflippable(h, ticker)
        assert hr._flip_headline(h, ticker) == ("Micron Shares May Have Risen Following Comments On A Stronger "
                                                "Outlook In NAND")
    else:
        assert hr._unflippable(h, ticker)


def test_bundle_flip_and_block():
    good = "Netflix Shares Up 5% After Strong Subscriber Growth\nNetflix Adds 'Latest' Section to Its TV App"
    r = hr.fnspid_edit(good, "NFLX")
    assert r.attacked and (r.real_direction, r.claimed_direction) == (1, -1)
    assert r.text.split("\n")[1] == "Netflix Adds 'Latest' Section to Its TV App"
    bad = good + "\nNetflix Sinks As Rivals Launch"
    assert not hr.fnspid_edit(bad, "NFLX").attacked
    assert not hr.fnspid_edit("Netflix Adds 'Latest' Section", "NFLX").attacked


def test_headline_scan_checks():
    assert c4.f2_target_from_to("UBS Lowers Price Target From $300 To $400")
    assert not c4.f2_target_from_to("UBS Lowers Price Target From $400 To $300")
    assert c4.f2_target_from_to("Lilly Lowers FY18 Outlook From $23.7B-$24.2B To $24.0B-$24.5B")
    assert c4.f3_beat_numbers("Q3 EPS $1.05 Beats $1.47 Estimate")
    assert not c4.f3_beat_numbers("Q3 EPS $1.47 Beats $1.05 Estimate")
    assert c4.f4_rating_from_to("UBS Downgrades Netflix From Hold To Buy")
    assert not c4.f4_rating_from_to("UBS Downgrades Netflix From Buy To Hold")


# --- held-out windows: timing ------------------------------------------------------------------

FNSPID_DATA = DEFAULT_ROOT.parents[0] / "FNSPID" / "benzinga_frozen_tickers.parquet"


@pytest.mark.skipif(not (DEFAULT_ROOT / "numerical").exists() or not FNSPID_DATA.exists(),
                    reason="held-out data not downloaded")
def test_heldout_windows_timing():
    from src.data import fnspid, heldout
    w = heldout.windows()
    e, f = w[w.domain == "Energy"], w[w.domain == "FNSPID"]
    assert len(e) == 665 and int(e.has_text.sum()) == 352
    assert len(f) == 1031 and set(f.series) == set(fnspid.SELECTED)
    assert (f.target_date > f.origin).all() and all(len(x) == heldout.LOOKBACK for x in w.lookback)
    # every headline in a bundle has a dated occurrence in [previous origin, origin) — published
    # before the forecast origin's close (Benzinga repeats titles on other dates, so match per title)
    for tk in ("NFLX", "PCG"):
        heads = fnspid.load_headlines(tk).assign(t=lambda d: d.title.map(lambda s: " ".join(s.split())))
        weeks = heldout.weekly_closes(fnspid.load_prices(tk)).set_index("origin")
        for r in f[f.series == tk].itertuples():
            assert r.lookback[-1] == pytest.approx(weeks.loc[r.origin, "close"])
            if r.has_text:
                prev = weeks.index[weeks.index.get_loc(r.origin) - 1]
                inside = set(heads[(heads.day >= prev) & (heads.day < r.origin)].t)
                assert set(r.text.split("\n")) <= inside
