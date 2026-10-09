"""P5b analysis guards: the Holm family must be exactly the pre-registered 76 tests."""

import itertools

import numpy as np
import pandas as pd
import pytest

from src import analyze_p5 as A

DOMS = ["Economy", "SocialGood", "Traffic", "Health_US"]


def _tables():
    scr = pd.DataFrame([dict(model=m, condition="real", domain=d, quantity="sensitivity",
                             value=0.1, p_raw=0.01)
                        for m, d in itertools.product(A.TEXT_MODELS, DOMS + ["pooled"])])
    att = pd.DataFrame([dict(model=m, condition=c, domain=d, dss=0.1, dss_p_raw=0.02,
                             flip_rate=0.1, flip_p_raw=0.03, toward=3, away=1)
                        for m, c, d in itertools.product(A.MODELS, A.ATTACKS, DOMS + ["pooled"])])
    att.loc[att.model.isin(A.TEXT_BLIND), ["dss_p_raw", "flip_p_raw"]] = np.nan
    fus = pd.DataFrame([dict(comparison=f"M5 - {k}", condition=c, domain="pooled", diff=0.2,
                             p_raw=0.04) for k, c in itertools.product(("M3", "M4"), A.ATTACKS)])
    return scr, att, fus


def test_holm_family_is_the_preregistered_76():
    fam = A.holm_family(*_tables())
    assert len(fam) == 76
    assert fam.block.value_counts().to_dict() == {"S": 12, "A1": 30, "A2": 30, "F": 4}
    assert (fam.p_holm >= fam.p_raw).all()


def test_holm_family_refuses_a_missing_test():
    scr, att, fus = _tables()
    with pytest.raises(ValueError, match="pre-registered"):
        A.holm_family(scr, att, fus.iloc[1:])
    att.loc[att.model == "DLinear+BERT", "dss_p_raw"] = np.nan    # an untested cell
    with pytest.raises(ValueError, match="pre-registered"):
        A.holm_family(scr, att, fus)


def test_year_mask_drops_2020_targets():
    w = pd.DataFrame({"sample_id": ["2019-12-01", "2020-01-06", "2020-12-01", "2021-01-01"]})
    assert A.year_mask(w, 2020).tolist() == [True, False, False, True]
