"""Judge questions for R4's held-out domains — v1's and v2's subjects for Energy and FNSPID.

Approved by Aryan 2026-10-16 and fixed in the R4 pre-registration (PROJECT_STATUS §10g) before any
judge output on either domain. **Plain dicts, no imports:** the held-out Kaggle judge notebooks
paste this file verbatim after the judge prompt module and extend it in place
(``QUESTIONS.update(HELDOUT_QUESTIONS)`` for v1, ``SUBJECTS.update(HELDOUT_SUBJECTS)`` for v2), so
v1's frozen ``judge_prompt.py`` and v2's ``judge_prompt_v2.py`` are never edited. Locally,
:func:`register` does the same.

Domain keys: ``"Energy"`` and ``"FNSPID:<TICKER>"`` (one subject per stock).

* **v1** follows v1's own convention — the claim's frame goes into the subject (as Traffic's
  "compared with the same month one year earlier"): Energy's attacked claim is week-over-week.
  Change question only; neither domain states a baseline, so no level question. POLARITY +1.
* **v2** names only the quantity; the frame is read from each unit (``units.frame``).
"""

COMPANY_NAMES = {"NFLX": "Netflix", "GOOGL": "Alphabet", "MRK": "Merck", "NVDA": "NVIDIA", "PFE": "Pfizer",
                 "JNJ": "Johnson & Johnson", "MU": "Micron", "FDX": "FedEx", "PCG": "PG&E", "LLY": "Eli Lilly"}

HELDOUT_QUESTIONS = {
    "Energy": [("change", "the U.S. national average retail price of regular gasoline compared with "
                          "the previous week")],
    **{f"FNSPID:{t}": [("change", f"the share price of {c} stock")] for t, c in COMPANY_NAMES.items()},
}

HELDOUT_SUBJECTS = {
    "Energy": "the U.S. national average retail price of regular gasoline",
    **{f"FNSPID:{t}": f"the share price of {c} stock" for t, c in COMPANY_NAMES.items()},
}


def domain_key(domain: str, series: str) -> str:
    """The judge / forecast key of a held-out window: ``Energy`` or ``FNSPID:<TICKER>``."""
    return domain if domain == "Energy" else f"{domain}:{series}"


def register() -> None:
    """Extend v1's QUESTIONS and v2's SUBJECTS in place (held-out runs only; no file is edited)."""
    from src.textguard import judge_prompt, judge_prompt_v2
    for k, v in HELDOUT_QUESTIONS.items():
        if judge_prompt.QUESTIONS.get(k, v) != v:
            raise ValueError(f"{k}: a different v1 question is already registered")
    judge_prompt.QUESTIONS.update(HELDOUT_QUESTIONS)
    judge_prompt_v2.SUBJECTS.update(HELDOUT_SUBJECTS)
