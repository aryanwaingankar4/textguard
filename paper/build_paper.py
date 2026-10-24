"""Build the TextGuard paper as an IEEE-style Word document: paper/TextGuard_paper.docx.

Run with the paper venv (python-docx; nothing else from the project is imported):
    .venv-paper\\Scripts\\python paper\\build_paper.py

How numbers get into the paper:
* **Tables** read their cells from ``results/summary_*.csv`` / ``*.json`` at build time through
  ``V()``; each cell is logged in ``paper/build/provenance.json`` (file, row filter, column, value,
  text written), which ``check_numbers.py`` re-verifies independently.
* **Prose** numbers are written by hand from PROJECT_STATUS §9 or the same tables;
  ``check_numbers.py`` extracts every number from the finished .docx and finds its source.

The content lives in ``content_front.py`` (title to Section IV), ``content_results.py``
(Sections V-VI) and ``content_back.py`` (Section VII to the appendices). Citations are written
``[@key]`` and numbered in order of first use, IEEE style.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
FIG = HERE / "figures"
OUT = HERE / "TextGuard_paper.docx"
BUILD = HERE / "build"

sys.path.insert(0, str(HERE))
from ieee_docx import IEEEDoc  # noqa: E402

MINUS = "−"
PROVENANCE: list[dict] = []
_CSV: dict[str, list[dict]] = {}


# ---- reading the summaries ----------------------------------------------------------------
def rows(name: str) -> list[dict]:
    if name not in _CSV:
        with open(RES / name, newline="", encoding="utf-8") as fh:
            _CSV[name] = list(csv.DictReader(fh))
    return _CSV[name]


def pick(name: str, **flt) -> dict:
    hits = [r for r in rows(name) if all(r.get(k) == v for k, v in flt.items())]
    if len(hits) != 1:
        raise KeyError(f"{name} {flt}: {len(hits)} rows")
    return hits[0]


def jget(name: str, path: str):
    with open(RES / name, encoding="utf-8") as fh:
        obj = json.load(fh)
    for key in path.split("/"):
        obj = obj[key]
    return obj


def signed(x: float, d: int = 3) -> str:
    s = f"{x:+.{d}f}"
    if float(s) == 0:
        s = f"{0:.{d}f}"
    return s.replace("-", MINUS)


def plain(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".replace("-", MINUS)


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%".replace("-", MINUS)


def V(name: str, col: str, fmt=signed, **flt) -> str:
    """One table cell from a summary CSV (logged for check_numbers.py)."""
    raw = pick(name, **flt)[col]
    value = float(raw)
    text = fmt(value)
    PROVENANCE.append({"file": name, "filter": flt, "column": col, "value": value,
                       "text": text, "fmt": fmt.__name__})
    return text


def J(name: str, path: str, fmt=signed) -> str:
    value = float(jget(name, path))
    text = fmt(value)
    PROVENANCE.append({"file": name, "json_path": path, "value": value, "text": text,
                       "fmt": fmt.__name__})
    return text


def ci(name: str, lo: str, hi: str, d: int = 3, **flt) -> str:
    return f"[{V(name, lo, lambda x: signed(x, d), **flt)}, {V(name, hi, lambda x: signed(x, d), **flt)}]"


# ---- references (IEEE style; numbered by first citation) -----------------------------------
REFS = {
    "timemmd": "H. Liu, S. Xu, Z. Zhao, L. Kong, H. Kamarthi, A. B. Sasanur, M. Sharma, J. Cui, "
               "Q. Wen, C. Zhang, and B. A. Prakash, “Time-MMD: Multi-domain multimodal "
               "dataset for time series analysis,” in *Proc. Adv. Neural Inf. Process. Syst. "
               "(NeurIPS), Datasets and Benchmarks Track*, 2024, arXiv:2406.08627.",
    "gruver": "N. Gruver, M. Finzi, S. Qiu, and A. G. Wilson, “Large language models are "
              "zero-shot time series forecasters,” in *Proc. Adv. Neural Inf. Process. Syst. "
              "(NeurIPS)*, 2023.",
    "cik": "A. R. Williams *et al.*, “Context is key: A benchmark for forecasting with essential "
           "textual information,” in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2025, "
           "arXiv:2410.18959.",
    "parnet": "E. Zhu, Z. Deng, S. Wang, Y.-K. Tang, and C. Liu, “PA-RNet: Perturbation-aware "
              "residual network for robust multimodal time series forecasting,” arXiv:2508.04750, "
              "2025.",
    "sridhar": "K. Sridhar, A. Gupta, N. Pradhan, M. Mandal, D. Kumar, and S. Deshpande, "
               "“Semantics or structure? Auditing text sensitivity in multimodal time-series "
               "forecasting,” arXiv:2608.22321, 2026.",
    "liu_adv": "F. Liu, S. Jiang, L. Miranda-Moreno, S. Choi, and L. Sun, “Adversarial "
               "vulnerabilities in large language models for time series forecasting,” in "
               "*Proc. Int. Conf. Artif. Intell. Statist. (AISTATS)*, 2025, arXiv:2412.08099.",
    "xie": "Y. Xie, D. Wang, P.-Y. Chen, J. Xiong, S. Liu, and S. Koyejo, “A word is worth a "
           "thousand dollars: Adversarial attack on tweets fools stock predictions,” in *Proc. "
           "Conf. North Amer. Chapter Assoc. Comput. Linguistics (NAACL)*, 2022, arXiv:2205.01094.",
    "rizvani": "A. Rizvani, G. Apruzzese, and P. Laskov, “Adversarial news and lost profits: "
               "Manipulating headlines in LLM-driven algorithmic trading,” in *Proc. IEEE Conf. "
               "Secure Trustworthy Mach. Learn. (SaTML)*, 2026, arXiv:2601.13082.",
    "corroboration": "Y. Lu and Y. Zhang, “The corroboration illusion: When more news makes LLM "
                     "forecasts less true,” arXiv:2609.22246, 2026.",
    "msi": "D. Lee and H. Park, “Market signal injection: Adversarial context manipulation of LLM "
           "pricing agents,” arXiv:2609.18357, 2026.",
    "dlinear": "A. Zeng, M. Chen, L. Zhang, and Q. Xu, “Are transformers effective for time series "
               "forecasting?” in *Proc. AAAI Conf. Artif. Intell.*, 2023.",
    "patchtst": "Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, “A time series is worth 64 "
                "words: Long-term forecasting with transformers,” in *Proc. Int. Conf. Learn. "
                "Represent. (ICLR)*, 2023.",
    "collapse": "H. H. Nguyen, M. H. Nguyen, D. Nguyen, and H. Le, “Does text actually help? "
                "Uncovering and resolving text collapse in multimodal time series forecasting,” "
                "arXiv:2606.19413, 2026.",
    "cfa": "S. Lee, J. Seo, J. Lee, S. Yoo, M. Kim, T. Y. Lim, D. Kang, H. Choi, S. Lee, and W. Ahn, "
           "“Rethinking multimodal fusion for time series: Text modalities need constrained "
           "fusion,” arXiv:2603.22372, 2026.",
    "lafp": "H. H. Nguyen, D. Nguyen, M. H. Nguyen, D. Do, and H. Le, “LLM as forecasting planner: "
            "Training-free text conditioning for time-series foundation models,” "
            "arXiv:2607.24892, 2026.",
    "andrews": "E. Andrews and G. Mengaldo, “When does text inform? Benchmarking "
               "information-theoretic metrics for multimodal time-series forecasting,” "
               "arXiv:2609.11282, 2026.",
    "ma_detect": "H. Ma, R. Sun, M. Xue, X. Yuan, C. Rudolph, S. Nepal, and L. Liu, “Keep the lights "
                 "on, keep the lengths in check: Plug-in adversarial detection for time-series LLMs "
                 "in energy forecasting,” arXiv:2512.12154, 2025.",
    "greshake": "K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and M. Fritz, “Not what "
                "you’ve signed up for: Compromising real-world LLM-integrated applications with "
                "indirect prompt injection,” in *Proc. ACM Workshop Artif. Intell. Secur. "
                "(AISec)*, 2023.",
    "autoredtrader": "Z. Liu *et al.*, “AutoRedTrader: Autonomous red teaming of trading agents "
                     "through synthetic misinformation injection,” arXiv:2605.09185, 2026.",
    "bert": "J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of deep "
            "bidirectional transformers for language understanding,” in *Proc. Conf. North "
            "Amer. Chapter Assoc. Comput. Linguistics (NAACL-HLT)*, 2019.",
    "qwen25": "Qwen Team (A. Yang *et al.*), “Qwen2.5 technical report,” arXiv:2412.15115, "
              "2024.",
    "finbert": "D. Araci, “FinBERT: Financial sentiment analysis with pre-trained language "
               "models,” arXiv:1908.10063, 2019.",
    "qlora": "T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “QLoRA: Efficient "
             "finetuning of quantized LLMs,” in *Proc. Adv. Neural Inf. Process. Syst. "
             "(NeurIPS)*, 2023.",
    "phi3": "M. Abdin *et al.*, “Phi-3 technical report: A highly capable language model locally "
            "on your phone,” arXiv:2404.14219, 2024.",
    "wilcoxon": "F. Wilcoxon, “Individual comparisons by ranking methods,” *Biometrics "
                "Bulletin*, vol. 1, no. 6, pp. 80–83, 1945.",
    "mcnemar": "Q. McNemar, “Note on the sampling error of the difference between correlated "
               "proportions or percentages,” *Psychometrika*, vol. 12, no. 2, pp. 153–157, "
               "1947.",
    "holm": "S. Holm, “A simple sequentially rejective multiple test procedure,” *Scand. J. "
            "Statist.*, vol. 6, no. 2, pp. 65–70, 1979.",
    "efron": "B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. New York, NY, USA: "
             "Chapman & Hall, 1993.",
    "lakens": "D. Lakens, “Equivalence tests: A practical primer for t tests, correlations, and "
              "meta-analyses,” *Soc. Psychol. Personality Sci.*, vol. 8, no. 4, pp. 355–362, "
              "2017.",
    "nosek": "B. A. Nosek, C. R. Ebersole, A. C. DeHaven, and D. T. Mellor, “The preregistration "
             "revolution,” *Proc. Natl. Acad. Sci. USA*, vol. 115, no. 11, pp. 2600–2606, "
             "2018.",
    "cohen": "J. Cohen, “A coefficient of agreement for nominal scales,” *Educ. Psychol. "
             "Meas.*, vol. 20, no. 1, pp. 37–46, 1960.",
    "fnspid": "Z. Dong, X. Fan, and Z. Peng, “FNSPID: A comprehensive financial news dataset in "
              "time series,” arXiv:2402.06698, 2024.",
}
ORDER: list[str] = []
_CITE = re.compile(r"\[@([^\]]+)\]")


def _cites(text: str) -> str:
    def repl(m):
        nums = []
        for key in (k.strip().lstrip("@") for k in m.group(1).split(";")):
            if key not in REFS:
                raise KeyError(f"unknown citation {key}")
            if key not in ORDER:
                ORDER.append(key)
            nums.append(ORDER.index(key) + 1)
        return ", ".join(f"[{n}]" for n in sorted(nums))
    return _CITE.sub(repl, text)


# ---- figure / table labels -----------------------------------------------------------------
TABLES = ["related", "sets", "v2dev", "p5", "p6", "r1", "r2", "r3", "r4", "errors",
          "sens_p5", "sens_r3", "sens_r4", "config", "checks"]
FIGS = ["pipeline", "p5dss", "p5flip", "p6trade", "p6gate", "r2", "r3", "r4dss", "r4fbr"]


def _labels(text: str) -> str:
    text = re.sub(r"\{T:(\w+)\}", lambda m: f"Table {_roman(TABLES.index(m.group(1)) + 1)}", text)
    return re.sub(r"\{F:(\w+)\}", lambda m: f"Fig. {FIGS.index(m.group(1)) + 1}", text)


def _roman(n):
    from ieee_docx import roman
    return roman(n)


class Paper(IEEEDoc):
    """IEEEDoc whose text methods resolve citations and labels first."""

    def _t(self, s):
        return _labels(_cites(s)) if isinstance(s, str) else s

    def p(self, text, runin=None, **kw):
        return super().p(self._t(text), runin=runin, **kw)

    def item(self, label, text, **kw):
        return super().item(label, self._t(text), **kw)

    def abstract(self, label, text):
        return super().abstract(label, self._t(text))

    def table(self, key, caption, header, rows_, **kw):
        assert TABLES[self.table_no] == key, (key, TABLES[self.table_no])
        rows_ = [[self._t(c) for c in r] for r in rows_]
        return super().table(self._t(caption), [self._t(h) for h in header], rows_,
                             note=self._t(kw.pop("note", None)) if kw.get("note") else None,
                             **{k: v for k, v in kw.items() if k != "note"})

    def figure(self, key, path, caption, **kw):
        assert FIGS[self.fig_no] == key, (key, FIGS[self.fig_no])
        return super().figure(path, self._t(caption), **kw)


def main():
    import content_back
    import content_front
    import content_results

    d = Paper()
    content_front.write(d)
    content_results.write(d)
    content_back.write(d)
    d.h1("References")
    for n, key in enumerate(ORDER, 1):
        d.reference(n, REFS[key])
    unused = sorted(set(REFS) - set(ORDER))
    assert not unused, f"references never cited: {unused}"
    n_refs = len(ORDER)
    content_back.write_appendices(d)
    assert len(ORDER) == n_refs, "the appendices cite a work after the reference list was written"
    assert d.table_no == len(TABLES) and d.fig_no == len(FIGS), (d.table_no, d.fig_no)
    d.save(OUT)
    BUILD.mkdir(exist_ok=True)
    (BUILD / "provenance.json").write_text(json.dumps(PROVENANCE, indent=1, ensure_ascii=False),
                                           encoding="utf-8")
    print(f"{OUT}  tables {d.table_no}, figures {d.fig_no}, references {len(ORDER)}, "
          f"table cells logged {len(PROVENANCE)}")


if __name__ == "__main__":
    # Run through the importable module so the content files share one PROVENANCE / ORDER.
    import build_paper
    build_paper.main()
