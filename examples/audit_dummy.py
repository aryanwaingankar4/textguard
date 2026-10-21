"""Worked example: audit the DummyForecaster on the synthetic sample series, with and without TextGuard.

    python examples/audit_dummy.py                    # no judge: undefended / never_trust / clip only
    python examples/audit_dummy.py --judge qwen       # + TextGuard v1 and v2 (Qwen2.5-1.5B judge;
                                                      #   GPU if available, else CPU — slow, ~10 min)

The DummyForecaster follows the direction words in a report ("rose" / "fell"), so it is fooled by
lies by construction — a stand-in for "your model". Replace it with any object that has
``horizon`` and ``predict(numbers, text)``. Output: ``examples/audit_out/`` (CSV, JSON, markdown).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from textguard import DummyForecaster, audit, judges, load_series  # noqa: E402

DATA = ROOT / "examples" / "sample_series.csv"
SUBJECT = "the average widget price"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--judge", choices=["none", "qwen"], default="none")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--out", type=Path, default=ROOT / "examples" / "audit_out")
    a = ap.parse_args()
    data = load_series(DATA, lookback=12, freq="weekly")
    model = DummyForecaster(sensitivity=1.0, step=data.sigma / 4, lookback=12)
    j1 = j2 = None
    if a.judge == "qwen":
        j1, j2 = judges("widgets", SUBJECT, frame_v1="compared with the previous week",
                        device=a.device)
    rep = audit(model, data, judge_v1=j1, judge_v2=j2, subject=SUBJECT)
    out = rep.save(a.out)
    print(rep.to_markdown())
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
