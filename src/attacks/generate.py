"""Generate every text condition for the val + test windows and save one CSV per condition.

Run: ``python -m src.attacks.generate`` → ``texts/<condition>.csv`` (gitignored) + README.
Deterministic: re-running gives identical files.
"""

from __future__ import annotations

import pandas as pd

from src.attacks import common, cross_domain, empty, exaggerated, flipped, shuffled, time_shifted
from src.attacks.rules import headline

BUILDERS = {"empty": empty.build, "shuffled": shuffled.build, "cross_domain": cross_domain.build,
            "time_shifted": time_shifted.build, "flipped": flipped.build,
            "exaggerated": exaggerated.build}


def build_all() -> dict[str, pd.DataFrame]:
    w = common.windows()
    real_dir = [headline(t, d).series_direction for t, d in zip(w["text"], w["domain"])]
    out = {"real": common.frame(w, "real", w["text"], real_direction=real_dir,
                                source_domain=w["domain"].values,
                                source_sample_id=w["sample_id"].values)}
    out.update({name: fn(w) for name, fn in BUILDERS.items()})
    return out


def main() -> None:
    for name, df in build_all().items():
        path = common.save(df, name)
        extra = f", attacked {int(df.attacked.sum())}" if name in ("flipped", "exaggerated") else ""
        print(f"{name:<13} {len(df)} rows, with text {int(df.has_text.sum())}{extra} -> {path}")


if __name__ == "__main__":
    main()
