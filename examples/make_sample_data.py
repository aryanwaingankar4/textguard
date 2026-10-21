"""Write ``examples/sample_series.csv`` — a tiny SYNTHETIC weekly series with short reports.

Nothing here is real: a random-walk "widget price" with a mild trend, and one-line reports that
state last week's change ("The average widget price rose 1.2% from the previous week."), some
neutral ("Trading was quiet ahead of the holiday."). The test rows also carry hand-written-style
**fake** reports (direction reversed) with their claimed direction, to show the ``fake_text`` /
``fake_direction`` columns. Fixed seed, so the file is identical on every run.

Run: ``python examples/make_sample_data.py``. Labelled synthetic in every row (``synthetic``).
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).with_name("sample_series.csv")
NEUTRAL = ["Trading was quiet ahead of the holiday.", "Retailers reported normal inventories.",
           "The weekly survey was published on schedule."]


def main() -> Path:
    rng = np.random.default_rng(7)
    n = 104
    dates = pd.date_range("2022-01-03", periods=n, freq="W-MON")
    steps = 0.15 * np.sin(np.arange(n) / 6.0) + rng.normal(0, 0.12, n)
    value = np.round(10 + np.cumsum(steps), 3)
    rows = []
    for t in range(n):
        split = "train" if t < 60 else ("val" if t < 78 else "test")
        text = fake = None
        fd = None
        if t >= 1 and rng.random() < 0.8:
            prev = value[t - 2] if t >= 2 else value[t - 1]
            chg = (value[t - 1] - prev) / prev * 100     # last known week: available before t
            if abs(chg) < 0.05 or rng.random() < 0.15:
                text = NEUTRAL[t % len(NEUTRAL)]
            else:
                verb, other = ("rose", "fell") if chg > 0 else ("fell", "rose")
                text = f"The average widget price {verb} {abs(chg):.1f}% from the previous week."
                if split == "test":
                    fake = text.replace(verb, other)
                    fd = 1 if other == "rose" else -1
        rows.append(dict(date=dates[t].date().isoformat(), value=value[t], text=text, split=split,
                         fake_text=fake, fake_direction=fd, synthetic=True))
    pd.DataFrame(rows).to_csv(OUT, index=False, lineterminator="\n")
    return OUT


if __name__ == "__main__":
    print(f"-> {main()}")
