# TextGuard

**Detecting and Defending Against Misleading Text Attacks on Multimodal Time-Series Forecasting Models**

Aryan Manoj Waingankar · B.Tech AI & ML, Symbiosis Institute of Technology, Pune

---

## What this is

Multimodal time-series forecasters combine numerical history with aligned text reports to
predict future values. Where these models genuinely consume text, that text is an
**unverified input channel** — an attack surface.

This repository provides:

- **An attack suite** — semantically misleading (direction-flipped) and exaggerated text,
  generated from any dataset's own reports
- **DSS (Directional Susceptibility Score)** — a metric for how far a forecast is pulled
  *toward the attacker's claimed direction*, which MSE alone cannot capture
- **TextGuard** — a **training-free, inference-time, black-box** defense. It runs any
  forecaster twice (with and without text) and admits the text's effect only in proportion
  to how well the text agrees with the numerical trend
- **An Auditor** — plug in any model and get a vulnerability report

> ⚠️ **Status: under active development.** Nothing here is validated yet.

---

## Quick start

```bash
git clone https://github.com/<your-username>/textguard.git
cd textguard
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Using it on your own model

Wrap your forecaster in one function:

```python
class MyModel:
    def predict(self, numbers, text):
        ...
        return prediction
```

Audit it:

```python
from textguard import Auditor
report = Auditor(model=MyModel(), domain="energy").run()
```

Defend it:

```python
from textguard import TextGuard
guard = TextGuard(model=MyModel())
prediction, gate, warning = guard.predict(numbers, text)
```

---

## Repo layout

| Path | Contents |
|---|---|
| `CLAUDE.md` | Full project context — read this first |
| `PHASES.md` | Phase-gated work plan |
| `PROJECT_STATUS.md` | Living progress log |
| `src/` | Source: models, attacks, textguard, metrics |
| `notebooks/` | Colab GPU runs |
| `results/` | Raw predictions + aggregated metrics |
| `paper/` | LaTeX draft and figures |

---

## Datasets

- **Time-MMD** (Liu et al., NeurIPS 2024) — primary benchmark
- **FNSPID** (Dong et al., 2024) — optional secondary validation.
  Used under **CC BY-NC-4.0, academic research only**

---

## Ethics

The misleading text in this repository is generated **solely for defense research**. All
generated adversarial text is clearly labelled as such. Do not deploy or distribute it as
genuine reporting.

---

## License

Code: MIT. Datasets retain their original licenses (see above).
