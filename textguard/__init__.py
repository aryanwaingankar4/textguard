"""TextGuard — public API (R7). Import it from the repository root (it re-exports ``src/``).

    from textguard import TextGuardV1, TextGuardV2, audit, load_series, judges

* ``TextGuardV1`` / ``TextGuardV2`` — wrap any object with ``horizon`` and
  ``predict(numbers, text)``; frozen paper settings by default.
* ``judges(key, subject)`` — the Qwen2.5-1.5B direction judges (v1, v2), run live; GPU strongly
  recommended (``src/textguard/live_judge.py``).
* ``load_series(csv, lookback, ...)`` + ``audit(model, data, ...)`` — DSS, flip rate, False
  Block Rate, real-text MSE and worst cases for undefended / never_trust / clip / v1 / v2.

The data format, the attacks and the limits are documented in ``src/textguard/toolkit.py`` and
in the README section "Use TextGuard on your own model". The implementation lives in
``src/`` — this package only re-exports it, so the toolkit and the paper share one code path.
"""

from src.models.base import BaseForecaster, Forecaster
from src.models.dummy import DummyForecaster
from src.textguard.live_judge import (LiveJudgeV1, LiveJudgeV2, QwenScorer, judges,
                                      register_subject)
from src.textguard.text_direction import Claim, fixed_judge, judge_from
from src.textguard.toolkit import (AuditReport, SeriesData, TextGuardV1, TextGuardV2, a2_attack,
                                   a2_sentence, audit, load_series, v1_window, v2_settings)

__all__ = ["Forecaster", "BaseForecaster", "DummyForecaster", "TextGuardV1", "TextGuardV2",
           "audit", "AuditReport", "load_series", "SeriesData", "a2_attack", "a2_sentence",
           "judges", "QwenScorer", "LiveJudgeV1", "LiveJudgeV2", "register_subject", "Claim",
           "fixed_judge", "judge_from", "v1_window", "v2_settings"]
