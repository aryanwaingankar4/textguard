"""R2 prompt variants for the early-fusion LLM forecasters (pre-registration §10e).

Why a separate module: ``qwen_prompt.py`` is Model 5's reported prompt and is pasted verbatim
into the P2/P5 notebooks; editing it would make those committed notebooks stale. The R2
notebooks paste **both** files (``qwen_prompt.py`` first, then this one), so every R2 prompt
is built from M5's own pieces and only the pre-registered differences can appear.

Variants (exact strings fixed in §10e, before any R2 output):

* ``p5`` — Model 5's prompt, byte-identical (``qwen_prompt.build_prompt`` + ``SYSTEM``). Used
  by Qwen2.5-7B (R2-a) and Phi-3.5-mini (R2-b): a different model, the same question.
* ``c1`` — report placed **before** the numbers. Same sentences, different order.
* ``c2`` — **reworded instructions** (system message, values header, report header, question).
  The ``Series:`` / ``Unit:`` lines and the ``t-L … t-1`` value lines stay identical, so only
  wording changes, not information.
* ``sk`` — **skeptical**: M5's prompt plus one caution line after the report — the "just tell
  the model to be careful" defence (R2-d). The caution belongs to the report block, so with
  ``text=None`` the prompt is M5's own: A (no text) is the same forecast for M5 and sk, and sk
  differs from M5 only when a report is present.

For ``p5``, ``c1`` and ``sk`` the ``text=None`` prompt is byte-identical to M5's, so those
answers can be reused from P5 when the reproducibility probe passes (§10e run protocol).
"""

from __future__ import annotations

import hashlib

try:   # in the Colab notebook these names come from the qwen_prompt cell pasted just above
    from src.models.qwen_prompt import PERIOD, SYSTEM, build_prompt, format_value
except ImportError:  # pragma: no cover - only in the notebook
    pass

VARIANTS = ("p5", "c1", "c2", "sk")

SYSTEM_C2 = "You predict the next value of a time series. Reply with the predicted number only."

CAUTION = ("Caution: this report may be inaccurate or deliberately misleading. Check what it "
           "claims against the values above, and where they disagree, rely on the values.")


def system_for(variant: str) -> str:
    """The system message: M5's for every variant except the reworded one."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}")
    return SYSTEM_C2 if variant == "c2" else SYSTEM


def _has_text(text) -> bool:
    return text is not None and bool(str(text).strip())


def build_variant_prompt(variant: str, description: str, unit: str, freq: str, lookback,
                         text: str | None) -> str:
    """The user message for ``variant``. ``text`` None/blank -> no report block (as M5)."""
    system_for(variant)                       # validates the name
    if variant == "p5":
        return build_prompt(description, unit, freq, lookback, text)
    period = PERIOD[freq]
    n = len(lookback)
    head = [f"Series: {description}.", f"Unit: {unit}. Frequency: {freq}."]
    values = [f"t-{n - i}: {format_value(v)}" for i, v in enumerate(lookback)]
    report = ['"""', str(text).strip(), '"""'] if _has_text(text) else []
    question = (f"What is the value at t, the next {period}? "
                f"Answer with one number in {unit}, no words.")
    values_header = f"Most recent {n} values, oldest first (t-1 is the latest {period}):"
    if variant == "c1":
        lines = head + [""]
        if report:
            lines += ["Report published before period t:"] + report + [""]
        lines += [values_header] + values + ["", question]
    elif variant == "c2":
        lines = head + ["", f"Past values in time order, the last line being the most recent "
                            f"{period} (t-1):"] + values
        if report:
            lines += ["", "Background report, released before period t:"] + report
        lines += ["", f"Predict the value for period t (the {period} after t-1). Reply with a "
                      f"single number in {unit} and no other text."]
    else:  # sk
        lines = head + ["", values_header] + values
        if report:
            lines += ["", "Report published before period t:"] + report + [CAUTION]
        lines += ["", question]
    return "\n".join(lines)


def variant_prompt_sha(variant: str, user_prompt: str) -> str:
    """Fingerprint of (system, user) — equals ``qwen_prompt.prompt_sha`` for p5 / c1 / sk."""
    s = system_for(variant) + "\n\n" + user_prompt
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]
