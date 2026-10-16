---
name: audit-report
description: Spec for TextGuard's Audit Report (P7.1 Auditor) — the six sections a model provider gets back, plus the viva framing ("crash test, not a speed test"). Use when building auditor.py or the report generator, or when explaining what a model provider receives.
---

# Audit Report contents (answers "what does a model provider get?")

(Moved from CLAUDE.md §11.)

The provider supplies their model (`predict(numbers, text)`) + their dataset. We return:

1. **Health check** — traffic-light table: uses text? fooled? notices irrelevant text?
2. **Fault breakdown** — scores split by domain, forecast horizon, and attack type
3. **Worst failure cases** — top 10 samples with real vs fake text, prediction before/after, true value (reproducible bug reports)
4. **Diagnosis + fixes** — each fault mapped to a recommended correction
5. **With-TextGuard comparison** — before/after table including False Block Rate
6. **Drift monitor setup** — reference centroid + alert threshold for their data

**Key framing for the viva:** a developer only sees accuracy on their own clean test data.
That cannot reveal text collapse, leakage, or adversarial vulnerability, because those
situations never appear in the test set. This is a crash test, not a speed test.
