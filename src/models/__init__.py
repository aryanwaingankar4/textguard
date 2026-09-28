"""Model wrappers (P2): every model exposes the same `predict(numbers, text)` interface.

That single uniform interface is what lets TextGuard and the Auditor treat any model as a
black box, so it must be identical across all wrappers.
"""
