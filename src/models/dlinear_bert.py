"""Model 3 — DLinear + frozen BERT, MM-TSFlib-style late fusion.

The implementation lives in ``late_fusion.py`` (shared with Model 4, so the two differ only
in their backbone). Read that module's docstring for the fusion formula, what was kept from
and changed relative to MM-TSFlib, and what ``text=None`` means.
"""

from src.models.late_fusion import DLinearBertForecaster

__all__ = ["DLinearBertForecaster"]
