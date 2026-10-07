"""Model 4 — PatchTST + frozen BERT, MM-TSFlib-style late fusion.

The implementation lives in ``late_fusion.py`` (shared with Model 3). PatchTST uses
``patching="window"`` by default — the backbone of Model 2, so it reads all 8 inputs at h=1
instead of the 2 that MM-TSFlib's patch_len clamp leaves (PROJECT_STATUS §7).
"""

from src.models.late_fusion import PatchTSTBertForecaster

__all__ = ["PatchTSTBertForecaster"]
