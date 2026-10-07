"""Import MM-TSFlib's network code without its experiment pipeline.

Why only the networks: MM-TSFlib's data loader splits each CSV by row position, and its
Economy CSV is unsorted (2024 first, 1989 last), so its "test set" is not the future
(issue #7). We reuse the *models* (``models/DLinear.py``, ``models/PatchTST.py``) so our
baselines are the same architectures as the published ones, and keep *our* chronological
splits and scaler.

MM-TSFlib is cloned into ``external/MM-TSFlib`` (gitignored); its modules import each
other as top-level packages (``layers``, ``utils``, ``models``), so its root goes on
``sys.path``.

``layers/SelfAttention_Family.py`` imports ``reformer_pytorch`` at module level, only for
the Reformer model, which we never use. Installing it would pull in extra packages and may
try to change our pinned torch, so a stub module stands in when the real one is absent.
"""

from __future__ import annotations

import importlib
import sys
import types
from argparse import Namespace
from pathlib import Path

MMTSFLIB_ROOT = Path(__file__).resolve().parents[2] / "external" / "MM-TSFlib"


def available() -> bool:
    return (MMTSFLIB_ROOT / "models" / "PatchTST.py").exists()


def _ensure_importable() -> None:
    if not available():
        raise FileNotFoundError(
            f"MM-TSFlib not found at {MMTSFLIB_ROOT}; clone "
            "https://github.com/AdityaLab/MM-TSFlib into external/MM-TSFlib")
    root = str(MMTSFLIB_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    if "reformer_pytorch" not in sys.modules:
        try:
            importlib.import_module("reformer_pytorch")
        except ImportError:
            stub = types.ModuleType("reformer_pytorch")

            class LSHSelfAttention:  # only Reformer uses it
                def __init__(self, *a, **k):
                    raise ImportError("reformer_pytorch is stubbed; Reformer is not supported")

            stub.LSHSelfAttention = LSHSelfAttention
            sys.modules["reformer_pytorch"] = stub


def model_module(name: str):
    """MM-TSFlib's ``models.<name>`` module, e.g. ``model_module("PatchTST")``."""
    _ensure_importable()
    return importlib.import_module(f"models.{name}")


def model_class(name: str):
    """MM-TSFlib's ``models.<name>.Model`` class, e.g. ``model_class("DLinear")``."""
    return model_module(name).Model


def default_configs(seq_len: int, pred_len: int, **overrides) -> Namespace:
    """MM-TSFlib ``run.py`` defaults, univariate (``features='S'``, ``enc_in=1``).

    ``run.py`` forces ``features='S'`` / ``enc_in=1`` for every Time-MMD run, so these
    are the settings behind the published numbers. Only the fields the two networks read
    are included.
    """
    cfg = dict(task_name="long_term_forecast", seq_len=seq_len, label_len=seq_len // 2,
               pred_len=pred_len, enc_in=1, dec_in=1, c_out=1, d_model=512, n_heads=8,
               e_layers=2, d_layers=1, d_ff=2048, moving_avg=25, factor=1, dropout=0.1,
               activation="gelu", output_attention=False, embed="timeF", freq="h")
    cfg.update(overrides)
    return Namespace(**cfg)
