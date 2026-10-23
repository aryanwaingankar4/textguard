"""R7: files whose sha256 a GPU run (or a pre-registration) recorded are written with CRLF on every OS.

The recorded hashes come from Windows, where pandas writes CRLF; on Linux / macOS pandas writes LF
(``os.linesep``), so without an explicit ``lineterminator`` every check of these files would fail
there. ``os.linesep`` is patched to "\\n" to simulate Linux on any machine.
"""

import inspect
import os

import pandas as pd

from src.attacks import heldout
from src.models import export_qwen_inputs, export_r3_inputs, export_r4_inputs
from src.textguard import export_v2_inputs

DF = pd.DataFrame({"a": [1], "text": ["line one\r\nline two"]})   # report texts hold CRLF (DATA_CARD)
WINDOWS_BYTES = b'a,text\r\n1,"line one\r\nline two"\r\n'


def test_simulated_linux_writes_lf_without_the_fix(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "linesep", "\n")
    DF.to_csv(tmp_path / "x.csv", index=False)
    assert (tmp_path / "x.csv").read_bytes() == b'a,text\n1,"line one\r\nline two"\n'


def test_r3_and_r4_exports_write_windows_bytes_on_linux(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "linesep", "\n")
    for mod in (export_r3_inputs, export_r4_inputs):
        monkeypatch.setattr(mod, "build", lambda: DF)
        out = tmp_path / f"{mod.__name__}.csv"
        mod.export(path=out)
        assert out.read_bytes() == WINDOWS_BYTES, mod.__name__


def test_every_recorded_hash_writer_pins_crlf():
    """The five writers of hash-checked files: R2 / R3 / R4 Kaggle inputs and R4's pre-registered texts."""
    for fn in (export_qwen_inputs.export_conditions, export_r3_inputs.export, export_r4_inputs.export,
               export_v2_inputs.export, heldout.main):
        assert 'lineterminator="\\r\\n"' in inspect.getsource(fn), fn.__qualname__
