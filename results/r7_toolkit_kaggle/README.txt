R7 toolkit test on Kaggle (2026-10-23, run by Aryan): notebooks/textguard_your_model.ipynb,
unchanged worked example = DummyForecaster on examples/sample_series.csv, real Qwen2.5-1.5B-Instruct
judge @989aa79 (live, CPU: no GPU was attached; ~47 min end to end incl. the model download).
Toolkit = dist/textguard_toolkit.zip built at 24bd16d-dirty; its 23 files are identical to the
committed ones (checked against HEAD in R7). Files are the notebook's audit_out/ folder, unedited.

SYNTHETIC DATA: the series, its reports and every fake text here (attack "a2" = genuine report +
appended reverse sentence; attack "user" = the CSV's fake_text column) are synthetic, made for
testing defences only. They describe no real product or price. Not genuine reporting.

A functional test of the toolkit, not evidence about TextGuard: the DummyForecaster follows
direction words by construction (PROJECT_STATUS.md section 9 "R7").
