# Ethics

## The risk

This project writes fake versions of real reports to test forecasters: reversing the direction
(P3 flip), reversing only where the defence won't notice (R3 A1), adding a sentence about next
period (A2), or inflating a change (×3, ×10). Someone could use the same ideas against a real
forecaster, for example one that reads market news. Below is why we think releasing it is
reasonable, and what we hold back.

## Why we release it

1. **The fakes are simple and visible.** Each is a small rule-based edit of a real report: a few
   direction words and the numbers tied to them. No hidden characters, no invisible text, no
   LLM-written fakes (R5 was not run). Anyone reading the text sees the claim. We used rules so the
   edits can be checked and so Qwen is not tested on Qwen-written text, not to make the attack
   strong. A chatbot can already do more.
2. **Text attacks on price models are already published** (Rizvani et al., 2026; *Market Signal
   Injection*, 2026). What we add is a way to measure the push (DSS) and a defence people can run
   on their own models. The toolkit is an audit of *your* model.
3. **The public API is for testing and defending.** `textguard.audit` and `TextGuardV1/V2` measure
   and defend. The only built-in attack is the one-sentence A2, worded so it fits any domain,
   because deployers need it to check the weakness we found. We don't offer fake-writing rules for
   new text styles; users supply their own test fakes.
4. **The failures are published too.** `MODEL_CARD.md` lists them (A2 beats v1; v2 blocks 22% of
   real headline bundles; no gain on Qwen2.5-7B), so nobody trusts the defence more than they
   should.

## Released and not released

| Released | Not released |
|---|---|
| Code for attacks, checks, defences, metrics, analyses and the toolkit (MIT) | The raw datasets: Time-MMD (no licence stated) and FNSPID (CC BY-NC 4.0) are fetched from their owners by `src/data/download.py` |
| Model and judge answers (numbers or answer words; texts only as fingerprints) | Generated fake texts (`texts/`, gitignored); they are rebuilt locally by the rules and every row has `synthetic=True` |
| 150 short hand-labelled report and headline excerpts used for checks (credited in `DATA_CARD.md`) | The R4 GPU input file with FNSPID headline bundles and fakes in the current tree (removed in R7, rebuilt by `export_r4_inputs`) |
| In git history only (`801c3f6` … `d93f98a`): that R4 input file. Its FNSPID content is shared under CC BY-NC 4.0, non-commercial only. History is kept as is (Aryan, 2026-10-23) because rewriting it would change the commit hashes that date the pre-registrations | |
| A synthetic example series (`examples/sample_series.csv`, marked `synthetic`) | Model checkpoints (rebuildable on a CPU) |

## Labels

Every generated fake has `synthetic`, `attacked`, `real_direction` and `claimed_direction`
columns; `texts/r4/README.txt` says the synthetic rows are fake. The old R4 input file in git
history has no `synthetic` column: its fakes are the forecast rows with `attacked = True` and the
judge rows whose `sources` don't include `real` (`DATA_CARD.md` §1a). Audit outputs name the attack
(`a2`, `user`) on every row. Never publish a fake or feed it to a live system outside a controlled
test.

## Data and people

The data are public statistical reports (U.S. series summarised by Time-MMD) and public news
headlines about listed companies (FNSPID / Benzinga). No personal data. No study with human
participants (R5's plausibility study was not run). The labels were written by the author.

## Licences

FNSPID (Dong et al., KDD 2024) is CC BY-NC 4.0. This is non-commercial academic work and FNSPID is
credited. That licence allows non-commercial sharing with credit, which covers the FNSPID extracts
in the repository and its history (`DATA_CARD.md` §1a); they are non-commercial only. Time-MMD states
no licence, so we cite it and don't share its raw files. Check both terms for your own use.
