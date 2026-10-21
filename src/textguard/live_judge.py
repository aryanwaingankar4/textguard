"""The direction judge, run live (R7 toolkit): Qwen2.5-1.5B scores the allowed answers.

Everywhere in our experiments the judge ran once on Colab / Kaggle and its answers were
served from a CSV (``text_direction.QwenJudge``, ``v2.JudgeV2``) — so every result can be
re-derived without a GPU. An outside user has new texts, so the judge must run on them. This
module is the notebooks' judge code (``notebooks/build_p6_judge_notebook.py`` cell 5),
unchanged in what it computes, packaged as a class:

* same model and weights: ``Qwen/Qwen2.5-1.5B-Instruct`` @ ``989aa79`` (P5's revision);
* same prompts: v1's ``judge_prompt`` and v2's ``judge_prompt_v2``, with the user's series
  registered as a new subject (:func:`register_subject` — the R4 held-out pattern; the frozen
  prompt files are not edited);
* same answer rule: log P(answer | prompt) by teacher forcing for each of the three allowed
  answers, softmax over the three, take the most likely; confidence = its probability.

**Hardware.** fp16 on a CUDA GPU is what the paper used (a free Colab / Kaggle T4 judges ~3–5
questions per second). On a CPU it runs in bfloat16 — usable for a few dozen texts, not for a
real audit — and its answers can differ from the GPU's in rare near-ties.

Known judge weaknesses (R4, §9): on neutral financial headlines it reads a direction into
about 1 in 3 (11 of 33 hand-labelled neutral FNSPID headlines), which is what makes v2 block
18–22% of genuine headline bundles. It never reversed a direction on the labelled set.

Answers are cached per prompt fingerprint, and :meth:`QwenScorer.answers` returns them in the
notebooks' column layout, so a run can be saved and re-loaded like any of ours.
"""

from __future__ import annotations

import math

import pandas as pd

from src.textguard import judge_prompt as JP1
from src.textguard import judge_prompt_v2 as JP2
from src.textguard.text_direction import Claim, stated_baseline
from src.textguard.units import frame, units
from src.textguard.v2 import ClaimV2, consolidate, level_reference, questions_for

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"   # = Model 5 and every judge run


def register_subject(key: str, subject: str, frame_v1: str = "") -> None:
    """Teach both judge prompts a new series: ``key`` is its name, ``subject`` what it is.

    v1 writes the claim's frame into its question (Traffic: "compared with the same month one
    year earlier"); pass it as ``frame_v1`` (e.g. "compared with the previous week") or leave it
    empty. v2 names only the quantity and reads the frame from each sentence. Registering the
    same key twice with a different subject is refused (one key = one question)."""
    q1 = [("change", f"{subject} {frame_v1}".strip())]
    if JP1.QUESTIONS.get(key, q1) != q1 or JP2.SUBJECTS.get(key, subject) != subject:
        raise ValueError(f"{key!r} is already registered with a different subject")
    JP1.QUESTIONS[key] = q1
    JP2.SUBJECTS[key] = subject


class QwenScorer:
    """Scores the three allowed answers of a judge prompt; caches by prompt fingerprint."""

    def __init__(self, device: str | None = None, model_id: str = MODEL_ID,
                 revision: str = REVISION):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.startswith("cuda") else torch.bfloat16
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        kw = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"   # renamed in 4.56
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
                                                          **{kw: dtype}).to(self.device).eval()
        self.model_id, self.revision = model_id, revision
        self._ids: dict[str, list[int]] = {}
        self._cache: dict[str, dict] = {}

    def _chat_ids(self, system: str, user_prompt: str):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        if not self.torch.is_tensor(ids):      # transformers >= 5 returns a BatchEncoding
            ids = ids["input_ids"]
        return ids.to(self.device)

    def _answer_ids(self, a: str) -> list[int]:
        if a not in self._ids:
            self._ids[a] = self.tok(a, add_special_tokens=False).input_ids
        return self._ids[a]

    def _logprobs(self, system: str, user_prompt: str, answers) -> list[float]:
        torch = self.torch
        with torch.no_grad():
            ids = self._chat_ids(system, user_prompt)
            if all(len(self._answer_ids(a)) == 1 for a in answers):
                lsm = torch.log_softmax(self.model(ids).logits[0, -1].float(), -1)
                return [lsm[self._answer_ids(a)[0]].item() for a in answers]
            out = []
            for a in answers:
                a_ids = torch.tensor([self._answer_ids(a)], device=self.device)
                logits = self.model(torch.cat([ids, a_ids], 1)).logits[0].float()
                lsm = torch.log_softmax(logits[ids.shape[1] - 1:-1], -1)
                out.append(lsm.gather(1, a_ids[0][:, None]).sum().item())
            return out

    def ask(self, system: str, user_prompt: str, sha: str, answers) -> tuple[str, float]:
        """(answer, confidence) — the notebooks' rule, ties to the earlier answer."""
        if sha not in self._cache:
            lp = self._logprobs(system, user_prompt, answers)
            m = max(lp)
            p = [math.exp(x - m) for x in lp]
            p = [x / sum(p) for x in p]
            best = max(range(3), key=lambda i: (p[i], -i))
            self._cache[sha] = dict(prompt_sha=sha, answer=answers[best], confidence=p[best],
                                    p_pos=p[0], p_neg=p[1], p_none=p[2],
                                    setup=f"{self.device} | {MODEL_ID}@{self.revision}")
        r = self._cache[sha]
        return r["answer"], r["confidence"]

    def answers(self) -> pd.DataFrame:
        """Every answer so far, one row per distinct prompt."""
        return pd.DataFrame(list(self._cache.values()))


def _literal(answers, answer: str) -> int:
    pos, neg, _ = answers
    return 1 if answer == pos else (-1 if answer == neg else 0)


class LiveJudgeV1:
    """TextGuard v1's judge on new text: one question about the whole report (change claim)."""

    def __init__(self, key: str, scorer: QwenScorer, polarity: int = 1):
        if key not in JP1.QUESTIONS:
            raise KeyError(f"{key!r}: call register_subject first")
        self.key, self.scorer, self.polarity = key, scorer, int(polarity)

    def __call__(self, text: str | None) -> list[Claim]:
        if not isinstance(text, str) or not text.strip():
            return []
        claims = []
        for kind, _ in JP1.QUESTIONS[self.key]:
            up = JP1.build_judge_prompt(self.key, kind, text)
            ans = JP1.ANSWERS[kind]
            a, c = self.scorer.ask(JP1.JUDGE_SYSTEM, up, JP1.judge_sha(up), ans)
            d = _literal(ans, a) * (1 if kind == "level" else self.polarity)
            claims.append(Claim(kind, d, c, stated_baseline(text) if kind == "level" else None))
        return claims


class LiveJudgeV2:
    """TextGuard v2's judge on new text: every unit, change + outlook (+ level), then V3 rules."""

    def __init__(self, key: str, scorer: QwenScorer, polarity: int = 1,
                 consolidated: bool = True):
        if key not in JP2.SUBJECTS:
            raise KeyError(f"{key!r}: call register_subject first")
        self.key, self.scorer = key, scorer
        self.polarity, self.consolidated = int(polarity), consolidated

    def __call__(self, text: str | None) -> list[ClaimV2]:
        out = []
        for u in units(text):
            for kind in questions_for(u):
                up = JP2.build_judge_prompt(self.key, kind, u)
                ans = JP2.ANSWERS[kind]
                a, c = self.scorer.ask(JP2.JUDGE_SYSTEM, up, JP2.judge_sha(up), ans)
                lit = _literal(ans, a)
                d = lit if kind == "level" else lit * self.polarity
                out.append(ClaimV2(kind, d, c, frame(u) if kind == "change" else "trend",
                                   level_reference(u) if kind == "level" else None, u))
        return consolidate(out) if self.consolidated else out


def judges(key: str, subject: str, frame_v1: str = "", device: str | None = None,
           polarity: int = 1, scorer: QwenScorer | None = None):
    """Register ``subject`` and return (judge_v1, judge_v2) sharing one loaded model."""
    register_subject(key, subject, frame_v1)
    scorer = scorer or QwenScorer(device)
    return LiveJudgeV1(key, scorer, polarity), LiveJudgeV2(key, scorer, polarity)


__all__ = ["MODEL_ID", "REVISION", "register_subject", "QwenScorer", "LiveJudgeV1",
           "LiveJudgeV2", "judges"]
