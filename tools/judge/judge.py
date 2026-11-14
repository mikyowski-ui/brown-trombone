"""The intelligibility judge.

    uv run python judge.py score human espeak classic physical
    uv run python judge.py score physical --profile out/profile-a.json --name physical-a
    uv run python judge.py score physical --sets sentence_heldout --profile best.json --name best-heldout
    uv run python judge.py report                 # rebuild report/judge.md from out/score-*.json
    uv run python judge.py items physical --set vcv   # per-item reference vs heard
    uv run python judge.py vowels physical        # probes (see probes.py): vowels, voicing, confusions
    uv run python judge.py voicing espeak
    uv run python judge.py confusions classic --profile p.json --name classic-p
    uv run python judge.py natural physical espeak --listener rec.wav   # naturalness (see natural.py)
    uv run python judge.py natural-report human espeak physical

For every rendered item it measures

* **PER** -- phoneme error rate.  ``facebook/wav2vec2-lv-60-espeak-cv-ft``
  (wav2vec 2.0 + CTC, eSpeak IPA output) transcribes the audio, greedy CTC
  decoding; the tokens are folded to the engine's ARPAbet alphabet by
  ``arpabet.py`` and aligned against the expected phonemes (edits / reference
  phonemes, summed over a set).
* **CTC log-likelihood** of the reference, in nats per reference phoneme
  (higher is better).  Each reference phoneme may be emitted as *any* model
  token that folds to it (``ɑː``, ``ɑ``, ``a`` ... for AA), so this is the log
  probability of everything the normalizer would accept; r-coloured pairs are
  tried merged (``ɑːɹ``) and split (``ɑː ɹ``) and the better is kept.  A
  smooth signal: it moves before the argmax flips.
* **confusions** -- substitutions, deletions (``-``) and insertions from the
  alignments.
* **word accuracy** with Whisper ``small.en`` (reference words found in the
  transcript / reference words, after Whisper's English text normalizer).
  Items without a written form (the VCVs) have no word score.
* for the engine, **PER vs intended**: the same alignment against the
  phonemes the engine says it spoke (first stdout line of ``say``), which
  separates the engine's own text-to-phoneme mistakes from acoustic ones.

Results go to ``out/score-<name>.json``; ``report/judge.md`` compares every
scored system.  The human anchors must score under 15 % PER (the gate).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("HF_HOME", str(HERE / ".cache" / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
if (HERE / ".cache" / "hf" / "hub" / "models--facebook--wav2vec2-lv-60-espeak-cv-ft" / "snapshots").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # downloaded: no network at all
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402

import arpabet as A  # noqa: E402
import material  # noqa: E402
import render  # noqa: E402

PHONEME_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
WHISPER_MODEL = "small.en"
WHISPER_DIR = HERE / ".cache" / "whisper"
OUT_DIR = HERE / "out"
REPORT_PATH = HERE / "report" / "judge.md"
GATE_PER = 0.15
ANCHOR_ORDER = ["human", "espeak", "classic", "physical"]
SET_ORDER = material.SYNTH_SETS + ["human"]
SET_LABEL = {"human": "librispeech"}
LL_FLOOR = -20.0  # nats/phoneme for a reference the model cannot align at all

#: ``score --fast``: the ten held-out sentences and the four listening
#: sentences (each alone, then all four as the one clip the owner hears).
FAST_SET = "sentence_heldout"
BENCH = [
    ("The birch canoe slid on the smooth planks.",
     "DH AX B ER CH K AX N UW S L IH D AO N DH AX S M UW DH P L AE NG K S"),
    ("Glue the sheet to the dark blue background.",
     "G L UW DH AX SH IY T T AX DH AX D AA R K B L UW B AE K G R AW N D"),
    ("She sells sea shells by the sea shore.",
     "SH IY S EH L Z S IY SH EH L Z B AY DH AX S IY SH AO R"),
    ("The juice of lemons makes fine punch.",
     "DH AX JH UW S AH V L EH M AX N Z M EY K S F AY N P AH N CH"),
]  # fmt: skip


def default_sets(system: str) -> list[str]:
    """Human recordings exist only for the LibriSpeech sentences; the
    synthesizers say everything, including those same sentences."""
    return ["human"] if system == "human" else list(SET_ORDER)


def device() -> str:
    import torch

    if os.environ.get("JUDGE_DEVICE"):
        return os.environ["JUDGE_DEVICE"]
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# Phoneme judge (wav2vec2 CTC)
# --------------------------------------------------------------------------


def build_unit_ids(vocab: dict[str, int]) -> dict[tuple[str, ...], list[int]]:
    """Unit (tuple of scoring symbols) -> ids of every token that folds to it.

    The flap is accepted for T and D, and T and D for the flap."""
    by_fold: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for tok, idx in vocab.items():
        if tok not in A.SPECIAL_TOKENS:
            by_fold[A.fold_token(tok)].append(idx)
    unit_ids: dict[tuple[str, ...], list[int]] = dict(by_fold)
    flap = by_fold.get((A.FLAP,), [])
    for s in ("T", "D"):
        unit_ids[(s,)] = by_fold.get((s,), []) + flap
    unit_ids[(A.FLAP,)] = flap + by_fold.get(("T",), []) + by_fold.get(("D",), [])
    return unit_ids


def scoring_phones(phones) -> list[str]:
    """Stress removed, only scoring symbols kept."""
    return [p for p in (A.strip_stress(x) for x in phones) if p in A.SCORING_SYMBOLS]


def units_of(phones: list[str], merge: bool, unit_ids: dict) -> list[tuple[str, ...]]:
    """Phonemes -> CTC units; with ``merge`` the r-coloured / syllabic pairs
    the model writes as one token (``AA R`` = ``ɑːɹ``) become one unit."""
    units: list[tuple[str, ...]] = []
    i = 0
    while i < len(phones):
        pair = tuple(phones[i : i + 2])
        if merge and len(pair) == 2 and pair in A.MERGED_TOKENS and pair in unit_ids:
            units.append(pair)
            i += 2
        else:
            units.append((phones[i],))
            i += 1
    return units


def class_matrix(lp, classes: list[tuple[str, ...]], unit_ids: dict, blank: int):
    """T x (1 + K) log-probabilities: the blank, then each unit class
    (log-sum over every model token that folds to it)."""
    import torch

    cols = [lp[:, blank]]
    for u in classes:
        ids = unit_ids.get(u) or unit_ids[(A.OTHER,)]
        cols.append(torch.logsumexp(lp[:, ids], dim=-1))
    return torch.stack(cols, dim=-1)


def sequences_logprob(lp, seqs: list[list[str]], unit_ids: dict, blank: int) -> list[float]:
    """CTC log P(string | audio), in nats (not normalised), for each phoneme
    string in ``seqs``, every accepted spelling of each phoneme pooled; the
    r-coloured pairs are tried merged and split and the better is kept.
    All strings are scored in one batched ``ctc_loss`` call.  ``-inf`` when
    the audio is too short for a string."""
    import torch

    variants: list[tuple[int, tuple]] = []
    for k, phones in enumerate(seqs):
        seen = set()
        for merge in (True, False):
            units = tuple(units_of(list(phones), merge, unit_ids))
            if units and units not in seen:
                seen.add(units)
                variants.append((k, units))
    best = [-math.inf] * len(seqs)
    if not variants:
        return best
    classes = sorted({u for _, units in variants for u in units})
    index = {u: c + 1 for c, u in enumerate(classes)}
    small = class_matrix(lp, classes, unit_ids, blank)  # T x (1 + K)
    n, t = len(variants), small.shape[0]
    width = max(len(units) for _, units in variants)
    target = torch.zeros((n, width), dtype=torch.long)
    for row, (_, units) in enumerate(variants):
        target[row, : len(units)] = torch.tensor([index[u] for u in units], dtype=torch.long)
    loss = torch.nn.functional.ctc_loss(
        small[:, None, :].expand(t, n, small.shape[1]),
        target,
        torch.full((n,), t, dtype=torch.long),
        torch.tensor([len(units) for _, units in variants], dtype=torch.long),
        blank=0,
        reduction="none",
        zero_infinity=False,
    )
    for (k, _), value in zip(variants, loss.tolist()):
        best[k] = max(best[k], -value)
    return best



@dataclass
class PhonemeResult:
    tokens: list[str]  # decoded model tokens
    heard: list[str]  # folded to scoring symbols
    ll: float  # reference log-likelihood, nats per reference phoneme


class PhonemeJudge:
    """wav2vec2 phoneme recognizer + the normalizer.  Load once, reuse."""

    def __init__(self, dev: str | None = None):
        import torch
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

        self.torch = torch
        self.device = dev or device()
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = Wav2Vec2ForCTC.from_pretrained(PHONEME_MODEL, dtype=dtype, low_cpu_mem_usage=True)
        self.model.to(self.device).eval()
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(PHONEME_MODEL)
        vocab = A.load_vocab()
        if self.model.config.vocab_size != len(vocab):
            raise RuntimeError("data/w2v_vocab.json does not match the model's vocabulary")
        self.id2tok = {i: t for t, i in vocab.items()}
        self.blank = vocab[A.BLANK_TOKEN]
        self.special = {vocab[t] for t in A.SPECIAL_TOKENS}
        self.unit_ids = build_unit_ids(vocab)

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()

    # -- acoustics ---------------------------------------------------------

    def log_probs(self, audios: list[np.ndarray], batch_size: int = 8, max_batch_seconds: float = 60.0):
        """Per-item (frames x vocab) log-probabilities, float32 CPU tensors."""
        torch = self.torch
        order = sorted(range(len(audios)), key=lambda k: len(audios[k]))
        out: list = [None] * len(audios)
        batch: list[int] = []

        def flush():
            if not batch:
                return
            feats = self.fe(
                [audios[k] for k in batch], sampling_rate=16_000, padding=True, return_tensors="pt", return_attention_mask=True
            )
            x = feats.input_values.to(self.device, dtype=self.model.dtype)
            mask = feats.attention_mask.to(self.device)
            with torch.inference_mode():
                logits = self.model(x, attention_mask=mask).logits
                lp = torch.log_softmax(logits.float(), dim=-1)
                lengths = self.model._get_feat_extract_output_lengths(mask.sum(-1)).tolist()
            for row, k in enumerate(batch):
                out[k] = lp[row, : int(lengths[row])].cpu()
            batch.clear()

        for k in order:
            if batch and (len(batch) >= batch_size or len(audios[k]) * (len(batch) + 1) > max_batch_seconds * 16_000):
                flush()
            batch.append(k)
        flush()
        return out

    def decode(self, lp) -> list[str]:
        ids = lp.argmax(-1).tolist()
        toks, prev = [], None
        for i in ids:
            if i != prev and i not in self.special:
                toks.append(self.id2tok[i])
            prev = i
        return toks

    def _units(self, phones: list[str], merge: bool) -> list[tuple[str, ...]]:
        return units_of(phones, merge, self.unit_ids)

    def sequences_logprob(self, lp, seqs: list[list[str]]) -> list[float]:
        """Total log P(phoneme string | audio) for each string (see
        :func:`sequences_logprob`)."""
        return sequences_logprob(lp, seqs, self.unit_ids, self.blank)

    def ref_loglik(self, lp, expected: list[str]) -> float:
        """log P(reference | audio) per reference phoneme, any accepted spelling."""
        phones = scoring_phones(expected)
        if not phones:
            return 0.0
        best = self.sequences_logprob(lp, [phones])[0]
        if not math.isfinite(best):
            return LL_FLOOR
        return max(LL_FLOOR, best / len(phones))

    def judge(self, audios: list[np.ndarray], expected: list[list[str]], batch_size: int = 8) -> list[PhonemeResult]:
        lps = self.log_probs(audios, batch_size=batch_size)
        out = []
        for lp, ref in zip(lps, expected):
            toks = self.decode(lp)
            out.append(PhonemeResult(toks, A.fold_tokens(toks), self.ref_loglik(lp, ref)))
        return out


# --------------------------------------------------------------------------
# Word judge (Whisper)
# --------------------------------------------------------------------------


class WordJudge:
    def __init__(self, dev: str | None = None):
        import torch
        import whisper
        from whisper.normalizers import EnglishTextNormalizer

        self.torch, self.whisper = torch, whisper
        self.device = dev or device()
        self.model = whisper.load_model(WHISPER_MODEL, device=self.device, download_root=str(WHISPER_DIR))
        self.normalize = EnglishTextNormalizer()

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()

    def transcribe(self, audios: list[np.ndarray], batch_size: int = 8) -> list[str]:
        w, torch = self.whisper, self.torch
        opts = w.DecodingOptions(language="en", without_timestamps=True, fp16=self.device.startswith("cuda"))
        out: list[str] = []
        for k in range(0, len(audios), batch_size):
            mels = [
                w.log_mel_spectrogram(w.pad_or_trim(torch.from_numpy(a)), n_mels=self.model.dims.n_mels)
                for a in audios[k : k + batch_size]
            ]
            with torch.inference_mode():
                res = w.decode(self.model, torch.stack(mels).to(self.device), opts)
            out.extend(r.text.strip() for r in res)
        return out

    def words(self, text: str) -> list[str]:
        return self.normalize(text).split()


def word_hits(ref: list[str], hyp: list[str]) -> tuple[int, int]:
    """(reference words matched, word edits) by Levenshtein alignment."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]), d[i - 1][j] + 1, d[i][j - 1] + 1)
    hits, i, j = 0, n, m
    while i > 0 and j > 0:
        if d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            hits += ref[i - 1] == hyp[j - 1]
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return hits, d[n][m]


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


@dataclass
class ItemScore:
    id: str
    set: str
    ref: list[str]
    heard: list[str]
    tokens: list[str]
    edits: int
    ll: float
    intended: list[str] | None = None
    edits_intended: int | None = None
    ortho: str = ""
    whisper: str | None = None
    word_hits: int | None = None
    word_edits: int | None = None
    word_n: int | None = None
    ops: list = field(default_factory=list)
    inserted: int | None = None

    def to_json(self) -> dict:
        d = {
            "id": self.id,
            "set": self.set,
            "ref": " ".join(self.ref),
            "heard": " ".join(self.heard),
            "tokens": " ".join(self.tokens),
            "per": round(self.edits / max(1, len(self.ref)), 4),
            "ll": round(self.ll, 4),
        }
        if self.intended is not None:
            d["intended"] = " ".join(self.intended)
            d["per_intended"] = round((self.edits_intended or 0) / max(1, len(self.intended)), 4)
        if self.inserted is not None:
            d["inserted"] = self.inserted
        if self.whisper is not None:
            d["ortho"] = self.ortho
            d["whisper"] = self.whisper
            d["word_acc"] = round((self.word_hits or 0) / max(1, self.word_n or 0), 4)
        return d


def score_phonemes(judge: PhonemeJudge, rendered: list[render.Rendered], batch_size: int = 8) -> list[ItemScore]:
    audios = [render.load_audio(r.path) for r in rendered]
    results = judge.judge(audios, [list(r.item.expected) for r in rendered], batch_size=batch_size)
    out = []
    for r, res in zip(rendered, results):
        ref = A.canonical(list(r.item.expected))
        heard = A.canonical(res.heard)
        ops = A.align(ref, heard)
        s = ItemScore(
            r.item.id,
            r.item.set,
            ref,
            heard,
            res.tokens,
            sum(1 for o in ops if o[0] != "ok"),
            res.ll,
            ortho=r.item.ortho,
            ops=ops,
        )
        if r.engine_phones is not None:
            s.intended = A.canonical(r.engine_phones)
            s.edits_intended = A.edit_distance(s.intended, heard)
        s.inserted = inserted_segments(s.intended or ref, heard)
        out.append(s)
    return out


def inserted_segments(said: list[str], heard: list[str]) -> int:
    """Segments heard that were never asked for: the insertions of the
    minimum-edit alignment of what was heard against what the speaker
    meant to say (the engine's own phonemes when it reports them, so its
    text front end is not counted; the reference otherwise).  Movements
    between sounds that are heard as sounds of their own show up here."""
    return sum(1 for op, _, _ in A.align(said, heard) if op == "ins")


def score_words(wj: WordJudge, rendered: list[render.Rendered], scores: list[ItemScore], batch_size: int = 8) -> None:
    idx = [k for k, r in enumerate(rendered) if r.item.ortho]
    if not idx:
        return
    texts = wj.transcribe([render.load_audio(rendered[k].path) for k in idx], batch_size=batch_size)
    for k, text in zip(idx, texts):
        ref = wj.words(rendered[k].item.ortho)
        hits, edits = word_hits(ref, wj.words(text))
        s = scores[k]
        s.whisper, s.word_hits, s.word_edits, s.word_n = text, hits, edits, len(ref)


def summarize_set(scores: list[ItemScore]) -> dict:
    n_ph = sum(len(s.ref) for s in scores)
    d = {
        "items": len(scores),
        "phonemes": n_ph,
        "per": sum(s.edits for s in scores) / max(1, n_ph),
        "ll": sum(s.ll * len(s.ref) for s in scores) / max(1, n_ph),
    }
    ins = [s.inserted for s in scores if s.inserted is not None]
    if ins:
        d["inserted_per_item"] = sum(ins) / len(ins)
    eng = [s for s in scores if s.intended is not None]
    if eng:
        d["per_intended"] = sum(s.edits_intended or 0 for s in eng) / max(1, sum(len(s.intended or []) for s in eng))
    wsc = [s for s in scores if s.word_n]
    if wsc:
        nw = sum(s.word_n or 0 for s in wsc)
        d["word_acc"] = sum(s.word_hits or 0 for s in wsc) / max(1, nw)
        d["wer"] = sum(s.word_edits or 0 for s in wsc) / max(1, nw)
    return d


def confusions(scores: list[ItemScore], top: int = 20) -> dict:
    pairs: Counter = Counter()
    per_ph: dict[str, Counter] = defaultdict(Counter)
    n_ref: Counter = Counter()
    for s in scores:
        for op, r, h in s.ops:
            if r is not None:
                n_ref[r] += 1
            if op == "ok":
                continue
            rr, hh = r or "-", h or "-"
            pairs[(rr, hh)] += 1
            if r is not None:
                per_ph[r][hh] += 1
    by_ph = {}
    for p in sorted(n_ref, key=lambda p: -sum(per_ph[p].values()) / n_ref[p]):
        err = sum(per_ph[p].values())
        by_ph[p] = {"n": n_ref[p], "err": err, "rate": round(err / n_ref[p], 4), "as": dict(per_ph[p].most_common(4))}
    return {
        "pairs": [[r, h, n] for (r, h), n in pairs.most_common(top)],
        "insertions": [[h, n] for (r, h), n in pairs.most_common() if r == "-"][:10],
        "by_phoneme": by_ph,
    }


def score(
    systems: list[str],
    sets: list[str] | None = None,
    profile: Path | None = None,
    name: str | None = None,
    whisper: bool = True,
    jobs: int | None = None,
    voice: str = render.DEFAULT_VOICE,
) -> dict[str, dict]:
    """Render, judge and write ``out/score-<name>.json`` for each system."""
    t0 = time.perf_counter()
    plan = []
    for system in systems:
        ss = sets or default_sets(system)
        if system == "human":
            ss = [s for s in ss if s == "human"]
            if not ss:
                raise SystemExit("the human system only has the 'human' set")
        items = material.load(ss)
        rendered = render.render(items, system, profile if system in render.ENGINE_SYSTEMS else None, voice, jobs)
        label = name if (name and len(systems) == 1) else (
            f"{system}-{Path(profile).stem}" if profile and system in render.ENGINE_SYSTEMS else system
        )
        plan.append((system, label, ss, rendered))
        fresh = sum(1 for r in rendered if not r.cached)
        print(f"[render] {label}: {len(rendered)} items ({fresh} rendered)", flush=True)
    t_render = time.perf_counter() - t0

    pj = PhonemeJudge()
    all_scores = {}
    for system, label, ss, rendered in plan:
        all_scores[label] = score_phonemes(pj, rendered)
        print(f"[phoneme] {label} done", flush=True)
    pj.close()
    del pj
    if whisper:
        wj = WordJudge()
        for system, label, ss, rendered in plan:
            score_words(wj, rendered, all_scores[label])
            print(f"[whisper] {label} done", flush=True)
        wj.close()
        del wj

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for system, label, ss, rendered in plan:
        scores = all_scores[label]
        by_set = {s: summarize_set([x for x in scores if x.set == s]) for s in ss}
        res = {
            "name": label,
            "system": system,
            "profile": str(profile) if profile and system in render.ENGINE_SYSTEMS else None,
            "voice": voice if system in render.ENGINE_SYSTEMS else None,
            "phoneme_model": PHONEME_MODEL,
            "whisper_model": WHISPER_MODEL if whisper else None,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sets": by_set,
            "overall": summarize_set(scores),
            "confusions": confusions(scores),
            "confusions_by_set": {s: confusions([x for x in scores if x.set == s], top=8)["pairs"] for s in ss},
            "items": [x.to_json() for x in scores],
        }
        path = OUT_DIR / f"score-{label}.json"
        path.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        results[label] = res
        print_summary(res)
        print(f"-> {path.relative_to(HERE)}")
    write_report()
    print(f"-> {REPORT_PATH.relative_to(HERE)}  ({time.perf_counter() - t0:.0f} s, render {t_render:.0f} s)")
    return results


def bench_items() -> list[material.Item]:
    """The four listening sentences (set ``bench``) and the four as one
    clip (set ``bench_clip``)."""
    items = [material.Item(f"bench_{k}", "bench", text, tuple(ph.split()), text) for k, (text, ph) in enumerate(BENCH, 1)]
    whole = " ".join(text for text, _ in BENCH)
    phones = tuple(" ".join(ph for _, ph in BENCH).split())
    return items + [material.Item("bench_all", "bench_clip", whole, phones, whole)]


def fast(
    system: str,
    profile: Path | None = None,
    name: str | None = None,
    jobs: int | None = None,
    voice: str = render.DEFAULT_VOICE,
) -> dict:
    """The quick check between changes: the ten held-out sentences and the
    four listening sentences, every clip in one pass through each model.
    Writes ``out/fast-<name>.json`` and the four-sentence clip as
    ``out/fast-<name>.wav``; prints the change against the last run of the
    same name.  Leaves the scoreboard (``out/score-*.json``, the report)
    alone."""
    t0 = time.perf_counter()
    engine = system in render.ENGINE_SYSTEMS
    items = material.load([FAST_SET]) + bench_items()
    rendered = render.render(items, system, profile if engine else None, voice, jobs)
    label = name or (f"{system}-{Path(profile).stem}" if profile and engine else system)
    t_render = time.perf_counter() - t0
    pj = PhonemeJudge()
    scores = score_phonemes(pj, rendered, batch_size=len(rendered))
    pj.close()
    del pj
    wj = WordJudge()
    score_words(wj, rendered, scores, batch_size=len(rendered))
    wj.close()
    del wj

    sets = [FAST_SET, "bench", "bench_clip"]
    res = {
        "name": label,
        "system": system,
        "profile": str(profile) if profile and engine else None,
        "voice": voice if engine else None,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sets": {s: summarize_set([x for x in scores if x.set == s]) for s in sets},
        "items": [x.to_json() for x in scores],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"fast-{label}.json"
    before = json.loads(path.read_text(encoding="utf-8"))["sets"] if path.exists() else {}
    path.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    clip = OUT_DIR / f"fast-{label}.wav"
    shutil.copyfile(rendered[-1].path, clip)

    def change(now: float, was: float | None) -> str:
        return "" if was is None else f" ({100 * (now - was):+.1f})"

    print(f"== {label} (fast)")
    titles = {FAST_SET: "held-out sentences (10)", "bench": "listening sentences (4)", "bench_clip": "the four as one clip"}
    for s in sets:
        d, b = res["sets"][s], before.get(s, {})
        print(
            f"  {titles[s]:24} PER {fmt_pct(d['per']):>5}%{change(d['per'], b.get('per'))}"
            f"   words {fmt_pct(d['word_acc']):>5}%{change(d['word_acc'], b.get('word_acc'))}"
        )
    for x in scores:
        if x.set != FAST_SET:
            print(f"  {x.id:10} {x.word_hits}/{x.word_n}  {x.whisper}")
    print(f"-> {path.relative_to(HERE)}, {clip.relative_to(HERE)}  ({time.perf_counter() - t0:.0f} s, render {t_render:.0f} s)")
    return res


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def fmt_pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def print_summary(res: dict) -> None:
    print(f"== {res['name']}")
    for s, d in res["sets"].items():
        wa = fmt_pct(d.get("word_acc"))
        pi = f"  per_intended {fmt_pct(d['per_intended'])}%" if "per_intended" in d else ""
        ins = f"  inserted {d['inserted_per_item']:.2f}/item" if "inserted_per_item" in d else ""
        print(f"  {SET_LABEL.get(s, s):17} PER {fmt_pct(d['per']):>5}%  ll {d['ll']:6.2f}  words {wa:>5}%{pi}{ins}")
    top = ", ".join(f"{r}→{h} {n}" for r, h, n in res["confusions"]["pairs"][:10])
    print(f"  top confusions: {top}")


def load_scores() -> dict[str, dict]:
    out = {}
    for p in sorted(OUT_DIR.glob("score-*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        out[d["name"]] = d
    order = {n: k for k, n in enumerate(ANCHOR_ORDER)}
    return dict(sorted(out.items(), key=lambda kv: (order.get(kv[0], 99), kv[0])))


def write_report(path: Path = REPORT_PATH) -> None:
    scores = load_scores()
    if not scores:
        return
    sets = [s for s in SET_ORDER if any(s in d["sets"] for d in scores.values())]
    head = "| system | " + " | ".join(SET_LABEL.get(s, s) for s in sets) + " |"
    rule = "|---|" + "---:|" * len(sets)

    def table(metric: str, f) -> list[str]:
        rows = [head, rule]
        for name, d in scores.items():
            cells = [f(d["sets"][s].get(metric)) if s in d["sets"] else "" for s in sets]
            if all(c in ("", "–") for c in cells):
                continue
            rows.append(f"| {name} | " + " | ".join(cells) + " |")
        return rows

    L = [
        "# Intelligibility judge",
        "",
        "Generated by `tools/judge/judge.py` (`uv run python judge.py report`); do not edit by hand.",
        "",
        f"Phoneme judge `{PHONEME_MODEL}` (greedy CTC, eSpeak IPA folded to ARPAbet by `arpabet.py`); "
        f"word judge Whisper `{WHISPER_MODEL}`. `librispeech` is the 16 LibriSpeech dev-clean sentences: "
        "recorded speech for `human`, synthesized from the transcript for every other system.",
        "",
    ]
    human = scores.get("human", {}).get("sets", {}).get("human")
    if human:
        ok = human["per"] < GATE_PER
        L += [
            f"**Gate: human anchors PER {fmt_pct(human['per'])}% "
            f"({'PASS' if ok else 'FAIL'}, threshold {100 * GATE_PER:.0f}%).**",
            "",
        ]
    L += ["## Phoneme error rate (%, lower is better)", ""] + table("per", fmt_pct)
    L += ["", "## Whisper word accuracy (%, higher is better)", ""] + table("word_acc", fmt_pct)
    L += ["", "## Reference CTC log-likelihood (nats per phoneme, higher is better)", ""]
    L += table("ll", lambda x: "–" if x is None else f"{x:.2f}")
    L += [
        "",
        "## Engine: PER against the phonemes the engine says it spoke (%)",
        "",
        "Same audio, but the reference is the engine's own phoneme string, so the engine's",
        "text-to-phoneme mistakes on plain-text sentences do not count.",
        "",
    ]
    L += table("per_intended", fmt_pct)
    L += [
        "",
        "## Inserted segments per item (lower is better)",
        "",
        "Insertions in the alignment of what was heard against what was meant (the engine's own",
        "phonemes when it reports them): movements between sounds heard as sounds of their own.",
        "",
    ]
    L += table("inserted_per_item", lambda x: "–" if x is None else f"{x:.2f}")
    L += ["", "## Top confusions", "", "`X→Y n`: reference X heard as Y n times (`-` = deleted / inserted).", ""]
    for name, d in scores.items():
        c = d["confusions"]
        pairs = ", ".join(f"{r}→{h} {n}" for r, h, n in c["pairs"][:12])
        worst = [
            f"{p} {fmt_pct(v['rate'])}% (as {', '.join(f'{h} {n}' for h, n in v['as'].items())})"
            for p, v in [kv for kv in c["by_phoneme"].items() if kv[1]["n"] >= 3][:8]
        ]
        L += [f"**{name}** — {pairs}", "", f"worst phonemes: {'; '.join(worst)}", ""]
    L += ["## Runs", "", "| name | system | profile | scored |", "|---|---|---|---|"]
    for name, d in scores.items():
        L.append(f"| {name} | {d['system']} | {d.get('profile') or ''} | {d['created']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def show_items(name: str, set_name: str | None) -> None:
    d = json.loads((OUT_DIR / f"score-{name}.json").read_text(encoding="utf-8"))
    for it in d["items"]:
        if set_name and it["set"] != set_name:
            continue
        extra = f"  | whisper: {it['whisper']}" if it.get("whisper") is not None else ""
        print(f"{it['id']:28} PER {100 * it['per']:5.1f}  ll {it['ll']:6.2f}{extra}")
        print(f"    ref   {it['ref']}")
        if "intended" in it and it["intended"] != it["ref"]:
            print(f"    said  {it['intended']}")
        print(f"    heard {it['heard']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("score", help="render + judge one or more systems")
    sp.add_argument("systems", nargs="+", choices=render.SYSTEMS)
    sp.add_argument("--sets", default=None, help="comma-separated material sets (default: all that apply)")
    sp.add_argument("--profile", type=Path, default=None, help="engine profile JSON")
    sp.add_argument("--name", default=None, help="result name (default: system, or system-<profile stem>)")
    sp.add_argument("--voice", default=render.DEFAULT_VOICE)
    sp.add_argument("--no-whisper", action="store_true", help="skip the word judge")
    sp.add_argument("--jobs", type=int, default=None, help="render threads (default: all cores)")
    sp.add_argument("--fast", action="store_true",
                    help="quick check: the ten held-out sentences and the four listening sentences only")  # fmt: skip
    sub.add_parser("report", help="rebuild report/judge.md from out/score-*.json")
    si = sub.add_parser("items", help="per-item reference vs heard for a scored system")
    si.add_argument("name")
    si.add_argument("--set", default=None)
    probe_help = {
        "vowels": "vowel error rate, PER, per-vowel confusions and margins (material: vowel)",
        "voicing": "voicing minimal pairs: forced choice and free recognition (material: voicing)",
        "confusions": "consonant confusions by place, manner and voicing (material: vcv)",
    }
    for metric, text in probe_help.items():
        pp = sub.add_parser(metric, help=text)
        pp.add_argument("system", choices=render.SYSTEMS)
        pp.add_argument("--profile", type=Path, default=None, help="engine profile JSON")
        pp.add_argument("--name", default=None, help="result name (default: system or system-<profile stem>)")
        pp.add_argument("--set", default=None, help="material set (default: the probe's own; 'human' = in-sentence)")
        pp.add_argument("--cut", action="store_true", help="human system: tokens cut out of the anchor recordings")
        pp.add_argument("--voice", default=render.DEFAULT_VOICE)
        pp.add_argument("--jobs", type=int, default=None, help="render threads (default: all cores)")
        pp.add_argument("--fresh", action="store_true", help="render into a scratch folder, no cache (timing)")
        pp.add_argument("--batch", type=int, default=8, help="GPU batch size")
    pr = sub.add_parser("probe-report", help="one Markdown page from out/<probe>-<name>.json")
    pr.add_argument("names", nargs="+")
    pr.add_argument("--refs", default="", help="comma-separated reference runs (ceilings) shown too")
    pr.add_argument("--scores", default="", help="comma-separated out/score-<name>.json for the sentence table")
    pr.add_argument("--out", type=Path, default=HERE / "report" / "probes.md")
    pr.add_argument("--notes", type=Path, default=None, help="Markdown inserted after the title")
    pn = sub.add_parser("natural", help="naturalness against the human recordings (see natural.py)")
    pn.add_argument("systems", nargs="+", choices=render.SYSTEMS)
    pn.add_argument("--profile", type=Path, default=None, help="engine profile JSON")
    pn.add_argument("--name", default=None, help="result name for the engine system (default: system)")
    pn.add_argument("--voice", default=render.DEFAULT_VOICE)
    pn.add_argument("--listener", type=Path, default=None,
                    help="a recording of the listener saying the two reference sentences (read, never copied)")
    pn.add_argument("--jobs", type=int, default=None, help="render threads (default: all cores)")
    pnr = sub.add_parser("natural-report", help="report/natural.md from out/natural-<name>.json")
    pnr.add_argument("names", nargs="+")
    pnr.add_argument("--out", type=Path, default=HERE / "report" / "natural.md")
    args = ap.parse_args(argv)
    if args.cmd == "natural":
        import natural

        natural.run(args.systems, args.profile, args.name, args.voice, args.listener, args.jobs)
    elif args.cmd == "natural-report":
        import natural

        natural.report(args.names, args.out)
        print(f"wrote {args.out}")
    elif args.cmd in probe_help:
        import probes

        probes.run(args.cmd, args.system, args.profile, args.name, args.set, args.cut, args.voice, args.jobs,
                   args.fresh, args.batch)  # fmt: skip
    elif args.cmd == "probe-report":
        import probes

        split = lambda s: [x for x in s.split(",") if x]  # noqa: E731
        probes.probe_report(args.names, args.out, split(args.refs), split(args.scores), args.notes)
    elif args.cmd == "score":
        sets = args.sets.split(",") if args.sets else None
        if args.fast:
            if len(args.systems) != 1 or sets or args.no_whisper:
                ap.error("--fast scores one system on its own sets, with Whisper")
            fast(args.systems[0], args.profile, args.name, args.jobs, args.voice)
        else:
            score(args.systems, sets, args.profile, args.name, not args.no_whisper, args.jobs, args.voice)
    elif args.cmd == "report":
        write_report()
        print(f"wrote {REPORT_PATH}")
    else:
        show_items(args.name, args.set)
    return 0


if __name__ == "__main__":
    sys.exit(main())
