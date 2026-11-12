"""Targeted probes for the vowel, voicing and consonant-place stages.

    uv run python judge.py vowels physical --profile ../../profiles/tuned-r3.json
    uv run python judge.py voicing espeak
    uv run python judge.py confusions classic
    uv run python judge.py vowels human            # in-sentence, LibriSpeech anchors
    uv run python judge.py vowels human --cut      # tokens cut out of the anchors
    uv run python judge.py probe-report r4-espeak r4-physical ... --out report/round4-baseline.md

Each probe scores one *slot* per item (or every matching slot of a
sentence): the phoneme that the stage under test is about.

* ``vowels`` (material ``vowel``: 16 vowels alone and in CVC frames).
  Vowel error rate: share of items whose vowel slot, after aligning the
  greedy transcription to the reference, is not the intended vowel (AH and AX
  count as one vowel: the model's alphabet writes the same quality ʌ when
  stressed and ə when not; ``vowel_error_merged`` also merges AA/AO, the
  cot-caught merger of much of American English).  Plain PER on the set.  A
  per-vowel confusion table.  The **margin**: CTC log P(reference) minus the
  best log P of the reference with the vowel replaced by any other vowel, in
  nats.  Positive means the intended vowel wins; it moves continuously
  before the transcription flips, so a tuner can use it
  (:class:`ProbeJudge`).
* ``voicing`` (material ``voicing``: P/B, T/D, K/G, S/Z, F/V minimal pairs
  in initial, medial and final position, and VCV syllables).  Forced
  choice: log P(intended) - log P(minimal-pair twin); accuracy = share of
  items where the intended member wins.  Free recognition: the consonant
  heard in the slot has the intended voicing (a flap counts half for T/D;
  a sonorant, vowel or deletion counts as wrong).  Every pair has both
  members, so a system that voices everything scores 50 %, not 100 %.
* ``confusions`` (material ``vcv``: every consonant between AA, IY, UW).
  The consonant heard in the slot: the full table, and tables by place,
  manner and voicing.

``--set human`` scores the same slots inside the 16 LibriSpeech sentences
(every vowel, every P B T D K G S Z F V, every consonant), for any system;
``--cut`` (the ``human`` system only) cuts tokens out of the recordings by
CTC forced alignment (a consonant-vowel-consonant excerpt around each vowel,
a phoneme-consonant-phoneme excerpt around each consonant) and scores them as
isolated items, the human counterpart of the synthetic material.
"""

from __future__ import annotations

import json
import math
import time
from functools import lru_cache
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

import arpabet as A
import judge as J
import material
import render

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"
REPORT_DIR = HERE / "report"
METRICS = ["vowels", "voicing", "confusions"]
DEFAULT_SET = {"vowels": "vowel", "voicing": "voicing", "confusions": "vcv"}
MARGIN_CAP = 30.0  # nats: a string the audio cannot carry at all

VOWELS = list(material.VOWEL_TARGETS)
#: Symbols that are one vowel for scoring (notation, not sound).
VOWEL_SAME = [{"AH", "AX"}]
#: ... and the cot-caught merger, for the lenient rate.
VOWEL_SAME_MERGED = [{"AH", "AX"}, {"AA", "AO"}]

TWIN = {}
for _vl, _vd in material.VOICING_PAIRS:
    TWIN[_vl], TWIN[_vd] = _vd, _vl
VOICELESS = {"P", "T", "K", "F", "TH", "S", "SH", "CH", "HH"}
OBSTRUENTS = set(A.STOPS + A.FRICATIVES + A.AFFRICATES + A.ASPIRATE)
HEADLINE_PAIRS = ["P/B", "T/D", "K/G", "S/Z"]  # F/V reported separately

PLACE = {
    **dict.fromkeys(["P", "B", "M", "W"], "bilabial"),
    **dict.fromkeys(["F", "V"], "labiodental"),
    **dict.fromkeys(["TH", "DH"], "dental"),
    **dict.fromkeys(["T", "D", "N", "S", "Z", "L", "DX"], "alveolar"),
    **dict.fromkeys(["SH", "ZH", "CH", "JH", "R"], "postalveolar"),
    "Y": "palatal",
    **dict.fromkeys(["K", "G", "NG"], "velar"),
    "HH": "glottal",
}
PLACE_ORDER = ["bilabial", "labiodental", "dental", "alveolar", "postalveolar", "palatal", "velar", "glottal"]
MANNER = {
    **dict.fromkeys(A.STOPS, "stop"),
    **dict.fromkeys(A.NASALS, "nasal"),
    **dict.fromkeys(A.FRICATIVES + A.ASPIRATE, "fricative"),
    **dict.fromkeys(A.AFFRICATES, "affricate"),
    **dict.fromkeys(A.APPROXIMANTS, "approximant"),
    "DX": "flap",
}
MANNER_ORDER = ["stop", "nasal", "fricative", "affricate", "approximant", "flap"]
# heard-side columns for things that are not consonants
DELETED, VOWEL_HEARD, OTHER_HEARD = "-", "vowel", "other"


def voicing_of(sym: str) -> str:
    if sym == A.FLAP:
        return "flap"
    return "voiceless" if sym in VOICELESS else "voiced"


# --------------------------------------------------------------------------
# Slot alignment
# --------------------------------------------------------------------------


def canonical_map(phones: list[str]) -> tuple[list[str], list[list[int]]]:
    """:func:`arpabet.canonical` of ``phones``, and for each output symbol
    the input indices it came from."""
    seq = [A.strip_stress(p) for p in phones]
    keep = [k for k, p in enumerate(seq) if p in A.SCORING_SYMBOLS]
    seq = [seq[k] for k in keep]
    out: list[str] = []
    src: list[list[int]] = []
    i = 0
    while i < len(seq):
        for pat, rep in A.CANONICAL_RULES:
            if tuple(seq[i : i + len(pat)]) == pat:
                for j, r in enumerate(rep):
                    out.append(r)
                    # the last output symbol takes what is left of the pattern
                    take = range(i + j, i + len(pat)) if j == len(rep) - 1 else [i + j]
                    src.append([keep[t] for t in take])
                i += len(pat)
                break
        else:
            out.append(seq[i])
            src.append([keep[i]])
            i += 1
    return out, src


def canonical_slot(phones: list[str], slot: int) -> tuple[list[str], int]:
    """:func:`arpabet.canonical` of ``phones`` and where ``slot`` ends up.

    A canonical rule that rewrites the slot together with its neighbour
    (``AX R`` -> ``ER``) moves the slot to the rewritten symbol."""
    out, src = canonical_map(phones)
    return out, next(j for j, ss in enumerate(src) if slot in ss)


def heard_with_tokens(tokens: list[str]) -> tuple[list[str], list[str]]:
    """Decoded tokens -> canonical heard symbols, and the model token each
    symbol came from (``+``-joined when a rule merged two)."""
    syms, tags = [], []
    for t in tokens:
        for f in A.fold_token(t):
            syms.append(f)
            tags.append(t)
    out, src = canonical_map(syms)
    return out, ["+".join(dict.fromkeys(tags[k] for k in ss)) for ss in src]


def slot_heard(ref: list[str], heard: list[str], slot: int) -> tuple[str, str | None]:
    """Align canonical ``ref`` to canonical ``heard`` and return
    ``(op, heard symbol)`` for reference position ``slot`` (``op`` is
    ``ok``, ``sub`` or ``del``; the symbol is ``None`` for a deletion)."""
    op, h, _ = slot_heard_at(ref, heard, slot)
    return op, h


def slot_heard_at(ref: list[str], heard: list[str], slot: int) -> tuple[str, str | None, int | None]:
    """:func:`slot_heard` plus the index of the heard symbol."""
    k, j = -1, -1
    for op, r, h in A.align(ref, heard):
        if h is not None:
            j += 1
        if r is not None:
            k += 1
            if k == slot:
                return op, h, (j if h is not None else None)
    raise IndexError(f"slot {slot} outside a reference of {len(ref)}")


def same_vowel(a: str | None, b: str | None, classes=VOWEL_SAME) -> bool:
    if a is None or b is None:
        return False
    return a == b or any(a in c and b in c for c in classes)


def heard_class(sym: str | None, table: dict | None = None) -> str:
    """Heard symbol -> column of a consonant table."""
    if sym is None:
        return DELETED
    if sym in A.VOCALIC:
        return VOWEL_HEARD
    if sym == A.OTHER:
        return OTHER_HEARD
    return table[sym] if table is not None else sym


# --------------------------------------------------------------------------
# Margins (forced choice) from CTC
# --------------------------------------------------------------------------


def _cap(x: float) -> float:
    if math.isnan(x):
        return 0.0
    return max(-MARGIN_CAP, min(MARGIN_CAP, x))


def choice_margin(lp, ref: list[str], slot: int, rivals: list[str], unit_ids: dict, blank: int):
    """log P(ref) - max over rivals r of log P(ref with ``ref[slot] = r``).

    Returns ``(margin, best rival, log P(ref))``, margin in nats, capped at
    +-``MARGIN_CAP``.  ``lp`` is a (frames x vocab) log-probability tensor."""
    rivals = [r for r in rivals if r != ref[slot]]
    seqs = [list(ref)] + [ref[:slot] + [r] + ref[slot + 1 :] for r in rivals]
    lps = J.sequences_logprob(lp, seqs, unit_ids, blank)
    own, others = lps[0], lps[1:]
    k = int(np.argmax(others))
    best = others[k]
    if own == -math.inf and best == -math.inf:
        return 0.0, rivals[k], own
    return _cap(own - best), rivals[k], own


def vowel_rivals(target: str) -> list[str]:
    """Competing vowels: every other vowel, except the ones that are the same
    vowel for scoring (AH for AX and back)."""
    return [v for v in VOWELS if not same_vowel(v, target)]


# --------------------------------------------------------------------------
# Slot records
# --------------------------------------------------------------------------


@dataclass
class Slot:
    item: str
    set: str
    ref: list[str]  # canonical reference of the whole item
    slot: int
    target: str
    heard: str | None  # symbol heard in the slot (None = deleted)
    op: str
    margin: float | None = None
    rival: str | None = None
    meta: dict = field(default_factory=dict)
    token: str | None = None  # the model token heard in the slot

    def to_json(self) -> dict:
        d = asdict(self)
        d["ref"] = " ".join(self.ref)
        if self.margin is not None:
            d["margin"] = round(self.margin, 3)
        return d


@dataclass
class ItemResult:
    id: str
    set: str
    ref: list[str]
    heard: list[str]
    tokens: list[str]
    edits: int
    ll: float
    slots: list[Slot]

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "set": self.set,
            "ref": " ".join(self.ref),
            "heard": " ".join(self.heard),
            "tokens": " ".join(self.tokens),
            "per": round(self.edits / max(1, len(self.ref)), 4),
            "ll": round(self.ll, 4),
            "slots": [s.to_json() for s in self.slots],
        }


def item_slots(item: material.Item, metric: str, sentence: bool) -> list[tuple[int, dict]]:
    """``(slot in canonical ref, meta)`` for every slot ``metric`` scores in
    ``item``: the one marked in the material, or, in sentences (``human``
    set, cut tokens), every matching phoneme."""
    ref = A.canonical(list(item.expected))
    info = item.info()
    if not sentence:
        if metric == "confusions":  # vcv set: the consonant of V C V
            if not item.id.startswith("vcv_"):
                return []
            raw = 1
            info = {"target": item.expected[1], "vowel": item.expected[0]}
        else:
            raw = int(info["slot"])
        _, slot = canonical_slot(list(item.expected), raw)
        return [(slot, info)]
    out = []
    for k, p in enumerate(ref):
        if metric == "vowels" and p in VOWELS:
            out.append((k, {"target": p, "frame": "sentence", "lex": "sentence"}))
        elif metric == "voicing" and p in TWIN:
            pair = "/".join(sorted({p, TWIN[p]}, key=lambda s: (s not in VOICELESS, s)))
            out.append((k, {"target": p, "pair": pair, "pos": "sentence", "voicing": voicing_of(p), "lex": "sentence"}))
        elif metric == "confusions" and p in A.CONSONANTS:
            out.append((k, {"target": p}))
    if item.set == "human" and metric == "vowels":
        marks = sentence_stress(item.ortho or item.text, list(item.expected))
        for k, m in out:
            if marks and len(marks) == len(ref):
                m["frame"] = "sentence, stressed" if marks[k] else "sentence, unstressed"
    if "stress" in info:  # a word cut out of a recording
        marks = info["stress"]
        for k, m in out:
            m["frame"] = m["pos"] = m["lex"] = "cut word"
            if metric == "vowels" and len(marks) == len(ref):
                m["frame"] = "cut word, stressed" if marks[k] == "1" else "cut word, unstressed"
        if metric == "confusions":  # only V C V inside the word, like the vcv set
            out = [(k, m) for k, m in out if 0 < k < len(ref) - 1 and ref[k - 1] in A.VOCALIC and ref[k + 1] in A.VOCALIC]
    return out


def score_slots(
    pj: J.PhonemeJudge,
    items: list[material.Item],
    audios: list[np.ndarray],
    metric: str,
    sentence: bool = False,
    batch_size: int = 8,
    margins: bool = True,
) -> list[ItemResult]:
    """Recognise every item once and score the slots of ``metric``."""
    lps = pj.log_probs(audios, batch_size=batch_size)
    out = []
    for item, lp in zip(items, lps):
        toks = pj.decode(lp)
        heard, heard_tok = heard_with_tokens(toks)
        ref = A.canonical(list(item.expected))
        ops = A.align(ref, heard)
        res = ItemResult(
            item.id, item.set, ref, heard, toks, sum(1 for o in ops if o[0] != "ok"), pj.ref_loglik(lp, list(item.expected)), []
        )
        for slot, info in item_slots(item, metric, sentence):
            target = ref[slot]
            op, h, j = slot_heard_at(ref, heard, slot)
            s = Slot(item.id, item.set, ref, slot, target, h, op, meta=dict(info), token=heard_tok[j] if j is not None else None)
            if margins and metric == "vowels":
                s.margin, s.rival, _ = choice_margin(lp, ref, slot, vowel_rivals(target), pj.unit_ids, pj.blank)
            elif margins and metric == "voicing":
                s.margin, s.rival, _ = choice_margin(lp, ref, slot, [TWIN[target]], pj.unit_ids, pj.blank)
            res.slots.append(s)
        out.append(res)
    return out


# --------------------------------------------------------------------------
# The API for tuners
# --------------------------------------------------------------------------


class ProbeJudge:
    """Continuous vowel and voicing scores for a tuner.  The phoneme model is
    loaded once (or pass an existing :class:`judge.PhonemeJudge`).

        pj = ProbeJudge()
        r = pj.vowel("out/x.wav", ["B", "IY", "D"])        # slot = the first vowel
        r["margin"]        # nats; > 0: the intended vowel beats every other vowel
        rs = pj.vowels([wav1, wav2], [ref1, ref2], [1, 1])   # batched on the GPU
        pj.voicing("y.wav", ["P", "AE", "T"], 0)["margin"]   # vs "B AE T"
    """

    def __init__(self, judge: J.PhonemeJudge | None = None):
        self.pj = judge or J.PhonemeJudge()

    def close(self) -> None:
        self.pj.close()

    @staticmethod
    def _audio(a) -> np.ndarray:
        return a if isinstance(a, np.ndarray) else render.load_audio(a)

    def _score(self, audios, refs, slots, rivals_of, batch_size: int) -> list[dict]:
        lps = self.pj.log_probs([self._audio(a) for a in audios], batch_size=batch_size)
        out = []
        for lp, ref, slot in zip(lps, refs, slots):
            raw = list(ref)
            if slot is None:
                slot = next(k for k, p in enumerate(J.scoring_phones(raw)) if p in A.VOCALIC)
            cref, cslot = canonical_slot(raw, slot)
            margin, rival, own = choice_margin(lp, cref, cslot, rivals_of(cref[cslot]), self.pj.unit_ids, self.pj.blank)
            heard = A.canonical(A.fold_tokens(self.pj.decode(lp)))
            op, h = slot_heard(cref, heard, cslot)
            out.append(
                {
                    "margin": margin,
                    "rival": rival,
                    "logprob": own,
                    "target": cref[cslot],
                    "heard": h,
                    "correct": same_vowel(h, cref[cslot]) if cref[cslot] in A.VOCALIC else h == cref[cslot],
                    "transcript": heard,
                }
            )
        return out

    def vowels(self, audios, refs, slots=None, batch_size: int = 8) -> list[dict]:
        """``audios``: wav paths or 16 kHz float arrays; ``refs``: phoneme
        lists (ARPAbet, stress digits allowed); ``slots``: index of the vowel
        in each ref (``None``: the first vowel)."""
        slots = slots if slots is not None else [None] * len(refs)
        return self._score(audios, refs, slots, vowel_rivals, batch_size)

    def vowel(self, audio, ref, slot: int | None = None) -> dict:
        return self.vowels([audio], [ref], [slot])[0]

    def voicing_many(self, audios, refs, slots, batch_size: int = 8) -> list[dict]:
        """Forced choice against the voicing twin of ``ref[slot]``."""
        return self._score(audios, refs, slots, lambda t: [TWIN[t]], batch_size)

    def voicing(self, audio, ref, slot: int) -> dict:
        return self.voicing_many([audio], [ref], [slot])[0]


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------


def _rate(xs) -> float | None:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def set_per(results: list[ItemResult]) -> float:
    return sum(r.edits for r in results) / max(1, sum(len(r.ref) for r in results))


def summarize_vowels(results: list[ItemResult]) -> dict:
    slots = [s for r in results for s in r.slots]
    ok = lambda s: same_vowel(s.heard, s.target)  # noqa: E731
    okm = lambda s: same_vowel(s.heard, s.target, VOWEL_SAME_MERGED)  # noqa: E731
    per_vowel = {}
    table: dict[str, Counter] = {}
    for v in VOWELS:
        ss = [s for s in slots if s.target == v]
        if not ss:
            continue
        heard = Counter(
            (s.heard if s.heard in A.VOCALIC else heard_class(s.heard)) if not ok(s) else v for s in ss
        )
        table[v] = heard
        per_vowel[v] = {
            "n": len(ss),
            "error": 1 - _rate(ok(s) for s in ss),
            "error_merged": 1 - _rate(okm(s) for s in ss),
            "margin": _mean(s.margin for s in ss),
            "won": _rate((s.margin or 0) > 0 for s in ss) if ss[0].margin is not None else None,
            "rivals": dict(Counter(s.rival for s in ss if (s.margin or 0) <= 0).most_common(3)),
            "heard_as": dict(Counter(heard_class(s.heard) if s.heard not in A.VOCALIC else s.heard
                                     for s in ss if not ok(s)).most_common(4)),
            "tokens": dict(Counter(s.token or DELETED for s in ss).most_common(5)),
        }  # fmt: skip
    by_frame = {}
    for fr in sorted({s.meta.get("frame", "") for s in slots}):
        ss = [s for s in slots if s.meta.get("frame", "") == fr]
        by_frame[fr] = {"n": len(ss), "error": 1 - _rate(ok(s) for s in ss), "margin": _mean(s.margin for s in ss)}
    pairs = Counter((s.target, s.heard or DELETED) for s in slots if not ok(s))
    # Heard as AA: the open front token a/aː (between AE and AA) or the back ɑ/ɑː?
    as_aa = Counter(s.token for s in slots if s.heard == "AA" and s.target != "AA")
    return {
        "items": len(results),
        "slots": len(slots),
        "heard_aa_tokens": dict(as_aa.most_common()),
        "vowel_error": 1 - (_rate(ok(s) for s in slots) or 0),
        "vowel_error_merged": 1 - (_rate(okm(s) for s in slots) or 0),
        "per": set_per(results),
        "margin": _mean(s.margin for s in slots),
        "won": _rate((s.margin or 0) > 0 for s in slots) if slots and slots[0].margin is not None else None,
        "per_vowel": per_vowel,
        "by_frame": by_frame,
        "worst": [[r, h, n] for (r, h), n in pairs.most_common(12)],
        "table": {v: dict(c) for v, c in table.items()},
    }


def _voicing_breakdown(slots: list[Slot], key) -> dict:
    groups: dict[str, list[Slot]] = defaultdict(list)
    for s in slots:
        groups[key(s)].append(s)
    return {g: _voicing_stats(ss) for g, ss in sorted(groups.items())}


def _free_credit(s: Slot) -> float:
    h = s.heard
    if h == A.FLAP and s.target in ("T", "D"):
        return 0.5
    if h is None or h not in OBSTRUENTS:
        return 0.0
    return 1.0 if voicing_of(h) == voicing_of(s.target) else 0.0


def _voicing_stats(ss: list[Slot]) -> dict:
    vl = [s for s in ss if voicing_of(s.target) == "voiceless"]
    vd = [s for s in ss if voicing_of(s.target) == "voiced"]
    forced = lambda xs: _rate((s.margin or 0) > 0 for s in xs)  # noqa: E731
    free = lambda xs: _rate(_free_credit(s) for s in xs)  # noqa: E731
    bal = lambda f: None if not vl or not vd else (f(vl) + f(vd)) / 2  # noqa: E731
    return {
        "n": len(ss),
        "forced": forced(ss),
        "forced_balanced": bal(forced),
        "margin": _mean(s.margin for s in ss),
        "free": free(ss),
        "free_balanced": bal(free),
        "forced_voiceless": forced(vl),
        "forced_voiced": forced(vd),
        "free_voiceless": free(vl),
        "free_voiced": free(vd),
    }


def summarize_voicing(results: list[ItemResult]) -> dict:
    slots = [s for r in results for s in r.slots]
    head = [s for s in slots if s.meta.get("pair") in HEADLINE_PAIRS]
    fv = [s for s in slots if s.meta.get("pair") == "F/V"]
    heard = Counter()
    for s in slots:
        h = s.heard
        cls = "deleted" if h is None else ("vowel" if h in A.VOCALIC else ("flap" if h == A.FLAP else
              (voicing_of(h) + (" obstruent" if h in OBSTRUENTS else " sonorant"))))  # fmt: skip
        heard[(voicing_of(s.target), cls)] += 1
    return {
        "items": len(results),
        "slots": len(slots),
        "per": set_per(results),
        "headline": _voicing_stats(head),
        "fv": _voicing_stats(fv),
        "by_pair": _voicing_breakdown(slots, lambda s: s.meta.get("pair", "")),
        "by_position": _voicing_breakdown(head, lambda s: s.meta.get("pos", "")),
        "by_pair_position": _voicing_breakdown(slots, lambda s: f"{s.meta.get('pair')} {s.meta.get('pos')}"),
        "by_target": _voicing_breakdown(slots, lambda s: s.target),
        "heard_class": {f"{a} -> {b}": n for (a, b), n in sorted(heard.items())},
        "worst": [[r, h or DELETED, n] for (r, h), n in Counter(
            (s.target, s.heard) for s in slots if s.heard != s.target).most_common(12)],
    }  # fmt: skip


def _class_table(slots: list[Slot], classes: dict | None, order: list[str] | None) -> dict:
    rows: dict[str, Counter] = defaultdict(Counter)
    for s in slots:
        r = classes[s.target] if classes else s.target
        rows[r][heard_class(s.heard, classes)] += 1
    keys = order or [c for c in A.CONSONANTS if c in rows]
    out = {}
    for r in keys:
        if r not in rows:
            continue
        n = sum(rows[r].values())
        out[r] = {"n": n, "correct": rows[r][r] / n, "heard": dict(rows[r].most_common())}
    return out


def summarize_confusions(results: list[ItemResult]) -> dict:
    slots = [s for r in results for s in r.slots]
    voi = {c: voicing_of(c) for c in A.CONSONANTS + [A.FLAP]}
    pairs = Counter((s.target, s.heard or DELETED) for s in slots if not A.same(s.target, s.heard or ""))
    return {
        "items": len(results),
        "slots": len(slots),
        "per": set_per(results),
        "accuracy": _rate(A.same(s.target, s.heard or "") for s in slots),
        "place_accuracy": _rate(PLACE.get(s.heard or "") == PLACE[s.target] for s in slots),
        "manner_accuracy": _rate(MANNER.get(s.heard or "") == MANNER[s.target] for s in slots),
        "voicing_accuracy": _rate(
            s.heard in voi and (voi[s.heard] == voi[s.target] or voi[s.heard] == "flap") for s in slots
        ),
        "full": _class_table(slots, None, None),
        "place": _class_table(slots, PLACE, PLACE_ORDER),
        "manner": _class_table(slots, MANNER, MANNER_ORDER),
        "voicing": _class_table(slots, voi, ["voiceless", "voiced"]),
        "worst": [[r, h, n] for (r, h), n in pairs.most_common(15)],
    }


SUMMARIZE = {"vowels": summarize_vowels, "voicing": summarize_voicing, "confusions": summarize_confusions}


# --------------------------------------------------------------------------
# Cut tokens from the human anchors
# --------------------------------------------------------------------------

FRAME = 320  # samples per wav2vec2 frame at 16 kHz (20 ms)


def ctc_viterbi(small: np.ndarray, labels: list[int]) -> list[tuple[int, int]]:
    """Best CTC path through ``labels`` (column indices of ``small``,
    T x C log-probabilities, blank = column 0).  Returns the first and last
    frame each label is emitted on."""
    t_n, n = small.shape[0], len(labels)
    ext = [0]
    for lab in labels:
        ext += [lab, 0]
    s_n = len(ext)
    neg = -1e30
    score = np.full(s_n, neg)
    score[0] = small[0, ext[0]]
    if s_n > 1:
        score[1] = small[0, ext[1]]
    back = np.zeros((t_n, s_n), dtype=np.int8)  # 0 stay, 1 from s-1, 2 from s-2
    ext_a = np.array(ext)
    skip_ok = np.zeros(s_n, dtype=bool)
    for s in range(2, s_n):
        skip_ok[s] = ext[s] != 0 and ext[s] != ext[s - 2]
    for t in range(1, t_n):
        stay = score
        one = np.concatenate([[neg], score[:-1]])
        two = np.where(skip_ok, np.concatenate([[neg, neg], score[:-2]]), neg)
        cand = np.stack([stay, one, two])
        choice = cand.argmax(0)
        score = cand[choice, np.arange(s_n)] + small[t, ext_a]
        back[t] = choice
    s = s_n - 1 if s_n == 1 or score[s_n - 1] >= score[s_n - 2] else s_n - 2
    path = [0] * t_n
    for t in range(t_n - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])
    spans: list[list[int]] = [[-1, -1] for _ in range(n)]
    for t, s in enumerate(path):
        if s % 2 == 1:
            j = s // 2
            if spans[j][0] < 0:
                spans[j][0] = t
            spans[j][1] = t
    return [tuple(x) for x in spans]


@lru_cache(maxsize=256)
def espeak_words(text: str) -> list[tuple[list[str], list[bool]]]:
    """eSpeak's words for ``text``: per word, its phonemes (scoring symbols)
    and whether each carries primary stress.  Concatenated, the phonemes are
    exactly :func:`espeak_util.phonemes` of ``text``."""
    import subprocess

    out = subprocess.run(["espeak-ng", "-q", "-v", "en-us", "--ipa", "--sep=_", text],
                         capture_output=True, text=True, check=True).stdout  # fmt: skip
    words: list = []
    for w in out.split():
        phones, stress, pending = [], [], False
        for tok in [t for t in w.split("_") if t]:
            pending |= "ˈ" in tok
            for f in A.fold_token(tok.replace("ˈ", "").replace("ˌ", "")):
                phones.append(f)
                stress.append(pending and f in A.VOCALIC)
                if f in A.VOCALIC:
                    pending = False
        if phones:
            words.append((phones, stress))
    return words


def sentence_stress(text: str, expected: list[str]) -> list[bool] | None:
    """Primary stress of each canonical reference phoneme (``None`` when
    eSpeak's words do not reproduce ``expected``)."""
    words = espeak_words(text)
    flat = [p for ph, _ in words for p in ph]
    if flat != list(expected):
        return None
    stressed = [x for _, st in words for x in st]
    _, src = canonical_map(flat)
    return [any(stressed[k] for k in ss) for ss in src]


def cut_tokens(pj: J.PhonemeJudge, metric: str, pad: float = 0.15) -> tuple[list[material.Item], list[np.ndarray]]:
    """Words cut out of the human anchors, to be scored as isolated items.

    Each recording is force-aligned to its reference (Viterbi CTC path,
    r-coloured pairs merged); eSpeak's word boundaries split it, each word
    running from halfway between the last spike of the previous word and its
    own first spike to halfway after its last, with 10 ms fades and ``pad``
    seconds of silence on each side.  The slots of ``metric`` are scored in
    every word (vowels carry ``stressed``; consonants only between two vowels
    of the same word)."""
    items_h = material.load(["human"])
    audios_h = [render.load_audio(render.ANCHOR_DIR / f"{it.id}.flac") for it in items_h]
    lps = pj.log_probs(audios_h, batch_size=2)
    items, audios = [], []
    fade = np.hanning(2 * 160).astype(np.float32)
    for it, audio, lp in zip(items_h, audios_h, lps):
        words = espeak_words(it.ortho or it.text)
        flat = [p for ph, _ in words for p in ph]
        if flat != list(it.expected):
            print(f"[cut] {it.id}: eSpeak words do not match the reference, skipped")
            continue
        word_of = [w for w, (ph, _) in enumerate(words) for _ in ph]
        stressed = [x for _, st in words for x in st]
        ref, src = canonical_map(flat)
        units = J.units_of(ref, True, pj.unit_ids)
        unit_of = [k for k, u in enumerate(units) for _ in u]
        classes = sorted(set(units))
        small = J.class_matrix(lp, classes, pj.unit_ids, pj.blank).numpy()
        spans = ctc_viterbi(small, [classes.index(u) + 1 for u in units])
        n_frames = small.shape[0]
        # canonical position -> word; a unit belongs to the word of its first phoneme
        cword = [word_of[ss[0]] for ss in src]
        cstress = [any(stressed[k] for k in ss) for ss in src]
        unit_word = [cword[unit_of.index(k)] for k in range(len(units))]
        for w in sorted(set(unit_word)):
            us = [k for k, x in enumerate(unit_word) if x == w]
            first, last = us[0], us[-1]
            a = 0 if first == 0 else (spans[first - 1][1] + spans[first][0] + 1) // 2
            b = n_frames if last == len(units) - 1 else (spans[last][1] + spans[last + 1][0] + 1) // 2
            x = audio[a * FRAME : min(len(audio), b * FRAME)].astype(np.float32).copy()
            if len(x) < 2 * len(fade):
                continue
            x[:160] *= fade[:160]
            x[-160:] *= fade[160:]
            pos = [k for k, u in enumerate(unit_of) if u in us]
            phones = [ref[k] for k in pos]
            marks = "".join("1" if cstress[k] else "0" for k in pos)
            meta = material._meta(src=it.id, word=w, stress=marks)
            items.append(material.Item(f"cut_{it.id[6:]}_w{w:02d}", "human_cut", "", tuple(phones), "", meta))
            z = np.zeros(int(pad * render.TARGET_SR), np.float32)
            audios.append(np.concatenate([z, x, z]))
    return items, audios


# --------------------------------------------------------------------------
# Running a probe
# --------------------------------------------------------------------------


def default_label(system: str, profile: Path | None, set_name: str, cut: bool) -> str:
    label = f"{system}-{Path(profile).stem}" if profile and system in render.ENGINE_SYSTEMS else system
    if cut:
        return f"{label}-cut"
    if set_name == "human":
        return f"{label}-insent"
    return label


def run(
    metric: str,
    system: str,
    profile: Path | None = None,
    name: str | None = None,
    set_name: str | None = None,
    cut: bool = False,
    voice: str = render.DEFAULT_VOICE,
    jobs: int | None = None,
    fresh: bool = False,
    batch_size: int = 8,
    pj: J.PhonemeJudge | None = None,
    quiet: bool = False,
) -> dict:
    """Render, judge, summarise; write ``out/<metric>-<name>.json`` and
    ``report/<metric>-<name>.md``."""
    set_name = set_name or ("human" if system == "human" else DEFAULT_SET[metric])
    if system == "human" and set_name != "human":
        raise SystemExit("the human system only has the 'human' set (in-sentence) or --cut")
    if cut and system != "human":
        raise SystemExit("--cut cuts tokens out of the human recordings: use it with the human system")
    sentence = set_name == "human" or cut
    label = name or default_label(system, profile, set_name, cut)
    own = pj is None
    t0 = time.perf_counter()
    scratch = None
    if cut:
        pj = pj or J.PhonemeJudge()
        t_render = 0.0
        t1 = time.perf_counter()
        items, audios = cut_tokens(pj, metric)
    else:
        items = material.load([set_name])
        if fresh and system != "human":
            scratch = render.scratch_dir("probe-")
            rendered = render.render(items, system, profile, voice, jobs, cache=False, out_dir=scratch)
        else:
            rendered = render.render(items, system, profile if system in render.ENGINE_SYSTEMS else None, voice, jobs)
        t_render = time.perf_counter() - t0
        audios = [render.load_audio(r.path) for r in rendered]
        if scratch is not None:
            import shutil

            shutil.rmtree(scratch, ignore_errors=True)
        pj = pj or J.PhonemeJudge()
        t1 = time.perf_counter()
    t_load = time.perf_counter() - t1 if own and not cut else 0.0
    t2 = time.perf_counter()
    results = score_slots(pj, items, audios, metric, sentence=sentence, batch_size=batch_size)
    t_judge = time.perf_counter() - t2
    if own:
        pj.close()
    summary = SUMMARIZE[metric](results)
    res = {
        "metric": metric,
        "name": label,
        "system": system,
        "set": "human_cut" if cut else set_name,
        "profile": str(profile) if profile and system in render.ENGINE_SYSTEMS else None,
        "voice": voice if system in render.ENGINE_SYSTEMS else None,
        "binary": str(render.say_bin()) if system in render.ENGINE_SYSTEMS else None,
        "phoneme_model": J.PHONEME_MODEL,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": {"render": round(t_render, 2), "judge": round(t_judge, 2), "model_load": round(t_load, 2),
                    "fresh_render": fresh},  # fmt: skip
        "summary": summary,
        "items": [r.to_json() for r in results],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{metric}-{label}.json"
    path.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    md = REPORT_DIR / f"{metric}-{label}.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(markdown(res)) + "\n", encoding="utf-8")
    if not quiet:
        print(summary_line(res))
        print(f"-> {path.relative_to(HERE)}, {md.relative_to(HERE)}")
    return res


# --------------------------------------------------------------------------
# Text output
# --------------------------------------------------------------------------


def pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def num(x, fmt="{:+.2f}") -> str:
    return "–" if x is None else fmt.format(x)


def summary_line(res: dict) -> str:
    s, m, name = res["summary"], res["metric"], res["name"]
    t = res["seconds"]
    tt = f"render {t['render']:.1f} s, judge {t['judge']:.1f} s"
    if m == "vowels":
        return (
            f"vowels {name}: vowel error {pct(s['vowel_error'])}% (AA=AO {pct(s['vowel_error_merged'])}%), "
            f"PER {pct(s['per'])}%, margin {num(s['margin'])} nats, intended wins {pct(s['won'])}% "
            f"[{s['slots']} slots; {tt}]"
        )
    if m == "voicing":
        h, f = s["headline"], s["fv"]
        return (
            f"voicing {name}: forced choice {pct(h['forced'])}% (voiceless {pct(h['forced_voiceless'])} / voiced "
            f"{pct(h['forced_voiced'])}), margin {num(h['margin'])}; free {pct(h['free'])}%; "
            f"F/V forced {pct(f['forced'])}% free {pct(f['free'])}% [{s['slots']} slots; {tt}]"
        )
    return (
        f"confusions {name}: consonant accuracy {pct(s['accuracy'])}%, place {pct(s['place_accuracy'])}%, "
        f"manner {pct(s['manner_accuracy'])}%, voicing {pct(s['voicing_accuracy'])}% [{s['slots']} slots; {tt}]"
    )


def _matrix_md(rows: dict[str, dict], cols: list[str], as_pct: bool = False) -> list[str]:
    """Rows: name -> Counter-like dict; one column per heard class."""
    out = ["| intended \\ heard | n | " + " | ".join(cols) + " |", "|---|---:|" + "---:|" * len(cols)]
    for r, c in rows.items():
        n = sum(c.values())
        cells = []
        for col in cols:
            v = c.get(col, 0)
            cell = "" if not v else (f"{100 * v / n:.0f}" if as_pct else str(v))
            cells.append(f"**{cell}**" if col == r and cell else cell)
        out.append(f"| {r} | {n} | " + " | ".join(cells) + " |")
    return out


def markdown(res: dict) -> list[str]:
    s, m = res["summary"], res["metric"]
    L = [f"# {m}: {res['name']}", "", summary_line(res), ""]
    if res.get("profile"):
        L += [f"Profile `{res['profile']}`.", ""]
    if m == "vowels":
        L += ["| vowel | n | error % | AA=AO error % | margin (nats) | wins % | heard as | loses to | model tokens |",
              "|---|---:|---:|---:|---:|---:|---|---|---|"]  # fmt: skip
        for v, d in s["per_vowel"].items():
            L.append(
                f"| {v} | {d['n']} | {pct(d['error'])} | {pct(d['error_merged'])} | {num(d['margin'])} | "
                f"{pct(d['won'])} | {', '.join(f'{h} {n}' for h, n in d['heard_as'].items())} | "
                f"{', '.join(f'{h} {n}' for h, n in d['rivals'].items())} | "
                f"{', '.join(f'{h} {n}' for h, n in d.get('tokens', {}).items())} |"
            )
        if s.get("heard_aa_tokens"):
            L += ["", "Other vowels heard as AA, by model token (`a`/`aː` is open front, between AE and AA; "
                  "`ɑ`/`ɑː` is the back AA): " + ", ".join(f"`{t}` {n}" for t, n in s["heard_aa_tokens"].items())]
        L += ["", "By frame:", "", "| frame | n | error % | margin |", "|---|---:|---:|---:|"]
        for fr, d in s["by_frame"].items():
            L.append(f"| {fr} | {d['n']} | {pct(d['error'])} | {num(d['margin'])} |")
        cols = [v for v in VOWELS if any(v in c for c in s["table"].values())]
        cols += [c for c in (DELETED, *A.CONSONANTS, OTHER_HEARD) if any(c in t for t in s["table"].values())]
        L += ["", "Confusion table (counts; rows intended, columns heard in the vowel slot):", ""]
        L += _matrix_md(s["table"], cols)
    elif m == "voicing":
        L += ["| group | n | forced % | balanced % | voiceless % | voiced % | margin | free % | free voiceless % | free voiced % |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]  # fmt: skip

        def row(g, d):
            return (
                f"| {g} | {d['n']} | {pct(d['forced'])} | {pct(d['forced_balanced'])} | {pct(d['forced_voiceless'])} | "
                f"{pct(d['forced_voiced'])} | {num(d['margin'])} | {pct(d['free'])} | {pct(d['free_voiceless'])} | "
                f"{pct(d['free_voiced'])} |"
            )

        L.append(row("**P/B T/D K/G S/Z**", s["headline"]))
        L.append(row("**F/V**", s["fv"]))
        for title, key in (("pair", "by_pair"), ("position (P/B T/D K/G S/Z)", "by_position"),
                           ("pair and position", "by_pair_position"), ("intended consonant", "by_target")):  # fmt: skip
            L.append(f"| _{title}_ | | | | | | | | | |")
            for g, d in s[key].items():
                L.append(row(g, d))
        L += ["", "Heard in the slot: " + ", ".join(f"{k} {v}" for k, v in s["heard_class"].items()), ""]
        L += ["Most frequent wrong symbols: " + ", ".join(f"{r}→{h} {n}" for r, h, n in s["worst"])]
    else:
        L += ["Most frequent confusions: " + ", ".join(f"{r}→{h} {n}" for r, h, n in s["worst"]), ""]
        for title, key in (("place", "place"), ("manner", "manner"), ("voicing", "voicing"), ("consonant", "full")):
            rows = {r: d["heard"] for r, d in s[key].items()}
            order = {"place": PLACE_ORDER, "manner": MANNER_ORDER, "voicing": ["voiceless", "voiced", "flap"]}.get(
                key, A.CONSONANTS + [A.FLAP]
            )
            seen = {c for d in rows.values() for c in d}
            cols = [c for c in order if c in seen] + [c for c in (VOWEL_HEARD, OTHER_HEARD, DELETED) if c in seen]
            L += [f"### By {title} (row %)", ""] + _matrix_md(rows, cols, as_pct=True) + [""]
    return L


# --------------------------------------------------------------------------
# Combined report
# --------------------------------------------------------------------------


def load(metric: str, name: str) -> dict | None:
    p = OUT_DIR / f"{metric}-{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def probe_report(
    names: list[str], out: Path, refs: list[str] | None = None, scores: list[str] | None = None, notes: Path | None = None
) -> None:
    """One Markdown page: every probe of every ``names`` run side by side,
    the reference runs (``refs``: eSpeak/human ceilings), and the sentence
    scoreboard from ``out/score-<name>.json`` for ``scores``."""
    names = list(names)
    allnames = names + [r for r in (refs or []) if r not in names]
    L = ["# Probe baseline", "", f"Generated by `judge.py probe-report` on {time.strftime('%Y-%m-%d %H:%M')}.", ""]
    if notes is not None:
        L += [Path(notes).read_text(encoding="utf-8").rstrip(), ""]

    get = load
    L += ["## Vowels", "", "| run | set | vowel error % | AA=AO error % | PER % | margin (nats) | intended wins % | slots |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
    for n in allnames:
        d = get("vowels", n)
        if d:
            s = d["summary"]
            L.append(f"| {n} | {d['set']} | {pct(s['vowel_error'])} | {pct(s['vowel_error_merged'])} | {pct(s['per'])} | "
                     f"{num(s['margin'])} | {pct(s['won'])} | {s['slots']} |")  # fmt: skip
    L += ["", "Vowel error by vowel (%):", ""]
    L += ["| run | " + " | ".join(VOWELS) + " |", "|---|" + "---:|" * len(VOWELS)]
    for n in allnames:
        d = get("vowels", n)
        if d:
            pv = d["summary"]["per_vowel"]
            L.append(f"| {n} | " + " | ".join(pct(pv[v]["error"]) if v in pv else "" for v in VOWELS) + " |")
    L += ["", "Vowel margin by vowel (nats):", ""]
    L += ["| run | " + " | ".join(VOWELS) + " |", "|---|" + "---:|" * len(VOWELS)]
    for n in allnames:
        d = get("vowels", n)
        if d:
            pv = d["summary"]["per_vowel"]
            L.append(f"| {n} | " + " | ".join(num(pv[v]["margin"], "{:+.1f}") if v in pv else "" for v in VOWELS) + " |")
    L += ["", "Worst vowel confusions (intended→heard count):", ""]
    for n in allnames:
        d = get("vowels", n)
        if d:
            L.append(f"- **{n}**: " + ", ".join(f"{r}→{h} {c}" for r, h, c in d["summary"]["worst"][:8]))

    pairs = [f"{a}/{b}" for a, b in material.VOICING_PAIRS]
    L += ["", "## Voicing", "", "Forced choice accuracy % (intended beats its voicing twin); free = recognised consonant has the intended voicing.", "",
          "| run | set | forced % | voiceless % | voiced % | margin | free % | " + " | ".join(f"{p} forced" for p in pairs) + " |",
          "|---|---|---:|---:|---:|---:|---:|" + "---:|" * len(pairs)]  # fmt: skip
    for n in allnames:
        d = get("voicing", n)
        if d:
            s = d["summary"]
            h = s["headline"]
            bp = s["by_pair"]
            L.append(f"| {n} | {d['set']} | {pct(h['forced'])} | {pct(h['forced_voiceless'])} | {pct(h['forced_voiced'])} | "
                     f"{num(h['margin'])} | {pct(h['free'])} | " + " | ".join(pct(bp[p]["forced"]) if p in bp else "" for p in pairs) + " |")  # fmt: skip
    L += ["", "By position (P/B T/D K/G S/Z), forced % / free %:", "",
          "| run | initial | medial | final | vcv | sentence |", "|---|---:|---:|---:|---:|---:|"]  # fmt: skip
    for n in allnames:
        d = get("voicing", n)
        if d:
            bp = d["summary"]["by_position"]
            L.append(f"| {n} | " + " | ".join(f"{pct(bp[p]['forced'])} / {pct(bp[p]['free'])}" if p in bp else ""
                                              for p in ("initial", "medial", "final", "vcv", "sentence")) + " |")  # fmt: skip

    L += ["", "## Consonants (VCV)", "", "| run | set | consonant % | place % | manner % | voicing % | slots |",
          "|---|---|---:|---:|---:|---:|---:|"]  # fmt: skip
    for n in allnames:
        d = get("confusions", n)
        if d:
            s = d["summary"]
            L.append(f"| {n} | {d['set']} | {pct(s['accuracy'])} | {pct(s['place_accuracy'])} | {pct(s['manner_accuracy'])} | "
                     f"{pct(s['voicing_accuracy'])} | {s['slots']} |")  # fmt: skip
    L += ["", "Worst consonant confusions:", ""]
    for n in allnames:
        d = get("confusions", n)
        if d:
            L.append(f"- **{n}**: " + ", ".join(f"{r}→{h} {c}" for r, h, c in d["summary"]["worst"][:10]))
    L += ["", "Place: row % heard at the intended place (stops, nasals, fricatives ...):", ""]
    L += ["| run | " + " | ".join(PLACE_ORDER) + " |", "|---|" + "---:|" * len(PLACE_ORDER)]
    for n in allnames:
        d = get("confusions", n)
        if d:
            pl = d["summary"]["place"]
            L.append(f"| {n} | " + " | ".join(pct(pl[p]["correct"]) if p in pl else "" for p in PLACE_ORDER) + " |")

    if scores:
        L += ["", "## Sentence scoreboard (`judge.py score`)", "",
              "| run | sentence_tune PER % | sentence_heldout PER % | librispeech PER % | tune WER % | heldout WER % | librispeech WER % | heldout word acc % |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
        for n in scores:
            p = OUT_DIR / f"score-{n}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text(encoding="utf-8"))["sets"]
            g = lambda s, k: pct(d[s].get(k)) if s in d else ""  # noqa: E731
            L.append(f"| {n} | {g('sentence_tune', 'per')} | {g('sentence_heldout', 'per')} | {g('human', 'per')} | "
                     f"{g('sentence_tune', 'wer')} | {g('sentence_heldout', 'wer')} | {g('human', 'wer')} | {g('sentence_heldout', 'word_acc')} |")  # fmt: skip
    L += ["", "## Timing", "", "| run | metric | render s | judge s | fresh render |", "|---|---|---:|---:|---|"]
    for n in allnames:
        for m in METRICS:
            d = get(m, n)
            if d:
                t = d["seconds"]
                L.append(f"| {n} | {m} | {t['render']} | {t['judge']} | {t.get('fresh_render')} |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}")
