"""Model-free tests of ``probes.py``: the probe material, slot alignment, and
the forced-choice scoring on synthetic posteriors (no GPU, no downloads)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import arpabet as A  # noqa: E402
import espeak_util as E  # noqa: E402
import judge as J  # noqa: E402
import material  # noqa: E402
import probes as P  # noqa: E402

VOCAB = A.load_vocab()
UNIT_IDS = J.build_unit_ids(VOCAB)
BLANK = VOCAB[A.BLANK_TOKEN]


# --------------------------------------------------------------------------
# Material
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", material.PROBE_SETS)
def test_probe_material_is_current(name):
    built = {
        "vowel": material.build_vowel,
        "voicing": material.build_voicing,
        "carrier_tune": lambda: material.build_carrier("carrier_tune"),
        "carrier_heldout": lambda: material.build_carrier("carrier_heldout"),
        "harvard_tune": lambda: material.build_sentences("harvard_tune", material.HARVARD_TUNE, "harv"),
        "harvard_heldout": lambda: material.build_sentences("harvard_heldout", material.HARVARD_HELDOUT, "harv"),
    }[name]()
    assert material.load([name]) == built, f"material/{name}.tsv is stale: run `uv run python material.py build`"


@pytest.mark.parametrize("name", [s for s in material.PROBE_SETS if s not in material.SENTENCE_SETS])
def test_probe_items_map_through_arpabet(name):
    items = material.load([name])
    assert len({it.id for it in items}) == len(items)
    for it in items:
        assert it.expected and all(p in A.ENGINE_PHONEMES for p in it.expected), it
        assert A.parse_arpabet(it.text) == list(it.expected), it
        for tok in A.arpabet_to_tokens(list(it.expected)) + A.arpabet_to_tokens(list(it.expected), merge=False):
            assert tok in VOCAB, (it.id, tok)
        assert E.engine_text_to_espeak(it.text).startswith("[["), it
        # the scored slot survives the canonical form unchanged
        slot = int(it.info()["slot"])
        canon, cslot = P.canonical_slot(list(it.expected), slot)
        assert canon == list(it.expected) and cslot == slot, it


def test_vowel_material_covers_every_vowel_in_every_frame():
    items = material.load(["vowel"])
    assert sorted({it.info()["target"] for it in items}) == sorted(material.VOWEL_TARGETS)
    assert len(material.VOWEL_TARGETS) == 16
    for it in items:
        info = it.info()
        assert it.expected[int(info["slot"])] == info["target"], it
        assert info["lex"] in ("word", "nonsense", "isolated")
        assert bool(it.ortho) == (info["lex"] == "word"), it
        # one vowel per item, except the AX words
        if info["target"] != "AX":
            assert sum(p in A.VOCALIC for p in it.expected) == 1, it
    per_vowel = {v: sum(it.info()["target"] == v for it in items) for v in material.VOWEL_TARGETS}
    assert set(per_vowel.values()) == {6}, per_vowel


def test_voicing_material_twins():
    items = material.load(["voicing"])
    by_phones = {it.expected: it for it in items}
    for it in items:
        info = it.info()
        slot, target = int(info["slot"]), info["target"]
        twin = tuple(info["twin"].split())
        assert it.expected[slot] == target
        diff = [k for k, (a, b) in enumerate(zip(it.expected, twin)) if a != b]
        assert len(twin) == len(it.expected) and diff == [slot], it
        assert P.TWIN[target] == twin[slot]
        assert info["voicing"] == P.voicing_of(target)
        assert twin in by_phones, f"{it.id}: its twin is not in the set"
        assert info["pos"] in ("initial", "medial", "final", "vcv")
    # balanced: every pair and position has as many voiced as voiceless items
    for pair in ("P/B", "T/D", "K/G", "S/Z", "F/V"):
        for pos in ("initial", "medial", "final", "vcv"):
            group = [it for it in items if it.info()["pair"] == pair and it.info()["pos"] == pos]
            assert group, (pair, pos)
            n_voiced = sum(it.info()["voicing"] == "voiced" for it in group)
            assert 2 * n_voiced == len(group), (pair, pos)


# --------------------------------------------------------------------------
# Slot alignment
# --------------------------------------------------------------------------


def test_slot_heard():
    ref = ["B", "IY", "D"]
    assert P.slot_heard(ref, ["B", "IY", "D"], 1) == ("ok", "IY")
    assert P.slot_heard(ref, ["B", "IH", "D"], 1) == ("sub", "IH")
    assert P.slot_heard(ref, ["B", "D"], 1) == ("del", None)
    # an inserted L does not move the slot
    assert P.slot_heard(ref, ["B", "L", "IY", "D"], 1) == ("ok", "IY")
    assert P.slot_heard(ref, ["P", "IY", "T"], 0) == ("sub", "P")
    assert P.slot_heard(ref, ["P", "IY", "T"], 2) == ("sub", "T")
    # the flap matches T and D
    assert P.slot_heard(["AA", "T", "AA"], ["AA", "DX", "AA"], 1) == ("ok", "DX")


def test_canonical_slot():
    assert P.canonical_slot(["B", "AA1", "T"], 1) == (["B", "AA", "T"], 1)
    # "AX R" becomes ER: a slot on either symbol lands on the ER
    assert P.canonical_slot(["S", "AX", "R", "T"], 1) == (["S", "ER", "T"], 1)
    assert P.canonical_slot(["S", "AX", "R", "T"], 3) == (["S", "ER", "T"], 2)
    assert P.canonical_slot(["HH", "IY", "R"], 1) == (["HH", "IH", "R"], 1)


def test_same_vowel():
    assert P.same_vowel("AH", "AX") and P.same_vowel("IY", "IY")
    assert not P.same_vowel("IY", "IH") and not P.same_vowel("AA", "AO")
    assert P.same_vowel("AA", "AO", P.VOWEL_SAME_MERGED)
    assert not P.same_vowel(None, "AA")
    assert "AH" not in P.vowel_rivals("AX") and "AX" not in P.vowel_rivals("AH")
    assert len(P.vowel_rivals("IY")) == 15  # every other vowel (AH and AX both: the max is the same)


# --------------------------------------------------------------------------
# Forced choice on synthetic posteriors
# --------------------------------------------------------------------------


def posteriors(frames: list[dict[str, float] | None], floor: float = 1e-4) -> torch.Tensor:
    """frames: token -> probability (``None`` = a blank frame); the rest of
    the mass is spread over the vocabulary.  Returns T x V log-probs."""
    v = len(VOCAB)
    rows = []
    for f in frames:
        p = np.full(v, floor)
        f = f or {A.BLANK_TOKEN: 1.0}
        for tok, q in f.items():
            p[VOCAB[tok]] = q
        rows.append(p / p.sum())
    return torch.log(torch.tensor(np.array(rows), dtype=torch.float32))


def spoken(*tokens, other=None):
    """blank, tok, blank, tok, ... with optional second token at each step."""
    frames = [None]
    for k, t in enumerate(tokens):
        frames += [t if isinstance(t, dict) else {t: 1.0}, None]
    return posteriors(frames)


def test_forced_choice_prefers_what_was_said():
    lp = spoken("b", "æ", "t")
    m, rival, own = P.choice_margin(lp, ["B", "AE", "T"], 0, ["P"], UNIT_IDS, BLANK)
    assert rival == "P" and m > 5 and own > -1
    m2, _, _ = P.choice_margin(lp, ["P", "AE", "T"], 0, ["B"], UNIT_IDS, BLANK)
    assert m2 < -5 and abs(m + m2) < 1e-3  # antisymmetric


def test_forced_choice_is_continuous():
    margins = []
    for q in (0.2, 0.4, 0.5, 0.6, 0.8):
        lp = spoken({"b": q, "p": 1 - q}, "æ", "t")
        margins.append(P.choice_margin(lp, ["B", "AE", "T"], 0, ["P"], UNIT_IDS, BLANK)[0])
    assert all(a < b for a, b in zip(margins, margins[1:])), margins
    assert abs(margins[2]) < 1e-3  # a coin toss is a zero margin


def test_vowel_margin_names_the_rival():
    # "bit" said, "beat" asked for: IH is the best rival and the margin negative
    lp = spoken("b", "ɪ", "t")
    m, rival, _ = P.choice_margin(lp, ["B", "IY", "T"], 1, P.vowel_rivals("IY"), UNIT_IDS, BLANK)
    assert rival == "IH" and m < -5
    m, rival, _ = P.choice_margin(lp, ["B", "IH", "T"], 1, P.vowel_rivals("IH"), UNIT_IDS, BLANK)
    assert m > 5
    # every token that folds to the vowel counts for it: "i" is IY too
    lp = spoken("b", "i", "t")
    assert P.choice_margin(lp, ["B", "IY", "T"], 1, P.vowel_rivals("IY"), UNIT_IDS, BLANK)[0] > 5


def test_margin_is_capped_when_audio_is_too_short():
    lp = posteriors([{"b": 1.0}])  # one frame cannot carry three phonemes
    m, _, own = P.choice_margin(lp, ["B", "AE", "T"], 1, P.vowel_rivals("AE"), UNIT_IDS, BLANK)
    assert own == -math.inf and m == 0.0


def test_sequences_logprob_merges_r_coloured_units():
    lp = spoken("k", "ɑːɹ")  # "car" as the model writes it
    merged, wrong = J.sequences_logprob(lp, [["K", "AA", "R"], ["K", "AO", "R"]], UNIT_IDS, BLANK)
    assert merged > -1 and wrong < merged - 5


def test_ctc_viterbi_spans():
    lp = spoken("b", "æ", "t")  # frames: _ b _ æ _ t _
    classes = [("B",), ("AE",), ("T",)]
    small = J.class_matrix(lp, classes, UNIT_IDS, BLANK).numpy()
    assert P.ctc_viterbi(small, [1, 2, 3]) == [(1, 1), (3, 3), (5, 5)]
    lp = posteriors([{"b": 1.0}, {"b": 1.0}, None, {"b": 1.0}])  # B B: blank between repeats
    small = J.class_matrix(lp, [("B",)], UNIT_IDS, BLANK).numpy()
    assert P.ctc_viterbi(small, [1, 1]) == [(0, 1), (3, 3)]


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------


def _slot(target, heard, margin, pair, pos="initial"):
    return P.Slot("x", "voicing", [target], 0, target, heard, "ok", margin, None,
                  {"pair": pair, "pos": pos, "voicing": P.voicing_of(target)})  # fmt: skip


def test_all_voiced_system_scores_half():
    slots = []
    for vl, vd in [("P", "B"), ("T", "D"), ("K", "G"), ("S", "Z")]:
        pair = f"{vl}/{vd}"
        slots.append(_slot(vl, vd, -3.0, pair))  # voiceless asked, voiced heard and preferred
        slots.append(_slot(vd, vd, +3.0, pair))
    res = [P.ItemResult("x", "voicing", ["B"], ["B"], [], 0, 0.0, slots)]
    s = P.summarize_voicing(res)["headline"]
    assert s["forced"] == 0.5 and s["forced_balanced"] == 0.5
    assert s["forced_voiced"] == 1.0 and s["forced_voiceless"] == 0.0
    assert s["free"] == 0.5 and s["free_balanced"] == 0.5


def test_free_voicing_credit():
    assert P._free_credit(_slot("T", "DX", 0, "T/D")) == 0.5
    assert P._free_credit(_slot("T", "N", 0, "T/D")) == 0.0  # sonorant: wrong
    assert P._free_credit(_slot("S", "SH", 0, "S/Z")) == 1.0  # right voicing, wrong place
    assert P._free_credit(_slot("Z", None, 0, "S/Z")) == 0.0


def test_vowel_summary_counts_ax_as_ah():
    def item(target, heard):
        s = P.Slot("x", "vowel", [target], 0, target, heard, "sub", 1.0, None, {"frame": "iso"})
        return P.ItemResult("x", "vowel", [target], [heard or "B"], [], 1, 0.0, [s])

    s = P.summarize_vowels([item("AX", "AH"), item("IY", "IH"), item("AA", "AO"), item("UW", None)])
    assert s["vowel_error"] == 0.75 and s["vowel_error_merged"] == 0.5
    assert s["table"]["IY"] == {"IH": 1} and s["table"]["UW"] == {"-": 1} and s["table"]["AX"] == {"AX": 1}


def test_consonant_tables():
    def item(target, heard):
        s = P.Slot("x", "vcv", ["AA", target, "AA"], 1, target, heard, "sub", None, None, {})
        return P.ItemResult("x", "vcv", ["AA", target, "AA"], [], [], 0, 0.0, [s])

    s = P.summarize_confusions([item("B", "D"), item("M", "N"), item("T", "T"), item("G", "AA")])
    assert s["accuracy"] == 0.25
    assert s["place"]["bilabial"]["heard"] == {"alveolar": 2}
    assert s["manner"]["stop"]["heard"] == {"stop": 2, "vowel": 1}
    assert s["voicing"]["voiced"]["n"] == 3


def test_heard_with_tokens():
    syms, toks = P.heard_with_tokens(["b", "ɑːɹ", "d"])
    assert syms == ["B", "AA", "R", "D"] and toks == ["b", "ɑːɹ", "ɑːɹ", "d"]
    syms, toks = P.heard_with_tokens(["s", "ə", "ɹ", "t"])  # AX R -> ER
    assert syms == ["S", "ER", "T"] and toks == ["s", "ə+ɹ", "t"]
    assert P.slot_heard_at(["B", "IY", "D"], ["L", "B", "IH", "D"], 1) == ("sub", "IH", 2)
