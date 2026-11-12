"""Tests for the ARPAbet <-> eSpeak-IPA normalizer (``arpabet.py``).

The phoneme judge's tokenizer vocabulary is bundled as
``data/w2v_vocab.json``; every check that says "the way the model tokenizes"
is checked against that file, not against assumptions.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import arpabet as A  # noqa: E402

VOCAB = A.load_vocab()

#: Fixed round-trip list: words that exercise stops, affricates, diphthongs,
#: r-coloured vowels, syllabic consonants and the flap.
ROUND_TRIP = [
    "B AE1 T",  # bat
    "CH IY1 P",  # cheap
    "JH AH1 JH",  # judge
    "V OY1 S",  # voice
    "HH AW1 S",  # house
    "L AY1 T",  # light
    "L EY1 T",  # late
    "B OW1 T",  # boat
    "K AA1 R",  # car
    "F AO1 R",  # four
    "HH EH1 R",  # hair
    "HH IH1 R",  # here
    "P UH1 R",  # poor
    "F AY1 ER",  # fire
    "T EY1 B AX L",  # table
    "G AA1 R D AX N",  # garden
    "M EH1 ZH ER",  # measure
    "TH IH1 NG",  # thing
    "DH AX",  # the
    "Y EH1 L OW",  # yellow
    "W ER1 L D",  # world
    "S K UW1 L",  # school
    "B EH1 DX ER",  # better (flap)
    "IY1 T",  # eat
    "P UH1 L",  # pull
    "AX B AW1 T",  # about
]


# --------------------------------------------------------------------------
# Forward map: every engine symbol has a model token
# --------------------------------------------------------------------------


def test_engine_alphabet_is_40_phonemes():
    # the CMU 39 plus AX (the engine has a separate schwa)
    assert len(A.ENGINE_PHONEMES) == 40
    assert len(set(A.ENGINE_PHONEMES)) == 40


@pytest.mark.parametrize("sym", A.ENGINE_PHONEMES + [A.FLAP])
def test_every_arpabet_symbol_maps_to_a_vocab_token(sym):
    tok = A.ARPA_TO_IPA[sym]
    assert tok in VOCAB, f"{sym} -> {tok!r} is not a model token"
    assert A.fold_token(tok) == (sym,), f"{tok!r} folds back to {A.fold_token(tok)}"


@pytest.mark.parametrize("sym", A.ENGINE_PHONEMES)
def test_arpabet_to_tokens_single_symbol(sym):
    assert A.arpabet_to_tokens([sym]) == [A.ARPA_TO_IPA[sym]]
    # stress digits are ignored
    assert A.arpabet_to_tokens([sym + "1"]) == [A.ARPA_TO_IPA[sym]]


def test_merged_tokens_are_vocab_tokens_and_fold_back():
    for pair, tok in A.MERGED_TOKENS.items():
        assert tok in VOCAB, tok
        assert A.fold_token(tok) == pair, (tok, A.fold_token(tok))


# --------------------------------------------------------------------------
# Reverse map: every model token folds into the scoring alphabet
# --------------------------------------------------------------------------


def test_every_vocab_token_folds_into_scoring_alphabet():
    allowed = set(A.SCORING_SYMBOLS)
    for tok in VOCAB:
        fold = A.fold_token(tok)
        if tok in A.SPECIAL_TOKENS:
            assert fold == ()
        else:
            assert fold, tok
            assert set(fold) <= allowed, (tok, fold)


def test_every_scoring_symbol_has_a_variant():
    v = A.variants()
    for sym in A.ENGINE_PHONEMES:
        assert v[sym], f"no vocab token folds to {sym}"
    assert "ɾ" in v["T"] and "ɾ" in v["D"]


def test_stress_marks_ignored():
    assert A.fold_token("ˈaɪ") == ("AY",)
    assert A.fold_token("ˌiː") == ("IY",)


def test_unknown_token_folds_to_other():
    assert A.fold_token("☃") == (A.OTHER,)


def test_fold_table_file_in_sync():
    assert A.FOLD_TABLE_PATH.read_text(encoding="utf-8") == A.render_fold_table(), (
        "data/fold_table.tsv is stale: run `uv run python arpabet.py --dump`"
    )


# --------------------------------------------------------------------------
# Diphthongs and affricates: ONE model token each
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sym", A.DIPHTHONGS + A.AFFRICATES)
def test_diphthongs_and_affricates_are_one_model_token(sym):
    tok = A.ARPA_TO_IPA[sym]
    assert len(tok) == 2, tok  # two IPA letters ...
    assert tok in VOCAB  # ... but one model token
    assert A.arpabet_to_tokens([sym]) == [tok]
    assert A.fold_tokens([tok]) == [sym]


@pytest.mark.parametrize(
    "parts, whole",
    [(["t", "ʃ"], "CH"), (["d", "ʒ"], "JH"), (["a", "ɪ"], "AY"), (["e", "ɪ"], "EY"), (["o", "ʊ"], "OW")],
)
def test_split_spelling_is_not_the_single_phoneme(parts, whole):
    """The model emits ``tʃ`` as one token for /tʃ/.  If it emits ``t`` then
    ``ʃ`` it heard two phonemes, and that is scored as two."""
    assert all(p in VOCAB for p in parts)
    assert A.fold_tokens(parts) != [whole]
    assert len(A.fold_tokens(parts)) == 2


def test_other_spellings_of_diphthongs_fold_to_same_symbol():
    # what other eSpeak languages in the vocabulary use for the same glides
    for tok, sym in [("ai", "AY"), ("au", "AW"), ("ei", "EY"), ("əʊ", "OW"), ("oɪ", "OY")]:
        if tok in VOCAB:
            assert A.fold_token(tok) == (sym,), tok


def test_affricate_variants():
    assert A.fold_token("tʃ") == ("CH",)
    assert A.fold_token("dʒ") == ("JH",)
    if "tɕ" in VOCAB:
        assert A.fold_token("tɕ") == ("CH",)
    if "dʑ" in VOCAB:
        assert A.fold_token("dʑ") == ("JH",)


# --------------------------------------------------------------------------
# Round trip ARPAbet -> model tokens -> ARPAbet
# --------------------------------------------------------------------------


@pytest.mark.parametrize("arpa", ROUND_TRIP)
@pytest.mark.parametrize("merge", [True, False])
def test_round_trip(arpa, merge):
    phones = A.parse_arpabet(arpa)
    tokens = A.arpabet_to_tokens(phones, merge=merge)
    assert all(t in VOCAB for t in tokens), tokens
    back = A.fold_tokens(tokens)
    assert A.canonical(back) == A.canonical(phones)
    assert A.error_rate(phones, back) == 0.0


def test_round_trip_merges_r_coloured_vowels():
    assert A.arpabet_to_tokens(["K", "AA1", "R"]) == ["k", "ɑːɹ"]
    assert A.arpabet_to_tokens(["K", "AA1", "R"], merge=False) == ["k", "ɑː", "ɹ"]


def test_ipa_string_round_trip():
    assert A.ipa_to_arpabet("k_ˈɑːɹ t_ʃ") == ["K", "AA", "R", "T", "SH"]
    assert A.ipa_to_arpabet("tʃ_ˈiː_p") == ["CH", "IY", "P"]


# --------------------------------------------------------------------------
# Engine-side parsing
# --------------------------------------------------------------------------


def test_parse_arpabet():
    assert A.parse_arpabet("[B AA1 T] [S IH1 P]") == ["B", "AA", "T", "S", "IH", "P"]
    assert A.parse_arpabet("SIL B AA1 SIL") == ["B", "AA"]


@pytest.mark.parametrize(
    "line, phones",
    [
        ("AA P P AA", ["AA", "P", "AA"]),  # aspirated stop before a vowel
        ("B AY AY T", ["B", "AY", "T"]),  # diphthong = two segments
        ("CH CH IY P", ["CH", "IY", "P"]),  # affricate = two segments
        ("K K L AE NG K S SIL", ["K", "L", "AE", "NG", "K", "S"]),
        ("B AE T", ["B", "AE", "T"]),  # already one per phoneme
        ("AA T", ["AA", "T"]),  # final stop: nothing to undo
        ("AA D D SIL", ["AA", "D"]),  # physical: a final stop's release
        ("SIL DH DH AX K AE T T", ["DH", "AX", "K", "AE", "T"]),  # dental onset after a pause
        ("AX DH AX", ["AX", "DH", "AX"]),
    ],
)
def test_parse_engine_line(line, phones):
    assert A.parse_engine_line(line) == phones


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def test_canonical_rules():
    assert A.canonical(["AX", "R"]) == ["ER"]
    assert A.canonical(["IY", "R"]) == ["IH", "R"]
    assert A.canonical(["B", "AA1", "T"]) == ["B", "AA", "T"]


def test_flap_matches_t_and_d():
    assert A.error_rate(["B", "EH", "T", "ER"], ["B", "EH", "DX", "ER"]) == 0.0
    assert A.error_rate(["B", "EH", "D", "ER"], ["B", "EH", "DX", "ER"]) == 0.0
    assert A.error_rate(["B", "EH", "P", "ER"], ["B", "EH", "DX", "ER"]) == 0.25


def test_align_ops():
    ops = A.align(["B", "AE", "T"], ["P", "AE", "T", "S"])
    assert [o for o, _, _ in ops] == ["sub", "ok", "ok", "ins"]
    ops = A.align(["S", "IH", "N"], ["S", "N"])
    assert ("del", "IH", None) in ops
    assert A.edit_distance(["S", "IH", "N"], ["S", "N"]) == 1


# --------------------------------------------------------------------------
# eSpeak agrees with the forward map (skipped without espeak-ng)
# --------------------------------------------------------------------------

needs_espeak = pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng not installed")


@needs_espeak
@pytest.mark.parametrize(
    "word, arpa",
    [
        ("bat", "B AE T"),
        ("cheap", "CH IY P"),
        ("judge", "JH AH JH"),
        ("voice", "V OY S"),
        ("house", "HH AW S"),
        ("light", "L AY T"),
        ("late", "L EY T"),
        ("boat", "B OW T"),
        ("thing", "TH IH NG"),
        ("measure", "M EH ZH ER"),
    ],
)
def test_espeak_plain_words_fold_to_expected(word, arpa):
    import espeak_util as E

    assert A.canonical(E.phonemes(word)) == A.canonical(arpa.split())


@needs_espeak
def test_espeak_tokens_are_all_vocab_tokens():
    import espeak_util as E

    toks = E.ipa_tokens("The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background.")
    assert all(t in VOCAB for t in toks), [t for t in toks if t not in VOCAB]


@needs_espeak
def test_espeak_speaks_bracketed_arpabet():
    """The eSpeak *system* turns ``[ARPAbet]`` into eSpeak mnemonics; check
    every material item comes back as the phonemes it was given."""
    import espeak_util as E
    import material

    bad = []
    for it in material.load(["vcv", "pair", "word"]):
        got = E.phonemes(E.engine_text_to_espeak(it.text))
        # eSpeak flaps intervocalic /t/ ("water"): DX matches T, so use the
        # scoring comparison, not string equality.
        if A.error_rate(list(it.expected), got) != 0.0:
            bad.append((it.id, " ".join(it.expected), " ".join(got)))
    assert not bad, bad
