"""ARPAbet <-> eSpeak-IPA normalizer.

The engine speaks 40 ARPAbet phonemes (CMU 39 + AX).  The phoneme judge
(``facebook/wav2vec2-lv-60-espeak-cv-ft``) emits eSpeak IPA tokens from a
392-entry multilingual vocabulary.  Everything that compares the two goes
through this module, so that error rates are computed in ONE alphabet:

* :func:`fold_token` / :data:`FOLD` -- any model token -> tuple of scoring
  symbols (the reverse map).  Allophones, length marks, tone digits,
  palatalisation, aspiration, nasalisation etc. are folded to the nearest
  engine phoneme; what has no sensible English neighbour becomes ``OTH``.
* :func:`arpabet_to_tokens` -- engine phonemes -> the token sequence eSpeak
  (and therefore the model) uses for them (the forward map), including the
  rhotic/syllabic tokens eSpeak writes as one unit (``AA R`` -> ``ɑːɹ``).
* :func:`canonical` -- sequence-level rules applied to BOTH reference and
  hypothesis before scoring, for distinctions that are notation rather than
  sound (``AX R`` vs ``ER``, ``IY R`` vs ``IH R`` ...).
* :func:`align` -- edit distance with the flap wildcard (``DX`` matches
  ``T`` and ``D``).

Scoring alphabet = the 40 engine phonemes + ``DX`` (flap: an allophone of
both /t/ and /d/, so it matches either) + ``OTH`` (not an English sound).

The complete token table is written to ``data/fold_table.tsv`` by
``python arpabet.py --dump`` so the mapping can be reviewed line by line;
a test checks the file is in sync with this code.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
VOCAB_PATH = HERE / "data" / "w2v_vocab.json"
FOLD_TABLE_PATH = HERE / "data" / "fold_table.tsv"

# --------------------------------------------------------------------------
# The engine's alphabet
# --------------------------------------------------------------------------

VOWELS = ["AA", "AE", "AH", "AO", "AX", "EH", "ER", "IH", "IY", "UH", "UW"]
DIPHTHONGS = ["AY", "AW", "EY", "OW", "OY"]
STOPS = ["P", "B", "T", "D", "K", "G"]
NASALS = ["M", "N", "NG"]
FRICATIVES = ["F", "V", "TH", "DH", "S", "Z", "SH", "ZH"]
AFFRICATES = ["CH", "JH"]
APPROXIMANTS = ["L", "R", "W", "Y"]
ASPIRATE = ["HH"]
CONSONANTS = STOPS + NASALS + FRICATIVES + AFFRICATES + APPROXIMANTS + ASPIRATE
ENGINE_PHONEMES = VOWELS + DIPHTHONGS + CONSONANTS  # 40 (CMU 39 + AX), SIL excluded
VOCALIC = set(VOWELS + DIPHTHONGS)

FLAP = "DX"  # matches T and D
OTHER = "OTH"  # no English neighbour
SCORING_SYMBOLS = ENGINE_PHONEMES + [FLAP, OTHER]

SPECIAL_TOKENS = {"<pad>", "<s>", "</s>", "<unk>"}
BLANK_TOKEN = "<pad>"  # CTC blank of the model, id 0

# --------------------------------------------------------------------------
# Forward map: engine phoneme -> eSpeak (en-us) IPA token
# --------------------------------------------------------------------------

#: The token eSpeak NG's en-us voice writes for each engine phoneme.
ARPA_TO_IPA: dict[str, str] = {
    "AA": "ɑː", "AE": "æ", "AH": "ʌ", "AO": "ɔː", "AX": "ə", "EH": "ɛ",
    "ER": "ɚ", "IH": "ɪ", "IY": "iː", "UH": "ʊ", "UW": "uː",
    "AY": "aɪ", "AW": "aʊ", "EY": "eɪ", "OW": "oʊ", "OY": "ɔɪ",
    "P": "p", "B": "b", "T": "t", "D": "d", "K": "k", "G": "ɡ",
    "M": "m", "N": "n", "NG": "ŋ",
    "F": "f", "V": "v", "TH": "θ", "DH": "ð", "S": "s", "Z": "z",
    "SH": "ʃ", "ZH": "ʒ", "CH": "tʃ", "JH": "dʒ",
    "L": "l", "R": "ɹ", "W": "w", "Y": "j", "HH": "h",
    FLAP: "ɾ",
}  # fmt: skip

#: Phoneme pairs eSpeak en-us writes as ONE token (r-coloured vowels and
#: syllabic consonants).  A human "car" is transcribed ``k ɑːɹ`` by the
#: model, never ``k ɑː ɹ``, so targets must be able to use these.
MERGED_TOKENS: dict[tuple[str, ...], str] = {
    ("AA", "R"): "ɑːɹ",
    ("AO", "R"): "ɔːɹ",
    ("EH", "R"): "ɛɹ",
    ("IH", "R"): "ɪɹ",
    ("UH", "R"): "ʊɹ",
    ("AY", "ER"): "aɪɚ",
    ("AY", "AX"): "aɪə",
    ("IY", "AX"): "iə",
    ("AX", "L"): "əl",
    ("AX", "N"): "n̩",
}

# --------------------------------------------------------------------------
# Reverse map: model token -> scoring symbols
# --------------------------------------------------------------------------

#: Explicit folds.  Every token eSpeak NG 1.52 en-us can produce is here
#: (61 tokens, collected over 37k dictionary words), plus the few non-English
#: tokens whose fold is a decision rather than a rule.
EXPLICIT_FOLD: dict[str, tuple[str, ...]] = {
    # --- en-us vowels -----------------------------------------------------
    "ɑː": ("AA",), "æ": ("AE",), "ʌ": ("AH",),
    "ɔː": ("AO",), "ɔ": ("AO",),          # "on", "want": short open o
    "ə": ("AX",), "ɐ": ("AX",),            # "about": eSpeak writes ɐ
    "ɛ": ("EH",),
    "ɚ": ("ER",), "ɜː": ("ER",), "ɜ": ("ER",),
    "ɪ": ("IH",), "ᵻ": ("IH",),            # "roses": CMUdict writes IH0
    "iː": ("IY",), "i": ("IY",),           # "happy": final unstressed i
    "ʊ": ("UH",), "uː": ("UW",),
    "aɪ": ("AY",), "aʊ": ("AW",), "eɪ": ("EY",), "oʊ": ("OW",), "ɔɪ": ("OY",),
    # --- en-us r-coloured and syllabic units (one token, two phonemes) ----
    "ɑːɹ": ("AA", "R"), "ɔːɹ": ("AO", "R"), "oːɹ": ("AO", "R"),
    "ɛɹ": ("EH", "R"), "ɪɹ": ("IH", "R"), "ʊɹ": ("UH", "R"),
    "aɪɚ": ("AY", "ER"), "aɪə": ("AY", "AX"), "iə": ("IY", "AX"),
    "əl": ("AX", "L"), "l̩": ("AX", "L"), "n̩": ("AX", "N"), "r̩": ("ER",),
    # --- en-us consonants -------------------------------------------------
    "p": ("P",), "b": ("B",), "t": ("T",), "d": ("D",), "k": ("K",), "ɡ": ("G",),
    "m": ("M",), "n": ("N",), "ŋ": ("NG",),
    "f": ("F",), "v": ("V",), "θ": ("TH",), "ð": ("DH",),
    "s": ("S",), "z": ("Z",), "ʃ": ("SH",), "ʒ": ("ZH",),
    "tʃ": ("CH",), "dʒ": ("JH",),
    "l": ("L",), "ɹ": ("R",), "w": ("W",), "j": ("Y",), "h": ("HH",),
    "ɾ": (FLAP,),                          # "better": flap, T or D
    "ʔ": ("T",),                           # "button": glottal stop for /t/
    "r": ("R",),                           # trilled r in loans
    "x": ("HH",),                          # "loch"
    "ç": ("HH",),                          # ich-laut (kept before NFD strips the cedilla)
    "ɑ̃": ("AA",),                          # French loans
    "e": ("EY",),
    # --- British / other English centring diphthongs ----------------------
    "eə": ("EH", "AX"), "ʊə": ("UH", "AX"), "əʊ": ("OW",),
    # --- junk the tokenizer vocabulary contains ---------------------------
    "??": (OTHER,), "1": (OTHER,), "ʲ": (OTHER,),
}  # fmt: skip

#: IPA base segments for the generic rule, longest match first at run time.
#: Vowels go to the nearest English vowel, consonants to the nearest place
#: and manner the engine has.
SEGMENTS: dict[str, tuple[str, ...]] = {
    # vowels
    "i": ("IY",), "ɪ": ("IH",), "ᵻ": ("IH",), "ɨ": ("IH",),
    "e": ("EY",), "ɛ": ("EH",), "æ": ("AE",),
    "a": ("AA",), "ɑ": ("AA",), "ɒ": ("AA",),
    "ɔ": ("AO",), "o": ("OW",),
    "ʊ": ("UH",), "ɵ": ("UH",),
    "u": ("UW",), "ʉ": ("UW",), "y": ("UW",), "ɯ": ("UW",),
    "ʌ": ("AH",), "ə": ("AX",), "ɐ": ("AX",),
    "ɜ": ("ER",), "ɚ": ("ER",), "ø": ("ER",), "œ": ("ER",),
    # diphthongs (and how other languages spell the same glides)
    "aɪ": ("AY",), "ai": ("AY",), "aʊ": ("AW",), "au": ("AW",), "ɑu": ("AW",),
    "eɪ": ("EY",), "ei": ("EY",), "ɛɪ": ("EY",),
    "oʊ": ("OW",), "ou": ("OW",), "əʊ": ("OW",),
    "ɔɪ": ("OY",), "oɪ": ("OY",), "ɔø": ("OY",), "øi": ("OY",),
    # stops
    "p": ("P",), "b": ("B",), "t": ("T",), "ʈ": ("T",), "d": ("D",), "ɖ": ("D",),
    "k": ("K",), "q": ("K",), "c": ("K",), "ɡ": ("G",), "g": ("G",), "ɟ": ("G",),
    "ʔ": ("T",),
    # nasals
    "m": ("M",), "n": ("N",), "ɲ": ("N",), "ɳ": ("N",), "ŋ": ("NG",), "ɴ": ("NG",),
    "N": ("NG",),
    # fricatives
    "f": ("F",), "ɸ": ("F",), "v": ("V",), "ʋ": ("V",), "β": ("V",),
    "θ": ("TH",), "ð": ("DH",), "s": ("S",), "z": ("Z",),
    "ʃ": ("SH",), "ɕ": ("SH",), "ʂ": ("SH",), "S": ("SH",),
    "ʒ": ("ZH",), "ʑ": ("ZH",), "ʐ": ("ZH",),
    "h": ("HH",), "x": ("HH",), "χ": ("HH",), "ç": ("HH",), "ħ": ("HH",), "X": ("HH",),
    "ɣ": ("G",),
    # affricates
    "tʃ": ("CH",), "tɕ": ("CH",), "tS": ("CH",),
    "dʒ": ("JH",), "dʑ": ("JH",), "dZ": ("JH",),
    # liquids and glides
    "l": ("L",), "ɫ": ("L",), "ɭ": ("L",), "ʎ": ("L",),
    "ɹ": ("R",), "r": ("R",), "ʁ": ("R",), "ɻ": ("R",), "ɽ": ("R",),
    "ɾ": (FLAP,),
    "w": ("W",), "j": ("Y",), "ʝ": ("Y",),
}  # fmt: skip

#: Characters that modify a segment without changing its English category:
#: length, palatalisation, aspiration, pharyngealisation, rounding, and the
#: ASCII leftovers of eSpeak mnemonics (``.`` retroflex, ``[`` dental ...).
MODIFIER_CHARS = set("ː:ʲʰˤᵝ.[^\"")
_TONE_DIGITS = "12345"
_SEGMENT_KEYS = sorted(SEGMENTS, key=len, reverse=True)


def _strip_combining(s: str) -> str:
    """Drop combining marks (nasal tilde, syllabic, dental, raised ...)."""
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def _generic_fold(token: str) -> tuple[str, ...]:
    """Rule-based fold for tokens without an explicit entry."""
    s = token
    # Mandarin finals carry the tone as a trailing digit; tone 3 came out of
    # eSpeak's mnemonic table as "ɜ" (yɛɜ, onɡɜ ...), never as a vowel there.
    stripped_tone = s.rstrip(_TONE_DIGITS)
    if stripped_tone != s and stripped_tone:
        s = stripped_tone
    elif len(s) > 1 and s.endswith("ɜ"):
        s = s[:-1]
    s = _strip_combining(s)
    s = "".join(c for c in s if c not in MODIFIER_CHARS)
    # Aspiration written as a plain h after a stop/affricate: th, kh, tsh.
    if len(s) > 1 and s.endswith("h"):
        s = s[:-1]
    if not s:
        return (OTHER,)
    out: list[str] = []
    i = 0
    while i < len(s):
        for key in _SEGMENT_KEYS:
            if s.startswith(key, i):
                out.extend(SEGMENTS[key])
                i += len(key)
                break
        else:
            return (OTHER,)
    # Doubled letters are another spelling of length (ɐɐ, ää, dˤdˤ).
    dedup = [p for k, p in enumerate(out) if k == 0 or p != out[k - 1]]
    return tuple(dedup)


def fold_token(token: str) -> tuple[str, ...]:
    """Model/eSpeak IPA token -> scoring symbols.

    Special tokens fold to the empty tuple; anything unrecognisable to
    ``(OTH,)``.  Stress marks are ignored.
    """
    token = token.replace("ˈ", "").replace("ˌ", "")
    if token in SPECIAL_TOKENS or not token:
        return ()
    if token in EXPLICIT_FOLD:
        return EXPLICIT_FOLD[token]
    return _generic_fold(token)


@lru_cache(maxsize=1)
def load_vocab() -> dict[str, int]:
    """The model's tokenizer vocabulary (token -> id), bundled copy."""
    return json.loads(VOCAB_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def fold_table() -> dict[str, tuple[str, ...]]:
    """Fold of every vocabulary token."""
    return {tok: fold_token(tok) for tok in load_vocab()}


FOLD = fold_table  # alias: FOLD()[token]


def fold_tokens(tokens: list[str]) -> list[str]:
    """Decoded token sequence -> flat list of scoring symbols."""
    out: list[str] = []
    for t in tokens:
        out.extend(fold_token(t))
    return out


def ipa_to_arpabet(ipa: str) -> list[str]:
    """Fold a separated IPA string (``_`` between phonemes, spaces between
    words, as ``espeak-ng --ipa --sep=_`` prints it) to scoring symbols."""
    tokens = [t for t in re.split(r"[\s_]+", ipa.strip()) if t]
    return fold_tokens(tokens)


# --------------------------------------------------------------------------
# Engine side
# --------------------------------------------------------------------------

_STRESS = re.compile(r"\d+$")


def strip_stress(symbol: str) -> str:
    return _STRESS.sub("", symbol.upper())


def parse_arpabet(text: str) -> list[str]:
    """``"B AA1 T"`` or ``"[B AA1 T] [S IH1 P]"`` -> ``[B, AA, T, S, IH, P]``.

    Stress digits are stripped, ``SIL`` and unknown symbols dropped.
    """
    out = []
    for sym in re.sub(r"[\[\],.?!]", " ", text).split():
        s = strip_stress(sym)
        if s in ENGINE_PHONEMES or s == FLAP:
            out.append(s)
    return out


def parse_engine_line(line: str) -> list[str]:
    """Phonemes from the first stdout line of ``say``.

    ``say`` prints one symbol per *segment*: diphthongs and affricates are
    two segments, a voiceless stop before a vowel or approximant gets an
    extra aspiration segment, pauses are ``SIL``.  This undoes that, so
    ``AA P P AA B AY AY T`` reads ``AA P AA B AY T``.  A line that already
    has one symbol per phoneme passes through unchanged.
    """
    segs = [s for s in (strip_stress(x) for x in line.split()) if s in ENGINE_PHONEMES]
    out: list[str] = []
    i = 0
    while i < len(segs):
        s = segs[i]
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        after = segs[i + 2] if i + 2 < len(segs) else None
        out.append(s)
        if nxt == s and (s in DIPHTHONGS or s in AFFRICATES):
            i += 2
        elif nxt == s and s in ("P", "T", "K") and (after in VOCALIC or after in APPROXIMANTS):
            i += 2
        else:
            i += 1
    return out


def arpabet_to_tokens(phones: list[str], merge: bool = True) -> list[str]:
    """Engine phonemes -> model tokens, the way eSpeak en-us writes them.

    With ``merge`` the r-coloured / syllabic pairs become their single
    token (``K AA R`` -> ``k ɑːɹ``); without, one token per phoneme.
    """
    phones = [strip_stress(p) for p in phones if strip_stress(p) != "SIL"]
    out: list[str] = []
    i = 0
    while i < len(phones):
        pair = tuple(phones[i : i + 2])
        if merge and pair in MERGED_TOKENS:
            out.append(MERGED_TOKENS[pair])
            i += 2
        else:
            out.append(ARPA_TO_IPA[phones[i]])
            i += 1
    return out


@lru_cache(maxsize=1)
def variants() -> dict[str, list[str]]:
    """Scoring symbol -> every vocabulary token that folds to exactly it.

    The flap token is listed under ``T`` and ``D`` as well.
    """
    out: dict[str, list[str]] = {p: [] for p in SCORING_SYMBOLS}
    for tok, fold in fold_table().items():
        if len(fold) == 1:
            out[fold[0]].append(tok)
    out["T"] = out["T"] + out[FLAP]
    out["D"] = out["D"] + out[FLAP]
    return out


# --------------------------------------------------------------------------
# Scoring: canonical form and alignment
# --------------------------------------------------------------------------

#: Sequence rewrites applied to reference AND hypothesis.  Each one removes
#: a distinction that is a spelling convention, not a difference in sound.
CANONICAL_RULES: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    # r-coloured schwa: "AX R" / "AH R" are the two-symbol spelling of ER,
    # and eSpeak writes "around" as ɚ ɹ.
    (("ER", "R"), ("ER",)),
    (("AX", "R"), ("ER",)),
    (("AH", "R"), ("ER",)),
    # tense/lax is neutralised before /r/ in American English: eSpeak writes
    # "here" ɪɹ, "poor" ʊɹ, "four" ɔːɹ; dictionaries often IY R, UW R, OW R.
    (("IY", "R"), ("IH", "R")),
    (("UW", "R"), ("UH", "R")),
    (("OW", "R"), ("AO", "R")),
    (("EY", "R"), ("EH", "R")),
]


def canonical(phones: list[str]) -> list[str]:
    """Scoring form of a phoneme sequence (see :data:`CANONICAL_RULES`)."""
    seq = [strip_stress(p) for p in phones]
    seq = [p for p in seq if p in SCORING_SYMBOLS]
    out: list[str] = []
    i = 0
    while i < len(seq):
        for pat, rep in CANONICAL_RULES:
            if tuple(seq[i : i + len(pat)]) == pat:
                out.extend(rep)
                i += len(pat)
                break
        else:
            out.append(seq[i])
            i += 1
    return out


def same(a: str, b: str) -> bool:
    """Do two scoring symbols count as the same sound?  The flap matches
    T and D (it is an allophone of both)."""
    if a == b:
        return True
    return (a == FLAP and b in ("T", "D")) or (b == FLAP and a in ("T", "D"))


def align(ref: list[str], hyp: list[str]) -> list[tuple[str, str | None, str | None]]:
    """Minimum-edit alignment.  Returns ``(op, ref_symbol, hyp_symbol)``
    with op in ``ok``, ``sub``, ``del`` (in ref, not heard), ``ins``."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        row, prev = d[i], d[i - 1]
        for j in range(1, m + 1):
            cost = 0 if same(ri, hyp[j - 1]) else 1
            row[j] = min(prev[j - 1] + cost, prev[j] + 1, row[j - 1] + 1)
    ops: list[tuple[str, str | None, str | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if same(ref[i - 1], hyp[j - 1]) else 1
            if d[i][j] == d[i - 1][j - 1] + cost:
                ops.append(("ok" if cost == 0 else "sub", ref[i - 1], hyp[j - 1]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    return sum(1 for op, _, _ in align(ref, hyp) if op != "ok")


def error_rate(ref: list[str], hyp: list[str]) -> float:
    """Edits / reference length, on canonical forms."""
    r, h = canonical(ref), canonical(hyp)
    return edit_distance(r, h) / max(1, len(r))


# --------------------------------------------------------------------------
# Table dump
# --------------------------------------------------------------------------


def render_fold_table() -> str:
    vocab = load_vocab()
    lines = ["id\ttoken\tfold\tsource"]
    for tok, idx in sorted(vocab.items(), key=lambda kv: kv[1]):
        if tok in SPECIAL_TOKENS:
            fold, src = "-", "special"
        else:
            fold = " ".join(fold_token(tok))
            src = "explicit" if tok in EXPLICIT_FOLD else "rule"
        lines.append(f"{idx}\t{tok}\t{fold}\t{src}")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    if "--dump" in argv:
        FOLD_TABLE_PATH.write_text(render_fold_table(), encoding="utf-8")
        print(f"wrote {FOLD_TABLE_PATH}")
        return 0
    table = fold_table()
    other = [t for t, f in table.items() if f == (OTHER,)]
    multi = [t for t, f in table.items() if len(f) > 1]
    print(f"{len(table)} tokens: {len(other)} fold to OTH, {len(multi)} to several phonemes")
    for p in ENGINE_PHONEMES:
        print(f"{p:3} {ARPA_TO_IPA[p]:3} <- {' '.join(variants()[p])}")
    print("OTH <-", " ".join(other))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
