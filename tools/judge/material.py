"""Test material: what the systems are asked to say.

``python material.py build`` (re)writes ``material/*.tsv``; the files are
committed, so normal runs only read them.  Columns:

    id  set  text  expected  ortho

* ``text``     -- what is handed to the engine (``[ARPAbet]`` groups or plain
                  English),
* ``expected`` -- the phonemes a listener should hear, in the scoring
                  alphabet (ARPAbet, no stress digits),
* ``ortho``    -- the written word/sentence for Whisper (empty for VCVs).

Sets: ``vcv``, ``pair``, ``word``, ``sentence_tune``, ``sentence_heldout``
and ``human`` (written by ``anchors.py``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arpabet as A  # noqa: E402

HERE = Path(__file__).resolve().parent
MATERIAL_DIR = HERE / "material"
SYNTH_SETS = ["vcv", "pair", "word", "sentence_tune", "sentence_heldout"]
ALL_SETS = SYNTH_SETS + ["human"]
HEADER = ["id", "set", "text", "expected", "ortho"]


@dataclass(frozen=True)
class Item:
    id: str
    set: str
    text: str
    expected: tuple[str, ...]
    ortho: str = ""


# --------------------------------------------------------------------------
# Content
# --------------------------------------------------------------------------

VCV_VOWELS = ["AA", "IY", "UW"]

#: A second vowel-bearing frame for each vowel: hVd and bVt.  AX only occurs
#: unstressed, so it gets real words ("sofa", "about").
AX_FRAMES = [("sofa", "S OW1 F AX"), ("about", "AX B AW1 T")]

#: Minimal pairs: (word, ARPAbet, word, ARPAbet, contrast).
PAIRS = [
    ("bat", "B AE1 T", "pat", "P AE1 T", "B/P"),
    ("tip", "T IH1 P", "dip", "D IH1 P", "T/D"),
    ("coat", "K OW1 T", "goat", "G OW1 T", "K/G"),
    ("fan", "F AE1 N", "van", "V AE1 N", "F/V"),
    ("sip", "S IH1 P", "zip", "Z IH1 P", "S/Z"),
    ("see", "S IY1", "she", "SH IY1", "S/SH"),
    ("thin", "TH IH1 N", "tin", "T IH1 N", "TH/T"),
    ("then", "DH EH1 N", "den", "D EH1 N", "DH/D"),
    ("cheap", "CH IY1 P", "jeep", "JH IY1 P", "CH/JH"),
    ("ship", "SH IH1 P", "chip", "CH IH1 P", "SH/CH"),
    ("rock", "R AA1 K", "lock", "L AA1 K", "R/L"),
    ("wet", "W EH1 T", "yet", "Y EH1 T", "W/Y"),
    ("map", "M AE1 P", "nap", "N AE1 P", "M/N"),
    ("sin", "S IH1 N", "sing", "S IH1 NG", "N/NG"),
    ("heat", "HH IY1 T", "eat", "IY1 T", "HH/-"),
    ("beat", "B IY1 T", "bit", "B IH1 T", "IY/IH"),
    ("pen", "P EH1 N", "pan", "P AE1 N", "EH/AE"),
    ("pull", "P UH1 L", "pool", "P UW1 L", "UH/UW"),
    ("cup", "K AH1 P", "cop", "K AA1 P", "AH/AA"),
    ("late", "L EY1 T", "light", "L AY1 T", "EY/AY"),
    ("three", "TH R IY1", "free", "F R IY1", "TH/F"),
    ("boat", "B OW1 T", "bought", "B AO1 T", "OW/AO"),
]

#: Common words covering every engine phoneme.
WORDS = [
    ("water", "W AO1 T ER"),
    ("people", "P IY1 P AX L"),
    ("about", "AX B AW1 T"),
    ("because", "B IH K AH1 Z"),
    ("little", "L IH1 T AX L"),
    ("mother", "M AH1 DH ER"),
    ("father", "F AA1 DH ER"),
    ("thing", "TH IH1 NG"),
    ("house", "HH AW1 S"),
    ("school", "S K UW1 L"),
    ("music", "M Y UW1 Z IH K"),
    ("yellow", "Y EH1 L OW"),
    ("children", "CH IH1 L D R AX N"),
    ("measure", "M EH1 ZH ER"),
    ("church", "CH ER1 CH"),
    ("judge", "JH AH1 JH"),
    ("young", "Y AH1 NG"),
    ("thought", "TH AO1 T"),
    ("voice", "V OY1 S"),
    ("around", "AX R AW1 N D"),
    ("world", "W ER1 L D"),
    ("number", "N AH1 M B ER"),
    ("picture", "P IH1 K CH ER"),
    ("money", "M AH1 N IY"),
    ("garden", "G AA1 R D AX N"),
    ("river", "R IH1 V ER"),
    ("window", "W IH1 N D OW"),
    ("table", "T EY1 B AX L"),
    ("paper", "P EY1 P ER"),
    ("orange", "AO1 R IH N JH"),
    ("city", "S IH1 T IY"),
    ("happy", "HH AE1 P IY"),
    ("question", "K W EH1 S CH AX N"),
    ("language", "L AE1 NG G W IH JH"),
    ("together", "T AX G EH1 DH ER"),
    ("morning", "M AO1 R N IH NG"),
    ("between", "B IH T W IY1 N"),
    ("good", "G UH1 D"),
    ("family", "F AE1 M AX L IY"),
    ("beautiful", "B Y UW1 T AX F AX L"),
    ("today", "T AX D EY1"),
    ("zero", "Z IY1 R OW"),
]

#: Harvard sentences (IEEE 1969, public domain), lists 1 and 2.
SENTENCES_TUNE = [
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It's easy to tell the depth of a well.",
    "These days a chicken leg is a rare dish.",
    "Rice is often served in round bowls.",
    "The juice of lemons makes fine punch.",
    "The box was thrown beside the parked truck.",
    "The hogs were fed chopped corn and garbage.",
    "Four hours of steady work faced us.",
    "A large size in stockings is hard to sell.",
]
SENTENCES_HELDOUT = [
    "The boy was there when the sun rose.",
    "A rod is used to catch pink salmon.",
    "The source of the huge river is the clear spring.",
    "Kick the ball straight and follow through.",
    "Help the woman get back to her feet.",
    "A pot of tea helps to pass the evening.",
    "Smoky fires lack flame and heat.",
    "The soft cushion broke the man's fall.",
    "The salt breeze came across from the sea.",
    "The girl at the booth sold fifty bonds.",
]


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def _bracket(arpa: str) -> str:
    return f"[{arpa}]"


def build_vcv() -> list[Item]:
    items = []
    for c in A.CONSONANTS:
        for v in VCV_VOWELS:
            arpa = f"{v} {c} {v}1"  # stress on the second vowel: C is an onset
            items.append(Item(f"vcv_{v.lower()}_{c.lower()}", "vcv", _bracket(arpa), tuple(A.parse_arpabet(arpa))))
    for v in A.VOWELS + A.DIPHTHONGS:
        if v == "AX":
            for word, arpa in AX_FRAMES:
                items.append(Item(f"vow_ax_{word}", "vcv", _bracket(arpa), tuple(A.parse_arpabet(arpa))))
            continue
        for name, frame in (("hvd", "HH {}1 D"), ("bvt", "B {}1 T")):
            arpa = frame.format(v)
            items.append(Item(f"vow_{name}_{v.lower()}", "vcv", _bracket(arpa), tuple(A.parse_arpabet(arpa))))
    return items


def build_pair() -> list[Item]:
    items = []
    for w1, a1, w2, a2, _ in PAIRS:
        for w, a in ((w1, a1), (w2, a2)):
            items.append(Item(f"pair_{w}", "pair", _bracket(a), tuple(A.parse_arpabet(a)), w))
    return items


def build_word() -> list[Item]:
    return [Item(f"word_{w}", "word", _bracket(a), tuple(A.parse_arpabet(a)), w) for w, a in WORDS]


def build_sentences(name: str, sentences: list[str]) -> list[Item]:
    import espeak_util as E

    prefix = "tune" if name.endswith("tune") else "held"
    return [
        Item(f"sent_{prefix}_{k + 1:02d}", name, s, tuple(E.phonemes(s)), s) for k, s in enumerate(sentences)
    ]


def write_tsv(path: Path, items: list[Item]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(HEADER)]
    for it in items:
        for field in (it.id, it.text, it.ortho):
            assert "\t" not in field and "\n" not in field, field
        lines.append("\t".join([it.id, it.set, it.text, " ".join(it.expected), it.ortho]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_tsv(path: Path) -> list[Item]:
    rows = path.read_text(encoding="utf-8").splitlines()
    assert rows[0].split("\t") == HEADER, f"{path}: unexpected header"
    items = []
    for row in rows[1:]:
        if not row.strip():
            continue
        f = row.split("\t") + [""] * 5
        items.append(Item(f[0], f[1], f[2], tuple(f[3].split()), f[4]))
    return items


def load(sets: list[str] | None = None) -> list[Item]:
    """Items of the given sets (default: every synthesizer set), in order."""
    out: list[Item] = []
    for name in sets or SYNTH_SETS:
        path = MATERIAL_DIR / f"{name}.tsv"
        if not path.exists():
            hint = "anchors.py fetch" if name == "human" else "material.py build"
            raise FileNotFoundError(f"{path} is missing: run `uv run python {hint}`")
        out.extend(read_tsv(path))
    return out


def build() -> None:
    sets = {
        "vcv": build_vcv(),
        "pair": build_pair(),
        "word": build_word(),
        "sentence_tune": build_sentences("sentence_tune", SENTENCES_TUNE),
        "sentence_heldout": build_sentences("sentence_heldout", SENTENCES_HELDOUT),
    }
    for name, items in sets.items():
        write_tsv(MATERIAL_DIR / f"{name}.tsv", items)
        n_ph = sum(len(i.expected) for i in items)
        print(f"{name:17} {len(items):4d} items {n_ph:5d} phonemes")
    covered = {p for items in sets.values() for it in items for p in it.expected}
    missing = [p for p in A.ENGINE_PHONEMES if p not in covered]
    print("phonemes never expected:", missing or "none")


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()
    else:
        for it in load(sys.argv[1:] or None):
            print(it.id, it.set, it.text, " ".join(it.expected), it.ortho, sep="\t")
