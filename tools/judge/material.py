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
and ``human`` (written by ``anchors.py``); the probe sets ``vowel`` and
``voicing`` (``probes.py``) have a sixth column, ``meta``: ``key=value``
pairs separated by ``;`` (the scored slot, the intended vowel, the
minimal-pair twin ...).
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
#: Targeted sets for ``probes.py`` (not part of the default scoreboard).
#: ``harvard_*`` and ``carrier_*`` are the vowel tuner's material: a tuning
#: split it optimises on and a held-out split it never sees.
PROBE_SETS = ["vowel", "voicing", "harvard_tune", "harvard_heldout", "carrier_tune", "carrier_heldout"]
#: Sets of whole sentences: the probes score every slot in them.
SENTENCE_SETS = ["human", "sentence_tune", "sentence_heldout", "harvard_tune", "harvard_heldout"]
ALL_SETS = SYNTH_SETS + ["human"] + PROBE_SETS
HEADER = ["id", "set", "text", "expected", "ortho"]
META = "meta"


@dataclass(frozen=True)
class Item:
    id: str
    set: str
    text: str
    expected: tuple[str, ...]
    ortho: str = ""
    meta: str = ""  # probe sets: "key=value;key=value"

    def info(self) -> dict[str, str]:
        """The ``meta`` column as a dict."""
        return dict(kv.split("=", 1) for kv in self.meta.split(";") if kv)


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

#: More Harvard sentences for the vowel tuner: lists 3 and 4 to tune on...
HARVARD_TUNE = [
    "The small pup gnawed a hole in the sock.",
    "The fish twisted and turned on the bent hook.",
    "Press the pants and sew a button on the vest.",
    "The swan dive was far short of perfect.",
    "The beauty of the view stunned the young boy.",
    "Two blue fish swam in the tank.",
    "Her purse was full of useless trash.",
    "The colt reared and threw the tall rider.",
    "It snowed, rained, and hailed the same morning.",
    "Read verse out loud for pleasure.",
    "Hoist the load to your left shoulder.",
    "Take the winding path to reach the lake.",
    "Note closely the size of the gas tank.",
    "Wipe the grease off his dirty face.",
    "Mend the coat before you go out.",
    "The wrist was badly strained and hung limp.",
    "The stray cat gave birth to kittens.",
    "The young girl gave no clear response.",
    "The meal was cooked before the bell rang.",
    "What joy there is in living.",
]

#: ...and lists 5 and 6, held out.
HARVARD_HELDOUT = [
    "A king ruled the state in the early days.",
    "The ship was torn apart on the sharp reef.",
    "Sickness kept him home the third week.",
    "The wide road shimmered in the hot sun.",
    "The lazy cow lay in the cool grass.",
    "Lift the square stone over the fence.",
    "The rope will bind the seven books at once.",
    "Hop over the fence and plunge in.",
    "The friendly gang left the drug store.",
    "Mesh wire keeps chicks inside.",
    "The frosty air passed through the coat.",
    "The crooked maze failed to fool the mouse.",
    "Adding fast leads to wrong sums.",
    "The show was a flop from the very start.",
    "A saw is a tool used for making boards.",
    "The wagon moved on well oiled wheels.",
    "March the soldiers past the next hill.",
    "A cup of sugar makes sweet fudge.",
    "Place a rosebush near the porch steps.",
    "Both lost their lives in the raging storm.",
]

# --- carrier phrases (vowel tuner) -------------------------------------------

#: Every stressed vowel in CVC syllables inside short carrier phrases, given
#: as phonemes; only the CVC's vowel is scored.  The tuning and held-out
#: splits share no carrier and no consonant frame.
CARRIER_SPLITS = {
    "carrier_tune": {
        "carriers": [("say", "[S EY1] {} [AX0 G EH1 N]."), ("word", "[DH AX0] [W ER1 D] [IH0 Z] {}.")],
        "frames": [("bvd", "B {}1 D"), ("hvd", "HH {}1 D"), ("kvt", "K {}1 T"), ("svp", "S {}1 P")],
        "ax": [("sofa", "S OW1 F AX", 3), ("panda", "P AE1 N D AX", 4), ("comma", "K AA1 M AX", 3)],
    },
    "carrier_heldout": {
        "carriers": [("said", "[AY1] [S EH1 D] {} [T AX0 D EY1]."), ("write", "[R AY1 T] {} [W AH1 N S].")],
        "frames": [("pvt", "P {}1 T"), ("dvg", "D {}1 G"), ("fvn", "F {}1 N"), ("tvk", "T {}1 K")],
        "ax": [("about", "AX B AW1 T", 0), ("ago", "AX G OW1", 0), ("salad", "S AE1 L AX D", 3)],
    },
}

# --- vowel probe ----------------------------------------------------------

#: Every English monophthong and diphthong, in the order of the report.
VOWEL_TARGETS = ["IY", "IH", "EH", "AE", "AH", "AA", "AO", "UH", "UW", "ER", "AX", "EY", "AY", "OW", "AW", "OY"]

#: Frames for the stressed vowels: the vowel alone, then five CVC frames.
VOWEL_FRAMES = [
    ("iso", "{}1"),
    ("hvd", "HH {}1 D"),
    ("bvd", "B {}1 D"),
    ("bvt", "B {}1 T"),
    ("svt", "S {}1 T"),
    ("dvd", "D {}1 D"),
]

#: Real English words for (frame, vowel); every other frame item is labelled
#: nonsense (the engine is given phonemes either way).
VOWEL_WORDS = {
    "hvd": {"IY": "heed", "IH": "hid", "EH": "head", "AE": "had", "AH": "hud", "AA": "hod", "AO": "hawed",
            "UH": "hood", "UW": "who'd", "ER": "heard", "EY": "hayed", "AY": "hide", "OW": "hoed", "AW": "how'd"},
    "bvd": {"IY": "bead", "IH": "bid", "EH": "bed", "AE": "bad", "AH": "bud", "AO": "bawd", "UW": "booed",
            "ER": "bird", "EY": "bayed", "AY": "bide", "OW": "bode", "AW": "bowed"},
    "bvt": {"IY": "beat", "IH": "bit", "EH": "bet", "AE": "bat", "AH": "but", "AA": "bot", "AO": "bought",
            "UW": "boot", "EY": "bait", "AY": "bite", "OW": "boat", "AW": "bout"},
    "svt": {"IY": "seat", "IH": "sit", "EH": "set", "AE": "sat", "AA": "sot", "AO": "sought", "UH": "soot",
            "UW": "suit", "EY": "sate", "AY": "site"},
    "dvd": {"IY": "deed", "IH": "did", "EH": "dead", "AE": "dad", "AH": "dud", "UW": "dude", "AY": "died"},
}  # fmt: skip

#: AX is never stressed, so besides the isolated vowel it gets real words
#: where it is the unstressed syllable: (word, ARPAbet, index of the AX).
AX_WORDS = [
    ("sofa", "S OW1 F AX", 3),
    ("about", "AX B AW1 T", 0),
    ("comma", "K AA1 M AX", 3),
    ("panda", "P AE1 N D AX", 4),
    ("ago", "AX G OW1", 0),
]

# --- voicing probe ---------------------------------------------------------

#: Voiceless/voiced obstruent pairs.  F/V is reported separately.
VOICING_PAIRS = [("P", "B"), ("T", "D"), ("K", "G"), ("S", "Z"), ("F", "V")]

#: Real minimal pairs: (pair, position, voiceless word, ARPAbet, voiced
#: word, ARPAbet).  The pair differs in exactly one phoneme.  Medial /t d/
#: has no real pair here: American English flaps it ("latter"/"ladder" are
#: both DX), so it is "a tie"/"a die" with the stop before a stressed vowel.
VOICING_WORDS = [
    ("P/B", "initial", "pat", "P AE1 T", "bat", "B AE1 T"),
    ("P/B", "initial", "pin", "P IH1 N", "bin", "B IH1 N"),
    ("P/B", "initial", "pole", "P OW1 L", "bowl", "B OW1 L"),
    ("P/B", "medial", "rapid", "R AE1 P IH D", "rabid", "R AE1 B IH D"),
    ("P/B", "medial", "ample", "AE1 M P AX L", "amble", "AE1 M B AX L"),
    ("P/B", "final", "rope", "R OW1 P", "robe", "R OW1 B"),
    ("P/B", "final", "cap", "K AE1 P", "cab", "K AE1 B"),
    ("P/B", "final", "mop", "M AA1 P", "mob", "M AA1 B"),
    ("T/D", "initial", "tie", "T AY1", "die", "D AY1"),
    ("T/D", "initial", "tip", "T IH1 P", "dip", "D IH1 P"),
    ("T/D", "initial", "ten", "T EH1 N", "den", "D EH1 N"),
    ("T/D", "medial", "a tie", "AX T AY1", "a die", "AX D AY1"),
    ("T/D", "medial", "a ten", "AX T EH1 N", "a den", "AX D EH1 N"),
    ("T/D", "final", "bat", "B AE1 T", "bad", "B AE1 D"),
    ("T/D", "final", "seat", "S IY1 T", "seed", "S IY1 D"),
    ("T/D", "final", "write", "R AY1 T", "ride", "R AY1 D"),
    ("K/G", "initial", "coat", "K OW1 T", "goat", "G OW1 T"),
    ("K/G", "initial", "cold", "K OW1 L D", "gold", "G OW1 L D"),
    ("K/G", "initial", "curl", "K ER1 L", "girl", "G ER1 L"),
    ("K/G", "medial", "bicker", "B IH1 K ER", "bigger", "B IH1 G ER"),
    ("K/G", "medial", "decree", "D IH K R IY1", "degree", "D IH G R IY1"),
    ("K/G", "final", "back", "B AE1 K", "bag", "B AE1 G"),
    ("K/G", "final", "pick", "P IH1 K", "pig", "P IH1 G"),
    ("K/G", "final", "duck", "D AH1 K", "dug", "D AH1 G"),
    ("S/Z", "initial", "sip", "S IH1 P", "zip", "Z IH1 P"),
    ("S/Z", "initial", "sue", "S UW1", "zoo", "Z UW1"),
    ("S/Z", "initial", "seal", "S IY1 L", "zeal", "Z IY1 L"),
    ("S/Z", "medial", "racer", "R EY1 S ER", "razor", "R EY1 Z ER"),
    ("S/Z", "medial", "fussy", "F AH1 S IY", "fuzzy", "F AH1 Z IY"),
    ("S/Z", "final", "bus", "B AH1 S", "buzz", "B AH1 Z"),
    ("S/Z", "final", "rice", "R AY1 S", "rise", "R AY1 Z"),
    ("S/Z", "final", "place", "P L EY1 S", "plays", "P L EY1 Z"),
    ("F/V", "initial", "fan", "F AE1 N", "van", "V AE1 N"),
    ("F/V", "initial", "fine", "F AY1 N", "vine", "V AY1 N"),
    ("F/V", "initial", "fat", "F AE1 T", "vat", "V AE1 T"),
    ("F/V", "medial", "rifle", "R AY1 F AX L", "rival", "R AY1 V AX L"),
    ("F/V", "medial", "safer", "S EY1 F ER", "saver", "S EY1 V ER"),
    ("F/V", "final", "leaf", "L IY1 F", "leave", "L IY1 V"),
    ("F/V", "final", "safe", "S EY1 F", "save", "S EY1 V"),
    ("F/V", "final", "half", "HH AE1 F", "halve", "HH AE1 V"),
]

#: Nonsense VCV syllables for every pair (the consonant is the onset of the
#: stressed second vowel, as in the ``vcv`` set).
VOICING_VCV_VOWELS = ["AA", "IY", "UW"]


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


def build_sentences(name: str, sentences: list[str], stem: str = "sent") -> list[Item]:
    import espeak_util as E

    prefix = "tune" if name.endswith("tune") else "held"
    return [
        Item(f"{stem}_{prefix}_{k + 1:02d}", name, s, tuple(E.phonemes(s)), s) for k, s in enumerate(sentences)
    ]


def build_carrier(name: str) -> list[Item]:
    """Every stressed vowel in the split's CVC frames inside each of its
    carrier phrases, and AX unstressed in real words.  ``slot`` is the index
    of the scored vowel in ``expected``."""
    split = CARRIER_SPLITS[name]
    prefix = "tune" if name.endswith("tune") else "held"
    items = []
    for cname, carrier in split["carriers"]:
        before = A.parse_arpabet(" ".join(carrier.split("{}")[0].replace("[", " ").replace("]", " ").split()))
        words = [(v, f, pat.format(v)) for v in VOWEL_TARGETS if v != "AX" for f, pat in split["frames"]]
        words += [("AX", w, arpa) for w, arpa, _ in split["ax"]]
        for v, frame, arpa in words:
            text = carrier.format(_bracket(arpa))
            phones = A.parse_arpabet(" ".join(text.replace("[", " ").replace("]", " ").replace(".", " ").split()))
            inner = A.parse_arpabet(arpa)
            at = next(k for w, a, k in split["ax"] if w == frame) if v == "AX" else inner.index(v)
            meta = _meta(target=v, slot=len(before) + at, frame=frame, carrier=cname, lex="carrier")
            items.append(Item(f"car_{prefix}_{cname}_{frame}_{v.lower()}", name, text, tuple(phones), "", meta))
    return items


def _meta(**kv) -> str:
    return ";".join(f"{k}={v}" for k, v in kv.items())


def build_vowel() -> list[Item]:
    """Every vowel alone and in five CVC frames (AX: alone and in five
    words where it is unstressed).  ``slot`` is the index of the vowel in
    ``expected``."""
    items = []
    for v in VOWEL_TARGETS:
        if v == "AX":
            arpa = "AX"
            items.append(Item("vowel_iso_ax", "vowel", _bracket(arpa), ("AX",), "",
                              _meta(target="AX", slot=0, frame="iso", lex="isolated")))  # fmt: skip
            for word, arpa, slot in AX_WORDS:
                items.append(
                    Item(f"vowel_word_ax_{word}", "vowel", _bracket(arpa), tuple(A.parse_arpabet(arpa)), word,
                         _meta(target="AX", slot=slot, frame="word", lex="word"))  # fmt: skip
                )
            continue
        for frame, pattern in VOWEL_FRAMES:
            arpa = pattern.format(v)
            phones = A.parse_arpabet(arpa)
            word = VOWEL_WORDS.get(frame, {}).get(v, "")
            lex = "isolated" if frame == "iso" else ("word" if word else "nonsense")
            items.append(
                Item(f"vowel_{frame}_{v.lower()}", "vowel", _bracket(arpa), tuple(phones), word,
                     _meta(target=v, slot=phones.index(v), frame=frame, lex=lex))  # fmt: skip
            )
    return items


def _swap(phones: list[str], slot: int, sym: str) -> list[str]:
    out = list(phones)
    out[slot] = sym
    return out


def build_voicing() -> list[Item]:
    """Real minimal pairs in three positions plus VCV syllables, both members
    of every pair.  ``twin`` is the other member's phonemes (one phoneme
    different, at ``slot``)."""
    voiced_of = dict(VOICING_PAIRS)
    items = []

    def add(group: str, pos: str, word: str, arpa: str, lex: str, ident: str):
        phones = A.parse_arpabet(arpa)
        pair = group.split("/")
        slot = next(k for k, p in enumerate(phones) if p in pair)
        target = phones[slot]
        other = pair[1] if target == pair[0] else pair[0]
        voicing = "voiceless" if target in voiced_of else "voiced"
        meta = _meta(pair=group, pos=pos, slot=slot, target=target, twin=" ".join(_swap(phones, slot, other)),
                     voicing=voicing, lex=lex)  # fmt: skip
        items.append(Item(ident, "voicing", _bracket(arpa), tuple(phones), word if lex == "word" else "", meta))

    for group, pos, w1, a1, w2, a2 in VOICING_WORDS:
        tag = group.replace("/", "").lower()
        for w, a in ((w1, a1), (w2, a2)):
            add(group, pos, w, a, "word", f"voicing_{tag}_{pos}_{w.replace(' ', '_')}")
    for vl, vd in VOICING_PAIRS:
        group = f"{vl}/{vd}"
        tag = group.replace("/", "").lower()
        for v in VOICING_VCV_VOWELS:
            for c in (vl, vd):
                add(group, "vcv", "", f"{v} {c} {v}1", "vcv", f"voicing_{tag}_vcv_{v.lower()}_{c.lower()}")
    return items


def write_tsv(path: Path, items: list[Item]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with_meta = any(it.meta for it in items)
    lines = ["\t".join(HEADER + ([META] if with_meta else []))]
    for it in items:
        for field in (it.id, it.text, it.ortho, it.meta):
            assert "\t" not in field and "\n" not in field, field
        row = [it.id, it.set, it.text, " ".join(it.expected), it.ortho] + ([it.meta] if with_meta else [])
        lines.append("\t".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_tsv(path: Path) -> list[Item]:
    rows = path.read_text(encoding="utf-8").splitlines()
    head = rows[0].split("\t")
    assert head in (HEADER, HEADER + [META]), f"{path}: unexpected header"
    items = []
    for row in rows[1:]:
        if not row.strip():
            continue
        f = row.split("\t") + [""] * 6
        items.append(Item(f[0], f[1], f[2], tuple(f[3].split()), f[4], f[5]))
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
        "vowel": build_vowel(),
        "voicing": build_voicing(),
        "harvard_tune": build_sentences("harvard_tune", HARVARD_TUNE, "harv"),
        "harvard_heldout": build_sentences("harvard_heldout", HARVARD_HELDOUT, "harv"),
        "carrier_tune": build_carrier("carrier_tune"),
        "carrier_heldout": build_carrier("carrier_heldout"),
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
