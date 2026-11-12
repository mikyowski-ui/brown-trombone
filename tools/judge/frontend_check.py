"""Compare the engine's text front end with eSpeak NG, word by word.

    uv run python frontend_check.py                  # every word of every material file
    uv run python frontend_check.py canoe smooth     # just these words
    uv run python frontend_check.py --all            # also list the benign differences

For each word the engine's phonemes are the first stdout line of ``bin/say``
(``SAY_BIN``) given the plain word, one symbol per segment, undone by
``arpabet.parse_engine_line``; eSpeak's are ``espeak-ng -v en-us --ipa``
folded by ``arpabet.py``.  Both are put in canonical form and aligned.

Differences that are notation or accent, not mistakes, are *benign* and do
not count: AX/AH/IH (unstressed vowels), IY/IH (final -y), AA/AO (the
cot-caught merger), the flap for T or D.  Words with any other difference
are listed, most edits first, with both transcriptions.  Results go to
``out/frontend.json`` and ``report/frontend.md``.

The engine's front end is ``phonemize`` in ``crates/synthie-trombone/src/
speech.rs`` (the ``lookup`` dictionary, then ``letter_to_sound``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import arpabet as A  # noqa: E402
import espeak_util as E  # noqa: E402
import material  # noqa: E402
import render  # noqa: E402

BENIGN = [{"AX", "AH", "IH"}, {"IY", "IH"}, {"AA", "AO"}, {"T", "DX"}, {"D", "DX"}]

#: Words whose difference is eSpeak's, or an accepted American variant:
#: listed separately, not counted as engine errors.
NOT_ENGINE = {
    "a": "eSpeak reads a lone 'a' as the letter name EY",
    "m": "a lone letter (from 'man's' split at the apostrophe); eSpeak spells it",
    "rabid": "eSpeak's error: rabid is R AE B IH D",
    "ruggedo": "a name; no reference",
    "was": "W AA Z is a common US form",
    "from": "F R AA M is a common US form",
    "your": "Y AO R is the usual US form",
    "our": "AW R is a common US form",
    "zero": "Z IH R OW is the usual US form (eSpeak's IY ER is the odd one)",
    "than": "stressed form of a function word",
    "often": "the T is an accepted variant",
    "dominions": "unstressed vowel quality",
    "himself": "unstressed vowel quality",
}
TEXT_SETS = ["sentence_tune", "sentence_heldout", "human"]


def benign(a: str | None, b: str | None) -> bool:
    return a is not None and b is not None and any(a in s and b in s for s in BENIGN)


def material_words() -> list[str]:
    words: dict[str, None] = {}
    for name in material.ALL_SETS:
        try:
            items = material.load([name])
        except FileNotFoundError:
            continue
        for it in items:
            for w in re.findall(r"[a-z']+", it.ortho.lower()):
                w = w.strip("'")
                if w:
                    words[w] = None
    return list(words)


def text_counts() -> dict[str, int]:
    """How often each word is read as plain text by the engine (sentences)."""
    n: dict[str, int] = {}
    for it in material.load(TEXT_SETS):
        for w in re.findall(r"[a-z']+", it.text.lower()):
            w = w.strip("'")
            n[w] = n.get(w, 0) + 1
    return n


def engine_phones(word: str, tmp: Path) -> list[str]:
    out = tmp / f"{abs(hash(word))}.wav"
    res = subprocess.run(
        [str(render.say_bin()), "--engine", "physical", "--out", str(out), word], capture_output=True, text=True
    )
    if res.returncode != 0:
        raise RuntimeError(f"say failed for {word!r}: {res.stderr.strip()[-200:]}")
    out.unlink(missing_ok=True)
    return A.canonical(A.parse_engine_line(res.stdout.splitlines()[0] if res.stdout else ""))


def compare(words: list[str], jobs: int | None = None) -> list[dict]:
    with tempfile.TemporaryDirectory(dir=render.scratch_dir("frontend-")) as d:
        tmp = Path(d)
        with ThreadPoolExecutor(max_workers=jobs or os.cpu_count() or 4) as pool:
            eng = list(pool.map(lambda w: engine_phones(w, tmp), words))
    counts = text_counts()
    rows = []
    for w, e in zip(words, eng):
        ref = A.canonical(E.phonemes(w))
        ops = A.align(ref, e)
        edits = [(op, r, h) for op, r, h in ops if op != "ok"]
        real = [(op, r, h) for op, r, h in edits if not (op == "sub" and benign(r, h))]
        rows.append({
            "word": w,
            "espeak": " ".join(ref),
            "engine": " ".join(e),
            "edits": len(edits),
            "errors": len(real),
            "diff": ", ".join(f"{r or '-'}→{h or '-'}" for _, r, h in real),
            "in_text": counts.get(w, 0),
            "not_engine": NOT_ENGINE.get(w),
        })  # fmt: skip
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words", nargs="*")
    ap.add_argument("--all", action="store_true", help="also list words with only benign differences")
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args(argv)
    words = args.words or material_words()
    rows = compare(words, args.jobs)
    bad = sorted((r for r in rows if r["errors"] and not r["not_engine"]),
                 key=lambda r: (-(r["in_text"] > 0), -r["errors"], -r["in_text"], r["word"]))  # fmt: skip
    other = [r for r in rows if r["errors"] and r["not_engine"]]
    soft = [r for r in rows if r["edits"] and not r["errors"]]
    spoken = sum(1 for r in bad if r["in_text"])
    L = [
        "# Engine text front end vs eSpeak NG",
        "",
        f"{len(rows)} words of the material; **{len(bad)} look like engine errors** ({spoken} of them in the "
        f"sentences the engine reads as text: `{', '.join(TEXT_SETS)}`; the other sets are given as ARPAbet), "
        f"{len(other)} differ for another reason, {len(soft)} only in notation (AX/AH/IH, IY/IH, AA/AO, flap). "
        "Engine front end: `crates/synthie-trombone/src/speech.rs` (`phonemize` -> `push_word` -> the `lookup` "
        "dictionary, else `letter_to_sound`).",
        "",
        "| word | read as text | eSpeak | engine | differences |",
        "|---|---:|---|---|---|",
    ]
    L += [f"| {r['word']} | {r['in_text']} | {r['espeak']} | {r['engine']} | {r['diff']} |" for r in bad]
    L += ["", "Not counted (eSpeak's error, a US variant, or no reference):", "",
          "| word | eSpeak | engine | why |", "|---|---|---|---|"]  # fmt: skip
    L += [f"| {r['word']} | {r['espeak']} | {r['engine']} | {r['not_engine']} |" for r in other]
    if args.all:
        L += ["", "Notation-only differences:", "", "| word | eSpeak | engine |", "|---|---|---|"]
        L += [f"| {r['word']} | {r['espeak']} | {r['engine']} |" for r in soft]
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "frontend.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    if not args.words:
        (HERE / "report").mkdir(exist_ok=True)
        (HERE / "report" / "frontend.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
