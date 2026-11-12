"""eSpeak NG helpers: reference phonemes for plain text, and eSpeak as a
"system" (a rule-based formant synthesizer saying the same phonemes).

Nothing here needs PyTorch.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arpabet as A  # noqa: E402

VOICE = "en-us"

#: Engine phoneme -> eSpeak phoneme mnemonic (en-us), for ``[[...]]`` input.
ARPA_TO_MNEMONIC: dict[str, str] = {
    "AA": "A:", "AE": "a", "AH": "V", "AO": "O:", "AX": "@", "EH": "E",
    "ER": "3:", "IH": "I", "IY": "i:", "UH": "U", "UW": "u:",
    "AY": "aI", "AW": "aU", "EY": "eI", "OW": "oU", "OY": "OI",
    "P": "p", "B": "b", "T": "t", "D": "d", "K": "k", "G": "g",
    "M": "m", "N": "n", "NG": "N",
    "F": "f", "V": "v", "TH": "T", "DH": "D", "S": "s", "Z": "z",
    "SH": "S", "ZH": "Z", "CH": "tS", "JH": "dZ",
    "L": "l", "R": "r", "W": "w", "Y": "j", "HH": "h",
}  # fmt: skip

#: Vowel + R inside one syllable: eSpeak has its own r-coloured phonemes.
MERGED_MNEMONIC: dict[tuple[str, str], str] = {
    ("AA", "R"): "A@",
    ("AO", "R"): "O@",
    ("EH", "R"): "e@",
    ("UH", "R"): "U@",
}


def have_espeak() -> bool:
    return shutil.which("espeak-ng") is not None


def _run(args: list[str]) -> str:
    res = subprocess.run(["espeak-ng", *args], capture_output=True, text=True, check=True)
    return res.stdout


@lru_cache(maxsize=4096)
def ipa_tokens(text: str) -> tuple[str, ...]:
    """IPA phoneme tokens eSpeak en-us speaks for ``text`` (stress removed).

    Uses ``--sep=_`` so the tokens are eSpeak's own phoneme units -- the
    same units the judge model was trained on.
    """
    out = _run(["-q", "-v", VOICE, "--ipa", "--sep=_", text])
    toks = []
    for t in re.split(r"[\s_]+", out.strip()):
        t = t.replace("ˈ", "").replace("ˌ", "")
        if t:
            toks.append(t)
    return tuple(toks)


def phonemes(text: str) -> list[str]:
    """Reference phonemes for plain text, in the scoring alphabet."""
    return A.fold_tokens(list(ipa_tokens(text)))


def arpabet_group_to_mnemonics(symbols: list[str]) -> str:
    """One bracket group (``["B", "AA1", "T"]``) -> eSpeak phoneme string
    (``b'A:t``).  The stress mark goes before the stressed vowel; if none
    is marked, before the first vowel (as the engine does)."""
    phones = [A.strip_stress(s) for s in symbols]
    stressed = [s.endswith("1") for s in symbols]
    if not any(stressed):
        for k, p in enumerate(phones):
            if p in A.VOCALIC:
                stressed[k] = True
                break
    parts: list[str] = []
    i = 0
    while i < len(phones):
        p = phones[i]
        mark = "'" if stressed[i] else ""
        pair = (p, phones[i + 1]) if i + 1 < len(phones) else None
        nxt2 = phones[i + 2] if i + 2 < len(phones) else None
        # r-coloured vowel only when the R closes the syllable.
        if pair in MERGED_MNEMONIC and nxt2 not in A.VOCALIC:
            parts.append(mark + MERGED_MNEMONIC[pair])
            i += 2
            continue
        m = ARPA_TO_MNEMONIC[p]
        if p == "ER" and not stressed[i]:
            m = "3"
        parts.append(mark + m)
        i += 1
    # "|" keeps neighbouring mnemonics from fusing (t|S is not tS).
    return "|".join(parts)


_BRACKET = re.compile(r"\[([^\]]*)\]")


def engine_text_to_espeak(text: str) -> str:
    """Engine text (plain words and ``[ARPAbet]`` groups) -> eSpeak text
    (plain words and ``[[mnemonics]]`` groups)."""

    def repl(m: re.Match[str]) -> str:
        syms = [s for s in m.group(1).split() if A.strip_stress(s) in A.ENGINE_PHONEMES]
        return f" [[{arpabet_group_to_mnemonics(syms)}]] " if syms else " "

    return re.sub(r"\s+", " ", _BRACKET.sub(repl, text)).strip()


def synth(text: str, out_path: Path | str) -> None:
    """Speak engine-style ``text`` with eSpeak NG into a WAV (22.05 kHz)."""
    _run(["-v", VOICE, "-w", str(out_path), engine_text_to_espeak(text)])


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        e = engine_text_to_espeak(arg)
        print(e)
        print(" ", " ".join(ipa_tokens(e)))
        print(" ", " ".join(phonemes(e)))
