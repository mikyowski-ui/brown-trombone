"""Tune the physical engine's vowel targets against the phoneme judge.

    uv run python vowel_tune.py --base base.json --out out/vowels-tuned.json
    uv run python vowel_tune.py --base base.json --out out/vowels-tuned.json --resume

Coordinate search over F1 and F2 of every vowel shape (vowels, diphthong
starts and ends; F3 follows F2 where it is tied to it, else stays), on the
**tuning split only** (``vowel_lab.TUNE_SETS``).  Each round tries, for
every shape at once, F1 and F2 one step up and one step down (each
candidate alone, on top of the current point), keeps for each shape the
best move that improves the objective, and then checks the combined
point; if the combination is worse than the best single move, only that
move is kept.  The step halves when a round keeps nothing.

Objective (higher is better)::

    J = mean over slots of clip(margin, -CLIP, CLIP)
        - PRIOR * sum over formants of (ln(f / published) / bound)^2

with ``bound`` = 12 % for F1 and 10 % for F2 (hard limits too), and the
vowels' natural order in F1 and F2 kept as hard constraints (``ORDER``).

Every candidate renders through ``bin/say`` with the profile's
``"vowels"`` object (shapes solved through the babble map at start-up);
the judge holds the GPU lock only while it scores a chunk.  Every
evaluated point is appended to ``out/vtune-<out stem>.jsonl``; ``--resume``
replays it.  ``--out`` gets the best profile so far after every round:
the base profile with the tuned ``"vowels"``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import vowel_lab as L  # noqa: E402

#: Shapes tuned (the glides W and Y are consonants and are left alone).
KEYS = ["IY", "IH", "EH", "AE", "AH", "AA", "AO", "UH", "UW", "ER", "AX",
        "EY<", "EY>", "AY<", "AY>", "OW<", "OW>", "AW<", "AW>", "OY<", "OY>"]  # fmt: skip
#: Largest move from the published value, as a ratio: F1, F2.
BOUND = (0.12, 0.10)
#: Pairs ``(lower, higher)`` whose order in F1, and in F2, must stay.
ORDER = {
    0: [("IY", "IH"), ("IH", "EH"), ("EH", "AE"), ("UW", "UH"), ("UH", "AO"), ("AH", "AA"), ("AO", "AA")],
    1: [("IH", "IY"), ("EH", "IH"), ("AH", "EH"), ("AO", "AH"), ("UW", "UH"), ("AO", "AA"), ("AH", "AE")],
}
CLIP = 6.0
PRIOR = 0.05
CHUNK = 12


def objective(rows: list[dict], point: dict, published: dict) -> tuple[float, float]:
    """(J, mean clipped margin)."""
    m = sum(max(-CLIP, min(CLIP, r["margin"])) for r in rows) / max(1, len(rows))
    pen = 0.0
    for k in KEYS:
        for i in (0, 1):
            pen += (math.log(point[k][i] / published[k][i]) / BOUND[i]) ** 2
    return m - PRIOR * pen, m


def allowed(point: dict, published: dict) -> bool:
    for k in KEYS:
        for i in (0, 1):
            if abs(math.log(point[k][i] / published[k][i])) > math.log(1 + BOUND[i]) + 1e-9:
                return False
    for i, pairs in ORDER.items():
        if any(point[a][i] >= point[b][i] for a, b in pairs):
            return False
    return True


def moved(point: dict, key: str, i: int, ratio: float) -> dict:
    p = {k: list(v) for k, v in point.items()}
    p[key][i] *= ratio
    if i == 1 and point[key][2] < point[key][1] + 500:  # keep F3 clear of F2
        p[key][2] = max(p[key][2], p[key][1] + 300)
    return p


def profile(base: dict, point: dict) -> dict:
    p = json.loads(json.dumps(base))
    p.setdefault("vowels", {}).update({k: [round(x, 1) for x in v] for k, v in point.items()})
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True, help="profile the vowels are tuned on top of")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--step", type=float, default=0.05, help="first step, as a ratio")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    base = json.loads(args.base.read_text())
    published = {k: v for k, v in L.targets().items() if k in KEYS}
    log = L.OUT_DIR / f"vtune-{args.out.stem}.jsonl"
    seen: dict[str, dict] = {}
    if args.resume and log.exists():
        for line in log.read_text().splitlines():
            e = json.loads(line)
            seen[json.dumps(e["point"], sort_keys=True)] = e
    elif log.exists():
        log.rename(log.with_suffix(".jsonl.bak"))
    lab = L.Lab()

    def evaluate(points: list[dict]) -> list[dict]:
        todo = [p for p in points if json.dumps(p, sort_keys=True) not in seen]
        for c in range(0, len(todo), CHUNK):
            part = todo[c : c + CHUNK]
            t0 = time.perf_counter()
            res = lab.slots(L.TUNE_SETS, [profile(base, p) for p in part])
            for p, rows in zip(part, res):
                j, m = objective(rows, p, published)
                s = L.summarize(rows)
                e = {"point": p, "J": j, "margin": m, "err": s["all"]["err"],
                     "vowels": {v: [st["err"], st["margin"]] for v, st in s["vowels"].items()}}  # fmt: skip
                seen[json.dumps(p, sort_keys=True)] = e
                with log.open("a") as f:
                    f.write(json.dumps(e) + "\n")
            print(f"  evaluated {len(part)} in {time.perf_counter() - t0:.0f} s", flush=True)
        return [seen[json.dumps(p, sort_keys=True)] for p in points]

    point = {k: list(v) for k, v in published.items()}
    if "vowels" in base:  # start where the base profile puts them
        for k, v in base["vowels"].items():
            if k in point:
                point[k] = list(v)
    best = evaluate([point])[0]
    print(f"start: J {best['J']:+.3f}, margin {best['margin']:+.3f}, error {100 * best['err']:.1f} %", flush=True)
    step = args.step
    for rnd in range(args.rounds):
        cands = []
        for k in KEYS:
            for i in (0, 1):
                for r in (1 + step, 1 / (1 + step)):
                    q = moved(point, k, i, r)
                    if allowed(q, published):
                        cands.append((k, i, r, q))
        res = evaluate([q for *_, q in cands])
        gains = {}
        for (k, i, r, q), e in zip(cands, res):
            d = e["J"] - best["J"]
            if d > 0 and (k not in gains or d > gains[k][0]):
                gains[k] = (d, i, r, e)
        if not gains:
            step /= 2
            print(f"round {rnd}: nothing kept, step -> {step:.3f}", flush=True)
            if step < 0.012:
                break
            continue
        combo = {k: list(v) for k, v in point.items()}
        for k, (d, i, r, e) in gains.items():
            combo = moved(combo, k, i, r)
        top = max(gains.values(), key=lambda g: g[0])
        if allowed(combo, published):
            ce = evaluate([combo])[0]
        else:
            ce = None
        if ce is not None and ce["J"] >= top[3]["J"]:
            point, best = combo, ce
            kept = ", ".join(f"{k} F{i + 1} x{r:.3f}" for k, (d, i, r, _) in gains.items())
        else:
            point, best = top[3]["point"], top[3]
            kept = "only the best single move"
        print(f"round {rnd}: J {best['J']:+.3f}, margin {best['margin']:+.3f}, error {100 * best['err']:.1f} % ({kept})", flush=True)
        args.out.write_text(json.dumps(profile(base, point), indent=1) + "\n")
    lab.close()
    args.out.write_text(json.dumps(profile(base, point), indent=1) + "\n")
    print("published -> tuned (F1, F2, F3):")
    for k in KEYS:
        print(f"  {k:4} {published[k]} -> {[round(x) for x in point[k]]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
