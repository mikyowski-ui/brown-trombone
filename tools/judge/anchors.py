"""Human anchors: real speech the judge must get right.

``uv run python anchors.py fetch`` downloads one small parquet file (9 MB,
73 utterances of LibriSpeech dev-clean: LibriVox readings of public-domain
books, corpus CC BY 4.0) from Hugging Face, picks a handful of short
utterances from different speakers, stores the audio under
``.cache/anchors/`` (not committed) and writes ``material/human.tsv`` with
the transcripts and the phonemes eSpeak NG expects for them.

The gate: these recordings must score under 15 % phoneme error.  If they do
not, the normalizer or the decoder is wrong and no other number from the
judge can be trusted.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import material  # noqa: E402

HERE = Path(__file__).resolve().parent
ANCHOR_DIR = HERE / ".cache" / "anchors"
SOURCE_DIR = ANCHOR_DIR / "source"
DATASET = "hf-internal-testing/librispeech_asr_dummy"
REVISION = "5be91486e11a2d616f4ec5db8d3fd248585ac07a"
FILENAME = "clean/validation-00000-of-00001.parquet"
N_ANCHORS = 16
MIN_SECONDS, MAX_SECONDS = 2.0, 9.0


def fetch(n: int = N_ANCHORS) -> list[material.Item]:
    os.environ.setdefault("HF_HOME", str(HERE / ".cache" / "hf"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    import espeak_util as E

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(DATASET, FILENAME, repo_type="dataset", revision=REVISION, local_dir=SOURCE_DIR)
    table = pq.read_table(path, columns=["id", "text", "speaker_id", "audio"]).to_pylist()

    rows = []
    for r in table:
        raw = r["audio"]["bytes"]
        info = sf.info(io.BytesIO(raw))
        seconds = info.frames / info.samplerate
        if MIN_SECONDS <= seconds <= MAX_SECONDS:
            rows.append((str(r["id"]), int(r["speaker_id"]), seconds, r["text"].strip(), raw))
    rows.sort(key=lambda r: r[0])

    # Round-robin over speakers so no single voice dominates.
    by_speaker: dict[int, list] = {}
    for r in rows:
        by_speaker.setdefault(r[1], []).append(r)
    picked = []
    while len(picked) < n and any(by_speaker.values()):
        for spk in sorted(by_speaker):
            if by_speaker[spk] and len(picked) < n:
                picked.append(by_speaker[spk].pop(0))

    items = []
    for utt_id, spk, seconds, text, raw in picked:
        item_id = f"human_{utt_id}"
        data, rate = sf.read(io.BytesIO(raw), dtype="int16")
        sf.write(str(ANCHOR_DIR / f"{item_id}.flac"), data, rate)
        sentence = text.capitalize()
        items.append(material.Item(item_id, "human", sentence, tuple(E.phonemes(sentence)), sentence))
        print(f"{item_id}  speaker {spk}  {seconds:4.1f} s  {sentence}")
    material.write_tsv(material.MATERIAL_DIR / "human.tsv", items)
    total = sum(r[2] for r in picked)
    print(f"{len(items)} anchors, {len({r[1] for r in picked})} speakers, {total:.0f} s -> material/human.tsv")
    return items


if __name__ == "__main__":
    if sys.argv[1:2] == ["fetch"]:
        fetch(int(sys.argv[2]) if len(sys.argv) > 2 else N_ANCHORS)
    else:
        print(__doc__)
