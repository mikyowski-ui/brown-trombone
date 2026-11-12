# Intelligibility judge and tuner

An offline judge that scores how intelligible synthesized speech is, and a
CMA-ES tuner that searches the physical engine's 30 tuning parameters
(`params.json`) to improve that score. Everything runs locally once the models
are downloaded.

## Setup

```sh
cd tools/judge
uv sync                                   # Python 3.12, CUDA 12.6 PyTorch, transformers, whisper, cma
uv run python anchors.py fetch            # 16 LibriSpeech recordings -> .cache/anchors, material/human.tsv
```

`espeak-ng` must be on `PATH` (reference phonemes for plain text, and the
`espeak` system). The engine is called through the prebuilt `bin/say` (the
`say` example of `crates/synthie-trombone`); set `SAY_BIN` to use another
build. The models download on first use into `.cache/` (Hugging Face
`facebook/wav2vec2-lv-60-espeak-cv-ft`, about 1.3 GB, and Whisper `small.en`,
about 0.5 GB). Once the phoneme model is there, `judge.py` sets
`HF_HUB_OFFLINE=1`, so no network is touched. A GPU is used when one is
available (`JUDGE_DEVICE=cpu` forces CPU). The phoneme model and Whisper are
loaded one after the other, never together.

```sh
uv run pytest                             # normalizer + model-free tests
```

## Scoring

```sh
uv run python judge.py score human espeak classic physical     # the anchors
uv run python judge.py score physical --profile out/profile-b.json --name physical-b
uv run python judge.py score physical --sets sentence_heldout --profile out/profile-b.json --name physical-b-heldout
uv run python judge.py report                                  # rebuild report/judge.md
uv run python judge.py items physical --set vcv                # what was heard, item by item
```

Each scored system writes `out/score-<name>.json` (per set, per item,
confusions) and the comparison table `report/judge.md` is rebuilt from every
`out/score-*.json`. Delete a score file to drop it from the report.

What is measured, per material set:

| metric | how |
|---|---|
| PER | wav2vec2 CTC phoneme recognizer, greedy decoding, eSpeak IPA tokens folded to the engine's ARPAbet by `arpabet.py`, aligned against the expected phonemes; edits / reference phonemes over the whole set. The flap `DX` matches `T` and `D`. |
| CTC log-likelihood | log P(reference phonemes given the audio), nats per reference phoneme, with every model token that folds to a phoneme accepted for it, and r-coloured pairs (`AA R` = `ɑːɹ`) tried merged and split. Higher is better; smoother than PER. |
| confusions | substitutions, deletions (`X→-`) and insertions (`-→Y`) from the alignments, overall and per set. |
| word accuracy | Whisper `small.en`, reference words found in the transcript after Whisper's English text normalizer (also stored as WER). VCVs have no written form, so they have no word score. |
| PER vs intended | engine only: the same alignment against the phonemes the engine says it spoke (first line of `say`'s output), so the engine's own text-to-phoneme mistakes on plain-text sentences are left out. |

Material sets (`material/*.tsv`, built by `material.py build`):

| set | items | content |
|---|---:|---|
| `vcv` | 104 | every consonant in `AA C AA`, `IY C IY`, `UW C UW`; every vowel in `hVd` and `bVt` |
| `pair` | 44 | 22 minimal pairs (B/P, S/SH, R/L, IY/IH ...) |
| `word` | 42 | common words that cover every phoneme |
| `sentence_tune` | 10 | Harvard sentences, lists 1 and 2: tuning |
| `sentence_heldout` | 10 | Harvard sentences: held out, never tuned on |
| `human` | 16 | LibriSpeech dev-clean sentences. The `human` system plays the recordings; every other system synthesizes the transcript, so all systems are compared on the same text (shown as `librispeech` in the report). |
| `vowel` | 96 | probe set (not in the default scoreboard): every vowel alone and in five CVC frames |
| `voicing` | 110 | probe set: P/B T/D K/G S/Z F/V minimal pairs (initial, medial, final) and VCVs |
| `harvard_tune` / `harvard_heldout` | 20 / 20 | vowel tuner: Harvard lists 3 and 4 (tuning) and 5 and 6 (held out); every vowel slot scored |
| `carrier_tune` / `carrier_heldout` | 126 / 126 | vowel tuner: every stressed vowel in four CVC frames inside two carrier phrases, AX in three words; tuning and held-out splits share no frame and no carrier; only the CVC's vowel is scored |

The probe sets have a sixth column, `meta` (`key=value;...`): the scored
slot, the intended phoneme, the frame, the minimal-pair twin and so on.

`vcv`, `pair` and `word` are given to the engine as bracketed ARPAbet
(`[B AE1 T]`), so the expected phonemes are exactly what it was told to say.
Sentences are plain text, and the expected phonemes come from eSpeak NG
(`espeak-ng -v en-us --ipa`, folded).

### Probes: vowels, voicing, consonant confusions

Targeted measurements for the vowel, voicing and consonant-place stages
(`probes.py`, run through `judge.py`):

```sh
uv run python judge.py vowels physical --profile ../../profiles/tuned-r3.json   # -> physical-tuned-r3
uv run python judge.py voicing espeak
uv run python judge.py confusions classic --name classic-x
uv run python judge.py vowels physical --set human          # the same slots inside the LibriSpeech sentences
uv run python judge.py vowels human                         # human ceiling, in-sentence
uv run python judge.py vowels human --cut                   # human words cut out of the recordings
uv run python judge.py probe-report r4-physical r4-classic --refs r4-espeak,human-cut \
    --scores r4-physical --out report/probes.md             # one page, side by side
uv run python frontend_check.py                             # engine text front end vs eSpeak
```

Each takes a system (`espeak`, `classic`, `physical`, and `human` for the
references), `--profile`, `--name` (default: the system, or
`system-<profile stem>`; `-insent` / `-cut` is appended for `--set human` /
`--cut`), honours `SAY_BIN`, uses the render cache (`--fresh` renders into a
scratch folder instead, for timing), writes `out/<probe>-<name>.json` (every
item and slot) and `report/<probe>-<name>.md`, and prints one summary line.

| probe | material | metrics |
|---|---|---|
| `vowels` | `vowel`: 96 items, the 16 vowels (IY IH EH AE AH AA AO UH UW ER AX EY AY OW AW OY) alone and in hVd, bVd, bVt, sVt, dVd (AX: alone and unstressed in five words); 50 real words, the rest labelled nonsense or isolated | **vowel error rate**: share of items whose vowel slot, after aligning the transcription to the reference, is not the intended vowel (AH = AX: the model writes the same vowel ʌ stressed and ə unstressed; a second rate also merges AA/AO, the cot-caught merger); **PER** on the set; a **confusion table** (intended x heard in the slot, with the model tokens); the **margin** (below) per vowel and frame |
| `voicing` | `voicing`: 110 items, P/B T/D K/G S/Z F/V as real minimal pairs in initial, medial and final position (medial T/D as "a tie"/"a die": American English flaps "latter"/"ladder") plus VCV syllables, both members of every pair | **forced choice**: CTC log P(intended) - log P(minimal-pair twin); accuracy = share where the intended member wins, and the mean margin; **free recognition**: the consonant heard in the slot has the intended voicing (a flap counts half for T/D; a sonorant, vowel or deletion is wrong). Broken down by pair, position, pair x position, intended consonant and intended voicing. Every pair and position has as many voiced as voiceless items, so voicing everything scores 50 %. The headline is P/B T/D K/G S/Z; F/V is reported separately. |
| `confusions` | `vcv` (the consonant of every `V C V`) | consonant accuracy; full consonant x consonant table; tables by place (bilabial, labiodental, dental, alveolar, postalveolar, palatal, velar, glottal), manner (stop, nasal, fricative, affricate, approximant) and voicing; heard-side columns for vowel, other and deletion |

**The margin** (`vowels`, and the forced choice of `voicing`) is computed
from the phoneme model's frame posteriors: the CTC log-probability of the
reference string minus that of the best competing string, where a competitor
is the reference with the slot replaced (by any other vowel; by the voicing
twin), every model token that folds to a phoneme pooled for it, as for the
CTC log-likelihood. In nats; positive means the intended phoneme wins; it
moves smoothly before the transcription flips, so a tuner can use it. Capped
at ±30 nats; 0 when the audio is too short to carry either string.

```python
import probes
pj = probes.ProbeJudge()                                  # loads the phoneme model once
r = pj.vowel("x.wav", ["B", "IY1", "D"])                  # slot: the first vowel by default
r["margin"], r["rival"], r["heard"], r["correct"]
rs = pj.vowels([a1, a2], [ref1, ref2], [slot1, slot2])    # wav paths or 16 kHz float arrays, batched
pj.voicing("y.wav", ["P", "AE", "T"], 0)["margin"]        # vs B AE T
```

**Human and in-sentence references.** `--set human` scores the same slots in
the 16 LibriSpeech sentences (every vowel, with stressed/unstressed from
eSpeak; every P B T D K G S Z F V against its twin; every consonant), for
the recordings (`human`) or any synthesizer saying the transcripts. `--cut`
(human only) force-aligns each recording to its reference (Viterbi CTC
path) and cuts out every word (eSpeak's word boundaries, halfway between the
spikes of neighbouring words, 10 ms fades, 0.15 s of silence each side);
each word is scored as an isolated item, the human counterpart of the
synthetic material. For the engine reading sentences, slots where its own
phonemes (the first line of `say`) differ from the reference are flagged
`frontend_ok: false`, and every in-sentence summary is repeated without them
("front end right"), so the engine's text-to-phoneme mistakes can be told
from its sound. Cutting three-phoneme excerpts was tried first and was
unusable (the excerpts spill into their neighbours and the model answers in
other languages).

`frontend_check.py` compares the engine's own text-to-phoneme output (the
first line of `say`) with eSpeak's for every word of the material, ignores
notation-only differences (AX/AH/IH, IY/IH, AA/AO, flap) and a short list of
eSpeak errors and US variants, and writes `report/frontend.md`.

#### What the probes can and cannot tell

- The judge is harsh on isolated items, also with real speech: stressed
  vowels in words cut out of the LibriSpeech recordings get about 35 % vowel
  error (about 4.5 % for the same vowels heard in their sentences), and eSpeak
  saying the `vowel` set gets 57 %. On isolated items the model drifts to
  non-English tokens (`a`, `aː`, `ɑ` without length, even Mandarin finals),
  which `arpabet.py` folds to AA, so "heard as AA" on isolated items is partly
  the judge: `report/vowels-*.md` lists the model tokens, and
  `heard_aa_tokens` in the JSON counts them.
- So read the isolated-item numbers against eSpeak's and the cut-word human
  scores, not against zero; the in-sentence numbers (`--set human`) are
  closer to what a listener hears, but mix in the engine's text front end.
  `report/round4-baseline.md` gives the ceilings.

### The gate

The human anchors must score **under 15 % PER**. If they do not, the
normalizer or the decoder is wrong and no other number from the judge can be
trusted: fix the judge, do not move the threshold. `report/judge.md` prints
PASS or FAIL at the top.

## Vowels: the lab and the tuner

```sh
uv run python vowel_lab.py eval --split tune                     # vowel error and margins, per vowel
uv run python vowel_lab.py eval --split heldout --profile p.json --name p
uv run python vowel_tune.py --base base.json --out out/vowels-tuned.json   # coordinate search on the tuning split
```

`vowel_lab.py` scores every vowel slot of a split under a profile (renders
in a scratch folder, GPU lock held only while judging): **tune** =
`sentence_tune`, `harvard_tune`, `carrier_tune`; **heldout** =
`sentence_heldout`, `harvard_heldout`, `carrier_heldout`, `human` (the
held-out split is never tuned on). A profile may carry a `"vowels"` object
of F1/F2/F3 targets (adult man, Hz) keyed `IH`, `AY<`, `AY>`, `IH>`; `bin/say`
solves their shapes through the babble map at start-up, so a candidate
needs no rebuild. `vowel_tune.py` moves F1 and F2 of every vowel shape in
steps, within 12 % (F1) and 10 % (F2) of the published value, keeping the
vowels' order in F1 and F2, with a mild pull back to the published value;
the objective is the mean clipped margin over the tuning split. Bake a
result into `crates/synthie-trombone/src/vowel_targets.rs` and regenerate
the shapes (`cargo run -p synthie-trombone --example vowel_size --release`).

## Tuning

```sh
uv run python tune.py --self-test                    # 7 evaluations on 12 items, then a resume check
uv run python tune.py --pass a --sets vcv,pair --budget 300 --seed 1 --out out/profile-a.json
uv run python tune.py --pass b --lock out/profile-a.json --sets word,sentence_tune \
    --budget 300 --seed 1 --out out/profile-b.json
uv run python judge.py score physical --sets sentence_heldout --profile out/profile-b.json --name physical-b-heldout
```

- Searches the parameters of one pass (`a`: 14 source/frication parameters,
  `b`: 16 timing parameters), minus those fixed by `--lock`. Each parameter
  is mapped to 0..1 over its `lo..hi` range (hard bounds) and the search starts
  at the defaults.
- Objective (lower is better), averaged over the sets:
  `mean PER + 0.1 * mean(-log-likelihood) + 0.1 * mean(((x - default)/(hi - lo))^2)`
  (`--ll-weight`, `--prior-weight`).
- Evaluation 0 is the starting point (defaults plus the lock), then CMA-ES
  generations (`--popsize`, default `4 + 3 ln n`: 11 for pass a, 12 for pass
  b; `--sigma` 0.25). The last generation may overrun `--budget` by less than
  one population.
- All items of a generation render in parallel (`--jobs`, default all cores)
  into `out/tmp/` (deleted afterwards); the phoneme model stays loaded on the GPU
  and judges in batches (`--batch 8`).
- Every evaluation is appended to `out/tune-<out stem>.jsonl` (unit-box
  point, parameter values, objective, per-set PER and log-likelihood, timings).
  `--resume` replays it: same seed, same candidates, and logged ones are not
  evaluated again. Without `--resume` an existing log is kept as `.bak`.
- The best profile so far is rewritten to `--out` after every generation, as
  `{"tuning": {...}}` with the tuned and locked parameters, the only shape
  `bin/say --profile` accepts.
- Whisper is not used in the loop. At the end the best profile is scored by
  `judge.py` (phonemes + Whisper) on the tuning sets, as
  `out/score-physical-<out stem>.json` (`--no-final-whisper` skips this).
- `--subset N` uses N items per set for quick experiments.

Measured on the RTX 3070 / 16 threads machine, with the default population
size: pass a on `vcv,pair` (148 items) took **2.5 s per evaluation** (1.8 s rendering, 0.65 s judging);
pass b on `word,sentence_tune` (52 items) took **1.8 s per evaluation**. A
300-evaluation pass is about 9 to 13 minutes, plus about 15 s to start up and 30 s for the final
Whisper check.

## Files

| file | what |
|---|---|
| `arpabet.py` | ARPAbet <-> eSpeak IPA normalizer: forward map, fold of all 392 model tokens, canonical rules, alignment. `python arpabet.py --dump` rewrites `data/fold_table.tsv`. |
| `espeak_util.py` | eSpeak NG: reference phonemes for text; ARPAbet -> eSpeak mnemonics for the `espeak` system. |
| `material.py` | builds and loads `material/*.tsv`. |
| `anchors.py` | fetches the LibriSpeech anchors (`fetch`). |
| `render.py` | renders a material set through a system (`classic`, `physical`, `espeak`, `human`); renders are cached under `out/render/` by binary, voice, profile and text. |
| `judge.py` | the judge: `score`, `report`, `items`, and the probes `vowels`, `voicing`, `confusions`, `probe-report`. |
| `probes.py` | the probes: slot alignment, CTC forced-choice margins, `ProbeJudge` (the tuner API), cut tokens from the anchors, summaries and Markdown. |
| `frontend_check.py` | the engine's text-to-phoneme output vs eSpeak's, per word. |
| `tune.py` | the CMA-ES tuner. |
| `vowel_lab.py` | vowel error and margins per vowel under a profile, on the tuning or held-out split. |
| `vowel_tune.py` | coordinate search over the vowel targets on the tuning split. |
| `params.json` | the tuning parameters: `lo`, `hi`, `default`, `pass`; passes a and b for `tune.py`, pass c (vowel lengths and glides) for reference. |
| `data/w2v_vocab.json` | the phoneme model's tokenizer vocabulary (bundled; checked against the model at load). |
| `data/fold_table.tsv` | every model token and the phonemes it folds to, for review (a test checks it is current). |
| `tests/` | pytest: normalizer (every symbol maps, round trips, diphthongs and affricates as single model tokens, eSpeak agreement) and model-free parts of the judge and tuner. |
| `report/judge.md` | generated comparison of every scored system. |
| `out/` | scores, tuner logs, profiles, render cache (not committed). |
| `.cache/` | models and anchor audio (not committed). |
| `bin/say` | prebuilt renderer (not committed). |

## Caveats

- The phoneme model is a multilingual CommonVoice model. It transcribes the
  recorded human speech well (about 6 % PER), but it is harsh on synthetic
  speech in isolation: eSpeak's own voice scores about 50 % PER on the
  `vcv`/`pair`/`word` sets while Whisper understands most of its words, and
  around 15 to 25 % on sentences. Treat PER on isolated items as a relative
  measure (better or worse than before), not as "the fraction a listener
  would mishear". Decoding variants (pooling token variants, restricting to
  English tokens, padding with silence) were tried and changed the scores by
  only a few points, so greedy decoding was kept.
- The 16 human anchors come from the small `hf-internal-testing/librispeech_asr_dummy`
  subset, which is a single LibriSpeech speaker (1272, male). The gate
  checks the normalizer and the decoder, not robustness across speakers.
- Rendering is deterministic (same profile, same audio), so the objective has
  no noise from the engine; with small sets it can still overfit them, which is
  why the held-out sentences exist.

## Matching a voice

`match_voice.py` listens to a recording of someone speaking and writes a
voice profile (`bin/say --profile`) that makes the physical engine sound
like that speaker: same pitch and pitch range, same body size (formant
scale), same voice quality, same pace.

```sh
uv run python match_voice.py speaker.wav                        # -> speaker.profile.json
uv run python match_voice.py speaker.flac --text "what they said" --transcript --out voice.json
uv run python match_voice.py speaker.mp3 --analyse              # measurements only
uv run python match_voice.py --calibrate                        # rebuild data/voice_calibration.json (about 2 min)
uv run pytest tests/test_match_voice.py -s                      # includes the self-consistency gate (about 1 min)
```

Any format `soundfile` reads (wav, flac, ogg), else ffmpeg (mp3, m4a, ...);
mono or stereo, any rate, a few seconds to a minute of speech. `--text` is
what the engine says while it is compared with the recording (default: four
Harvard sentences); give the transcript when you have it, and `--transcript`
when it is exactly what the recording says (speaking rate is then matched by
speech time instead of by syllable rate). A match takes 5 to 30 s (2 to 8
renders of the comparison text). It prints the recording's measurements,
the same measurements of the engine speaking with the new profile, and the
differences; `--report r.json` keeps everything, round by round.

The profile sets `pitch`, `tract_length`, `tenseness`, `expressiveness`,
`rate` and `tuning.posterior_gap` / `tuning.lung_pressure` (plus
`"engine": "Physical"`), all inside the engine's ranges.

### What is measured

| quantity | how |
|---|---|
| median pitch, range | autocorrelation pitch (Boersma: normalised autocorrelation, 4 candidates per 10 ms frame, Viterbi path with octave-jump cost, frames > 0.8 octave from the median dropped); median and 10th..90th percentile in semitones |
| formant scale | LPC (order 20/22/26 by the speaker's formant spacing) on pitch-smoothed spectra of the steady voiced frames; the long-term distribution of formant candidates over log frequency is lined up with the default voice's saying the same text. The scale factor that lines them up is the formant scale (1/tract length for the engine, within 2 %). Where the alignment is ambiguous (several scales about 25 % apart fit almost equally well, as for human speech), the ratio of formant spacings `dF` (uniform tube fitted to median F1..F4) chooses |
| H1*-H2* | levels of the first two harmonics (48 ms frames), corrected for the first two formants (Iseli and Alwan), median over voiced frames; the uncorrected H1-H2 is printed too |
| tilt | alpha ratio: energy 1..5 kHz over 0..1 kHz of the voiced frames |
| breathiness | harmonics-to-noise ratio from the autocorrelation peak (median), and cepstral peak prominence |
| pace | syllable nuclei (envelope peaks in voiced speech) per second of speech; speech time; pauses of 0.25 s or more are left out of both |

### How it is calibrated

`data/voice_calibration.json` holds the measurements above taken on the
engine itself, saying the default comparison text at 129 settings (a Latin
hypercube over pitch 75..340 Hz, tract length 0.64..1.4, tenseness
0.25..0.92, expressiveness 0..2, rate 0.65..1.6, posterior gap 0.05..0.5 and
lung pressure 5..13, plus the default voice, whose analysis is the reference
for the formant scale). At start-up a quadratic model of each measurement is
fitted to it (robustly: samples the analysis got badly wrong are dropped).
Matching inverts the model for a first guess, then renders, re-analyses and
corrects: pitch, tract length and rate by ratio, expressiveness by the range
error over its slope, and tenseness, posterior gap and lung pressure together
by a Broyden step on H1*-H2*, HNR and tilt. It stops when everything is
close, or after 8 rounds, and writes the best round. Rebuild the calibration
when `bin/say` changes (the tool notes it when the binary differs).

### The gate

The engine speaking with the produced profile must re-analyse to within
5 % of the recording on median pitch, 3 % on formant scale and 3 dB on
H1*-H2*. `tests/test_match_voice.py` checks this on eight hidden engine voices
(pitch 90..300 Hz, tract length 0.7..1.25, tenseness 0.35..0.85) and prints
the errors and the recovered settings.

### Limits

- The engine cannot make every voice. With an adult tract (length near 1)
  its H1*-H2* does not go below about 5 dB (eSpeak's male voices measure
  about -1 dB, and fail the H1*-H2* gate by about 7 dB); its pitch range tops
  out at 9 to 10 semitones at expressiveness 2 (the LibriSpeech reader
  already needs 1.4 to 2); its HNR is lower than very clean voices'.
- At high pitch (above about 220 Hz) different tenseness / lung pressure /
  posterior gap combinations measure almost alike, so the voice quality is
  matched but the settings are not unique (the hidden tenseness is not
  recovered there).
- The formant scale of human speech is less certain than the engine's: on 16
  utterances of one LibriSpeech speaker the guide (`dF`) varies by about 5 %
  and the chosen scale by more (tract length 0.78..0.97 across five of
  them). Long tracts at high pitch (above 250 Hz with tract length above 1.2)
  and near-whispers (tenseness below about 0.3 with a wide posterior gap) are
  measured unreliably, as are recordings with little voiced speech.
- Not measured, so left at the defaults: jitter, vibrato, nasality, the
  persona dials, and every timing/articulation tuning parameter except
  `rate`. Accent, articulation and intonation patterns are not copied; only
  their overall pitch range is.
- The scale and the rate depend on what is said; give `--text` (and
  `--transcript`) when you know it.
