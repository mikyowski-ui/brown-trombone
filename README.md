# synthie-trombone

This is [Pink Trombone](https://dood.al/pinktrombone/), Neil Thapen's
physical model of the human voice, rebuilt in Rust: a glottal source pushes
air through a 44-cell waveguide of the vocal tract, you squeeze the tract
somewhere, and out comes something between a vowel and a very confused
trombone. You can play it in your terminal, have it read your text aloud, or
give the characters in your game a voice.

**Play it in your browser: https://sunsided.github.io/synthie-trombone/**

![Singing an "ah" with the critter voice, a finger on the tongue tip](docs/singing.png)

![The bright voice mid-sentence, about to close the lips for the "m" in "am"](docs/speaking.png)

## Trying it out

The browser version runs the same Rust engine through WebAssembly in an
AudioWorklet; `task web:dev` starts it locally with hot reload (it needs
`wasm-pack` and Node with corepack). Everything below is about the
terminal app and the library.

You'll need a Rust toolchain (1.88 or newer for the library) and, on Linux,
the ALSA headers for audio output (`libasound2-dev` on Debian and Ubuntu).
The project uses [go-task](https://taskfile.dev/); `task` on its own lists
everything it can do.

```sh
task run                                   # start the terminal app
task install                               # install it as ~/.cargo/bin/synthie-trombone
task say -- --voice bright "Hello there!"  # speak a line into say.wav
task voices -- "Welcome, traveller!"       # the same line in every voice, in voices/*.wav
```

Without `task`, run `cargo run --release -p synthie-trombone-cli` for the app
and `cargo run --release -p synthie-trombone --example say -- "…"` for a WAV
file. Stick to release builds: the tract model takes 96,000 steps per second
of audio, and a debug build can't keep up with the sound card.

## Playing the terminal app

The app starts in play mode, with the piano on your home row. Hold a key to
sing, change the vowel with the number keys, and work everything else with
the mouse: the tract, the tongue, the velum, and every meter in the panels.

| Key | Action |
|---|---|
| `a w s e d f t g y h u j k o l p ;` | sing, piano layout; `z` / `x` change the octave |
| `Space` or `Esc` | stop singing (and speaking) |
| `1` … `0` | vowels IY IH EH AE AA AH AO UH UW ER |
| arrow keys | move the tongue body |
| `v` / `V` | next / previous voice |
| `'`, `PgUp` / `PgDn`, `Del` | pick a persona dial, turn it, back to neutral |
| `n`, `b`, `i` | nasal (velum), lip pop, drone |
| `[` `]`, `-` `=` | tenseness, vibrato |
| `m`, `r`, `c` | pitch wobble, reverb amount, clear constrictions |
| `\` | switch between the classic and the physical engine |
| `/` | Formant Lock: steer the vowel by F1 and F2 on a vowel chart (arrow keys then nudge F2 / F1) |
| `Tab` | type a line; `Enter` speaks it at the current note's pitch |
| `,` `.` | speech rate |
| `q` | quit |

| Mouse | Action |
|---|---|
| left-drag in the mouth or throat | a finger on the wall: the row of the midline shuts the tract (a stop), the row beside it leaves a gap that hisses (a fricative), rows further out only shape it |
| left-drag the tongue handle `◆`, or Shift-drag anywhere | move the tongue body (right-drag also works where the terminal passes it on) |
| click a vowel mark (`i e a o u`) | that vowel |
| click in the nose; drag in the gap under it | open or shut the velum; set how far it hangs open |
| click or drag a meter | tenseness, tongue, velum, vibrato, reverb, loudness (lung pressure, physical engine), and the four persona dials |
| click `[-]` `[+]`, `wobble`, `drone`, `engine` | octave and the toggles |
| click `[/ vowels]` under the scope; left-drag on the vowel chart | Formant Lock on or off; steer the vowel by its formants, or click a vowel mark |

**Formant Lock** (`/`, or `[/ vowels]` under the scope) lets you find
vowels by how they sound rather than by where the tongue goes. The scope
gives way to a vowel chart, drawn the way phoneticians draw it: F2 across,
high on the left, and F1 down, low at the top. The shaded region is what
this voice's tract can reach (it moves with the persona's size and the
voice's tract length), the IPA letters mark where this voice makes the
English vowels, and `◆` is where the voice is now, with its F1 and F2 in the
panel's title. Press or drag inside the chart and the physical engine's
babble map ([`inverse`](crates/synthie-trombone/src/inverse.rs)) finds the
tongue, lips and narrowing that give those formants; the tract view above
shows them move. A point out of reach is held at the nearest one in reach,
clicking a letter takes that vowel, and the arrow keys nudge F2 (left
raises it, as on the chart) and F1 (up lowers it) by 2 %. Formant Lock
belongs to the physical engine, so turning it on switches to it, and
switching to the classic engine turns it off; turning it off brings the
scope back and leaves the tract as it is.

Notes stop when you let go of the key in terminals that report key releases
(kitty, WezTerm, foot, Ghostty). In other terminals a note holds until you
press `Space`.

## Voices

You pick a voice with `v` in the app, `--voice` on the command line, or a
`Preset` in code:

| Preset | Character |
|---|---|
| `deep` | 105 Hz, the model's native tract |
| `bright` | 210 Hz, tract 0.86, livelier intonation |
| `child` | 290 Hz, tract 0.72 |
| `elder` | breathy and slow, with a tremor and a shaky pitch |
| `giant` | 62 Hz, tract 1.4, slow |
| `critter` | 360 Hz, tract 0.6, fast, nasal and very excitable |
| `robot` | monotone, no jitter, pressed |
| `whisper` | breath only, no voicing |

Pitch and vocal tract length vary independently from person to person:
raising only the pitch sounds like the same speaker reaching higher, while a
shorter tract shifts every resonance and sounds like a smaller speaker. The
tract values above are relative to the model's native tract of about 17 cm.
The presets combine both with tenseness, jitter, vibrato, nasality, speaking
rate and how much the intonation moves, and all of these are public fields
of `VoiceSettings`, so your own voice starts from a preset and changes what
you like. With the `serde` feature, the settings can live in your game's
data files.

### Personas and profiles

Describing a voice by its parameters is slow. A `Persona` describes it the
way you would describe a person, with four dials that each move several
parameters together, as they move together in people:

| Dial | From … to | What moves |
|---|---|---|
| `size` | 0.6 (small body) … 1.4 (large) | tract length, and pitch faster than length |
| `effort` | 0 (murmur) … 1 (calling out) | lung pressure, firmer folds, higher and livelier pitch, quicker articulators |
| `breathiness` | 0 (pressed) … 1 (breathy) | how much air escapes past the folds |
| `clarity` | 0 (casual) … 1 (careful) | earlier and faster articulators, longer sounds, firmer closures |

```sh
cargo run --release -p synthie-trombone-cli -- --engine physical --voice bright --size 0.8 --effort 0.7
cargo run --release -p synthie-trombone --example say -- --engine physical --breathiness 0.7 "Hello there."
```

```rust
use synthie_trombone::{Persona, Preset};

let voice = Preset::Bright.settings().with_persona(&Persona {
    size: 0.8,
    effort: 0.7,
    ..Persona::NEUTRAL
});
```

Underneath sit the thirty numbers of `Tuning`: lung pressure and how the
folds start, the efficiency and colour of hiss, the width of consonant
gaps, how long each kind of sound lasts and how fast the articulators move.
A voice with its persona and tuning can be kept as a JSON profile (feature
`profile`); every key is optional and a misspelt one is an error:

```json
{
  "persona": { "size": 0.9, "clarity": 0.7 },
  "pitch": 120.0,
  "tuning": { "lead": 0.035, "fricative_length": 0.11 }
}
```

```sh
cargo run --release -p synthie-trombone --example say -- --engine physical --profile my-voice.json "Hello there."
cargo run --release -p synthie-trombone --example say -- --dump-profile     # the voice in use, as a profile
cargo run --release -p synthie-trombone-cli -- --engine physical --profile my-voice.json
```

## The physical engine

Next to Pink Trombone's own model there is a second one, `Engine::Physical`,
that uses the same tract shape and controls but works in real units and
lets the sound come out of the air flow instead of a script:

| | Classic | Physical |
|---|---|---|
| Voice source | a fixed pulse shape | vocal folds as masses on springs, driven by lung pressure through the glottis; pitch from their tension |
| Losses | the same at every frequency | lip radiation, yielding walls and viscous losses, each with its own frequency dependence |
| Side cavities | one lossless nose | two piriform sinuses, two nasal passages with a sinus each |
| Hiss | noise wherever a finger narrows the tract | turbulence where a jet is fast enough, loudest where it strikes the teeth |
| Stop bursts | a scripted pop | pressure builds behind the closure and is released |
| Articulators | constant speed | critically damped, as tissue with inertia |

```sh
cargo run --release -p synthie-trombone-cli -- --engine physical   # `\` switches engines while playing
cargo run --release -p synthie-trombone --example say -- --engine physical "Hello there!"
cargo run --release -p synthie-trombone --example ab                # both engines side by side, in report/
cargo run --release -p synthie-trombone --example probe             # what each engine measures
```

```rust
use synthie_trombone::{Engine, Preset, Talker};

let voice = Preset::Deep.settings().with_engine(Engine::Physical);
let line = Talker::new(44_100.0, voice).bake("Stay a while, and listen.");
```

The physical voice also listens to itself. Each block it reads what its
own air is doing and corrects two things a person corrects by ear: a gap
that is meant to hiss but is shut, choked, or too fine for its stream to
turn turbulent is widened until it does (and the place remembers, so the
next fricative there starts right); and a vowel that starts to hiss, as a
close /u/ does in a loud voice, is opened a little around the place that
hisses and let back when it is quiet. Neither moves anything by more than
0.15 in diameter, and the default voice is left as it was. That is what
lets a persona make the body small or the breath strong without breaking
the consonants, and what lets a finger in the app find a fricative without
landing on its exact width. `examples/hiss.rs` prints the levels over body
size and lung pressure; `Trombone::set_regulation(false)` switches the
correction off to compare.

It is held to published measurements of real voices by tests
(`crates/synthie-trombone/tests/physical_gates.rs` and
`src/phys/tests.rs`); the classic engine is pinned bit for bit by
`tests/classic_golden.rs`, so it is the unchanged baseline to compare
against. A few of the figures:

- Formant bandwidths of 30 to 110 Hz, rising with frequency (classic: 70 to
  1900 Hz), and vowels fitted to Peterson & Barney's F1, F2 and F3.
- The walls resonate against the enclosed air near 190 Hz and give about
  2.7 cm³ per cmH2O, as measured in vivo (Hanna, Smith & Wolfe 2016).
- The piriform sinuses cut a notch near 4.4 kHz (Dang & Honda 1997).
- The folds sing within a cent of the requested note over every preset's
  range, stop below about 3.4 cmH2O of lung pressure, and keep voicing for
  some 40 ms into a /b/ before the mouth fills.
- /s/ and /sh/ come out 15 to 20 dB louder than /f/ and /th/.

It costs more: about 6 times faster than real time on one core at the
native tract length, against 170 for the classic engine. Notes outside the
folds' range (above about 400 Hz for the native tract) are sung with a
prescribed glottal pulse. Body-cover folds under muscle control (Titze &
Story 2002) are implemented and tested but not yet what sets the pitch.

The models and numbers come from: Steinecke & Herzel (1995) for the
two-mass folds; Story & Titze (1995) and Titze & Story (2002) for the
body-cover folds and muscle rules, with tissue constants as in Brad
Story's LeTalker 1.22; Titze (1984, 2002) for flow through the glottis;
Hanna, Smith & Wolfe (2016) for the walls; Dang & Honda (1997) and Dang,
Honda & Suzuki (1994) for the piriform sinuses and nose; Flanagan (1972)
for lip radiation; Stevens (1971) and Shadle (1985) for turbulence;
Peterson & Barney (1952) for the vowel targets.

## Making your game talk

Add the library (`synthie-trombone`, in `crates/synthie-trombone`) to your
project, tune a `Talker` once, and turn each line into mono `f32` samples
for whatever plays your audio:

```rust
use synthie_trombone::{Preset, Talker, VoiceSettings, normalize};

let mut shopkeeper = Talker::new(44_100.0, Preset::Elder);
let mut line = shopkeeper.bake("Welcome, traveller!");
normalize(&mut line, 0.9);
// with game-toolkit: audio.play_samples(&line, 44_100)?;

let goblin = VoiceSettings { nasality: 0.4, rate: 1.5, ..Preset::Critter.settings() };
let squeak = Talker::new(44_100.0, goblin).bake("Shiny! Mine!");
```

For live dialogue, `Talker::say` queues text and `Talker::render` fills your
audio buffers as they come. The library never opens an audio device, so it
doesn't compete with kira (or whatever else you use) for the output, and it
builds for `wasm32-unknown-unknown` and without `std` (enable `libm`, plus
`alloc` for speech).

If you'd rather play it as an instrument from your own audio callback,
`Trombone` gives you the voice directly: note on and off, MIDI pitch, glide,
tongue, up to four constrictions, and the velum, with no allocation in the
audio path.

## When the pronunciation is off

Text goes through English letter-to-sound rules (digraphs, the silent *e*,
soft *c* and *g*, unstressed vowels) and a dictionary of about a hundred
common and irregular words, and numbers are spelled out. Consonants take the
tongue position of the next vowel, stops get a closure, a burst and a
breath, and the pitch drifts down over a phrase, falls at `.` and rises at
`?`. Rules get some English spellings wrong; when a word comes out odd, you
can write its phonemes in ARPAbet in square brackets (a `1` marks the
stressed vowel):

```text
[HH AX L OW1], pink trombone!
```

## A trombone walks into a synthesizer

I had been building [synthie](https://github.com/sunsided/synth), a
chiptune synthesizer in Rust, and at some point the question came up whether
the trombone could live there too. It turns out somebody already did the
port: the [`pink-trombone`](https://crates.io/crates/pink-trombone) crate
translates [chdh's TypeScript version](https://github.com/chdh/pink-trombone-mod)
to Rust, more or less line by line. I did my own anyway, partly because I
wanted it inside an audio callback (the port is a faithful translation, but
it allocates in the audio path, and a real-time callback is the one place
that can't have that), and partly because once the tract was running I
wanted it to sing notes, speak typed text, and sound like more than one
person.

The vowels were the part I enjoyed most. My first formant probe was a plain
DFT peak search, which happily reported 210 Hz for every vowel I threw at it
(the glottal source tilts the spectrum so hard that the low harmonics win
every time). Switching to LPC gave sensible numbers, and from there a grid
search over tongue position, tongue height and lip rounding fitted each vowel
to Peterson & Barney's 1952 averages for adult male speakers; all nine land
within about 2 % on F1 and F2. The search runs for about four minutes
(`task tune:vowels`), and `task formants` prints what each vowel measures
today.

The screenshots above come from the real UI driven by a real voice;
`task screenshots` renders them again.

## Development

| Task | What it does |
|---|---|
| `task check` | rustfmt, Clippy with warnings as errors, `cargo check`, the `no_std` builds |
| `task test` | all tests |
| `task lint:actions` | actionlint and zizmor over the workflows, same versions as CI |

CI runs the same gates (plus wasm32, docs and the MSRV) on every push and
pull request. Pushing a `v*` tag that matches the workspace version builds
the terminal app for Linux, macOS and Windows and attaches it to a GitHub
release.

## License

EUPL-1.2 (`LICENSE`), like synthie and game-toolkit. Derived from Pink
Trombone, © 2017 Neil Thapen, used under the MIT licence; its notice is in
`NOTICE-pink-trombone`.
