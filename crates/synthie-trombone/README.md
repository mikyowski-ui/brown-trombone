# synthie-trombone

A real-time [Pink Trombone](https://dood.al/pinktrombone/) vocal tract
synthesizer for Rust, made for [synthie](https://crates.io/crates/synthie):
play it like an instrument, shape vowels and consonants, or type text and
hear it spoken.

Pink Trombone (Neil Thapen, 2017) drives a 44-cell Kelly–Lochbaum waveguide
of the vocal tract, with a nasal branch, from a Liljencrants–Fant glottal
source, and adds turbulence at constrictions and bursts when closures
release. This crate keeps that model and adds:

- **Real-time safe:** fixed-size state; no allocation, locks or trait
  objects in the audio path; recovers from non-finite state on its own.
- **Sample-rate independent:** runs at a fixed 48 kHz internally and
  resamples, so formants don't shift at 44.1 or 96 kHz. Parameter blocks
  don't depend on the host buffer size either.
- **Built as an instrument:** note on/off, MIDI pitch, glide, voicing
  attack/release, up to four simultaneous constrictions, explicit velum.
- **Measured vowels:** ARPAbet phonemes. The vowels were grid-searched
  against Peterson & Barney formant data, and every vowel lands within
  about 2 % on F1/F2 (`examples/tune_vowels.rs`, `examples/formants.rs`).
- **Text to speech** (`alloc`): a pronouncing dictionary (2,000 words
  built in, 1,200 more with the default `dictionary` feature), known stems
  with regular endings, English spelling rules for the rest, ARPAbet in
  `[brackets]`, numbers, stress, coarticulation, declining intonation,
  falls at `.` and rises at `?`. `speech::phonemize_with` takes your own
  lexicon in place of the built-in one.
- **`no_std`:** `default-features = false, features = ["libm"]` (add
  `alloc` for speech).

## Voices for games

Pick a voice once, then turn lines of dialogue into sample buffers for
whatever plays your audio:

```rust
use synthie_trombone::{Preset, Talker, VoiceSettings, normalize};

// Built-in voices: Deep, Bright, Child, Elder, Giant, Critter, Robot, Whisper.
let mut shopkeeper = Talker::new(44_100.0, Preset::Elder);
let mut line = shopkeeper.bake("Welcome, traveller!");
normalize(&mut line, 0.9); // mono f32 at 44.1 kHz, ready to play

// Or tune your own: a nasal, fast-talking goblin.
let goblin = VoiceSettings {
    nasality: 0.4,
    rate: 1.5,
    ..Preset::Critter.settings()
};
let mut goblin = Talker::new(44_100.0, goblin);
let squeak = goblin.bake("Shiny!");
# assert!(!squeak.is_empty());
```

Pitch and vocal tract length vary independently from person to person:
raising only the pitch sounds like the same speaker reaching higher, while a
shorter tract shifts every resonance and sounds like a smaller speaker.
Tenseness, jitter, vibrato, nasality, speaking rate and intonation range
complete the character. With the `serde` feature, `VoiceSettings` can live
in your game's data files.

## Instrument and streaming

```rust
use synthie_trombone::{Phoneme, Trombone};
use synthie_trombone::speech::{Prosody, Speaker, Utterance};

let mut voice = Trombone::new(48_000.0);

// Sing an "ah" at A3.
voice.articulate_phoneme(Phoneme::Aa);
voice.note_on_midi(57.0);
let mut buf = [0.0f32; 256];
voice.render(&mut buf);

// Speak. Compile off the audio thread; Speaker::process never allocates.
let utterance = Utterance::from_text("Hello, pink trombone!", &Prosody::default());
let mut speaker = Speaker::new();
Speaker::prepare(&mut voice);
speaker.speak(utterance);
for s in &mut buf {
    *s = speaker.process(&mut voice);
}
```

To drive a running voice from another thread (a controller, a language
model sending text a few words at a time), use a
[`VoiceStream`](src/stream.rs): a single-producer single-consumer pair of
fixed capacity. The controller pushes timed events, gestures, utterances and
text; the audio callback renders with the `VoicePlayer`, which carries each
event out before its sample and never allocates, locks or waits. A push
that does not fit returns `Full` with what was pushed.

```rust
use synthie_trombone::stream::{Event, voice_stream};
use synthie_trombone::{Phoneme, Trombone};

let (mut stream, mut player) = voice_stream(1024);   // 1024 words of ring
let mut voice = Trombone::new(48_000.0);
stream.push(Event::Articulate(Phoneme::Aa)).unwrap(); // as soon as possible
stream.push_at(0, Event::NoteOn(57.0)).unwrap();      // before sample 0
stream.push_at(24_000, Event::NoteOff).unwrap();      // half a second later
stream.push_text("Hello the");                        // held: the sentence
stream.push_text("re, my friend. And so");            // ends here, so it goes
let mut block = [0.0f32; 256];
player.render(&mut voice, &mut block);                // in the audio callback
```

Text is cut at the ends of sentences (or of phrases), each piece compiled on
the controller's thread once all of it has arrived and queued after what is
being said, with the pause the compiler leaves between the two in one text:
a text sent in chunks is spoken as the whole text would be (to within 2 ms
over three sentences, in `tests/stream_tests.rs`). Its latency is the time
the rest of the sentence takes to arrive, plus 20 to 50 µs of compiling. The
ring and the events are `no_std` (`StreamBuffer::split`); utterances and text
need `std`. `examples/stream.rs` feeds a voice from a script of timed events
and from stdin.

## Gestures

`gesture` is a task-dynamic gesture score (Saltzman & Munhall 1989) on six
independent tiers, `Tier::ALL`: `GlottalAdduction`, `LarynxAndPitch`,
`Velum`, `TongueBody`, `TongueTip` and `LipsAndJaw`. A `Gesture` is a
target for its tier's variables (`gesture::var`: a constriction's place and
degree, the lips' aperture and protrusion, the port, the glottis's spread,
the pitch and the larynx's height; the protrusion and the height lengthen
the physical tract with `AnatomyConfig::lip_horn` and `larynx_travel` on;
with `AnatomyConfig::tongue_model` 1 the tongue body is three numbers of
measured MRI modes, `mode_1..3`, see `inverse::Shape::modes`),
moved towards as a mass on a spring of its own stiffness and damping,
weighted by its dominance against the other gestures active on the tier,
inside an activation window with ramps. Push one into a playing voice and
read what each tier is doing:

```rust
use synthie_trombone::gesture::{Target, Window, var};
use synthie_trombone::{Engine, Gesture, Phoneme, Tier, Trombone};

let mut voice = Trombone::new(48_000.0);
voice.set_engine(Engine::Physical);
voice.articulate_phoneme(Phoneme::Aa);
voice.note_on();
// A tongue-tip closure for 80 ms, from 10 ms from now, ramped in and out.
let closure = Target::NONE
    .with(var::tip::LOCATION, 36.0)
    .with(var::tip::DEGREE, 0.0);
let taken = voice.push_gesture(
    Gesture::new(Tier::TongueTip, closure).window(Window::between(0.01, 0.09).ramps(0.01, 0.02)),
);
assert!(taken);
let mut buf = [0.0f32; 1024]; // 21 ms
voice.render(&mut buf);
let tip = voice.tier_activity()[Tier::TongueTip.index()];
assert_eq!(tip.gestures, 1);
```

The score holds `gesture::Score::CAPACITY` gestures at once and never
allocates; a gesture is any target on any tier, not a phoneme. Speech on
the physical engine plays through it: each segment becomes gestures on the
tiers that make it, and a consonant made by the tip or the lips leaves the
tongue body and the lips to the vowel it leans on, whose gestures span it
(`ArticulationConfig::gesture_score`, on by default; `0` is the segment
path of earlier rounds, `gesture_overlap` which rules are left to overlap).

## Two engines

`Engine::Classic` (the default) is Pink Trombone's own model.
`Engine::Physical` runs the same tract in real units: self-oscillating
vocal folds under lung pressure, frequency-dependent losses, yielding
walls, side cavities and airflow-driven noise. Notes outside the folds'
range (`Trombone::fold_range`) are sung by a prescribed pulse unless
`set_fallback(Fallback::Never)`; `Gauges::sounding` says which source is
singing. Its speech has a voiced/voiceless contrast made by the larynx
(the glottis spread for /p t k s/, a voice onset time after a release,
a voice that lasts through /b d g/), timed from the moment the mouth
actually shuts (`Trombone::mouth_shut`, `Gauges::mouth_shut`); a slow
tongue body and jaw under quick tip and lips; vowels from published
formants; and breath and intonation shaped per phrase. Choose it per
voice with `VoiceSettings::with_engine`, or on a `Trombone` with
`set_engine`. The physical tract's lowest cells, the epilaryngeal tube
above the folds and its flare into the pharynx, are shaped by
`AnatomyConfig::epilarynx_*` (the original's tube by default); after
changing them, solve the vowel shapes again
(`cargo run -p synthie-trombone --example vowel_size --release`, after
`examples/babble.rs` for the babble map).

An utterance can also be spoken with a person's prosody:
`speech::Schedule` holds a recording's phone times, pitch track and
breath per phone (the repository's `tools/judge/prosody.py` measures
them), `Schedule::phrase` moves the text's phrase breaks to the person's
pauses, and `Utterance::follow` gives every phone the person's duration
(shared out over its segments as the rules had it, closures kept at their
floor) and the utterance a pitch track (`Utterance::set_pitch_track`) that
follows the person's contour phone by phone, in place of the rules'
declination, accents and final fall. `Utterance::follow_with` takes more
of the speaker, within the limits of `speech::FollowOptions` (from the
`follow_*` fields of `Tuning`): each sonorant's tenseness, each vowel's
F1 and F2 (a shape solved through the babble map, held by the vowel and
the consonants made on its body; `Trombone::hold_speech_vowel`), and
faster articulators through phones shorter than the rules make them
(`Trombone::set_articulation_boost`), which are themselves made no shorter
than 70 % of their own rule length (by class since round 11, see
`Tuning::follow_shortest_light`).

Both engines read their numbers from `Tuning`, in three parts:
`AnatomyConfig` (the body, its tissues and sources: 33 numbers),
`ArticulationConfig` (how sounds are made and joined: 71, with a
`TierDynamics` spring for each of the six tiers of the gesture score) and
`ProsodyConfig` (the rhythm and melody of a phrase: 43), each number with
a default and a plausible range (`Tuning::FIELDS`; the classic engine
reads twelve of them). A JSON profile can set them by name, flat or by
part (`profile` feature, `VoiceSettings::with_profile`). The repository README has the details,
the measurements the physical engine is held to and how a speech
recognizer scores it.

## Examples

```sh
cargo run -p synthie-trombone --example say --release -- --voice bright "Can you hear me?"  # → say.wav
cargo run -p synthie-trombone --example voices --release                       # → voices/*.wav
cargo run -p synthie-trombone --example vowels --release                       # → vowels.wav
cargo run -p synthie-trombone --example formants --release                     # F1–F3 table
cargo run -p synthie-trombone --example say --release -- --engine physical "Hello!"  # the physical engine
cargo run -p synthie-trombone --example say --release -- --engine physical \
    --match-prosody schedule.json "the recorded text"                          # a person's timing, pitch and voice
    # (--prosody-voice / -tenseness / -speed off leave a part out; --prosody-formants on adds the speaker's vowels)
cargo run -p synthie-trombone --example ab --release                           # both engines → report/
cargo run -p synthie-trombone --example probe --release                        # measurements of both
cargo run -p synthie-trombone --example epilarynx --release                    # the tube above the folds: F3-F5, the 3 kHz band
cargo run -p synthie-trombone --example epilarynx --release -- --coupling      # ... and what it does to the folds
echo "Text to say." | cargo run -p synthie-trombone --example stream --release -- --engine physical  # → stream.wav
```

## License

EUPL-1.2 (`LICENSE`). Derived from Pink Trombone, © 2017 Neil Thapen,
used under the MIT licence; its notice is in `NOTICE-pink-trombone`.
