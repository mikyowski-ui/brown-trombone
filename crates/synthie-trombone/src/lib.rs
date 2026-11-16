//! # synthie-trombone
//!
//! A real-time [Pink Trombone] vocal tract synthesizer, made to slot into
//! [`synthie`](https://crates.io/crates/synthie) and anything else that
//! wants `f32` samples.
//!
//! Pink Trombone, by Neil Thapen (2017), models speech with a
//! Liljencrants–Fant glottal source driving a 44-cell Kelly–Lochbaum
//! waveguide of the vocal tract, with a nasal side branch, turbulence noise
//! at constrictions and bursts when closures release.
//!
//! This implementation keeps that model and adds:
//!
//! - **Real-time safety:** fixed-size state, no allocation, no trait objects
//!   or locks in the audio path; non-finite state resets itself.
//! - **Sample-rate independence:** the model runs at a fixed
//!   [`INTERNAL_RATE`] and is resampled, so formants do not shift with the
//!   host rate, and parameter blocks do not depend on the host buffer size.
//! - **An instrument API:** note on/off, MIDI pitch, glide, voicing rates,
//!   up to [`MAX_CONSTRICTIONS`] simultaneous constrictions, an explicit
//!   velum.
//! - **Phonemes:** an ARPAbet [`Phoneme`] set with articulations.
//! - **Voices:** [`Preset`]s (deep, bright, child, elder, giant, critter,
//!   robot, whisper) built from pitch, *vocal tract length*, tenseness,
//!   jitter and intonation; tune your own with [`VoiceSettings`].
//! - **[`Talker`]** (`alloc`): tune once, then [`bake`](Talker::bake)
//!   sentences into buffers for any audio engine, or stream them.
//! - **Text-to-speech** (`alloc` feature): [`speech`] turns English text into
//!   a timed articulation track with simple prosody, and plays it.
//! - **`no_std`:** disable `std` and enable `libm`.
//!
//! For games and other one-shot uses:
//!
//! ```
//! use synthie_trombone::{Preset, Talker};
//!
//! let mut npc = Talker::new(44_100.0, Preset::Elder);
//! let samples: Vec<f32> = npc.bake("Stay a while, and listen.");
//! // hand `samples` (mono, 44.1 kHz) to your audio engine
//! ```
//!
//! As an instrument:
//!
//! ```
//! use synthie_trombone::{Phoneme, Trombone};
//!
//! let mut voice = Trombone::new(48_000.0);
//! voice.articulate_phoneme(Phoneme::Aa);
//! voice.note_on_midi(57.0);
//! let mut buf = [0.0f32; 512];
//! voice.render(&mut buf);
//! ```
//!
//! [Pink Trombone]: https://dood.al/pinktrombone/

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod gesture;
mod glottis;
pub mod inverse;
pub mod listen;
mod math;
#[cfg(feature = "alloc")]
pub mod midi;
pub mod mimic;
mod noise;
mod phoneme;
mod phys;
mod preset;
#[cfg(feature = "profile")]
mod profile;
#[cfg(feature = "alloc")]
pub mod sing;
#[cfg(feature = "alloc")]
pub mod speech;
#[cfg(feature = "alloc")]
mod talker;
mod tract;
mod tuning;
mod voice;
mod vowel_size;
pub mod vowel_targets;

/// LPC formant estimation, shared with the examples.
#[cfg(test)]
#[path = "../examples/support/lpc.rs"]
mod lpc;

pub use gesture::{Gesture, Tier, TierActivity};
pub use math::midi_to_hz;
pub use phoneme::{Articulation, Manner, Phoneme};
pub use preset::{Persona, Preset, VoiceSettings};
#[cfg(feature = "profile")]
pub use profile::{ProfileError, tuning_parameters};
#[cfg(feature = "alloc")]
pub use talker::{Talker, bake, normalize};
pub use tract::{
    BLADE_START, Constriction, LIP_START, MAX_CONSTRICTIONS, N, NOSE_LEN, NOSE_START, PHARYNX,
    TIP_START, TONGUE_DIAMETER, TONGUE_INDEX,
};
pub use tuning::{Field, Pass, Tuning};
pub use voice::{Engine, Fallback, Gauges, GlottalSource, SLOT_LIPS, SLOT_MAIN, Trombone};

/// Rate at which the glottis runs; the tract runs at twice this.
pub const INTERNAL_RATE: f32 = 48_000.0;

/// The README examples compile and run.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
