//! [`VoiceStream`]: feed a running voice from another thread, without
//! allocating or waiting in the audio path.
//!
//! A stream is a single-producer single-consumer pair. The controller keeps
//! the [`VoiceStream`] and pushes into it: control events the voice already
//! understands ([`Event`]: notes, pitch, the tract, the voice), gestures for
//! the gesture score, and (with `std`) whole utterances and text arriving
//! piece by piece. Whoever renders the voice keeps the [`VoicePlayer`] and
//! calls [`VoicePlayer::process`] in place of [`Trombone::process`]: it
//! carries out each event at its sample and plays the speech.
//!
//! - **Fixed capacity, nothing blocks.** Events go through a ring of
//!   32-bit words whose size is fixed when the stream is made (an event
//!   takes 2 to 10 words). A push that does not fit returns
//!   [`Full`] with what was pushed, and nothing of it is queued; the
//!   controller can try again once the player has caught up. The player
//!   never waits for the controller.
//! - **No allocation, no lock in [`VoicePlayer::process`].** The ring is
//!   atomics only (and safe code). Utterances are compiled by the controller
//!   (compiling allocates) and handed over whole through a bounded channel
//!   whose slots exist from the start; the player only swaps them in, and
//!   sends the ones it is done with back to be freed by the controller.
//! - **Timing.** [`VoiceStream::push`] is carried out as soon as the player
//!   gets to it: at the next sample it renders. [`VoiceStream::push_at`]
//!   names a sample on the player's clock ([`VoicePlayer::now`]: samples
//!   rendered since the stream was made) and is carried out exactly before
//!   that sample, or at once if it is already past. Events are carried out
//!   in the order they were pushed: one timed later than the event after it
//!   holds that one back.
//! - **`no_std`.** The ring and the events need neither `std` nor an
//!   allocator: a [`StreamBuffer`] holds the words, and
//!   [`StreamBuffer::split`] gives the two ends. With `alloc`,
//!   [`voice_stream`] makes a pair that owns its buffer. Utterances, text
//!   and whole [`VoiceSettings`](crate::VoiceSettings) need `std`.
//!
//! ## Text, piece by piece
//!
//! [`VoiceStream::push_text`] takes text as it arrives (from a language
//! model, a typist, a socket) and speaks it as one continuous voice. The
//! text is cut into sentences ([`Granularity::Sentence`], the default) or
//! phrases ([`Granularity::Phrase`]); each is compiled as soon as all of it
//! has arrived, on the controller's thread, and queued after what is being
//! said. A piece is held back until its end is known, so a sentence that
//! arrives in chunks has the intonation, rhythm and breath of the same
//! sentence arriving whole: nothing resets where the chunks meet. The
//! pause between two pieces is the one the compiler leaves between them in
//! one text (the next piece is compiled once more after the one before to
//! measure it, and the player waits, or hands over early in the closing
//! silence, by the difference), so the stream comes out as long as the
//! whole text spoken as one utterance, to within a few milliseconds.
//!
//! The latency of a piece is therefore the time the rest of it takes to
//! arrive, plus its compiling (some 20 to 50 µs for a sentence of eight
//! words, measured in `stream_tests`), plus nothing on the player's side
//! if something is still being said (the piece waits its turn) or the start
//! of the next sample if not. A piece that never ends (no full stop) is
//! spoken by [`VoiceStream::flush_text`].
//!
//! ## Gestures
//!
//! Gestures ([`Gesture`](crate::Gesture)) go to the voice's gesture score
//! through the same ring ([`VoiceStream::push_gesture`]): the player gives
//! each to [`Trombone::push_gesture`] at its sample, and its window counts
//! from there.
//!
//! ```
//! use synthie_trombone::stream::{Event, voice_stream};
//! use synthie_trombone::{Phoneme, Trombone};
//!
//! let (mut stream, mut player) = voice_stream(1024);
//! let mut voice = Trombone::new(48_000.0);
//! stream.push(Event::Articulate(Phoneme::Aa)).unwrap();
//! stream.push(Event::NoteOn(57.0)).unwrap();
//! stream.push_at(4_800, Event::NoteOff).unwrap();
//! # #[cfg(feature = "std")]
//! stream.push_text("Hello there, ");
//! # #[cfg(feature = "std")]
//! stream.push_text("my friend. And");
//! let mut block = [0.0f32; 512];
//! player.render(&mut voice, &mut block); // in the audio callback
//! ```

mod ring;
#[cfg(feature = "std")]
mod text;

#[cfg(feature = "std")]
pub use text::Granularity;

use crate::gesture::{Target, VARIABLES, Window};
use crate::inverse::Shape;
use crate::{Constriction, Listener, Phoneme, Stereo, Trombone};
use crate::{Gesture, Tier};
#[cfg(feature = "std")]
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::{AtomicU32, Ordering};
use ring::{Counters, MAX_MESSAGE};

#[cfg(feature = "std")]
use crate::VoiceSettings;
#[cfg(feature = "std")]
use crate::speech::{Prosody, Speaker, Utterance};
#[cfg(feature = "std")]
use alloc::string::String;
#[cfg(feature = "alloc")]
use alloc::{boxed::Box, sync::Arc, vec::Vec};
#[cfg(feature = "std")]
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, TrySendError, sync_channel};

/// A control event the voice understands.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Event {
    /// Start phonating at a (fractional) MIDI note.
    NoteOn(f32),
    /// Stop phonating.
    NoteOff,
    /// Pitch as a MIDI note, voicing left alone.
    Note(f32),
    /// Pitch in Hz.
    Pitch(f32),
    /// The voicing gate.
    Voiced(bool),
    /// Phonate continuously (see [`Trombone::set_always_voice`]).
    AlwaysVoice(bool),
    /// Glottal tenseness, `0‥1`.
    Tenseness(f32),
    /// Lung pressure of the physical engine, cmH2O.
    Lung(f32),
    /// Vibrato depth (relative) and rate (Hz).
    Vibrato {
        /// Depth.
        amount: f32,
        /// Rate, Hz.
        hz: f32,
    },
    /// Glide speed of the pitch (see [`Trombone::set_glide`]).
    Glide(f32),
    /// Tongue body target.
    Tongue {
        /// Position (cell index).
        index: f32,
        /// Height (diameter under the tongue).
        diameter: f32,
    },
    /// Place, move or release (`None`) the constriction in a slot.
    Constriction {
        /// Slot, `0..MAX_CONSTRICTIONS`.
        slot: u8,
        /// The constriction.
        constriction: Option<Constriction>,
    },
    /// Velum target, `0‥1`.
    Velum(f32),
    /// Move to a phoneme's articulation.
    Articulate(Phoneme),
    /// Hold a vowel shape of the physical engine.
    Shape(Shape),
    /// Vocal tract length (relative; see [`Trombone::set_tract_length`]).
    TractLength(f32),
    /// Articulator speed multiplier.
    ArticulationSpeed(f32),
    /// A gesture for the gesture score (see [`VoiceStream::push_gesture`];
    /// a score that is full drops it, as [`Trombone::push_gesture`] does).
    Gesture(Gesture),
    /// Stop speaking, and drop the speech queued after it.
    StopSpeaking,
    /// Move the listener (see [`Trombone::set_listener`]): it glides there.
    Listener(Listener),
    /// How the two channels are heard (see [`Trombone::set_stereo`]).
    Stereo(Stereo),
}

/// A push that did not fit: the stream is full. Nothing of it was queued;
/// it is handed back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Full<T>(pub T);

impl<T> core::fmt::Display for Full<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("the voice stream is full")
    }
}

/// When a pushed item is carried out.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum When {
    /// As soon as the player gets to it.
    #[default]
    Now,
    /// Before this sample of the player's clock (see [`VoicePlayer::now`]).
    At(u64),
}

impl When {
    fn at(self) -> Option<u64> {
        match self {
            When::Now => None,
            When::At(t) => Some(t),
        }
    }
}

// ----- the wire format -------------------------------------------------------

const TAG_NOTE_ON: u8 = 1;
const TAG_NOTE_OFF: u8 = 2;
const TAG_NOTE: u8 = 3;
const TAG_PITCH: u8 = 4;
const TAG_VOICED: u8 = 5;
const TAG_ALWAYS: u8 = 6;
const TAG_TENSENESS: u8 = 7;
const TAG_LUNG: u8 = 8;
const TAG_VIBRATO: u8 = 9;
const TAG_GLIDE: u8 = 10;
const TAG_TONGUE: u8 = 11;
const TAG_CONSTRICTION: u8 = 12;
const TAG_VELUM: u8 = 13;
const TAG_ARTICULATE: u8 = 14;
const TAG_SHAPE: u8 = 15;
const TAG_LENGTH: u8 = 16;
const TAG_SPEED: u8 = 17;
const TAG_GESTURE: u8 = 18;
const TAG_STOP: u8 = 19;
const TAG_LISTENER: u8 = 40;
const TAG_STEREO: u8 = 41;
/// An utterance waits in the hand-over channel (payload: [`SpeakMode`]).
#[cfg(feature = "std")]
const TAG_SPEAK: u8 = 20;
/// Voice settings wait in the hand-over channel.
#[cfg(feature = "std")]
const TAG_VOICE: u8 = 21;

/// Header bit: two words of sample time follow.
const TIMED: u32 = 1 << 16;

fn message_len(header: u32) -> usize {
    ((header >> 8) & 0xFF) as usize
}

/// One message being written.
struct Message {
    words: [u32; MAX_MESSAGE],
    len: usize,
}

impl Message {
    fn new(tag: u8, at: Option<u64>) -> Self {
        let mut m = Self {
            words: [0; MAX_MESSAGE],
            len: 1,
        };
        m.words[0] = u32::from(tag);
        if let Some(t) = at {
            m.words[0] |= TIMED;
            #[allow(clippy::cast_possible_truncation)]
            {
                m.word(t as u32);
                m.word((t >> 32) as u32);
            }
        }
        m
    }

    fn word(&mut self, w: u32) {
        self.words[self.len] = w;
        self.len += 1;
    }

    fn float(&mut self, x: f32) {
        self.word(x.to_bits());
    }

    /// The message's words, its length in the header.
    fn finish(&mut self) -> &[u32] {
        #[allow(clippy::cast_possible_truncation)]
        let len = self.len as u32;
        self.words[0] |= len << 8;
        &self.words[..self.len]
    }
}

/// One message being read.
struct Reader<'a> {
    words: &'a [u32],
    at: usize,
}

impl Reader<'_> {
    fn word(&mut self) -> u32 {
        let w = self.words.get(self.at).copied().unwrap_or(0);
        self.at += 1;
        w
    }

    fn float(&mut self) -> f32 {
        f32::from_bits(self.word())
    }

    /// A float that stands for `None` when it is not a number.
    fn maybe(&mut self) -> Option<f32> {
        let x = self.float();
        (!x.is_nan()).then_some(x)
    }
}

#[allow(clippy::too_many_lines)]
fn encode(event: &Event, at: Option<u64>) -> Message {
    let tag = match event {
        Event::NoteOn(_) => TAG_NOTE_ON,
        Event::NoteOff => TAG_NOTE_OFF,
        Event::Note(_) => TAG_NOTE,
        Event::Pitch(_) => TAG_PITCH,
        Event::Voiced(_) => TAG_VOICED,
        Event::AlwaysVoice(_) => TAG_ALWAYS,
        Event::Tenseness(_) => TAG_TENSENESS,
        Event::Lung(_) => TAG_LUNG,
        Event::Vibrato { .. } => TAG_VIBRATO,
        Event::Glide(_) => TAG_GLIDE,
        Event::Tongue { .. } => TAG_TONGUE,
        Event::Constriction { .. } => TAG_CONSTRICTION,
        Event::Velum(_) => TAG_VELUM,
        Event::Articulate(_) => TAG_ARTICULATE,
        Event::Shape(_) => TAG_SHAPE,
        Event::TractLength(_) => TAG_LENGTH,
        Event::ArticulationSpeed(_) => TAG_SPEED,
        Event::Gesture(_) => TAG_GESTURE,
        Event::StopSpeaking => TAG_STOP,
        Event::Listener(_) => TAG_LISTENER,
        Event::Stereo(_) => TAG_STEREO,
    };
    let mut m = Message::new(tag, at);
    match *event {
        Event::NoteOn(x)
        | Event::Note(x)
        | Event::Pitch(x)
        | Event::Tenseness(x)
        | Event::Lung(x)
        | Event::Glide(x)
        | Event::Velum(x)
        | Event::TractLength(x)
        | Event::ArticulationSpeed(x) => m.float(x),
        Event::Voiced(b) | Event::AlwaysVoice(b) => m.word(u32::from(b)),
        Event::Vibrato { amount, hz } => {
            m.float(amount);
            m.float(hz);
        }
        Event::Tongue { index, diameter } => {
            m.float(index);
            m.float(diameter);
        }
        Event::Constriction { slot, constriction } => {
            m.word(u32::from(slot));
            m.float(constriction.map_or(f32::NAN, |c| c.index));
            m.float(constriction.map_or(f32::NAN, |c| c.diameter));
        }
        Event::Articulate(p) => {
            let k = Phoneme::ALL.iter().position(|&q| q == p).unwrap_or(0);
            #[allow(clippy::cast_possible_truncation)]
            m.word(k as u32);
        }
        Event::Shape(s) => {
            m.float(s.tongue_index);
            m.float(s.tongue_diameter);
            m.float(s.lips.unwrap_or(f32::NAN));
            m.float(s.narrowing.map_or(f32::NAN, |n| n.0));
            m.float(s.narrowing.map_or(f32::NAN, |n| n.1));
            m.float(s.pharynx);
            for k in 0..3 {
                m.float(s.modes.map_or(f32::NAN, |q| q[k]));
            }
        }
        Event::Gesture(g) => {
            #[allow(clippy::cast_possible_truncation)]
            m.word(g.tier.index() as u32 | u32::from(g.flags) << 8 | u32::from(g.layer) << 16);
            let set = |v: usize| g.target.get(v);
            let mask = (0..VARIABLES)
                .filter(|&v| set(v).is_some())
                .fold(0u32, |m, v| m | 1 << v);
            m.word(mask);
            for v in 0..VARIABLES {
                if let Some(x) = set(v) {
                    m.float(x);
                }
            }
            for x in [
                g.stiffness,
                g.damping,
                g.dominance,
                g.window.start,
                g.window.end,
                g.window.ramp_in,
                g.window.ramp_out,
            ] {
                m.float(x);
            }
        }
        Event::Listener(l) => {
            for x in [
                l.azimuth,
                l.elevation,
                l.distance,
                l.heard_azimuth,
                l.heard_elevation,
            ] {
                m.float(x);
            }
        }
        Event::Stereo(s) => m.word(match s {
            Stereo::Off => 0,
            Stereo::Binaural => 1,
            Stereo::Pair => 2,
        }),
        Event::NoteOff | Event::StopSpeaking => {}
    }
    m
}

/// What a message read off the ring says.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Item {
    Event(Event),
    /// An utterance waits in the hand-over channel, and the gap before it.
    #[cfg(feature = "std")]
    Speak(SpeakMode, f32),
    /// Voice settings wait in the hand-over channel.
    #[cfg(feature = "std")]
    Voice,
}

#[allow(clippy::too_many_lines)]
fn decode(words: &[u32]) -> (Option<u64>, Option<Item>) {
    let header = words.first().copied().unwrap_or(0);
    let mut r = Reader { words, at: 1 };
    let at = (header & TIMED != 0).then(|| {
        let lo = u64::from(r.word());
        lo | (u64::from(r.word()) << 32)
    });
    #[allow(clippy::cast_possible_truncation)]
    let tag = header as u8;
    let event = match tag {
        TAG_NOTE_ON => Event::NoteOn(r.float()),
        TAG_NOTE_OFF => Event::NoteOff,
        TAG_NOTE => Event::Note(r.float()),
        TAG_PITCH => Event::Pitch(r.float()),
        TAG_VOICED => Event::Voiced(r.word() != 0),
        TAG_ALWAYS => Event::AlwaysVoice(r.word() != 0),
        TAG_TENSENESS => Event::Tenseness(r.float()),
        TAG_LUNG => Event::Lung(r.float()),
        TAG_VIBRATO => Event::Vibrato {
            amount: r.float(),
            hz: r.float(),
        },
        TAG_GLIDE => Event::Glide(r.float()),
        TAG_TONGUE => Event::Tongue {
            index: r.float(),
            diameter: r.float(),
        },
        TAG_CONSTRICTION => {
            #[allow(clippy::cast_possible_truncation)]
            let slot = r.word() as u8;
            let (index, diameter) = (r.maybe(), r.float());
            Event::Constriction {
                slot,
                constriction: index.map(|i| Constriction::new(i, diameter)),
            }
        }
        TAG_VELUM => Event::Velum(r.float()),
        TAG_ARTICULATE => {
            let k = r.word() as usize;
            Event::Articulate(Phoneme::ALL.get(k).copied().unwrap_or(Phoneme::Sil))
        }
        TAG_SHAPE => Event::Shape(Shape {
            tongue_index: r.float(),
            tongue_diameter: r.float(),
            lips: r.maybe(),
            narrowing: {
                let (a, b) = (r.maybe(), r.float());
                a.map(|a| (a, b))
            },
            pharynx: r.float(),
            modes: {
                let q = [r.float(), r.float(), r.float()];
                q.iter().all(|v| v.is_finite()).then_some(q)
            },
        }),
        TAG_LENGTH => Event::TractLength(r.float()),
        TAG_SPEED => Event::ArticulationSpeed(r.float()),
        TAG_GESTURE => {
            let head = r.word();
            let tier = Tier::ALL
                .get((head & 0xFF) as usize)
                .copied()
                .unwrap_or(Tier::GlottalAdduction);
            let mask = r.word();
            let mut target = Target::NONE;
            for v in 0..VARIABLES {
                if mask & (1 << v) != 0 {
                    target.set(v, r.float());
                }
            }
            #[allow(clippy::cast_possible_truncation)]
            let (flags, layer) = ((head >> 8) as u8, (head >> 16) as u8);
            Event::Gesture(Gesture {
                tier,
                target,
                stiffness: r.float(),
                damping: r.float(),
                dominance: r.float(),
                window: Window {
                    start: r.float(),
                    end: r.float(),
                    ramp_in: r.float(),
                    ramp_out: r.float(),
                },
                flags,
                layer,
            })
        }
        TAG_STOP => Event::StopSpeaking,
        TAG_LISTENER => Event::Listener(Listener {
            azimuth: r.float(),
            elevation: r.float(),
            distance: r.float(),
            heard_azimuth: r.float(),
            heard_elevation: r.float(),
        }),
        TAG_STEREO => Event::Stereo(match r.word() {
            1 => Stereo::Binaural,
            2 => Stereo::Pair,
            _ => Stereo::Off,
        }),
        #[cfg(feature = "std")]
        TAG_SPEAK => {
            let mode = if r.word() == 1 {
                SpeakMode::Interrupt
            } else {
                SpeakMode::Queue
            };
            return (at, Some(Item::Speak(mode, r.float())));
        }
        #[cfg(feature = "std")]
        TAG_VOICE => return (at, Some(Item::Voice)),
        _ => return (at, None),
    };
    (at, Some(Item::Event(event)))
}

// ----- the shared buffer -----------------------------------------------------

#[cfg(target_has_atomic = "64")]
type ClockWord = core::sync::atomic::AtomicU64;
#[cfg(not(target_has_atomic = "64"))]
type ClockWord = AtomicU32;

/// What both ends share besides the words.
#[derive(Debug)]
struct Shared {
    counters: Counters,
    /// The player's clock, as it last told it.
    clock: ClockWord,
    /// Utterances sent and not yet begun or dropped by the player.
    #[cfg(feature = "std")]
    queued: AtomicUsize,
}

impl Shared {
    const fn new() -> Self {
        Self {
            counters: Counters::new(),
            clock: ClockWord::new(0),
            #[cfg(feature = "std")]
            queued: AtomicUsize::new(0),
        }
    }

    fn set_clock(&self, now: u64) {
        #[allow(clippy::cast_possible_truncation)]
        self.clock.store(now as _, Ordering::Relaxed);
    }

    #[allow(clippy::useless_conversion)] // a narrower clock on some targets
    fn clock(&self) -> u64 {
        u64::from(self.clock.load(Ordering::Relaxed))
    }
}

/// The words of a stream, for a `no_std` build without an allocator: put
/// it where both ends can reach it (a `static`, or the stack of a scope
/// both threads live in) and [`split`](Self::split) it. `WORDS` must be a
/// power of two.
#[derive(Debug)]
pub struct StreamBuffer<const WORDS: usize> {
    shared: Shared,
    words: [AtomicU32; WORDS],
}

impl<const WORDS: usize> Default for StreamBuffer<WORDS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const WORDS: usize> StreamBuffer<WORDS> {
    /// An empty buffer.
    ///
    /// # Panics
    ///
    /// If `WORDS` is not a power of two of at least 16.
    #[must_use]
    pub const fn new() -> Self {
        assert!(
            WORDS.is_power_of_two() && WORDS >= MAX_MESSAGE,
            "a stream buffer holds a power of two of at least 16 words"
        );
        Self {
            shared: Shared::new(),
            words: [const { AtomicU32::new(0) }; WORDS],
        }
    }

    /// The two ends of a stream through this buffer, empty: the controller's
    /// and the player's. Gestures and events only; utterances and text need
    /// [`voice_stream`] (and `std`).
    pub fn split(&mut self) -> (VoiceStream<'_>, VoicePlayer<'_>) {
        self.shared.counters.clear();
        self.shared.set_clock(0);
        let link = || Link::Borrowed {
            shared: &self.shared,
            words: &self.words,
        };
        (VoiceStream::new(link()), VoicePlayer::new(link()))
    }
}

#[cfg(feature = "alloc")]
#[derive(Debug)]
struct Owned {
    shared: Shared,
    words: Box<[AtomicU32]>,
}

#[derive(Debug)]
enum Link<'a> {
    Borrowed {
        shared: &'a Shared,
        words: &'a [AtomicU32],
    },
    #[cfg(feature = "alloc")]
    Owned(Arc<Owned>),
}

impl Link<'_> {
    fn parts(&self) -> (&Shared, &[AtomicU32]) {
        match self {
            Link::Borrowed { shared, words } => (shared, words),
            #[cfg(feature = "alloc")]
            Link::Owned(o) => (&o.shared, &o.words),
        }
    }
}

/// Utterances queued at most at a time (sent and not yet begun).
#[cfg(feature = "std")]
pub const UTTERANCES: usize = 8;
/// Utterances the player can hand back to be freed before it has to free
/// one itself.
#[cfg(feature = "std")]
const TRASH: usize = 64;

/// How an utterance is started.
#[cfg(feature = "std")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpeakMode {
    /// After everything queued before it.
    Queue,
    /// At once, dropping what is being said and queued.
    Interrupt,
}

/// What travels by the hand-over channel.
#[cfg(feature = "std")]
#[allow(clippy::large_enum_variant)] // the channel's slots exist up front
enum Heavy {
    Speech(Utterance),
    Voice(VoiceSettings),
}

/// A stream with a ring of `words` 32-bit words (rounded up to a power of
/// two, at least 16): the controller's end and the player's. An event takes
/// 2 to 10 words, so 1024 words hold some hundred events. With `std` it also
/// carries utterances, text and voice settings (see the [module](self)).
#[cfg(feature = "alloc")]
#[must_use]
pub fn voice_stream(words: usize) -> (VoiceStream<'static>, VoicePlayer<'static>) {
    let n = words.max(MAX_MESSAGE).next_power_of_two();
    let owned = Arc::new(Owned {
        shared: Shared::new(),
        words: (0..n).map(|_| AtomicU32::new(0)).collect::<Vec<_>>().into(),
    });
    #[allow(unused_mut)]
    let (mut stream, mut player) = (
        VoiceStream::new(Link::Owned(Arc::clone(&owned))),
        VoicePlayer::new(Link::Owned(owned)),
    );
    #[cfg(feature = "std")]
    {
        let (heavy_tx, heavy_rx) = sync_channel(UTTERANCES + 4);
        let (trash_tx, trash_rx) = sync_channel(TRASH);
        stream.lane = Some(Lane {
            heavy: heavy_tx,
            trash: trash_rx,
        });
        player.talk = Some(Talk::new(heavy_rx, trash_tx));
    }
    (stream, player)
}

// ----- the controller's end --------------------------------------------------

/// The controller's end of the hand-over channel.
#[cfg(feature = "std")]
#[derive(Debug)]
struct Lane {
    heavy: SyncSender<Heavy>,
    trash: Receiver<Utterance>,
}

/// What [`VoiceStream::push_text`] did.
#[cfg(feature = "std")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextSent {
    /// Pieces compiled and queued by this call.
    pub utterances: usize,
    /// Bytes of text held back: the piece still arriving, and anything
    /// that did not fit (it goes out on a later push or flush).
    pub held: usize,
    /// Whether the stream was full, so that complete text is held back.
    pub full: bool,
}

/// The controller's end of a stream (see the [module](self)). Pushing never
/// blocks.
#[derive(Debug)]
pub struct VoiceStream<'a> {
    link: Link<'a>,
    #[cfg(feature = "std")]
    lane: Option<Lane>,
    #[cfg(feature = "std")]
    text: String,
    #[cfg(feature = "std")]
    prosody: Prosody,
    #[cfg(feature = "std")]
    granularity: Granularity,
    #[cfg(feature = "std")]
    last_piece: Option<Piece>,
}

impl<'a> VoiceStream<'a> {
    fn new(link: Link<'a>) -> Self {
        Self {
            link,
            #[cfg(feature = "std")]
            lane: None,
            #[cfg(feature = "std")]
            text: String::new(),
            #[cfg(feature = "std")]
            prosody: Prosody::default(),
            #[cfg(feature = "std")]
            granularity: Granularity::default(),
            #[cfg(feature = "std")]
            last_piece: None,
        }
    }

    fn send(&self, m: &mut Message) -> bool {
        let (shared, words) = self.link.parts();
        shared.counters.write(words, m.finish())
    }

    /// Carry out `event` as soon as the player gets to it.
    ///
    /// # Errors
    ///
    /// [`Full`] with the event when the ring has no room for it.
    pub fn push(&mut self, event: Event) -> Result<(), Full<Event>> {
        self.push_when(When::Now, event)
    }

    /// Carry out `event` before sample `at` of the player's clock.
    ///
    /// # Errors
    ///
    /// [`Full`] with the event when the ring has no room for it.
    pub fn push_at(&mut self, at: u64, event: Event) -> Result<(), Full<Event>> {
        self.push_when(When::At(at), event)
    }

    /// Carry out `event` `when` says.
    ///
    /// # Errors
    ///
    /// [`Full`] with the event when the ring has no room for it.
    pub fn push_when(&mut self, when: When, event: Event) -> Result<(), Full<Event>> {
        self.collect();
        if self.send(&mut encode(&event, when.at())) {
            #[cfg(feature = "std")]
            if event == Event::StopSpeaking {
                self.last_piece = None;
            }
            Ok(())
        } else {
            Err(Full(event))
        }
    }

    /// Send a gesture to the voice's gesture score
    /// ([`Trombone::push_gesture`]); its window counts from when the player
    /// carries it out.
    ///
    /// # Errors
    ///
    /// [`Full`] with the gesture when the ring has no room for it.
    pub fn push_gesture(&mut self, gesture: Gesture, when: When) -> Result<(), Full<Gesture>> {
        self.push_when(when, Event::Gesture(gesture))
            .map_err(|_| Full(gesture))
    }

    /// Take a breath: the voice's lungs draw `cm_h2o` below the air's
    /// pressure for `seconds` (a [`Gesture::inhalation`] with 30 ms ramps,
    /// as [`Trombone::breathe_in`] takes; physical engine).
    ///
    /// # Errors
    ///
    /// [`Full`] with the gesture when the ring has no room for it.
    pub fn breathe_in(
        &mut self,
        cm_h2o: f32,
        seconds: f32,
        when: When,
    ) -> Result<(), Full<Gesture>> {
        self.push_gesture(Gesture::inhalation(cm_h2o, seconds, 0.03), when)
    }

    /// Words free in the ring.
    #[must_use]
    pub fn free(&self) -> usize {
        let (shared, words) = self.link.parts();
        shared.counters.free(words)
    }

    /// The player's clock as it last told it (it tells it every sample;
    /// see [`VoicePlayer::now`]).
    #[must_use]
    pub fn now(&self) -> u64 {
        self.link.parts().0.clock()
    }

    /// Free the utterances the player is done with (every push does this
    /// too).
    pub fn collect(&mut self) {
        #[cfg(feature = "std")]
        if let Some(lane) = &self.lane {
            while lane.trash.try_recv().is_ok() {}
        }
    }
}

// What does not fit is handed back whole, settings and utterances included.
#[cfg(feature = "std")]
#[allow(clippy::result_large_err)]
impl VoiceStream<'_> {
    /// Send something by the hand-over channel with the ring's message
    /// that says it is there; `false`, with nothing sent, if either has
    /// no room.
    fn hand_over(&mut self, heavy: Heavy, mut marker: Message) -> Result<(), Heavy> {
        self.collect();
        let Some(lane) = &self.lane else {
            return Err(heavy);
        };
        let speech = matches!(heavy, Heavy::Speech(_));
        let (shared, words) = self.link.parts();
        let marker = marker.finish();
        // The ring's room only grows while the controller waits, so it is
        // checked first: an item in the channel always gets its message.
        if shared.counters.free(words) < marker.len()
            || (speech && shared.queued.load(Ordering::Acquire) >= UTTERANCES)
        {
            return Err(heavy);
        }
        if speech {
            shared.queued.fetch_add(1, Ordering::AcqRel);
        }
        match lane.heavy.try_send(heavy) {
            Ok(()) => {
                let sent = shared.counters.write(words, marker);
                debug_assert!(sent);
                Ok(())
            }
            Err(TrySendError::Full(h) | TrySendError::Disconnected(h)) => {
                if speech {
                    shared.queued.fetch_sub(1, Ordering::AcqRel);
                }
                Err(h)
            }
        }
    }

    /// Speak `utterance` `when` says, after everything queued before it
    /// (seamlessly: see the [module](self)).
    ///
    /// # Errors
    ///
    /// [`Full`] with the utterance when [`UTTERANCES`] are already queued,
    /// or the ring has no room; also for a stream without a hand-over
    /// channel (one from [`StreamBuffer::split`]).
    pub fn queue(&mut self, utterance: Utterance, when: When) -> Result<(), Full<Utterance>> {
        self.last_piece = None;
        self.utter(utterance, when, SpeakMode::Queue, 0.0)
    }

    /// Speak `utterance` `when` says, dropping what is being said and
    /// queued then.
    ///
    /// # Errors
    ///
    /// As [`queue`](Self::queue).
    pub fn speak(&mut self, utterance: Utterance, when: When) -> Result<(), Full<Utterance>> {
        self.last_piece = None;
        self.utter(utterance, when, SpeakMode::Interrupt, 0.0)
    }

    /// Hand `utterance` over; a queued one starts `gap` seconds after the
    /// one before it ends (before, if negative, while that one is still in
    /// its closing silence).
    fn utter(
        &mut self,
        utterance: Utterance,
        when: When,
        mode: SpeakMode,
        gap: f32,
    ) -> Result<(), Full<Utterance>> {
        let mut m = Message::new(TAG_SPEAK, when.at());
        m.word(u32::from(mode == SpeakMode::Interrupt));
        m.float(gap);
        self.hand_over(Heavy::Speech(utterance), m)
            .map_err(|h| match h {
                Heavy::Speech(u) => Full(u),
                Heavy::Voice(_) => unreachable!(),
            })
    }

    /// Apply voice settings (preset, size, tuning…) `when` says. They
    /// also become the prosody text is compiled with.
    ///
    /// # Errors
    ///
    /// [`Full`] with the settings when the hand-over channel or the ring
    /// has no room.
    pub fn push_voice(
        &mut self,
        settings: VoiceSettings,
        when: When,
    ) -> Result<(), Full<VoiceSettings>> {
        let m = Message::new(TAG_VOICE, when.at());
        self.hand_over(Heavy::Voice(settings), m)
            .map_err(|h| match h {
                Heavy::Voice(s) => Full(s),
                Heavy::Speech(_) => unreachable!(),
            })?;
        self.prosody = settings.prosody();
        Ok(())
    }

    /// How text pushed from now on is compiled (pitch, rate, engine…).
    pub fn set_prosody(&mut self, prosody: Prosody) {
        self.prosody = prosody;
    }

    /// The prosody text is compiled with.
    #[must_use]
    pub fn prosody(&self) -> &Prosody {
        &self.prosody
    }

    /// Where text is cut for compiling (see [`Granularity`]).
    pub fn set_granularity(&mut self, granularity: Granularity) {
        self.granularity = granularity;
    }

    /// Text held back: the piece still arriving, and anything complete
    /// that did not fit yet.
    #[must_use]
    pub fn held_text(&self) -> &str {
        &self.text
    }

    /// Text arriving: `chunk` is added to what has arrived, and every piece
    /// that is now complete is compiled here and queued to be spoken after
    /// what is queued before it. Nothing is lost: what does not fit stays
    /// held back and goes out with a later push or [`flush_text`](Self::flush_text).
    pub fn push_text(&mut self, chunk: &str) -> TextSent {
        self.text.push_str(chunk);
        let mut sent = TextSent::default();
        while let Some(end) = text::first_piece(&self.text, self.granularity) {
            if !self.send_piece(end, &mut sent) {
                break;
            }
        }
        sent.held = self.text.len();
        sent
    }

    /// Speak the text held back, as the end of a piece (a sentence that
    /// never got its full stop).
    pub fn flush_text(&mut self) -> TextSent {
        let mut sent = self.push_text("");
        if !sent.full && !self.text.is_empty() {
            self.send_piece(self.text.len(), &mut sent);
        }
        sent.held = self.text.len();
        sent
    }

    /// Compile and queue the first `end` bytes of the held text; `false`
    /// (with the text kept) if the stream is full.
    fn send_piece(&mut self, end: usize, sent: &mut TextSent) -> bool {
        let piece = &self.text[..end];
        if piece.trim().is_empty() {
            self.text.drain(..end);
            return true;
        }
        let mut u = Utterance::from_text(piece, &self.prosody);
        if u.is_empty() {
            self.text.drain(..end);
            return true;
        }
        // The pause the compiler leaves between this piece and the one
        // before, had they come as one text; less what the two leave on
        // their own (the end of the one, the start of the other).
        let mut gap = 0.0;
        if let Some(last) = &self.last_piece {
            let both = Utterance::from_text(&(last.text.clone() + piece), &self.prosody);
            // Had they come as one text, the lungs would draw a breath in
            // the pause between them (wave 3: between sentences, where the
            // utterance itself opens without one, `breath_first` 0): the
            // piece takes it before itself, and the gap is what is left of
            // the pause.
            if breath_between(&both, last.phones) && !breath_before(&u) {
                let mut first = self.prosody;
                first.tuning.prosody.breath_first = 1.0;
                u = Utterance::from_text(piece, &first);
            }
            gap = pause_between(&both, last.phones)
                .map_or(0.0, |between| between - last.trailing - silence(&u, false));
        }
        let last = Piece {
            text: piece.into(),
            phones: phones(&u),
            trailing: silence(&u, true),
        };
        if self.utter(u, When::Now, SpeakMode::Queue, gap).is_err() {
            sent.full = true;
            return false;
        }
        self.last_piece = Some(last);
        self.text.drain(..end);
        sent.utterances += 1;
        true
    }

    /// Forget the text before: what is pushed next starts afresh (after a
    /// [`speak`](Self::speak), a [`queue`](Self::queue) or
    /// [`Event::StopSpeaking`] this happens by itself).
    pub fn forget_text(&mut self) {
        self.last_piece = None;
    }
}

/// The last piece of text sent: what the next one is compiled after.
#[cfg(feature = "std")]
#[derive(Clone, Debug)]
struct Piece {
    text: String,
    /// Phones it has.
    phones: usize,
    /// Seconds of silence it ends with.
    trailing: f32,
}

/// Phones of the text `u` was compiled from.
#[cfg(feature = "std")]
fn phones(u: &Utterance) -> usize {
    (0..u.segments().len())
        .filter_map(|k| u.phone_of(k))
        .max()
        .map_or(0, |p| p + 1)
}

/// Seconds of silence `u` begins with (or, with `end`, ends with): its
/// segments before its first phone (after its last).
#[cfg(feature = "std")]
fn silence(u: &Utterance, end: bool) -> f32 {
    let s = u.segments();
    let quiet = |k: &usize| u.phone_of(*k).is_none();
    if end {
        (0..s.len())
            .rev()
            .take_while(quiet)
            .map(|k| s[k].duration)
            .sum()
    } else {
        (0..s.len()).take_while(quiet).map(|k| s[k].duration).sum()
    }
}

/// Whether `u` draws a breath between its phone `phones − 1` and its
/// phone `phones`.
#[cfg(feature = "std")]
fn breath_between(u: &Utterance, phones: usize) -> bool {
    let s = u.segments();
    let before = (0..s.len()).rfind(|&k| u.phone_of(k).is_some_and(|p| p < phones));
    let after = (0..s.len()).find(|&k| u.phone_of(k).is_some_and(|p| p >= phones));
    match (before, after) {
        (Some(b), Some(a)) => s[b + 1..a].iter().any(|x| x.gesture.intake > 0.0),
        _ => false,
    }
}

/// Whether `u` draws a breath before its first phone.
#[cfg(feature = "std")]
fn breath_before(u: &Utterance) -> bool {
    let s = u.segments();
    (0..s.len())
        .take_while(|&k| u.phone_of(k).is_none())
        .any(|k| s[k].gesture.intake > 0.0)
}

/// Seconds of silence in `u` between its phone `phones − 1` and its phone
/// `phones`.
#[cfg(feature = "std")]
fn pause_between(u: &Utterance, phones: usize) -> Option<f32> {
    let s = u.segments();
    let before = (0..s.len()).rfind(|&k| u.phone_of(k).is_some_and(|p| p < phones))?;
    let after = (0..s.len()).find(|&k| u.phone_of(k).is_some_and(|p| p >= phones))?;
    Some(s[before + 1..after].iter().map(|x| x.duration).sum())
}

// ----- the player's end ------------------------------------------------------

/// The next event, read off the ring.
#[derive(Clone, Copy, Debug)]
struct Due {
    at: Option<u64>,
    item: Item,
}

/// The player's end of a stream: render the voice with it in place of
/// [`Trombone::process`]. Nothing in it allocates, locks or waits.
#[derive(Debug)]
pub struct VoicePlayer<'a> {
    link: Link<'a>,
    now: u64,
    due: Option<Due>,
    #[cfg(feature = "std")]
    talk: Option<Talk>,
}

impl<'a> VoicePlayer<'a> {
    fn new(link: Link<'a>) -> Self {
        Self {
            link,
            now: 0,
            due: None,
            #[cfg(feature = "std")]
            talk: None,
        }
    }

    /// Samples rendered since the stream was made: the clock
    /// [`VoiceStream::push_at`] times events on.
    #[must_use]
    pub fn now(&self) -> u64 {
        self.now
    }

    /// Whether nothing is waiting: no event in the ring or due, nothing
    /// being said or queued.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        let (shared, _) = self.link.parts();
        let quiet = self.due.is_none() && shared.counters.waiting() == 0;
        #[cfg(feature = "std")]
        let quiet = quiet && self.talk.as_ref().is_none_or(Talk::is_idle);
        quiet
    }

    /// The speaker playing the stream's speech (`None` for a stream without
    /// utterances).
    #[cfg(feature = "std")]
    #[must_use]
    pub fn speaker(&self) -> Option<&Speaker> {
        self.talk.as_ref().map(|t| &t.speaker)
    }

    /// Carry out what is due, and render one sample of `voice`.
    pub fn process(&mut self, voice: &mut Trombone) -> f32 {
        loop {
            if self.due.is_none() {
                let (shared, words) = self.link.parts();
                let mut buf = [0u32; MAX_MESSAGE];
                let Some(len) = shared.counters.read(words, &mut buf, message_len) else {
                    break;
                };
                if let (at, Some(item)) = decode(&buf[..len]) {
                    self.due = Some(Due { at, item });
                }
                continue;
            }
            let Some(due) = self.due else { break };
            if due.at.is_some_and(|t| t > self.now) {
                break;
            }
            self.due = None;
            self.carry_out(voice, due.item);
        }
        #[cfg(feature = "std")]
        let out = match &mut self.talk {
            Some(talk) => talk.process(voice, self.now, self.link.parts().0),
            None => voice.process(),
        };
        #[cfg(not(feature = "std"))]
        let out = voice.process();
        self.now += 1;
        self.link.parts().0.set_clock(self.now);
        out
    }

    /// Fill `out` with mono samples (see [`process`](Self::process)).
    pub fn render(&mut self, voice: &mut Trombone, out: &mut [f32]) {
        for s in out {
            *s = self.process(voice);
        }
    }

    /// [`process`](Self::process) as a frame, `[left, right]` (see
    /// [`Trombone::set_stereo`]).
    pub fn process_stereo(&mut self, voice: &mut Trombone) -> [f32; 2] {
        self.process(voice);
        voice.frame()
    }

    /// Fill `out` with interleaved frames, left then right (see
    /// [`process_stereo`](Self::process_stereo); an odd last sample is
    /// left alone).
    pub fn render_stereo(&mut self, voice: &mut Trombone, out: &mut [f32]) {
        for frame in out.as_chunks_mut::<2>().0 {
            *frame = self.process_stereo(voice);
        }
    }

    fn carry_out(&mut self, voice: &mut Trombone, item: Item) {
        let event = match item {
            Item::Event(e) => e,
            #[cfg(feature = "std")]
            Item::Speak(mode, gap) => {
                let shared = self.link.parts().0;
                if let Some(talk) = &mut self.talk {
                    talk.take_speech(voice, mode, gap, shared);
                }
                return;
            }
            #[cfg(feature = "std")]
            Item::Voice => {
                if let Some(talk) = &mut self.talk
                    && let Ok(Heavy::Voice(settings)) = talk.heavy_next()
                {
                    settings.apply(voice);
                }
                return;
            }
        };
        match event {
            Event::NoteOn(note) => {
                voice.set_note(note);
                voice.note_on();
            }
            Event::NoteOff => voice.note_off(),
            Event::Note(note) => voice.set_note(note),
            Event::Pitch(hz) => voice.set_frequency(hz),
            Event::Voiced(on) => voice.set_voiced(on),
            Event::AlwaysVoice(on) => voice.set_always_voice(on),
            Event::Tenseness(t) => voice.set_tenseness(t),
            Event::Lung(p) => voice.set_lung_pressure(p),
            Event::Vibrato { amount, hz } => voice.set_vibrato(amount, hz),
            Event::Glide(speed) => voice.set_glide(speed),
            Event::Tongue { index, diameter } => voice.set_tongue(index, diameter),
            Event::Constriction { slot, constriction } => {
                voice.set_constriction(usize::from(slot), constriction);
            }
            Event::Velum(x) => voice.set_velum(x),
            Event::Articulate(p) => voice.articulate_phoneme(p),
            Event::Shape(s) => voice.set_shape(&s),
            Event::TractLength(l) => voice.set_tract_length(l),
            Event::ArticulationSpeed(f) => voice.set_articulation_speed(f),
            Event::Gesture(g) => {
                voice.push_gesture(g);
            }
            Event::StopSpeaking => {
                #[cfg(feature = "std")]
                {
                    let shared = self.link.parts().0;
                    if let Some(talk) = &mut self.talk {
                        talk.stop(voice, shared);
                    }
                }
            }
            Event::Listener(l) => voice.set_listener(l),
            Event::Stereo(s) => voice.set_stereo(s),
        }
    }
}

/// The player's speech: the speaker, the utterances queued for it, and the
/// way back for those it is done with.
#[cfg(feature = "std")]
#[derive(Debug)]
struct Talk {
    heavy: Receiver<Heavy>,
    trash: SyncSender<Utterance>,
    speaker: Speaker,
    /// Utterances queued, oldest first from `first`, each with the gap
    /// before it (seconds; see [`VoiceStream::utter`]).
    queue: [Option<(Utterance, f32)>; UTTERANCES],
    first: usize,
    queued: usize,
    state: State,
    /// The segment being spoken and the sample it began at.
    segment: (usize, u64),
    /// Utterances waiting for room on the way back.
    spare: [Option<Utterance>; 4],
    /// The voice's output rate, Hz.
    sample_rate: f32,
}

#[cfg(feature = "std")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// Nothing of the stream's is being said.
    Quiet,
    Speaking,
    /// Between two utterances: the next begins at this sample.
    Waiting(u64),
}

#[cfg(feature = "std")]
impl Talk {
    fn new(heavy: Receiver<Heavy>, trash: SyncSender<Utterance>) -> Self {
        Self {
            heavy,
            trash,
            speaker: Speaker::new(),
            queue: Default::default(),
            first: 0,
            queued: 0,
            state: State::Quiet,
            segment: (usize::MAX, 0),
            spare: Default::default(),
            sample_rate: 48_000.0,
        }
    }

    fn is_idle(&self) -> bool {
        self.state == State::Quiet && self.queued == 0
    }

    fn heavy_next(&self) -> Result<Heavy, TryRecvError> {
        self.heavy.try_recv()
    }

    /// Hand an utterance back to be freed by the controller; while there
    /// is no room it waits in a spare slot, and only when those are full
    /// too is it freed here (the controller has stopped collecting).
    fn trash(&mut self, u: Utterance) {
        // An utterance never compiled holds no memory.
        if u.is_empty() {
            return;
        }
        let u = match self.trash.try_send(u) {
            Ok(()) => return,
            Err(TrySendError::Full(u) | TrySendError::Disconnected(u)) => u,
        };
        if let Some(slot) = self.spare.iter_mut().find(|s| s.is_none()) {
            *slot = Some(u);
        }
    }

    /// Retry handing back what waits in the spare slots.
    fn retry_trash(&mut self) {
        for k in 0..self.spare.len() {
            if let Some(u) = self.spare[k].take() {
                match self.trash.try_send(u) {
                    Ok(()) => {}
                    Err(TrySendError::Full(u) | TrySendError::Disconnected(u)) => {
                        self.spare[k] = Some(u);
                        return;
                    }
                }
            }
        }
    }

    /// Take the utterance the ring said is waiting.
    fn take_speech(&mut self, voice: &mut Trombone, mode: SpeakMode, gap: f32, shared: &Shared) {
        let Ok(Heavy::Speech(u)) = self.heavy.try_recv() else {
            return;
        };
        if mode == SpeakMode::Interrupt {
            self.clear(shared);
            self.start(voice, u, shared);
            return;
        }
        if self.state == State::Quiet && self.queued == 0 {
            self.start(voice, u, shared);
        } else if self.queued < UTTERANCES {
            let at = (self.first + self.queued) % UTTERANCES;
            self.queue[at] = Some((u, gap));
            self.queued += 1;
        } else {
            // The controller never sends more than there are slots.
            shared.queued.fetch_sub(1, Ordering::AcqRel);
            self.trash(u);
        }
    }

    /// Begin speaking `u` (it leaves the count of those queued).
    fn start(&mut self, voice: &mut Trombone, u: Utterance, shared: &Shared) {
        shared.queued.fetch_sub(1, Ordering::AcqRel);
        if self.state != State::Speaking {
            Speaker::prepare(voice);
        }
        let old = self.speaker.speak(u);
        self.trash(old);
        self.state = State::Speaking;
        self.segment = (usize::MAX, 0);
    }

    /// Drop everything queued.
    fn clear(&mut self, shared: &Shared) {
        while let Some((u, _)) = self.next() {
            shared.queued.fetch_sub(1, Ordering::AcqRel);
            self.trash(u);
        }
    }

    fn stop(&mut self, voice: &mut Trombone, shared: &Shared) {
        self.clear(shared);
        if self.state == State::Speaking {
            let old = self.speaker.stop(voice);
            self.trash(old);
        }
        if self.state != State::Quiet {
            self.state = State::Quiet;
            restore_instrument(voice);
        }
    }

    fn next(&mut self) -> Option<(Utterance, f32)> {
        if self.queued == 0 {
            return None;
        }
        let u = self.queue[self.first].take();
        self.first = (self.first + 1) % UTTERANCES;
        self.queued -= 1;
        u
    }

    /// The gap before the next queued utterance.
    fn next_gap(&self) -> Option<f32> {
        if self.queued == 0 {
            return None;
        }
        self.queue[self.first].as_ref().map(|(_, gap)| *gap)
    }

    /// Whether the next utterance, due before this one ends, begins now:
    /// this one is in its closing silence and has no more than the gap
    /// left of it.
    fn hand_over_early(&self, now: u64) -> bool {
        let Some(gap) = self.next_gap().filter(|g| *g < 0.0) else {
            return false;
        };
        let Some((index, _)) = self.speaker.current() else {
            return false;
        };
        let rest = &self.speaker.utterance().segments()[index..];
        if rest.iter().any(|s| s.phoneme != Phoneme::Sil) {
            return false;
        }
        #[allow(clippy::cast_precision_loss)]
        let into = now.saturating_sub(self.segment.1) as f32 / self.sample_rate;
        let left: f32 = rest.iter().map(|s| s.duration).sum::<f32>() - into;
        left <= -gap
    }

    fn process(&mut self, voice: &mut Trombone, now: u64, shared: &Shared) -> f32 {
        if self.spare[0].is_some() {
            self.retry_trash();
        }
        match self.state {
            State::Quiet => return voice.process(),
            State::Waiting(at) => {
                if now < at {
                    return voice.process();
                }
                if let Some((u, _)) = self.next() {
                    self.start(voice, u, shared);
                } else {
                    self.state = State::Quiet;
                    restore_instrument(voice);
                    return voice.process();
                }
            }
            State::Speaking => {}
        }
        self.sample_rate = voice.sample_rate();
        let index = self.speaker.current().map_or(usize::MAX, |(i, _)| i);
        if index != self.segment.0 {
            self.segment = (index, now);
        }
        if self.hand_over_early(now)
            && let Some((u, _)) = self.next()
        {
            self.start(voice, u, shared);
        }
        let out = self.speaker.process(voice);
        if !self.speaker.is_speaking() {
            if let Some(gap) = self.next_gap() {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let wait = (gap.max(0.0) * self.sample_rate).round() as u64;
                self.state = State::Waiting(now + 1 + wait);
            } else {
                self.state = State::Quiet;
                restore_instrument(voice);
            }
        }
        out
    }
}

/// Back to the instrument's voicing rates, articulator speed and glide
/// after speech (the library's defaults; the rest stays as speech left it).
#[cfg(feature = "std")]
fn restore_instrument(voice: &mut Trombone) {
    voice.reset_voicing_rates();
    voice.set_articulation_speed(1.0);
    voice.set_glide(0.1);
}
