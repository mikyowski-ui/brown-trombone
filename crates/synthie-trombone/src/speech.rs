//! Text-to-speech front end.
//!
//! [`Utterance::from_text`] turns English text into a timed articulation
//! track; a [`Speaker`] plays it on a [`Trombone`] from the audio thread.
//!
//! Text goes through:
//!
//! 1. **Normalisation:** lower case, numbers spelled out, punctuation kept as
//!    phrase breaks.
//! 2. **Letter to sound:** a small exceptions dictionary for common and
//!    irregular words, then context-sensitive English spelling rules
//!    (digraphs, magic *e*, soft *c*/*g*, vowel reduction). Write ARPAbet in
//!    brackets to bypass the rules: `"[HH AX L OW1] world"`.
//! 3. **Timing:** per-manner durations, lengthened under stress and at phrase
//!    ends; stops become closure, burst and aspiration; diphthongs glide.
//! 4. **Coarticulation:** consonants take the tongue body of the next vowel.
//! 5. **Intonation:** declining pitch over each phrase, accents on stressed
//!    syllables, a fall at `.` and a rise at `?`.
//!
//! Compiling allocates; do it off the audio thread and hand the
//! [`Utterance`] to the [`Speaker`].

#![allow(clippy::match_same_arms, clippy::if_same_then_else)] // rule tables

use crate::math::{exp, lerp};
use crate::phoneme::{Articulation, Manner, Phoneme};
use crate::vowel_targets::VowelKey;
use crate::{Engine, SLOT_MAIN, Trombone, Tuning};
use alloc::string::String;
use alloc::vec::Vec;

/// Voice settings for compiling speech.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct Prosody {
    /// Baseline pitch in Hz.
    pub pitch: f32,
    /// Pitch excursion of accents and phrase ends, relative (`0.0` is a
    /// monotone robot, `1.0` is lively).
    pub expressiveness: f32,
    /// Speaking rate multiplier (`1.0` is about 4.5 syllables per second).
    pub rate: f32,
    /// Glottal tenseness while speaking.
    pub tenseness: f32,
    /// Velum opening during vowels and approximants (`0` oral, `1` very
    /// nasal).
    pub nasality: f32,
    /// How long sounds last and how early the articulators start for the
    /// next one (the timing fields of the voice's [`Tuning`]).
    pub tuning: Tuning,
    /// The sound model the speech is for. The physical engine gets the
    /// laryngeal and wall gestures of [`Gesture`] and the timing that goes
    /// with them (voice onset times, longer voiceless closures, vowels
    /// clipped before voiceless codas), and each vowel has a length of its
    /// own and glides as people's do (pass C of the [`Tuning`]); the
    /// classic engine keeps its own.
    pub engine: Engine,
}

impl Default for Prosody {
    fn default() -> Self {
        Self {
            pitch: 110.0,
            expressiveness: 0.5,
            rate: 1.0,
            tenseness: 0.65,
            nasality: 0.0,
            tuning: Tuning::DEFAULT,
            engine: Engine::Classic,
        }
    }
}

/// One timed target of an [`Utterance`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    /// The phoneme this segment realises.
    pub phoneme: Phoneme,
    /// Tract target (tongue always filled in).
    pub articulation: Articulation,
    /// Length in seconds.
    pub duration: f32,
    /// Pitch at the start and end, Hz.
    pub pitch: (f32, f32),
    /// Glottal tenseness.
    pub tenseness: f32,
    /// Whether the folds vibrate.
    pub voiced: bool,
    /// What the larynx and the walls do besides (physical engine).
    pub gesture: Gesture,
}

/// What the larynx and the walls of the tract do through a [`Segment`], on
/// top of the voicing gate: the second channel of a consonant, timed with
/// the tract's closure and release. The physical engine plays it; the
/// classic engine has nothing to play it on. All zero by default.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Gesture {
    /// Glottal opening held behind the folds, cm² at native size (see
    /// [`Trombone::set_glottal_spread`]).
    pub spread: f32,
    /// Share by which the fold cover slackens (see
    /// [`Trombone::set_fold_slack`]).
    pub slack: f32,
    /// How soft the voice is, `0‥1` (see
    /// [`Trombone::set_fold_softness`]).
    pub softness: f32,
    /// Active widening of the pharynx, cm³/s (see
    /// [`Trombone::set_cavity_expansion`]).
    pub expansion: f32,
    /// Pitch added at the start of the segment, Hz, decaying over the
    /// voice's [`Tuning::kick_time`].
    pub kick: f32,
    /// Semitones the pitch stands off the straight line from the start of
    /// the segment to its end at its middle, falling away as a parabola
    /// to either end (the physical engine's intonation: an accent peaking
    /// inside its vowel).
    pub pitch_arc: f32,
    /// Share of the voice's lung pressure held back at the start and at
    /// the end of the segment, in between in proportion (see
    /// [`Trombone::set_breath`]): `0` pushes with all of it, `1` not at
    /// all. A phrase builds up from nothing and dies away at its end.
    pub pressure_drop: (f32, f32),
    /// For a fricative: seconds at its start and at its end over which its
    /// gap is still forming and already opening again, wider by
    /// `gap_widening` (twice that at the end); `(0, 0)` holds the gap
    /// throughout.
    pub gap_opening: (f32, f32),
    /// How much wider the gap is while it forms, diameter units.
    pub gap_widening: f32,
    /// For the breath after a voiceless stop's release: how long after the
    /// mouth actually opens the voice comes back, seconds (a voice onset
    /// time, less the folds' own few milliseconds). The release is the
    /// instant the tract stops being shut, however early or late the
    /// articulators get there.
    pub hold: f32,
    /// For the release of a stop into a liquid (/bl pr gl kr/): how long
    /// after the mouth actually opens the next sound's shape is sent,
    /// seconds; until then the tract holds this segment's shape (the
    /// liquid's tongue body without its tip). `0` sends it as usual, a
    /// head start before the segment ends.
    pub transition: f32,
    /// How long before this segment begins its shape is sent, seconds, in
    /// place of the voice's [`Tuning::lead`] (or `release_lead` out of a
    /// closure): a quick gesture thrown late, as the tip of an alveolar
    /// tap is (see [`Tuning::tap_lead`]). `0` keeps the voice's.
    pub approach: f32,
}

/// A compiled, ready-to-play articulation track.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Utterance {
    segments: Vec<Segment>,
    /// Head start the articulators get on each sound, and out of a
    /// closure, seconds (physical engine).
    lead: f32,
    release_lead: f32,
    /// Seconds over which a segment's pitch kick decays.
    kick_time: f32,
    /// Head start of the larynx into a voiceless consonant, seconds.
    devoicing_lead: f32,
    /// Head start of the velum out of a nasal into an obstruent, seconds.
    velum_lead: f32,
    /// Shortest complete closure of a voiceless stop, seconds.
    closure_floor: f32,
    /// How long the voice goes on into a voiceless stop's closure, seconds.
    voicing_tail: f32,
    /// Out of a sibilant into /f v/: how long before the tongue lets go
    /// the lower lip starts, and how long it takes to reach the teeth,
    /// seconds (see [`Tuning::labial_lead`]).
    labial_lead: f32,
    labial_closing: f32,
    sibilant_exit: f32,
    /// Which phone of the text each segment realises (its place among the
    /// [`Token::Phone`]s, [`NO_PHONE`] for a pause); empty for hand-made
    /// segments.
    phones: Vec<u32>,
    /// Where each phone of the text stands in its word and syllable (empty
    /// for hand-made segments).
    roles: Vec<rhythm::Role>,
    /// A pitch track the [`Speaker`] follows in place of the segments' own
    /// pitch: Hz every `pitch_hop` seconds from the start of the utterance,
    /// on the clock of the segments' durations (see
    /// [`Utterance::set_pitch_track`]). Empty: the segments' pitch.
    pitch_track: Vec<f32>,
    pitch_hop: f32,
    /// What following a speaker's schedule asks of the [`Speaker`] beyond
    /// the segments (see [`Utterance::follow_with`]).
    followed: follow::Followed,
}

/// The phone of a pause, in [`Utterance`]'s record of which phone each
/// segment realises.
const NO_PHONE: u32 = u32::MAX;

/// A phoneme with its lexical context, the output of [`phonemize`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Token {
    /// A sound; `stressed` marks the accented syllable of a content word.
    Phone {
        /// The sound.
        phoneme: Phoneme,
        /// Primary stress.
        stressed: bool,
    },
    /// Boundary between words.
    WordBreak,
    /// Phrase break from punctuation.
    Break(Break),
}

/// Kinds of phrase break.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Break {
    /// `,` `;` `:` — short pause, pitch held.
    Comma,
    /// `.` — pitch falls.
    Period,
    /// `?` — pitch rises.
    Question,
    /// `!` — emphatic fall.
    Exclamation,
}

impl Utterance {
    /// Compile English text.
    #[must_use]
    pub fn from_text(text: &str, prosody: &Prosody) -> Self {
        Self::from_tokens(&phonemize(text), prosody)
    }

    /// An utterance from hand-made segments (see [`crate::sing`]).
    #[must_use]
    pub fn from_segments(segments: Vec<Segment>) -> Self {
        Self {
            segments,
            lead: Tuning::DEFAULT.lead,
            release_lead: Tuning::DEFAULT.release_lead,
            kick_time: Tuning::DEFAULT.kick_time,
            devoicing_lead: Tuning::DEFAULT.devoicing_lead,
            velum_lead: Tuning::DEFAULT.velum_lead,
            closure_floor: 0.0,
            voicing_tail: 0.0,
            labial_lead: 0.0,
            labial_closing: 0.0,
            sibilant_exit: 0.0,
            phones: Vec::new(),
            roles: Vec::new(),
            pitch_track: Vec::new(),
            pitch_hop: 0.0,
            followed: follow::Followed::default(),
        }
    }

    /// Compile a token stream (see [`phonemize`]).
    #[must_use]
    pub fn from_tokens(tokens: &[Token], prosody: &Prosody) -> Self {
        compile(tokens, prosody)
    }

    /// The segments, in order.
    #[must_use]
    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// The segments, to set their durations, pitch or breath by hand (see
    /// [`Utterance::follow`]). A voiceless stop's closure is still held for
    /// [`Tuning::closure_floor`] however short it is made.
    pub fn segments_mut(&mut self) -> &mut [Segment] {
        &mut self.segments
    }

    /// Which phone of the text segment `index` realises: its place among
    /// the [`Token::Phone`]s the utterance was compiled from (a stop's
    /// closure, release and breath, a diphthong's two parts and a lax
    /// vowel's glide all belong to one phone). `None` for a pause, and for
    /// hand-made segments.
    #[must_use]
    pub fn phone_of(&self, index: usize) -> Option<usize> {
        self.phones
            .get(index)
            .filter(|&&p| p != NO_PHONE)
            .map(|&p| p as usize)
    }

    /// Give the utterance a pitch track, Hz every `hop` seconds from its
    /// start, which the [`Speaker`] follows in place of each segment's own
    /// `pitch` (a segment's [`Gesture::kick`] still adds to it). Time on
    /// the track is the time the segments' durations add up to, so a
    /// closure held past its segment's end (see [`Tuning::closure_floor`])
    /// does not shift it. Past its end the last value holds. An empty track
    /// (or a `hop` that is not positive) gives the segments their own pitch
    /// back.
    pub fn set_pitch_track(&mut self, hop: f32, hz: Vec<f32>) {
        if hop > 0.0 && hop.is_finite() {
            self.pitch_track = hz;
            self.pitch_hop = hop;
        } else {
            self.pitch_track = Vec::new();
            self.pitch_hop = 0.0;
        }
    }

    /// The pitch track at `t` seconds (see [`Utterance::set_pitch_track`]).
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn track_at(&self, t: f32) -> Option<f32> {
        let track = &self.pitch_track;
        let last = track.len().checked_sub(1)?;
        let x = (t / self.pitch_hop).max(0.0);
        let i = (x as usize).min(last);
        let j = (i + 1).min(last);
        Some(lerp(track[i], track[j], (x - i as f32).min(1.0)))
    }

    /// Total length in seconds.
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.segments.iter().map(|s| s.duration).sum()
    }

    /// Whether there is nothing to say.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }
}

/// Plays [`Utterance`]s on a [`Trombone`]. Never allocates.
#[derive(Clone, Debug, Default)]
#[allow(clippy::struct_excessive_bools)] // independent flags
pub struct Speaker {
    utterance: Utterance,
    index: usize,
    elapsed: f32,
    started: bool,
    /// The next segment's shape has already been sent (physical engine).
    ahead: bool,
    /// The larynx has already let go of the voice for the next segment.
    larynx_ahead: bool,
    /// In a pause the articulators are still to be set for what follows
    /// (physical engine; see [`PAUSE_SETTLE`]).
    pause_pending: bool,
    /// They have been set, for the rest of this pause.
    readied: bool,
    /// Seconds into the pause being spoken.
    pause_time: f32,
    /// How far a fricative's gap has got (see [`Gesture::gap_opening`]):
    /// `0` still forming, `1` made, `2` opening again.
    gap_stage: u8,
    /// Seconds since the utterance began.
    clock: f32,
    /// The mouth was shut at the last sample (physical engine).
    was_shut: bool,
    /// When the voice comes back after a voiceless stop's release, on
    /// `clock`, while it is still held off.
    voice_at: Option<f32>,
    /// Time on `clock` the pitch kick of the segment counts from.
    kick_from: f32,
    /// A voiceless stop has begun but the mouth is not shut yet: the voice
    /// stops when it is, or at this time on `clock` at the latest.
    devoice_at: Option<f32>,
    /// Seconds the mouth has been sealed (shut, the velum too) without a
    /// break (physical engine).
    sealed_for: f32,
    /// A voiceless stop has shut the mouth with the voice still going: the
    /// voice is let go at this time on `clock` (see
    /// [`Tuning::voicing_tail`]).
    tail_until: Option<f32>,
    /// The segment's closure has been held for its floor (see
    /// [`Speaker::holding_closure`]).
    closure_made: bool,
    /// When the mouth last opened, on `clock` (physical engine).
    opened_at: f32,
    /// The lower lip on its way up to the teeth for an /f v/ after a
    /// sibilant (see [`Speaker::move_lip`]).
    labial: Option<LabialGesture>,
    /// Where the segment being spoken starts by the durations of those
    /// before it, seconds: the clock of a pitch track.
    nominal: f32,
}

/// The way out of a sibilant into /f v/: the sibilant's gap as it was,
/// the lip's gap the /f v/ wants, and the time on `clock` the two
/// gestures set off.
#[derive(Clone, Copy, Debug)]
struct LabialGesture {
    sibilant: crate::Constriction,
    lip: crate::Constriction,
    /// The lip's opening when it set off (touch diameter).
    open: f32,
    since: f32,
}

/// The constriction slot the lower lip moves in while the tongue still
/// holds the sibilant before it.
const LABIAL_SLOT: usize = 2;
/// Opening (touch diameter) the tongue's gap widens to as it lets go of a
/// sibilant into /f v/: the widest that still counts as a fricative.
const SIBILANT_LET_GO: f32 = 0.8;

/// How long the voice must have been let go in a pause before the
/// articulators are set for the sound after it, seconds (physical engine).
///
/// In a pause the mouth keeps the shape it ended the phrase with while the
/// voice and the breath die away (opening it with the voice still sounding
/// would end every phrase on a vowel nobody asked for, `... bowls-uh`).
/// Once all is quiet, and at the start of an utterance, the articulators
/// are put where the next sound begins, without moving through the air: a
/// speaker readies the tongue and lips before speaking, slowly enough that
/// nothing is heard, but moved at speech speed in a silent tract they push
/// out a puff of air that is heard as a thump. The next sound then starts
/// from its own shape, its tongue body already at the vowel it leads into.
const PAUSE_SETTLE: f32 = 0.08;

/// Pressure below the folds, cmH2O, under which a pause counts as quiet.
const QUIET_PRESSURE: f32 = 0.3;

/// Longest a voiceless stop waits for the mouth to shut before the voice
/// is let go anyway, seconds.
const DEVOICE_WAIT: f32 = 0.025;

/// Longest a voiceless stop's closure segment is drawn out to give it its
/// [`Tuning::closure_floor`], seconds: a closure that never forms is not
/// waited for.
const CLOSURE_OVERRUN: f32 = 0.03;

/// Velum openness under which the nose counts as shut for a stop's
/// closure.
const SEALED_VELUM: f32 = 0.05;

// On the physical engine the articulators get a head start on each sound
// (`Tuning::lead`): their tissue has inertia there, and people begin moving
// towards a sound before it is due; without it every shape arrives late
// and speech slurs. Out of a closure the head start is short
// (`Tuning::release_lead`): stops and nasals need their closure held, and
// the release should fall on the boundary.

impl Speaker {
    /// An idle speaker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Start speaking `utterance`, interrupting anything in progress. Returns
    /// the previous utterance so the caller can drop it off the audio thread.
    pub fn speak(&mut self, utterance: Utterance) -> Utterance {
        self.index = 0;
        self.elapsed = 0.0;
        self.started = false;
        self.ahead = false;
        self.larynx_ahead = false;
        self.pause_pending = false;
        self.readied = false;
        self.pause_time = 0.0;
        self.clock = 0.0;
        self.was_shut = false;
        self.voice_at = None;
        self.kick_from = 0.0;
        self.devoice_at = None;
        self.sealed_for = 0.0;
        self.tail_until = None;
        self.closure_made = false;
        self.opened_at = 0.0;
        self.labial = None;
        self.nominal = 0.0;
        core::mem::replace(&mut self.utterance, utterance)
    }

    /// Stop speaking (the voice fades out). Returns the utterance.
    pub fn stop(&mut self, voice: &mut Trombone) -> Utterance {
        voice.set_voiced(false);
        voice.set_breath(1.0);
        voice.set_glottal_spread(0.0);
        voice.set_fold_slack(0.0);
        voice.set_fold_softness(0.0);
        voice.set_cavity_expansion(0.0);
        voice.articulate_phoneme(Phoneme::Sil);
        voice.set_constriction(LABIAL_SLOT, None);
        self.labial = None;
        self.index = usize::MAX;
        core::mem::take(&mut self.utterance)
    }

    /// Whether an utterance is playing.
    #[must_use]
    pub fn is_speaking(&self) -> bool {
        self.index < self.utterance.segments.len()
    }

    /// Index and segment being spoken.
    #[must_use]
    pub fn current(&self) -> Option<(usize, &Segment)> {
        self.utterance
            .segments
            .get(self.index)
            .map(|s| (self.index, s))
    }

    /// The utterance being spoken (or last spoken).
    #[must_use]
    pub fn utterance(&self) -> &Utterance {
        &self.utterance
    }

    /// Configure `voice` for speech: fast voicing, nimble articulators,
    /// quick pitch tracking, no wobble. Call once before speaking.
    pub fn prepare(voice: &mut Trombone) {
        voice.set_voicing_rates(40.0, 40.0);
        voice.set_articulation_speed(2.2);
        voice.set_glide(f32::INFINITY);
        voice.set_wobble(false);
        voice.set_always_voice(false);
    }

    /// Advance one output sample and render it.
    #[allow(clippy::too_many_lines)] // one sample of every channel's timing
    pub fn process(&mut self, voice: &mut Trombone) -> f32 {
        if let Some(&seg) = self.utterance.segments.get(self.index) {
            let physical = voice.engine() == Engine::Physical;
            if !self.started {
                self.started = true;
                let pause = physical && seg.phoneme == Phoneme::Sil;
                if !pause {
                    self.readied = false;
                    self.pause_time = 0.0;
                }
                // The pause that ends the utterance readies nothing: the
                // breath dies away in it as its gesture says.
                let last = self.utterance.segments[self.index..]
                    .iter()
                    .all(|s| s.phoneme == Phoneme::Sil);
                self.pause_pending = pause && !self.readied && !last;
                self.gap_stage = u8::from(seg.gesture.gap_opening.0 <= 0.0);
                self.closure_made = false;
                if !self.ahead && !pause {
                    articulate_segment(
                        voice,
                        &self.utterance.followed,
                        self.index,
                        &entry_shape(&seg),
                    );
                }
                self.ahead = false;
                self.larynx_ahead = false;
                self.kick_from = self.clock;
                // A voice still held off after a voiceless release waits.
                let held = self.voice_at.is_some_and(|t| t > self.clock);
                if !held {
                    self.voice_at = None;
                }
                // Into a voiceless stop the voice goes on until the mouth
                // shuts (see `time_larynx`).
                let defer = physical
                    && !seg.voiced
                    && matches!(seg.phoneme.manner(), Manner::Stop | Manner::Affricate)
                    && voice.is_voiced()
                    && !voice.mouth_shut();
                self.devoice_at = defer.then_some(self.clock + DEVOICE_WAIT);
                // A voice dying away in a closure already made is let be;
                // into a voiceless stop from a nasal's closure, the voice of
                // the murmur dies away the same way.
                let closed_already = physical
                    && !seg.voiced
                    && matches!(seg.phoneme.manner(), Manner::Stop | Manner::Affricate)
                    && voice.is_voiced()
                    && voice.mouth_shut()
                    && self.tail_until.is_none();
                if closed_already {
                    self.tail_until = Some(self.clock + self.utterance.voicing_tail);
                    voice.set_glottal_spread(seg.gesture.spread);
                }
                let dying = physical && self.tail_until.is_some() && !seg.voiced;
                if !dying {
                    self.tail_until = None;
                }
                if !defer && !dying {
                    voice.set_voiced(seg.voiced && !held);
                    voice.set_tenseness(seg.tenseness);
                }
                if voice.engine() == Engine::Physical {
                    let g = &seg.gesture;
                    if (!held || !seg.voiced) && !defer && !dying {
                        voice.set_glottal_spread(g.spread);
                    }
                    voice.set_fold_slack(g.slack);
                    voice.set_fold_softness(g.softness);
                    voice.set_cavity_expansion(g.expansion);
                }
            }
            let t = (self.elapsed / seg.duration).min(1.0);
            if physical {
                // In a pause the lungs wait until the articulators are set
                // for what follows.
                let (from, to) = seg.gesture.pressure_drop;
                let breath = if self.pause_pending {
                    0.0
                } else {
                    1.0 - lerp(from, to, t)
                };
                voice.set_breath(breath);
            }
            let mut hz = self
                .utterance
                .track_at(self.nominal + self.elapsed.min(seg.duration))
                .unwrap_or_else(|| {
                    let hz = lerp(seg.pitch.0, seg.pitch.1, t);
                    if seg.gesture.pitch_arc == 0.0 {
                        hz
                    } else {
                        let arc = seg.gesture.pitch_arc * 4.0 * t * (1.0 - t);
                        hz * crate::math::powf(2.0, arc / 12.0)
                    }
                });
            if seg.gesture.kick.abs() > 0.0 && self.clock >= self.kick_from {
                hz += seg.gesture.kick
                    * exp(-(self.clock - self.kick_from) / self.utterance.kick_time);
            }
            voice.set_frequency(hz);
            if physical {
                self.time_larynx(voice, &seg);
            }
            self.elapsed += 1.0 / voice.sample_rate();
            self.clock += 1.0 / voice.sample_rate();
            if physical && seg.phoneme == Phoneme::Sil {
                self.pause_time += 1.0 / voice.sample_rate();
            }
            if self.pause_pending
                && (self.index == 0 || self.pause_time >= PAUSE_SETTLE)
                && !voice.is_voiced()
                && voice.intensity() <= 0.0
                && voice.gauges().subglottal_pressure < QUIET_PRESSURE
            {
                let segments = &self.utterance.segments[self.index..];
                if let Some(k) = segments.iter().position(|s| s.phoneme != Phoneme::Sil) {
                    let next = &segments[k];
                    articulate_segment(
                        voice,
                        &self.utterance.followed,
                        self.index + k,
                        &entry_shape(next),
                    );
                    voice.settle();
                    // What little still rings in the tract belongs to the
                    // old shape; let it go with it.
                    voice.reset();
                    self.ahead = true;
                }
                self.pause_pending = false;
                self.readied = true;
            }
            if !self.ahead
                && physical
                && let Some(&next) = self.utterance.segments.get(self.index + 1)
                && next.phoneme != Phoneme::Sil
            {
                let closed = matches!(
                    seg.phoneme.manner(),
                    Manner::Stop | Manner::Affricate | Manner::Nasal
                ) && seg
                    .articulation
                    .constriction
                    .is_some_and(|c| c.diameter <= 0.3);
                let lead = if next.gesture.approach > 0.0 {
                    next.gesture.approach
                } else if closed {
                    self.utterance.release_lead
                } else {
                    self.utterance.lead
                };
                // Out of a sibilant into /f v/ the lower lip sets off
                // for the teeth before the tongue lets go (see
                // `Speaker::move_lip`).
                if self.labial.is_none()
                    && self.utterance.sibilant_exit > 0.0
                    && is_sibilant(seg.phoneme)
                    && matches!(next.phoneme, Phoneme::F | Phoneme::V)
                    && let Some(c) = next.articulation.constriction
                    && self.elapsed
                        >= seg.duration
                            - (lead + self.utterance.labial_lead).min(0.6 * seg.duration)
                    && let Some(sibilant) = voice.constriction(SLOT_MAIN)
                {
                    let lip = crate::Constriction::new(
                        c.index,
                        (c.diameter + voice.tuning().fricative_gap).max(0.32),
                    );
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let cell = ((c.index + 0.5) as usize).min(crate::N - 1);
                    self.labial = Some(LabialGesture {
                        sibilant,
                        lip,
                        open: (voice.diameters()[cell] + 0.3).max(lip.diameter),
                        since: self.clock,
                    });
                }
                let send = if seg.gesture.transition > 0.0 {
                    // A stop's release into a liquid: the liquid comes once
                    // the release has been heard for its while.
                    !voice.mouth_shut() && self.clock - self.opened_at >= seg.gesture.transition
                } else {
                    self.elapsed >= seg.duration - lead.min(0.5 * seg.duration)
                        && !self.holding_closure(&seg)
                        && self
                            .labial
                            .is_none_or(|l| self.clock - l.since >= self.utterance.sibilant_exit)
                };
                if send {
                    articulate_segment(
                        voice,
                        &self.utterance.followed,
                        self.index + 1,
                        &entry_shape(&next),
                    );
                    self.ahead = true;
                }
            }
            self.move_lip(voice, is_sibilant(seg.phoneme) && !self.ahead);
            // Out of a nasal into an obstruent the velum shuts ahead of the
            // mouth's opening (`Tuning::velum_lead`), so the obstruent
            // starts with the nose shut.
            if physical
                && seg.phoneme.manner() == Manner::Nasal
                && let Some(next) = self.utterance.segments.get(self.index + 1)
                && is_obstruent(next.phoneme)
                && self.elapsed >= seg.duration - self.utterance.velum_lead.min(0.5 * seg.duration)
            {
                voice.set_velum(next.articulation.velum);
            }
            // A fricative's gap forming and opening again.
            let (into, out_of) = seg.gesture.gap_opening;
            let lead = self.utterance.lead;
            if physical && self.gap_stage == 0 && self.elapsed >= into - lead.min(0.5 * into) {
                articulate_segment(
                    voice,
                    &self.utterance.followed,
                    self.index,
                    &seg.articulation,
                );
                self.gap_stage = 1;
            }
            if physical
                && self.gap_stage == 1
                && out_of > 0.0
                && !self.ahead
                && self.elapsed >= seg.duration - out_of - lead.min(0.5 * out_of)
            {
                let a = widened(&seg.articulation, 2.0 * seg.gesture.gap_widening);
                articulate_segment(voice, &self.utterance.followed, self.index, &a);
                self.gap_stage = 2;
            }
            // Into a voiceless fricative the larynx moves a little ahead
            // too, so the voice stops as the tract narrows rather than
            // carrying on into the hiss. Into a voiceless stop it stops as
            // the mouth shuts (see `time_larynx`): letting go earlier blew
            // the open glottis's breath through the closing gap, a hiss
            // before every /t/.
            if !self.larynx_ahead
                && seg.voiced
                && physical
                && let Some(next) = self.utterance.segments.get(self.index + 1)
                && !next.voiced
                && next.phoneme.manner() == Manner::Fricative
                && self.elapsed
                    >= seg.duration - self.utterance.devoicing_lead.min(0.5 * seg.duration)
            {
                voice.set_voiced(false);
                voice.set_glottal_spread(next.gesture.spread);
                self.larynx_ahead = true;
            }
            if physical {
                let sealed = voice.mouth_shut() && voice.velum() < SEALED_VELUM;
                self.sealed_for = if sealed {
                    self.sealed_for + 1.0 / voice.sample_rate()
                } else {
                    0.0
                };
                self.closure_made |= self.sealed_for >= self.utterance.closure_floor;
            }
            if self.elapsed >= seg.duration && !self.holding_closure(&seg) {
                self.elapsed -= seg.duration;
                self.nominal += seg.duration;
                self.index += 1;
                self.started = false;
                if self.index == self.utterance.segments.len() {
                    voice.set_voiced(false);
                    voice.set_glottal_spread(0.0);
                    voice.set_fold_slack(0.0);
                    voice.set_fold_softness(0.0);
                    voice.set_cavity_expansion(0.0);
                    // The physical tract keeps the shape it ended on (see
                    // `PAUSE_SETTLE`); the next utterance starts from its
                    // own. The lungs stop pushing: an utterance that ends
                    // without a pause (a single word) would otherwise go on
                    // breathing through its last shape, a hiss or a murmur
                    // after it.
                    if physical {
                        voice.set_breath(0.0);
                    } else {
                        voice.articulate_phoneme(Phoneme::Sil);
                    }
                }
            }
        }
        voice.process()
    }

    /// Whether a voiceless stop's closure is still to be held: the mouth
    /// has not yet been sealed (shut, with the velum shut) for
    /// [`Tuning::closure_floor`] in this segment, and the segment has not
    /// run over by more than [`CLOSURE_OVERRUN`]. Its release waits, and so does the next
    /// segment: in a fast voice the articulators sent on to the next sound
    /// as a closure barely formed let the mouth open again after a few
    /// milliseconds, and the stop was not heard at all.
    fn holding_closure(&self, seg: &Segment) -> bool {
        let closure = !seg.voiced
            && matches!(seg.phoneme, Phoneme::P | Phoneme::T | Phoneme::K)
            && seg
                .articulation
                .constriction
                .is_some_and(|c| c.diameter <= 0.3);
        closure
            && !self.closure_made
            && self.sealed_for < self.utterance.closure_floor
            && self.elapsed < seg.duration + CLOSURE_OVERRUN
    }

    /// Time the larynx from what the mouth does (physical engine): the
    /// voice stops the moment a voiceless stop shuts the mouth, and after
    /// its release it comes back the segment's `hold` after the mouth
    /// opens. Timed from the segments alone, both drifted with the
    /// articulators: a slower tongue body or a longer vowel moved the
    /// closure and the release against the segment boundaries, and with
    /// them the voice onset time and the voicing left in a closure.
    fn time_larynx(&mut self, voice: &mut Trombone, seg: &Segment) {
        let shut = voice.mouth_shut();
        let next = self.utterance.segments.get(self.index + 1).copied();
        let closing = !matches!(
            seg.phoneme.manner(),
            Manner::Stop | Manner::Affricate | Manner::Nasal
        );
        if shut
            && !self.was_shut
            && seg.voiced
            && closing
            && !self.larynx_ahead
            && let Some(n) = next.filter(|n| {
                !n.voiced && matches!(n.phoneme.manner(), Manner::Stop | Manner::Affricate)
            })
        {
            // The arytenoids part at once and the folds, still vibrating,
            // die away as they go (see `Tuning::voicing_tail`).
            voice.set_glottal_spread(n.gesture.spread);
            self.tail_until = Some(self.clock + self.utterance.voicing_tail);
            self.larynx_ahead = true;
        }
        if !shut && self.was_shut && self.voice_at.is_none() && !seg.voiced {
            // The aspiration after this release: this segment, or the one
            // after a closure.
            let hold = if seg.gesture.hold > 0.0 {
                seg.gesture.hold
            } else {
                next.filter(|n| n.phoneme == seg.phoneme)
                    .map_or(0.0, |n| n.gesture.hold)
            };
            if hold > 0.0 {
                self.voice_at = Some(self.clock + hold);
            }
        }
        if let Some(t) = self.devoice_at
            && (shut || self.clock >= t)
        {
            self.devoice_at = None;
            // Shut in time: the voice dies away in the closure.
            voice.set_glottal_spread(seg.gesture.spread);
            self.tail_until = Some(if shut {
                self.clock + self.utterance.voicing_tail
            } else {
                self.clock
            });
        }
        if let Some(t) = self.tail_until
            && self.clock >= t
        {
            self.tail_until = None;
            // Still in the sound before the stop, or in the stop itself.
            let stop = if seg.voiced {
                next.unwrap_or(*seg)
            } else {
                *seg
            };
            voice.set_voiced(false);
            if !seg.voiced {
                voice.set_tenseness(seg.tenseness);
            }
            voice.set_glottal_spread(stop.gesture.spread);
        }
        if !shut && self.was_shut {
            self.opened_at = self.clock;
        }
        self.was_shut = shut;
        if let Some(t) = self.voice_at
            && self.clock >= t
        {
            self.voice_at = None;
            if seg.voiced {
                voice.set_voiced(true);
                voice.set_glottal_spread(seg.gesture.spread);
                self.kick_from = self.clock;
            } else if seg.gesture.hold > 0.0
                && let Some(n) = next.filter(|n| n.voiced)
            {
                // Released early: the voice comes during the breath.
                voice.set_tenseness(n.tenseness);
                voice.set_voiced(true);
                voice.set_glottal_spread(n.gesture.spread);
            }
        }
    }

    /// The lower lip coming up to the teeth for an /f v/ after a sibilant.
    /// A speaker's lip rises while the tongue still holds the sibilant, and
    /// the two gestures overlap: the lip narrows the way out in front of
    /// the teeth, so the hiss of /s/ fades under it as its jet is choked
    /// and the lip's own hiss grows, and when the tongue lets go the air
    /// has the lip's gap to pass already. Sent at once (the tongue letting
    /// go, the lip shutting in some 15 ms behind it), the sibilant stopped
    /// within 4 ms, the air rushed out of the open mouth (lip flow from
    /// about 120 to 950 cm³/s) and only then met the lip: a gap, a puff and
    /// a faint hiss, the pattern of /p/ ("makes fine" as "makes pine").
    /// The target is moved from the lip's opening to its gap over
    /// `labial_closing`, starting `labial_lead` before the tongue lets go.
    fn move_lip(&mut self, voice: &mut Trombone, under_sibilant: bool) {
        let Some(l) = self.labial else {
            return;
        };
        if !under_sibilant {
            // The tongue has let go: the lip is the /f v/ now.
            voice.set_constriction(LABIAL_SLOT, None);
            self.labial = None;
            return;
        }
        let since = self.clock - l.since;
        let t = (since / self.utterance.labial_closing.max(1e-6)).min(1.0);
        let eased = t * t * (3.0 - 2.0 * t);
        let lip = l.open + (l.lip.diameter - l.open) * eased;
        voice.set_constriction(
            LABIAL_SLOT,
            Some(crate::Constriction::new(l.lip.index, lip)),
        );
        let s = l.sibilant;
        let t = (since / self.utterance.sibilant_exit.max(1e-6)).min(1.0);
        let d = s.diameter + (SIBILANT_LET_GO - s.diameter).max(0.0) * t;
        voice.set_constriction(SLOT_MAIN, Some(crate::Constriction::new(s.index, d)));
    }

    /// Render a block.
    pub fn render(&mut self, voice: &mut Trombone, out: &mut [f32]) {
        for s in out {
            *s = self.process(voice);
        }
    }
}

/// Send segment `k`'s articulation `a` to the voice, with the speed and
/// the vowel shape a followed utterance gives it (see
/// [`Utterance::follow_with`]); an utterance that follows nothing has the
/// voice's own.
fn articulate_segment(
    voice: &mut Trombone,
    followed: &follow::Followed,
    k: usize,
    a: &Articulation,
) {
    voice.set_articulation_boost(followed.speed_of(k));
    match followed.shape_of(k) {
        Some((key, shape)) => voice.hold_speech_vowel(Some((key, &shape))),
        None => voice.hold_speech_vowel(None),
    }
    voice.articulate(a);
}

/// The shape a segment begins with: a fricative's gap still forming (see
/// [`Gesture::gap_opening`]), anything else its own.
fn entry_shape(seg: &Segment) -> Articulation {
    if seg.gesture.gap_opening.0 > 0.0 {
        widened(&seg.articulation, seg.gesture.gap_widening)
    } else {
        seg.articulation
    }
}

/// `a` with its gap wider by `by` (diameter units).
fn widened(a: &Articulation, by: f32) -> Articulation {
    Articulation {
        constriction: a
            .constriction
            .map(|c| crate::Constriction::new(c.index, c.diameter + by)),
        ..*a
    }
}

// ===== text → tokens ======================================================

mod follow;
mod lexicon;
mod rhythm;
mod rules;

pub use follow::{FollowOptions, FollowReport, PhoneSpan, PitchMode, Schedule, ScheduleError};

/// Convert English text (with optional `[ARPAbet]` spans) to tokens.
///
/// Words are looked up in the built-in dictionary, taken apart into known
/// stems and regular endings (`-s`, `-ed`, `-ing`, `-ly` ...), and spelled
/// out by English spelling rules otherwise.
#[must_use]
pub fn phonemize(text: &str) -> Vec<Token> {
    phonemize_words(text, &lexicon::builtin)
}

/// [`phonemize`] with `lexicon` consulted in place of the built-in
/// dictionary.
///
/// `lexicon` maps a lower-case word (apostrophes kept) to ARPAbet with `1`
/// after the accented vowel, `"K AX N UW1"`; a word with no `1` is spoken
/// unaccented, like a function word. Words it does not know go through the
/// same stems, endings and spelling rules as in [`phonemize`]. Useful for a
/// custom vocabulary, and for checking the spelling rules alone
/// (`|_| None`).
#[must_use]
pub fn phonemize_with<'a>(text: &str, lexicon: &dyn Fn(&str) -> Option<&'a str>) -> Vec<Token> {
    phonemize_words(text, &|w: &str| lexicon(w).map(lexicon::parse_arpabet))
}

fn phonemize_words(text: &str, lex: lexicon::Lexicon) -> Vec<Token> {
    let mut out = Vec::new();
    let mut word = String::new();
    let mut chars = text.chars().peekable();

    // Function words with a weak form, and where their phonemes start.
    let mut weak: Vec<(usize, &'static WeakForm)> = Vec::new();
    let mut before = String::new();
    let mut flush = |word: &mut String, out: &mut Vec<Token>| {
        if !word.is_empty() {
            if let Some(form) = WEAK_FORMS.iter().find(|f| f.word == word.as_str())
                && !(form.noun && DETERMINERS.contains(&before.as_str()))
            {
                weak.push((out.len(), form));
            }
            push_word(word, lex, out);
            before.clone_from(word);
            word.clear();
        }
    };

    while let Some(c) = chars.next() {
        match c {
            '[' => {
                flush(&mut word, &mut out);
                let mut span = String::new();
                for c in chars.by_ref() {
                    if c == ']' {
                        break;
                    }
                    span.push(c);
                }
                push_arpabet(&span, &mut out);
            }
            c if c.is_alphabetic() || c == '\'' || c == '’' => {
                if c == '’' {
                    word.push('\'');
                } else {
                    word.extend(c.to_lowercase());
                }
            }
            c if c.is_ascii_digit() => {
                flush(&mut word, &mut out);
                let mut n = String::from(c);
                while let Some(&d) = chars.peek() {
                    if d.is_ascii_digit() {
                        n.push(d);
                        chars.next();
                    } else {
                        break;
                    }
                }
                for w in number_words(&n).split(' ').filter(|w| !w.is_empty()) {
                    push_word(w, lex, &mut out);
                }
            }
            _ => {
                flush(&mut word, &mut out);
                let brk = match c {
                    ',' | ';' | ':' | '—' | '–' => Some(Break::Comma),
                    '.' | '\n' => Some(Break::Period),
                    '?' => Some(Break::Question),
                    '!' => Some(Break::Exclamation),
                    _ => None,
                };
                if let Some(b) = brk {
                    // Collapse runs like "?!" or "..." into the first.
                    if !matches!(out.last(), Some(Token::Break(_))) {
                        if matches!(out.last(), Some(Token::WordBreak)) {
                            out.pop();
                        }
                        out.push(Token::Break(b));
                    }
                }
            }
        }
    }
    flush(&mut word, &mut out);
    if matches!(out.last(), Some(Token::WordBreak)) {
        out.pop();
    }
    reduce_function_words(&mut out, &weak);
    out
}

/// A function word that is said in a weak form in running speech.
struct WeakForm {
    word: &'static str,
    /// Its phonemes before a consonant.
    weak: &'static [Phoneme],
    /// Its phonemes before a vowel, where they differ.
    before_vowel: Option<&'static [Phoneme]>,
    /// Also a noun ("a can"): full after a determiner.
    noun: bool,
}

/// Weak forms of function words, used when another word follows in the
/// same phrase ("to send", "bread and butter", "what he can do"); at the
/// end of a phrase the word keeps the dictionary's full form ("who is it
/// for?", "yes I can"). A speaker says these with a schwa nearly always;
/// said in full, every one of them was a vowel the recognizer marked wrong.
const WEAK_FORMS: &[WeakForm] = {
    use Phoneme as P;
    const fn form(word: &'static str, weak: &'static [Phoneme]) -> WeakForm {
        WeakForm {
            word,
            weak,
            before_vowel: None,
            noun: false,
        }
    }
    &[
        WeakForm {
            before_vowel: Some(&[P::T, P::Uh]),
            ..form("to", &[P::T, P::Ax])
        },
        WeakForm {
            // "The apple": /ðɪ/ before a vowel, /ðə/ (the dictionary's)
            // before a consonant.
            before_vowel: Some(&[P::Dh, P::Ih]),
            ..form("the", &[P::Dh, P::Ax])
        },
        form("was", &[P::W, P::Ah, P::Z]),
        form("has", &[P::Hh, P::Ax, P::Z]),
        form("had", &[P::Hh, P::Ax, P::D]),
        form("than", &[P::Dh, P::Ax, P::N]),
        form("your", &[P::Y, P::Uh, P::R]),
    ]
};

/// Words after which "can" is the noun.
const DETERMINERS: &[&str] = &[
    "a", "the", "this", "that", "my", "his", "her", "your", "our", "tin",
];

/// Put the weak form of every function word in `weak` (where it starts in
/// `tokens`, and which word) that is followed by another word in its
/// phrase.
fn reduce_function_words(tokens: &mut Vec<Token>, weak: &[(usize, &'static WeakForm)]) {
    // Last first, so that the places of the earlier ones still hold.
    for &(start, form) in weak.iter().rev() {
        let len = tokens[start..]
            .iter()
            .take_while(|t| matches!(t, Token::Phone { .. }))
            .count();
        let next = match tokens.get(start + len..start + len + 2) {
            Some([Token::WordBreak, Token::Phone { phoneme, .. }]) => *phoneme,
            _ => continue,
        };
        let phones = match form.before_vowel {
            Some(v) if is_vowel(next) => v,
            _ => form.weak,
        };
        tokens.splice(
            start..start + len,
            phones.iter().map(|&phoneme| Token::Phone {
                phoneme,
                stressed: false,
            }),
        );
    }
}

/// An `[ARPAbet]` span. A `1` after a vowel marks the accent; with no `1`
/// the first vowel is accented, unless a vowel is marked `0` (`[DH AX0]`),
/// which keeps the word unaccented.
fn push_arpabet(span: &str, out: &mut Vec<Token>) {
    let mut any = false;
    let mut stressed_seen = false;
    let mut unstressed_marked = false;
    for sym in span.split_whitespace() {
        if let Some(p) = Phoneme::from_symbol(sym) {
            let stressed = sym.ends_with('1');
            stressed_seen |= stressed;
            unstressed_marked |= sym.ends_with('0');
            out.push(Token::Phone {
                phoneme: p,
                stressed,
            });
            any = true;
        }
    }
    if any {
        if !stressed_seen && !unstressed_marked {
            stress_first_vowel(out);
        }
        out.push(Token::WordBreak);
    }
}

fn push_word(word: &str, lex: lexicon::Lexicon, out: &mut Vec<Token>) {
    let word = word.trim_matches('\'');
    if word.is_empty() {
        return;
    }
    for (phoneme, stressed) in lexicon::pronounce(word, lex) {
        out.push(Token::Phone { phoneme, stressed });
    }
    if matches!(out.last(), Some(Token::Phone { .. })) {
        out.push(Token::WordBreak);
    }
}

fn stress_first_vowel(tokens: &mut [Token]) {
    // Within the current word: walk back to the last break.
    let start = tokens
        .iter()
        .rposition(|t| !matches!(t, Token::Phone { .. }))
        .map_or(0, |i| i + 1);
    for t in &mut tokens[start..] {
        if let Token::Phone { phoneme, stressed } = t
            && is_vowel(*phoneme)
        {
            *stressed = true;
            return;
        }
    }
}

fn is_vowel(p: Phoneme) -> bool {
    matches!(p.manner(), Manner::Vowel | Manner::Diphthong)
}

fn number_words(digits: &str) -> String {
    const ONES: [&str; 20] = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ];
    const TENS: [&str; 10] = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];

    fn below_thousand(n: usize, out: &mut String) {
        if n >= 100 {
            out.push_str(ONES[n / 100]);
            out.push_str(" hundred ");
        }
        let r = n % 100;
        if r >= 20 {
            out.push_str(TENS[r / 10]);
            out.push(' ');
            if !r.is_multiple_of(10) {
                out.push_str(ONES[r % 10]);
                out.push(' ');
            }
        } else if r > 0 || n == 0 {
            out.push_str(ONES[r]);
            out.push(' ');
        }
    }

    let mut out = String::new();
    match digits.parse::<usize>() {
        Ok(n) if digits.len() <= 9 && !(digits.len() > 1 && digits.starts_with('0')) => {
            let (m, rest) = (n / 1_000_000, n % 1_000_000);
            let (k, r) = (rest / 1000, rest % 1000);
            if m > 0 {
                below_thousand(m, &mut out);
                out.push_str("million ");
            }
            if k > 0 {
                below_thousand(k, &mut out);
                out.push_str("thousand ");
            }
            if r > 0 || n == 0 {
                below_thousand(r, &mut out);
            }
        }
        _ => {
            // Too long, or a code like "007": read digit by digit.
            for d in digits.bytes() {
                out.push_str(ONES[usize::from(d - b'0')]);
                out.push(' ');
            }
        }
    }
    out
}

// ===== tokens → timed segments ============================================

// English vowels have lengths of their own. In the hVd words of
// Hillenbrand et al. (1995) the men's lax /ɪ ɛ ʌ ʊ/ last about 0.81 of
// their mean and the long tense /æ ɑ ɔ ɝ/ about 1.17 (/i u/ about 1). In
// running speech the lax vowels are shorter still against the tense ones
// (a speaker reading LibriSpeech holds his stressed /ɪ ɛ ʌ/ for about 60 %
// of his /i eɪ æ ɔ/): stressed lax vowels take `Tuning::lax_length` of a
// tense vowel's length, unstressed ones `Tuning::unstressed_lax_length`.

/// Length of the long tense vowels /æ ɑ ɔ ɝ/ against the others (physical
/// engine).
const LONG_LENGTH: f32 = 1.085;
/// How much longer /aɪ aʊ ɔɪ/ are than /eɪ oʊ/ (Peterson & Lehiste 1960),
/// against `Tuning::diphthong_length` (physical engine).
const WIDE_DIPHTHONG_LENGTH: f32 = 1.05;

/// Share of a diphthong spent on the vowel it starts from, before it
/// glides to its end.
const NUCLEUS: f32 = 0.55;

/// The lax vowels /ɪ ɛ ʌ ʊ/: short, and gliding towards schwa as they end
/// (physical engine).
fn is_lax(p: Phoneme) -> bool {
    matches!(p, Phoneme::Ih | Phoneme::Eh | Phoneme::Ah | Phoneme::Uh)
}

/// Whether an /u/ in `context` is made further front (physical engine; see
/// [`VowelKey::Fronted`] and [`Tuning::fronted_nucleus`]): after /t d s z/
/// in its own word ("two", "do", "tune", "soon", "zoo"), and not before
/// the dark /l/ of its own syllable ("tool", "school"), which draws the
/// tongue back again.
///
/// Measured in the LibriSpeech reader's sentences (aligned, LPC; F2 at a
/// fifth, half and four fifths of the vowel): after /t d n/ his /u/ starts
/// at 1700 to 2200 Hz and ends at 1250 to 1650 ("two" 1690, 1360, 1300;
/// "do" 1700, 1570, 1810; "the new" 1880, 2220, 2240; "too" 1760, 1630,
/// 1340; "do in" 1790, 1390, 1180), with F1 250 to 310; after /j/ 1840,
/// 1770, 1650; after his /l/ ("lucidity", unstressed) 1300; he has no /u/
/// before /l/. The owner's "canoe" holds 1860 to 1940 Hz for its first 60
/// ms and glides to 1330 by its end; the owner's "smooth", "glue" and
/// "who'd" are back /u/s (900 to 1300 Hz). The engine's /u/ was a back one
/// everywhere, F2 1000 to 1100 with F1 390 to 440.
///
/// After /n/, the postalveolars and /j/ the engine's fronted /u/ was heard
/// as /ju/ whatever its shape: "canoe" as "can use", "the juice" as "the
/// use", "the new king" as "the n-ee-oo king" (F2 1600 to 1800 gliding
/// back: the recognizer heard /i/ inserted before 3 of 20 /u/ after
/// coronals; held at 1400 to 1800, it heard 3 to 6 of them as /i/). So the
/// /u/ stays back there. After /t d s z/ alone the recognizer heard 13 of
/// those 20 /u/ right (12 before), none with an /i/ before it.
fn fronts_u(context: Context) -> bool {
    use Phoneme as P;
    let after_front = context
        .prev
        .is_some_and(|p| matches!(p, P::T | P::D | P::S | P::Z));
    let dark_l = context.next.is_some_and(|(n, _, same)| same && n == P::L);
    after_front && !dark_l
}

/// The vowel after an alveolar tap takes the time the tap saves against a
/// stop's closure (physical engine): the tongue body, heavy and slow, glides
/// from the vowel before the tap to the one after it underneath the tip, and
/// in the 35 ms of a tap it had not got there: the recognizer heard the
/// /ɪ/ of "fled in" as /æ/ and the /ɛ/ of "not exactly" as /ʌ/ (both right
/// again with the time given back; the stop's own length, `stop_length`, is
/// what the rules gave the body before).
fn give_taps_time(drafts: &mut [Draft], timing: &Tuning, rate: f32) {
    for k in 0..drafts.len().saturating_sub(1) {
        let tap = drafts[k];
        if tap.gesture.approach > 0.0 && drafts[k + 1].vowel {
            drafts[k + 1].duration += (timing.stop_length / rate - tap.duration).max(0.0);
        }
    }
}

/// Whether the /t d/ `p` in `context` is said as an alveolar tap [ɾ]
/// (physical engine; [`Tuning::tap_length`]), as American English says
/// it between vowels when the second is unstressed: at the end of a word
/// before an unstressed vowel that starts the next one in the phrase
/// ("slid on", "get it", "a lot of", "that it"), and inside a word after a
/// stressed vowel ("water", "little", "city", "middle"). Not before a
/// stressed vowel ("return", "attack", "a toe"), not after a consonant
/// ("winter", "quilter"), never at the end of a phrase, and not /t/ before
/// a syllabic /n/ ("button", "Latin"), which English closes in the glottis
/// instead.
///
/// Measured in the LibriSpeech reader's sentences (aligned; level in 25
/// ms frames against the louder vowel beside it; voiced share of 10 ms
/// frames): his taps last 37 to 66 ms as aligned, dip 5 to 13 dB and are
/// voiced throughout ("that it" 45 ms, 5 dB; "fled in" 50 ms, 9 dB;
/// "British" 66 ms, 9 dB; "middle" 37 ms, 7 dB; "steadily" 45 ms, 8 dB);
/// his /t/ after a consonant ("quilter", "mantel", "painting") or before a
/// stressed vowel ("returned") is a stop, 55 to 140 ms, 18 to 31 dB down
/// and mostly voiceless. The engine's /t/ between vowels was 98 ms, 45 to
/// 57 dB down and voiceless, its /d/ 60 ms, 18 to 47 dB down.
fn taps(p: Phoneme, context: Context) -> bool {
    use Phoneme as P;
    if !matches!(p, P::T | P::D) || !context.prev.is_some_and(is_vowel) {
        return false;
    }
    let Some((next, next_stressed, same_word)) = context.next else {
        return false;
    };
    if !is_vowel(next) || next_stressed {
        return false;
    }
    if !same_word {
        return true;
    }
    let syllabic_n = matches!(next, P::Ax | P::Ih)
        && context
            .after_next
            .is_some_and(|(q, _, same)| same && q == P::N);
    context.prev_stressed && !(p == P::T && syllabic_n)
}

/// How long vowel or diphthong `p` lasts against others of its kind, on
/// the physical engine (pass C of the tuning): every vowel is held longer
/// than on the classic engine (`Tuning::vowel_hold`), lax vowels are
/// shorter than tense ones, long tense ones longer, /aɪ aʊ ɔɪ/ longer than
/// /eɪ oʊ/.
fn own_length(p: Phoneme, stressed: bool, timing: &Tuning) -> f32 {
    use Phoneme as P;
    match p {
        Phoneme::Ih if stressed => timing.ih_length * timing.vowel_hold,
        _ if is_lax(p) && stressed => timing.lax_length * timing.vowel_hold,
        _ if is_lax(p) => timing.unstressed_lax_length * timing.vowel_hold,
        P::Ae | P::Aa | P::Ao | P::Er => LONG_LENGTH * timing.vowel_hold,
        P::Ay | P::Aw | P::Oy => WIDE_DIPHTHONG_LENGTH * timing.vowel_hold,
        _ => timing.vowel_hold,
    }
}

/// Base durations in seconds at rate 1 (`physical`: for the physical
/// engine, which gives each vowel and /z ʒ/ a length of its own).
fn base_duration(p: Phoneme, stressed: bool, timing: &Tuning, physical: bool) -> f32 {
    let own = |d: f32| {
        if physical {
            d * own_length(p, stressed, timing)
        } else {
            d
        }
    };
    let d = match p.manner() {
        Manner::Vowel => own(timing.vowel_length),
        Manner::Diphthong => own(timing.diphthong_length),
        Manner::Stop | Manner::Affricate => timing.stop_length,
        Manner::Nasal => timing.nasal_length,
        Manner::Fricative => {
            if matches!(p, Phoneme::Z | Phoneme::Zh) && physical {
                timing.voiced_sibilant_length
            } else if p.is_voiced() {
                timing.voiced_fricative_length
            } else {
                timing.fricative_length
            }
        }
        Manner::Approximant => timing.approximant_length,
        Manner::Aspirate => timing.aspirate_length,
        Manner::Pause => 0.1,
    };
    if stressed && is_vowel(p) {
        d * timing.stress_stretch
    } else {
        d
    }
}

/// Which part of a sound a draft is.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Part {
    /// A whole sound, or the first part of a vowel.
    Head,
    /// A diphthong's glide to its end.
    Glide,
    /// A lax vowel's glide towards schwa (physical engine): part of the
    /// vowel, not a sound after it, for the phrase's pitch and breath.
    Offglide,
}

/// The first draft of the vowel whose last draft is `k` (a lax vowel's
/// offglide belongs to the vowel before it).
fn vowel_head(drafts: &[Draft], k: usize) -> usize {
    if k > 0 && drafts[k].part == Part::Offglide {
        k - 1
    } else {
        k
    }
}

#[derive(Clone, Copy)]
#[allow(clippy::struct_excessive_bools)] // independent flags
struct Draft {
    phoneme: Phoneme,
    articulation: Articulation,
    duration: f32,
    voiced: bool,
    tenseness: Option<f32>,
    /// Relative pitch target at the end of the segment (1.0 = baseline).
    accent: f32,
    /// The same at its middle, for a vowel the physical engine's
    /// intonation has passed (`0`: none, the pitch goes straight from the
    /// last segment's to this one's).
    mid_accent: f32,
    vowel: bool,
    part: Part,
    /// Laryngeal and wall gesture (physical engine).
    gesture: Gesture,
    /// The breath after a voiceless stop's release: takes the shape of
    /// the sound that follows (physical engine).
    aspiration: bool,
    /// A /b p g k/ released into an /l r/ of its own word (physical
    /// engine; see `release_into_liquids`).
    into_liquid: bool,
    /// Which phone of the text this draft realises ([`NO_PHONE`]: a pause).
    phone: u32,
}

/// Where a phone stands, for the physical engine's voicing rules.
#[derive(Clone, Copy, Debug, Default)]
struct Context {
    /// The phone before, if it is in the same word.
    prev: Option<Phoneme>,
    /// Whether that phone is stressed.
    prev_stressed: bool,
    /// The phone after: the sound, whether it is stressed, whether it is in
    /// the same word.
    next: Option<(Phoneme, bool, bool)>,
    /// The one after that, the same way (in the same word as `next`).
    after_next: Option<(Phoneme, bool, bool)>,
}

impl Context {
    /// The context of the phone at `tokens[i]`. A phrase break ends it.
    fn at(tokens: &[Token], i: usize) -> Self {
        let (prev, prev_stressed) = match i.checked_sub(1).map(|k| tokens[k]) {
            Some(Token::Phone { phoneme, stressed }) => (Some(phoneme), stressed),
            _ => (None, false),
        };
        let mut found = [None; 2];
        let mut same_word = true;
        let mut n = 0;
        for t in &tokens[i + 1..] {
            match *t {
                Token::Phone { phoneme, stressed } => {
                    found[n] = Some((phoneme, stressed, same_word));
                    n += 1;
                    if n == 2 {
                        break;
                    }
                    same_word = true;
                }
                Token::WordBreak => same_word = false,
                Token::Break(_) => break,
            }
        }
        Self {
            prev,
            prev_stressed,
            next: found[0],
            after_next: found[1],
        }
    }
}

/// /s z ʃ ʒ/.
fn is_sibilant(p: Phoneme) -> bool {
    matches!(p, Phoneme::S | Phoneme::Z | Phoneme::Sh | Phoneme::Zh)
}

fn is_obstruent(p: Phoneme) -> bool {
    matches!(
        p.manner(),
        Manner::Stop | Manner::Fricative | Manner::Affricate
    )
}

#[allow(clippy::too_many_lines)]
fn compile(tokens: &[Token], prosody: &Prosody) -> Utterance {
    let rate = prosody.rate.clamp(0.2, 5.0);
    let expr = prosody.expressiveness.clamp(0.0, 2.0);
    let tense = prosody.tenseness.clamp(0.0, 1.0);
    let nasality = prosody.nasality.clamp(0.0, 1.0);
    let timing = prosody.tuning.clamped();
    let physical = prosody.engine == Engine::Physical;

    // Split into phrases at breaks; each phrase gets its own contour.
    let mut drafts: Vec<Draft> = Vec::new();
    let mut phrase_start = 0;
    let mut i = 0;
    // Each phrase's drafts and how it ends, for the breath (physical).
    let mut phrases: Vec<(usize, usize, Break)> = Vec::new();
    // The place of each phone among the text's, and where it stands.
    let mut phone: u32 = 0;
    let roles = rhythm::roles(tokens);

    // The physical engine shapes its phrases once they have their final
    // lengths (see `intonation`).
    let push_phrase_end = |drafts: &mut Vec<Draft>, start: usize, brk: Option<Break>| {
        if !physical {
            shape_phrase(&mut drafts[start..], brk, expr);
        }
    };

    while i < tokens.len() {
        match tokens[i] {
            Token::Phone { phoneme, stressed } => {
                let next_vowel_or_approx = tokens[i + 1..]
                    .iter()
                    .find_map(|t| match t {
                        Token::Phone { phoneme, .. } => Some(*phoneme),
                        _ => None,
                    })
                    .is_some_and(|p| is_vowel(p) || p.manner() == Manner::Approximant);
                let context = physical.then(|| Context::at(tokens, i));
                let first = drafts.len();
                let role = roles.get(phone as usize).copied().unwrap_or_default();
                expand(
                    phoneme,
                    (stressed, rhythm::stretch(phoneme, role, &timing)),
                    next_vowel_or_approx,
                    context,
                    rate,
                    &timing,
                    &mut drafts,
                );
                for d in &mut drafts[first..] {
                    d.phone = phone;
                }
                phone += 1;
            }
            Token::WordBreak => {}
            Token::Break(b) => {
                push_phrase_end(&mut drafts, phrase_start, Some(b));
                phrases.push((phrase_start, drafts.len(), b));
                let pause = match b {
                    Break::Comma => 0.18,
                    _ if physical => timing.sentence_pause,
                    _ => 0.38,
                } / rate;
                drafts.push(silence(pause));
                phrase_start = drafts.len();
            }
        }
        i += 1;
    }
    push_phrase_end(&mut drafts, phrase_start, Some(Break::Period));
    phrases.push((phrase_start, drafts.len(), Break::Period));
    if physical && !drafts.is_empty() {
        // Speakers set their tongue before they speak: a moment of
        // silence in which the slow tongue body gets to the first vowel
        // (it takes that body's shape below, as every pause takes the
        // next vowel's), so the first word does not start from rest.
        drafts.insert(0, silence(PRE_POSITION / rate));
        for p in &mut phrases {
            (p.0, p.1) = (p.0 + 1, p.1 + 1);
        }
    }

    if physical {
        give_taps_time(&mut drafts, &timing, rate);
    }

    // Lengthen the last syllable of each phrase.
    for k in 1..drafts.len() {
        if drafts[k].phoneme == Phoneme::Sil
            && let Some(v) = drafts[..k].iter().rposition(|d| d.vowel)
        {
            drafts[v].duration *= 1.4;
            // On the physical engine the whole vowel, not only its end.
            let mut head = v;
            while physical && head > 0 && drafts[head].part != Part::Head {
                head -= 1;
                drafts[head].duration *= 1.4;
            }
            // And the consonants that close it (see `Tuning::final_coda_stretch`).
            if physical && drafts[k - 1].phoneme != Phoneme::Sil {
                for d in &mut drafts[v + 1..k] {
                    d.duration *= timing.final_coda_stretch;
                }
            }
        }
    }

    if physical {
        darken_before_coda_l(&mut drafts, &mut phrases, &timing);
    }

    // Coarticulation: consonants borrow the tongue of the next vowel (or
    // the previous one at the end of an utterance).
    let mut next_tongue = None;
    for d in drafts.iter_mut().rev() {
        if d.vowel {
            next_tongue = d.articulation.tongue;
        } else if d.articulation.tongue.is_none() {
            d.articulation.tongue = next_tongue;
        }
    }
    // On the physical engine only a vowel's: after the /k/ of "planks" an
    // utterance-final /s/ took the /k/'s own tongue, and with it the /k/'s
    // place (a consonant on the velar body is made at the velum, see
    // `phoneme::velar_place`), a faint velar hiss 56 dB under the vowel in
    // place of the /s/ (mid-phrase, and before a comma, it takes the next
    // vowel's tongue above and hisses).
    let mut prev_tongue = Phoneme::Ax.articulation().tongue;
    for d in &mut drafts {
        match d.articulation.tongue {
            Some(t) if !physical || d.vowel || d.phoneme == Phoneme::Sil => prev_tongue = Some(t),
            Some(_) => {}
            None => d.articulation.tongue = prev_tongue,
        }
    }
    if physical {
        let light = function_words(tokens);
        for &(a, b, _) in &phrases {
            relax_the_folds(&mut drafts[a..b], tense, &timing);
            shade_syllables(&mut drafts[a..b], &light, tense, &timing);
        }
        // Which vowels carry an accent, before the intonation turns the
        // stress marks into pitch.
        let accented: Vec<bool> = drafts.iter().map(|d| d.vowel && d.accent > 0.0).collect();
        for &(a, b, brk) in &phrases {
            intonation(&mut drafts[a..b], brk, expr, &timing);
        }
        breathe(&mut drafts, &mut phrases, &accented, &timing);
        lean_on_vowels(&mut drafts);
        assimilate_place(&mut drafts);
        voicing_gestures(&mut drafts, prosody.pitch, &timing);
        release_into_liquids(&mut drafts, &timing, rate);
        lateral_onto_vowels(&mut drafts, &timing, rate);
    }

    // Pitch: start each segment where the last ended.
    let mut segments = Vec::with_capacity(drafts.len() + 1);
    let phones = drafts.iter().map(|d| d.phone).collect();
    let mut last = prosody.pitch;
    for mut d in drafts {
        if d.vowel || d.phoneme.manner() == Manner::Approximant {
            d.articulation.velum = d.articulation.velum.max(nasality);
        }
        let target = prosody.pitch * d.accent;
        let mut gesture = d.gesture;
        if d.vowel && d.mid_accent > 0.0 {
            // How far the intonation's curve stands off the straight line
            // from the last segment's pitch to this one's at the middle.
            let mid = prosody.pitch * d.mid_accent;
            gesture.pitch_arc =
                12.0 * crate::math::ln(2.0 * mid / (last + target)) / core::f32::consts::LN_2;
        }
        segments.push(Segment {
            phoneme: d.phoneme,
            articulation: d.articulation,
            duration: d.duration,
            pitch: (last, target),
            tenseness: d.tenseness.unwrap_or(tense),
            voiced: d.voiced,
            gesture,
        });
        last = target;
    }
    // Trim a trailing pause to a short release.
    if let Some(s) = segments.last_mut()
        && s.phoneme == Phoneme::Sil
    {
        s.duration = 0.05;
    }
    Utterance {
        segments,
        lead: timing.lead,
        release_lead: timing.release_lead,
        kick_time: timing.kick_time,
        devoicing_lead: timing.devoicing_lead,
        velum_lead: timing.velum_lead,
        closure_floor: if physical { timing.closure_floor } else { 0.0 },
        voicing_tail: timing.voicing_tail,
        labial_lead: if physical { timing.labial_lead } else { 0.0 },
        labial_closing: if physical { timing.labial_closing } else { 0.0 },
        sibilant_exit: if physical { timing.sibilant_exit } else { 0.0 },
        phones,
        roles,
        pitch_track: Vec::new(),
        pitch_hop: 0.0,
        followed: follow::Followed::default(),
    }
}

/// The physical engine's place assimilation: /t d/ right before /θ ð/
/// are made where the dental is, with the tip against the teeth ("get the"
/// is [t̪ ð]), so the tongue goes straight on into the fricative's gap
/// instead of letting a closure behind the ridge burst into a gap at the
/// teeth. Not /n/: a speaker's [n̪] in "in the" touches from the ridge to
/// the teeth, and the mouth behind it, which gives a nasal its
/// antiresonance, is an /n/'s. Closed at the teeth alone, this tract's
/// mouth behind the closure grew by a centimetre, its antiresonance fell
/// to /m/'s (840 Hz), and "on the" was heard as "om the".
fn assimilate_place(drafts: &mut [Draft]) {
    for k in 0..drafts.len().saturating_sub(1) {
        let next = drafts[k + 1];
        if !matches!(drafts[k].phoneme, Phoneme::T | Phoneme::D)
            || !matches!(next.phoneme, Phoneme::Th | Phoneme::Dh)
        {
            continue;
        }
        let (Some(c), Some(dental)) = (
            drafts[k].articulation.constriction,
            next.articulation.constriction,
        ) else {
            continue;
        };
        // The closure and, for a stop, its release segment.
        let p = drafts[k].phoneme;
        for d in drafts[..=k].iter_mut().rev().take_while(|d| d.phoneme == p) {
            if let Some(own) = &mut d.articulation.constriction
                && own.diameter <= c.diameter
            {
                own.index = dental.index;
            }
        }
    }
}

/// How firmly the folds are held, and how soft the voice is, through a
/// phrase (physical engine). A speaker's voice is brightest on the accented
/// vowels: through nasals and liquids, and through unaccented vowels, its
/// spectrum falls faster above 2 kHz, and it softens again over the last
/// 150 ms before a pause. Held at the accented vowel's brightness
/// throughout, every voiced sound had the same bright buzz.
///
/// Measured in the LibriSpeech reader's 16 sentences against his stressed
/// vowels (`tools/judge` alignment; level in 2-4 kHz against 0-1 kHz):
/// nasals 5.6 dB darker, liquids 5.1, unaccented vowels 0.6, glides 4.4;
/// the engine's, with the folds alike throughout, 3.8, 2.1, -0.5 and 11.6.
/// So nasals, liquids and unaccented vowels are softened
/// (`sonorant_softness`, `unstressed_softness`, `final_softness`, through
/// `Trombone::set_fold_softness`, which leaves the folds alone), and the
/// glides, already darker than his under their narrow lips, are not.
///
/// Until round 8 the folds were held more loosely instead
/// (`sonorant_slack`, `final_slack`: a share of the tenseness off), which
/// opens them: past 12 % the child voice fell out of its folds' range and
/// a /g/ lost its voice bar, and at 12 % it made the nasals and liquids
/// brighter, not darker (their breath noise grew).
fn relax_the_folds(phrase: &mut [Draft], tense: f32, timing: &Tuning) {
    /// The stretch before a pause over which the voice softens, seconds.
    const FINAL: f32 = 0.15;
    for d in phrase.iter_mut() {
        let sonorant = matches!(d.phoneme.manner(), Manner::Nasal | Manner::Approximant);
        let liquid = matches!(d.phoneme, Phoneme::L | Phoneme::R);
        let unaccented = d.vowel && d.accent <= 0.0;
        if !d.voiced {
            continue;
        }
        if d.tenseness.is_none() && (sonorant || unaccented) && timing.sonorant_slack > 0.0 {
            d.tenseness = Some(tense * (1.0 - timing.sonorant_slack));
        }
        if d.phoneme.manner() == Manner::Nasal || liquid {
            d.gesture.softness = d.gesture.softness.max(timing.sonorant_softness);
        } else if unaccented {
            d.gesture.softness = d.gesture.softness.max(timing.unstressed_softness);
        }
    }
    let mut left = FINAL;
    for d in phrase.iter_mut().rev() {
        if d.phoneme == Phoneme::Sil {
            continue;
        }
        if left <= 0.0 {
            break;
        }
        if d.voiced {
            // Slackest and softest at the very end, as before 150 ms
            // before it.
            let share = (left - 0.5 * d.duration.min(left)).max(0.0) / FINAL;
            if timing.final_slack > 0.0 {
                let now = d.tenseness.unwrap_or(tense);
                d.tenseness = Some(now * (1.0 - timing.final_slack * share));
            }
            d.gesture.softness = d.gesture.softness.max(timing.final_softness * share);
        }
        left -= d.duration;
    }
}

/// For each phone of `tokens`, whether its word has no stressed vowel of
/// its own: a function word ("the", "of", "to", a weak form).
fn function_words(tokens: &[Token]) -> Vec<bool> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut stressed = false;
    for t in tokens {
        match *t {
            Token::Phone { stressed: s, .. } => {
                out.push(false);
                stressed |= s;
            }
            Token::WordBreak | Token::Break(_) => {
                out[start..].fill(!stressed);
                start = out.len();
                stressed = false;
            }
        }
    }
    out[start..].fill(!stressed);
    out
}

/// The voice quality of each syllable (physical engine): the folds are held
/// more loosely through the vowel of a function word, by
/// [`Tuning::light_laxness`] of the tenseness, within the range a followed
/// schedule's tenseness keeps to ([`Tuning::follow_tenseness_low`]). The
/// consonants keep theirs: a voiced closure needs folds held firmly enough
/// to vibrate against the pressure in the mouth.
///
/// Measured in the LibriSpeech reader's 16 sentences (aligned; H1*-H2* of
/// each vowel against the mean of its sentence's sonorants, fitted over
/// stress, function word, place in the phrase and vowel height, each half
/// of the set alone): his function words' vowels are 1.3 to 1.7 dB
/// breathier than the rest, where the rules' were as breathy as the rest.
/// His last two syllables are also 0.6 to 1.6 dB less breathy than the
/// rest, the rules' 0.8 to 1.3 dB more, the breath dying away under them;
/// folds held 12 % more firmly there made the voice start 43 Hz low after
/// the /p/ of "a pa" (`the_voice_starts_higher_after_a_voiceless_stop`),
/// so that is left as it is. These explain little of his contour (held out,
/// 2 to 8 % of its variance); the rest moves from phone to phone with
/// nothing the text says.
fn shade_syllables(phrase: &mut [Draft], light: &[bool], tense: f32, timing: &Tuning) {
    let lo = timing.follow_tenseness_low;
    for k in 0..phrase.len() {
        let function_vowel = phrase[k].vowel
            && phrase[k].part == Part::Head
            && light
                .get(phrase[k].phone as usize)
                .copied()
                .unwrap_or(false);
        if !function_vowel {
            continue;
        }
        let share = (1.0 - timing.light_laxness).clamp(lo.min(1.0), 1.0);
        if (share - 1.0).abs() < 1e-6 {
            continue;
        }
        let end = k
            + 1
            + phrase[k + 1..]
                .iter()
                .take_while(|d| d.vowel && d.part != Part::Head)
                .count();
        for d in &mut phrase[k..end] {
            if d.voiced {
                d.tenseness = Some((d.tenseness.unwrap_or(tense) * share).clamp(0.0, 1.0));
            }
        }
    }
}

/// The physical engine makes consonants on the tongue body of the vowel
/// they lead into (of the one before, at the end of an utterance, and for
/// the stops, nasals and liquids that end a phrase): each
/// consonant is told which vowel shape that is, so that its tongue tip, lips
/// or tongue back act on that vowel's body instead of moving the whole
/// tongue to a body of their own (see `Articulation::leaning_on`).
fn lean_on_vowels(drafts: &mut [Draft]) {
    let leans = |d: &Draft| !d.vowel && d.phoneme != Phoneme::Sil && d.articulation.vowel.is_none();
    let hisses = |d: &Draft| {
        d.articulation
            .constriction
            .is_some_and(|c| c.diameter > 0.3 && c.diameter <= crate::tract::HISS_WIDEST)
    };
    let mut next = None;
    let mut across = false;
    for d in drafts.iter_mut().rev() {
        if d.vowel {
            next = d.articulation.vowel;
            across = false;
        } else if d.phoneme == Phoneme::Sil {
            across = true;
        } else if leans(d) && (!across || hisses(d)) {
            // Across a pause only a fricative's gap is made on the next
            // phrase's first vowel (on the last vowel's body the final /z/
            // of "shells;" did not hiss as the breath ran out); the stops,
            // nasals and liquids that end a phrase are made on its last
            // vowel. Made on the next one, the /l/ of "shells; who" took
            // the lips of its /u/, and the /z/ after it was unrounding as
            // it should have hissed.
            d.articulation.vowel = next;
        }
    }
    let mut prev = None;
    for d in drafts.iter_mut() {
        if d.vowel {
            prev = d.articulation.vowel;
        } else if leans(d) {
            d.articulation.vowel = prev;
        }
    }
}

/// Length of the frication of /tʃ/, as a share of a voiceless
/// fricative's (physical engine).
const AFFRICATE_FRICATION: f32 = 0.8;

/// How long after the release of a stop that follows /s/ (/sp st sk/) the
/// voice comes back, seconds (physical engine), less the folds' own few
/// milliseconds: a speaker's glottis opens once for the cluster, widest in
/// the /s/, and is closing again by the release, so the voice onset time is
/// short (10 to 20 ms). Waiting for the vowel to begin instead, the voice
/// came a release lead late through a wide-open glottis, and the release
/// was a loud breathy burst.
const AFTER_S_ONSET: f32 = 0.004;

/// The dental closure an /ð/ starts with after a pause, seconds (physical
/// engine).
const DENTAL_ONSET: f32 = 0.025;

/// The silence an utterance starts with on the physical engine, seconds:
/// long enough for the tongue body (a 40 ms time constant) to get most of
/// the way to the first vowel.
const PRE_POSITION: f32 = 0.1;

/// Pitch, Hz, at which a [`Tuning::onset_kick`] is as given; a voice
/// pitched elsewhere gets it in proportion.
const KICK_PITCH: f32 = 110.0;

/// The physical engine's finishing touches on voicing, once every segment
/// is in place: the breath after a voiceless release takes the shape of the
/// sound it leads into, so it is that sound's formants the [h]-like noise
/// excites; and the voice starts higher after a voiceless consonant.
fn voicing_gestures(drafts: &mut [Draft], pitch: f32, timing: &Tuning) {
    for k in 0..drafts.len().saturating_sub(1) {
        if drafts[k].aspiration {
            let next = drafts[k + 1].articulation;
            drafts[k].articulation = Articulation {
                voiced: false,
                tenseness: None,
                ..next
            };
        }
    }
    for k in 1..drafts.len() {
        let before = &drafts[k - 1];
        let after_voiceless = !before.voiced && is_obstruent(before.phoneme);
        let sonorant = drafts[k].vowel || drafts[k].phoneme.manner() == Manner::Approximant;
        if after_voiceless && sonorant && drafts[k].voiced {
            drafts[k].gesture.kick = timing.onset_kick * pitch / KICK_PITCH;
        }
    }
}

/// Whether stop `p` is released into an /l r/ of its own word: /bl pl gl
/// kl br pr gr kr/ (physical engine). Not /tr dr/: the tip makes both.
fn into_liquid(p: Phoneme, context: Context) -> bool {
    matches!(p, Phoneme::B | Phoneme::P | Phoneme::G | Phoneme::K)
        && context
            .next
            .is_some_and(|(n, _, same)| same && matches!(n, Phoneme::L | Phoneme::R))
}

/// How far the tongue tip of an /l r/ has got while it is held back for
/// the release of the stop before it (touch diameter; see
/// `release_into_liquids`): on its way, not yet the liquid's narrowing.
/// Taken away altogether, the tract opened to an F1 of 650 Hz between the
/// stop and the liquid, and "blew" was heard as "b-ah-lew".
const HELD_TIP: f32 = 1.0;

/// The physical engine's release of a stop into a liquid ("blue", "play",
/// "glue", "cream"). A speaker's tongue is often in place for the liquid
/// during the closure, but what a listener hears first is the stop: its
/// burst and a stretch of its own formant transition, the lips or the back
/// of the tongue letting go, before the liquid takes over (voiceless stops
/// leave the start of the liquid voiceless). Sent with the release, the
/// liquid's tip arrived within one articulator block (10 ms) of the lips
/// parting, and of the stop's transition there was next to nothing left.
/// (Most of the burst "blue" lacked against "background", some 25 dB above
/// 2.5 kHz, was lips that parted at the tongue's speed; see
/// `Tract::compute_shaped_targets`.)
///
/// So the stop is made on the liquid's tongue body (a labial leaves the
/// tongue free), and its release goes into that body with the liquid's tip
/// still on its way ([`HELD_TIP`]; /r/'s lips stay rounded). The Speaker sends the liquid itself
/// [`Tuning::liquid_transition`] after the mouth actually opens
/// ([`Gesture::transition`]): after a voiced stop in a release segment of
/// its own, taken out of the liquid's time; after a voiceless one in its
/// breath, the rest of which is the liquid made voiceless.
fn release_into_liquids(drafts: &mut Vec<Draft>, timing: &Tuning, rate: f32) {
    if timing.liquid_transition <= 0.0 {
        return;
    }
    let mut k = 0;
    while k + 1 < drafts.len() {
        if !drafts[k].into_liquid {
            k += 1;
            continue;
        }
        let p = drafts[k].phoneme;
        // The stop's last draft: its closure, or the breath after it.
        let last = if drafts[k + 1].aspiration && drafts[k + 1].phoneme == p {
            k + 1
        } else {
            k
        };
        let Some(liquid) = drafts
            .get(last + 1)
            .copied()
            .filter(|d| matches!(d.phoneme, Phoneme::L | Phoneme::R))
        else {
            k += 1;
            continue;
        };
        // The liquid's own body (/r/'s that of /ɚ/), not the vowel's after it.
        let held = Articulation {
            constriction: liquid
                .articulation
                .constriction
                .map(|c| crate::Constriction::new(c.index, HELD_TIP.max(c.diameter))),
            vowel: None,
            ..liquid.articulation
        };
        if matches!(p, Phoneme::B | Phoneme::P) {
            let closure = &mut drafts[k].articulation;
            closure.tongue = held.tongue;
            closure.vowel = None;
        }
        let window = timing.liquid_transition / rate;
        if last > k {
            let breath = &mut drafts[last];
            breath.articulation = Articulation {
                voiced: false,
                tenseness: None,
                ..held
            };
            breath.gesture.transition = window.min(breath.duration);
        } else {
            let d = window.min(0.5 * liquid.duration);
            let share = d / liquid.duration;
            let (from, to) = liquid.gesture.pressure_drop;
            let between = from + (to - from) * share;
            let before = drafts[k].accent;
            let release = Draft {
                phoneme: p,
                articulation: held,
                duration: d,
                voiced: true,
                accent: before * crate::math::powf(liquid.accent / before, share),
                mid_accent: 0.0,
                vowel: false,
                part: Part::Head,
                gesture: Gesture {
                    pressure_drop: (from, between),
                    transition: d,
                    ..liquid.gesture
                },
                aspiration: false,
                into_liquid: false,
                phone: drafts[k].phone,
                ..liquid
            };
            let after = &mut drafts[last + 1];
            after.duration -= d;
            after.gesture.pressure_drop = (between, to);
            drafts.insert(last + 1, release);
        }
        k = last + 1;
    }
}

/// The physical engine's release of an /l/ into the vowel after it ("slid",
/// "lid", "blue"). The /l/ holds a body of its own, low and back, under its
/// tip; the tip is light and lets go within some 20 ms, the body is heavy.
/// Released together, the tip had gone while the front of the tongue was
/// still down for the /l/: the mouth stood open over a low tongue, the
/// first formant rose 100 to 150 Hz above both the /l/'s and the vowel's
/// and fell again (in "slid" 470, then 585, then 455 Hz), and with the
/// second formant rising that is the pattern of /aɪ/: "slid" was heard as
/// "slide". A speaker's first formant runs flat from the /l/ into a close
/// vowel (the LibriSpeech reader's "sleep": 311 to 328 Hz throughout) or
/// rises some 60 Hz above both at most ("exactly").
///
/// So the end of an /l/ before a vowel, [`Tuning::lateral_release`] of it,
/// is made on the vowel's body: the tongue body sets off for the vowel
/// while the tip still holds the /l/, and the tip lets go as the body
/// arrives.
fn lateral_onto_vowels(drafts: &mut Vec<Draft>, timing: &Tuning, rate: f32) {
    if timing.lateral_release <= 0.0 {
        return;
    }
    let mut k = 0;
    while k + 1 < drafts.len() {
        let (l, next) = (drafts[k], drafts[k + 1]);
        if l.phoneme != Phoneme::L || !next.vowel || l.articulation.vowel.is_none() {
            k += 1;
            continue;
        }
        let d = (timing.lateral_release / rate).min(0.6 * l.duration);
        let share = d / l.duration;
        let (from, to) = l.gesture.pressure_drop;
        let between = from + (to - from) * (1.0 - share);
        let before = if k > 0 {
            drafts[k - 1].accent
        } else {
            l.accent
        };
        let accent = if before > 0.0 && l.accent > 0.0 {
            before * crate::math::powf(l.accent / before, 1.0 - share)
        } else {
            l.accent
        };
        // The /l/'s tip on the vowel's tongue body and lips.
        let on_vowel = Draft {
            articulation: Articulation {
                tongue: next.articulation.tongue,
                lips: None,
                ..l.articulation
            },
            duration: d,
            gesture: Gesture {
                pressure_drop: (between, to),
                ..l.gesture
            },
            ..l
        };
        let own = &mut drafts[k];
        own.duration -= d;
        own.accent = accent;
        own.gesture.pressure_drop = (from, between);
        drafts.insert(k + 1, on_vowel);
        k += 2;
    }
}

/// The vowel before a dark /l/ (physical engine): an /l/ that closes its
/// syllable, with no vowel after it ("welcome", "also", "middle",
/// "principles"). English makes such an /l/ with the back of the tongue
/// first and the tip after, the back drawn up and back during the vowel
/// before it (Sproat & Fujimura 1993), so that vowel is dark all through:
/// the LibriSpeech reader's vowels before a coda /l/ have their F2 near
/// 1000 Hz from a quarter of the way in (the /ɪ/ of "quilter" 1090, the /ɛ/
/// of "welcome" 1070, the schwa of "pascal" 720), where the engine's,
/// made on their own shape until the /l/ began, stood at 1460 to 1660 and
/// rose into the /l/; a fall of some 500 Hz into the lateral, where his
/// falls by 70, and 15 of the 26 such vowels in the LibriSpeech texts were
/// heard wrong (11 before round 10's lateral, whose darker /l/ made the
/// fall larger), the recognizer's own choice among the vowels at a margin
/// of -0.3 nats (his +3.0).
///
/// So the last [`Tuning::dark_l_share`] of such a vowel, a lax vowel's
/// glide towards schwa included, glides towards /ʊ/, the vowel of the
/// back of the tongue raised (the slow tongue body gets part of the way).
/// Not a diphthong, which glides to its own end, and not /ʊ/ itself.
/// `phrases` (each one's drafts) is kept up to date with the split.
fn darken_before_coda_l(
    drafts: &mut Vec<Draft>,
    phrases: &mut [(usize, usize, Break)],
    timing: &Tuning,
) {
    let share = timing.dark_l_share;
    if share <= 0.0 {
        return;
    }
    let target = Phoneme::Uh.articulation();
    let mut k = 0;
    while k + 1 < drafts.len() {
        let coda = drafts[k].vowel
            && drafts[k + 1].phoneme == Phoneme::L
            && drafts.get(k + 2).is_none_or(|d| !d.vowel)
            && drafts[k].part != Part::Glide
            && target.vowel != Some(VowelKey::Vowel(drafts[k].phoneme));
        if !coda {
            k += 1;
            continue;
        }
        let head = vowel_head(drafts, k);
        let total: f32 = drafts[head..=k].iter().map(|d| d.duration).sum();
        let tail = share * total;
        // A lax vowel's glide (or a fronted /u/'s) takes the dark shape
        // instead of its own end and is drawn out to the share; any other
        // vowel's end is split off for it.
        if drafts[k].part == Part::Offglide && head + 1 == k {
            drafts[k].articulation = Articulation {
                voiced: drafts[k].articulation.voiced,
                tenseness: drafts[k].articulation.tenseness,
                ..target
            };
            // A stressed lax vowel keeps its nucleus (`Tuning::lax_floor`).
            let keep = drafts[head].duration.min(timing.lax_floor);
            let tail = tail.min(total - keep);
            drafts[head].duration += drafts[k].duration - tail;
            drafts[k].duration = tail;
        } else if drafts[k].part == Part::Head && drafts[k].duration > tail {
            drafts[k].duration -= tail;
            let mut d = drafts[k];
            d.articulation = Articulation {
                voiced: d.articulation.voiced,
                tenseness: d.articulation.tenseness,
                ..target
            };
            d.duration = tail;
            d.accent = 0.0;
            d.part = Part::Offglide;
            drafts.insert(k + 1, d);
            for q in phrases.iter_mut() {
                if q.0 > k {
                    q.0 += 1;
                }
                if q.1 > k {
                    q.1 += 1;
                }
            }
            k += 1;
        }
        k += 1;
    }
}

fn silence(duration: f32) -> Draft {
    Draft {
        phoneme: Phoneme::Sil,
        articulation: Phoneme::Sil.articulation(),
        duration,
        voiced: false,
        tenseness: None,
        accent: 1.0,
        vowel: false,
        part: Part::Head,
        gesture: Gesture::default(),
        aspiration: false,
        into_liquid: false,
        phone: NO_PHONE,
        mid_accent: 0.0,
    }
}

/// The drafts of phone `p`; `(stressed, stretch)`: whether it is stressed,
/// and how much longer than its own length the physical engine makes it
/// where it stands ([`rhythm::stretch`]).
#[allow(clippy::too_many_lines)]
fn expand(
    p: Phoneme,
    (stressed, stretch): (bool, f32),
    before_vowel: bool,
    context: Option<Context>,
    rate: f32,
    timing: &Tuning,
    out: &mut Vec<Draft>,
) {
    let mut a = p.articulation();
    if context.is_some()
        && p == Phoneme::L
        && let Some(c) = &mut a.constriction
    {
        // The physical /l/ is made closer than the classic one.
        c.diameter += timing.lateral_gap;
    }
    let a = a;
    let mut dur = base_duration(p, stressed, timing, context.is_some()) / rate;
    let accent = if stressed { 1.0 } else { 0.0 };
    let draft = |phoneme: Phoneme, articulation: Articulation, duration: f32, voiced: bool| Draft {
        phoneme,
        articulation,
        duration,
        voiced,
        tenseness: articulation.tenseness,
        // Marks stress for `shape_phrase`; replaced by a pitch ratio there.
        accent,
        vowel: is_vowel(phoneme),
        part: Part::Head,
        gesture: Gesture::default(),
        aspiration: false,
        into_liquid: false,
        phone: NO_PHONE,
        mid_accent: 0.0,
    };
    let Some(context) = context else {
        // The classic engine.
        match p.manner() {
            Manner::Diphthong => {
                // Start on the diphthong's own shape (OW starts on /o/, not on
                // AO's more open "aw"), then glide to its end (on the physical
                // engine the diphthong's own, on the classic its second vowel).
                let end = p.glide_end().expect("diphthong");
                out.push(draft(p, a, dur * 0.55, true));
                let mut tail = draft(p, end, dur * 0.45, true);
                tail.accent = 0.0;
                out.push(tail);
            }
            Manner::Stop => {
                out.push(draft(p, a, dur, a.voiced));
                if !a.voiced && before_vowel {
                    // Aspiration after the burst: open tract, breathy folds.
                    let mut asp = Phoneme::Hh.articulation();
                    asp.tongue = a.tongue;
                    out.push(draft(p, asp, timing.aspiration / rate, true));
                }
            }
            Manner::Affricate => {
                out.push(draft(p, a, dur * 0.6, a.voiced));
                let fric = if a.voiced { Phoneme::Zh } else { Phoneme::Sh };
                out.push(draft(p, fric.articulation(), dur, a.voiced));
            }
            _ => out.push(draft(p, a, dur, a.voiced)),
        }
        return;
    };

    // The physical engine's rhythm: where the phone stands in its word and
    // syllable (see `rhythm`).
    dur *= stretch;
    // The physical engine: the larynx is a second channel, timed with the
    // tract. Voiceless obstruents spread the glottis, voiced stops slacken
    // the folds and widen the pharynx, voiced fricatives leave a little
    // opening behind the vibrating folds.
    let voiceless = Gesture {
        spread: timing.voiceless_spread,
        ..Gesture::default()
    };
    let voiced_stop = Gesture {
        slack: timing.voiced_stop_slack,
        expansion: timing.cavity_expansion,
        ..Gesture::default()
    };
    let voiced_fricative = Gesture {
        spread: timing.voiced_fricative_spread,
        ..Gesture::default()
    };
    // Inside a word /ʒ/ (and /z/, with less air) is a long, loud hiss with
    // the voice under it all the way: air behind the folds, and the
    // pharynx widened a little, as for a voiced stop, so the mouth does not
    // fill and stop the voice before the hiss is over.
    let medial_zh = Gesture {
        spread: timing.voiced_sibilant_spread,
        expansion: SIBILANT_EXPANSION * timing.cavity_expansion,
        ..Gesture::default()
    };
    let with = |mut d: Draft, g: Gesture| {
        d.gesture = g;
        d
    };
    // A consonant that closes its syllable: the next sound is not the
    // start of a stressed syllable in the same word.
    let onset_of_stress = |next: Option<(Phoneme, bool, bool)>,
                           after: Option<(Phoneme, bool, bool)>| {
        after.is_some_and(|(v, stress, same)| is_vowel(v) && stress && same)
            && next.is_some_and(|(n, _, _)| is_obstruent(n))
    };
    if is_vowel(p)
        && let Some((n, _, same_word)) = context.next
        && same_word
        && is_obstruent(n)
        && !onset_of_stress(context.next, context.after_next)
    {
        // Vowels are clipped before a voiceless coda and drawn out before a
        // voiced one, by about 2:3 between the two. Not before the first
        // sound of the next word, which is no coda of theirs: clipped by
        // the /s/ of "slid", the /u/ of "canoe" was 128 ms where the two
        // reference recordings and the owner hold it for 180 to 215, and
        // drawn out by the /b/ of "birch", "the" was longer than people's.
        dur *= if n.is_voiced() {
            timing.prelenis_stretch
        } else {
            timing.prefortis_clip
        };
    }
    match p.manner() {
        Manner::Diphthong => {
            let end = p.glide_end().expect("diphthong");
            let head = NUCLEUS;
            out.push(draft(p, a, dur * head, true));
            let mut tail = draft(p, end, dur * (1.0 - head), true);
            tail.accent = 0.0;
            tail.part = Part::Glide;
            out.push(tail);
        }
        Manner::Stop if timing.tap_length > 0.0 && taps(p, context) => {
            // A touch of the tip on the way from vowel to vowel: voiced
            // throughout, no closure held, no pressure built behind it, no
            // burst and no breath.
            let touch = Articulation { voiced: true, ..a };
            let mut tap = draft(p, touch, timing.tap_length / rate, true);
            tap.gesture.approach = timing.tap_lead / rate;
            out.push(tap);
        }
        Manner::Stop if a.voiced => {
            let mut closure = with(draft(p, a, dur, true), voiced_stop);
            closure.into_liquid = into_liquid(p, context);
            out.push(closure);
            if context.next.is_none() {
                out.push(final_release(p, rate));
            }
        }
        Manner::Stop => {
            // After /s/ the glottis has been open through the hiss and is
            // closing again by the release: no aspiration, and a short
            // closure. Otherwise the closure is long and the glottis stays
            // open after the release for a voice onset time, longer the
            // further back the closure, shorter before an unstressed vowel.
            let after_s = context.prev == Some(Phoneme::S);
            let closure = if after_s {
                dur
            } else {
                dur * timing.fortis_closure
            };
            let released_into = context
                .next
                .filter(|&(n, _, same)| same && (is_vowel(n) || n.manner() == Manner::Approximant));
            let mut g = voiceless;
            if after_s && released_into.is_some() {
                // The glottis opened once, for the /s/, and is closing by
                // the release: the voice follows it at once.
                g.hold = AFTER_S_ONSET / rate;
            }
            let mut shut = with(draft(p, a, closure, false), g);
            shut.into_liquid = into_liquid(p, context);
            out.push(shut);
            if context.next.is_none() {
                out.push(final_release(p, rate));
            } else if let Some((n, n_stressed, _)) = released_into
                && !after_s
            {
                let stressed = if is_vowel(n) {
                    n_stressed
                } else {
                    context.after_next.is_some_and(|(v, s, _)| is_vowel(v) && s)
                };
                let hold = match p {
                    Phoneme::P => timing.vot_labial,
                    Phoneme::K => timing.vot_velar,
                    _ => timing.vot_alveolar,
                } * if stressed { 1.0 } else { timing.unstressed_vot };
                let mut asp = with(draft(p, a, hold / rate, false), voiceless);
                asp.aspiration = true;
                asp.gesture.hold = hold / rate;
                out.push(asp);
            }
        }
        Manner::Affricate => {
            let fric = if a.voiced { Phoneme::Zh } else { Phoneme::Sh };
            let g = if a.voiced { voiced_stop } else { voiceless };
            // The frication of /dʒ/ comes straight out of the closure, with
            // the folds still pressed for its voice.
            let fg = if a.voiced {
                Gesture::default()
            } else {
                voiceless
            };
            out.push(with(draft(p, a, dur * 0.6, a.voiced), g));
            // The frication of /tʃ/ is nearly as long as a fricative's (70
            // to 100 ms in English), or its hiss has no time to build after
            // the release; that of /dʒ/ is shorter.
            let frication = if a.voiced {
                dur
            } else {
                (AFFRICATE_FRICATION * timing.fricative_length / rate).max(dur)
            };
            out.push(with(draft(p, fric.articulation(), frication, a.voiced), fg));
        }
        Manner::Fricative => {
            let word_final = context.next.is_none_or(|(_, _, same)| !same);
            let g = if matches!(p, Phoneme::Z | Phoneme::Zh) && word_final {
                // At the end of a word English lets /z/ go partly
                // voiceless: more air behind the folds, a plainer hiss.
                Gesture {
                    spread: timing.voiced_sibilant_spread,
                    ..Gesture::default()
                }
            } else if p == Phoneme::Zh {
                medial_zh
            } else if p == Phoneme::Z {
                Gesture {
                    spread: f32::midpoint(
                        timing.voiced_fricative_spread,
                        timing.voiced_sibilant_spread,
                    ),
                    ..medial_zh
                }
            } else if a.voiced {
                voiced_fricative
            } else {
                voiceless
            };
            let after_pause = out.last().is_none_or(|d| d.phoneme == Phoneme::Sil);
            if p == Phoneme::Dh
                && after_pause
                && let Some(c) = a.constriction
            {
                // After a pause /ð/ starts as a dental stop ([d̪ð], "The
                // ..."): out of silence a voiced gap alone is heard as
                // /l/.
                let shut = Articulation {
                    constriction: Some(crate::Constriction::new(c.index, 0.0)),
                    ..a
                };
                let closure = DENTAL_ONSET / rate;
                out.push(with(draft(p, shut, closure, true), voiced_stop));
                dur = (dur - closure).max(0.5 * dur);
            }
            let g = if matches!(p, Phoneme::S | Phoneme::Sh) {
                // The gap narrows into the hiss and widens out of it: the
                // first and last part of /s ʃ/ are made with a wider gap
                // (the Speaker sends the shapes), so the hiss swells in and
                // dies away over tens of milliseconds, as a speaker's does,
                // instead of switching on and off with the gap; out of it
                // the breath is flowing freely already, and the gap has to
                // open further for the hiss to fade. Next to another
                // obstruent or a closure the tongue goes straight from one
                // to the other, and a wider gap there would be a vowel
                // between them. Not /z ʒ/, whose hiss is weak enough beside
                // their voice (any less and they are heard as /ð/), nor the
                // faint /f θ v ð/.
                let ramp = timing.frication_ramp * dur;
                let closes = |q: Phoneme| is_obstruent(q) || q.manner() == Manner::Nasal;
                let into = if context.prev.is_some_and(closes) {
                    0.0
                } else {
                    ramp
                };
                let out_of = if context.next.is_some_and(|(q, _, _)| closes(q)) {
                    0.0
                } else {
                    ramp
                };
                Gesture {
                    gap_opening: (into, out_of),
                    gap_widening: timing.frication_widening,
                    ..g
                }
            } else {
                g
            };
            out.push(with(draft(p, a, dur, a.voiced), g));
        }
        Manner::Vowel if p == Phoneme::Uw && timing.fronted_nucleus > 0.0 && fronts_u(context) => {
            // Made further front after the front of the tongue, and gliding
            // back towards the plain /u/ over the rest of a long vowel.
            let fronted = VowelKey::Fronted(p).articulation().unwrap_or(a);
            let back = VowelKey::End(p).articulation().unwrap_or(a);
            let head = timing.fronted_nucleus;
            out.push(draft(p, fronted, dur * head, true));
            if head < 1.0 {
                let mut tail = draft(p, back, dur * (1.0 - head), true);
                tail.accent = 0.0;
                tail.part = Part::Offglide;
                out.push(tail);
            }
        }
        Manner::Vowel if is_lax(p) || (p == Phoneme::Ae && stressed) => {
            // A lax vowel holds its shape for `lax_nucleus` of its length
            // and then glides towards schwa, as English lax vowels do (the
            // tense ones hold theirs): its end shape is part of the way
            // there (see `vowel_targets`). A stressed one holds its shape
            // for at least `lax_floor`, however fast the speech. A stressed
            // /æ/ glides too, over its last `1 - ae_nucleus`.
            let (head, floor) = if p == Phoneme::Ae {
                (timing.ae_nucleus, 0.0)
            } else if stressed {
                (timing.lax_nucleus, timing.lax_floor)
            } else {
                (timing.lax_nucleus, 0.0)
            };
            let end = VowelKey::End(p).articulation().unwrap_or(a);
            out.push(draft(p, a, (dur * head).max(floor), true));
            if head < 1.0 {
                let mut tail = draft(p, end, dur * (1.0 - head), true);
                tail.accent = 0.0;
                tail.part = Part::Offglide;
                out.push(tail);
            }
        }
        _ => out.push(draft(p, a, dur, a.voiced)),
    }
}

/// The physical engine's breath over each phrase (`phrases`: the drafts
/// of each and the break that ends it).
///
/// People start a phrase by raising the pressure with the folds coming
/// together and end it with the pressure falling and the folds parting: the
/// level swells in and dies away instead of switching on and off. So the
/// lungs build up from nothing over [`Tuning::phrase_onset`], lose
/// [`Tuning::phrase_decline`] of their pressure by the end, and let go
/// [`Tuning::final_drop`] of it over the last vowel, the consonants after
/// it keeping what is left (half of that before a comma); in a pause they
/// stop.
///
/// A phrase that opens on a vowel or a sonorant starts from
/// [`Tuning::onset_pressure`] of the pressure, about what the folds need
/// to start, and the voice grows from there; one that opens on a nasal or
/// a liquid starts that sound [`Tuning::sonorant_onset`] earlier, out of
/// the pause, so that it is voiced by the time it was due rather than
/// silent for the 40 ms the lungs take to get there. Before a phrase that opens on
/// an obstruent or /h/ the lungs start pushing in the pause, its last part
/// split off for it, with all their pressure, for [`Tuning::lead`]: the
/// voice starts as from a closure in the middle of speech, the release
/// bursts and the first hiss is there from the start (built up from
/// nothing, an opening /f/, /s/ or /h/ was mostly lost). `phrases` is kept
/// up to date with the split.
fn breathe(
    drafts: &mut Vec<Draft>,
    phrases: &mut [(usize, usize, Break)],
    accented: &[bool],
    timing: &Tuning,
) {
    for d in drafts.iter_mut() {
        if d.phoneme == Phoneme::Sil {
            d.gesture.pressure_drop = (1.0, 1.0);
        }
    }
    for p in (0..phrases.len()).rev() {
        let (a, b, brk) = phrases[p];
        let Some(first) = drafts.get(a).filter(|_| a < b) else {
            continue;
        };
        let pause = a
            .checked_sub(1)
            .filter(|&k| drafts[k].phoneme == Phoneme::Sil);
        // Seconds the lungs push before the phrase, the share of their
        // pressure they start from, and the seconds over which they build
        // up to all of it.
        let breathy = is_obstruent(first.phoneme) || first.phoneme.manner() == Manner::Aspirate;
        // A phrase that opens on a nasal or a liquid starts it a little
        // earlier, out of the pause: the voice needs the lungs' first 40 ms
        // to start, and the /n m l/ is voiced from then on rather than
        // heard only in its release (`Tuning::sonorant_onset`).
        let sonorant =
            !first.vowel && matches!(first.phoneme.manner(), Manner::Nasal | Manner::Approximant);
        if let Some(k) = pause.filter(|_| sonorant && timing.sonorant_onset > 0.0) {
            let early = timing.sonorant_onset.min(0.5 * drafts[k].duration);
            drafts[k].duration -= early;
            drafts[a].duration += early;
        }
        let (ready, floor, swell_time) = match pause {
            Some(k) if breathy => (drafts[k].duration.min(timing.lead), 1.0, 0.0),
            _ => (0.0, timing.onset_pressure, timing.phrase_onset),
        };
        let phrase = &drafts[a..b];
        let total: f32 = phrase.iter().map(|d| d.duration).sum();
        let last_vowel = phrase.iter().rposition(|d| d.vowel).unwrap_or(0);
        let final_start: f32 = phrase[..vowel_head(phrase, last_vowel)]
            .iter()
            .map(|d| d.duration)
            .sum();
        let final_end = phrase[..last_vowel].iter().map(|d| d.duration).sum::<f32>()
            + phrase.get(last_vowel).map_or(0.0, |d| d.duration);
        let final_drop = match brk {
            Break::Comma => 0.5 * timing.final_drop,
            _ => timing.final_drop,
        };
        let held = |t: f32| {
            let swell = if swell_time > 0.0 {
                floor + (1.0 - floor) * ((t + ready) / swell_time).clamp(0.0, 1.0)
            } else {
                1.0
            };
            let decline = 1.0 - timing.phrase_decline * (t / total.max(DECLINE_SPAN)).max(0.0);
            let fade = if t > final_start && final_end > final_start {
                let x = ((t - final_start) / (final_end - final_start)).min(1.0);
                1.0 - final_drop * x * x * (3.0 - 2.0 * x)
            } else {
                1.0
            };
            1.0 - swell * decline * fade
        };
        let mut t = 0.0;
        for d in &mut drafts[a..b] {
            let start = held(t);
            t += d.duration;
            d.gesture.pressure_drop = (start, held(t));
        }
        decline_through_vowels(&mut drafts[a..b], timing);
        if let Some(acc) = accented.get(a..b) {
            weigh_syllables(&mut drafts[a..b], acc, timing);
        }
        // After the last phrase the breath dies away over the closing pause,
        // so its last sound is not cut off.
        if b + 1 == drafts.len()
            && let Some(end) = drafts.get_mut(b).filter(|d| d.phoneme == Phoneme::Sil)
        {
            end.gesture.pressure_drop = (held(total), 1.0);
        }
        let Some(k) = pause.filter(|_| ready > 0.0) else {
            continue;
        };
        let mut lead_in = drafts[k];
        lead_in.duration = ready;
        lead_in.gesture.pressure_drop = (held(-ready), held(0.0));
        if drafts[k].duration > ready {
            drafts[k].duration -= ready;
            drafts.insert(k + 1, lead_in);
            for q in &mut phrases[p..] {
                (q.0, q.1) = (q.0 + 1, q.1 + 1);
            }
        } else {
            drafts[k] = lead_in;
        }
    }
}

/// The shortest phrase that loses all of [`Tuning::phrase_decline`] of its
/// breath, seconds; a shorter one loses as much as the lungs would in its
/// time. The lungs run down with time, not with the share of a phrase
/// said: at round 10's 0.3, a two-syllable "a pa" lost it all, and its
/// last vowel's voice started 40 Hz off, or not at all, after the /p/
/// (`the_voice_starts_higher_after_a_voiceless_stop`).
const DECLINE_SPAN: f32 = 2.0;

/// Level of the physical voice against lung pressure: it rises as the
/// pressure to this power (`examples/tiltprobe.rs --level`: /a/ at 115 Hz,
/// 3.2 dB from 6.4 to 8 cmH2O).
const LEVEL_EXPONENT: f32 = 1.64;

/// The breath sinks through each vowel (physical engine): a speaker's
/// vowel is loudest soon after it starts and falls by some 2 dB to its end
/// (the LibriSpeech reader's stressed vowels: 2.1 dB from their peak to
/// their end, median), where the engine's held its level or grew. So the
/// lungs let go of [`Tuning::vowel_decline`] dB of level across each
/// vowel (its glide or offglide included) and take it back in the sounds
/// that follow; they follow within some 30 ms.
fn decline_through_vowels(phrase: &mut [Draft], timing: &Tuning) {
    if timing.vowel_decline <= 0.0 {
        return;
    }
    let end = crate::math::powf(10.0, -timing.vowel_decline / (20.0 * LEVEL_EXPONENT));
    let mut k = 0;
    while k < phrase.len() {
        if !phrase[k].vowel {
            k += 1;
            continue;
        }
        let last = k + phrase[k + 1..]
            .iter()
            .take_while(|d| d.vowel && d.part != Part::Head)
            .count();
        let total: f32 = phrase[k..=last].iter().map(|d| d.duration).sum();
        if total > 0.0 {
            let share = |t: f32| 1.0 + (end - 1.0) * (t / total);
            let mut t = 0.0;
            for d in &mut phrase[k..=last] {
                let (a, b) = d.gesture.pressure_drop;
                let s0 = share(t);
                t += d.duration;
                d.gesture.pressure_drop = (1.0 - (1.0 - a) * s0, 1.0 - (1.0 - b) * share(t));
            }
        }
        k = last + 1;
    }
}

/// The breath of each syllable against the accented ones (physical
/// engine): the lungs push less through an unaccented vowel, by
/// [`Tuning::unaccented_level`] dB of level, and the nasals and liquids
/// between two vowels take the breath of the line from the middle of the
/// one to the middle of the other (those before the first vowel and after
/// the last take theirs), as a followed schedule's phones do. The
/// obstruents keep the phrase's breath, as the lungs take back what they
/// let go through each vowel (`decline_through_vowels`): with their breath
/// on that line too, the /z/ of "his" and "has" lost so much of its hiss
/// that it stood 9.6 dB under /s/ at 5 to 8 kHz (8.3 without), where a
/// /z/ heard as /ð/ begins (`tests/sibilant_gate.rs`).
///
/// Measured in the LibriSpeech reader's 16 sentences (aligned; each vowel's
/// level against the mean of its sentence's vowels, fitted over stress,
/// function word, place in the phrase and vowel height): his accented
/// vowels stand 2.7 dB over his unaccented ones, the rules' 1.2 (the pitch
/// accent and the length alone); the two reference recordings 1.9 and 2.2,
/// the owner's 6.5.
#[allow(clippy::many_single_char_names)]
fn weigh_syllables(phrase: &mut [Draft], accented: &[bool], timing: &Tuning) {
    if timing.unaccented_level <= 0.0 {
        return;
    }
    let low = crate::math::powf(10.0, -timing.unaccented_level / (20.0 * LEVEL_EXPONENT));
    // Each vowel's middle, seconds into the phrase, and its share.
    let mut mids: Vec<(f32, f32)> = Vec::new();
    let mut t = 0.0;
    let mut k = 0;
    while k < phrase.len() {
        if !phrase[k].vowel {
            t += phrase[k].duration;
            k += 1;
            continue;
        }
        let last = k + phrase[k + 1..]
            .iter()
            .take_while(|d| d.vowel && d.part != Part::Head)
            .count();
        let total: f32 = phrase[k..=last].iter().map(|d| d.duration).sum();
        let share = if accented.get(k).copied().unwrap_or(true) {
            1.0
        } else {
            low
        };
        mids.push((t + 0.5 * total, share));
        t += total;
        k = last + 1;
    }
    let share_at = |tau: f32| -> f32 {
        let i = mids.partition_point(|m| m.0 < tau);
        match (i.checked_sub(1).map(|j| mids[j]), mids.get(i)) {
            (Some(a), Some(b)) if b.0 > a.0 => lerp(a.1, b.1, (tau - a.0) / (b.0 - a.0)),
            (Some(a), _) => a.1,
            (_, Some(b)) => b.1,
            _ => 1.0,
        }
    };
    let mut t = 0.0;
    for d in phrase.iter_mut() {
        let start = t;
        t += d.duration;
        if is_obstruent(d.phoneme) || d.phoneme == Phoneme::Sil {
            continue;
        }
        let (a, b) = d.gesture.pressure_drop;
        d.gesture.pressure_drop = (
            1.0 - (1.0 - a) * share_at(start),
            1.0 - (1.0 - b) * share_at(t),
        );
    }
}

/// Share of a voiced stop's widening of the pharynx (`Tuning::cavity_expansion`)
/// that /z ʒ/ make inside a word (physical engine).
const SIBILANT_EXPANSION: f32 = 0.5;

/// How long the breath goes on through the release of a stop at the end
/// of a phrase, seconds (physical engine).
const FINAL_RELEASE: f32 = 0.04;

/// The release of a stop that ends a phrase (physical engine): the mouth
/// opens towards the vowel before it on the air still pent up behind the
/// closure, so the closure ends in an audible burst rather than fading out
/// shut. Word-final stops are often unreleased in running speech, but a
/// listener hearing no release hears no stop.
fn final_release(p: Phoneme, rate: f32) -> Draft {
    Draft {
        phoneme: p,
        articulation: Articulation {
            voiced: false,
            ..Phoneme::Hh.articulation()
        },
        duration: FINAL_RELEASE / rate,
        voiced: false,
        tenseness: None,
        accent: 0.0,
        vowel: false,
        part: Part::Head,
        gesture: Gesture::default(),
        aspiration: false,
        into_liquid: false,
        phone: NO_PHONE,
        mid_accent: 0.0,
    }
}

/// The physical engine's intonation for one phrase, in semitones at
/// expressiveness 1 from the [`Tuning`]: the pitch declines by
/// [`Tuning::declination`] over the phrase; on each stressed vowel it rises
/// by [`Tuning::accent_size`] and sinks back over [`Tuning::accent_decay`]
/// seconds through what follows (so it moves inside vowels and from
/// syllable to syllable, as a speaker's does, rather than stepping up for
/// each stress and down again at its consonant); a statement falls by
/// [`Tuning::final_fall`] across its last vowel, after an accent on it that
/// has risen on the way in; a question rises instead and a comma lifts a
/// little.
#[allow(clippy::cast_precision_loss)]
fn intonation(drafts: &mut [Draft], brk: Break, expr: f32, timing: &Tuning) {
    let total: f32 = drafts.iter().map(|d| d.duration).sum();
    if total <= 0.0 {
        return;
    }
    // Start and end time of every draft.
    let mut spans = Vec::with_capacity(drafts.len());
    let mut t = 0.0;
    for d in drafts.iter() {
        spans.push((t, t + d.duration));
        t += d.duration;
    }
    // The last vowel's last draft, and its first (a lax vowel's offglide
    // is part of it).
    let last_draft = drafts.iter().rposition(|d| d.vowel);
    let last_vowel = last_draft.map(|k| vowel_head(drafts, k));
    let falling = matches!(brk, Break::Period | Break::Exclamation);
    let accents: Vec<(f32, f32)> = drafts
        .iter()
        .zip(&spans)
        .enumerate()
        .filter(|(_, (d, _))| d.accent > 0.0 && d.vowel)
        .map(|(k, (_, &(a, b)))| {
            let b = match drafts.get(k + 1) {
                Some(n) if n.part == Part::Offglide => spans[k + 1].1,
                _ => b,
            };
            // The nucleus of a statement peaks as its vowel begins, its rise
            // carried by the sounds before it, and falls from there; any
            // other accent rises from the start of its syllable and peaks
            // `accent_peak` of the way through its vowel.
            let onset = spans[..k]
                .iter()
                .rev()
                .zip(drafts[..k].iter().rev())
                .take_while(|(_, d)| !d.vowel)
                .last()
                .map_or(a - 0.05, |(&(s, _), _)| s);
            if falling && Some(k) == last_vowel {
                (onset, a)
            } else if timing.accent_peak < 1.0 {
                (onset.min(a), a + timing.accent_peak * (b - a))
            } else {
                (a, b)
            }
        })
        .collect();
    let size = timing.accent_size * expr;
    let accent = |t: f32| {
        accents
            .iter()
            .map(|&(a, b)| {
                if t < a {
                    0.0
                } else if t <= b {
                    size * (t - a) / (b - a).max(1e-3)
                } else {
                    size * (1.0 - (t - b) / timing.accent_decay).max(0.0)
                }
            })
            .fold(0.0f32, f32::max)
    };
    let (vowel_start, vowel_end) = last_vowel
        .zip(last_draft)
        .map_or((total, total), |(k, end)| (spans[k].0, spans[end].1));
    let boundary = |t: f32| {
        let x = ((t - vowel_start) / (vowel_end - vowel_start).max(1e-3)).clamp(0.0, 1.0);
        let shape = x * x * (3.0 - 2.0 * x);
        match brk {
            Break::Period => -timing.final_fall * expr * shape,
            Break::Exclamation => -1.4 * timing.final_fall * expr * shape,
            Break::Question => QUESTION_RISE * expr * shape,
            Break::Comma => COMMA_RISE * expr * shape,
        }
    };
    let declination = timing.declination * expr;
    let semitones = |t: f32| declination * (0.5 - t / total) + accent(t) + boundary(t);
    for (d, &(start, end)) in drafts.iter_mut().zip(&spans) {
        d.accent = crate::math::powf(2.0, semitones(end) / 12.0);
        if d.vowel {
            d.mid_accent = crate::math::powf(2.0, semitones(f32::midpoint(start, end)) / 12.0);
        }
    }
    // After a question's rise the consonants stay up.
    if let (Some(v), Break::Question) = (last_draft, brk) {
        let top = drafts[v].accent;
        for d in &mut drafts[v + 1..] {
            d.accent = top;
        }
    }
}

/// Rise at the end of a question, semitones at expressiveness 1.
const QUESTION_RISE: f32 = 8.0;
/// Lift at a comma, semitones at expressiveness 1: the phrase is not over.
const COMMA_RISE: f32 = 1.5;

/// Intonation for one phrase: gentle declination, accents on stressed
/// vowels, and a boundary tone.
#[allow(clippy::cast_precision_loss)]
fn shape_phrase(drafts: &mut [Draft], brk: Option<Break>, expr: f32) {
    let total: f32 = drafts.iter().map(|d| d.duration).sum();
    if total <= 0.0 {
        return;
    }
    let last_vowel = drafts.iter().rposition(|d| d.vowel);
    let mut t = 0.0;
    for (k, d) in drafts.iter_mut().enumerate() {
        t += d.duration;
        let pos = t / total;
        let declination = 1.0 + expr * (0.12 - 0.22 * pos);
        let accent = if d.accent > 0.0 && d.vowel {
            1.0 + 0.18 * expr
        } else {
            1.0
        };
        let boundary = if Some(k) == last_vowel {
            match brk {
                Some(Break::Question) => 1.0 + 0.6 * expr,
                Some(Break::Period) => 1.0 - 0.18 * expr,
                Some(Break::Exclamation) => 1.0 - 0.25 * expr,
                Some(Break::Comma) => 1.0 + 0.08 * expr,
                None => 1.0,
            }
        } else {
            1.0
        };
        d.accent = declination * accent * boundary;
    }
    // After a question rise, keep the remaining consonants up.
    if let (Some(v), Some(Break::Question)) = (last_vowel, brk) {
        let top = drafts[v].accent;
        for d in &mut drafts[v + 1..] {
            d.accent = top;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Phoneme as P;
    use alloc::vec;

    fn phones(text: &str) -> Vec<Phoneme> {
        phonemize(text)
            .into_iter()
            .filter_map(|t| match t {
                Token::Phone { phoneme, .. } => Some(phoneme),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn dictionary_words() {
        assert_eq!(
            phones("hello world"),
            vec![P::Hh, P::Ax, P::L, P::Ow, P::W, P::Er, P::L, P::D]
        );
    }

    #[test]
    fn spelling_rules() {
        assert_eq!(phones("cat"), vec![P::K, P::Ae, P::T]);
        assert_eq!(phones("make"), vec![P::M, P::Ey, P::K]);
        assert_eq!(phones("time"), vec![P::T, P::Ay, P::M]);
        assert_eq!(phones("ship"), vec![P::Sh, P::Ih, P::P]);
        assert_eq!(phones("night"), vec![P::N, P::Ay, P::T]);
        assert_eq!(phones("city"), vec![P::S, P::Ih, P::T, P::Iy]);
        assert_eq!(phones("sing"), vec![P::S, P::Ih, P::Ng]);
        assert_eq!(phones("nation"), vec![P::N, P::Ey, P::Sh, P::Ax, P::N]);
        assert_eq!(phones("dogs"), vec![P::D, P::Aa, P::G, P::Z]);
        assert_eq!(phones("bell"), vec![P::B, P::Eh, P::L]);
        assert_eq!(phones("knight"), vec![P::N, P::Ay, P::T]);
        assert_eq!(phones("queen"), vec![P::K, P::W, P::Iy, P::N]);
        assert_eq!(phones("boat"), vec![P::B, P::Ow, P::T]);
    }

    #[test]
    fn arpabet_spans_bypass_rules() {
        assert_eq!(phones("[K AE1 T]"), vec![P::K, P::Ae, P::T]);
        assert_eq!(phones("[nonsense]"), vec![]);
    }

    #[test]
    fn numbers_are_spelled_out() {
        assert_eq!(number_words("0").trim(), "zero");
        assert_eq!(number_words("42").trim(), "forty two");
        assert_eq!(
            number_words("1984").trim(),
            "one thousand nine hundred eighty four"
        );
        assert_eq!(number_words("007").trim(), "zero zero seven");
        assert_ne!(phones("route 66"), vec![]);
    }

    #[test]
    fn punctuation_makes_breaks() {
        let t = phonemize("hi, you?! ok.");
        let breaks: Vec<_> = t
            .iter()
            .filter_map(|t| match t {
                Token::Break(b) => Some(*b),
                _ => None,
            })
            .collect();
        assert_eq!(breaks, vec![Break::Comma, Break::Question, Break::Period]);
    }

    #[test]
    fn content_words_are_stressed() {
        let stressed: Vec<_> = phonemize("the cat")
            .into_iter()
            .filter_map(|t| match t {
                Token::Phone {
                    phoneme,
                    stressed: true,
                } => Some(phoneme),
                _ => None,
            })
            .collect();
        assert_eq!(stressed, vec![P::Ae]);
    }

    #[test]
    fn questions_rise_statements_fall() {
        let p = Prosody::default();
        let q = Utterance::from_text("you are here?", &p);
        let s = Utterance::from_text("you are here.", &p);
        let last_vowel_pitch = |u: &Utterance| {
            u.segments()
                .iter()
                .rev()
                .find(|s| is_vowel(s.phoneme))
                .unwrap()
                .pitch
                .1
        };
        assert!(last_vowel_pitch(&q) > p.pitch * 1.1);
        assert!(last_vowel_pitch(&s) < p.pitch);
    }

    #[test]
    fn consonants_borrow_the_next_vowel_tongue() {
        let u = Utterance::from_text("[S IY]", &Prosody::default());
        let s = u.segments()[0];
        assert_eq!(s.phoneme, P::S);
        assert_eq!(s.articulation.tongue, P::Iy.articulation().tongue);
    }

    #[test]
    fn faster_rate_is_shorter() {
        let slow = Utterance::from_text(
            "pink trombone",
            &Prosody {
                rate: 0.5,
                ..Prosody::default()
            },
        );
        let fast = Utterance::from_text(
            "pink trombone",
            &Prosody {
                rate: 2.0,
                ..Prosody::default()
            },
        );
        assert!(slow.duration() > 3.0 * fast.duration());
    }

    #[test]
    fn speaker_plays_and_finishes() {
        let mut voice = Trombone::new(16_000.0);
        Speaker::prepare(&mut voice);
        let mut sp = Speaker::new();
        let u = Utterance::from_text("Hello, pink trombone!", &Prosody::default());
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let n = ((u.duration() + 0.3) * 16_000.0) as usize;
        sp.speak(u);
        assert!(sp.is_speaking());
        let mut buf = vec![0.0; n];
        sp.render(&mut voice, &mut buf);
        assert!(!sp.is_speaking());
        assert!(buf.iter().all(|s| s.is_finite()));
        let energy: f32 = buf.iter().map(|s| s * s).sum();
        assert!(energy > 1.0, "{energy}");
    }

    #[test]
    fn physical_vowels_have_lengths_of_their_own() {
        let length = |engine: Engine, text: &str, p: Phoneme| {
            let prosody = Prosody {
                engine,
                ..Prosody::default()
            };
            Utterance::from_text(text, &prosody)
                .segments()
                .iter()
                .filter(|s| s.phoneme == p)
                .map(|s| s.duration)
                .sum::<f32>()
        };
        // Lax /ɪ/ is shorter than tense /i/ on the physical engine...
        let (bit, beat) = (
            length(Engine::Physical, "[S IH1 L] [AX0]", P::Ih),
            length(Engine::Physical, "[S IY1 L] [AX0]", P::Iy),
        );
        assert!(bit < 0.95 * beat, "{bit} {beat}");
        // ...and the classic engine keeps them alike.
        let (bit, beat) = (
            length(Engine::Classic, "[S IH1 L] [AX0]", P::Ih),
            length(Engine::Classic, "[S IY1 L] [AX0]", P::Iy),
        );
        assert!((bit - beat).abs() < 1e-6, "{bit} {beat}");
    }

    #[test]
    fn never_panics_on_odd_text() {
        for t in [
            "",
            "   ",
            "!!!",
            "[",
            "]]",
            "ÄÖÜ ß",
            "a",
            "e",
            "y",
            "x",
            "123456789012",
            "'''",
            "—",
        ] {
            let u = Utterance::from_text(t, &Prosody::default());
            assert!(
                u.segments()
                    .iter()
                    .all(|s| s.duration > 0.0 && s.duration.is_finite())
            );
        }
    }
}
