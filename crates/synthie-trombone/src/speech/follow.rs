//! Speaking with a person's timing and pitch.
//!
//! An [`Utterance`] is compiled from the text as always, so every
//! articulatory rule, coarticulation and laryngeal gesture stays; then
//! [`Utterance::follow`] overwrites, phone by phone, how long each sound
//! lasts and what the pitch does, from a [`Schedule`] measured on a
//! recording of someone saying that text (`tools/judge/prosody.py` writes
//! one: forced alignment for the phones, a pitch track, the level of each
//! phone). [`Schedule::phrase`] first moves the phrase breaks of the text
//! to where the speaker paused.
//!
//! What the rules still decide: how a phone's time is shared out over its
//! segments (a stop's closure, release and breath, a diphthong's nucleus
//! and glide, a lax vowel's glide to schwa) in the proportions they gave
//! them; the closure of a voiceless stop keeps its floor
//! ([`Tuning::closure_floor`](crate::Tuning::closure_floor)); the pitch
//! kick after a voiceless consonant ([`Gesture::kick`](super::Gesture)),
//! which is the folds' physics and is added on top of the speaker's
//! contour; the voice quality of each sound and the breath's swell and
//! fade over a phrase. Replaced: every duration, the declination, the
//! accents and the fall or rise at a phrase end; within limits, how hard
//! the lungs push through each phone.

use super::{Break, NO_PHONE, Token, Utterance};
use crate::inverse::{self, Shape};
use crate::math::{lerp, ln, powf};
use crate::vowel_targets::VowelKey;
use crate::{Manner, Phoneme, Tuning};
use alloc::vec::Vec;
use core::fmt;

/// Shortest segment the schedule may make, seconds.
const MIN_SEGMENT: f32 = 0.006;
/// What a voiceless stop's closure segment is given beyond its floor, so
/// that the mouth has shut and held for the floor before the segment
/// ends, seconds: a shorter one is drawn out by the Speaker and the
/// schedule slips behind.
const CLOSURE_MARGIN: f32 = 0.008;
/// Shortest pause, seconds.
const MIN_PAUSE: f32 = 0.03;
/// A gap between two phones of the schedule at least this long is a
/// pause, seconds.
pub const PAUSE_GAP: f32 = 0.02;
/// A phone this much shorter than the rules make it, or more, has its
/// articulators sped up (a ratio of durations).
const SPED_UP: f32 = 1.01;
/// Step of the pitch track the utterance is given, seconds.
const TRACK_HOP: f32 = 0.005;

/// One phone of a [`Schedule`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhoneSpan {
    /// The phone, if known: checked against the utterance's.
    pub phoneme: Option<Phoneme>,
    /// Start and end on the recording, seconds.
    pub start: f32,
    /// End on the recording, seconds.
    pub end: f32,
    /// How hard the lungs push through it, a share of what the rules give
    /// (`1`), limited to [`FollowOptions::breath_range`].
    pub breath: f32,
    /// How firmly the folds are held through it, a share of the rules'
    /// tenseness (`1`), limited to [`FollowOptions::tenseness_range`]: the
    /// speaker's voice quality from phone to phone.
    pub tenseness: f32,
    /// For a vowel: its first two formants as the speaker has them, each a
    /// share of the formant of the voice's own vowel shape (`(1, 1)` is the
    /// voice's vowel), limited to ±[`FollowOptions::formant_range`]. The
    /// vowel, and the consonants made on its body, get a shape solved for
    /// them (see [`Utterance::follow_with`]).
    pub formants: Option<(f32, f32)>,
}

impl PhoneSpan {
    /// A phone from `start` to `end` seconds, everything else as the rules
    /// have it.
    #[must_use]
    pub const fn new(phoneme: Option<Phoneme>, start: f32, end: f32) -> Self {
        Self {
            phoneme,
            start,
            end,
            breath: 1.0,
            tenseness: 1.0,
            formants: None,
        }
    }
}

/// What [`Utterance::follow_with`] takes from a [`Schedule`] besides the
/// durations, and within which limits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FollowOptions {
    /// Which pitch.
    pub pitch: PitchMode,
    /// The range of a phone's breath share ([`PhoneSpan::breath`]).
    pub breath_range: (f32, f32),
    /// The range of a phone's tenseness share ([`PhoneSpan::tenseness`]);
    /// `(1, 1)` leaves the rules' tenseness.
    pub tenseness_range: (f32, f32),
    /// How far a vowel's formants may be moved, a share either way
    /// ([`PhoneSpan::formants`]); `0` leaves the voice's vowels.
    pub formant_range: f32,
    /// The voice's tract length (see
    /// [`Trombone::set_tract_length`](crate::Trombone::set_tract_length)),
    /// which the vowel shapes are solved for.
    pub tract_length: f32,
    /// Fastest the articulators may be driven, a multiple of their rule
    /// speed: a phone the schedule makes shorter than the rules do has
    /// them moved faster in proportion, up to this (`1` never).
    pub max_speed: f32,
    /// Shortest a phone may be made, a share of its rule duration; `0`
    /// gives every phone the schedule's duration.
    pub shortest: f32,
}

impl FollowOptions {
    /// Everything followed, within the limits of `tuning`, for a voice of
    /// `tract_length`.
    #[must_use]
    pub fn new(pitch: PitchMode, tuning: &Tuning, tract_length: f32) -> Self {
        let t = tuning.clamped();
        Self {
            pitch,
            breath_range: (t.follow_breath_low, t.follow_breath_high),
            tenseness_range: (t.follow_tenseness_low, t.follow_tenseness_high),
            formant_range: t.follow_formant_range,
            tract_length,
            max_speed: t.follow_speed_max,
            shortest: t.follow_shortest,
        }
    }

    /// Only the durations, pauses, pitch and breath, as until round 9.
    #[must_use]
    pub fn timing_only(pitch: PitchMode) -> Self {
        Self {
            tenseness_range: (1.0, 1.0),
            formant_range: 0.0,
            max_speed: 1.0,
            shortest: 0.0,
            ..Self::new(pitch, &Tuning::DEFAULT, 1.0)
        }
    }
}

/// What a followed utterance asks of the [`Speaker`](super::Speaker) beyond
/// its segments: per segment, how fast the articulators move to its shape
/// and the vowel shape it is made with. Empty for an utterance that
/// follows nothing.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Followed {
    /// Articulation speed per segment, a multiple of the voice's.
    pub(crate) speed: Vec<f32>,
    /// Per segment, the vowel shape it is made with in place of the
    /// voice's own for that key: the vowel's, and that of the vowel a
    /// consonant leans on.
    pub(crate) shape: Vec<Option<(VowelKey, Shape)>>,
}

impl Followed {
    /// Articulation speed for segment `k`.
    pub(crate) fn speed_of(&self, k: usize) -> f32 {
        self.speed.get(k).copied().unwrap_or(1.0)
    }

    /// The vowel shape segment `k` is made with, if it has one of its own.
    pub(crate) fn shape_of(&self, k: usize) -> Option<(VowelKey, Shape)> {
        self.shape.get(k).copied().flatten()
    }
}

/// A person's timing and pitch for a text: one [`PhoneSpan`] per phone of
/// the text, in order, and a pitch track, all on the recording's clock.
/// A gap of [`PAUSE_GAP`] or more between two phones is a pause.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Schedule {
    /// The phones.
    pub phones: Vec<PhoneSpan>,
    /// Time of the pitch track's first value, seconds.
    pub pitch_start: f32,
    /// Step of the pitch track, seconds.
    pub pitch_hop: f32,
    /// Pitch, Hz, a value everywhere (through unvoiced stretches too).
    pub pitch: Vec<f32>,
    /// The speaker's median pitch over voiced frames, Hz (for
    /// [`PitchMode::Voice`]).
    pub median_hz: f32,
}

/// Which pitch the utterance is given.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PitchMode {
    /// The speaker's pitch as it is.
    #[default]
    Speaker,
    /// The speaker's contour moved, in semitones, so that its median is
    /// the median of the voice's own (rule) contour for the text.
    Voice,
}

/// What [`Utterance::follow`] did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FollowReport {
    /// Phones the schedule made shorter than the engine can make them
    /// (their segments' floors); they were given their floors.
    pub too_short: usize,
    /// Seconds added to those phones in all.
    pub added: f32,
    /// Semitones the speaker's contour was moved by.
    pub shift: f32,
    /// Breath shares that had to be limited to
    /// [`FollowOptions::breath_range`].
    pub breath_limited: usize,
    /// Tenseness shares that had to be limited to
    /// [`FollowOptions::tenseness_range`].
    pub tenseness_limited: usize,
    /// Vowels given a shape of their own for the speaker's formants.
    pub shaped: usize,
    /// Of those, the formants that had to be limited to
    /// ±[`FollowOptions::formant_range`].
    pub formants_limited: usize,
    /// Phones whose articulators were sped up (shorter than the rules
    /// make them), and the fastest speed given.
    pub sped: usize,
    /// The fastest articulation speed given, a multiple of the voice's.
    pub fastest: f32,
    /// Each phone's start and end on the utterance's clock, seconds.
    pub spans: Vec<(f32, f32)>,
}

/// Why a schedule does not fit an utterance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduleError {
    /// The schedule has another number of phones than the utterance.
    PhoneCount {
        /// The utterance's.
        utterance: usize,
        /// The schedule's.
        schedule: usize,
    },
    /// A phone of the schedule is not the utterance's.
    Phoneme {
        /// Which phone.
        index: usize,
        /// The utterance's.
        utterance: Phoneme,
        /// The schedule's.
        schedule: Phoneme,
    },
    /// The schedule has no pitch, or a step that is not positive.
    NoPitch,
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PhoneCount {
                utterance,
                schedule,
            } => write!(
                f,
                "the text has {utterance} phones and the schedule {schedule}"
            ),
            Self::Phoneme {
                index,
                utterance,
                schedule,
            } => write!(
                f,
                "phone {index} is {utterance} in the text and {schedule} in the schedule"
            ),
            Self::NoPitch => write!(f, "the schedule has no pitch track"),
        }
    }
}

impl Schedule {
    /// Whether the speaker paused after phone `k`.
    #[must_use]
    pub fn pause_after(&self, k: usize) -> bool {
        self.gap_after(k) >= PAUSE_GAP
    }

    /// Seconds between the end of phone `k` and the start of the next.
    fn gap_after(&self, k: usize) -> f32 {
        match (self.phones.get(k), self.phones.get(k + 1)) {
            (Some(a), Some(b)) => b.start - a.end,
            _ => 0.0,
        }
    }

    /// `tokens` (see [`super::phonemize`]) with their phrase breaks where
    /// the speaker paused: a break inside the text the speaker went
    /// straight through becomes a word break, and a word break the speaker
    /// paused at becomes a comma. The phones stay as they are (the weak
    /// forms of function words are already chosen), so the utterance
    /// compiled from the result fits the schedule; its phrases breathe and
    /// end where the speaker's did.
    #[must_use]
    pub fn phrase(&self, tokens: &[Token]) -> Vec<Token> {
        let total = tokens
            .iter()
            .filter(|t| matches!(t, Token::Phone { .. }))
            .count();
        let mut seen = 0;
        tokens
            .iter()
            .map(|&t| match t {
                Token::Phone { .. } => {
                    seen += 1;
                    t
                }
                Token::WordBreak if seen > 0 && self.pause_after(seen - 1) => {
                    Token::Break(Break::Comma)
                }
                Token::Break(_) if seen > 0 && seen < total && !self.pause_after(seen - 1) => {
                    Token::WordBreak
                }
                _ => t,
            })
            .collect()
    }

    /// The pitch at `t` seconds on the recording, Hz.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn pitch_at(&self, t: f32) -> f32 {
        let last = self.pitch.len() - 1;
        let x = ((t - self.pitch_start) / self.pitch_hop).max(0.0);
        let i = (x as usize).min(last);
        let j = (i + 1).min(last);
        lerp(self.pitch[i], self.pitch[j], (x - i as f32).clamp(0.0, 1.0))
    }
}

/// Share `target` seconds out over segments whose rules gave them `rule`,
/// in those proportions, none under its `floor`. Returns the durations
/// (all at their floors when `target` is less than the floors add up to).
fn share_out(rule: &[f32], floor: &[f32], target: f32) -> Vec<f32> {
    let mut fixed: Vec<bool> = floor.iter().map(|_| false).collect();
    let mut out = floor.to_vec();
    if target <= floor.iter().sum::<f32>() {
        return out;
    }
    for _ in 0..rule.len() {
        let held: f32 = floor
            .iter()
            .zip(&fixed)
            .filter(|(_, f)| **f)
            .map(|(m, _)| m)
            .sum();
        let free: f32 = rule
            .iter()
            .zip(&fixed)
            .filter(|(_, f)| !**f)
            .map(|(r, _)| r)
            .sum();
        let scale = if free > 0.0 {
            (target - held) / free
        } else {
            0.0
        };
        let mut again = false;
        for i in 0..rule.len() {
            if fixed[i] {
                continue;
            }
            out[i] = rule[i] * scale;
            if out[i] < floor[i] {
                out[i] = floor[i];
                fixed[i] = true;
                again = true;
            }
        }
        if !again {
            break;
        }
    }
    out
}

impl Utterance {
    /// Speak with the timing and pitch of `schedule` (see the module):
    /// every phone gets the schedule's duration, shared out over its
    /// segments in the proportions the rules gave them (a voiceless stop's
    /// closure no shorter than its floor), every pause the speaker's
    /// length, and the utterance a pitch track that follows the speaker's
    /// contour phone by phone (moved by `mode`), with the rules' pitch kick
    /// after voiceless consonants on top. Each phone's
    /// [`PhoneSpan::breath`] scales the breath the rules give it. The
    /// pause the utterance opens with and the one it ends with keep their
    /// lengths.
    ///
    /// Compile the utterance from [`Schedule::phrase`] of the text's
    /// tokens, so its pauses are where the speaker's were; a pause of the
    /// speaker's the utterance has no pause for (inside a word) is shared
    /// between the phones on either side.
    ///
    /// This follows the timing, pitch and breath only, as
    /// [`FollowOptions::timing_only`]; [`follow_with`](Self::follow_with)
    /// takes the speaker's voice quality and vowels too.
    ///
    /// # Errors
    ///
    /// When the schedule's phones are not the utterance's, or it has no
    /// pitch track.
    pub fn follow(
        &mut self,
        schedule: &Schedule,
        mode: PitchMode,
    ) -> Result<FollowReport, ScheduleError> {
        self.follow_with(schedule, &FollowOptions::timing_only(mode))
    }

    /// [`follow`](Self::follow), and within the limits of `options`:
    ///
    /// * **tenseness:** each voiced sonorant's tenseness is scaled by its
    ///   phone's [`PhoneSpan::tenseness`] (the obstruents keep the rules');
    /// * **vowels:** a vowel with [`PhoneSpan::formants`] is given a shape
    ///   solved through the babble map ([`inverse::solve_for`], starting
    ///   from the voice's own shape for it, for the voice's tract length)
    ///   for the voice's formants there scaled by those shares, rounding
    ///   and all; the consonants made on its body (see `lean_on_vowels`)
    ///   lean on that shape too;
    /// * **speed:** a phone the schedule makes shorter than the rules do
    ///   has its articulators moved faster by the rules' duration over the
    ///   schedule's, up to [`FollowOptions::max_speed`], so the tongue gets
    ///   to its shapes in the time there is. A phone given more time than
    ///   the rules keeps the voice's speed.
    ///
    /// # Errors
    ///
    /// When the schedule's phones are not the utterance's, or it has no
    /// pitch track.
    #[allow(
        clippy::too_many_lines,
        clippy::many_single_char_names,
        clippy::cast_precision_loss
    )]
    pub fn follow_with(
        &mut self,
        schedule: &Schedule,
        options: &FollowOptions,
    ) -> Result<FollowReport, ScheduleError> {
        let mode = options.pitch;
        if schedule.pitch.is_empty() || schedule.pitch_hop <= 0.0 || !schedule.pitch_hop.is_finite()
        {
            return Err(ScheduleError::NoPitch);
        }
        let n = self
            .phones
            .iter()
            .filter(|&&p| p != NO_PHONE)
            .map(|&p| p as usize + 1)
            .max()
            .unwrap_or(0);
        if n != schedule.phones.len() {
            return Err(ScheduleError::PhoneCount {
                utterance: n,
                schedule: schedule.phones.len(),
            });
        }
        for (seg, &p) in self.segments.iter().zip(&self.phones) {
            if p == NO_PHONE {
                continue;
            }
            if let Some(q) = schedule.phones[p as usize].phoneme
                && q != seg.phoneme
            {
                return Err(ScheduleError::Phoneme {
                    index: p as usize,
                    utterance: seg.phoneme,
                    schedule: q,
                });
            }
        }
        // The voice's own pitch: the median of its rule contour over its
        // voiced sounds, time-weighted.
        let shift = match mode {
            PitchMode::Speaker => 0.0,
            PitchMode::Voice => {
                let mut voiced: Vec<(f32, f32)> = self
                    .segments
                    .iter()
                    .zip(&self.phones)
                    .filter(|(s, p)| s.voiced && **p != NO_PHONE)
                    .map(|(s, _)| {
                        (
                            ln(s.pitch.0.max(1.0) * s.pitch.1.max(1.0)) * 0.5,
                            s.duration,
                        )
                    })
                    .collect();
                voiced.sort_by(|a, b| a.0.total_cmp(&b.0));
                let half = 0.5 * voiced.iter().map(|v| v.1).sum::<f32>();
                let mut acc = 0.0;
                let own = voiced
                    .iter()
                    .find(|v| {
                        acc += v.1;
                        acc >= half
                    })
                    .map(|v| v.0);
                match own {
                    Some(own) if schedule.median_hz > 0.0 => {
                        12.0 * (own - ln(schedule.median_hz)) / core::f32::consts::LN_2
                    }
                    _ => 0.0,
                }
            }
        };
        let mut report = FollowReport {
            shift,
            ..FollowReport::default()
        };

        // Each phone's duration: the speaker's, and half of any pause of
        // his the utterance has no pause for on either side.
        let mut target: Vec<f32> = schedule
            .phones
            .iter()
            .map(|p| (p.end - p.start).max(0.0))
            .collect();
        let pause_between = |k: usize| {
            // Whether a pause segment stands between phone k and k + 1.
            let last = self.phones.iter().rposition(|&p| p as usize == k);
            last.and_then(|i| self.phones.get(i + 1))
                .is_some_and(|&p| p == NO_PHONE)
        };
        for k in 0..n.saturating_sub(1) {
            let gap = schedule.gap_after(k).max(0.0);
            if gap > 0.0 && !pause_between(k) {
                target[k] += 0.5 * gap;
                target[k + 1] += 0.5 * gap;
            }
        }

        // Retime, run by run of segments of one phone (or of one pause).
        let floor_of = |s: &super::Segment| {
            let closure = !s.voiced
                && matches!(s.phoneme, Phoneme::P | Phoneme::T | Phoneme::K)
                && s.articulation
                    .constriction
                    .is_some_and(|c| c.diameter <= 0.3);
            if closure {
                self.closure_floor + CLOSURE_MARGIN
            } else {
                MIN_SEGMENT
            }
        };
        let mut i = 0;
        let mut new = Vec::with_capacity(self.segments.len());
        let mut speed = alloc::vec![1.0; self.segments.len()];
        report.fastest = 1.0;
        while i < self.segments.len() {
            let p = self.phones[i];
            let mut j = i + 1;
            while j < self.segments.len() && self.phones[j] == p {
                j += 1;
            }
            let rule: Vec<f32> = self.segments[i..j].iter().map(|s| s.duration).collect();
            if p == NO_PHONE {
                // A pause: the speaker's, between the phones either side; the
                // first and last keep theirs.
                let before = self.phones[..i].iter().rev().find(|&&q| q != NO_PHONE);
                let after = self.phones[j..].iter().find(|&&q| q != NO_PHONE);
                match (before, after) {
                    (Some(&b), Some(_)) => {
                        let total = schedule.gap_after(b as usize).max(MIN_PAUSE);
                        // The lungs' lead-in before a phrase that opens on
                        // an obstruent is the last part; it keeps its length
                        // up to half the pause.
                        let mut d = rule.clone();
                        if d.len() > 1 {
                            let lead = d[d.len() - 1].min(0.5 * total);
                            let rest: f32 = d[..d.len() - 1].iter().sum();
                            let k = d.len() - 1;
                            for x in &mut d[..k] {
                                *x = if rest > 0.0 {
                                    *x * (total - lead) / rest
                                } else {
                                    (total - lead) / k as f32
                                };
                            }
                            d[k] = lead;
                        } else {
                            d[0] = total;
                        }
                        new.extend(d);
                    }
                    _ => new.extend(rule),
                }
            } else {
                let floor: Vec<f32> = self.segments[i..j].iter().map(floor_of).collect();
                let had: f32 = rule.iter().sum();
                let want = target[p as usize].max(options.shortest * had);
                let least: f32 = floor.iter().sum();
                if want < least {
                    report.too_short += 1;
                    report.added += least - want;
                }
                let got = share_out(&rule, &floor, want);
                // Shorter than the rules: the articulators get there faster.
                let has: f32 = got.iter().sum();
                if options.max_speed > 1.0 && has > 0.0 && had > SPED_UP * has {
                    let f = (had / has).min(options.max_speed);
                    speed[i..j].fill(f);
                    report.sped += 1;
                    report.fastest = report.fastest.max(f);
                }
                new.extend(got);
            }
            i = j;
        }
        self.followed.speed = if report.sped > 0 { speed } else { Vec::new() };
        for (s, &d) in self.segments.iter_mut().zip(&new) {
            let f = if s.duration > 0.0 {
                d / s.duration
            } else {
                1.0
            };
            let g = &mut s.gesture;
            g.gap_opening = (g.gap_opening.0 * f, g.gap_opening.1 * f);
            if g.hold >= 0.5 * s.duration {
                // The breath after a release: its voice onset time.
                g.hold *= f;
            }
            if g.transition > 0.0 {
                g.transition = (g.transition * f).min(d);
            }
            s.duration = d;
        }

        // Where every phone now is, on the utterance's clock.
        let mut spans = alloc::vec![(f32::NAN, f32::NAN); n];
        let mut t = 0.0;
        let mut starts = Vec::with_capacity(self.segments.len());
        for (s, &p) in self.segments.iter().zip(&self.phones) {
            starts.push(t);
            if p != NO_PHONE {
                let span = &mut spans[p as usize];
                if span.0.is_nan() {
                    span.0 = t;
                }
                span.1 = t + s.duration;
            }
            t += s.duration;
        }
        let total = t;

        // The utterance's clock to the recording's: phone onto phone, and
        // linearly through the pauses between.
        let to_recording = |tau: f32| -> f32 {
            let k = spans.partition_point(|s| s.1 < tau);
            if k >= n {
                return schedule.phones[n - 1].end + (tau - spans[n - 1].1);
            }
            let (a, b) = spans[k];
            let ph = &schedule.phones[k];
            if tau >= a {
                let x = if b > a { (tau - a) / (b - a) } else { 0.0 };
                return lerp(ph.start, ph.end, x);
            }
            if k == 0 {
                return ph.start - (a - tau);
            }
            // In the pause before phone k.
            let (_, pb) = spans[k - 1];
            let prev = schedule.phones[k - 1].end;
            let x = if a > pb { (tau - pb) / (a - pb) } else { 1.0 };
            lerp(prev, ph.start, x)
        };
        let shift = powf(2.0, report.shift / 12.0);
        let mut last_good = schedule.median_hz.max(1.0);
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let steps = (total / TRACK_HOP) as usize + 2;
        let mut track = Vec::with_capacity(steps);
        #[allow(clippy::cast_precision_loss)]
        for q in 0..steps {
            let hz = schedule.pitch_at(to_recording(q as f32 * TRACK_HOP));
            if hz.is_finite() && hz > 0.0 {
                last_good = hz;
            }
            track.push(last_good * shift);
        }
        self.set_pitch_track(TRACK_HOP, track);
        let pitches: Vec<(f32, f32)> = self
            .segments
            .iter()
            .zip(&starts)
            .map(|(s, &a)| {
                (
                    self.track_at(a).unwrap_or(s.pitch.0),
                    self.track_at(a + s.duration).unwrap_or(s.pitch.1),
                )
            })
            .collect();
        for (s, p) in self.segments.iter_mut().zip(pitches) {
            s.pitch = p;
        }

        // The breath: each phone's share at its middle, in a line between.
        let mids: Vec<(f32, f32)> = spans
            .iter()
            .zip(&schedule.phones)
            .map(|(&(a, b), ph)| {
                let share = if ph.breath.is_finite() {
                    ph.breath
                } else {
                    1.0
                };
                let (lo, hi) = options.breath_range;
                let limited = share.clamp(lo, hi.max(lo));
                if (limited - share).abs() > 1e-6 {
                    report.breath_limited += 1;
                }
                (f32::midpoint(a, b), limited)
            })
            .collect();
        let breath_at = |tau: f32| -> f32 {
            let k = mids.partition_point(|m| m.0 < tau);
            match (k.checked_sub(1).map(|i| mids[i]), mids.get(k)) {
                (Some(a), Some(b)) if b.0 > a.0 => lerp(a.1, b.1, (tau - a.0) / (b.0 - a.0)),
                (Some(a), _) => a.1,
                (_, Some(b)) => b.1,
                _ => 1.0,
            }
        };
        for (k, s) in self.segments.iter_mut().enumerate() {
            let a = starts[k];
            let b = a + s.duration;
            let (from, to) = s.gesture.pressure_drop;
            s.gesture.pressure_drop = (
                1.0 - (1.0 - from) * breath_at(a),
                1.0 - (1.0 - to) * breath_at(b),
            );
        }

        // The voice quality: each voiced sonorant's tenseness by its phone's
        // share. Not the obstruents': a voiced closure needs folds held
        // firmly enough to go on vibrating against the pressure building
        // in the mouth (with a loose baseline and its breath at the bottom
        // of its range, a /b/ or /g/ between vowels lost most of its voice
        // bar), and the speaker's voice quality is measured on the
        // sonorants anyway.
        let (lo, hi) = options.tenseness_range;
        let mut limited = alloc::vec![false; n];
        for (s, &p) in self.segments.iter_mut().zip(&self.phones) {
            let sonorant = matches!(
                s.phoneme.manner(),
                Manner::Vowel | Manner::Diphthong | Manner::Nasal | Manner::Approximant
            );
            if p == NO_PHONE || !s.voiced || !sonorant {
                continue;
            }
            let share = schedule.phones[p as usize].tenseness;
            let share = if share.is_finite() { share } else { 1.0 };
            let f = share.clamp(lo, hi.max(lo));
            limited[p as usize] |= (f - share).abs() > 1e-6;
            s.tenseness = (s.tenseness * f).clamp(0.0, 1.0);
        }
        report.tenseness_limited = limited.iter().filter(|&&l| l).count();

        self.shape_vowels(schedule, options, &mut report);
        report.spans = spans;
        Ok(report)
    }

    /// The speaker's vowels (see [`follow_with`](Self::follow_with)): a
    /// shape solved for each vowel with [`PhoneSpan::formants`], held by
    /// the vowel's segments and by the consonants made on its body.
    fn shape_vowels(
        &mut self,
        schedule: &Schedule,
        options: &FollowOptions,
        report: &mut FollowReport,
    ) {
        self.followed.shape = Vec::new();
        let range = options.formant_range.clamp(0.0, 0.5);
        if range <= 0.0 {
            return;
        }
        let length = options.tract_length;
        let mut shapes: Vec<Option<(VowelKey, Shape)>> = alloc::vec![None; self.segments.len()];
        let mut solved: Vec<Option<(VowelKey, Shape)>> = alloc::vec![None; schedule.phones.len()];
        let mut limited = 0;
        for (i, s) in self.segments.iter().enumerate() {
            let p = self.phones[i];
            if p == NO_PHONE || s.phoneme.manner() != Manner::Vowel {
                continue;
            }
            // The vowel's own shape, not its glide towards schwa.
            let key = VowelKey::Vowel(s.phoneme);
            if s.articulation.vowel != Some(key) {
                continue;
            }
            let Some((r1, r2)) = schedule.phones[p as usize].formants else {
                continue;
            };
            if solved[p as usize].is_none() {
                let Some(own) = key.articulation().and_then(|a| a.physical_shape(length)) else {
                    continue;
                };
                let Some(native) = inverse::predict(&own) else {
                    continue;
                };
                let at = inverse::formants_at_length(native, length);
                let clip = |r: f32| {
                    let r = if r.is_finite() && r > 0.0 { r } else { 1.0 };
                    let c = r.clamp(1.0 - range, 1.0 + range);
                    ((c - r).abs() > 1e-6, c)
                };
                let ((l1, r1), (l2, r2)) = (clip(r1), clip(r2));
                limited += usize::from(l1 || l2);
                let shape =
                    inverse::solve_for(at[0] * r1, at[1] * r2, Some(at[2]), Some(&own), length);
                solved[p as usize] = Some((key, shape));
                report.shaped += 1;
            }
            shapes[i] = solved[p as usize];
        }
        report.formants_limited = limited;
        if report.shaped == 0 {
            return;
        }
        // The consonants made on a vowel's body lean on its shape: the
        // vowel they lead into, or the one before.
        let is_vowel =
            |s: &super::Segment| matches!(s.phoneme.manner(), Manner::Vowel | Manner::Diphthong);
        for i in 0..self.segments.len() {
            let s = &self.segments[i];
            let Some(key) = s.articulation.vowel else {
                continue;
            };
            if is_vowel(s) || s.phoneme == Phoneme::Sil {
                continue;
            }
            let next = self.segments[i + 1..]
                .iter()
                .position(is_vowel)
                .map(|j| i + 1 + j);
            let prev = self.segments[..i].iter().rposition(is_vowel);
            let on = [next, prev]
                .into_iter()
                .flatten()
                .find(|&j| self.segments[j].articulation.vowel == Some(key));
            if let Some(j) = on {
                shapes[i] = shapes[j];
            }
        }
        self.followed.shape = shapes;
    }

    /// Where each phone of the text starts and ends on the utterance's
    /// clock (the sum of the segments' durations before it), seconds; a
    /// phone the utterance has no segment for is `(NaN, NaN)`. Empty for
    /// hand-made segments.
    #[must_use]
    pub fn phone_spans(&self) -> Vec<(f32, f32)> {
        let n = self
            .phones
            .iter()
            .filter(|&&p| p != NO_PHONE)
            .map(|&p| p as usize + 1)
            .max()
            .unwrap_or(0);
        let mut spans = alloc::vec![(f32::NAN, f32::NAN); n];
        let mut t = 0.0;
        for (s, &p) in self.segments.iter().zip(&self.phones) {
            if p != NO_PHONE {
                let span = &mut spans[p as usize];
                if span.0.is_nan() {
                    span.0 = t;
                }
                span.1 = t + s.duration;
            }
            t += s.duration;
        }
        spans
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Prosody, phonemize};
    use super::*;
    use crate::Engine;

    fn physical() -> Prosody {
        Prosody {
            engine: Engine::Physical,
            ..Prosody::default()
        }
    }

    /// A schedule for `text` with every phone `d` seconds long, a pause of
    /// `pause` after phone `at`, and a pitch gliding from 100 to 200 Hz.
    #[allow(clippy::cast_precision_loss)]
    fn schedule(text: &str, d: f32, at: usize, pause: f32) -> Schedule {
        let n = phonemize(text)
            .iter()
            .filter(|t| matches!(t, Token::Phone { .. }))
            .count();
        let mut t = 0.5;
        let phones = (0..n)
            .map(|k| {
                let s = PhoneSpan::new(None, t, t + d);
                t += d + if k == at { pause } else { 0.0 };
                s
            })
            .collect();
        let steps = 400;
        Schedule {
            phones,
            pitch_start: 0.0,
            pitch_hop: 0.01,
            pitch: (0..steps).map(|q| 100.0 + q as f32 * 0.25).collect(),
            median_hz: 150.0,
        }
    }

    #[test]
    fn phones_get_the_schedule_durations() {
        let text = "she sells sea shells by the sea shore";
        let s = schedule(text, 0.09, 99, 0.0);
        let mut u = Utterance::from_tokens(&s.phrase(&phonemize(text)), &physical());
        let report = u.follow(&s, PitchMode::Speaker).unwrap();
        assert_eq!(report.too_short, 0);
        for (k, &(a, b)) in u.phone_spans().iter().enumerate() {
            assert!((b - a - 0.09).abs() < 1e-4, "phone {k}: {}", b - a);
        }
        // The pitch follows the speaker's, phone onto phone.
        let spans = u.phone_spans();
        let mid = f32::midpoint(spans[10].0, spans[10].1);
        let want = s.pitch_at(0.5 + 10.5 * 0.09);
        let got = u.track_at(mid).unwrap();
        assert!((got - want).abs() < 0.5, "{got} {want}");
    }

    #[test]
    fn a_closure_keeps_its_floor() {
        let text = "a cat ate it";
        let s = schedule(text, 0.012, 99, 0.0);
        let mut u = Utterance::from_tokens(&phonemize(text), &physical());
        let report = u.follow(&s, PitchMode::Speaker).unwrap();
        assert!(report.too_short > 0);
        let floor = physical().tuning.closure_floor;
        assert!(
            u.segments()
                .iter()
                .filter(|s| s.phoneme == Phoneme::T && !s.voiced)
                .any(|s| s.duration >= floor)
        );
    }

    #[test]
    fn pauses_move_to_where_the_speaker_paused() {
        let text = "the birch canoe slid on the smooth planks. glue the sheet";
        let tokens = phonemize(text);
        // A pause after "canoe" (its last phone is 8) and none at the full
        // stop (after phone 26).
        let s = schedule(text, 0.08, 8, 0.3);
        let phrased = s.phrase(&tokens);
        let breaks = |t: &[Token]| {
            t.iter()
                .enumerate()
                .filter(|(_, t)| matches!(t, Token::Break(_)))
                .map(|(i, _)| {
                    t[..i]
                        .iter()
                        .filter(|t| matches!(t, Token::Phone { .. }))
                        .count()
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(breaks(&tokens), [27]);
        assert_eq!(breaks(&phrased), [9]);
        let mut u = Utterance::from_tokens(&phrased, &physical());
        u.follow(&s, PitchMode::Speaker).unwrap();
        let spans = u.phone_spans();
        let gap = spans[9].0 - spans[8].1;
        assert!((gap - 0.3).abs() < 1e-4, "{gap}");
        let none = spans[27].0 - spans[26].1;
        assert!(none.abs() < 1e-4, "{none}");
    }

    #[test]
    fn the_voice_mode_moves_the_contour_to_the_voice() {
        let text = "glue the sheet to the dark blue background";
        let s = schedule(text, 0.08, 99, 0.0);
        let mut u = Utterance::from_tokens(&phonemize(text), &physical());
        let r = u.follow(&s, PitchMode::Voice).unwrap();
        // The rule voice speaks near 110 Hz, the schedule's median is 150.
        assert!(r.shift < -3.0 && r.shift > -8.0, "{}", r.shift);
    }

    #[test]
    fn a_schedule_for_other_phones_is_refused() {
        let s = schedule("one two three", 0.08, 99, 0.0);
        let mut u = Utterance::from_tokens(&phonemize("four five"), &physical());
        assert!(matches!(
            u.follow(&s, PitchMode::Speaker),
            Err(ScheduleError::PhoneCount { .. })
        ));
    }

    #[test]
    fn share_out_keeps_proportions_and_floors() {
        let d = share_out(&[0.05, 0.05], &[0.03, 0.006], 0.04);
        assert!(
            (d[0] - 0.03).abs() < 1e-6 && (d[1] - 0.01).abs() < 1e-6,
            "{d:?}"
        );
        let d = share_out(&[0.02, 0.06], &[0.006, 0.006], 0.16);
        assert!(
            (d[0] - 0.04).abs() < 1e-6 && (d[1] - 0.12).abs() < 1e-6,
            "{d:?}"
        );
        let d = share_out(&[0.02, 0.06], &[0.03, 0.006], 0.01);
        assert_eq!(d, [0.03, 0.006]);
    }

    fn everything(tract_length: f32) -> FollowOptions {
        FollowOptions::new(PitchMode::Speaker, &Tuning::DEFAULT, tract_length)
    }

    #[test]
    fn tenseness_follows_the_schedule_within_its_limits() {
        let text = "she sells sea shells";
        let mut s = schedule(text, 0.09, 99, 0.0);
        s.phones[1].tenseness = 1.3; // the /i/ of "she"
        s.phones[3].tenseness = 0.95; // the /e/ of "sells"
        s.phones[4].tenseness = 0.5; // its /l/
        let mut plain = Utterance::from_tokens(&phonemize(text), &physical());
        plain.follow(&s, PitchMode::Speaker).unwrap();
        let mut u = Utterance::from_tokens(&phonemize(text), &physical());
        let r = u.follow_with(&s, &everything(1.0)).unwrap();
        assert_eq!(r.tenseness_limited, 2);
        let t = Tuning::DEFAULT;
        for (k, want) in [
            (1, t.follow_tenseness_high),
            (3, 0.95),
            (4, t.follow_tenseness_low),
            (5, 1.0),
        ] {
            for (i, (a, b)) in plain.segments.iter().zip(&u.segments).enumerate() {
                if plain.phone_of(i) == Some(k) && a.voiced {
                    assert!((b.tenseness - a.tenseness * want).abs() < 1e-5, "phone {k}");
                }
            }
        }
        // the voiceless /s/ keeps its tenseness
        let s0 = plain.segments.iter().position(|x| x.phoneme == Phoneme::S);
        assert!(
            s0.is_some_and(
                |i| (plain.segments[i].tenseness - u.segments[i].tenseness).abs() < 1e-9
            )
        );
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn phones_shorter_than_the_rules_move_the_tongue_faster() {
        let text = "the juice of lemons";
        let rule = Utterance::from_tokens(&phonemize(text), &physical()).phone_spans();
        let mut s = schedule(text, 0.1, 99, 0.0);
        // phone 3 at 80 % of its rule length, phone 4 at a third, phone 5 at
        // twice: sped up by 1.25, by the most allowed, and not at all
        let mut t = 0.5;
        for (k, p) in s.phones.iter_mut().enumerate() {
            let d = rule[k].1 - rule[k].0;
            let d = match k {
                3 => 0.8 * d,
                4 => d / 3.0,
                5 => 2.0 * d,
                _ => d,
            };
            (p.start, p.end) = (t, t + d);
            t += d;
        }
        let mut u = Utterance::from_tokens(&phonemize(text), &physical());
        let mut options = everything(1.0);
        options.shortest = 0.0;
        let r = u.follow_with(&s, &options).unwrap();
        let max = Tuning::DEFAULT.follow_speed_max;
        for i in 0..u.segments.len() {
            let want = match u.phone_of(i) {
                Some(3) => 1.25,
                Some(4) => max,
                _ => 1.0,
            };
            let got = u.followed.speed_of(i);
            assert!((got - want).abs() < 0.02, "segment {i}: {got} {want}");
        }
        assert!(r.sped == 2 && (r.fastest - max).abs() < 1e-6);
        // With a floor at 70 % of the rules the third-length phone gets 70 %
        // of its rule duration, and the speed that goes with it.
        let mut u = Utterance::from_tokens(&phonemize(text), &physical());
        let mut options = everything(1.0);
        options.shortest = 0.7;
        u.follow_with(&s, &options).unwrap();
        let spans = u.phone_spans();
        let d = spans[4].1 - spans[4].0;
        assert!((d / (rule[4].1 - rule[4].0) - 0.7).abs() < 0.01, "{d}");
        let i = (0..u.segments.len())
            .find(|&i| u.phone_of(i) == Some(4))
            .unwrap();
        assert!((u.followed.speed_of(i) - 1.0 / 0.7).abs() < 0.02);
        // and never with the speed left alone
        let mut u = Utterance::from_tokens(&phonemize(text), &physical());
        u.follow(&s, PitchMode::Speaker).unwrap();
        assert!(u.followed.speed.iter().all(|&f| (f - 1.0).abs() < 1e-9));
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn a_vowel_takes_the_speakers_formants_and_its_consonants_lean_on_them() {
        let text = "a big dog";
        let length = 0.9;
        let mut s = schedule(text, 0.1, 99, 0.0);
        let ih = 2; // AX B IH G D AO G
        s.phones[ih].formants = Some((1.1, 0.92));
        s.phones[5].formants = Some((1.4, 1.0)); // the /ɔ/: F1 limited to +15 %
        let mut u = Utterance::from_tokens(&phonemize(text), &physical());
        let r = u.follow_with(&s, &everything(length)).unwrap();
        assert_eq!((r.shaped, r.formants_limited), (2, 1));
        let key = VowelKey::Vowel(Phoneme::Ih);
        let vowel = (0..u.segments.len())
            .find(|&i| u.phone_of(i) == Some(ih) && u.segments[i].articulation.vowel == Some(key))
            .unwrap();
        let (k, shape) = u.followed.shape_of(vowel).unwrap();
        assert_eq!(k, key);
        let own = key.articulation().unwrap().physical_shape(length).unwrap();
        let f = |sh: &Shape| inverse::formants_at_length(inverse::predict(sh).unwrap(), length);
        let (a, b) = (f(&own), f(&shape));
        assert!((b[0] / a[0] / 1.1 - 1.0).abs() < 0.04, "{a:?} {b:?}");
        assert!((b[1] / a[1] / 0.92 - 1.0).abs() < 0.04, "{a:?} {b:?}");
        // the /b/ and /g/ around it are made on that shape
        for i in 0..u.segments.len() {
            if matches!(u.phone_of(i), Some(1 | 3)) && u.segments[i].articulation.vowel == Some(key)
            {
                assert_eq!(u.followed.shape_of(i), Some((key, shape)), "segment {i}");
            }
        }
        let ao = (0..u.segments.len())
            .find(|&i| u.phone_of(i) == Some(5))
            .unwrap();
        let (_, shape) = u.followed.shape_of(ao).unwrap();
        let own = VowelKey::Vowel(Phoneme::Ao)
            .articulation()
            .unwrap()
            .physical_shape(length)
            .unwrap();
        let (a, b) = (f(&own), f(&shape));
        assert!(b[0] / a[0] < 1.15 * 1.04, "{a:?} {b:?}");

        // The speaker holds the shape while the vowel is spoken.
        let mut voice = crate::Trombone::new(48_000.0);
        voice.set_engine(Engine::Physical);
        voice.set_tract_length(length);
        super::super::Speaker::prepare(&mut voice);
        let mut speaker = super::super::Speaker::new();
        let want = u.followed.shape_of(vowel).unwrap().1;
        speaker.speak(u);
        let mut seen = false;
        for _ in 0..48_000 * 3 {
            speaker.process(&mut voice);
            // Early in the vowel: by its end the /g/ after it is on its way.
            if !seen && speaker.current().is_some_and(|(i, _)| i == vowel) {
                let (ti, td) = voice.tongue();
                assert!(
                    (ti - want.tongue_index).abs() < 1e-3
                        && (td - want.tongue_diameter).abs() < 1e-3,
                    "{:?} {want:?}",
                    voice.tongue()
                );
                seen = true;
            }
            if !speaker.is_speaking() {
                break;
            }
        }
        assert!(seen);
    }
}
