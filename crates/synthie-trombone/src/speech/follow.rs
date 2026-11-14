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
use crate::Phoneme;
use crate::math::{lerp, ln, powf};
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
/// The range of a phone's breath, a share of the voice's lung pressure.
const BREATH_RANGE: (f32, f32) = (0.5, 1.5);
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
    /// (`1`), limited to 0.5 to 1.5.
    pub breath: f32,
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
    /// Breath shares that had to be limited to 0.5 to 1.5.
    pub breath_limited: usize,
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
        let held: f32 = floor.iter().zip(&fixed).filter(|(_, f)| **f).map(|(m, _)| m).sum();
        let free: f32 = rule.iter().zip(&fixed).filter(|(_, f)| !**f).map(|(r, _)| r).sum();
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
    /// # Errors
    ///
    /// When the schedule's phones are not the utterance's, or it has no
    /// pitch track.
    #[allow(clippy::too_many_lines)]
    pub fn follow(
        &mut self,
        schedule: &Schedule,
        mode: PitchMode,
    ) -> Result<FollowReport, ScheduleError> {
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
        let mut report = FollowReport::default();

        // The voice's own pitch: the median of its rule contour over its
        // voiced sounds, time-weighted.
        report.shift = match mode {
            PitchMode::Speaker => 0.0,
            PitchMode::Voice => {
                let mut voiced: Vec<(f32, f32)> = self
                    .segments
                    .iter()
                    .zip(&self.phones)
                    .filter(|(s, p)| s.voiced && **p != NO_PHONE)
                    .map(|(s, _)| (ln(s.pitch.0.max(1.0) * s.pitch.1.max(1.0)) * 0.5, s.duration))
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
                let want = target[p as usize];
                let least: f32 = floor.iter().sum();
                if want < least {
                    report.too_short += 1;
                    report.added += least - want;
                }
                new.extend(share_out(&rule, &floor, want));
            }
            i = j;
        }
        for (s, &d) in self.segments.iter_mut().zip(&new) {
            let f = if s.duration > 0.0 { d / s.duration } else { 1.0 };
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
                let share = if ph.breath.is_finite() { ph.breath } else { 1.0 };
                let limited = share.clamp(BREATH_RANGE.0, BREATH_RANGE.1);
                if (limited - share).abs() > 1e-6 {
                    report.breath_limited += 1;
                }
                (0.5 * (a + b), limited)
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
        report.spans = spans;
        Ok(report)
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
                let s = PhoneSpan {
                    phoneme: None,
                    start: t,
                    end: t + d,
                    breath: 1.0,
                };
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
        let mid = 0.5 * (spans[10].0 + spans[10].1);
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
        assert!((d[0] - 0.03).abs() < 1e-6 && (d[1] - 0.01).abs() < 1e-6, "{d:?}");
        let d = share_out(&[0.02, 0.06], &[0.006, 0.006], 0.16);
        assert!((d[0] - 0.04).abs() < 1e-6 && (d[1] - 0.12).abs() < 1e-6, "{d:?}");
        let d = share_out(&[0.02, 0.06], &[0.03, 0.006], 0.01);
        assert_eq!(d, [0.03, 0.006]);
    }
}
