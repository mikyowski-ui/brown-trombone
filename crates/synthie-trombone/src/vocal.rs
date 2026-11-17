//! Non-verbal vocalizations (wave 3, stage 3B): a sigh, a laugh, a
//! chuckle, a throat clearing and a cough, each a macro that lays gestures
//! on the tiers of the gesture score for one event, with the lungs' push as
//! a gesture of its own (see [`var::glottis::PUSH`]).
//!
//! What each is, from what people do:
//!
//! - **Laugh** ([`Vocalization::Laugh`]): a quick breath in, then a train
//!   of calls ("ha") at [`ArticulationConfig::laugh_rate`] per second
//!   (4.37 in Bachorowski, Smoski & Owren 2001), each some
//!   [`ArticulationConfig::laugh_voiced`] of voice (0.11 s in voiced,
//!   open-mouth calls there, the first about twice as long) followed by
//!   breath through the open glottis, the intervals growing towards the end
//!   of the bout. Each call is a pulse of subglottal pressure and a burst of
//!   the folds' adductors at about 5 Hz (Luschei et al. 2006), on a mouth
//!   held open and not re-articulated; the pulses fall from
//!   [`AnatomyConfig::laugh_pressure`] to six tenths of it along the bout as
//!   the lungs empty, and the pitch sits well above speech (male laughter
//!   282 Hz against 120 in speech in the same study: here 1.8 times the
//!   voice's pitch) and falls within each call and a little along the
//!   bout (Bachorowski et al. found no fall of the mean along a bout; the
//!   lungs' falling push lowers the level here, the pitch falls by an
//!   eighth).
//! - **Chuckle** ([`Vocalization::Chuckle`]): a short, closed-mouth bout of
//!   three weaker, shorter calls with the lips shut and the voice through
//!   the nose ("hm-hm-hm"; closed-mouth calls of men 216 Hz in the same
//!   study: here 1.5 times the voice's pitch).
//! - **Cough** ([`Vocalization::Cough`]): an inspiration (3A's breath in),
//!   a compression phase of some 0.2 s against the folds pressed shut
//!   ([`var::glottis::SEAL`]) and the tube above them drawn in, while the
//!   lungs build [`AnatomyConfig::cough_pressure`]; then the folds are
//!   thrown wide open: a burst of air (the expulsive phase, 30 to 50 ms at
//!   up to 12 L/s in people, then 0.2 to 0.5 s of lesser flow), its noise
//!   the jets' own, and a short voiced, breathy tail ("uh") as the push
//!   dies away.
//! - **Throat clearing** ([`Vocalization::ClearThroat`]): two shorter,
//!   weaker coughs, more voiced, with the epilaryngeal tube narrowed
//!   ([`var::larynx::SQUEEZE`]) and the folds pressed (tenseness 0.9),
//!   low in pitch, the second weaker.
//! - **Sigh** ([`Vocalization::Sigh`]): a deep breath in (twice the depth of
//!   a breath between sentences) and a long, breathy exhalation, voiced
//!   with the glottis held open (high airflow, soft adduction), its pitch
//!   falling from a quarter above the voice's to a fifth below and the
//!   tract relaxing from an open vowel towards schwa, ending in breath.
//!
//! The model has no false folds: the closure of a cough is the true folds
//! pressed past touching (their pressed edges feel none of the pressure
//! behind them, so they hold at any), with the epilaryngeal tube drawn in
//! above them; the lungs are a pressure source that follows its target
//! within 30 ms. Everything here is gestures, and the folds, the jets and
//! the tract do the rest: the burst of a cough, the breath between a
//! laugh's calls and the hum of a chuckle come out of the model's physics.
//!
//! [`Trombone::vocalize`](crate::Trombone::vocalize) plays one on a voice;
//! speech plays a tag in its text (`"Well, [sigh] I suppose so."`).

use crate::gesture::{Gesture, LAYER_VOCAL, Target, Tier, Window, var};
use crate::tuning::Tuning;

/// A non-verbal vocalization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Vocalization {
    /// A deep breath in and a long, breathy, falling breath out.
    Sigh,
    /// A train of voiced calls on pulses of pressure ("ha ha ha").
    Laugh,
    /// A short, closed-mouth laugh through the nose.
    Chuckle,
    /// Two short, weak, voiced coughs with the throat narrowed.
    ClearThroat,
    /// A breath in, the folds shut against the lungs' pressure, a burst.
    Cough,
}

impl Vocalization {
    /// Every vocalization, in the order of their tags in the docs.
    pub const ALL: [Self; 5] = [
        Self::Sigh,
        Self::Laugh,
        Self::Chuckle,
        Self::ClearThroat,
        Self::Cough,
    ];

    /// Its tag, as text writes it in square brackets: `sigh`, `laugh`,
    /// `chuckle`, `clear-throat`, `cough`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sigh => "sigh",
            Self::Laugh => "laugh",
            Self::Chuckle => "chuckle",
            Self::ClearThroat => "clear-throat",
            Self::Cough => "cough",
        }
    }

    /// The vocalization of a tag (exactly as [`name`](Self::name) gives it,
    /// lower case; anything else is `None`, so `[HH AX L OW1]` stays
    /// phonemes).
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|v| v.name() == name)
    }

    /// How many calls (or coughs) it makes at length 1.
    fn calls(self) -> usize {
        match self {
            Self::Laugh => 5,
            Self::Chuckle => 3,
            Self::ClearThroat => 2,
            Self::Sigh | Self::Cough => 1,
        }
    }
}

/// One vocalization with the numbers a caller sets: how long, how strong,
/// at what pitch.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Vocal {
    /// Which one.
    pub kind: Vocalization,
    /// How long, as a multiple of its own length (`0.5‥2`; `1`): a laugh
    /// of length 2 has twice the calls, a sigh draws and lets out its
    /// breath for twice as long.
    pub length: f32,
    /// How strong (`0.3‥2`; `1`): the lungs' pressure, and with it the
    /// level; a breath in this much deeper.
    pub strength: f32,
    /// The pitch it is made around, Hz (`0`: the voice's own, as it is when
    /// the event starts); a laugh's calls sit above it, a cough's tail near
    /// it, a sigh falls through it.
    pub pitch: f32,
}

impl Vocal {
    /// `kind` at its own length and strength, around the voice's pitch.
    #[must_use]
    pub const fn new(kind: Vocalization) -> Self {
        Self {
            kind,
            length: 1.0,
            strength: 1.0,
            pitch: 0.0,
        }
    }

    /// With length `x` (see [`length`](Self::length)).
    #[must_use]
    pub const fn with_length(mut self, x: f32) -> Self {
        self.length = x;
        self
    }

    /// With strength `x` (see [`strength`](Self::strength)).
    #[must_use]
    pub const fn with_strength(mut self, x: f32) -> Self {
        self.strength = x;
        self
    }

    /// Around pitch `hz` (see [`pitch`](Self::pitch)).
    #[must_use]
    pub const fn with_pitch(mut self, hz: f32) -> Self {
        self.pitch = hz;
        self
    }

    /// The same with its numbers in their ranges (anything not a number
    /// back to its default).
    #[must_use]
    pub fn clamped(self) -> Self {
        let within = |x: f32, lo: f32, hi: f32, d: f32| {
            if x.is_finite() { x.clamp(lo, hi) } else { d }
        };
        Self {
            kind: self.kind,
            length: within(self.length, 0.5, 2.0, 1.0),
            strength: within(self.strength, 0.3, 2.0, 1.0),
            pitch: within(self.pitch, 0.0, 1000.0, 0.0),
        }
    }

    /// How long it lasts on a voice of `tuning`, seconds (as
    /// [`Trombone::vocalize`](crate::Trombone::vocalize) plays it).
    #[must_use]
    pub fn seconds(&self, tuning: &Tuning) -> f32 {
        plan(self, 100.0, tuning).seconds
    }
}

impl From<Vocalization> for Vocal {
    fn from(kind: Vocalization) -> Self {
        Self::new(kind)
    }
}

/// A shape of the tract a vocalization holds, which the voice resolves
/// into gestures on its tract tiers as it resolves a vowel's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Posture {
    /// The mouth open, as for /ɑ/: a laugh, the start of a sigh.
    Open,
    /// Half open, as for /ʌ/: a cough.
    Mid,
    /// Relaxed, as for schwa: the end of a sigh, a throat clearing.
    Relaxed,
    /// The lips shut and the velum open, as for /m/: a chuckle's hum.
    Hum,
}

/// A posture held over a window, with its dominance.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Hold {
    pub posture: Posture,
    pub window: Window,
    pub dominance: f32,
}

/// The most gestures an event lays (a laugh of twelve calls).
pub(crate) const MOST: usize = 56;
/// The most postures it holds.
pub(crate) const MOST_HOLDS: usize = 3;

/// One event as gestures: on the glottal tier and the larynx's (laid
/// here), and the postures of the tract (resolved by the voice).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Plan {
    gestures: [Option<Gesture>; MOST],
    len: usize,
    holds: [Option<Hold>; MOST_HOLDS],
    held: usize,
    /// Seconds from its start to its end.
    pub seconds: f32,
}

impl Plan {
    const fn new() -> Self {
        Self {
            gestures: [None; MOST],
            len: 0,
            holds: [None; MOST_HOLDS],
            held: 0,
            seconds: 0.0,
        }
    }

    pub(crate) fn gestures(&self) -> impl Iterator<Item = &Gesture> {
        self.gestures[..self.len].iter().flatten()
    }

    pub(crate) fn holds(&self) -> impl Iterator<Item = &Hold> {
        self.holds[..self.held].iter().flatten()
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// A gesture on `tier` setting `sets`, from `start` to `end` with
    /// ramps in and out, at `dominance`.
    fn lay(
        &mut self,
        tier: Tier,
        sets: &[(usize, f32)],
        (start, end): (f32, f32),
        (ramp_in, ramp_out): (f32, f32),
        dominance: f32,
    ) {
        if self.len == MOST {
            return;
        }
        let mut target = Target::NONE;
        for &(v, x) in sets {
            target.set(v, x);
        }
        let g = Gesture::new(tier, target)
            .window(Window::between(start.max(0.0), end.max(start)).ramps(ramp_in, ramp_out))
            .dominance(dominance)
            .layer(LAYER_VOCAL);
        self.gestures[self.len] = Some(g);
        self.len += 1;
    }

    /// The glottal tier's `sets` over a window, winning over speech.
    fn glottis(&mut self, sets: &[(usize, f32)], window: (f32, f32), ramps: (f32, f32)) {
        self.lay(Tier::GlottalAdduction, sets, window, ramps, STRONG);
    }

    /// The same at the weight of a background the other gestures of the
    /// event lie over.
    fn glottis_under(&mut self, sets: &[(usize, f32)], window: (f32, f32), ramps: (f32, f32)) {
        self.lay(Tier::GlottalAdduction, sets, window, ramps, 1.0);
    }

    /// The lungs pushing `cm_h2o` over a window (its own gesture, so that
    /// the pressure leads or lags the voice).
    fn push(&mut self, cm_h2o: f32, window: (f32, f32), ramps: (f32, f32)) {
        self.glottis(&[(var::glottis::PUSH, cm_h2o)], window, ramps);
    }

    /// The pitch going from `from` to `to` Hz between `start` and `end`:
    /// two gestures crossfading (the blend of the two by their activations
    /// goes smoothly from one to the other), winning over speech.
    fn glide(&mut self, (from, to): (f32, f32), (start, end): (f32, f32), lead: f32) {
        let across = (end - start).max(0.01);
        let pitch = var::larynx::PITCH;
        self.lay(
            Tier::LarynxAndPitch,
            &[(pitch, from)],
            (start - lead, start),
            (0.005, across),
            STRONG,
        );
        self.lay(
            Tier::LarynxAndPitch,
            &[(pitch, to)],
            (start, end),
            (across, 0.02),
            STRONG,
        );
    }

    fn hold(
        &mut self,
        posture: Posture,
        (start, end): (f32, f32),
        ramps: (f32, f32),
        dominance: f32,
    ) {
        if self.held == MOST_HOLDS {
            return;
        }
        self.holds[self.held] = Some(Hold {
            posture,
            window: Window::between(start.max(0.0), end.max(start)).ramps(ramps.0, ramps.1),
            dominance,
        });
        self.held += 1;
    }

    /// A breath in of `depth` cmH2O from `start` for `seconds` (Stage 3A's
    /// [`Gesture::inhalation`]), the voice off and the glottis left to it.
    fn breathe_in(&mut self, depth: f32, start: f32, seconds: f32) {
        if self.len == MOST {
            return;
        }
        let mut g = Gesture::inhalation(depth, seconds, BREATH_RAMP)
            .dominance(STRONG)
            .layer(LAYER_VOCAL);
        g.window.start += start;
        g.window.end += start;
        self.gestures[self.len] = Some(g);
        self.len += 1;
        self.glottis(
            &[(var::glottis::VOICING, 0.0), (var::glottis::SPREAD, 0.0)],
            (start, start + seconds),
            (0.01, 0.02),
        );
    }
}

/// Dominance of a vocalization's gestures: they win over the speech's
/// (dominance 1, a consonant's own lips 1000) where both act.
const STRONG: f32 = 1000.0;
/// Ramps of a vocalization's breaths in, seconds (as `Trombone::breathe_in`).
const BREATH_RAMP: f32 = 0.03;
/// Seconds from a breath in letting go to the first sound after it (as
/// speech's turn, `speech::BREATH_TURN`).
const TURN: f32 = 0.1;
/// Glottal opening between a laugh's calls, cm² at native size: the
/// breath of an /h/ (an aspirated stop's is 0.25 to 0.4).
const LAUGH_BREATH: f32 = 0.3;
/// Opening held behind the vibrating folds in a laugh's call, cm²: a
/// breathy voice.
const LAUGH_SPREAD: f32 = 0.05;
/// Lung pressure between a laugh's calls, cmH2O: the breath that carries
/// the /h/.
const LAUGH_FLOOR: f32 = 2.5;
/// Lung pressure an event holds where none of its gestures push, cmH2O.
const LUNGS_HELD: f32 = 0.3;
/// How far past touching the folds are pressed in a cough's compression,
/// cm (a glottal attack presses them 0.04).
const COUGH_SEAL: f32 = 0.03;
/// Glottal opening the folds are thrown to as a cough bursts, cm².
const COUGH_OPENING: f32 = 0.8;

/// The gestures of `v` (its numbers clamped) around `pitch` Hz on a voice
/// of `tuning`, from its start.
#[allow(clippy::too_many_lines)]
pub(crate) fn plan(v: &Vocal, pitch: f32, tuning: &Tuning) -> Plan {
    use var::glottis::{PUSH, SEAL, SPREAD, TENSENESS, VOICING};
    let v = v.clamped();
    let (len, s) = (v.length, v.strength);
    let p0 = if v.pitch > 0.0 {
        v.pitch
    } else {
        pitch.max(40.0)
    };
    let mut plan = Plan::new();
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let calls = ((v.kind.calls() as f32 * len + 0.5) as usize).clamp(1, 12);
    match v.kind {
        Vocalization::Laugh | Vocalization::Chuckle => {
            let laugh = v.kind == Vocalization::Laugh;
            let (inhale, posture, rise, pressure, voiced) = if laugh {
                (0.25, Posture::Open, 1.8, 1.0, 1.0)
            } else {
                (0.0, Posture::Hum, 1.5, 0.8, 0.7)
            };
            let rate = tuning.articulation.laugh_rate;
            let voice = tuning.articulation.laugh_voiced * voiced;
            let start = if inhale > 0.0 {
                plan.breathe_in(3.5 * s, 0.0, inhale);
                inhale + TURN
            } else {
                0.05
            };
            let mut t = start;
            let mut ends = start;
            for k in 0..calls {
                #[allow(clippy::cast_precision_loss)]
                let along = if calls > 1 {
                    k as f32 / (calls - 1) as f32
                } else {
                    0.0
                };
                let length = if k == 0 && laugh { 1.4 * voice } else { voice };
                let push = tuning.anatomy.laugh_pressure * pressure * s * (1.0 - 0.4 * along);
                let hz = p0 * rise * (1.0 - 0.125 * along);
                // The pressure leads the voice by the lungs' own time, and
                // lets go through its second half, so the call dies away
                // with it and the breath between the calls is let out at
                // the floor's pressure.
                plan.push(push, (t - 0.03, t + 0.55 * length), (0.025, 0.04));
                plan.glottis(
                    &[(VOICING, 1.0), (SPREAD, LAUGH_SPREAD), (TENSENESS, 0.55)],
                    (t, t + length),
                    (0.01, 0.015),
                );
                plan.glide((1.06 * hz, 0.9 * hz), (t, t + length), 0.03);
                ends = t + length;
                // The interval to the next grows along the bout (by a quarter,
                // about the rate), and leaves the breath at least 70 ms.
                t += ((0.875 + 0.25 * along) / rate).max(length + 0.07);
            }
            // Between the calls: breath through the open glottis.
            plan.glottis_under(
                &[
                    (VOICING, 0.0),
                    (SPREAD, LAUGH_BREATH),
                    (TENSENESS, 0.55),
                    (PUSH, LAUGH_FLOOR * s.min(1.0) * pressure),
                ],
                (start - 0.05, ends + 0.06),
                (0.03, 0.06),
            );
            plan.seconds = ends + 0.18;
            plan.hold(posture, (0.0, plan.seconds - 0.08), (0.04, 0.08), STRONG);
        }
        Vocalization::Cough => {
            let inhale = 0.35 * len;
            plan.breathe_in(5.0 * s, 0.0, inhale);
            // The folds shut and come to rest before the lungs push: pressed
            // together under pressure they are blown open and vibrate (a
            // pressed voice at 40 cmH2O, 0 to 0.8 cm² every cycle), shut at
            // rest they feel none of it.
            let shut = inhale + 0.01;
            let push_from = shut + 0.05;
            let burst = push_from + 0.2 * len;
            plan.hold(Posture::Mid, (0.0, burst + 0.35), (0.05, 0.1), STRONG);
            // The folds pressed shut and the tube above them drawn in.
            plan.glottis(
                &[(SEAL, COUGH_SEAL), (VOICING, 0.0), (SPREAD, 0.0)],
                (shut, burst),
                (0.02, 0.0),
            );
            plan.lay(
                Tier::LarynxAndPitch,
                &[(var::larynx::SQUEEZE, 0.5)],
                (shut, burst),
                (0.05, 0.03),
                STRONG,
            );
            // The lungs build their pressure behind them and keep it through
            // the expulsive phase, then let go.
            let peak = tuning.anatomy.cough_pressure * s;
            plan.push(peak, (push_from, burst + 0.04), (0.12 * len, 0.25 * len));
            // Thrown open...
            plan.glottis(
                &[(VOICING, 0.0), (SPREAD, COUGH_OPENING)],
                (burst, burst + 0.07),
                (0.0, 0.04),
            );
            // ...and a short, breathy voice as the push dies away.
            let tail = burst + 0.09;
            plan.glottis(
                &[(VOICING, 1.0), (SPREAD, 0.1), (TENSENESS, 0.55)],
                (tail, tail + 0.16 * len),
                (0.02, 0.04),
            );
            plan.push(
                0.25 * peak.min(24.0),
                (burst + 0.05, tail + 0.12 * len),
                (0.1, 0.08),
            );
            plan.glide((1.15 * p0, 0.9 * p0), (tail, tail + 0.16 * len), 0.05);
            plan.seconds = tail + 0.16 * len + 0.2;
        }
        Vocalization::ClearThroat => {
            plan.breathe_in(2.5 * s, 0.0, 0.2);
            let mut t = 0.2 + 0.06;
            plan.hold(Posture::Relaxed, (0.0, 2.0), (0.05, 0.1), STRONG);
            for k in 0..calls {
                #[allow(clippy::cast_precision_loss)]
                let weaker = 1.0 - 0.25 * k as f32;
                // Shut at rest first (see the cough), then the push, then the
                // release; the push dies away through the voice, which stops
                // with it.
                let shut = 0.14 * len;
                let voice = 0.12 * len;
                plan.glottis(
                    &[(SEAL, COUGH_SEAL), (VOICING, 0.0), (SPREAD, 0.0)],
                    (t, t + shut),
                    (0.015, 0.0),
                );
                plan.push(
                    16.0 * s * weaker,
                    (t + 0.05, t + shut + 0.3 * voice),
                    (0.05, 0.7 * voice),
                );
                // Released into a pressed, low, creaky voice (Stage 3A's
                // creak: the cover slack, the folds pressed).
                plan.glottis(
                    &[
                        (VOICING, 1.0),
                        (SPREAD, 0.03),
                        (TENSENESS, 0.9),
                        (var::glottis::CREAK, 1.0),
                    ],
                    (t + shut, t + shut + voice),
                    (0.005, 0.03),
                );
                plan.glide((0.95 * p0, 0.82 * p0), (t + shut, t + shut + voice), 0.03);
                t += shut + voice + 0.08;
            }
            plan.lay(
                Tier::LarynxAndPitch,
                &[(var::larynx::SQUEEZE, 0.7)],
                (0.15, t - 0.1),
                (0.05, 0.05),
                STRONG,
            );
            plan.seconds = t + 0.08;
            // Re-hold the posture to the event's end.
            if let Some(Some(h)) = plan.holds.first_mut() {
                h.window.end = plan.seconds - 0.1;
            }
        }
        Vocalization::Sigh => {
            let inhale = 0.6 * len;
            plan.breathe_in(6.5 * s, 0.0, inhale);
            let out = inhale + TURN;
            let exhale = 1.3 * len;
            let end = out + exhale;
            // Open, relaxing towards schwa over the breath out.
            plan.hold(
                Posture::Open,
                (0.0, out + 0.2 * exhale),
                (0.06, 0.7 * exhale),
                STRONG,
            );
            plan.hold(
                Posture::Relaxed,
                (out + 0.2 * exhale, end + 0.2),
                (0.7 * exhale, 0.15),
                STRONG,
            );
            // A breathy voice: the glottis held open, the folds lax.
            plan.glottis(
                &[(VOICING, 1.0), (SPREAD, 0.08), (TENSENESS, 0.45)],
                (out, end - 0.3 * len),
                (0.04, 0.1),
            );
            plan.glottis_under(
                &[(VOICING, 0.0), (SPREAD, 0.3)],
                (out, end + 0.1),
                (0.04, 0.1),
            );
            // The push falls through the breath out.
            plan.push(
                6.0 * s,
                (out - 0.03, out + 0.15 * exhale),
                (0.05, 0.85 * exhale),
            );
            plan.push(1.2 * s, (out + 0.15 * exhale, end), (0.85 * exhale, 0.15));
            plan.glide((1.25 * p0, 0.8 * p0), (out, end - 0.3 * len), 0.04);
            plan.seconds = end + 0.25;
        }
    }
    // The lungs are the event's from its start to its end: where nothing of
    // it pushes they push a breath's worth (a voice let go of its push would
    // otherwise go back to the voice's own lung pressure while the event's
    // voice is still on).
    let floor = plan.seconds - 0.03;
    plan.lay(
        Tier::GlottalAdduction,
        &[(var::glottis::PUSH, LUNGS_HELD)],
        (0.0, floor),
        (0.0, 0.02),
        0.001,
    );
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_round_trip_by_name() {
        for v in Vocalization::ALL {
            assert_eq!(Vocalization::from_name(v.name()), Some(v));
        }
        assert_eq!(Vocalization::from_name("Laugh"), None);
        assert_eq!(Vocalization::from_name("HH AX"), None);
    }

    #[test]
    fn every_plan_fits_and_ends() {
        for kind in Vocalization::ALL {
            for length in [0.5, 1.0, 2.0] {
                let v = Vocal::new(kind).with_length(length).with_strength(2.0);
                let p = plan(&v, 110.0, &Tuning::DEFAULT);
                assert!(p.len() < MOST, "{kind:?} {length}: {}", p.len());
                assert!(
                    p.seconds > 0.3 && p.seconds < 6.0,
                    "{kind:?}: {}",
                    p.seconds
                );
                for g in p.gestures() {
                    assert!(
                        g.window.end + g.window.ramp_out <= p.seconds + 1e-3,
                        "{kind:?} {g:?}"
                    );
                }
            }
        }
    }
}
