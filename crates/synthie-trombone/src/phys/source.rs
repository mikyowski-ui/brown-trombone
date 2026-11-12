//! What drives the glottis: lung pressure and the area between the folds.
//!
//! Two sources share the same controls:
//!
//! * [`SourceKind::Folds`]: self-oscillating folds (see [`folds`]). Pitch,
//!   loudness and breathiness come out of tension, lung pressure and how
//!   far apart the folds are held.
//! * [`SourceKind::Kinematic`]: the area follows a prescribed pulse at the
//!   requested pitch, and the flow through it is whatever lung pressure
//!   and the tract allow. It sings when asked to, and (unless
//!   [`Fallback::Never`]) the notes outside the folds' range: below it as
//!   vocal fry, above it as the falsetto these folds have no register
//!   for. Inside the range the folds sing everything, stalls included.
//!
//! The classic controls are reinterpreted rather than replaced:
//!
//! * tenseness → how tightly the folds are pressed together;
//! * the voicing gate → adducted (vibrating) or pulled apart (breathing);
//! * anything that needs air — voicing or a consonant — keeps the lungs
//!   pushing.
//!
//! [`folds`]: super::folds

use super::folds::{FoldParams, Folds};
use super::{CM_H2O, Gauge};
use crate::glottis::Glottis;
use crate::math::{cos, exp, ln, sqrt};
use crate::tuning::Tuning;
use crate::voice::Fallback;
use core::f32::consts::PI;

/// Seconds for lung pressure to build or fall.
const LUNG_TIME: f32 = 0.03;
// Lung pressure, the time the folds take to come together or part, the
// attack's depth and hold, the modal and posterior gaps and the opening
// for voiceless sounds come from the voice's `Tuning`.
/// When voicing begins the folds are first pressed together, this far past
/// touching (cm), and held until they are together and there has been
/// enough pressure across them for a moment (`attack_hold`); then they relax to where
/// the tenseness holds them. Speakers do this (a glottal attack). Without
/// it, folds brought together in a stream that is already flowing (after an
/// /s/, or behind lips that are still parting) find a quiet resting place
/// in it and take a tenth of a second to start; pressed shut, they are
/// blown open and are vibrating at once. The same is done whenever the
/// voice is wanted but the folds have stalled, as at the end of a /b/.
///
/// It also gives stops their timing for free: after /b/ the folds are
/// already together and start within a few milliseconds of the release;
/// after /p/ they first have to come together.
/// Seconds for pressed folds to come back to touching once the attack
/// lets go (much quicker than bringing them together from apart).
const ATTACK_RELEASE_TIME: f32 = 0.002;
/// Slack folds are held closer together: below this tension the rest gap
/// shrinks in proportion, down to `CLOSEST` of itself. Low notes and fry
/// are sung that way, and it is what keeps slack folds vibrating under a
/// close vowel, where they otherwise stall below about 75 Hz.
const SLACK_TENSION: f32 = 0.8;
/// Lung pressure, dyn/cm², below which the edges are held closer in
/// proportion: softly blown folds swing less, and held as far apart as for
/// a full voice they stop meeting (at 5 cmH2O on close vowels and high
/// notes about half the cycles stayed open).
const SOFT_VOICE: f32 = 8.0 * CM_H2O;
const CLOSEST: f32 = 0.3;
/// Longest the glottis stays open in each pulse of the pulse source,
/// seconds. Below about 100 Hz this turns the pulse into vocal fry: short
/// sharp openings with long closures between them, which is what a voice
/// does below the notes its folds can sustain.
const FRY_OPEN_TIME: f32 = 0.007;
/// Pressure across the folds, dyn/cm², that counts as enough to start:
/// just above what these folds need to keep going.
const START_PRESSURE: f32 = 3.5 * CM_H2O;
/// Swing of the area between the folds, cm² at native size, below which
/// they count as stalled.
const STALL_SWING: f32 = 0.03;
/// Seconds over which the voice is handed between the folds and the pulse
/// source, the glottal area of one fading into the other's. Swapped in one
/// tick, the two areas differed by up to 0.04 cm² (folds wide in their
/// open phase, the pulse shut, or the reverse), and the water hammer of
/// that step left the mouth as a click several times the vowel's peak:
/// the critter's high notes cross its folds' range within a vowel.
const HANDOVER_TIME: f32 = 0.006;
/// Peak area swept by the vibrating folds (kinematic source), cm², native
/// size.
const VIBRATION_AREA: f32 = 0.14;
/// Lung pressure at which the kinematic folds sweep `VIBRATION_AREA`,
/// dyn/cm². Real folds swing wider the harder they are blown, and the
/// self-oscillating ones here do; the kinematic ones follow the same way,
/// so that loudness answers lung pressure whichever model is singing.
const VIBRATION_PRESSURE: f32 = 8.0 * CM_H2O;
/// Standing leak of a fully breathy voice (kinematic source), cm².
const BREATHY_LEAK: f32 = 0.12;
/// Point in the open phase where the kinematic area peaks.
const PEAK: f32 = 0.62;

/// Rest half-width of the vibrating edges, cm, native size: 0.012 cm at
/// modal tenseness 0.65 (Steinecke & Herzel's standard is 0.018, which
/// these folds cannot sustain under every vowel), closing in proportion
/// to zero (pressed) at tenseness 1 and never wider than `WIDEST_GAP`
/// while voicing (it was 0.014 cm: held that far apart, the slack edges
/// of the elder voice did not meet on close vowels and higher notes).
/// Breathiness beyond that does not come from
/// holding the edges further apart (they would stop vibrating) but, as in
/// a real larynx, from a gap left open between the cartilages behind them.
const MODAL_TENSENESS: f32 = 0.65;
const WIDEST_GAP: f32 = 0.012;
/// Below this tenseness the edges themselves part and voicing gives way
/// to a whisper.
pub(crate) const WHISPER_TENSENESS: f32 = 0.12;

/// Pitch the folds settle at, Hz, by rest gap (rows, cm) and tension
/// (columns, `TABLE_TENSION_START · TABLE_TENSION_RATIO^k`), at native size
/// with lung pressure following the effort rule in `begin_block`. `0` means
/// they do not vibrate. Generated by
/// `cargo run --release --example folds -- --table`.
const TABLE_GAPS: [f32; 3] = [0.0, 0.007, 0.014];
const TABLE_TENSION_START: f32 = 0.35;
const TABLE_TENSION_RATIO: f32 = 1.18;
const TABLE_COLUMNS: usize = 15;
#[rustfmt::skip]
const PITCH_TABLE: [[f32; TABLE_COLUMNS]; 3] = [
    [45.5, 52.7, 62.6, 75.1, 89.1, 101.9, 124.5, 148.5, 172.3, 202.4, 246.1, 272.0, 0.0, 0.0, 0.0],
    [0.0, 53.6, 62.7, 73.4, 88.0, 106.0, 126.0, 147.4, 171.1, 199.6, 235.4, 270.5, 306.3, 0.0, 0.0],
    [0.0, 0.0, 0.0, 74.4, 87.9, 103.3, 121.1, 141.7, 168.9, 197.3, 228.8, 267.0, 295.7, 323.1, 0.0],
];

/// Tension, relative to the standard folds, that gives native-size pitch
/// `hz` at rest gap `gap`; and whether the folds can actually sing it. A
/// note counts as singable only where the folds vibrate at the gaps on
/// both sides of this one, with a little margin.
fn tension_from_table(hz: f32, gap: f32) -> (f32, bool) {
    let step = TABLE_GAPS[1] - TABLE_GAPS[0];
    #[allow(clippy::cast_precision_loss)]
    let top = (TABLE_GAPS.len() - 1) as f32 - 0.001;
    let at = (gap / step).clamp(0.0, top);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let row = at as usize;
    #[allow(clippy::cast_precision_loss)]
    let weight = at - row as f32;
    let mut pitch = [0.0f32; TABLE_COLUMNS];
    for (k, p) in pitch.iter_mut().enumerate() {
        let (a, b) = (PITCH_TABLE[row][k], PITCH_TABLE[row + 1][k]);
        *p = if weight < 0.02 {
            a
        } else if a == 0.0 || b == 0.0 {
            // Slack folds: the edge of where they vibrate lies between
            // these two gaps; put it halfway, rather than at the gap that
            // vibrates (which left notes the folds sing well, such as the
            // default voice at 1.4 times its size, to the pulse). At the
            // stiff end the edge stays where both gaps vibrate.
            let slack = k < TABLE_COLUMNS / 2;
            if !slack || (a == 0.0) == (weight < 0.5) {
                0.0
            } else {
                a.max(b)
            }
        } else {
            a + (b - a) * weight
        };
    }
    let Some(first) = pitch.iter().position(|&p| p > 0.0) else {
        return (1.0, false);
    };
    let last = pitch.iter().rposition(|&p| p > 0.0).unwrap_or(first);
    let log_ratio = ln(TABLE_TENSION_RATIO);
    #[allow(clippy::cast_precision_loss)]
    let column = |k: usize, towards: f32| k as f32 + towards;
    let position = if hz <= pitch[first] || first == last {
        // Outside the table pitch goes nearly in proportion to tension.
        column(first, ln(hz / pitch[first]) / log_ratio)
    } else if hz >= pitch[last] {
        column(last, ln(hz / pitch[last]) / log_ratio)
    } else {
        let mut k = first;
        while k + 1 < last && pitch[k + 1] < hz {
            k += 1;
        }
        column(k, ln(hz / pitch[k]) / ln(pitch[k + 1] / pitch[k]))
    };
    let covered = first < last && hz >= 1.04 * pitch[first] && hz <= 0.97 * pitch[last];
    (TABLE_TENSION_START * exp(position * log_ratio), covered)
}
/// How lung pressure rises with fold tension above the natural pitch.
const EFFORT_EXPONENT: f32 = 0.6;
/// Seconds over which the pitch trim acts.
const TRIM_TIME: f32 = 0.12;
/// Furthest the trim may bend the tension (natural log; about 2 semitones).
const TRIM_RANGE: f32 = 0.12;

/// Which model moves the folds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SourceKind {
    Kinematic,
    Folds,
}

/// Parts of the fold control that can be switched off one at a time, to
/// find out which of them a behaviour comes from. All on in normal use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)] // independent switches
pub(crate) struct Probe {
    /// The slow pitch trim.
    pub trim: bool,
    /// The stall watch, which presses stalled folds together again.
    pub monitor: bool,
    /// The folds feel the pressure the tract sends back above them.
    pub supraglottal: bool,
    /// The folds feel the pressure the trachea sends back below them.
    pub subglottal: bool,
}

impl Probe {
    pub(crate) const ON: Self = Self {
        trim: true,
        monitor: true,
        supraglottal: true,
        subglottal: true,
    };
}

/// Glottal source.
#[derive(Clone, Debug)]
#[allow(clippy::struct_excessive_bools)] // independent states
pub(crate) struct Source {
    pub(crate) kind: SourceKind,
    phase: f32,
    lung: f32,
    lung_target: f32,
    /// Lung pressure asked for while sounding, dyn/cm².
    lung_base: f32,
    /// The voice's settings for the breath and the folds.
    tuning: Tuning,
    /// Standing opening: an area for the kinematic source, a half-width for
    /// the folds.
    leak: f32,
    leak_target: f32,
    gap: f32,
    gap_target: f32,
    /// Standing gap between the cartilages behind the vibrating edges.
    chink: f32,
    chink_target: f32,
    /// Glottal opening behind the folds that the speaker asks for, apart
    /// from voicing (a voiceless consonant's abduction), cm²: wanted and
    /// present. It moves at the arytenoids' pace (`abduction_time`).
    pub(crate) spread_target: f32,
    spread: f32,
    /// Share by which the fold cover is slackened, wanted and present.
    pub(crate) slack_target: f32,
    slack: f32,
    /// Share of the glottal flow that passes between the vibrating edges.
    fold_share: f32,
    f0: f32,
    f1: f32,
    tense0: f32,
    tense1: f32,
    amp0: f32,
    amp1: f32,
    /// Relative larynx size (the tract length scale).
    size: f32,

    folds: Folds,
    voicing: bool,
    /// The folds are being pressed together to start the voice.
    pressing: bool,
    /// Seconds they have been touching with enough pressure across them.
    pressed_for: f32,
    /// Pressure across the folds, averaged over a few milliseconds.
    across: f32,
    /// Watching for a stall: smallest and largest area between the folds
    /// since the watch began, and how long ago that was.
    watch_lo: f32,
    watch_hi: f32,
    watch_for: f32,
    /// Whether the folds are singing the current note (it lies inside
    /// their range, or the pulse is not allowed to take it).
    reachable: bool,
    /// Share of the glottal area that comes from the folds while the voice
    /// is handed over (1 when they sing, 0 when the pulse does).
    blend: f32,
    tension0: f32,
    tension1: f32,
    /// Pitch tracking for the trim.
    mean_edge: f32,
    was_above: bool,
    since_crossing: f32,
    clean_cycles: u32,
    last_period: f32,
    trim: f32,
    /// Measured pitch of the folds, Hz (0 while not vibrating).
    pub(crate) measured: f32,
    /// Fixed tension and rest gap, for measuring the folds themselves.
    pub(crate) tension_override: Option<f32>,
    pub(crate) gap_override: Option<f32>,
    /// Fixed muscle activations (cricothyroid, thyroarytenoid, lateral
    /// cricoarytenoid) for measuring the body–cover folds.
    pub(crate) muscle_override: Option<(f32, f32, f32)>,
    /// Area between the vibrating edges on the last tick, cm²: the glottal
    /// area less the gap behind them (or, for the pulse, less its leak).
    pub(crate) edge_area: f32,
    pub(crate) probe: Probe,
    /// When the pulse may take over from the folds.
    pub(crate) fallback: Fallback,
}

impl Source {
    pub(crate) fn new() -> Self {
        Self {
            kind: SourceKind::Kinematic,
            phase: 0.0,
            lung: 0.0,
            lung_target: 0.0,
            lung_base: Tuning::DEFAULT.lung_pressure * CM_H2O,
            tuning: Tuning::DEFAULT,
            leak: Tuning::DEFAULT.abducted_area,
            leak_target: Tuning::DEFAULT.abducted_area,
            gap: Tuning::DEFAULT.abducted_area / (2.0 * 1.4),
            gap_target: Tuning::DEFAULT.abducted_area / (2.0 * 1.4),
            chink: 0.0,
            chink_target: 0.0,
            spread_target: 0.0,
            spread: 0.0,
            slack_target: 0.0,
            slack: 0.0,
            fold_share: 1.0,
            f0: 140.0,
            f1: 140.0,
            tense0: 0.6,
            tense1: 0.6,
            amp0: 0.0,
            amp1: 0.0,
            size: 1.0,
            folds: Folds::new(FoldParams::steinecke_herzel()),
            voicing: false,
            pressing: false,
            pressed_for: 0.0,
            across: 0.0,
            watch_lo: f32::INFINITY,
            watch_hi: 0.0,
            watch_for: 0.0,
            reachable: true,
            blend: 1.0,
            tension0: 1.0,
            tension1: 1.0,
            mean_edge: 0.0,
            was_above: false,
            since_crossing: 0.0,
            clean_cycles: 0,
            last_period: 0.0,
            trim: 0.0,
            measured: 0.0,
            tension_override: None,
            gap_override: None,
            muscle_override: None,
            edge_area: 0.0,
            probe: Probe::ON,
            fallback: Fallback::Auto,
        }
    }

    pub(crate) fn reset(&mut self) {
        let (size, kind, probe, fallback) = (self.size, self.kind, self.probe, self.fallback);
        let overrides = (self.tension_override, self.gap_override);
        let muscles = self.muscle_override;
        let (lung_base, tuning) = (self.lung_base, self.tuning);
        let (spread, slack) = (self.spread_target, self.slack_target);
        *self = Self::new();
        (self.spread_target, self.slack_target) = (spread, slack);
        self.kind = kind;
        self.probe = probe;
        self.fallback = fallback;
        self.lung_base = lung_base;
        self.tuning = tuning;
        (self.tension_override, self.gap_override) = overrides;
        self.muscle_override = muscles;
        self.set_size(size);
    }

    /// Begin a glottal attack.
    fn press(&mut self) {
        self.pressing = true;
        self.pressed_for = 0.0;
        self.restart_watch();
    }

    fn restart_watch(&mut self) {
        self.watch_lo = f32::INFINITY;
        self.watch_hi = 0.0;
        self.watch_for = 0.0;
    }

    /// Take the breath and fold settings of a voice (lung pressure
    /// included).
    pub(crate) fn set_tuning(&mut self, tuning: &Tuning) {
        self.tuning = *tuning;
        self.lung_base = tuning.lung_pressure * CM_H2O;
    }

    /// Lung pressure while sounding, cmH2O.
    pub(crate) fn set_lung_pressure(&mut self, cm_h2o: f32) {
        self.lung_base = cm_h2o * CM_H2O;
    }

    pub(crate) fn lung_pressure(&self) -> f32 {
        self.lung_base / CM_H2O
    }

    /// Whether the lungs are pushing, and have been for long enough that
    /// the pressure is all there (2 cmH2O at least).
    pub(crate) fn breath_is_up(&self) -> bool {
        self.lung_target > 2.0 * CM_H2O && self.lung >= 0.9 * self.lung_target
    }

    /// Fold size follows the tract: `len` is the relative tract length.
    pub(crate) fn set_size(&mut self, len: f32) {
        self.size = len;
        let params = match self.muscle_override {
            Some((ct, ta, lc)) => FoldParams::from_muscles(ct, ta, lc),
            None => FoldParams::steinecke_herzel(),
        };
        self.folds.set_params(params.sized(len));
    }

    /// Tension that puts the folds at `hz`, and whether they can sing it.
    /// A larynx of relative size `s` sounds `1/s` times as high at the same
    /// tension, so the table is read at the native-size pitch.
    fn tension_for(&self, hz: f32) -> (f32, bool) {
        if let Some(t) = self.tension_override {
            return (t, true);
        }
        let (tension, covered) = tension_from_table(hz * self.size, self.gap_target / self.size);
        (tension.clamp(0.2, 5.0), covered)
    }

    /// Rest half-width of the vibrating edges, cm at native size, for a
    /// modal voice at pitch `hz` and tenseness `tense`.
    fn modal_gap(&self, hz: f32, tense: f32) -> f32 {
        let slack = 1.0 - tense;
        let gap = (self.tuning.modal_gap * slack / (1.0 - MODAL_TENSENESS)).min(WIDEST_GAP)
            * (self.lung_base / SOFT_VOICE).min(1.0);
        let (tension, _) = tension_from_table(hz * self.size, gap);
        gap * (tension / SLACK_TENSION).clamp(CLOSEST, 1.0)
    }

    /// Lowest and highest pitch, Hz, the folds can sing at the present
    /// size and tenseness (the pitch table read the way `begin_block`
    /// reads it). Not for the audio path: it walks the whole range.
    pub(crate) fn range(&self) -> (f32, f32) {
        /// Pitch step of the walk (about a sixth of a semitone).
        const STEP: f32 = 1.01;
        if self.tension_override.is_some() || self.tense1 < WHISPER_TENSENESS {
            return (0.0, 0.0);
        }
        let (mut lo, mut hi) = (0.0f32, 0.0f32);
        let mut hz = 20.0f32;
        while hz < 2000.0 {
            let gap = match self.gap_override {
                Some(gap) => gap,
                None => self.modal_gap(hz, self.tense1),
            };
            if tension_from_table(hz * self.size, gap).1 {
                if lo == 0.0 {
                    lo = hz;
                }
                hi = hz;
            }
            hz *= STEP;
        }
        (lo, hi)
    }

    /// The model in charge right now: the folds, unless the note lies
    /// outside what they can sing.
    #[inline]
    fn active(&self) -> SourceKind {
        if self.kind == SourceKind::Folds && self.reachable {
            SourceKind::Folds
        } else {
            SourceKind::Kinematic
        }
    }

    /// Read the controls for the coming block. `constricted` says a
    /// consonant is being held, which needs breath even without voice.
    pub(crate) fn begin_block(&mut self, g: &Glottis, constricted: bool) {
        let (f0, f1) = g.block_frequency();
        let (t0, t1) = g.block_tenseness();
        self.f0 = f0;
        self.f1 = f1;
        self.tense0 = t0.clamp(0.0, 1.0);
        self.tense1 = t1.clamp(0.0, 1.0);
        self.amp0 = self.amp1;
        self.amp1 = g.intensity;

        let voicing = g.voiced || g.always_voice;
        if voicing && !self.voicing {
            self.press();
        }
        if !voicing {
            self.pressing = false;
        }
        self.voicing = voicing;
        // A spread glottis is held open for air: the lungs keep pushing.
        let breathing = voicing || constricted || g.intensity > 0.01 || self.spread_target > 0.0;
        let area_scale = self.size * self.size;
        let slack = 1.0 - self.tense1;
        self.leak_target = area_scale
            * if voicing {
                BREATHY_LEAK * slack * slack
            } else {
                self.tuning.abducted_area
            };
        let abducted = self.tuning.abducted_area / (2.0 * 1.4);
        self.gap_target = self.size
            * if !voicing {
                abducted
            } else if let Some(gap) = self.gap_override {
                gap
            } else if self.tense1 < WHISPER_TENSENESS {
                // Letting go: the edges part and the voice turns to breath.
                WIDEST_GAP + (abducted - WIDEST_GAP) * (1.0 - self.tense1 / WHISPER_TENSENESS)
            } else {
                self.modal_gap(f1, self.tense1)
            };
        let loose = (slack - 0.3).max(0.0) / 0.7;
        self.chink_target = if voicing && self.gap_override.is_none() {
            area_scale * self.tuning.posterior_gap * loose * loose
        } else {
            0.0
        };

        // Stiffer folds need more breath to keep going, as singers know.
        let mut effort = 1.0;
        if self.kind == SourceKind::Folds {
            let (tension, covered) = self.tension_for(f1);
            // Whether the folds can sing the note is read at the gap they
            // sing it with, also while they are apart for a pause or a
            // voiceless sound (at the open gap nothing is "covered"); and
            // the voice is only handed over while it sounds. The folds were
            // otherwise handed to the pulse source at every voiceless
            // consonant and back at the vowel, and the swap of the folds'
            // opening for the pulse source's leak in one tick went out as a
            // click.
            let covered = if voicing || self.tension_override.is_some() {
                covered
            } else {
                let gap = self.modal_gap(f1, self.tense1.max(WHISPER_TENSENESS));
                tension_from_table(f1 * self.size, gap).1
            };
            // Inside their range the folds sing every note themselves, even
            // one they keep stalling on: the stall watch presses them
            // together again, as a speaker would, but never hands the note
            // to the pulse.
            let covered = covered || self.fallback == Fallback::Never;
            if voicing && covered != self.reachable {
                // Handing over between the models: start the folds cleanly
                // if they were silent; the areas cross-fade (see
                // `HANDOVER_TIME`).
                self.reachable = covered;
                if covered && self.blend <= 0.0 {
                    self.folds.reset();
                }
                self.clean_cycles = 0;
                self.trim = 0.0;
            }
            self.tension0 = self.tension1;
            self.tension1 = tension;
            if self.reachable {
                effort = crate::math::powf(tension.max(1.0), EFFORT_EXPONENT);
            }
        }
        self.lung_target = if breathing {
            self.lung_base * effort
        } else {
            0.0
        };
    }

    /// One tick: `(glottal area cm², lung pressure dyn/cm²)`.
    #[inline]
    pub(crate) fn tick(&mut self, lambda: f32, dt: f32) -> (f32, f32) {
        self.lung += (self.lung_target - self.lung) * (dt / LUNG_TIME);
        let adduction = self.tuning.adduction_time;
        self.leak += (self.leak_target - self.leak) * (dt / adduction);
        let gap_target = if self.pressing && self.tense1 >= WHISPER_TENSENESS {
            -self.tuning.attack_depth * self.size
        } else {
            self.gap_target
        };
        let time = if !self.pressing && self.gap < 0.0 {
            ATTACK_RELEASE_TIME
        } else {
            adduction
        };
        self.gap += (gap_target - self.gap) * (dt / time);
        self.chink += (self.chink_target - self.chink) * (dt / adduction);
        let arytenoids = self.tuning.abduction_time;
        let spread = self.spread_target * self.size * self.size;
        self.spread += (spread - self.spread) * (dt / arytenoids);
        self.slack += (self.slack_target - self.slack) * (dt / arytenoids);
        let f = self.f0 + (self.f1 - self.f0) * lambda;

        let to = if self.active() == SourceKind::Folds {
            1.0
        } else {
            0.0
        };
        self.blend = if self.kind == SourceKind::Folds {
            let step = dt / HANDOVER_TIME;
            (self.blend + (to - self.blend).clamp(-step, step)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let mut folds = 0.0;
        if self.blend > 0.0 {
            self.folds.hold_at(self.gap, dt);
            let tension = self.tension0 + (self.tension1 - self.tension0) * lambda;
            self.folds.tension = tension * exp(self.trim) * (1.0 - self.slack);
            let between = self.folds.area();
            let total = between + self.chink + self.spread;
            self.fold_share = if total > 0.0 { between / total } else { 1.0 };
            if self.blend >= 1.0 {
                self.edge_area = between;
                return (total, self.lung);
            }
            folds = total;
        }

        self.phase += f * dt;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        let tense = self.tense0 + (self.tense1 - self.tense0) * lambda;
        let amp = self.amp0 + (self.amp1 - self.amp0) * lambda;

        // Pressed folds stay shut for most of the cycle, slack ones barely
        // close; fully slack folds do not vibrate at all (a whisper).
        let open_quotient = (0.85 - 0.45 * tense).min(FRY_OPEN_TIME * f);
        let x = self.phase / open_quotient;
        let pulse = if x >= 1.0 {
            0.0
        } else if x < PEAK {
            0.5 * (1.0 - cos(PI * x / PEAK))
        } else {
            cos(0.5 * PI * (x - PEAK) / (1.0 - PEAK))
        };
        let engage = (tense / 0.3).min(1.0);
        let engage = engage * engage * (3.0 - 2.0 * engage);
        let blown = (self.lung / VIBRATION_PRESSURE).clamp(0.25, 2.5);
        let swing = self.size * self.size * VIBRATION_AREA * blown;
        let edge = swing * amp * engage * pulse;
        let kinematic = self.leak + self.spread + edge;
        let w = self.blend;
        self.edge_area = w * self.folds.area() + (1.0 - w) * edge;
        (w * folds + (1.0 - w) * kinematic, self.lung)
    }

    /// Which model is making the voice right now, or `None` while no voice
    /// is wanted.
    pub(crate) fn sounding(&self) -> Option<SourceKind> {
        self.voicing.then(|| self.active())
    }

    /// Let the folds feel what the tract solver just found, and keep the
    /// pitch honest. `lambda` is the position within the block.
    #[inline]
    pub(crate) fn react(&mut self, gauge: &Gauge, lambda: f32, dt: f32) {
        if self.blend <= 0.0 {
            self.measured = 0.0;
            return;
        }
        if self.probe.supraglottal && self.probe.subglottal {
            self.folds.step(gauge, self.fold_share, dt);
        } else {
            // A load that sends nothing back: lungs straight below, open
            // air straight above, and the flow those two alone would drive.
            let mut alone = *gauge;
            if !self.probe.subglottal {
                alone.p_sub = self.lung;
            }
            if !self.probe.supraglottal {
                alone.p_supra = 0.0;
            }
            let area = self.folds.area() + self.chink + self.spread;
            let drop = (alone.p_sub - alone.p_supra).max(0.0);
            alone.u_glottis = area * sqrt(2.0 * drop / (1.1 * super::RHO));
            self.folds.step(&alone, self.fold_share, dt);
        }

        if self.active() != SourceKind::Folds {
            // Fading out: the folds still move, but no longer sing.
            self.measured = 0.0;
            return;
        }

        // The attack ends once the folds touch with pressure behind them.
        let between = self.folds.area();
        self.across += (gauge.p_sub - gauge.p_supra - self.across) * (dt / 0.003);
        if self.pressing {
            // "Together" is where the cartilages hold them, not whether the
            // edges happen to touch: under pressure one edge can stand a
            // hair open for as long as it likes.
            if self.gap <= 0.0 && self.across > START_PRESSURE {
                self.pressed_for += dt;
                if self.pressed_for >= self.tuning.attack_hold {
                    self.pressing = false;
                    self.restart_watch();
                }
            }
        } else if self.voicing && self.gap_override.is_none() && self.probe.monitor {
            // Wanted but stalled: press together and start again.
            self.watch_lo = self.watch_lo.min(between);
            self.watch_hi = self.watch_hi.max(between);
            self.watch_for += dt;
            let period = 1.0 / (self.f0 + (self.f1 - self.f0) * lambda).max(20.0);
            if self.watch_for > (1.5 * period).max(0.012) {
                if self.watch_hi - self.watch_lo < STALL_SWING * self.size * self.size {
                    self.press();
                }
                self.restart_watch();
            }
        }

        // Pitch: time between upward crossings of the lower edge through
        // its own slow average.
        let edge = self.folds.lower_edge();
        self.mean_edge += (edge - self.mean_edge) * (dt / 0.02);
        let above = edge > self.mean_edge;
        self.since_crossing += dt;
        if above && !self.was_above {
            let period = self.since_crossing;
            self.since_crossing = 0.0;
            let target = self.f0 + (self.f1 - self.f0) * lambda;
            let ratio = period * target;
            let steady = self.last_period > 0.0
                && (period / self.last_period - 1.0).abs() < 0.15
                && (0.6..1.6).contains(&ratio);
            self.last_period = period;
            if steady && self.voicing {
                self.clean_cycles += 1;
                self.measured = 1.0 / period;
            } else {
                self.clean_cycles = 0;
            }
            // The table sets the pitch at once; this only removes what is
            // left, slowly, and only once the voice is running cleanly.
            // Not while the cover is slackened on purpose: that pitch dip is
            // wanted.
            if self.clean_cycles >= 3
                && self.tension_override.is_none()
                && self.probe.trim
                && self.slack < 0.005
            {
                self.trim =
                    (self.trim + (period / TRIM_TIME) * ln(ratio)).clamp(-TRIM_RANGE, TRIM_RANGE);
            }
        }
        self.was_above = above;
        if self.since_crossing > 0.05 {
            // Not vibrating.
            self.clean_cycles = 0;
            self.measured = 0.0;
        }
    }
}
