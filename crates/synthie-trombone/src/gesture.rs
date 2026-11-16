//! A task-dynamic gesture score: speech as overlapping gestures on six
//! independent tiers of the vocal tract, after Saltzman & Munhall (1989)
//! and Browman & Goldstein's articulatory phonology.
//!
//! A [`Gesture`] is a target for the variables of one [`Tier`] (where the
//! tongue tip constricts and how narrowly, how open the velar port is, how
//! far the glottis is spread), with a stiffness and a damping ratio (a
//! critically damped mass on a spring by default), a dominance (its weight
//! against the other gestures active on its tier at the same time) and an
//! activation [`Window`] with ramps in and out. A [`Score`] holds the
//! gestures of an utterance in the order they start (a fixed number at
//! once, [`Score::CAPACITY`]: it never allocates), and at any moment
//! blends the active gestures of each tier (the targets and stiffnesses
//! weighted by activation times dominance; where a tier has nothing
//! active, its variables are drawn to a neutral attractor, or held where
//! they are), and integrates each tier's dynamics exactly over the control
//! block, as the tract's own articulators are (so the motion does not
//! depend on the block length).
//!
//! Coarticulation is overlap: a consonant made by the tongue tip puts a
//! gesture on [`Tier::TongueTip`] alone and leaves the tongue body and the
//! lips to the vowels around it, whose gestures span it; a rounded vowel's
//! lip gesture starts during the consonants before it.
//!
//! The score knows nothing of phonemes: a gesture is any target on any
//! tier, so breaths, creaks and laughs can be scored as well as speech.
//! [`Trombone::push_gesture`](crate::Trombone::push_gesture) adds one to a
//! playing voice; [`Trombone::tier_activity`](crate::Trombone::tier_activity)
//! reads what each tier is doing.
//!
//! The variables of each tier are listed in [`var`]. Two of them are slots
//! the present tract ignores (the height of the larynx, the protrusion of
//! the lips): they wait for a tract whose length can change.

use crate::math::{ln, spring_step};

/// One of the six independent tiers a gesture acts on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tier {
    /// The folds brought together or held apart: voicing, the spread
    /// glottis of a voiceless consonant and its aspiration (voice onset
    /// time), breathiness.
    GlottalAdduction,
    /// The pitch contour, and the height of the larynx (a slot, ignored
    /// until the tract's length can change).
    LarynxAndPitch,
    /// The velar port: how much the nose is coupled to the mouth.
    Velum,
    /// The tongue body: vowels' tongue shapes and the dorsal (velar)
    /// constrictions.
    TongueBody,
    /// The tongue tip and blade: dental, alveolar, postalveolar and
    /// retroflex constrictions, and the lateral.
    TongueTip,
    /// The lips and the jaw: rounding, protrusion (a slot, ignored until the
    /// tract's length can change), bilabial and labiodental constrictions.
    LipsAndJaw,
}

impl Tier {
    /// Every tier, in the order [`TierActivity`] arrays are given.
    pub const ALL: [Tier; 6] = [
        Tier::GlottalAdduction,
        Tier::LarynxAndPitch,
        Tier::Velum,
        Tier::TongueBody,
        Tier::TongueTip,
        Tier::LipsAndJaw,
    ];

    /// The number of tiers.
    pub const COUNT: usize = 6;

    /// The tier's place in [`Tier::ALL`].
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// The tier's name, as in [`Tier::ALL`] (`"TongueTip"`).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Tier::GlottalAdduction => "GlottalAdduction",
            Tier::LarynxAndPitch => "LarynxAndPitch",
            Tier::Velum => "Velum",
            Tier::TongueBody => "TongueBody",
            Tier::TongueTip => "TongueTip",
            Tier::LipsAndJaw => "LipsAndJaw",
        }
    }

    /// The names of the tier's variables, in the order of their indices in
    /// [`var`].
    #[must_use]
    pub const fn variables(self) -> &'static [&'static str] {
        match self {
            Tier::GlottalAdduction => &["voicing", "spread", "tenseness", "slack", "softness"],
            Tier::LarynxAndPitch => &["pitch", "height"],
            Tier::Velum => &["port", "haste"],
            Tier::TongueBody => &[
                "location",
                "degree",
                "root",
                "narrowing_location",
                "narrowing_degree",
                "dorsal_location",
                "dorsal_degree",
                "mode_1",
                "mode_2",
                "mode_3",
            ],
            Tier::TongueTip => &["location", "degree"],
            Tier::LipsAndJaw => &[
                "aperture_location",
                "aperture",
                "protrusion",
                "jaw",
                "closure_location",
                "closure_degree",
            ],
        }
    }

    /// The natural frequency (rad/s) of a gesture on this tier that gives
    /// none of its own, and of the tier's neutral attractor: the speeds of
    /// the tract's articulators at speech speed (the tongue body some 25
    /// rad/s, the tip and the lips 88; see `Tuning::body_rate`).
    #[must_use]
    pub const fn default_stiffness(self) -> f32 {
        match self {
            Tier::GlottalAdduction => 60.0,
            Tier::LarynxAndPitch => 40.0,
            Tier::Velum => 50.0,
            Tier::TongueBody => 25.0,
            Tier::TongueTip | Tier::LipsAndJaw => 88.0,
        }
    }

    /// Where variable `v` of this tier is drawn while no gesture on the tier
    /// sets it: `Some(value)` (a constriction's degree: open; the velum:
    /// shut), or `None`, held where it is (the tongue body's place, the
    /// pitch).
    #[must_use]
    pub const fn neutral(self, v: usize) -> Option<f32> {
        use var::{body, lips, tip, velum};
        match self {
            Tier::Velum if v == velum::PORT => Some(0.0),
            Tier::Velum if v == velum::HASTE => Some(1.0),
            Tier::TongueBody if v == body::NARROWING_DEGREE || v == body::DORSAL_DEGREE => {
                Some(OPEN)
            }
            Tier::TongueTip if v == tip::DEGREE => Some(OPEN),
            Tier::LipsAndJaw if v == lips::APERTURE || v == lips::CLOSURE_DEGREE => Some(OPEN),
            Tier::LipsAndJaw if v == lips::PROTRUSION => Some(0.0),
            _ => None,
        }
    }

    /// The variable [`TierActivity`] reports for the tier.
    #[must_use]
    pub const fn main_variable(self) -> usize {
        match self {
            Tier::GlottalAdduction => var::glottis::SPREAD,
            Tier::LarynxAndPitch => var::larynx::PITCH,
            Tier::Velum => var::velum::PORT,
            Tier::TongueBody => var::body::DEGREE,
            Tier::TongueTip => var::tip::DEGREE,
            Tier::LipsAndJaw => var::lips::APERTURE,
        }
    }
}

/// The opening (constriction touch diameter, as
/// [`Constriction::diameter`](crate::Constriction)) of a constriction that
/// is no longer there: the neutral of every constriction degree. A
/// constriction let go gradually opens towards it; the tract ignores
/// openings this wide.
pub const OPEN: f32 = 3.3;

/// How many variables a tier has at most.
pub const VARIABLES: usize = 10;

/// The indices of each tier's variables in a [`Target`].
///
/// Places are tract cell indices (`0` the glottis, 44 the lips; as
/// [`Constriction::index`](crate::Constriction)), degrees touch diameters
/// (`≤ 0.3` shut, about `0.35‥0.75` narrow enough to hiss, wider ones shape
/// the tube).
pub mod var {
    /// [`Tier::GlottalAdduction`](super::Tier::GlottalAdduction).
    pub mod glottis {
        /// Whether the folds are set vibrating: `≥ 0.5` on.
        pub const VOICING: usize = 0;
        /// Opening held behind the folds, cm² at native size (see
        /// [`Trombone::set_glottal_spread`](crate::Trombone::set_glottal_spread)).
        pub const SPREAD: usize = 1;
        /// Tenseness of the folds, `0‥1`.
        pub const TENSENESS: usize = 2;
        /// Slackening of the fold cover, `0‥0.5`.
        pub const SLACK: usize = 3;
        /// Softness of the voice, `0‥1`.
        pub const SOFTNESS: usize = 4;
    }
    /// [`Tier::LarynxAndPitch`](super::Tier::LarynxAndPitch).
    pub mod larynx {
        /// Pitch, Hz.
        pub const PITCH: usize = 0;
        /// Height of the larynx, cm from rest (a slot for a tract whose
        /// length changes; the present tract ignores it).
        pub const HEIGHT: usize = 1;
    }
    /// [`Tier::Velum`](super::Tier::Velum).
    pub mod velum {
        /// Port openness, `0` shut to `1` open.
        pub const PORT: usize = 0;
        /// How many times faster than usual the port shuts.
        pub const HASTE: usize = 1;
    }
    /// [`Tier::TongueBody`](super::Tier::TongueBody).
    pub mod body {
        /// Tongue body place (constriction location), tract index `12‥29`.
        pub const LOCATION: usize = 0;
        /// Tongue body height (constriction degree): the diameter under it,
        /// `2.05‥3.5`, small is high.
        pub const DEGREE: usize = 1;
        /// The tongue root: `ln` of the widening of the pharynx.
        pub const ROOT: usize = 2;
        /// A vowel shape's own narrowing along the tongue: place...
        pub const NARROWING_LOCATION: usize = 3;
        /// ...and degree.
        pub const NARROWING_DEGREE: usize = 4;
        /// A dorsal (velar) constriction: place...
        pub const DORSAL_LOCATION: usize = 5;
        /// ...and degree.
        pub const DORSAL_DEGREE: usize = 6;
        /// The modal body's three numbers (see
        /// [`Shape::modes`](crate::inverse::Shape::modes); a tract with
        /// `AnatomyConfig::tongue_model` 1): the first...
        pub const MODE_1: usize = 7;
        /// ...the second...
        pub const MODE_2: usize = 8;
        /// ...and the third. A tract with the modal body takes them, when
        /// all three are set, in place of the tongue's place and height.
        pub const MODE_3: usize = 9;
    }
    /// [`Tier::TongueTip`](super::Tier::TongueTip).
    pub mod tip {
        /// Constriction location.
        pub const LOCATION: usize = 0;
        /// Constriction degree.
        pub const DEGREE: usize = 1;
    }
    /// [`Tier::LipsAndJaw`](super::Tier::LipsAndJaw).
    pub mod lips {
        /// Where the lips round (tract index).
        pub const APERTURE_LOCATION: usize = 0;
        /// Lip aperture of rounding (touch diameter; not a closure).
        pub const APERTURE: usize = 1;
        /// Lip protrusion, cm (a slot for a tract whose length changes; the
        /// present tract ignores it).
        pub const PROTRUSION: usize = 2;
        /// Jaw opening (a slot: the present jaw follows the tongue body).
        pub const JAW: usize = 3;
        /// A bilabial or labiodental constriction: place...
        pub const CLOSURE_LOCATION: usize = 4;
        /// ...and degree.
        pub const CLOSURE_DEGREE: usize = 5;
    }
}

/// Flags a gesture carries to the tract, for what a target value alone
/// does not say.
pub mod flag {
    /// A tongue tip gesture that is a lateral (/l/): the air goes round the
    /// sides of the tongue, and the gap is never one meant to hiss.
    pub const LATERAL: u8 = 1;
    /// A lip gesture that is the consonant's own (/ʃ ʒ tʃ dʒ r w/), moving
    /// at the lips' speed rather than with a vowel's shape.
    pub const OWN_LIPS: u8 = 2;
    /// A tongue body gesture of a consonant with a body of its own (/k g ŋ
    /// l r/): the back of the tongue moves it, at its speed.
    pub const OWN_BODY: u8 = 4;
    /// A velum gesture of a nasal: the mouth behind its closure is shaped as
    /// the nose's side branch.
    pub const NASAL: u8 = 8;
    /// A constriction that shares the mouth with another one (the lower lip
    /// rising to the teeth under a sibilant): it never takes the main
    /// constriction slot.
    pub const SECONDARY: u8 = 16;
    /// A gesture whose target is driven from outside as it plays (the
    /// pitch contour, the folds' settings): letting go of its layer on one
    /// tier ([`Score::end_layer`](super::Score::end_layer)) leaves it be.
    pub const DRIVEN: u8 = 32;
}

/// The values a gesture sets on its tier's variables; the others it leaves
/// to the other gestures and the tier's neutral attractor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Target {
    values: [f32; VARIABLES],
    mask: u16,
}

impl Default for Target {
    fn default() -> Self {
        Self::NONE
    }
}

impl Target {
    /// A target that sets nothing.
    pub const NONE: Self = Self {
        values: [0.0; VARIABLES],
        mask: 0,
    };

    /// This target with variable `v` set to `value` (out-of-range `v` and
    /// values that are not finite are ignored).
    #[must_use]
    pub const fn with(mut self, v: usize, value: f32) -> Self {
        if v < VARIABLES && value.is_finite() {
            self.values[v] = value;
            self.mask |= 1 << v;
        }
        self
    }

    /// Set variable `v` (see [`with`](Self::with)).
    pub fn set(&mut self, v: usize, value: f32) {
        *self = self.with(v, value);
    }

    /// Leave variable `v` unset.
    pub fn clear(&mut self, v: usize) {
        if v < VARIABLES {
            self.mask &= !(1 << v);
        }
    }

    /// The value set on variable `v`, if any.
    #[must_use]
    pub const fn get(&self, v: usize) -> Option<f32> {
        if v < VARIABLES && self.mask & (1 << v) != 0 {
            Some(self.values[v])
        } else {
            None
        }
    }

    /// Whether this target sets anything.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.mask == 0
    }
}

/// When a gesture acts, in seconds from the moment it is given to the
/// voice ([`Trombone::push_gesture`](crate::Trombone::push_gesture)): its
/// activation rises from `0` at `start` to `1` over `ramp_in`, holds, and
/// falls from `end` to `0` over `ramp_out`, each ramp a smooth step
/// (raised cosine). A ramp of `0` is a step. `end` may be infinite: the
/// gesture then lasts until it is ended.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Window {
    /// Seconds from now the gesture starts.
    pub start: f32,
    /// Seconds from now it starts to let go (may be `f32::INFINITY`).
    pub end: f32,
    /// Seconds its activation takes to rise.
    pub ramp_in: f32,
    /// Seconds it takes to fall.
    pub ramp_out: f32,
}

impl Window {
    /// From now on, until ended: a step.
    pub const NOW: Self = Self {
        start: 0.0,
        end: f32::INFINITY,
        ramp_in: 0.0,
        ramp_out: 0.0,
    };

    /// From `start` to `end` seconds from now, as steps.
    #[must_use]
    pub const fn between(start: f32, end: f32) -> Self {
        Self {
            start,
            end,
            ramp_in: 0.0,
            ramp_out: 0.0,
        }
    }

    /// This window with ramps of `ramp_in` and `ramp_out` seconds.
    #[must_use]
    pub const fn ramps(mut self, ramp_in: f32, ramp_out: f32) -> Self {
        self.ramp_in = ramp_in;
        self.ramp_out = ramp_out;
        self
    }
}

/// A smooth step from `0` at `x ≤ 0` to `1` at `x ≥ 1`.
fn smooth(x: f32) -> f32 {
    if x <= 0.0 {
        0.0
    } else if x >= 1.0 {
        1.0
    } else {
        x * x * (3.0 - 2.0 * x)
    }
}

/// Activation `0‥1` at `t` of a gesture active from `start` to `end`, all on
/// the score's clock.
#[allow(clippy::cast_possible_truncation)]
fn activation(t: f64, start: f64, end: f64, ramp_in: f32, ramp_out: f32) -> f32 {
    if t < start {
        return 0.0;
    }
    let rise = if ramp_in > 0.0 {
        smooth(((t - start) / f64::from(ramp_in)) as f32)
    } else {
        1.0
    };
    if t < end {
        return rise;
    }
    let fall = if ramp_out > 0.0 {
        1.0 - smooth(((t - end) / f64::from(ramp_out)) as f32)
    } else {
        0.0
    };
    rise * fall
}

/// One gesture: a target for one tier's variables, the dynamics that move
/// the tier towards it, its weight against the other gestures on the tier
/// and when it acts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gesture {
    /// The tier it acts on.
    pub tier: Tier,
    /// The values it sets (see [`var`]).
    pub target: Target,
    /// Natural frequency of the mass-spring that moves the tier towards it,
    /// rad/s.
    pub stiffness: f32,
    /// Damping ratio: `1` critically damped.
    pub damping: f32,
    /// Weight against the other gestures active on its tier (Saltzman &
    /// Munhall's blending strength): a weak gesture (a schwa) is coloured by
    /// strong ones beside it.
    pub dominance: f32,
    /// When it acts.
    pub window: Window,
    /// What the target alone does not say (see [`flag`]).
    pub flags: u8,
    /// Who scheduled it: the speech front end plays its own gestures on
    /// layer `0` ([`LAYER_SEGMENT`]) and its overlaps on `1`
    /// ([`LAYER_SCORE`]); `2` and up are free (an outside controller).
    pub layer: u8,
}

/// The layer the [`Speaker`](crate::speech::Speaker) plays each segment's
/// own targets on.
pub const LAYER_SEGMENT: u8 = 0;
/// The layer of the gestures a compiled utterance's score adds.
pub const LAYER_SCORE: u8 = 1;
/// The layer [`Trombone::push_gesture`](crate::Trombone::push_gesture)
/// gives a gesture that names none (`layer` `0` or `1` is moved here).
pub const LAYER_EXTERNAL: u8 = 2;

impl Gesture {
    /// A gesture on `tier` towards `target`: critically damped at the
    /// tier's [default stiffness](Tier::default_stiffness), dominance `1`,
    /// from now on until ended.
    #[must_use]
    pub const fn new(tier: Tier, target: Target) -> Self {
        Self {
            tier,
            target,
            stiffness: tier.default_stiffness(),
            damping: 1.0,
            dominance: 1.0,
            window: Window::NOW,
            flags: 0,
            layer: LAYER_EXTERNAL,
        }
    }

    /// With natural frequency `w` rad/s.
    #[must_use]
    pub const fn stiffness(mut self, w: f32) -> Self {
        self.stiffness = w;
        self
    }

    /// With damping ratio `zeta`.
    #[must_use]
    pub const fn damping(mut self, zeta: f32) -> Self {
        self.damping = zeta;
        self
    }

    /// With dominance `d`.
    #[must_use]
    pub const fn dominance(mut self, d: f32) -> Self {
        self.dominance = d;
        self
    }

    /// Acting in `window`.
    #[must_use]
    pub const fn window(mut self, window: Window) -> Self {
        self.window = window;
        self
    }

    /// With `flags` (see [`flag`]).
    #[must_use]
    pub const fn flags(mut self, flags: u8) -> Self {
        self.flags = flags;
        self
    }

    /// On `layer`.
    #[must_use]
    pub const fn layer(mut self, layer: u8) -> Self {
        self.layer = layer;
        self
    }

    /// The same gesture with its numbers kept in their ranges: stiffness
    /// `0.1‥2000` rad/s, damping `0.3‥2`, dominance `0.001‥1000`, times not
    /// negative. `None` if something is not a number.
    fn sanitized(mut self) -> Option<Self> {
        let finite = self.stiffness.is_finite()
            && self.damping.is_finite()
            && self.dominance.is_finite()
            && self.window.start.is_finite()
            && !self.window.end.is_nan()
            && self.window.ramp_in.is_finite()
            && self.window.ramp_out.is_finite();
        if !finite {
            return None;
        }
        self.stiffness = self.stiffness.clamp(0.1, 2000.0);
        self.damping = self.damping.clamp(0.3, 2.0);
        self.dominance = self.dominance.clamp(0.001, 1000.0);
        self.window.start = self.window.start.max(0.0);
        self.window.end = self.window.end.max(self.window.start);
        self.window.ramp_in = self.window.ramp_in.max(0.0);
        self.window.ramp_out = self.window.ramp_out.max(0.0);
        Some(self)
    }
}

/// A handle on a gesture in a [`Score`], to change its target or end it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GestureId(u32);

/// What one tier is doing, for a display: in `0‥1` throughout.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TierActivity {
    /// How strongly gestures act on the tier: their activations weighted
    /// by dominance, summed, at most `1`.
    pub activation: f32,
    /// The tier's main variable as the blended target has it, `0‥1`: the
    /// glottis's adduction (`1` the folds together and vibrating), the
    /// pitch (50 to 400 Hz on a log scale), the velar port's opening, the
    /// tongue body's height, the closeness of the tongue tip's and of the
    /// lips' constrictions (`1` shut).
    pub target: f32,
    /// Where the tier's main variable is now, on the same scale.
    pub value: f32,
    /// How many gestures are active on the tier.
    pub gestures: u8,
}

/// The blend of the gestures active on one tier at one moment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Blend {
    /// Each variable's blended target; `NaN` where nothing sets it and the
    /// tier's neutral holds it where it is (see [`Tier::neutral`]), or where
    /// nothing has set it and its neutral is a constriction let go.
    pub values: [f32; VARIABLES],
    /// The blended natural frequency, rad/s.
    pub stiffness: f32,
    /// The blended damping ratio.
    pub damping: f32,
    /// Summed activation times dominance, at most `1`.
    pub activation: f32,
    /// Active gestures.
    pub gestures: u8,
    /// The flags of the active gestures at least half as strong as the
    /// strongest.
    pub flags: u8,
    /// When the strongest active gesture started, on the score's clock.
    pub since: f64,
    /// For each variable, the flags of the strongest gesture setting it
    /// and when it started.
    pub setter: [(u8, f64); VARIABLES],
}

impl Blend {
    /// The blended value of variable `v`, if anything sets it.
    #[must_use]
    pub fn get(&self, v: usize) -> Option<f32> {
        self.values.get(v).copied().filter(|x| !x.is_nan())
    }
}

#[derive(Clone, Copy, Debug)]
struct Entry {
    gesture: Gesture,
    start: f64,
    end: f64,
    id: u32,
}

impl Entry {
    fn activation(&self, t: f64) -> f32 {
        let w = &self.gesture.window;
        activation(t, self.start, self.end, w.ramp_in, w.ramp_out)
    }

    /// Whether it has let go entirely by `t`.
    fn finished(&self, t: f64) -> bool {
        t >= self.end + f64::from(self.gesture.window.ramp_out)
    }
}

/// The task-space state of one tier: each variable and its velocity.
#[derive(Clone, Copy, Debug)]
struct TierState {
    x: [f32; VARIABLES],
    v: [f32; VARIABLES],
}

impl TierState {
    const REST: Self = Self {
        x: [f32::NAN; VARIABLES],
        v: [0.0; VARIABLES],
    };
}

/// The gestures of an utterance, sorted by when they start, of which at
/// most [`Score::CAPACITY`] are held at once (those that have let go are
/// dropped), and the dynamics of the six tiers. Fixed size: nothing here
/// allocates.
#[derive(Clone, Debug)]
pub struct Score {
    entries: [Option<Entry>; Score::CAPACITY],
    len: usize,
    next_id: u32,
    clock: f64,
    state: [TierState; Tier::COUNT],
    /// Tiers that had a gesture at the last look (see [`Score::engaged`]).
    engaged: u8,
}

impl Default for Score {
    fn default() -> Self {
        Self::new()
    }
}

impl Score {
    /// The most gestures held at once, pending and active together.
    pub const CAPACITY: usize = 64;

    /// An empty score at time `0`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: [None; Self::CAPACITY],
            len: 0,
            next_id: 1,
            clock: 0.0,
            state: [TierState::REST; Tier::COUNT],
            engaged: 0,
        }
    }

    /// Seconds on the score's clock.
    #[must_use]
    pub fn clock(&self) -> f64 {
        self.clock
    }

    /// Gestures held, pending and active.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether no gesture is held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn entries(&self) -> impl Iterator<Item = &Entry> {
        self.entries[..self.len].iter().flatten()
    }

    /// Add `gesture`, its window counted from now. Returns its handle, or
    /// `None` if the score is full or the gesture is not made of numbers.
    pub fn push(&mut self, gesture: Gesture) -> Option<GestureId> {
        let g = gesture.sanitized()?;
        if self.len == Self::CAPACITY {
            // Make room by dropping what has let go.
            self.prune();
            if self.len == Self::CAPACITY {
                return None;
            }
        }
        let start = self.clock + f64::from(g.window.start);
        let end = if g.window.end.is_finite() {
            self.clock + f64::from(g.window.end)
        } else {
            f64::INFINITY
        };
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1).max(1);
        // Kept sorted by start: later starts after earlier ones, equal
        // starts in the order they came.
        let mut at = self.len;
        while at > 0 && self.entries[at - 1].is_some_and(|e| e.start > start) {
            self.entries[at] = self.entries[at - 1];
            at -= 1;
        }
        self.entries[at] = Some(Entry {
            gesture: g,
            start,
            end,
            id,
        });
        self.len += 1;
        self.engaged |= 1 << g.tier.index();
        Some(GestureId(id))
    }

    fn find_mut(&mut self, id: GestureId) -> Option<&mut Entry> {
        self.entries[..self.len]
            .iter_mut()
            .flatten()
            .find(|e| e.id == id.0)
    }

    /// Give gesture `id` a new target, at once. Returns whether it is still
    /// held.
    pub fn retarget(&mut self, id: GestureId, target: Target) -> bool {
        self.find_mut(id)
            .map(|e| e.gesture.target = target)
            .is_some()
    }

    /// Change one variable of gesture `id`'s target (see
    /// [`retarget`](Self::retarget)).
    pub fn retarget_var(&mut self, id: GestureId, v: usize, value: f32) -> bool {
        self.find_mut(id)
            .map(|e| e.gesture.target.set(v, value))
            .is_some()
    }

    /// The tier of gesture `id`, while it is held.
    #[must_use]
    pub fn tier_of(&self, id: GestureId) -> Option<Tier> {
        self.entries()
            .find(|e| e.id == id.0)
            .map(|e| e.gesture.tier)
    }

    /// The target of gesture `id`, while it is held.
    #[must_use]
    pub fn target_of(&self, id: GestureId) -> Option<Target> {
        self.entries()
            .find(|e| e.id == id.0)
            .map(|e| e.gesture.target)
    }

    /// Let gesture `id` go from now, over `ramp_out` seconds. Returns
    /// whether it was held.
    pub fn end(&mut self, id: GestureId, ramp_out: f32) -> bool {
        let now = self.clock;
        self.find_mut(id)
            .map(|e| {
                if e.end > now {
                    e.end = now.max(e.start);
                    e.gesture.window.ramp_out = ramp_out.max(0.0);
                }
            })
            .is_some()
    }

    /// Let every gesture on `layer` (on `tier`, or on every tier for
    /// `None`) go from now, at once, but the [driven](flag::DRIVEN) ones;
    /// those not started yet are dropped.
    pub fn end_layer(&mut self, layer: u8, tier: Option<Tier>) {
        self.end_where(|g| {
            g.layer == layer && tier.is_none_or(|t| t == g.tier) && g.flags & flag::DRIVEN == 0
        });
    }

    /// Let every gesture on `layer` go from now, at once, the driven ones
    /// too.
    pub fn clear_layer(&mut self, layer: u8) {
        self.end_where(|g| g.layer == layer);
    }

    fn end_where(&mut self, which: impl Fn(&Gesture) -> bool) {
        let now = self.clock;
        for e in self.entries[..self.len].iter_mut().flatten() {
            if which(&e.gesture) && e.end > now {
                e.end = now.max(e.start);
                e.gesture.window.ramp_out = 0.0;
            }
        }
        self.prune();
    }

    /// Set variables of the latest acting gesture on `layer` and `tier` that
    /// is neither [driven](flag::DRIVEN) nor [secondary](flag::SECONDARY):
    /// `sets` as `(variable, value)`. Returns whether there was one.
    pub fn retarget_layer(&mut self, layer: u8, tier: Tier, sets: &[(usize, f32)]) -> bool {
        self.retarget_where(
            layer,
            tier,
            |g| g.flags & (flag::DRIVEN | flag::SECONDARY) == 0,
            sets,
        )
    }

    /// Set variables of the latest acting gesture on `layer` and `tier` that
    /// `which` picks (see [`retarget_layer`](Self::retarget_layer)).
    pub fn retarget_where(
        &mut self,
        layer: u8,
        tier: Tier,
        which: impl Fn(&Gesture) -> bool,
        sets: &[(usize, f32)],
    ) -> bool {
        let now = self.clock;
        let found = self.entries[..self.len]
            .iter_mut()
            .flatten()
            .rev()
            .find(|e| {
                e.gesture.layer == layer
                    && e.gesture.tier == tier
                    && e.end > now
                    && which(&e.gesture)
            });
        let Some(e) = found else {
            return false;
        };
        for &(v, x) in sets {
            e.gesture.target.set(v, x);
        }
        true
    }

    /// Let go, at once, of the gestures on `layer` and `tier` that `which`
    /// picks.
    pub fn end_layer_where(&mut self, layer: u8, tier: Tier, which: impl Fn(&Gesture) -> bool) {
        self.end_where(|g| g.layer == layer && g.tier == tier && which(g));
    }

    /// Whether gesture `id` is held and has not been let go.
    #[must_use]
    pub fn is_acting(&self, id: GestureId) -> bool {
        self.entries().any(|e| e.id == id.0 && e.end > self.clock)
    }

    /// Drop the gestures that have let go.
    fn prune(&mut self) {
        let now = self.clock;
        let mut kept = 0;
        for k in 0..self.len {
            if let Some(e) = self.entries[k]
                && !(e.finished(now) || (e.end <= e.start && now >= e.start))
            {
                self.entries[kept] = Some(e);
                kept += 1;
            }
        }
        for e in &mut self.entries[kept..self.len] {
            *e = None;
        }
        self.len = kept;
    }

    /// Move the clock on by `dt` seconds.
    pub fn tick(&mut self, dt: f64) {
        self.clock += dt;
    }

    /// The blend of the gestures active on `tier` now (Saltzman & Munhall):
    /// each variable's target is the mean of the targets set on it,
    /// weighted by each gesture's activation times its dominance, with the
    /// tier's neutral attractor taking the share of activation the gestures
    /// leave (`1` less their summed activation); the stiffness and the
    /// damping are blended the same way over all the tier's gestures. With
    /// one gesture fully active the blend is that gesture's own numbers,
    /// exactly.
    #[must_use]
    pub fn blend(&self, tier: Tier) -> Blend {
        let now = self.clock;
        let mut out = Blend {
            values: [f32::NAN; VARIABLES],
            stiffness: tier.default_stiffness(),
            damping: 1.0,
            activation: 0.0,
            gestures: 0,
            flags: 0,
            since: f64::NEG_INFINITY,
            setter: [(0, f64::NEG_INFINITY); VARIABLES],
        };
        // The strongest gesture is the reference the means are taken
        // about, so that a single gesture's numbers come out exactly.
        let mut strongest: Option<(&Entry, f32)> = None;
        let mut sum_a = 0.0f32;
        let mut sum_w = 0.0f32;
        for e in self.entries().filter(|e| e.gesture.tier == tier) {
            let a = e.activation(now);
            if a <= 0.0 {
                continue;
            }
            let w = a * e.gesture.dominance;
            out.gestures = out.gestures.saturating_add(1);
            sum_a += a;
            sum_w += w;
            if strongest.is_none_or(|(_, s)| w > s) {
                strongest = Some((e, w));
            }
        }
        out.activation = sum_w.min(1.0);
        let neutral_w = (1.0 - sum_a).max(0.0);
        let Some((top, _)) = strongest else {
            // Nothing active: each variable at its neutral, if that is a
            // value of its own (the velum shut); held where it is, or a
            // constriction gone, otherwise.
            for v in 0..VARIABLES {
                out.values[v] = match tier.neutral(v) {
                    Some(n) if n != OPEN => n,
                    _ => f32::NAN,
                };
            }
            return out;
        };
        // The flags of every gesture at least half as strong as the
        // strongest.
        let top_w = strongest.map_or(0.0, |(_, w)| w);
        for e in self.entries().filter(|e| e.gesture.tier == tier) {
            if e.activation(now) * e.gesture.dominance >= 0.5 * top_w {
                out.flags |= e.gesture.flags;
            }
        }
        out.since = top.start;
        // Stiffness and damping over every gesture of the tier.
        let (w0, z0) = (top.gesture.stiffness, top.gesture.damping);
        let (mut dw, mut dz) = (0.0f32, 0.0f32);
        for e in self.entries().filter(|e| e.gesture.tier == tier) {
            let w = e.activation(now) * e.gesture.dominance;
            if w > 0.0 && e.id != top.id {
                dw += w * (e.gesture.stiffness - w0);
                dz += w * (e.gesture.damping - z0);
            }
        }
        dw += neutral_w * (tier.default_stiffness() - w0);
        dz += neutral_w * (1.0 - z0);
        let total = sum_w + neutral_w;
        out.stiffness = w0 + dw / total;
        out.damping = z0 + dz / total;
        for v in 0..VARIABLES {
            (out.values[v], out.setter[v]) = self.blend_variable(tier, v);
        }
        out
    }

    /// The blended target of `tier`'s variable `v` now (see
    /// [`blend`](Self::blend)).
    fn blend_variable(&self, tier: Tier, v: usize) -> (f32, (u8, f64)) {
        let now = self.clock;
        // The reference: the strongest gesture setting it.
        let mut reference: Option<(f32, f32, u8, f64)> = None;
        let mut set_a = 0.0f32;
        let mut set_w = 0.0f32;
        for e in self.entries().filter(|e| e.gesture.tier == tier) {
            let Some(x) = e.gesture.target.get(v) else {
                continue;
            };
            let a = e.activation(now);
            if a <= 0.0 {
                continue;
            }
            let w = a * e.gesture.dominance;
            set_a += a;
            set_w += w;
            if reference.is_none_or(|(_, s, _, _)| w > s) {
                reference = Some((x, w, e.gesture.flags, e.start));
            }
        }
        let Some((x0, _, flags, since)) = reference else {
            // Nothing sets it: its neutral, if it has one that is not a
            // constriction let go.
            let x = match tier.neutral(v) {
                Some(n) if n != OPEN => n,
                _ => f32::NAN,
            };
            return (x, (0, f64::NEG_INFINITY));
        };
        let mut dx = 0.0f32;
        for e in self.entries().filter(|e| e.gesture.tier == tier) {
            let Some(x) = e.gesture.target.get(v) else {
                continue;
            };
            let w = e.activation(now) * e.gesture.dominance;
            if w > 0.0 {
                dx += w * (x - x0);
            }
        }
        let mut total = set_w;
        if let Some(n) = tier.neutral(v) {
            let nw = (1.0 - set_a).max(0.0);
            dx += nw * (n - x0);
            total += nw;
        }
        (x0 + dx / total, (flags, since))
    }

    /// Whether `tier` has had a gesture since the score last let go of all
    /// of its gestures (see [`advance`](Self::advance)): while it has, the
    /// voice writes the tier's blend to the tract.
    #[must_use]
    pub fn engaged(&self, tier: Tier) -> bool {
        self.engaged & (1 << tier.index()) != 0
    }

    /// The [engaged](Self::engaged) tiers as a bit set by [`Tier::index`].
    #[must_use]
    pub fn engaged_tiers(&self) -> u8 {
        self.engaged
    }

    /// Integrate each tier's dynamics over `dt` seconds towards its blend
    /// now, exactly (the blend held over the step), and let go of the
    /// gestures that have ended. A tier left with no gesture is no longer
    /// [engaged](Self::engaged) after this. Returns the tiers that were
    /// engaged before it, as a bit set by [`Tier::index`].
    pub fn advance(&mut self, dt: f32) -> u8 {
        let before = self.engaged;
        for tier in Tier::ALL {
            if before & (1 << tier.index()) == 0 {
                continue;
            }
            let b = self.blend(tier);
            let s = &mut self.state[tier.index()];
            for v in 0..VARIABLES {
                let target = b.values[v];
                if target.is_nan() {
                    s.v[v] = 0.0;
                    if tier.neutral(v) == Some(OPEN) {
                        s.x[v] = OPEN;
                    }
                    continue;
                }
                if s.x[v].is_nan() {
                    s.x[v] = target;
                    s.v[v] = 0.0;
                    continue;
                }
                let (e, vel) = spring_step(s.x[v] - target, s.v[v], b.stiffness, b.damping, dt);
                s.x[v] = target + e;
                s.v[v] = vel;
            }
        }
        self.prune();
        let mut now = 0u8;
        for e in self.entries() {
            now |= 1 << e.gesture.tier.index();
        }
        self.engaged = now;
        before
    }

    /// Put every tier where its blend is now, at rest.
    pub fn settle(&mut self) {
        for tier in Tier::ALL {
            let b = self.blend(tier);
            let s = &mut self.state[tier.index()];
            for v in 0..VARIABLES {
                if !b.values[v].is_nan() {
                    s.x[v] = b.values[v];
                }
                s.v[v] = 0.0;
            }
        }
    }

    /// Put `tier`'s variable `v` at `x`, at rest, in task space (where the
    /// plant shows it is: a constriction appearing starts from the opening
    /// the tract has there, not from [`OPEN`]).
    pub fn seed(&mut self, tier: Tier, v: usize, x: f32) {
        if v < VARIABLES && x.is_finite() {
            let s = &mut self.state[tier.index()];
            s.x[v] = x;
            s.v[v] = 0.0;
        }
    }

    /// Where `tier`'s variable `v` is now in task space (`None` before
    /// anything has set it).
    #[must_use]
    pub fn state(&self, tier: Tier, v: usize) -> Option<f32> {
        self.state[tier.index()]
            .x
            .get(v)
            .copied()
            .filter(|x| !x.is_nan())
    }

    /// What each tier is doing, in [`Tier::ALL`] order.
    #[must_use]
    pub fn activity(&self) -> [TierActivity; Tier::COUNT] {
        let mut out = [TierActivity::default(); Tier::COUNT];
        for tier in Tier::ALL {
            let b = self.blend(tier);
            let s = &self.state[tier.index()];
            let target = scale(tier, &b.values);
            let value = scale(tier, &s.x);
            out[tier.index()] = TierActivity {
                activation: b.activation,
                target: target.unwrap_or(0.0),
                value: value.or(target).unwrap_or(0.0),
                gestures: b.gestures,
            };
        }
        out
    }
}

/// A tier's main variable on the `0‥1` scale of [`TierActivity`], from its
/// variables `x` (`None` where they are not set).
fn scale(tier: Tier, x: &[f32; VARIABLES]) -> Option<f32> {
    use var::{body, glottis, larynx, lips, tip, velum};
    let get = |v: usize| Some(x[v]).filter(|x| !x.is_nan());
    let closeness = |d: f32| (1.0 - d / 1.5).clamp(0.0, 1.0);
    let y = match tier {
        Tier::GlottalAdduction => {
            let spread = (get(glottis::SPREAD).unwrap_or(0.0) / 0.3).clamp(0.0, 1.0);
            let voiced = get(glottis::VOICING)?;
            if voiced >= 0.5 {
                1.0 - spread
            } else {
                0.6 * (1.0 - spread)
            }
        }
        Tier::LarynxAndPitch => {
            let hz = get(larynx::PITCH)?.max(1.0);
            ln(hz / 50.0) / ln(8.0)
        }
        Tier::Velum => get(velum::PORT)?,
        Tier::TongueBody => {
            let height = (3.5 - get(body::DEGREE)?) / (3.5 - 2.05);
            let dorsal = get(body::DORSAL_DEGREE).map_or(0.0, closeness);
            height.max(dorsal)
        }
        Tier::TongueTip => closeness(get(tip::DEGREE).unwrap_or(OPEN)),
        Tier::LipsAndJaw => {
            let round = get(lips::APERTURE).unwrap_or(OPEN);
            let shut = get(lips::CLOSURE_DEGREE).unwrap_or(OPEN);
            closeness(round.min(shut))
        }
    };
    Some(y.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec::Vec;

    const DT: f32 = 128.0 / 48_000.0;

    fn tip(degree: f32) -> Target {
        Target::NONE
            .with(var::tip::LOCATION, 36.0)
            .with(var::tip::DEGREE, degree)
    }

    /// The tip's degree in task space every `dt` for `seconds`.
    fn run(score: &mut Score, dt: f32, seconds: f32) -> Vec<f32> {
        let mut out = Vec::new();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let steps = (seconds / dt).round() as usize;
        for _ in 0..steps {
            score.advance(dt);
            score.tick(f64::from(dt));
            out.push(score.state(Tier::TongueTip, var::tip::DEGREE).unwrap());
        }
        out
    }

    #[test]
    fn tiers_are_named_as_the_owner_wrote_them() {
        let names: Vec<_> = Tier::ALL.iter().map(|t| t.name()).collect();
        assert_eq!(
            names,
            [
                "GlottalAdduction",
                "LarynxAndPitch",
                "Velum",
                "TongueBody",
                "TongueTip",
                "LipsAndJaw"
            ]
        );
        for (k, t) in Tier::ALL.iter().enumerate() {
            assert_eq!(t.index(), k);
            assert!(t.main_variable() < t.variables().len());
        }
    }

    #[test]
    fn a_step_is_critically_damped() {
        // From 1.0 to 0.0 at 88 rad/s: half way at 1.678 / w (19 ms), no
        // overshoot.
        let mut s = Score::new();
        s.push(Gesture::new(Tier::TongueTip, tip(1.0)).layer(0));
        s.settle();
        let id = s
            .push(Gesture::new(Tier::TongueTip, tip(0.0)).dominance(1000.0))
            .unwrap();
        let w = 88.0;
        let x = run(&mut s, DT, 0.2);
        let half = x.iter().position(|&d| d <= 0.5).unwrap();
        #[allow(clippy::cast_precision_loss)]
        let t_half = (half + 1) as f32 * DT;
        assert!((t_half - 1.678 / w).abs() < DT, "{t_half}");
        assert!(x.iter().all(|&d| d >= -1e-4), "no overshoot");
        assert!(x.last().unwrap().abs() < 1e-3);
        assert!(s.end(id, 0.0));
    }

    #[test]
    fn one_gesture_is_its_own_blend_exactly() {
        let mut s = Score::new();
        let t = Target::NONE
            .with(var::body::LOCATION, 21.37)
            .with(var::body::DEGREE, 2.713);
        s.push(
            Gesture::new(Tier::TongueBody, t)
                .dominance(0.37)
                .stiffness(11.3),
        );
        let b = s.blend(Tier::TongueBody);
        assert_eq!(b.get(var::body::LOCATION), Some(21.37));
        assert_eq!(b.get(var::body::DEGREE), Some(2.713));
        assert_eq!(b.stiffness, 11.3);
        // Untouched variables are held (no neutral) or let go.
        assert_eq!(b.get(var::body::ROOT), None);
        assert_eq!(b.get(var::body::DORSAL_DEGREE), None);
    }

    #[test]
    fn overlapping_gestures_blend_by_dominance() {
        let mut s = Score::new();
        let at = |d: f32| Target::NONE.with(var::body::DEGREE, d);
        s.push(Gesture::new(Tier::TongueBody, at(2.0)).dominance(3.0));
        s.push(Gesture::new(Tier::TongueBody, at(3.0)).dominance(1.0));
        let b = s.blend(Tier::TongueBody);
        // 3:1 towards 2.0; both fully active, so the neutral has no share.
        assert!((b.get(var::body::DEGREE).unwrap() - 2.25).abs() < 1e-5);
        assert_eq!(b.gestures, 2);
        assert!((b.activation - 1.0).abs() < 1e-6);
        // A weak gesture alone, half active, shares with the neutral: a
        // tip constriction half let go is half way open.
        let mut s = Score::new();
        s.push(
            Gesture::new(Tier::TongueTip, tip(0.0))
                .window(Window::between(0.0, 0.0).ramps(0.0, 0.02)),
        );
        s.tick(0.01);
        let b = s.blend(Tier::TongueTip);
        assert!((b.get(var::tip::DEGREE).unwrap() - 0.5 * OPEN).abs() < 1e-4);
        assert_eq!(b.get(var::tip::LOCATION), Some(36.0));
    }

    #[test]
    fn the_neutral_attractor_takes_over_when_gestures_end() {
        let mut s = Score::new();
        let port = Target::NONE.with(var::velum::PORT, 1.0);
        s.push(Gesture::new(Tier::Velum, port).window(Window::between(0.0, 0.05)));
        s.push(Gesture::new(Tier::TongueTip, tip(0.0)).window(Window::between(0.0, 0.05)));
        assert!(s.engaged(Tier::Velum));
        s.tick(0.06);
        // The velum is drawn shut, the tip constriction is gone.
        assert_eq!(s.blend(Tier::Velum).get(var::velum::PORT), Some(0.0));
        assert_eq!(s.blend(Tier::TongueTip).get(var::tip::DEGREE), None);
        s.advance(DT);
        assert!(s.is_empty());
        assert!(!s.engaged(Tier::Velum));
    }

    #[test]
    fn windows_start_and_end_on_time() {
        let mut s = Score::new();
        s.push(Gesture::new(Tier::TongueTip, tip(0.0)).window(Window::between(0.02, 0.04)));
        assert_eq!(s.blend(Tier::TongueTip).gestures, 0);
        s.tick(0.02);
        assert_eq!(s.blend(Tier::TongueTip).gestures, 1);
        s.tick(0.02);
        assert_eq!(s.blend(Tier::TongueTip).gestures, 0);
        // Ramps are smooth steps.
        assert!((activation(0.5, 0.0, 10.0, 1.0, 0.0) - 0.5).abs() < 1e-6);
        assert!((activation(10.25, 0.0, 10.0, 0.0, 1.0) - 0.84375).abs() < 1e-6);
    }

    #[test]
    fn the_motion_does_not_depend_on_the_block_length() {
        // The same gestures, integrated in blocks of 7.8, 2 and 0.5 ms,
        // agree wherever their clocks meet: the step is exact, and the
        // windows fall on the long blocks' boundaries (times in binary
        // fractions of a second, so that they fall exactly on the blocks of
        // every length).
        let long = 1.0 / 128.0;
        let make = || {
            let mut s = Score::new();
            s.push(Gesture::new(Tier::TongueTip, tip(1.2)));
            s.settle();
            s.push(
                Gesture::new(Tier::TongueTip, tip(0.0))
                    .dominance(100.0)
                    .window(Window::between(2.0 * long, 9.0 * long)),
            );
            s
        };
        let reference = run(&mut make(), long, 0.25);
        for div in [4usize, 16] {
            #[allow(clippy::cast_precision_loss)]
            let dt = long / div as f32;
            let fine = run(&mut make(), dt, 0.25);
            for (k, &x) in reference.iter().enumerate() {
                let y = fine[(k + 1) * div - 1];
                assert!((x - y).abs() < 2e-4, "block {k}: {x} against {y}");
            }
        }
    }

    #[test]
    fn a_full_score_refuses_and_lets_go() {
        let mut s = Score::new();
        for _ in 0..Score::CAPACITY {
            assert!(
                s.push(Gesture::new(Tier::Velum, Target::NONE).window(Window::between(0.0, 0.01)))
                    .is_some()
            );
        }
        assert!(s.push(Gesture::new(Tier::Velum, Target::NONE)).is_none());
        s.tick(0.02);
        assert!(s.push(Gesture::new(Tier::Velum, Target::NONE)).is_some());
        assert_eq!(s.len(), 1);
        // Garbage is refused.
        assert!(
            s.push(Gesture::new(Tier::Velum, Target::NONE).stiffness(f32::NAN))
                .is_none()
        );
    }

    #[test]
    fn the_score_stays_in_start_order() {
        let mut s = Score::new();
        for start in [0.3, 0.1, 0.2, 0.1] {
            s.push(
                Gesture::new(Tier::LipsAndJaw, Target::NONE).window(Window::between(start, 1.0)),
            );
        }
        let starts: Vec<f64> = s.entries().map(|e| e.start).collect();
        assert!(starts.windows(2).all(|w| w[0] <= w[1]), "{starts:?}");
    }

    #[test]
    fn activity_reads_the_main_variables() {
        let mut s = Score::new();
        s.push(Gesture::new(Tier::TongueTip, tip(0.0)));
        s.push(Gesture::new(
            Tier::Velum,
            Target::NONE.with(var::velum::PORT, 0.6),
        ));
        s.push(Gesture::new(
            Tier::LarynxAndPitch,
            Target::NONE.with(var::larynx::PITCH, 100.0),
        ));
        s.settle();
        let a = s.activity();
        assert!((a[Tier::TongueTip.index()].target - 1.0).abs() < 1e-6);
        assert!((a[Tier::Velum.index()].value - 0.6).abs() < 1e-6);
        assert!((a[Tier::LarynxAndPitch.index()].target - 1.0 / 3.0).abs() < 1e-4);
        assert_eq!(a[Tier::TongueTip.index()].gestures, 1);
        assert_eq!(a[Tier::LipsAndJaw.index()].gestures, 0);
        assert!(a.iter().all(|t| (0.0..=1.0).contains(&t.target)));
    }
}
