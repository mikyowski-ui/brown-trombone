//! Vocal tract: a 44-cell Kelly–Lochbaum waveguide with a 28-cell nasal
//! branch, plus the articulation model that shapes it (tongue, constrictions,
//! velum).
//!
//! The waveguide runs at twice [`INTERNAL_RATE`](crate::INTERNAL_RATE).
//! Everything here is fixed-size: no allocation, ever.

use crate::INTERNAL_RATE;
use crate::math::{cos, exp, lerp, move_towards, powf};
use crate::phys::Stream;
use crate::tuning::Tuning;
use core::f32::consts::PI;

/// Cells in the oral tract, glottis (0) to lips (`N - 1`).
pub const N: usize = 44;
/// Cells in the nasal tract.
pub const NOSE_LEN: usize = 28;
/// Cell where the nasal tract branches off (the velum).
pub const NOSE_START: usize = N - NOSE_LEN + 1;
/// First cell of the tongue blade.
pub const BLADE_START: usize = 10;
/// First cell of the tongue tip.
pub const TIP_START: usize = 32;
/// First cell of the lips.
pub const LIP_START: usize = 39;
/// Number of simultaneous constrictions.
pub const MAX_CONSTRICTIONS: usize = 4;

/// Valid tongue body positions (cell index, back to front).
pub const TONGUE_INDEX: core::ops::RangeInclusive<f32> = 12.0..=29.0;
/// Valid tongue body heights (diameter under the tongue; small is high).
pub const TONGUE_DIAMETER: core::ops::RangeInclusive<f32> = 2.05..=3.5;

const GLOTTAL_REFLECTION: f32 = 0.75;
const LIP_REFLECTION: f32 = -0.85;
const DAMPING: f32 = 0.999;
/// The original's articulator speed, diameter units per second.
pub(crate) const MOVEMENT_SPEED: f32 = 15.0;
const GRID_OFFSET: f32 = 1.7;
const VELUM_CLOSED: f32 = 0.01;
const VELUM_OPEN: f32 = 0.4;
const FRICATIVE_ATTACK: f32 = 0.1;
const MAX_TRANSIENTS: usize = 8;
const TRACT_RATE: f32 = 2.0 * INTERNAL_RATE;
/// The slot that holds lip rounding: a shaping of the vowel, never meant to
/// hiss, however narrow.
pub(crate) const ROUNDING_SLOT: usize = 1;
/// Rest diameter of the lip cells.
const LIP_REST: f32 = 1.5;
/// How far the lips part beyond [`LIP_REST`] when the jaw is fully open
/// (physical engine; see `Tract::jaw`), diameter units. The jaw lowers with
/// the tongue, so open vowels open the mouth too: real /a/ and /æ/ have lip
/// openings of 4 to 6 cm², twice the 2.8 cm² that rest lips give the
/// physical tract. With this the lips part at most 2.5 across (7.9 cm²); the
/// open vowels part them about 2.0 to 2.3 (5 to 6.6 cm²).
const JAW_OPENING: f32 = 1.0;
/// Lip rounding (touch diameter) at which the jaw no longer parts the lips
/// at all: that of /u/...
const JAW_ROUNDED: f32 = 0.65;
/// ...and at which it parts them fully: lips at rest.
const JAW_SPREAD: f32 = 1.8;
/// Consonant opening (touch diameter) at or below which the jaw is raised
/// fully: a fricative's gap; approximants raise it part way.
const JAW_CONSONANT: f32 = 0.75;
/// Time constant of the jaw, seconds: a heavier articulator than the lips.
/// A stop released into /a/ parts its lips first and the jaw follows, so
/// the burst comes out of a mouth that is still nearly closed.
const JAW_TIME: f32 = 0.02;
/// Lip cells over which the jaw's opening grows to its full width, from
/// [`LIP_START`] on: the corners of the mouth open less than its middle.
const JAW_RAMP: f32 = 3.0;

// ----- articulators of two speeds (physical engine) ----------------------
//
// The tongue body and the jaw are heavy; the tongue tip, the blade and the
// lips are light. A speaker moves the vowel shape slowly underneath the
// consonants and flicks the tip or the lips on top of it, so a /t/ is a
// quick touch on a tongue body still on its way to the next vowel, and the
// formants that come out of the release move with that body: the locus
// transitions. Each cell follows whichever articulator is shaping it.

/// Cells behind a tongue-tip constriction (towards the glottis) over which
/// its shoulder runs on the physical engine: the blade rises behind the tip.
const TIP_SHOULDER_BACK: f32 = 5.0;
/// Cells in front of it (towards the lips): the tip touches the ridge and
/// the mouth in front of it stays open, a small front cavity of its own.
const TIP_SHOULDER_FRONT: f32 = 1.5;
/// The same behind the ridge (/ʃ ʒ tʃ dʒ/, from [`TIP_START`] to
/// [`BLADE_PLACE`]): the blade makes a long channel, sloping down to the
/// teeth, as long in front as behind (the original's shoulder; a shorter
/// one takes a fifth off the hiss of /ʃ/).
const BLADE_SHOULDER_FRONT: f32 = 5.0;
/// Constriction index from which the tip, not the blade, makes it.
const BLADE_PLACE: f32 = 34.5;
/// The front shoulder of /s z/: their jet crosses a short, narrow channel
/// to the teeth (the jaw is raised), whose small cavity gives them their
/// high hiss.
const SIBILANT_SHOULDER_FRONT: f32 = 4.0;
/// Constriction index from which the tongue tip is at the teeth (/θ ð/):
/// nothing in front of it but the lips.
const DENTAL_PLACE: f32 = 38.5;
/// Shoulder of a closure of the lips, cells either way: the lips are the
/// last four cells of the tract, and closing them leaves the tongue alone.
const LIP_SHOULDER: f32 = 2.0;
/// Where the lips begin, as a constriction index (labials sit at 40.5 to
/// 41).
const LIP_PLACE: f32 = 40.0;
/// A cell is being shaped by a consonant when the consonant takes this
/// much off what the vowel shape alone would give it, diameter units.
const CARVED: f32 = 0.05;
/// A cell a consonant has let go is back with the tongue body once it is
/// this close to where the body wants it, diameter units.
const RETURNED: f32 = 0.08;

/// Which articulator moves a cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mover {
    /// The tongue body and the jaw: the vowel shape.
    Body,
    /// The back of the tongue making a consonant (/k g ŋ/).
    Dorsum,
    /// The tongue tip and blade, or the lips, making a consonant.
    Tip,
}

/// The articulator that makes a consonant constriction at `index`.
fn mover_of(index: f32) -> Mover {
    #[allow(clippy::cast_precision_loss)]
    if index >= (TIP_START - 2) as f32 {
        Mover::Tip
    } else {
        Mover::Dorsum
    }
}

// ----- the hiss regulator (physical engine) ------------------------------
//
// A speaker does not place the tongue by rule and hope: they listen, and
// shift it until the /s/ hisses and the /u/ does not. `Tract::regulate`
// does the same from what the air is doing (see `phys::Stream`).

/// Narrowest opening that is meant as a fricative; anything tighter is a
/// closure.
const HISS_NARROWEST: f32 = 0.28;
/// Widest opening that is meant as a fricative.
pub(crate) const HISS_WIDEST: f32 = 0.75;
/// Furthest the regulator will move a gap, either way (touch-diameter
/// units): a nudge, not a different sound.
pub(crate) const NUDGE: f32 = 0.15;
/// Cells that share one remembered nudge (a place of articulation).
const PLACE: usize = 2;
/// How fast a fricative gap that fails to hiss is widened, per second.
const WIDEN_RATE: f32 = 1.9;
/// How long a gap must fail to hiss before it is moved, seconds: the air
/// needs a moment to get going.
const PATIENCE: f32 = 0.02;
/// A gap is where it was sent once it is this close to its target and
/// this slow, diameter units and units per second.
const SETTLED: f32 = 0.03;
const SETTLED_SPEED: f32 = 2.0;
/// Share of the lung pressure that viscosity may take in a fricative gap
/// before the gap counts as choked.
const CHOKE: f32 = 0.25;
/// A stream at least this fast (relative to the speed lung pressure gives)
/// is held back by its own gap, not by the glottis...
const FAST: f32 = 0.8;
/// ...and it must then reach this Reynolds number to hiss fully: twice the
/// critical one.
const TURBULENT: f32 = 3400.0;
/// Noise of a jet in a vowel, relative to a free jet at lung-pressure
/// speed, above which the vowel is opened, and below which it is let back.
/// The first is 5 dB under what `examples/hiss.rs` and the gates call a
/// hissing vowel.
const VOWEL_HISS: f32 = 0.016;
const VOWEL_QUIET: f32 = 0.006;
/// How long a vowel must hiss before it is opened, seconds.
const VOWEL_PATIENCE: f32 = 0.03;
/// How fast a hissing vowel is opened and let back, per second.
const RELIEF_RATE: f32 = 1.5;
const RELAX_RATE: f32 = 0.3;
/// Half-width of the stretch that is opened, cells.
const RELIEF_REACH: usize = 4;

/// What a constriction is for.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Intent {
    /// Shut: a stop or a nasal.
    Closure,
    /// A gap meant to hiss.
    Hiss,
    /// A shaping of the tract that is not meant to hiss.
    Open,
}

fn intent(slot: usize, diameter: f32) -> Intent {
    if slot == ROUNDING_SLOT {
        if diameter <= 0.3 {
            Intent::Closure
        } else {
            Intent::Open
        }
    } else if diameter < HISS_NARROWEST {
        Intent::Closure
    } else if diameter <= HISS_WIDEST {
        Intent::Hiss
    } else {
        Intent::Open
    }
}

/// How many cells either side of `index` a constriction there reshapes.
#[allow(clippy::cast_precision_loss)]
fn reach(index: f32) -> f32 {
    if index < 25.0 {
        10.0
    } else if index >= TIP_START as f32 {
        5.0
    } else {
        10.0 - 5.0 * (index - 25.0) / (TIP_START as f32 - 25.0)
    }
}

// The damped articulators' natural frequency, how much faster they close
// than open, the speed at which a closure lets go (a spring like these
// peaks at 0.37 of `w·distance`; a release starts there) and how far past
// contact a closing articulator aims come from the voice's `Tuning`.

/// How cells move towards their targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Dynamics {
    /// The original's constant-speed ramps.
    Linear,
    /// Critically damped springs: articulators accelerate, coast and settle
    /// like tissue with inertia. Closures aim past contact and stop at it.
    Damped,
}

/// A narrowing of the tract made by the tongue tip, the lips, or the back of
/// the tongue, as a finger on the original's tract diagram.
///
/// `diameter` uses the original's touch scale: `≤ 0.3` closes the tract
/// (stops, nasals), about `0.35..0.7` is narrow enough for turbulence
/// (fricatives), larger values only shade the shape (approximants, lip
/// rounding).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Constriction {
    /// Cell index, `2.0..44.0`.
    pub index: f32,
    /// Opening at the constriction.
    pub diameter: f32,
}

impl Constriction {
    /// Construct a constriction.
    #[must_use]
    pub const fn new(index: f32, diameter: f32) -> Self {
        Self { index, diameter }
    }
}

#[derive(Clone, Copy, Debug)]
struct Slot {
    c: Constriction,
    alive: bool,
    /// Turbulence envelope, `0..1`, ramps on and off over 100 ms.
    fricative: f32,
    /// Seconds the gap has sat where it was sent without hissing as meant
    /// (hiss regulator).
    failing: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct Transient {
    position: usize,
    amplitude: f32,
    steps_left: u32,
}

/// Oral and nasal waveguides plus their articulation state.
#[derive(Clone, Debug)]
pub(crate) struct Tract {
    right: [f32; N],
    left: [f32; N],
    junction_right: [f32; N],
    junction_left: [f32; N + 1],
    reflection: [f32; N],
    new_reflection: [f32; N],
    max_amplitude: [f32; N],
    pub(crate) diameter: [f32; N],
    pub(crate) target: [f32; N],

    nose_right: [f32; NOSE_LEN],
    nose_left: [f32; NOSE_LEN],
    nose_junction_right: [f32; NOSE_LEN],
    nose_junction_left: [f32; NOSE_LEN + 1],
    nose_reflection: [f32; NOSE_LEN],
    pub(crate) nose_diameter: [f32; NOSE_LEN],
    nose_max_amplitude: [f32; NOSE_LEN],

    reflection_left: f32,
    reflection_right: f32,
    reflection_nose: f32,
    new_reflection_left: f32,
    new_reflection_right: f32,
    new_reflection_nose: f32,

    pub(crate) tongue_index: f32,
    pub(crate) tongue_diameter: f32,
    pub(crate) velum_target: f32,
    pub(crate) movement_speed: f32,
    pub(crate) dynamics: Dynamics,
    /// Whether the lips part with the tongue's opening, as a jaw lowered
    /// with the tongue body parts them (physical engine; see
    /// [`JAW_OPENING`]). The classic engine keeps the original's fixed lips.
    pub(crate) jaw: bool,
    /// How far the jaw parts the lips now (see [`Tract::jaw_opening`]): it
    /// follows where it is going with [`JAW_TIME`].
    jaw_open: f32,
    /// How the damped articulators move (the speeds of the voice's tuning).
    pub(crate) motion: Tuning,
    velocity: [f32; N],
    /// The vowel shape alone, before the consonants are carved into it
    /// (physical engine), and which articulator moves each cell.
    body: [f32; N],
    mover: [Mover; N],
    /// The tongue body is being held by a consonant of its own (/k g ŋ l
    /// r/): it moves at the back of the tongue's speed, until it is back
    /// with the vowels.
    pub(crate) quick_body: bool,
    /// A narrowing along the tongue that belongs to the vowel shape itself
    /// (physical engine), apart from the consonant slots.
    narrowing: Option<Constriction>,
    slots: [Option<Slot>; MAX_CONSTRICTIONS],
    /// Hiss regulator: the widening each place of articulation has needed
    /// to make its fricatives hiss, how far the vowel shape is being
    /// opened to stop it hissing, and where.
    trim: [f32; N / PLACE],
    relief: f32,
    relief_at: usize,
    /// Seconds a vowel has been hissing.
    hissing: f32,
    transients: [Transient; MAX_TRANSIENTS],
    transient_decay: f32,
    transient_life: u32,
    damping: f32,
    last_obstruction: Option<usize>,
}

impl Tract {
    pub(crate) fn new() -> Self {
        let mut t = Self {
            right: [0.0; N],
            left: [0.0; N],
            junction_right: [0.0; N],
            junction_left: [0.0; N + 1],
            reflection: [0.0; N],
            new_reflection: [0.0; N],
            max_amplitude: [0.0; N],
            diameter: [0.0; N],
            target: [0.0; N],
            nose_right: [0.0; NOSE_LEN],
            nose_left: [0.0; NOSE_LEN],
            nose_junction_right: [0.0; NOSE_LEN],
            nose_junction_left: [0.0; NOSE_LEN + 1],
            nose_reflection: [0.0; NOSE_LEN],
            nose_diameter: [0.0; NOSE_LEN],
            nose_max_amplitude: [0.0; NOSE_LEN],
            reflection_left: 0.0,
            reflection_right: 0.0,
            reflection_nose: 0.0,
            new_reflection_left: 0.0,
            new_reflection_right: 0.0,
            new_reflection_nose: 0.0,
            tongue_index: 12.9,
            tongue_diameter: 2.43,
            velum_target: VELUM_CLOSED,
            movement_speed: MOVEMENT_SPEED,
            dynamics: Dynamics::Linear,
            jaw: false,
            jaw_open: 0.0,
            motion: Tuning::DEFAULT,
            velocity: [0.0; N],
            body: [0.0; N],
            mover: [Mover::Body; N],
            quick_body: false,
            narrowing: None,
            slots: [None; MAX_CONSTRICTIONS],
            trim: [0.0; N / PLACE],
            relief: 0.0,
            relief_at: 0,
            hissing: 0.0,
            transients: [Transient::default(); MAX_TRANSIENTS],
            // Original: strength · 2^(−200·t) over 0.2 s.
            transient_decay: powf(2.0, -200.0 / TRACT_RATE),
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            transient_life: (0.2 * TRACT_RATE) as u32,
            damping: DAMPING,
            last_obstruction: None,
        };

        // Nasal cavity shape. Its reflections are computed once, with the
        // velum open, as in the original.
        #[allow(clippy::cast_precision_loss)]
        for i in 0..NOSE_LEN {
            let d = 2.0 * i as f32 / NOSE_LEN as f32;
            let dia = if i == 0 {
                VELUM_OPEN
            } else if d < 1.0 {
                0.4 + 1.6 * d
            } else {
                0.5 + 1.5 * (2.0 - d)
            };
            t.nose_diameter[i] = dia.min(1.9);
        }
        t.calculate_nose_reflections();
        t.nose_diameter[0] = VELUM_CLOSED;

        t.compute_targets();
        t.diameter = t.target;
        t.calculate_reflections();
        t.calculate_reflections();
        t
    }

    /// Change the glottal rate the waveguide runs at (twice this). A higher
    /// rate makes sound cross the 44 cells faster: a shorter tract. Loss and
    /// burst timing are kept constant in seconds.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub(crate) fn set_rate(&mut self, internal_rate: f32) {
        let tract_rate = 2.0 * internal_rate;
        self.transient_decay = powf(2.0, -200.0 / tract_rate);
        self.transient_life = (0.2 * tract_rate) as u32;
        self.damping = powf(DAMPING, TRACT_RATE / tract_rate);
    }

    /// Silence the waveguides without touching the articulation.
    pub(crate) fn reset_waves(&mut self) {
        self.right = [0.0; N];
        self.left = [0.0; N];
        self.junction_right = [0.0; N];
        self.junction_left = [0.0; N + 1];
        self.nose_right = [0.0; NOSE_LEN];
        self.nose_left = [0.0; NOSE_LEN];
        self.nose_junction_right = [0.0; NOSE_LEN];
        self.nose_junction_left = [0.0; NOSE_LEN + 1];
        self.max_amplitude = [0.0; N];
        self.nose_max_amplitude = [0.0; NOSE_LEN];
        self.transients = [Transient::default(); MAX_TRANSIENTS];
    }

    pub(crate) fn set_tongue(&mut self, index: f32, diameter: f32) {
        self.tongue_index = index.clamp(*TONGUE_INDEX.start(), *TONGUE_INDEX.end());
        self.tongue_diameter = diameter.clamp(*TONGUE_DIAMETER.start(), *TONGUE_DIAMETER.end());
    }

    pub(crate) fn set_velum(&mut self, openness: f32) {
        self.velum_target = lerp(VELUM_CLOSED, VELUM_OPEN, openness.clamp(0.0, 1.0));
    }

    pub(crate) fn velum_openness(&self) -> f32 {
        ((self.nose_diameter[0] - VELUM_CLOSED) / (VELUM_OPEN - VELUM_CLOSED)).clamp(0.0, 1.0)
    }

    /// Place, move or release (`None`) the constriction in `slot`. A released
    /// constriction stops shaping the tract at once but its turbulence fades
    /// out over 100 ms.
    pub(crate) fn set_constriction(&mut self, slot: usize, c: Option<Constriction>) {
        let Some(s) = self.slots.get_mut(slot) else {
            return;
        };
        match (c, s.as_mut()) {
            (Some(c), Some(s)) => {
                s.c = sanitize(c);
                s.alive = true;
            }
            (Some(c), None) => {
                *s = Some(Slot {
                    c: sanitize(c),
                    alive: true,
                    fricative: 0.0,
                    failing: 0.0,
                });
            }
            (None, Some(s)) => s.alive = false,
            (None, None) => {}
        }
    }

    pub(crate) fn constriction(&self, slot: usize) -> Option<Constriction> {
        self.slots
            .get(slot)
            .copied()
            .flatten()
            .filter(|s| s.alive)
            .map(|s| s.c)
    }

    /// Set or clear the vowel shape's own narrowing.
    pub(crate) fn set_narrowing(&mut self, c: Option<Constriction>) {
        self.narrowing = c.map(sanitize);
    }

    /// Whether any constriction is being held.
    pub(crate) fn any_constriction(&self) -> bool {
        self.slots.iter().flatten().any(|s| s.alive)
    }

    pub(crate) fn amplitudes(&self) -> &[f32; N] {
        &self.max_amplitude
    }

    pub(crate) fn nose_amplitudes(&self) -> &[f32; NOSE_LEN] {
        &self.nose_max_amplitude
    }

    /// Rest shape of cell `i` for the current tongue (after the original's
    /// `TractUI.setRestDiameter`).
    #[allow(clippy::cast_precision_loss)]
    fn rest_diameter(&self, i: usize) -> f32 {
        if i < 7 {
            return 0.6;
        }
        if i < BLADE_START {
            return 1.1;
        }
        if i >= LIP_START {
            let ramp = ((i + 1 - LIP_START) as f32 / JAW_RAMP).min(1.0);
            return LIP_REST + self.jaw_open * ramp;
        }
        let t = 1.1 * PI * (self.tongue_index - i as f32) / (TIP_START - BLADE_START) as f32;
        let fixed = 2.0 + (self.tongue_diameter - 2.0) / 1.5;
        let mut curve = (1.5 - fixed + GRID_OFFSET) * cos(t);
        if i == BLADE_START - 2 || i == LIP_START - 1 {
            curve *= 0.8;
        }
        if i == BLADE_START || i == LIP_START - 2 {
            curve *= 0.94;
        }
        1.5 - curve
    }

    /// How far the jaw parts the lips beyond rest, diameter units: by
    /// [`JAW_OPENING`] times how open the tongue is (0 high, 1 low), less as
    /// the lips round, and not at all while a consonant narrows the tract:
    /// the jaw rises to make a closure or a fricative's gap, whatever vowel
    /// the tongue body is waiting in. Rounded lips purse to their own small opening
    /// whatever the jaw does; were they to keep the jaw's width just
    /// behind the rounding, a lowered tongue would widen the mouth of a
    /// /u/ and give it a second way to its formants, one that a gliding
    /// voice would have to leave by a jump.
    fn jaw_opening(&self) -> f32 {
        let open = (self.tongue_diameter - *TONGUE_DIAMETER.start())
            / (*TONGUE_DIAMETER.end() - *TONGUE_DIAMETER.start());
        let smooth = |v: f32, lo: f32, hi: f32| {
            let t = ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        let mut share = 1.0;
        for (k, s) in self.slots.iter().enumerate() {
            let Some(s) = s.filter(|s| s.alive) else {
                continue;
            };
            share *= if k == ROUNDING_SLOT {
                smooth(s.c.diameter, JAW_ROUNDED, JAW_SPREAD)
            } else {
                // A consonant raises the jaw to make its closure or gap.
                smooth(s.c.diameter, JAW_CONSONANT, JAW_SPREAD)
            };
        }
        JAW_OPENING * open.clamp(0.0, 1.0) * share
    }

    /// Target shape: tongue rest shape, then every live constriction carved
    /// in with a raised-cosine shoulder.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    fn compute_targets(&mut self) {
        for i in 0..N {
            self.target[i] = self.rest_diameter(i);
        }
        let floor = match self.dynamics {
            Dynamics::Linear => 0.0,
            Dynamics::Damped => -self.motion.closure_overshoot,
        };
        if self.dynamics == Dynamics::Damped {
            self.compute_shaped_targets(floor);
            return;
        }
        for (k, s) in self.slots.iter().enumerate() {
            let Some(s) = s.filter(|s| s.alive) else {
                continue;
            };
            let mut c = s.c;
            if intent(k, c.diameter) == Intent::Hiss {
                c.diameter += self.trim[place(c.index)];
            }
            carve(&mut self.target, c, floor);
        }
        if let Some(c) = self.narrowing {
            carve(&mut self.target, c, floor);
        }
        if self.relief > 0.0 {
            self.relieve();
        }
    }

    /// [`compute_targets`](Self::compute_targets) on the physical engine:
    /// the vowel shape first (tongue, jaw, lip rounding and the vowel's own
    /// narrowing, exactly as before), kept as [`Tract::body`]; then each
    /// consonant carved into it with the shoulder of the articulator that
    /// makes it, and every cell assigned the articulator that moves it.
    fn compute_shaped_targets(&mut self, floor: f32) {
        if let Some(Some(s)) = self.slots.get(ROUNDING_SLOT)
            && s.alive
        {
            carve(&mut self.target, s.c, floor);
        }
        if let Some(c) = self.narrowing {
            carve(&mut self.target, c, floor);
        }
        self.body = self.target;
        let mut by = [None; N];
        for (k, s) in self.slots.iter().enumerate() {
            let Some(s) = s.filter(|s| s.alive) else {
                continue;
            };
            if k == ROUNDING_SLOT {
                continue;
            }
            let mut c = s.c;
            if intent(k, c.diameter) == Intent::Hiss {
                c.diameter += self.trim[place(c.index)];
            }
            let before = self.target;
            if intent(k, c.diameter) == Intent::Open {
                // An approximant (/l r/) shapes the tongue as a whole, as
                // the original's did.
                carve(&mut self.target, c, floor);
            } else {
                let hiss = intent(k, c.diameter) == Intent::Hiss;
                carve_consonant(&mut self.target, c, floor, hiss);
            }
            for i in 0..N {
                if self.target[i] < before[i] - CARVED * 0.1 {
                    by[i] = Some(mover_of(c.index));
                }
            }
        }
        for i in 0..N {
            if self.target[i] < self.body[i] - CARVED
                && let Some(m) = by[i]
            {
                self.mover[i] = m;
            } else if self.quick_body {
                self.mover[i] = Mover::Dorsum;
            } else if self.mover[i] != Mover::Body
                && (self.diameter[i] - self.target[i]).abs() <= RETURNED
            {
                self.mover[i] = Mover::Body;
            }
        }
        if self.relief > 0.0 {
            self.relieve();
        }
    }

    /// Open the vowel shape around the place that hisses, leaving alone
    /// whatever a consonant is holding.
    #[allow(clippy::cast_precision_loss)]
    fn relieve(&mut self) {
        let held = self.held();
        let from = self.relief_at.saturating_sub(RELIEF_REACH);
        let to = (self.relief_at + RELIEF_REACH).min(N - 1);
        let cells = self.target.iter_mut().zip(&held).enumerate();
        for (i, (target, _)) in cells
            .take(to + 1)
            .skip(from)
            .filter(|(_, (_, held))| !**held)
        {
            let away = i.abs_diff(self.relief_at) as f32 / (RELIEF_REACH + 1) as f32;
            *target += self.relief * 0.5 * (1.0 + cos(PI * away));
        }
    }

    /// The cells a consonant (a closure or a fricative, live or just let
    /// go) is shaping.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn held(&self) -> [bool; N] {
        let mut held = [false; N];
        for (k, s) in self.slots.iter().enumerate() {
            let Some(s) = s else { continue };
            if intent(k, s.c.diameter) == Intent::Open {
                continue;
            }
            let width = reach(s.c.index);
            let from = (s.c.index - width).max(0.0) as usize;
            let to = ((s.c.index + width + 1.0) as usize).min(N - 1);
            for h in &mut held[from..=to] {
                *h = true;
            }
        }
        held
    }

    /// The hiss regulator: one step, from what the air did over the last
    /// block of `dt` seconds. `breath` says the lungs are pushing, `voiced`
    /// that the folds are meant to be vibrating, and `stream` reports the
    /// stream leaving a cell.
    ///
    /// * A gap that is meant to hiss, has arrived where it was sent, and is
    ///   shut, choked by viscosity, or too fine for its fast stream to turn
    ///   turbulent, is widened. The widening is remembered by place, so the
    ///   next fricative there starts from it.
    /// * A voiced shape that is not meant to hiss, but does, is opened a
    ///   little around the place that hisses, and let back once it is quiet.
    ///
    /// Neither moves anything by more than [`NUDGE`].
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    pub(crate) fn regulate(
        &mut self,
        dt: f32,
        breath: bool,
        voiced: bool,
        stream: impl Fn(usize) -> Stream,
    ) {
        if !breath {
            self.relief = (self.relief - RELAX_RATE * dt).max(0.0);
            self.hissing = 0.0;
            return;
        }
        for k in 0..MAX_CONSTRICTIONS {
            let Some(s) = self.slots[k].filter(|s| s.alive) else {
                continue;
            };
            if intent(k, s.c.diameter) != Intent::Hiss {
                continue;
            }
            // The narrowest cell of the gap.
            let centre = ((s.c.index + 0.5) as usize).clamp(1, N - 2);
            let mut c = centre;
            for i in [centre - 1, centre + 1] {
                if self.diameter[i] < self.diameter[c] {
                    c = i;
                }
            }
            let arrived = (self.diameter[c] - self.target[c]).abs() <= SETTLED
                && self.velocity[c].abs() <= SETTLED_SPEED;
            let air = stream(c);
            let too_fine = air.share >= FAST && air.reynolds < TURBULENT;
            let failing = arrived && (air.sealed || air.choke > CHOKE || too_fine);
            let mut waited = 0.0;
            if failing {
                waited = s.failing + dt;
                if waited >= PATIENCE {
                    let t = &mut self.trim[place(s.c.index)];
                    *t = (*t + WIDEN_RATE * dt).min(NUDGE);
                }
            }
            if let Some(slot) = &mut self.slots[k] {
                slot.failing = waited;
            }
        }

        // The loudest jet that nobody asked for.
        let held = self.held();
        let (mut worst, mut at) = (0.0f32, 0);
        for (i, _) in held.iter().enumerate().filter(|&(i, h)| i >= 2 && !h) {
            let noise = stream(i).noise;
            if noise > worst {
                (worst, at) = (noise, i);
            }
        }
        if voiced && worst > VOWEL_HISS {
            // A breathy moment on the way into a vowel is not a hissing
            // vowel: wait for it to last.
            self.hissing += dt;
            if self.hissing >= VOWEL_PATIENCE {
                self.relief_at = at;
                self.relief = (self.relief + RELIEF_RATE * dt).min(NUDGE);
            }
        } else {
            self.hissing = 0.0;
            if !voiced || worst < VOWEL_QUIET {
                self.relief = (self.relief - RELAX_RATE * dt).max(0.0);
            }
        }
    }

    /// Forget what the regulator has learnt (a different voice).
    pub(crate) fn forget(&mut self) {
        self.trim = [0.0; N / PLACE];
        self.relief = 0.0;
        self.hissing = 0.0;
    }

    /// How fast each cell's diameter is changing, diameter units per
    /// second (damped articulators; zero for the classic ones).
    pub(crate) fn velocity(&self) -> &[f32; N] {
        &self.velocity
    }

    /// The largest widening of a fricative gap in use, and how far the
    /// vowel shape is being opened.
    pub(crate) fn nudges(&self) -> (f32, f32) {
        let widest = self.trim.iter().fold(0.0f32, |m, &t| m.max(t));
        (widest, self.relief)
    }

    /// Per-block articulation update: move the tract towards its target,
    /// fire release bursts, update reflection coefficients.
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn update_block(&mut self, dt: f32) {
        // Fricative envelopes; free slots whose noise has faded out.
        let ramp = dt / FRICATIVE_ATTACK;
        for slot in &mut self.slots {
            if let Some(s) = slot {
                if s.alive {
                    s.fricative = (s.fricative + ramp).min(1.0);
                } else {
                    s.fricative -= ramp;
                    if s.fricative <= 0.0 {
                        *slot = None;
                    }
                }
            }
        }

        let jaw = if self.jaw { self.jaw_opening() } else { 0.0 };
        self.jaw_open += (jaw - self.jaw_open) * (dt / JAW_TIME).min(1.0);
        self.compute_targets();

        if self.dynamics == Dynamics::Damped {
            self.move_damped(dt);
            return;
        }

        let amount = dt * self.movement_speed;
        let mut obstruction = None;
        for i in 0..N {
            if self.diameter[i] <= 0.0 {
                obstruction = Some(i);
            }
            // The back of the tract moves slower than the tip and lips.
            let slow_return = if i < NOSE_START {
                0.6
            } else if i >= TIP_START {
                1.0
            } else {
                0.6 + 0.4 * (i - NOSE_START) as f32 / (TIP_START - NOSE_START) as f32
            };
            self.diameter[i] = move_towards(
                self.diameter[i],
                self.target[i],
                slow_return * amount,
                2.0 * amount,
            );
        }
        // Releasing a closure with the velum shut makes a plosive burst.
        if let (Some(pos), None) = (self.last_obstruction, obstruction)
            && self.nose_diameter[0] < 0.223
        {
            self.add_transient(pos);
        }
        self.last_obstruction = obstruction;

        self.nose_diameter[0] = move_towards(
            self.nose_diameter[0],
            self.velum_target,
            amount * 0.25,
            amount * 0.1,
        );

        self.calculate_reflections();
    }

    /// [`Dynamics::Damped`] articulation update. Each cell is a critically
    /// damped spring pulled towards its target; the step below is the exact
    /// solution over `dt`, so it is stable for any block length.
    #[allow(clippy::cast_precision_loss)]
    fn move_damped(&mut self, dt: f32) {
        // Natural frequencies of the three articulators at this speed.
        let speed = self.movement_speed / MOVEMENT_SPEED;
        let tip = speed * self.motion.articulation_rate;
        let dorsum = speed * self.motion.dorsum_rate;
        let body = speed * self.motion.body_rate;
        for i in 0..N {
            let e0 = self.diameter[i] - self.target[i];
            // Consonants close twice as fast as they open, as in the
            // original; the vowel shape moves alike either way.
            let closing = if e0 > 0.0 {
                self.motion.closing_ratio
            } else {
                1.0
            };
            let w = match self.mover[i] {
                Mover::Body => body,
                Mover::Dorsum => dorsum * closing,
                Mover::Tip => tip * closing,
            };
            let mut v0 = self.velocity[i];
            if self.diameter[i] <= 0.0 && e0 < 0.0 && v0 <= 0.0 {
                // Letting go of a closure: surfaces pressed together spring
                // apart (and the air behind them pushes), so the release
                // starts at speed instead of from rest.
                v0 = self.motion.release_speed * w * -e0;
            }
            let decay = exp(-w * dt);
            let c = v0 + w * e0;
            let mut d = self.target[i] + (e0 + c * dt) * decay;
            let mut v = (v0 - w * c * dt) * decay;
            if d <= 0.0 {
                // Contact: the surfaces meet and stop.
                d = 0.0;
                v = 0.0;
            }
            self.diameter[i] = d;
            self.velocity[i] = v;
            // Targets past contact are only meaningful to the springs.
            self.target[i] = self.target[i].max(0.0);
        }
        self.last_obstruction = None;

        let amount = dt * self.movement_speed;
        self.nose_diameter[0] = move_towards(
            self.nose_diameter[0],
            self.velum_target,
            amount * 0.25,
            amount * 0.1,
        );
        self.calculate_reflections();
    }

    fn add_transient(&mut self, position: usize) {
        let life = self.transient_life;
        let slot = self
            .transients
            .iter_mut()
            .min_by_key(|t| t.steps_left)
            .expect("MAX_TRANSIENTS > 0");
        *slot = Transient {
            position,
            amplitude: 0.3,
            steps_left: life,
        };
    }

    fn calculate_nose_reflections(&mut self) {
        let mut a = [0.0; NOSE_LEN];
        for (a, d) in a.iter_mut().zip(&self.nose_diameter) {
            *a = (d * d).max(1e-6);
        }
        for i in 1..NOSE_LEN {
            self.nose_reflection[i] = (a[i - 1] - a[i]) / (a[i - 1] + a[i]);
        }
    }

    fn calculate_reflections(&mut self) {
        let mut a = [0.0; N];
        for (a, d) in a.iter_mut().zip(&self.diameter) {
            *a = d * d;
        }
        for i in 1..N {
            self.reflection[i] = self.new_reflection[i];
            let sum = a[i - 1] + a[i];
            self.new_reflection[i] = if sum.abs() > 1e-6 {
                (a[i - 1] - a[i]) / sum
            } else {
                1.0
            };
        }

        self.reflection_left = self.new_reflection_left;
        self.reflection_right = self.new_reflection_right;
        self.reflection_nose = self.new_reflection_nose;
        let velum = self.nose_diameter[0] * self.nose_diameter[0];
        let a0 = a[NOSE_START];
        let a1 = a[NOSE_START + 1];
        let sum = a0 + a1 + velum;
        if sum.abs() > 1e-6 {
            self.new_reflection_left = (2.0 * a0 - sum) / sum;
            self.new_reflection_right = (2.0 * a1 - sum) / sum;
            self.new_reflection_nose = (2.0 * velum - sum) / sum;
        } else {
            self.new_reflection_left = 1.0;
            self.new_reflection_right = 1.0;
            self.new_reflection_nose = 1.0;
        }
    }

    /// One waveguide step. `frication` is band-passed noise already scaled by
    /// the glottal noise modulator. Returns lip plus nostril output.
    pub(crate) fn step(&mut self, glottal: f32, frication: f32, lambda: f32) -> f32 {
        // Plosive bursts.
        for t in &mut self.transients {
            if t.steps_left > 0 {
                self.right[t.position] += t.amplitude * 0.5;
                self.left[t.position] += t.amplitude * 0.5;
                t.amplitude *= self.transient_decay;
                t.steps_left -= 1;
            }
        }

        // Turbulence at each constriction.
        for k in 0..MAX_CONSTRICTIONS {
            if let Some(s) = self.slots[k]
                && s.fricative > 0.0
            {
                self.add_turbulence(0.66 * frication * s.fricative, s.c.index, s.c.diameter);
            }
        }

        self.junction_right[0] = self.left[0] * GLOTTAL_REFLECTION + glottal;
        self.junction_left[N] = self.right[N - 1] * LIP_REFLECTION;

        for i in 1..N {
            let r = lerp(self.reflection[i], self.new_reflection[i], lambda);
            let w = r * (self.right[i - 1] + self.left[i]);
            self.junction_right[i] = self.right[i - 1] - w;
            self.junction_left[i] = self.left[i] + w;
        }

        // Three-way junction at the velum.
        let i = NOSE_START;
        let r = lerp(self.reflection_left, self.new_reflection_left, lambda);
        self.junction_left[i] =
            r * self.right[i - 1] + (1.0 + r) * (self.nose_left[0] + self.left[i]);
        let r = lerp(self.reflection_right, self.new_reflection_right, lambda);
        self.junction_right[i] =
            r * self.left[i] + (1.0 + r) * (self.right[i - 1] + self.nose_left[0]);
        let r = lerp(self.reflection_nose, self.new_reflection_nose, lambda);
        self.nose_junction_right[0] =
            r * self.nose_left[0] + (1.0 + r) * (self.left[i] + self.right[i - 1]);

        for i in 0..N {
            self.right[i] = self.junction_right[i] * self.damping;
            self.left[i] = self.junction_left[i + 1] * self.damping;
            let amp = (self.right[i] + self.left[i]).abs();
            self.max_amplitude[i] = (self.max_amplitude[i] * 0.9999).max(amp);
        }
        let lips = self.right[N - 1];

        self.nose_junction_left[NOSE_LEN] = self.nose_right[NOSE_LEN - 1] * LIP_REFLECTION;
        for i in 1..NOSE_LEN {
            let w = self.nose_reflection[i] * (self.nose_right[i - 1] + self.nose_left[i]);
            self.nose_junction_right[i] = self.nose_right[i - 1] - w;
            self.nose_junction_left[i] = self.nose_left[i] + w;
        }
        for i in 0..NOSE_LEN {
            self.nose_right[i] = self.nose_junction_right[i];
            self.nose_left[i] = self.nose_junction_left[i + 1];
            let amp = (self.nose_right[i] + self.nose_left[i]).abs();
            self.nose_max_amplitude[i] = (self.nose_max_amplitude[i] * 0.9999).max(amp);
        }
        let nose = self.nose_right[NOSE_LEN - 1];

        lips + nose
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn add_turbulence(&mut self, noise: f32, index: f32, diameter: f32) {
        if !(2.0..=N as f32).contains(&index) || diameter <= 0.0 {
            return;
        }
        let i = index as usize;
        let delta = index - i as f32;
        let thinness = (8.0 * (0.7 - diameter)).clamp(0.0, 1.0);
        let openness = (30.0 * (diameter - 0.3)).clamp(0.0, 1.0);
        let n0 = noise * (1.0 - delta) * thinness * openness;
        let n1 = noise * delta * thinness * openness;
        if i + 1 < N {
            self.right[i + 1] += n0 * 0.5;
            self.left[i + 1] += n0 * 0.5;
        }
        if i + 2 < N {
            self.right[i + 2] += n1 * 0.5;
            self.left[i + 2] += n1 * 0.5;
        }
    }
}

/// Carve a constriction into a target shape with a raised-cosine shoulder.
/// `floor` is the lowest target diameter (negative: aim past contact).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn carve(target: &mut [f32; N], c: Constriction, floor: f32) {
    let index = c.index;
    let d = (c.diameter - 0.3).max(floor);
    if !(2.0..N as f32).contains(&index) || d >= 3.0 {
        return;
    }
    let width = reach(index);
    let centre = (index + 0.5) as i32;
    let reach = width as i32 + 2;
    for k in -reach..=reach {
        let j = centre + k;
        if !(0..N as i32).contains(&j) {
            continue;
        }
        let j = j as usize;
        let rel = (j as f32 - index).abs() - 0.5;
        let shrink = if rel <= 0.0 {
            0.0
        } else if rel > width {
            1.0
        } else {
            0.5 * (1.0 - cos(PI * rel / width))
        };
        if d < target[j] {
            target[j] = d + (target[j] - d) * shrink;
        }
    }
}

/// Carve a consonant into a target shape (physical engine), with the
/// shoulder of the articulator that makes it: lips close only the lips; the
/// tongue tip raises the blade behind it but leaves the mouth in front of
/// it open; the back of the tongue humps over its usual reach either way.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn carve_consonant(target: &mut [f32; N], c: Constriction, floor: f32, hiss: bool) {
    let index = c.index;
    let d = (c.diameter - 0.3).max(floor);
    if !(2.0..N as f32).contains(&index) || d >= 3.0 {
        return;
    }
    let (back, front) = if index >= LIP_PLACE {
        (LIP_SHOULDER, LIP_SHOULDER)
    } else if index >= BLADE_PLACE && hiss && index < DENTAL_PLACE {
        (TIP_SHOULDER_BACK, SIBILANT_SHOULDER_FRONT)
    } else if index >= BLADE_PLACE {
        (TIP_SHOULDER_BACK, TIP_SHOULDER_FRONT)
    } else if index >= TIP_START as f32 {
        (TIP_SHOULDER_BACK, BLADE_SHOULDER_FRONT)
    } else {
        (reach(index), reach(index))
    };
    let centre = (index + 0.5) as i32;
    let span = back.max(front) as i32 + 2;
    for k in -span..=span {
        let j = centre + k;
        if !(0..N as i32).contains(&j) {
            continue;
        }
        let j = j as usize;
        let signed = j as f32 - index;
        let width = if signed < 0.0 { back } else { front };
        let rel = signed.abs() - 0.5;
        let shrink = if rel <= 0.0 {
            0.0
        } else if rel > width {
            1.0
        } else {
            0.5 * (1.0 - cos(PI * rel / width))
        };
        if d < target[j] {
            target[j] = d + (target[j] - d) * shrink;
        }
    }
}

/// The place of articulation (an entry of `Tract::trim`) a cell index
/// belongs to.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn place(index: f32) -> usize {
    ((index.max(0.0) as usize) / PLACE).min(N / PLACE - 1)
}

fn sanitize(c: Constriction) -> Constriction {
    #[allow(clippy::cast_precision_loss)]
    let index = if c.index.is_finite() {
        c.index.clamp(0.0, N as f32)
    } else {
        0.0
    };
    let diameter = if c.diameter.is_finite() {
        c.diameter.clamp(-1.0, 5.0)
    } else {
        5.0
    };
    Constriction { index, diameter }
}
