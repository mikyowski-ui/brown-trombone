//! Vocal tract: a 44-cell Kelly–Lochbaum waveguide with a 28-cell nasal
//! branch, plus the articulation model that shapes it (tongue, constrictions,
//! velum).
//!
//! The waveguide runs at twice [`INTERNAL_RATE`](crate::INTERNAL_RATE).
//! Everything here is fixed-size: no allocation, ever.

use crate::INTERNAL_RATE;
use crate::math::{cos, exp, lerp, move_towards, powf, spring_step, sqrt};
use crate::phys::Stream;
use crate::tract_modes as modes;
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
/// Valid tongue root settings (`ln` of the widening of the middle of the
/// pharynx; see [`Shape::pharynx`](crate::inverse::Shape::pharynx)).
pub const PHARYNX: core::ops::RangeInclusive<f32> = -1.2..=0.8;

const GLOTTAL_REFLECTION: f32 = 0.75;
const LIP_REFLECTION: f32 = -0.85;
const DAMPING: f32 = 0.999;
/// The original's articulator speed, diameter units per second.
pub(crate) const MOVEMENT_SPEED: f32 = 15.0;
const GRID_OFFSET: f32 = 1.7;
const VELUM_CLOSED: f32 = 0.01;
const VELUM_OPEN: f32 = 0.4;
const FRICATIVE_ATTACK: f32 = 0.1;
/// How fast the velum shuts on the physical engine, in the units of the
/// original's `0.1` (diameter per unit of articulator movement).
const VELUM_SHUTTING: f32 = 0.25;
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

// ----- the tongue root (physical engine) ---------------------------------
//
// The root of the tongue faces the pharynx. Drawn forward it widens the
// pharynx (an advanced tongue root, as in the tense vowels /i e u o/),
// pulled back it narrows it (/a/). The single cosine of the tongue body
// ties the pharynx to where the tongue is high; the root moves it on its
// own, with the body and as slowly, and consonants made on a vowel's body
// keep that vowel's root (see `Tract::pharynx`).

/// First and last cell the tongue root reshapes: from just above the
/// larynx tube to behind the tongue body's back...
const PHARYNX_FROM: f32 = 6.0;
const PHARYNX_TO: f32 = 22.0;

/// How much of the root's widening cell `i` takes: a raised cosine over
/// [`PHARYNX_FROM`]..[`PHARYNX_TO`], fullest half-way, in the lower and
/// middle pharynx.
#[allow(clippy::cast_precision_loss)]
fn pharynx_weight(i: usize) -> f32 {
    let x = (i as f32 - PHARYNX_FROM) / (PHARYNX_TO - PHARYNX_FROM);
    if (0.0..=1.0).contains(&x) {
        0.5 * (1.0 - cos(2.0 * PI * x))
    } else {
        0.0
    }
}

// ----- the tongue body from measured modes (physical engine) -------------
//
// With `AnatomyConfig::tongue_model` 1 the body from the top of the
// epilarynx to the lips is not the cosine hump but a measured neutral shape
// plus three principal modes of eleven vowels of one man (Story 2008, MRI;
// `src/tract_modes.rs`), weighted by the body's three numbers
// (`Tract::modes`). A cosine tongue set by its place and height (the app's
// handle, the consonants with a body of their own) is projected onto the
// modes. The lips, the consonants carved on the body and the nose are as
// on the cosine body.

/// Narrowest diameter the modal body gives a cell: some 0.1 cm², the
/// narrowest the measured vowels have (/ɔ/'s pharynx).
const MODAL_FLOOR: f32 = 0.3;
/// How far each body number may go beyond the measured vowels' range, in
/// spans of that range, either way.
const MODAL_REACH: f32 = 1.0;

/// The body numbers a tract allows: the measured range widened by
/// [`MODAL_REACH`] of its span either way.
pub(crate) fn mode_range(k: usize) -> (f32, f32) {
    let (lo, hi) = modes::RANGE[k];
    let span = hi - lo;
    (lo - MODAL_REACH * span, hi + MODAL_REACH * span)
}

/// `q` with every number inside [`mode_range`]; numbers not finite are `0`
/// (the neutral shape).
pub(crate) fn clamp_modes(q: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0; 3];
    for (k, (o, v)) in out.iter_mut().zip(q).enumerate() {
        let (lo, hi) = mode_range(k);
        *o = if v.is_finite() { v.clamp(lo, hi) } else { 0.0 };
    }
    out
}

/// Diameter of cell `i` (from [`modes::FROM`] to before [`modes::TO`]) of
/// the modal body with numbers `q`.
fn modal_cell(i: usize, q: &[f32; 3]) -> f32 {
    let j = i - modes::FROM;
    let mut d = modes::NEUTRAL[j];
    for (m, qk) in modes::MODES.iter().zip(q) {
        d += qk * m[j];
    }
    d.max(MODAL_FLOOR)
}

/// The cosine tongue `(index, diameter)` of the cells before the lips (as
/// `Tract::rest_diameter` makes it), projected onto the modes: the body
/// numbers whose shape is nearest to it in the least-squares sense over the
/// cosine's own cells.
#[must_use]
pub(crate) fn project_tongue(index: f32, diameter: f32) -> [f32; 3] {
    clamp_modes(projection(index, diameter))
}

/// [`project_tongue`] without the clamping.
fn projection(index: f32, diameter: f32) -> [f32; 3] {
    let mut q = [0.0f32; 3];
    for i in modes::COSINE_FROM..modes::TO {
        let d = cosine_tongue(i, index, diameter) - modes::NEUTRAL[i - modes::FROM];
        for (qk, p) in q.iter_mut().zip(&modes::PROJECTION) {
            *qk += p[i - modes::COSINE_FROM] * d;
        }
    }
    q
}

/// How the modal body's numbers move when a cosine tongue at `from` is
/// drawn `w` of the way towards `to` (as a consonant's tip draws the
/// vowel's body forward and up): the projection of that change of shape,
/// added to whatever numbers the body has. Drawn towards the projection of
/// `to` itself instead, the body moved much further than the cosine one
/// does (the /ɔ/ of a body drawn 0.3 of the way to the tip's: F2 1030 to
/// 1470 Hz on the modal body, 1030 to 1120 on the cosine one, heard as
/// /ɑ/), since the tip's own tongue projects onto an /i/-like body.
#[must_use]
pub(crate) fn drawn_modes(q: [f32; 3], from: (f32, f32), to: (f32, f32), w: f32) -> [f32; 3] {
    let a = projection(from.0, from.1);
    let b = projection(from.0 + (to.0 - from.0) * w, from.1 + (to.1 - from.1) * w);
    [0, 1, 2].map(|k| q[k] + b[k] - a[k])
}

/// The cosine tongue's diameter of cell `i` (`BLADE_START..LIP_START`).
#[allow(clippy::cast_precision_loss)]
fn cosine_tongue(i: usize, index: f32, diameter: f32) -> f32 {
    let t = 1.1 * PI * (index - i as f32) / (TIP_START - BLADE_START) as f32;
    let fixed = 2.0 + (diameter - 2.0) / 1.5;
    let mut curve = (1.5 - fixed + GRID_OFFSET) * cos(t);
    if i == BLADE_START - 2 || i == LIP_START - 1 {
        curve *= 0.8;
    }
    if i == BLADE_START || i == LIP_START - 2 {
        curve *= 0.94;
    }
    1.5 - curve
}

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
const BLADE_SHOULDER_FRONT: f32 = 3.5;
/// Cells behind an /l/'s channel over which its shoulder runs: behind the
/// tip of a lateral the blade is not raised but drawn down, the sides of the
/// tongue with it. With the tip's own shoulder (five cells) the raised blade
/// was a narrowing at the palate as the tip let go into a front vowel: the
/// second formant of the /ɪ/ of "slid" overshot to 2100 Hz before it settled
/// at 1900, an /i/ (the owner's rises from his /l/'s 1150 to 1900 and stays).
const LATERAL_SHOULDER_BACK: f32 = 2.0;
/// Constriction index from which the tip, not the blade, makes it.
const BLADE_PLACE: f32 = 34.5;
/// The front shoulder of /s z/: their jet crosses a short, narrow channel
/// to the teeth (the jaw is raised), whose small cavity gives them their
/// high hiss.
const SIBILANT_SHOULDER_FRONT: f32 = 5.0;
/// Constriction index from which the tongue tip is at the teeth (/θ ð/):
/// nothing in front of it but the lips.
const DENTAL_PLACE: f32 = 38.5;
/// The blade of /θ ð/: raised to this diameter just behind the tip, and
/// less over [`DENTAL_BLADE_BACK`] cells behind it. A dental's broad, flat
/// blade lies along the ridge and the front of the palate, so the mouth
/// in front of the tongue body is shallow, but the narrow gap is the
/// tip's alone (a long narrow channel chokes the air). With the
/// alveolars' shoulder alone the gap at the teeth was a pinch at the end
/// of an open mouth, and the formants into and out of it a labial's (F2
/// near 900 Hz before /ə/, where a speaker's /ð/ has some 1400 to 1600):
/// with the low F1 of the narrow gap and next to no hiss, an /l/ or /w/.
const DENTAL_BLADE: f32 = 0.5;
const DENTAL_BLADE_BACK: f32 = 9.0;
/// Shoulder of a closure of the lips, cells either way: the lips are the
/// last four cells of the tract, and closing them leaves the tongue alone.
const LIP_SHOULDER: f32 = 2.5;
/// Where the lips begin, as a constriction index (labials sit at 40.5 to
/// 41).
const LIP_PLACE: f32 = 40.0;
/// A cell is being shaped by a consonant when the consonant takes this
/// much off what the vowel shape alone would give it, diameter units.
const CARVED: f32 = 0.05;
/// Widest constriction that is a consonant's rather than a shading of the
/// vowel (touch diameter): approximants included.
const CONSONANT_WIDEST: f32 = 1.3;
/// A cell a consonant has let go is back with the tongue body once it is
/// this close to where the body wants it, diameter units.
const RETURNED: f32 = 0.08;
/// ...and how slowly it must be moving there, diameter units per second.
const RETURNED_SPEED: f32 = 1.0;
/// A tongue-tip closure let go with the velum at least this open is a
/// nasal's (see [`ArticulationConfig::nasal_tip_release`](crate::ArticulationConfig::nasal_tip_release)): out of /n/ into a vowel it
/// is still some two-thirds open at the release; a /d/ after /n/ is
/// released with it shut (`Tuning::velum_lead`).
const NASAL_RELEASE_VELUM: f32 = 0.2;
/// Tenseness of the folds at which `AnatomyConfig::epilarynx_effort`
/// leaves the epilaryngeal tube as wide as `epilarynx_diameter`: the modal
/// voice's (the voices' default).
const EPILARYNX_TENSENESS: f32 = 0.65;

/// Which articulator moves a cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mover {
    /// The tongue body and the jaw: the vowel shape.
    Body,
    /// The back of the tongue making a consonant (/k g ŋ/).
    Dorsum,
    /// The tongue tip and blade, or the lips, making a consonant.
    Tip,
    /// The tongue dropping behind a nasal's closure (see
    /// [`Tract::open_nasal_cavity`]).
    Cavity,
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
#[allow(clippy::struct_excessive_bools)] // independent articulator flags
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
    /// The tongue root (physical engine): `ln` of how much wider it makes
    /// the middle of the pharynx, positive drawn forward, negative pulled
    /// back, `0` at rest (see [`pharynx_weight`]). Part of the vowel shape,
    /// moved with the tongue body. The classic engine leaves it at rest.
    pub(crate) pharynx: f32,
    /// The modal body's three numbers (physical engine with
    /// `AnatomyConfig::tongue_model` 1), or `None`: the cosine tongue
    /// `(tongue_index, tongue_diameter)` projected onto the modes. Setting
    /// the tongue lets go of them.
    pub(crate) modes: Option<[f32; 3]>,
    /// Tenseness the folds are set to (`Trombone::set_tenseness`): the
    /// epilaryngeal tube narrows with it (physical engine, see
    /// [`Tract::epilarynx`]).
    pub(crate) tenseness: f32,
    /// How far the epilaryngeal tube is drawn in, `0‥1`
    /// (`Trombone::set_epilarynx_squeeze`; wave 3, w3-vocal): its area is
    /// `1 − squeeze` of itself (physical engine, see [`Tract::epilarynx`]).
    pub(crate) squeeze: f32,
    pub(crate) velum_target: f32,
    /// How many times faster than [`VELUM_SHUTTING`] the velum shuts now
    /// (physical engine; `1` but out of a nasal into a vowel, see
    /// [`ArticulationConfig::velum_release`](crate::ArticulationConfig::velum_release)).
    pub(crate) velum_haste: f32,
    pub(crate) movement_speed: f32,
    /// How much faster than `movement_speed` the tongue body and its back
    /// move (physical engine; see
    /// [`Trombone::set_articulation_boost`](crate::Trombone::set_articulation_boost)).
    pub(crate) boost: f32,
    /// How much faster than the tuning's speeds the tongue body (and its
    /// back), the tip and the lips move, as the stiffness of the gesture
    /// score's [`TongueBody`](crate::Tier::TongueBody),
    /// [`TongueTip`](crate::Tier::TongueTip) and
    /// [`LipsAndJaw`](crate::Tier::LipsAndJaw) tiers asks (physical
    /// engine; `1` when no score drives the tract).
    pub(crate) tier_speed: [f32; 3],
    pub(crate) dynamics: Dynamics,
    /// Whether the lips part with the tongue's opening, as a jaw lowered
    /// with the tongue body parts them (physical engine; see
    /// [`JAW_OPENING`]). The classic engine keeps the original's fixed lips.
    pub(crate) jaw: bool,
    /// How far the jaw parts the lips now (see [`Tract::jaw_opening`]): it
    /// follows where it is going with [`JAW_TIME`].
    jaw_open: f32,
    /// Length of a control block as a share of the long one (512 internal
    /// samples); see [`crate::math::approach`].
    pub(crate) block_share: f32,
    /// How the damped articulators move (the speeds of the voice's tuning).
    pub(crate) motion: Tuning,
    velocity: [f32; N],
    /// The vowel shape alone, before the consonants are carved into it
    /// (physical engine), and which articulator moves each cell.
    body: [f32; N],
    mover: [Mover; N],
    /// Lip cells let go of a closure and still on their way out of it.
    parting: [bool; N],
    /// Tongue-tip cells that let go of a closure while the velum was open
    /// (out of an /n/), still on their way out of it: they move
    /// [`ArticulationConfig::nasal_tip_release`](crate::ArticulationConfig::nasal_tip_release) times as fast.
    nasal_release: [bool; N],
    /// Share of the tongue's drop behind a nasal closure held back (see
    /// [`Tract::open_nasal_cavity`]; the speaker sets it for an /n/ that
    /// leads into a fronted vowel, [`ArticulationConfig::nasal_cavity_held`](crate::ArticulationConfig::nasal_cavity_held)).
    pub(crate) cavity_held: f32,
    /// The tongue body is being held by a consonant of its own (/k g ŋ l
    /// r/): it moves at the back of the tongue's speed, until it is back
    /// with the vowels.
    pub(crate) quick_body: bool,
    /// The sound being made is a nasal (the velum is lowered for it): the
    /// mouth behind its closure is shaped as the nose's side branch (see
    /// [`Tract::open_nasal_cavity`]), until another sound is articulated,
    /// even while the velum already rises for what follows.
    pub(crate) nasal: bool,
    /// The lip rounding being held is the consonant's own gesture (/ʃ ʒ tʃ
    /// dʒ r w/), not the rounding of the vowel the consonant is made on
    /// (physical engine; see [`Tract::compute_shaped_targets`]).
    pub(crate) own_lips: bool,
    /// The main constriction is a lateral (/l/; physical engine): the tip
    /// touches the ridge in the middle and the air goes round its sides.
    /// However narrow, it is not a gap meant to hiss: the jaw does not
    /// close the teeth for it, the hiss regulator leaves it alone, and it
    /// is carved as a channel (see [`carve_lateral`]).
    pub(crate) lateral: bool,
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
    /// The tract's length (wave 2A; physical engine, read by the sound only
    /// with `AnatomyConfig::lip_horn` or `larynx_travel` on). How far the
    /// lips are pushed forward, cm at native size, and how fast that is
    /// changing, cm/s: moved by the lips' spring towards what they are told
    /// (see [`Tract::protrusion_goal`]).
    protrusion: f32,
    protrusion_speed: f32,
    /// Protrusion set by the [`LipsAndJaw`](crate::Tier::LipsAndJaw)
    /// tier's protrusion variable, and by
    /// [`Trombone::set_lip_protrusion`](crate::Trombone::set_lip_protrusion),
    /// cm; `None` follows the rounding slot.
    pub(crate) protrusion_tier: Option<f32>,
    pub(crate) protrusion_set: Option<f32>,
    /// How far a lowered larynx lengthens the tract (negative: a raised
    /// one shortens it), cm at native size, and how fast, cm/s; where it is
    /// going (the [`LarynxAndPitch`](crate::Tier::LarynxAndPitch) tier's
    /// height, or [`Trombone::set_larynx_height`](crate::Trombone::set_larynx_height),
    /// as lengthening); and how much faster than tuned it moves, as the
    /// tier's stiffness asks.
    larynx: f32,
    larynx_speed: f32,
    pub(crate) larynx_target: f32,
    pub(crate) larynx_tier: Option<f32>,
    pub(crate) larynx_speed_ratio: f32,
}

/// Furthest the lips can be pushed forward, cm at native size.
pub(crate) const PROTRUSION_MOST: f32 = 2.0;
/// How far a lowered larynx can lengthen the tract and a raised one shorten
/// it, cm at native size.
pub(crate) const LARYNX_RANGE: core::ops::RangeInclusive<f32> = -1.0..=0.8;

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
            pharynx: 0.0,
            modes: None,
            tenseness: 0.6,
            squeeze: 0.0,
            velum_target: VELUM_CLOSED,
            velum_haste: 1.0,
            movement_speed: MOVEMENT_SPEED,
            boost: 1.0,
            tier_speed: [1.0; 3],
            dynamics: Dynamics::Linear,
            jaw: false,
            jaw_open: 0.0,
            block_share: 1.0,
            motion: Tuning::DEFAULT,
            velocity: [0.0; N],
            body: [0.0; N],
            mover: [Mover::Body; N],
            parting: [false; N],
            nasal_release: [false; N],
            cavity_held: 0.0,
            quick_body: false,
            nasal: false,
            own_lips: false,
            lateral: false,
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
            protrusion: 0.0,
            protrusion_speed: 0.0,
            protrusion_tier: None,
            protrusion_set: None,
            larynx: 0.0,
            larynx_speed: 0.0,
            larynx_target: 0.0,
            larynx_tier: None,
            larynx_speed_ratio: 1.0,
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
        self.modes = None;
    }

    /// Set the modal body's three numbers (see [`Tract::modes`]), inside
    /// [`mode_range`]; `None` follows the cosine tongue again.
    pub(crate) fn set_modes(&mut self, q: Option<[f32; 3]>) {
        self.modes = q.map(clamp_modes);
    }

    /// Whether the tongue body is the modal one (physical engine with
    /// `AnatomyConfig::tongue_model` 1).
    pub(crate) fn modal(&self) -> bool {
        self.jaw && self.motion.anatomy.tongue_model >= 0.5
    }

    /// First cell the modal body shapes: above the tube over the folds
    /// (`AnatomyConfig::epilarynx_cells`; a cell the tube's top falls in is
    /// the tube's), never below the first cell the modes are measured over.
    /// The measured body has a lower pharynx of its own, flare included
    /// (the speaker's tube flares over sections 5 to 8), which takes the
    /// place of the epilarynx's flare and its ramp
    /// (`epilarynx_flare_cells`): a ramp does not change the modal body;
    /// the larynx slide (cells 8 and 9, `larynx_travel`) takes their mean
    /// area.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn modal_from(&self) -> usize {
        let a = &self.motion.anatomy;
        let end = a.epilarynx_cells.max(0.0);
        // Rounded up (no `ceil` without `std`).
        let whole = end as usize;
        #[allow(clippy::cast_precision_loss)]
        let top = if (whole as f32) < end {
            whole + 1
        } else {
            whole
        };
        top.clamp(modes::FROM, BLADE_START)
    }

    /// The modal body's three numbers now: those set, or the cosine
    /// tongue's projection.
    pub(crate) fn body_modes(&self) -> [f32; 3] {
        self.modes
            .unwrap_or_else(|| project_tongue(self.tongue_index, self.tongue_diameter))
    }

    /// Set the tongue root (see [`Tract::pharynx`]), within
    /// [`PHARYNX`]; anything not finite leaves it at rest.
    pub(crate) fn set_pharynx(&mut self, pharynx: f32) {
        self.pharynx = if pharynx.is_finite() {
            pharynx.clamp(*PHARYNX.start(), *PHARYNX.end())
        } else {
            0.0
        };
    }

    pub(crate) fn set_velum(&mut self, openness: f32) {
        self.velum_target = lerp(VELUM_CLOSED, VELUM_OPEN, openness.clamp(0.0, 1.0));
    }

    pub(crate) fn velum_openness(&self) -> f32 {
        ((self.nose_diameter[0] - VELUM_CLOSED) / (VELUM_OPEN - VELUM_CLOSED)).clamp(0.0, 1.0)
    }

    pub(crate) fn velum_target_openness(&self) -> f32 {
        ((self.velum_target - VELUM_CLOSED) / (VELUM_OPEN - VELUM_CLOSED)).clamp(0.0, 1.0)
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

    /// What the constriction in slot `k` of touch diameter `diameter` is
    /// for: [`intent`], except that a lateral is never a gap meant to hiss
    /// (see [`Tract::lateral`]).
    fn intent_in(&self, k: usize, diameter: f32) -> Intent {
        match intent(k, diameter) {
            Intent::Hiss if self.lateral && k == crate::voice::SLOT_MAIN => Intent::Open,
            i => i,
        }
    }

    /// How far the jaw brings the incisors together (physical engine): all
    /// the way while a gap meant to hiss is held between the tongue tip and
    /// the alveolar ridge (/s z/), not at all otherwise. The physical tract
    /// narrows the gap between the teeth by it (`Phys::set_teeth`).
    pub(crate) fn teeth(&self) -> f32 {
        if self.dynamics != Dynamics::Damped {
            return 0.0;
        }
        let sibilant = self.slots.iter().enumerate().any(|(k, s)| {
            s.is_some_and(|s| {
                s.alive
                    && k != ROUNDING_SLOT
                    && self.intent_in(k, s.c.diameter) == Intent::Hiss
                    && (BLADE_PLACE..DENTAL_PLACE).contains(&s.c.index)
            })
        });
        if sibilant { 1.0 } else { 0.0 }
    }

    /// Whether a gap meant to hiss is held at the edges of the upper
    /// incisors (physical engine), as `(dental, labiodental)`, each `0` or
    /// `1`: the tongue tip against them (/θ ð/) or the lower lip (/f v/).
    /// The jet out of such a gap strikes the edges of the teeth and the
    /// upper lip at once (`Phys::set_edge`).
    pub(crate) fn edges(&self) -> (f32, f32) {
        if self.dynamics != Dynamics::Damped {
            return (0.0, 0.0);
        }
        let at = |range: core::ops::Range<f32>| {
            self.slots.iter().enumerate().any(|(k, s)| {
                s.is_some_and(|s| {
                    s.alive
                        && k != ROUNDING_SLOT
                        && self.intent_in(k, s.c.diameter) == Intent::Hiss
                        && range.contains(&s.c.index)
                })
            })
        };
        let dental = at(DENTAL_PLACE..LIP_PLACE);
        #[allow(clippy::cast_precision_loss)]
        let labial = at(LIP_PLACE..N as f32 + 1.0);
        (f32::from(u8::from(dental)), f32::from(u8::from(labial)))
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

    /// The epilaryngeal tube and its flare into the lower pharynx (physical
    /// engine): the rest diameter of cell `i` if it is one of theirs (cells
    /// below [`BLADE_START`], where the tongue's shape begins), from the
    /// voice's [`AnatomyConfig`](crate::AnatomyConfig): `epilarynx_cells`
    /// of tube at `epilarynx_diameter`, narrowed or widened with the folds'
    /// tenseness by `epilarynx_effort`, then a step or a ramp of
    /// `epilarynx_flare_cells` up to `epilarynx_flare`. `None` above them
    /// and on the classic engine, whose tube is the original's.
    ///
    /// A cell wholly in the tube or wholly past the ramp takes its diameter
    /// as it is (the defaults give the original's cells, 0.6 × 7 and 1.1 ×
    /// 3, exactly); a cell the tube's top or the ramp crosses takes the mean
    /// area over its length.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn epilarynx(&self, i: usize) -> Option<f32> {
        if !self.jaw || i >= BLADE_START {
            return None;
        }
        let a = &self.motion.anatomy;
        let top = a.epilarynx_cells.clamp(0.0, BLADE_START as f32);
        let flare = a.epilarynx_flare;
        let mut tube = a.epilarynx_diameter;
        if a.epilarynx_effort > 0.0 {
            let x = -a.epilarynx_effort * (self.tenseness - EPILARYNX_TENSENESS) / 0.3;
            let share = powf(2.0, x).clamp(0.5, 2.0);
            tube = (tube * sqrt(share)).min(flare);
        }
        if self.squeeze > 0.0 {
            // Drawn in by the aryepiglottic and false folds (wave 3,
            // w3-vocal): a throat clearing's narrowed tube.
            tube *= sqrt(1.0 - self.squeeze.min(1.0));
        }
        let ramp = a.epilarynx_flare_cells.max(0.0);
        let (from, to) = (i as f32, i as f32 + 1.0);
        if to <= top {
            return Some(tube);
        }
        if from >= top + ramp {
            return Some(flare);
        }
        // The mean area over the cell, from eight points along it.
        let (at, af) = (tube * tube, flare * flare);
        let mut sum = 0.0;
        for k in 0..8 {
            let s = from + (k as f32 + 0.5) / 8.0;
            sum += if s < top {
                at
            } else if ramp <= 0.0 {
                af
            } else {
                at + (af - at) * ((s - top) / ramp).min(1.0)
            };
        }
        Some(sqrt(sum / 8.0))
    }

    /// Rest shape of cell `i` for the current tongue (after the original's
    /// `TractUI.setRestDiameter`).
    #[allow(clippy::cast_precision_loss)]
    fn rest_diameter(&self, i: usize) -> f32 {
        if let Some(d) = self.epilarynx(i) {
            return d;
        }
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
        cosine_tongue(i, self.tongue_index, self.tongue_diameter)
    }

    /// [`rest_diameter`](Self::rest_diameter) with the tongue root's own
    /// shaping of the pharynx (see [`Tract::pharynx`]); the root at rest
    /// changes nothing.
    fn body_diameter(&self, i: usize) -> f32 {
        let d = self.rest_diameter(i);
        if self.pharynx.abs() <= f32::EPSILON || !self.jaw {
            return d;
        }
        d * exp(self.pharynx * pharynx_weight(i))
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
    ///
    /// On the modal body too the jaw follows the height of the cosine
    /// tongue the vowel shape carries (the cosine table's, or what the
    /// body's numbers project nearest to): the jaw is the `LipsAndJaw` tier's,
    /// as before, and only the body under it changes. Following the modal
    /// body's own opening instead (its narrowest cell from the soft palate
    /// on, between that of the closest and of the most open measured vowel)
    /// opened the jaw of /ʌ ɔ/ to 0.7 to 0.8 where the cosine's is under
    /// 0.2, and in sentences /ʌ/ was heard as /ɑ/ 8 times in 18 (vowels
    /// wrong in the LibriSpeech texts 31.1 % against 26.2 % with this jaw).
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
        let modal = self.modal().then(|| self.body_modes());
        let from = self.modal_from();
        for i in 0..N {
            self.target[i] = match &modal {
                Some(q) if (from..modes::TO).contains(&i) => modal_cell(i, q),
                _ => self.body_diameter(i),
            };
        }
        let floor = match self.dynamics {
            Dynamics::Linear => 0.0,
            Dynamics::Damped => -self.motion.articulation.closure_overshoot,
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
    #[allow(clippy::needless_range_loop)] // five arrays, cell by cell
    fn compute_shaped_targets(&mut self, floor: f32) {
        // Lip rounding that belongs to a consonant (/ʃ ʒ tʃ dʒ r/) is the
        // lips' own gesture and moves at their speed. A vowel's moves with
        // the vowel shape, but while a consonant is made on that vowel (the
        // rounding of /u/ under the /n/ of "canoe") the lips, lighter than
        // the tongue body, round at the speed of the back of the tongue
        // (and, as it closes, twice that, `closing_ratio`), so that the
        // consonant opens onto rounded lips: at the body's
        // speed they were a third of the way short at the release of the
        // /n/ and the second formant fell by 200 Hz into the /u/; at the
        // lips' own speed they rounded the end of the vowel before the /n/
        // too, and "canoe" was heard as "can-oh".
        let consonant = self.slots.iter().enumerate().any(|(k, s)| {
            k != ROUNDING_SLOT && s.is_some_and(|s| s.alive && s.c.diameter < CONSONANT_WIDEST)
        });
        if let Some(Some(s)) = self.slots.get(ROUNDING_SLOT)
            && s.alive
        {
            let before = self.target;
            carve(&mut self.target, s.c, floor);
            if consonant {
                let by = if self.own_lips {
                    Mover::Tip
                } else {
                    Mover::Dorsum
                };
                let lips = self.target[LIP_START..].iter().zip(&before[LIP_START..]);
                for (m, (t, b)) in self.mover[LIP_START..].iter_mut().zip(lips) {
                    if *t < b - CARVED {
                        *m = by;
                    }
                }
            }
        }
        if let Some(c) = self.narrowing {
            carve(&mut self.target, c, floor);
        }
        self.body = self.target;
        let (opened, cavity) = self.open_nasal_cavity();
        let mut by = [None; N];
        for (k, s) in self.slots.iter().enumerate() {
            let Some(s) = s.filter(|s| s.alive) else {
                continue;
            };
            if k == ROUNDING_SLOT {
                continue;
            }
            let mut c = s.c;
            let lateral = self.lateral && k == crate::voice::SLOT_MAIN;
            if lateral && intent(k, c.diameter) != Intent::Closure {
                let before = self.target;
                carve_lateral(&mut self.target, c, floor, &self.motion);
                for i in 0..N {
                    if self.target[i] < before[i] - CARVED * 0.1 {
                        by[i] = Some(mover_of(c.index));
                    }
                }
                continue;
            }
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
        for i in LIP_START..N {
            self.parting[i] |= self.diameter[i] <= 0.0;
        }
        for i in 0..N {
            if self.target[i] < self.body[i] - CARVED
                && let Some(m) = by[i]
            {
                self.mover[i] = m;
            } else if opened[i] {
                // Lowered behind a nasal's closure.
                self.mover[i] = cavity;
            } else if self.quick_body && !(self.parting[i] && self.mover[i] == Mover::Tip) {
                // The back of the tongue making a consonant of its own. Not
                // lips parting from a closure of theirs: opening from a
                // /b p/ into an /l r/ they moved at the tongue's speed, half
                // their own, and the burst came out of lips barely apart.
                self.mover[i] = Mover::Dorsum;
            } else if self.mover[i] != Mover::Body
                && (self.diameter[i] - self.target[i]).abs() <= RETURNED
                && self.velocity[i].abs() <= RETURNED_SPEED
            {
                // Back where the vowel wants it and at rest there: the
                // body has it again. Handed over on the way past, still
                // moving at the quick articulator's speed, the slow body
                // carried it on beyond its place and brought it back over
                // the next 100 ms, by more or less as the control blocks
                // happened to fall.
                self.mover[i] = Mover::Body;
                self.parting[i] = false;
            }
        }
        if self.relief > 0.0 {
            self.relieve();
        }
    }

    /// The mouth behind a nasal's closure (physical engine). With the velum
    /// open and the mouth shut in front of it, the mouth between the velum
    /// and the closure is a closed side branch of the nose, and its own
    /// resonance is the murmur's antiresonance: near 1 kHz for /m/, 1.5 to
    /// 2 kHz for /n/, above 3 kHz for /ŋ/, whatever the vowel around them.
    /// Left in the vowel's shape the cavity followed the vowel instead: the
    /// hump of a central or back vowel's body near the velum, a narrow neck
    /// at the cavity's open end, pulled /n/'s antiresonance down to 700 to
    /// 1000 Hz before /ʌ ə ɔ oʊ/ and /m/'s to 450 Hz, where /m/ and nothing
    /// are heard. So the tongue drops behind a nasal closure, to at least
    /// [`ArticulationConfig::nasal_cavity`](crate::ArticulationConfig::nasal_cavity) all the way from the velum to the closure,
    /// before the tip's or the lips' shoulder is carved into it: lowered
    /// only as far as the shoulder, the high front of /i ɪ æ/ stayed under
    /// the closed end of the branch, /n/'s antiresonance rose to 1.75 to
    /// 1.9 kHz, and the recognizer heard such an /n/ as /ŋ/ (8 of the 38
    /// above 1.7 kHz in the sentences, 2 of the 37 below 1.4 kHz). Returns
    /// the cells it moved and the articulator that moves them.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn open_nasal_cavity(&mut self) -> ([bool; N], Mover) {
        let mut moved = [false; N];
        if !self.nasal {
            return (moved, Mover::Dorsum);
        }
        // The closure furthest back in front of the velum.
        let closure = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(k, s)| s.filter(|s| s.alive).map(|s| (k, s.c)))
            .filter(|&(k, c)| {
                k != ROUNDING_SLOT
                    && intent(k, c.diameter) == Intent::Closure
                    && c.index >= (NOSE_START + 3) as f32
            })
            .map(|(_, c)| c.index)
            .fold(None, |m: Option<f32>, i| Some(m.map_or(i, |m| m.min(i))));
        let Some(index) = closure else {
            return (moved, Mover::Dorsum);
        };
        if index < TIP_START as f32 - 2.0 {
            // A velar closure: nothing of the mouth is left behind it.
            return (moved, Mover::Dorsum);
        }
        // Up to the closure: the tip's or the lips' shoulder is carved into
        // the lowered tongue afterwards.
        let cavity_end = (index.max(0.0) as usize).min(LIP_START).min(N);
        // Before a fronted vowel the tongue is kept up, all or part of the
        // way (`ArticulationConfig::nasal_cavity_held`).
        let floor =
            self.motion.articulation.nasal_cavity * (1.0 - self.cavity_held.clamp(0.0, 1.0));
        for (target, moved) in self.target[NOSE_START..cavity_end.max(NOSE_START)]
            .iter_mut()
            .zip(&mut moved[NOSE_START..])
        {
            if *target < floor {
                *target = floor;
                *moved = true;
            }
        }
        // Behind the tongue tip the blade drops as the tip rises, quickly
        // (`Tuning::nasal_cavity_rate`); behind the lips the tongue body
        // lowers at the back of the tongue's speed.
        let by = if index < LIP_PLACE {
            Mover::Cavity
        } else {
            Mover::Dorsum
        };
        (moved, by)
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
            if self.intent_in(k, s.c.diameter) != Intent::Hiss {
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

        // The cells are sent where the jaw will be half-way through the
        // block: sent where it ends the block, they ran ahead of it by half
        // a block, 5 ms at the long block and 1 ms at a short one.
        let jaw = if self.jaw { self.jaw_opening() } else { 0.0 };
        let whole = crate::math::approach(dt, JAW_TIME, self.block_share);
        let half = 1.0 - crate::math::sqrt(1.0 - whole);
        self.jaw_open += (jaw - self.jaw_open) * half;
        self.compute_targets();

        if self.dynamics == Dynamics::Damped {
            self.move_damped(dt);
            self.jaw_open += (jaw - self.jaw_open) * half;
            return;
        }
        self.jaw_open += (jaw - self.jaw_open) * half;

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
        // The gesture score's tiers scale each articulator's speed by the
        // stiffness of what they are doing (`1` otherwise).
        let [body_k, tip_k, lips_k] = self.tier_speed;
        let tip_rate = self.motion.articulation.tongue_tip.stiffness;
        let tip = speed * tip_k * tip_rate;
        let lips = speed * lips_k * tip_rate;
        let dorsum = speed * self.boost * body_k * self.motion.articulation.dorsum_rate;
        let body = speed * self.boost * body_k * self.motion.articulation.tongue_body.stiffness;
        let cavity = speed * self.motion.articulation.nasal_cavity_rate;
        // The tongue tip letting go of an /n/ (the velum open) moves quickly
        // (`ArticulationConfig::nasal_tip_release`), as a speaker's alveolar release
        // does (50 to 100 cm²/s, Stevens): the closure and the tip's
        // shoulder around it, which had narrowed with it, until each cell
        // is back with the vowel or closing again. The shoulder left at
        // the tip's own speed kept a gap of 0.01 cm² for 10 ms after the
        // closure had opened. Only cells let go onto the vowel's shape:
        // out of /n/ into /ð s l/ the tip moves on to its next place, and
        // hastened there too the /ð/ of "in the", "on the" and the /s/ of
        // "at once" were lost (Whisper's words over the 30 held-out
        // sentences 74.3 to 72.2 %, every one of the losses at such an /n/).
        let tip_lets_go = self.velum_openness() > NASAL_RELEASE_VELUM
            && (0..LIP_START).any(|i| {
                self.mover[i] == Mover::Tip
                    && self.diameter[i] <= 0.0
                    && self.diameter[i] < self.target[i]
                    && self.velocity[i] <= 0.0
            });
        if tip_lets_go {
            for i in 0..LIP_START {
                if self.mover[i] == Mover::Tip
                    && self.diameter[i] < self.target[i]
                    && self.target[i] >= self.body[i] - CARVED
                {
                    self.nasal_release[i] = true;
                }
            }
        }
        for i in 0..N {
            let e0 = self.diameter[i] - self.target[i];
            // Consonants close twice as fast as they open, as in the
            // original; the vowel shape moves alike either way.
            let closing = if e0 > 0.0 {
                self.motion.articulation.closing_ratio
            } else {
                1.0
            };
            let w = match self.mover[i] {
                Mover::Body => body,
                Mover::Dorsum => dorsum * closing,
                Mover::Tip if i >= LIP_START => lips * closing,
                Mover::Tip => tip * closing,
                Mover::Cavity => cavity * closing,
            };
            let mut v0 = self.velocity[i];
            let letting_go = self.diameter[i] <= 0.0 && e0 < 0.0 && v0 <= 0.0;
            if self.nasal_release[i] && (e0 > 0.0 || self.mover[i] != Mover::Tip) {
                // Closing again, or handed back to the tongue body once at
                // rest where the vowel wants it (cleared on the way there,
                // part of the way slow again, it got there by more or less
                // as the control blocks happened to fall).
                self.nasal_release[i] = false;
            }
            let w = if self.nasal_release[i] {
                w * self.motion.articulation.nasal_tip_release
            } else {
                w
            };
            if letting_go {
                // Letting go of a closure: surfaces pressed together spring
                // apart (and the air behind them pushes), so the release
                // starts at speed instead of from rest.
                v0 = self.motion.articulation.release_speed * w * -e0;
            }
            let zeta = if self.mover[i] == Mover::Body {
                self.motion.articulation.tongue_body.damping
            } else {
                1.0
            };
            let (e, mut v) = spring_step(e0, v0, w, zeta, dt);
            let mut d = self.target[i] + e;
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
        self.move_lengths(speed, lips_k, dt);

        let amount = dt * self.movement_speed;
        // The velum shuts as fast as it opens (the original's took two
        // and a half times longer): after a nasal, a stop at the same
        // place (/nd/ in "and", "round") must have the nose shut by its
        // release, or it builds no pressure and is not heard at all.
        self.nose_diameter[0] = move_towards(
            self.nose_diameter[0],
            self.velum_target,
            amount * 0.25,
            amount * VELUM_SHUTTING * self.velum_haste,
        );
        self.calculate_reflections();
    }

    /// Where the lips' protrusion is going, cm at native size: what the
    /// [`LipsAndJaw`](crate::Tier::LipsAndJaw) tier or
    /// [`Trombone::set_lip_protrusion`](crate::Trombone::set_lip_protrusion)
    /// sets, or else the rounding the lips are told to make, through the
    /// curve that sets the lumped lip tube's length
    /// ([`crate::phys::protrusion`]) times the fully protruded tube
    /// (`AnatomyConfig::lip_horn`). It comes from the opening the rounding
    /// slot commands, not from the lips' opening now: a /b/ closing over an
    /// /u/ keeps the lips forward, and lips passing through a rounded-sized
    /// opening on their way to a closure do not protrude. The consonants
    /// with lips of their own (/ʃ ʒ tʃ dʒ r w/) set the rounding slot too.
    fn protrusion_goal(&self) -> f32 {
        if let Some(cm) = self.protrusion_tier.or(self.protrusion_set) {
            return cm.clamp(0.0, PROTRUSION_MOST);
        }
        match self.slots.get(ROUNDING_SLOT).copied().flatten() {
            Some(s) if s.alive => {
                let opening = (s.c.diameter - 0.3).max(0.0);
                self.motion.anatomy.lip_horn * crate::phys::protrusion(opening)
            }
            _ => 0.0,
        }
    }

    /// Move the lips' protrusion and the larynx's height over `dt` seconds
    /// (physical engine): the protrusion on the lips' spring (their tier's
    /// stiffness, `lips_rate`, at the articulation speed `speed` and the
    /// tier's speed `lips_k`; critically damped as the lip cells are), the
    /// larynx on its own (`larynx_rate`, `larynx_damping`).
    fn move_lengths(&mut self, speed: f32, lips_k: f32, dt: f32) {
        let a = &self.motion.articulation;
        let w = speed * lips_k * a.lips_and_jaw.stiffness;
        let goal = self.protrusion_goal();
        let (e, v) = spring_step(self.protrusion - goal, self.protrusion_speed, w, 1.0, dt);
        self.protrusion = (goal + e).max(0.0);
        self.protrusion_speed = v;
        let w = speed * self.larynx_speed_ratio * a.larynx_and_pitch.stiffness;
        // The larynx's rest is `larynx_depth` lower: what the gestures and
        // the caller ask for is counted from there.
        let goal = (self.larynx_tier.unwrap_or(self.larynx_target)
            + self.motion.anatomy.larynx_depth)
            .clamp(*LARYNX_RANGE.start(), *LARYNX_RANGE.end());
        let zeta = a.larynx_and_pitch.damping;
        let (e, v) = spring_step(self.larynx - goal, self.larynx_speed, w, zeta, dt);
        self.larynx = goal + e;
        self.larynx_speed = v;
    }

    /// The lips' protrusion and the larynx's lengthening of the tract now,
    /// cm at native size, and how fast each is changing, cm/s (see
    /// `Phys::set_lengths`).
    pub(crate) fn lengths(&self) -> ([f32; 2], [f32; 2]) {
        (
            [self.protrusion, self.larynx],
            [self.protrusion_speed, self.larynx_speed],
        )
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

/// Carve a lateral (/l/) into a target shape (physical engine). The tip
/// touches the ridge in the middle and the air leaves round the sides of
/// the tongue, between them and the teeth: in the tube a narrow channel of
/// their area from the place [`ArticulationConfig::lateral_channel`](crate::ArticulationConfig::lateral_channel) cells forward, with
/// short shoulders at its ends ([`LATERAL_SHOULDER_BACK`] behind it, the
/// tip's in front), the mouth behind it the tongue body's.
///
/// Until round 10 the /l/'s gap counted as one meant to hiss (its opening
/// was under [`HISS_WIDEST`]), so the jaw shut the incisors in front of it as
/// for /s/ and its shoulder was the sibilant's; its first formant stood at
/// 450 to 480 Hz (LPC, middle of the /l/; the LibriSpeech reader's 350 to
/// 450, the owner's 310 to 330) and it was 4 to 6 dB under the vowel after
/// it (theirs about 10). Carved as a short pinch with the tip's shoulders it
/// was heard as /r/ (16 of 27 /l/ after a consonant in short phrases heard
/// right, 21 before); with the channel running back from the place instead
/// the third formant fell to 2700 Hz, the second rose to 1400, and 6 of 27
/// were. With the channel run forward to the front of the ridge (three
/// cells, 35 to 38) or on to the teeth (four) all 27 were, and 28 of 28 in
/// a larger set: the side channels' open end, not the tip, ends the cavity
/// behind the lips. Into the teeth's cell it narrowed the opening there
/// that the lips of a /b/ released into it must pass.
#[allow(clippy::cast_precision_loss)]
fn carve_lateral(target: &mut [f32; N], c: Constriction, floor: f32, motion: &Tuning) {
    let d = (c.diameter - 0.3).max(floor);
    if !(2.0..N as f32).contains(&c.index) || d >= 3.0 {
        return;
    }
    let mut at = c.index;
    while at <= c.index + motion.articulation.lateral_channel.max(0.0) {
        shoulder(target, at, d, LATERAL_SHOULDER_BACK, TIP_SHOULDER_FRONT);
        at += 1.0;
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
    shoulder(target, index, d, back, front);
    if (DENTAL_PLACE..LIP_PLACE).contains(&index) {
        // The broad blade of a dental, raised along the ridge behind the
        // tip, but open: the narrow gap is the tip's alone.
        shoulder(
            target,
            index - 1.0,
            DENTAL_BLADE.max(d),
            DENTAL_BLADE_BACK,
            0.5,
        );
    }
}

/// Narrow `target` to `d` at `index`, with raised-cosine shoulders `back`
/// cells towards the glottis and `front` cells towards the lips.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn shoulder(target: &mut [f32; N], index: f32, d: f32, back: f32, front: f32) {
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
