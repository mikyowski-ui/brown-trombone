//! The physical engine: the same 44-cell tract as the classic engine, but in
//! real units (centimetres, grams, seconds; pressure in dyn/cm²) and with
//! the parts the classic engine scripts replaced by the physics that causes
//! them.
//!
//! The articulation model is shared with the classic engine: this module
//! only reads cell diameters and turns them into areas. What it adds:
//!
//! * [`junction`]: one scattering rule for every joint, with kinetic and
//!   viscous losses and room for shunts (walls, side branches).
//! * [`radiation`]: a frequency-dependent load at the lips and nostrils,
//!   behind the short tube that rounded lips push forward
//!   ([`protrusion`]).
//! * Yielding walls: each cell's wall is a mass on a soft spring. Its mass
//!   bouncing on the enclosed air gives the closed-tract resonance near
//!   200 Hz; its softness lets the mouth swell during a voiced stop.
//! * A subglottal tube fed by lung pressure, joined to the tract through
//!   the glottis as an ordinary (very narrow, very lossy) joint.
//!
//! Everything is parameterised by the tick `dt` and the cell length
//! `dx = c·dt`, because [`Trombone::set_tract_length`] changes the tick
//! rate.
//!
//! [`Trombone::set_tract_length`]: crate::Trombone::set_tract_length

pub(crate) mod branch;
pub(crate) mod folds;
pub(crate) mod junction;
pub(crate) mod noise;
pub(crate) mod radiation;
pub(crate) mod slide;
pub(crate) mod source;
pub(crate) mod tilt;

use crate::math::sqrt;
use crate::tract::{N, NOSE_LEN, NOSE_START};
use branch::{Branch, End, Resonator};
use core::f32::consts::PI;
use junction::{Joint, Shunt, Y_MIN, expansion_loss, solve};
use noise::{JET_HEADROOM, Turbulence, V_MAX, full_jet, glottal_noise, jet_noise, reynolds};
use radiation::{Head, PORTS, Radiation};
use slide::Slide;

/// Seconds over which the share of the glottal jet's breath noise follows
/// the voicing (see `Phys::voiced_target`).
const BREATH_SHARE_TIME: f32 = 0.005;

/// Density of warm, moist air, g/cm³.
pub(crate) const RHO: f32 = 1.14e-3;
/// Speed of sound in the tract, cm/s.
pub(crate) const C: f32 = 35_000.0;
/// Characteristic impedance times area.
pub(crate) const RHO_C: f32 = RHO * C;
pub(crate) const INV_RHO_C: f32 = 1.0 / RHO_C;
/// Viscosity of air, dyn·s/cm².
pub(crate) const MU: f32 = 1.86e-4;
/// One centimetre of water, dyn/cm².
pub(crate) const CM_H2O: f32 = 980.665;
/// Lumen area below which a cell counts as shut, cm².
pub(crate) const A_MIN: f32 = 1.0e-5;

/// Cross-section scale that reads the articulation model's diameters as
/// anatomy. Fitted with `examples/tune_physical.rs`: at 1.6 every vowel
/// reaches its Peterson & Barney F1 and F2 within 2 %, and the tract holds
/// about as much air as measured ones do.
pub(crate) const AREA_SCALE: f32 = 1.6;

/// Boundary-layer (viscous and thermal) attenuation per centimetre for a
/// tube of 1 cm radius, evaluated at 1.5 kHz; it scales with 1/radius.
const WALL_FRICTION: f32 = 1.17e-3;

/// Wall tissue per unit area, from in-vivo measurements on seven men with
/// the glottis closed (Hanna, Smith & Wolfe 2016, JASA 139, Table III):
/// 0.17 kg of moving tissue and 4.3 kN/m of stiffness over 110 cm² of
/// wall. Mass g/cm², stiffness dyn/cm³. The tissue's own resonance is then
/// near 25 Hz and its yield about 2.8 cm³ per cmH2O, as measured.
const WALL_MASS: f32 = 1.55;
const WALL_STIFFNESS: f32 = 3.9e4;
/// The same study found the walls three times lossier at their 212 Hz
/// resonance against the enclosed air (0.86 MPa·s/m³) than at the tissue's
/// own slow resonance (0.27). One damper cannot be both, and both matter:
/// the first sets how sharp low formants are, the second how fast the
/// mouth swells behind a closure. So the wall drags a second layer of
/// tissue through a damper: slow motion carries the layer along and loses
/// little, fast motion leaves it behind and loses more. Resistances in
/// dyn·s/cm³, the layer's mass in g/cm².
const WALL_RESISTANCE: f32 = 300.0;
const WALL_LAYER_RESISTANCE: f32 = 650.0;
const WALL_LAYER_MASS: f32 = 0.8;

/// Seconds over which the pressure that drives a jet is averaged (see
/// `Phys::jet_cap`): long enough to smooth the voice's own sound waves out
/// of it in the vowel a closure is made from (at 2 ms the closing tap of "a
/// pot of" still made the utterance's loudest sample), short enough for the
/// mouth's pressure to rise with a fricative's gap (at 8 ms the hiss of /s/
/// between vowels lasted 62 ms where it lasts 78).
const PRESSURE_AVERAGING: f32 = 0.004;
/// Diameter units per second (the articulators' speed at the end of the
/// block) above which a closing cell's jet is bounded by the pressure
/// behind it (see `Phys::jet_cap`): at 10 the /f/ of "this fact" still lost
/// 2.4 dB as the /s/ ran into it, at 20 and 40 1.7; the closing bursts went
/// at 5 already.
const SHUTTING_SPEED: f32 = 20.0;

/// Seconds over which displaced-air flow is smoothed (articulators are
/// updated in blocks; their velocity must not step).
const DISPLACEMENT_SMOOTHING: f32 = 0.004;
/// Lumen, cm², below which a closing (or opening) cell displaces less and
/// less air, in proportion to what is left of it. The articulators move a
/// whole block at a steady speed, so a cell that shuts would otherwise go
/// on squeezing out air at full rate through a gap that has all but gone:
/// in the last ticks before the seal the pressure in it rose to thousands
/// of cmH2O and was cut off at the seal, and every closure and release
/// clicked. Real tissue slows as it meets; the air it squeezes out at the
/// end goes sideways, which a tube cannot. Over the last 0.03 cm² the
/// volume lost is a few thousandths of a cm³.
const SQUEEZE_AREA: f32 = 0.03;

/// Cells in the subglottal tube (trachea), lungs at `0`.
pub(crate) const NS: usize = 33;
/// Tracheal area, cm², at the native tract length.
const TRACHEA_AREA: f32 = 2.5;
/// Lung-end resistance relative to the tracheal impedance.
const LUNG_KAPPA: f32 = 0.15;
/// Wave loss per subglottal cell.
const TRACHEA_LOSS: f32 = 0.9985;

/// Kinetic pressure coefficient of the glottal jet (no pressure recovery).
const GLOTTAL_KT: f32 = 1.1;
/// Fold length and thickness for the viscous term, cm, native size.
const FOLD_LENGTH: f32 = 1.4;
const FOLD_THICKNESS: f32 = 0.3;

/// Share of the walls' motion that reaches the outside air as sound. Bone
/// (jaw, palate, teeth) does not move; cheeks and throat do. This is what
/// is heard during a closed /b/: the voice bar.
const SKIN_RADIATION: f32 = 0.4;

/// First cell whose walls radiate from the cheeks (and the floor of the
/// mouth) rather than from the throat: the oral cavity from the velum on.
const CHEEKS_FROM: usize = NOSE_START;

/// Joint in front of which the upper teeth stand in the air stream.
pub(crate) const TEETH_AT: usize = 40;
/// Cell between the edges of the incisors, which the jaw closes to a
/// narrow gap for /s z/ (see [`Phys::set_teeth`]). A jet driven through it
/// strikes them at its entrance, one cell further on than a jet that
/// crosses an open mouth to the back of the upper teeth.
pub(crate) const TEETH_GAP: usize = TEETH_AT + 1;

/// How far rounded lips are pushed forward, at the most, in cells of tube:
/// about 0.7 cm at the native size. Rounding is not only a narrowing: the
/// lips protrude and the lip tube grows longer (by 0.5 to 1 cm for /u/ in
/// men; Fant 1960), which lowers F1 and F2 further than the narrowing
/// alone can without hissing.
const PROTRUSION_CELLS: f32 = 2.0;
/// First cell of the lip opening itself, where lip rounding is centred: a
/// tongue tip's gap just behind the teeth (/s/, /t/) narrows the cells
/// before it, but is not the lips rounding, and must not muffle the hiss.
const LIP_OPENING: usize = 41;
/// Lip opening (articulation diameter of the narrowest cell from
/// [`LIP_OPENING`] on) at which the lips start to protrude...
const PROTRUSION_START: f32 = 1.2;
/// ...and at which they are fully forward: the opening of /u/.
const PROTRUSION_FULL: f32 = 0.4;
/// Lips nearly shut belong to consonants, which do not protrude: /f v/
/// press the lower lip against the teeth, /p b m/ close it. Protrusion
/// fades out again between these two openings.
const PROTRUSION_FADE: (f32, f32) = (0.2, 0.32);

/// `0` to `1` with zero slope at both ends.
#[inline]
fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// How far forward the lips are, `0` (not at all) to `1`
/// ([`PROTRUSION_CELLS`]), for a lip opening of articulation diameter
/// `diameter`: it grows smoothly as the lips round, so moving lips never
/// jump.
#[inline]
pub(crate) fn protrusion(diameter: f32) -> f32 {
    let rounding = smoothstep((PROTRUSION_START - diameter) / (PROTRUSION_START - PROTRUSION_FULL));
    let open = smoothstep((diameter - PROTRUSION_FADE.0) / (PROTRUSION_FADE.1 - PROTRUSION_FADE.0));
    rounding * open
}
/// How far behind the teeth (cells) a jet can form and still strike them.
const JET_REACH: usize = 9;
/// Length of a jet's core, in diameters of the gap it leaves: so far it
/// keeps the speed it left with, and beyond it slows as it spreads, in
/// proportion to the distance (a free round jet's core is 5 to 7
/// diameters; one confined between the tongue and the palate carries a
/// little further). The jet of /s z/ reaches the teeth inside its core;
/// that of /ʃ ʒ/, made twice as far back, has slowed by a tenth when it
/// gets there and strikes them some 2 dB more weakly.
const JET_CORE: f32 = 8.0;
/// Seconds over which the jaw raises the lower incisors to the upper ones,
/// or lowers them again (see [`Phys::set_teeth`]).
const TEETH_TIME: f32 = 0.02;
/// Seconds over which the edge source of /f v θ ð/ comes and goes with
/// the lip or the tongue tip at the teeth (see [`Phys::set_edge`]).
const EDGE_TIME: f32 = 0.02;
/// First cell a jet that strikes the edges of the upper incisors from in
/// front of the tongue can leave: the tongue tip's gap at the teeth.
const EDGE_FROM: usize = TEETH_AT - 2;

/// When a closure opens while the voice is wanted, the air the walls took
/// up behind it is let go for this long, seconds, with this time constant,
/// instead of all being pushed back out through the opening. Otherwise the
/// mouth stays pressurised for some 20 to 40 ms and the folds cannot
/// restart until it has drained. It is not applied to voiceless releases,
/// whose burst and aspiration are that very air.
const BLEED_TIME: f32 = 0.012;
const BLEED_TAU: f32 = 0.002;

/// First cell that counts for [`Gauge::shut`]: the pharynx below it is
/// never closed by a consonant.
const SEAL_FROM: usize = 10;

/// Cells (glottis end first) whose walls a speaker widens to keep a voiced
/// closure voiced: the pharynx, behind every place of closure. Lowering the
/// larynx, pushing the tongue root forward and lifting the soft palate all
/// enlarge it (Westbury 1983).
const EXPANDING: core::ops::Range<usize> = 2..15;
/// Seconds for that widening to get going or stop.
const EXPANSION_TIME: f32 = 0.01;
/// Most the pharynx can be widened by, cm³ at native size; once that much
/// has been taken up the widening stops until the gesture is let go.
const EXPANSION_LIMIT: f32 = 15.0;
/// The widening yields to the air rather than sucking at it: it goes at
/// full rate once the pressure in the pharynx is this high, dyn/cm², and in
/// proportion below (none below the outside air's). A relaxed, pulled-out
/// wall, in effect, not a pump: the mouth never goes below atmospheric.
const EXPANSION_PRESSURE: f32 = 2.0 * CM_H2O;
/// Cell whose pressure the widening yields to, and the seconds over which
/// that pressure is averaged (so the yield is slow, not an audio-rate load).
const EXPANSION_PROBE: usize = 8;
const EXPANSION_PRESSURE_TIME: f32 = 0.004;

/// Loss multiplier for the nasal passages (large wetted surface).
const NASAL_LOSS: f32 = 3.0;
/// One nasal passage, cm², from the port (cell 0, set by the velum) to the
/// nostril: half the nasopharynx, the main cavity, the narrow nasal valve
/// and the nostril. Rounded from MRI area functions (Dang, Honda & Suzuki
/// 1994).
const NASAL_PASSAGE: [f32; NOSE_LEN] = [
    0.0, 1.0, 1.2, 1.4, 1.5, 1.6, 1.6, 1.7, 1.8, 1.9, 2.0, 2.0, 2.0, 2.0, 1.9, 1.8, 1.8, 1.7, 1.6,
    1.5, 1.4, 1.2, 0.9, 0.65, 0.5, 0.5, 0.6, 0.7,
];
/// The two sides are never alike (septum, nasal cycle): area factors and
/// lengths in cells.
const NASAL_SIDES: [(f32, usize); 2] = [(1.15, NOSE_LEN), (0.85, NOSE_LEN - 1)];
/// Maxillary sinus on each side: volume cm³, ostium area cm², ostium
/// length cm, quality factor, cell where it opens.
const SINUS: [(f32, f32, f32, f32, usize); 2] =
    [(15.0, 0.12, 0.6, 6.0, 12), (18.0, 0.10, 0.6, 6.0, 11)];

/// Cells in a piriform sinus branch.
pub(crate) const PIRIFORM_LEN: usize = 6;
/// Main-tube joint where the piriform sinuses open: just above the
/// laryngeal tube.
pub(crate) const PIRIFORM_AT: usize = 8;
/// The two piriform sinuses, cm², mouth to blind end: about 2 cm deep and
/// 2.5 cm³ together, the left shorter (Dang & Honda 1997). A zero
/// in the last slot ends the shorter one early.
const PIRIFORM: [[f32; PIRIFORM_LEN]; 2] = [
    [0.7, 0.7, 0.65, 0.55, 0.45, 0.0],
    [0.7, 0.7, 0.7, 0.65, 0.55, 0.45],
];

/// Length of one cell at the native size, cm: a tick of travel at the
/// native tick rate. Lengths given in centimetres at the native size are
/// this many cells at every size (the cells grow with the speaker).
pub(crate) const CELL_CM: f32 = C / (2.0 * crate::INTERNAL_RATE);

/// The cell the lip horn lengthens when it is on (`AnatomyConfig::lip_horn`):
/// the lips' narrowest, where rounding is centred. It becomes a slide one
/// tick long plus the protrusion, of the cell's own area, so the horn grows
/// out of the lip opening from nothing. (Between the last cell and the
/// radiation load, where the lumped lip tube sits, a slide would have to be
/// shorter than a tick while the lips start to protrude, which a tract
/// whose joints are all solved at once from the last tick's waves cannot
/// do.)
const HORN_CELL: usize = LIP_OPENING;
/// The cells the larynx slide replaces when it is on
/// (`AnatomyConfig::larynx_travel`): the uniform stretch above the
/// laryngeal tube from the piriform sinuses' joint up, two ticks at rest
/// (so it can be shortened by one, 0.36 cm at native size). Its lower end
/// is joint 8 ([`PIRIFORM_AT`]): the sinuses hang on the larynx and move
/// with it, and the pharynx stretches above them. (Made of cells 7 to 9,
/// with the sinuses at joint 7, the slide at rest lowered the F3 of /i/ by
/// 4.7 % and of /u/ by 2.3 %, all of it from the sinuses having moved a
/// cell down: at joint 10 they raised it by 4.1 %.)
pub(crate) const LARYNX_CELLS: core::ops::Range<usize> = PIRIFORM_AT..10;
#[allow(clippy::cast_precision_loss)]
const LARYNX_REST: f32 = (LARYNX_CELLS.end - LARYNX_CELLS.start) as f32;

/// Attenuation of a wave crossing one cell.
#[inline]
pub(crate) fn cell_loss(area: f32, dx: f32, scale: f32) -> f32 {
    if area < A_MIN {
        return 0.9;
    }
    let radius = sqrt(area / PI);
    (1.0 - scale * WALL_FRICTION * dx / radius).clamp(0.5, 1.0)
}

/// Poiseuille resistance of one cell; only matters when nearly shut.
#[inline]
pub(crate) fn viscous_resistance(inv_area: f32, dx: f32) -> f32 {
    8.0 * PI * MU * dx * inv_area * inv_area
}

/// What excites the tract at its lower end.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Drive {
    /// An ideal flow source at a closed glottis (for measuring the tract).
    Flow(f32),
    /// Lung pressure behind a glottis of the given area.
    Glottis {
        /// Glottal area, cm².
        area: f32,
        /// Lung pressure, dyn/cm².
        lung: f32,
    },
}

/// The stream leaving one cell over the last block, as the hiss regulator
/// reads it (see [`Tract::regulate`](crate::tract::Tract::regulate)).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Stream {
    /// The cell is shut.
    pub sealed: bool,
    /// Speed of the stream relative to the speed lung pressure alone would
    /// give it.
    pub share: f32,
    /// Its Reynolds number.
    pub reynolds: f32,
    /// Share of the lung pressure lost to viscosity in and around the cell.
    pub choke: f32,
    /// Noise of the free jet it forms, relative to a jet at lung-pressure
    /// speed.
    pub noise: f32,
}

/// Readings from the last tick, for tests and measurements.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Gauge {
    /// The mouth is shut somewhere in front of the pharynx (a cell from
    /// `SEAL_FROM` on has no lumen).
    pub shut: bool,
    /// Flow through the glottis, cm³/s.
    pub u_glottis: f32,
    /// Pressure just below the glottis, dyn/cm².
    pub p_sub: f32,
    /// Pressure just above the glottis.
    pub p_supra: f32,
    /// Flow out of the lips.
    pub u_lips: f32,
    /// Flow out of the nose.
    pub u_nose: f32,
    /// Flow radiated through the skin.
    pub u_skin: f32,
    /// Flow the edge source of /f v θ ð/ radiates as (see
    /// [`Phys::set_edge`]).
    pub u_edge: f32,
    /// Noise pressure of the strongest jet in the mouth, dyn/cm².
    pub jet: f32,
    /// Speed of that jet, cm/s, and the cell it leaves.
    pub jet_speed: f32,
    pub jet_cell: usize,
    /// Power the air gives up to the slides' moving ends (the piston of a
    /// moving larynx and of lips pushed forward), erg/s: the pressure at
    /// each moving end times the flow its sweep takes up. Negative while
    /// the walls do work on the air.
    pub piston: f32,
}

/// The physical tract.
#[derive(Clone, Debug)]
#[allow(clippy::struct_excessive_bools)] // independent switches
pub(crate) struct Phys {
    dt: f32,
    dx: f32,
    /// Tract length relative to native; cross-sections scale with it too.
    len: f32,
    area_scale: f32,
    lossy: bool,
    walls: bool,
    primed: bool,

    r: [f32; N],
    l: [f32; N],
    a0: [f32; N],
    a1: [f32; N],
    att: [f32; N],
    /// Attenuation at the start of the block (it is interpolated like the area).
    att0: [f32; N],
    /// Each cell's diameter (articulation units) at the start and end of
    /// the block, and its rate of change there in diameter per block: the
    /// shape moves through the block on the cubic these fix, so its speed
    /// carries on smoothly from one block to the next.
    d0: [f32; N],
    d1: [f32; N],
    m0: [f32; N],
    m1: [f32; N],
    /// The articulators' speeds at the end of the coming block, diameter
    /// units per second (see [`Phys::set_motion`]).
    speed: [f32; N],
    visc: f32,

    /// Wall surface of each cell, cm²: now, and at the start and end of
    /// the block (it follows the moving shape tick by tick, like the area).
    wall_s: [f32; N],
    wall_s0: [f32; N],
    wall_s1: [f32; N],
    wall_v: [f32; N],
    wall_y: [f32; N],
    /// Velocity of the dragged tissue layer.
    wall_w: [f32; N],
    wall_cv: f32,
    wall_cg: f32,
    wall_ck: f32,
    wall_cw: f32,
    wall_drag: f32,
    adot: [f32; N],
    adot_target: [f32; N],
    adot_smooth: f32,
    /// Seconds left of a release bleed.
    bleed: f32,
    /// Active widening of the pharynx the speaker asks for, cm³/s at native
    /// size; the rate it is happening at, cm³/s; and the volume taken up so
    /// far, cm³.
    pub(crate) expansion_target: f32,
    expansion: f32,
    expanded: f32,
    /// Pressure in the pharynx, averaged, dyn/cm².
    pharynx_pressure: f32,
    /// Velar port fully open, cm² (both sides together).
    nasal_port: f32,

    sub_r: [f32; NS],
    sub_l: [f32; NS],
    sub_y: [f32; NS],
    sub_inv: [f32; NS],
    glottal_visc: f32,

    nose: [Branch<NOSE_LEN>; 2],
    piriform: [Branch<PIRIFORM_LEN>; 2],
    side_cavities: bool,
    lips: Radiation,

    /// Flow through each joint, averaged over about a millisecond (what
    /// the noise rule sees).
    flow: [f32; N + 1],
    flow_smooth: f32,
    /// Pressure at each joint, averaged over [`PRESSURE_AVERAGING`],
    /// dyn/cm²: what drives a jet through the cell in front of it (see
    /// [`Phys::jet_cap`]).
    press: [f32; N + 1],
    press_smooth: f32,
    /// Narrowest cell behind the teeth this block, and how far its jet
    /// travels to strike them, cm (see [`JET_CORE`]).
    jet_cell: usize,
    jet_travel: f32,
    turbulence: Turbulence,
    noisy: bool,
    /// Fastest jet the present lung pressure can drive, cm/s.
    v_max: f32,
    /// Flow through each joint and noise of the free jet there, summed
    /// over the ticks of the current block, and their means over the last.
    flow_sum: [f32; N + 1],
    jet_sum: [f32; N + 1],
    summed: f32,
    flow_mean: [f32; N + 1],
    jet_mean: [f32; N + 1],
    /// How far the jaw has brought the incisors together, `0` (as the
    /// lips and tongue leave them) to `1` (to the gap `teeth_gap`), at the
    /// start and end of the block, and where it is going.
    teeth0: f32,
    teeth1: f32,
    teeth_target: f32,
    /// Gap between the incisors when they are together, articulation
    /// diameter units (see `Tuning::sibilant_teeth`).
    teeth_gap: f32,
    /// Strength of the edge source (see [`Phys::set_edge`]) at the start
    /// and end of the block, and where it is going; and its strength for a
    /// dental and a labiodental gap (`Tuning::dental_edge`,
    /// `Tuning::labial_edge`).
    edge0: f32,
    edge1: f32,
    /// Length of a control block as a share of the long one (512 internal
    /// samples); see [`crate::math::approach`].
    pub(crate) block_share: f32,
    edge_target: f32,
    dental_edge: f32,
    labial_edge: f32,
    /// A measuring pressure source, dyn/cm², in series at joint `probe.0`
    /// (between cells `probe.0 − 1` and `probe.0`) for the next tick only.
    probe: (usize, f32),
    /// The source tilt the voice takes as it leaves (see [`tilt`]), and its
    /// state.
    tilt: tilt::Shelf,
    radiated: tilt::Tilt,
    /// Whether the folds are asked to sing, `0‥1`, wanted and present (it
    /// moves over `BREATH_SHARE_TIME`): the voice's tilt is on as far as
    /// this says, and the breath noise of the narrow glottal slit of a
    /// voice is pitched higher than that of an open glottis
    /// (`Tuning::voiced_breath_corner`).
    pub(crate) voiced_target: f32,
    voiced: f32,

    /// The tract's length (wave 2A): the fully protruded lip tube, cm at
    /// native size (`AnatomyConfig::lip_horn`; `0` keeps the lumped lip
    /// tube), and the share of the larynx's travel the slide in the lower
    /// pharynx follows (`AnatomyConfig::larynx_travel`; `0` keeps cells
    /// 8 and 9 as cells).
    lip_horn: f32,
    larynx_travel: f32,
    /// Where the lips' protrusion and the larynx's lengthening of the
    /// tract are going (cm at native size) and how fast (cm/s), at the end
    /// of the coming block (see [`Phys::set_lengths`]).
    length_target: [f32; 2],
    length_speed: [f32; 2],
    /// The two slides' lengths, ticks, at the start and end of the block,
    /// and their change per block there: they move on the same cubic as
    /// the areas.
    slide0: [f32; 2],
    slide1: [f32; 2],
    slide_m0: [f32; 2],
    slide_m1: [f32; 2],
    horn: Slide,
    larynx: Slide,
    /// Pressure at the lower and upper ends of the larynx slide and of the
    /// horn this tick, dyn/cm².
    slide_ends: [f32; 4],
    /// Whether the slides' moving ends take up the air they sweep (off only
    /// to measure the delay lines alone).
    pistons: bool,

    pub(crate) gauge: Gauge,

    /// The flows leaving by each port in the last tick, cm³/s (see
    /// [`radiation::LIPS`] and on): the lips with the edge source, each
    /// nostril, the throat's walls and each cheek's (with
    /// [`SKIN_RADIATION`] on them). They sum to what [`Phys::tick`]
    /// returns.
    pub(crate) ports: [f32; PORTS],
    /// The head the ports radiate round (wave 3), and the source tilt's
    /// state for each of its channels.
    pub(crate) head: Head,
    head_tilt: [tilt::Tilt; 3],
}

impl Phys {
    pub(crate) fn new() -> Self {
        let mut p = Self {
            dt: 0.0,
            dx: 0.0,
            len: 1.0,
            area_scale: AREA_SCALE,
            lossy: true,
            walls: true,
            primed: false,
            r: [0.0; N],
            l: [0.0; N],
            a0: [0.0; N],
            a1: [0.0; N],
            att: [1.0; N],
            att0: [1.0; N],
            d0: [0.0; N],
            d1: [0.0; N],
            m0: [0.0; N],
            m1: [0.0; N],
            speed: [0.0; N],
            visc: 0.0,
            wall_s: [0.0; N],
            wall_s0: [0.0; N],
            wall_s1: [0.0; N],
            wall_v: [0.0; N],
            wall_y: [0.0; N],
            wall_w: [0.0; N],
            wall_cv: 1.0,
            wall_cg: 0.0,
            wall_ck: 0.0,
            wall_cw: 0.0,
            wall_drag: 0.0,
            adot: [0.0; N],
            adot_target: [0.0; N],
            adot_smooth: 0.0,
            bleed: 0.0,
            expansion_target: 0.0,
            expansion: 0.0,
            expanded: 0.0,
            pharynx_pressure: 0.0,
            nasal_port: crate::Tuning::DEFAULT.anatomy.nasal_port,
            sub_r: [0.0; NS],
            sub_l: [0.0; NS],
            sub_y: [0.0; NS],
            sub_inv: [0.0; NS],
            glottal_visc: 0.0,
            nose: [Branch::new(End::Open), Branch::new(End::Open)],
            piriform: [Branch::new(End::Closed), Branch::new(End::Closed)],
            side_cavities: true,
            lips: Radiation::default(),
            flow: [0.0; N + 1],
            flow_smooth: 1.0,
            press: [0.0; N + 1],
            press_smooth: 1.0,
            jet_cell: TEETH_AT - 1,
            jet_travel: 0.0,
            turbulence: Turbulence::new(0x7B0E_5EED),
            noisy: true,
            v_max: V_MAX,
            flow_sum: [0.0; N + 1],
            jet_sum: [0.0; N + 1],
            summed: 0.0,
            flow_mean: [0.0; N + 1],
            jet_mean: [0.0; N + 1],
            teeth0: 0.0,
            teeth1: 0.0,
            teeth_target: 0.0,
            teeth_gap: crate::Tuning::DEFAULT.articulation.sibilant_teeth,
            edge0: 0.0,
            edge1: 0.0,
            block_share: 1.0,
            edge_target: 0.0,
            dental_edge: crate::Tuning::DEFAULT.anatomy.dental_edge,
            labial_edge: crate::Tuning::DEFAULT.anatomy.labial_edge,
            probe: (0, 0.0),
            tilt: tilt::Shelf::FLAT,
            radiated: tilt::Tilt::default(),
            voiced_target: 0.0,
            voiced: 0.0,
            lip_horn: crate::Tuning::DEFAULT.anatomy.lip_horn,
            larynx_travel: crate::Tuning::DEFAULT.anatomy.larynx_travel,
            length_target: [0.0; 2],
            length_speed: [0.0; 2],
            slide0: [1.0, LARYNX_REST],
            slide1: [1.0, LARYNX_REST],
            slide_m0: [0.0; 2],
            slide_m1: [0.0; 2],
            horn: Slide::new(),
            larynx: Slide::new(),
            slide_ends: [0.0; 4],
            pistons: true,
            gauge: Gauge::default(),
            ports: [0.0; PORTS],
            head: Head::new(),
            head_tilt: [tilt::Tilt::default(); 3],
        };
        p.set_rate(1.0 / (2.0 * crate::INTERNAL_RATE), 1.0);
        p
    }

    /// Set the tick length and the tract length scale it corresponds to.
    pub(crate) fn set_rate(&mut self, dt: f32, len: f32) {
        self.dt = dt;
        self.dx = C * dt;
        self.len = len;
        self.visc = 8.0 * PI * MU * self.dx;

        // Implicit step of  m·v' = p − b·v − k·y − b₂·(v − w),
        //                   m₂·w' = b₂·(v − w).
        let drag = dt * WALL_LAYER_RESISTANCE / WALL_LAYER_MASS;
        let layer = WALL_LAYER_RESISTANCE / (1.0 + drag);
        let damp = 1.0 / (1.0 + dt * (WALL_RESISTANCE + layer) / WALL_MASS);
        self.wall_cv = damp;
        self.wall_cg = damp * dt / WALL_MASS;
        self.wall_ck = dt * WALL_STIFFNESS / WALL_MASS;
        self.wall_cw = dt * layer / WALL_MASS;
        self.wall_drag = drag;
        self.adot_smooth = (dt / DISPLACEMENT_SMOOTHING).min(1.0);

        // Trachea: uniform, narrowing over the last centimetre to the folds.
        let scale = len * len;
        for k in 0..NS {
            let taper = match NS - 1 - k {
                0 => 0.36,
                1 => 0.56,
                2 => 0.8,
                _ => 1.0,
            };
            let a = TRACHEA_AREA * taper * scale;
            self.sub_y[k] = a * INV_RHO_C;
            self.sub_inv[k] = 1.0 / a;
        }
        let fl = FOLD_LENGTH * len;
        self.glottal_visc = 12.0 * MU * fl * fl * FOLD_THICKNESS * len;

        self.configure_branches();
        self.turbulence.set_rate(dt, len);
        self.tilt = self.turbulence.tilt(dt, len);
        self.flow_smooth = (dt / noise::FLOW_AVERAGING).min(1.0);
        self.press_smooth = (dt / PRESSURE_AVERAGING).min(1.0);
        self.head.set_rate(dt, len);
        // Areas depend on the scale: rebuild from the next block.
        self.primed = false;
    }

    /// Take the noise and nasal settings of a voice.
    pub(crate) fn set_tuning(&mut self, tuning: &crate::Tuning) {
        self.nasal_port = tuning.anatomy.nasal_port;
        self.teeth_gap = tuning.articulation.sibilant_teeth;
        self.dental_edge = tuning.anatomy.dental_edge;
        self.labial_edge = tuning.anatomy.labial_edge;
        let horn = tuning.anatomy.lip_horn.max(0.0);
        let larynx = tuning.anatomy.larynx_travel.max(0.0);
        if (horn > 0.0) != (self.lip_horn > 0.0) {
            // The lips' cell becomes a slide or a cell again: it starts
            // silent (a tuning is set before the voice speaks).
            self.horn.reset();
            self.horn.snap_length(1.0);
        }
        if (larynx > 0.0) != (self.larynx_travel > 0.0) {
            self.larynx.reset();
            self.larynx.snap_length(LARYNX_REST);
        }
        self.lip_horn = horn;
        self.larynx_travel = larynx;
        self.head.set_tuning(tuning);
        self.turbulence.set_tuning(tuning);
        self.turbulence.set_rate(self.dt, self.len);
        self.tilt = self.turbulence.tilt(self.dt, self.len);
    }

    /// The flow a tick returned, as it is heard: while the folds are asked
    /// to sing, with the source tilt (`Tuning::voice_tilt`) on it. The
    /// noise sources took its inverse on the way in, so they leave as they
    /// would without it; while the voice is off (a voiceless consonant, the
    /// breath before a phrase, a burst's flow) nothing is tilted.
    #[inline]
    pub(crate) fn radiate(&mut self, flow: f32) -> f32 {
        self.radiated.blend(&self.tilt, flow, self.voiced)
    }

    /// The head's channels for the ports of the tick just run, `[mixdown,
    /// left ear, right ear]` (see [`Head`]; a channel the head is not
    /// asked for is `0`), each with the source tilt on as [`Phys::radiate`]
    /// puts it on the mono sum.
    #[inline]
    pub(crate) fn radiate_head(&mut self) -> [f32; 3] {
        let ch = self.head.tick(&self.ports);
        let (mix, ears) = self.head.parts();
        let mut out = [0.0f32; 3];
        for k in 0..3 {
            if (k == 0 && mix) || (k > 0 && ears) {
                out[k] = self.head_tilt[k].blend(&self.tilt, ch[k], self.voiced);
            }
        }
        out
    }

    /// Seed the turbulence. Equal seeds give equal sound.
    pub(crate) fn seed(&mut self, seed: u32) {
        let old = self.turbulence.clone();
        self.turbulence = Turbulence::new(seed);
        self.turbulence.jet_eff = old.jet_eff;
        self.turbulence.teeth_eff = old.teeth_eff;
        self.turbulence.glottal_eff = old.glottal_eff;
        self.turbulence.corner = old.corner;
        self.turbulence.teeth_floor = old.teeth_floor;
        self.turbulence.teeth_corner = old.teeth_corner;
        self.turbulence.tilt_depth = old.tilt_depth;
        self.turbulence.tilt_corner = old.tilt_corner;
        self.turbulence.tilt_top = old.tilt_top;
        self.turbulence.tilt_top_corner = old.tilt_top_corner;
        self.turbulence.slit_corner = old.slit_corner;
        self.turbulence.slit_gain = old.slit_gain;
        self.turbulence.set_rate(self.dt, self.len);
    }

    /// Global cross-section scale (fitted once against vowel formants).
    pub(crate) fn set_area_scale(&mut self, scale: f32) {
        self.area_scale = scale;
        self.primed = false;
    }

    pub(crate) fn area_scale(&self) -> f32 {
        self.area_scale
    }

    /// Side cavities are anatomy in real centimetres: they follow the
    /// speaker's size, not the articulation model's units.
    fn configure_branches(&mut self) {
        let size = self.len * self.len;
        for (side, &(factor, cells)) in NASAL_SIDES.iter().enumerate() {
            let mut areas = [0.0f32; NOSE_LEN];
            for (a, &shape) in areas.iter_mut().zip(&NASAL_PASSAGE) {
                *a = shape * factor * size;
            }
            let mut nose = Branch::new(End::Open);
            nose.configure(&areas[..cells], NASAL_LOSS, self.dx);
            let (volume, area, length, q, at) = SINUS[side];
            nose.set_sinus(
                at,
                Resonator::new(
                    volume * size * self.len,
                    area * size,
                    length * self.len,
                    q,
                    self.dt,
                ),
            );
            self.nose[side] = nose;
        }
        for (side, shape) in PIRIFORM.iter().enumerate() {
            let cells = shape.iter().take_while(|&&a| a > 0.0).count();
            let mut areas = [0.0f32; PIRIFORM_LEN];
            for (a, &s) in areas.iter_mut().zip(shape) {
                *a = s * size;
            }
            let mut pocket = Branch::new(End::Closed);
            pocket.configure(&areas[..cells], 1.0, self.dx);
            self.piriform[side] = pocket;
        }
    }

    /// Switch losses and walls off, for checks against lossless theory.
    #[cfg(test)]
    pub(crate) fn set_ideal(&mut self, ideal: bool) {
        self.lossy = !ideal;
        self.walls = !ideal;
        self.side_cavities = !ideal;
        self.noisy = !ideal;
    }

    /// Switch the slides' pistons on or off (to measure the delay lines
    /// alone).
    #[cfg(test)]
    pub(crate) fn set_pistons(&mut self, on: bool) {
        self.pistons = on;
    }

    /// Switch the piriform sinuses on or off (to measure what they do).
    #[cfg(test)]
    pub(crate) fn set_side_cavities(&mut self, on: bool) {
        self.side_cavities = on;
    }

    /// Cross-section, cm², of a cell with the articulation model's diameter.
    #[inline]
    fn area_of(&self, diameter: f32) -> f32 {
        if diameter <= 0.0 {
            return 0.0;
        }
        let d = diameter * self.len;
        self.area_scale * 0.25 * PI * d * d
    }

    /// Inertance, g/cm⁴, of the stretch of tube that protruding lips add in
    /// front of an opening of `area`, cm² (see [`protrusion`]).
    #[inline]
    fn lip_tube(&self, area: f32) -> f32 {
        if area < A_MIN {
            return 0.0;
        }
        // The opening as a diameter of the articulation model.
        let unit = self.area_scale * 0.25 * PI * self.len * self.len;
        let share = protrusion(sqrt(area / unit));
        if share <= 0.0 {
            return 0.0;
        }
        RHO * PROTRUSION_CELLS * self.dx * share / area
    }

    /// Silence everything; the shape is kept.
    pub(crate) fn reset(&mut self) {
        self.r = [0.0; N];
        self.l = [0.0; N];
        self.wall_v = [0.0; N];
        self.wall_y = [0.0; N];
        self.wall_w = [0.0; N];
        self.adot = [0.0; N];
        self.bleed = 0.0;
        self.expansion = 0.0;
        self.expanded = 0.0;
        self.pharynx_pressure = 0.0;
        self.sub_r = [0.0; NS];
        self.sub_l = [0.0; NS];
        for b in &mut self.nose {
            b.reset();
        }
        for b in &mut self.piriform {
            b.reset();
        }
        self.lips.reset();
        self.horn.reset();
        self.larynx.reset();
        self.slide_ends = [0.0; 4];
        self.flow = [0.0; N + 1];
        self.flow_sum = [0.0; N + 1];
        self.jet_sum = [0.0; N + 1];
        self.summed = 0.0;
        self.flow_mean = [0.0; N + 1];
        self.jet_mean = [0.0; N + 1];
        self.turbulence.reset();
        self.radiated = tilt::Tilt::default();
        self.gauge = Gauge::default();
        self.ports = [0.0; PORTS];
        self.head.reset();
        self.head_tilt = [tilt::Tilt::default(); 3];
    }

    /// Close the books on the block just played: average what the streams
    /// did over it, for the hiss regulator.
    pub(crate) fn end_block(&mut self) {
        if self.summed > 0.0 {
            let scale = 1.0 / self.summed;
            for j in 0..=N {
                self.flow_mean[j] = self.flow_sum[j] * scale;
                self.jet_mean[j] = self.jet_sum[j] * scale;
            }
        }
        self.flow_sum = [0.0; N + 1];
        self.jet_sum = [0.0; N + 1];
        self.summed = 0.0;
    }

    /// Speed, cm/s, that the present lung pressure alone would give a jet.
    pub(crate) fn lung_speed(&self) -> f32 {
        self.v_max / JET_HEADROOM
    }

    /// The stream that left cell `i` over the last block.
    pub(crate) fn stream(&self, i: usize) -> Stream {
        let speed = self.lung_speed();
        if speed <= 0.0 {
            return Stream::default();
        }
        let area = self.a1[i];
        let flow = self.flow_mean[i + 1].max(0.0);
        let inv = 1.0 / area.max(A_MIN);
        // Viscous loss over the narrow stretch the cell sits in.
        let mut drop = 0.0;
        for k in i.saturating_sub(2)..(i + 3).min(N) {
            let a = self.a1[k].max(A_MIN);
            drop += self.visc * flow / (a * a);
        }
        Stream {
            sealed: area < A_MIN,
            share: flow * inv / speed,
            reynolds: reynolds(flow, inv),
            choke: drop / (0.5 * RHO * speed * speed),
            noise: self.jet_mean[i + 1] / (self.turbulence.jet_eff * full_jet(speed)),
        }
    }

    /// Whether the jaw is bringing the incisors together, `0..1`: it does
    /// for /s z/, whose jet must strike the teeth through a narrow gap.
    /// The teeth are bone, not part of the tongue and lip shape the
    /// articulation model makes; they narrow the cell between them
    /// ([`TEETH_GAP`]) to `Tuning::sibilant_teeth`, following the jaw over
    /// [`TEETH_TIME`]. The small cavity this closes off between the tongue
    /// tip and the teeth is what gives /s/ its peak at 5 to 8 kHz: open to
    /// the lips, the mouth in front of the tongue is a horn with no
    /// resonance below 12 kHz. Call before [`begin_block`](Self::begin_block).
    pub(crate) fn set_teeth(&mut self, share: f32) {
        self.teeth_target = share.clamp(0.0, 1.0);
    }

    /// Whether a gap meant to hiss is held at the edges of the upper
    /// incisors, `(dental, labiodental)`, each `0..1` (see
    /// [`Tract::edges`](crate::tract::Tract::edges)). The jet of /f v/
    /// leaves the gap between the lower lip and the upper teeth and strikes
    /// their edges and the upper lip at once; that of /θ ð/ the edges of
    /// the teeth in front of the tongue tip. An obstacle in a jet makes
    /// the sound of the force the eddies put on it, a dipole (Curle 1955;
    /// Shadle 1985), and with nothing in front of these edges but open air
    /// it radiates straight out, not through the gap behind it: the noise
    /// of the free jet, a pressure in series with the narrow gap, drives
    /// only as much flow as the gap's inertance lets through, some 20 dB
    /// less. The force is its noise pressure over the face it strikes, and
    /// a force `F` radiates as a flow `F/ρc` would; it scales with the jet
    /// as the free jet's noise does (`Tuning::labial_edge`,
    /// `Tuning::dental_edge` are how much face the eddies load, in areas
    /// of the gap). It adds nothing to the flow through the gap. Call
    /// before [`begin_block`](Self::begin_block).
    pub(crate) fn set_edge(&mut self, (dental, labial): (f32, f32)) {
        self.edge_target =
            dental.clamp(0.0, 1.0) * self.dental_edge + labial.clamp(0.0, 1.0) * self.labial_edge;
    }

    /// How fast each cell's diameter will be changing at the end of the
    /// coming block, diameter units per second (the articulators' own
    /// speeds). Call before [`begin_block`](Self::begin_block).
    pub(crate) fn set_motion(&mut self, speed: &[f32; N]) {
        self.speed = *speed;
    }

    /// How far the lips are pushed forward and the larynx lengthens the
    /// tract at the end of the coming block, `(protrusion, lengthening)`, cm
    /// at native size, and how fast each is changing there, cm/s (see
    /// `Tract::lengths`). Read only while the lip horn or the larynx slide
    /// is on. Call before [`begin_block`](Self::begin_block).
    pub(crate) fn set_lengths(&mut self, (at, speed): ([f32; 2], [f32; 2])) {
        self.length_target = at;
        self.length_speed = speed;
    }

    /// The slides' lengths, ticks, for lengths `cm` (protrusion and
    /// lengthening, cm at native size): the horn is the lips' cell and the
    /// protrusion, the larynx slide its three cells and the share
    /// `larynx_travel` of the lengthening.
    fn slide_lengths(&self, cm: [f32; 2]) -> [f32; 2] {
        [
            1.0 + cm[0].max(0.0) / CELL_CM,
            LARYNX_REST + self.larynx_travel * cm[1] / CELL_CM,
        ]
    }

    /// Take the articulation model's shape for the coming block. `velum` is
    /// how far the velar port is open (`0‥1`), `block_dt` the block length,
    /// `voiced` whether the voice is wanted (see `BLEED_TIME`).
    pub(crate) fn begin_block(
        &mut self,
        diameter: &[f32; N],
        velum: f32,
        block_dt: f32,
        voiced: bool,
    ) {
        self.a0 = self.a1;
        self.d0 = self.d1;
        self.m0 = self.m1;
        self.teeth0 = self.teeth1;
        self.teeth1 += (self.teeth_target - self.teeth1)
            * crate::math::approach(block_dt, TEETH_TIME, self.block_share);
        self.edge0 = self.edge1;
        self.edge1 += (self.edge_target - self.edge1)
            * crate::math::approach(block_dt, EDGE_TIME, self.block_share);
        for (i, &d) in diameter.iter().enumerate() {
            self.a1[i] = self.area_of(d);
            self.d1[i] = d;
            self.m1[i] = self.speed[i] * block_dt;
        }
        if self.lip_horn > 0.0 || self.larynx_travel > 0.0 {
            self.slide0 = self.slide1;
            self.slide_m0 = self.slide_m1;
            self.slide1 = self.slide_lengths(self.length_target);
            let speed = self.slide_lengths(self.length_speed);
            // Speeds through the same map, without its rest lengths.
            self.slide_m1 = [speed[0] - 1.0, speed[1] - LARYNX_REST].map(|v| v * block_dt);
        }
        if voiced && self.primed && (0..N).any(|i| self.a0[i] < A_MIN && self.a1[i] >= A_MIN) {
            self.bleed = BLEED_TIME;
        }
        // A velum resting on the back wall is shut, not a pinhole.
        let open = if velum > 0.01 { velum.min(1.0) } else { 0.0 };
        let port = 0.5 * self.nasal_port * self.len * self.len * open;
        for (nose, &(factor, _)) in self.nose.iter_mut().zip(&NASAL_SIDES) {
            nose.set_port(port * factor);
        }
        let first = !self.primed;
        if first {
            self.a0 = self.a1;
            self.d0 = self.d1;
            self.teeth1 = self.teeth_target;
            self.teeth0 = self.teeth1;
            self.edge1 = self.edge_target;
            self.edge0 = self.edge1;
            self.m0 = [0.0; N];
            self.m1 = [0.0; N];
            for (nose, &(factor, _)) in self.nose.iter_mut().zip(&NASAL_SIDES) {
                nose.set_port(port * factor);
            }
            if self.lip_horn > 0.0 || self.larynx_travel > 0.0 {
                self.slide0 = self.slide1;
                self.slide_m0 = [0.0; 2];
                self.slide_m1 = [0.0; 2];
                self.horn.snap_length(self.slide1[0]);
                self.larynx.snap_length(self.slide1[1]);
            }
            self.primed = true;
        }
        self.refresh(block_dt);
        if first {
            self.wall_s0 = self.wall_s1;
            self.wall_s = self.wall_s1;
        }
    }

    /// Stop the shape where it is at the end of the current block: no
    /// further motion, no displaced air (for measuring a moving tract).
    pub(crate) fn freeze(&mut self) {
        self.a0 = self.a1;
        self.d0 = self.d1;
        self.m0 = [0.0; N];
        self.m1 = [0.0; N];
        self.wall_s0 = self.wall_s1;
        self.adot = [0.0; N];
        self.adot_target = [0.0; N];
        self.slide0 = self.slide1;
        self.slide_m0 = [0.0; 2];
        self.slide_m1 = [0.0; 2];
        self.horn.snap_length(self.slide1[0]);
        self.larynx.snap_length(self.slide1[1]);
    }

    /// Jump straight to a shape, with no motion (for measurements).
    pub(crate) fn snap(&mut self, diameter: &[f32; N], velum: f32) {
        self.primed = false;
        self.speed = [0.0; N];
        self.begin_block(diameter, velum, 1.0, false);
        self.adot = [0.0; N];
    }

    #[allow(clippy::cast_precision_loss)]
    fn refresh(&mut self, block_dt: f32) {
        self.wall_s0 = self.wall_s1;
        self.att0 = self.att;
        for i in 0..N {
            // A cell that is shutting during this block is still open for
            // most of it: its losses and wall are those of the wider end.
            let a = self.a1[i].max(self.a0[i]);
            self.att[i] = if self.lossy {
                cell_loss(a, self.dx, 1.0)
            } else {
                1.0
            };
            let end = self.a1[i];
            self.wall_s1[i] = if self.walls && end >= A_MIN {
                2.0 * sqrt(PI * end) * self.dx
            } else {
                0.0
            };
            self.adot_target[i] = if self.walls {
                (self.a1[i] - self.a0[i]) / block_dt
            } else {
                0.0
            };
        }
        // Where a jet aimed at the teeth would come from.
        let mut narrowest = TEETH_AT - 1;
        for i in TEETH_AT - JET_REACH..TEETH_AT {
            if self.a1[i] < self.a1[narrowest] {
                narrowest = i;
            }
        }
        self.jet_cell = narrowest;
        // The jet leaves at the joint after its cell.
        let strike = TEETH_AT as f32 + self.teeth1;
        self.jet_travel = (strike - (narrowest + 1) as f32) * self.dx;
    }

    /// Switch the noise sources off or on (for measuring around a flow).
    pub(crate) fn set_noisy(&mut self, on: bool) {
        self.noisy = on;
    }

    /// Put a pressure pulse of `p`, dyn/cm², in series at joint `joint`
    /// for the next tick (for measuring how a noise source there is heard).
    pub(crate) fn kick(&mut self, joint: usize, p: f32) {
        self.probe = (joint, p);
    }

    /// Pressure in cell `i`, dyn/cm².
    pub(crate) fn pressure(&self, i: usize) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        if self.larynx_travel > 0.0 && LARYNX_CELLS.contains(&i) {
            // Within the larynx slide: where the cell's middle would be.
            let x = (i - LARYNX_CELLS.start) as f32 + 0.5;
            return self.larynx.pressure_at(x / LARYNX_REST);
        }
        if self.lip_horn > 0.0 && i == HORN_CELL {
            return self.horn.pressure_at(0.5);
        }
        self.r[i] + self.l[i]
    }

    /// Acoustic energy stored in cell `i` now (arbitrary units, those of
    /// [`energy`](Self::energy)).
    pub(crate) fn cell_energy(&self, i: usize) -> f32 {
        self.a1[i] * (self.r[i] * self.r[i] + self.l[i] * self.l[i])
    }

    /// The slides' lengths this tick, ticks: `(horn, larynx)` (the lips'
    /// cell and the protrusion; the three cells of the lower pharynx and
    /// the lengthening).
    #[cfg(test)]
    pub(crate) fn slide_ticks(&self) -> (f32, f32) {
        (self.horn.length(), self.larynx.length())
    }

    /// Acoustic energy stored in the main tube and its slides (arbitrary
    /// units: area times squared pressure waves, summed over the ticks of
    /// travel they hold).
    #[cfg(test)]
    pub(crate) fn energy(&self, lambda: f32) -> f32 {
        let area = |i: usize| self.a0[i] + (self.a1[i] - self.a0[i]) * lambda;
        let mut e: f32 = (0..N)
            .map(|i| area(i) * (self.r[i] * self.r[i] + self.l[i] * self.l[i]))
            .sum();
        if self.lip_horn > 0.0 {
            e += self.horn.energy(area(HORN_CELL));
        }
        if self.larynx_travel > 0.0 {
            #[allow(clippy::cast_precision_loss)]
            let mean = LARYNX_CELLS.map(area).sum::<f32>() / LARYNX_REST;
            e += self.larynx.energy(mean);
        }
        e
    }

    /// The wall of cell `i` (lumen `area`, cm²) as a shunt on the node at
    /// its right end, plus the air the moving articulators displace.
    #[inline]
    fn wall_shunt(&self, i: usize, area: f32) -> Shunt {
        let s = self.wall_s[i];
        if s == 0.0 {
            return Shunt::NONE;
        }
        let rest = self.wall_cv
            * (self.wall_v[i] - self.wall_ck * self.wall_y[i] + self.wall_cw * self.wall_w[i]);
        Shunt {
            g: s * self.wall_cg,
            h: s * rest + self.displaced(i, area),
        }
    }

    /// The air the moving articulators displace out of cell `i` (lumen
    /// `area`, cm²), cm³/s.
    #[inline]
    fn displaced(&self, i: usize, area: f32) -> f32 {
        let squeeze = (area / (SQUEEZE_AREA * self.len * self.len)).min(1.0);
        self.dx * self.adot[i] * squeeze
    }

    /// Move the wall of cell `i` under pressure `p`; returns the flow its
    /// motion takes up, cm³/s.
    #[inline]
    fn wall_commit(&mut self, i: usize, p: f32) -> f32 {
        if self.wall_s[i] == 0.0 {
            // A shut cell's wall just settles.
            self.wall_v[i] *= 0.999;
            self.wall_w[i] *= 0.999;
            return 0.0;
        }
        let v = self.wall_cv
            * (self.wall_v[i] - self.wall_ck * self.wall_y[i] + self.wall_cw * self.wall_w[i])
            + self.wall_cg * p;
        self.wall_v[i] = v;
        self.wall_y[i] += self.dt * v;
        self.wall_w[i] = (self.wall_w[i] + self.wall_drag * v) / (1.0 + self.wall_drag);
        self.wall_s[i] * v
    }

    /// Advance one tick. `lambda` is the position within the current block.
    /// Returns the flow radiated from lips and nose, cm³/s.
    #[allow(clippy::many_single_char_names, clippy::too_many_lines)]
    pub(crate) fn tick(&mut self, lambda: f32, drive: Drive) -> f32 {
        let mut y = [0.0f32; N];
        let mut inv = [0.0f32; N];
        let keep = if self.bleed > 0.0 {
            self.bleed -= self.dt;
            1.0 - (self.dt / BLEED_TAU).min(1.0)
        } else {
            1.0
        };
        // The shape follows the cubic through the block's ends and the
        // articulators' speeds there (Hermite), not straight lines between
        // block ends: a cell closing on a line changed its speed of closing
        // at once at every block boundary, and in a narrow gap that sudden
        // change of the air squeezed out of it went out as a click.
        let (l2, l3) = (lambda * lambda, lambda * lambda * lambda);
        let (h00, h10, h01, h11) = (
            2.0 * l3 - 3.0 * l2 + 1.0,
            l3 - 2.0 * l2 + lambda,
            3.0 * l2 - 2.0 * l3,
            l3 - l2,
        );
        let mut sealed = false;
        let teeth = self.teeth0 + (self.teeth1 - self.teeth0) * lambda;
        let edge = self.edge0 + (self.edge1 - self.edge0) * lambda;
        // The lip opening, as the lips alone make it (the teeth are not
        // the lips rounding).
        let mut lip_opening = f32::INFINITY;
        for i in 0..N {
            self.wall_y[i] *= keep;
            let d = h00 * self.d0[i] + h10 * self.m0[i] + h01 * self.d1[i] + h11 * self.m1[i];
            let mut a = self.area_of(d);
            if i >= LIP_OPENING {
                lip_opening = lip_opening.min(a);
            }
            if i == TEETH_GAP && teeth > 0.0 {
                let gap = self.area_of(self.teeth_gap);
                if gap < a {
                    a += (gap - a) * teeth;
                }
            }
            sealed |= i >= SEAL_FROM && a < A_MIN;
            y[i] = a * INV_RHO_C;
            inv[i] = 1.0 / a.max(A_MIN);
            self.wall_s[i] = self.wall_s0[i] + (self.wall_s1[i] - self.wall_s0[i]) * lambda;
            self.adot[i] += (self.adot_target[i] - self.adot[i]) * self.adot_smooth;
        }
        self.gauge.shut = sealed;
        // The slides (see `slide`): their lengths on the same cubic as the
        // areas, the horn of its cell's area, the larynx slide of the mean
        // of its three cells'. Their walls are their cells' walls, as long
        // as they are; the air their moving ends sweep is the piston.
        let (horn_on, larynx_on) = (self.lip_horn > 0.0, self.larynx_travel > 0.0);
        let mut lar = (0.0f32, 0.0f32);
        let mut piston = [0.0f32; 2];
        if horn_on || larynx_on {
            let at = |k: usize| {
                h00 * self.slide0[k]
                    + h10 * self.slide_m0[k]
                    + h01 * self.slide1[k]
                    + h11 * self.slide_m1[k]
            };
            let sweep = if self.pistons { self.dx / self.dt } else { 0.0 };
            if horn_on {
                self.horn.set_length(at(0));
                self.wall_s[HORN_CELL] *= self.horn.length();
                piston[0] = y[HORN_CELL] * RHO_C * self.horn.growth() * sweep;
            }
            if larynx_on {
                self.larynx.set_length(at(1));
                let mean = LARYNX_CELLS.map(|k| y[k]).sum::<f32>() / LARYNX_REST;
                lar = (mean, 1.0 / (mean * RHO_C).max(A_MIN));
                // The slide's wall hangs on the node at its upper end.
                let top = LARYNX_CELLS.end - 1;
                let walls: f32 = LARYNX_CELLS.map(|k| self.wall_s[k]).sum();
                for k in LARYNX_CELLS.start..top {
                    self.wall_s[k] = 0.0;
                }
                self.wall_s[top] = walls * self.larynx.length() / LARYNX_REST;
                piston[1] = mean * RHO_C * self.larynx.growth() * sweep;
            }
        }
        let (horn_up, horn_down) = if horn_on {
            self.horn.arrivals()
        } else {
            (0.0, 0.0)
        };
        let (lar_up, lar_down) = if larynx_on {
            self.larynx.arrivals()
        } else {
            (0.0, 0.0)
        };
        // Waves sent into the slides: the horn's up and down, the larynx
        // slide's up and down.
        let mut sent = [0.0f32; 4];
        let mut piston_power = 0.0f32;
        let (lar_low, lar_high) = (LARYNX_CELLS.start, LARYNX_CELLS.end);
        // (The larynx slide starts at the piriform sinuses' joint.)
        let piriform_at = PIRIFORM_AT;
        let mut nr = [0.0f32; N];
        let mut nl = [0.0f32; N];
        let widen = self.widening();

        match drive {
            Drive::Flow(u) => {
                nr[0] = self.l[0] + if y[0] >= Y_MIN { u / y[0] } else { 0.0 };
                self.gauge.u_glottis = u;
                self.v_max = V_MAX;
            }
            Drive::Glottis { area, lung } => {
                self.v_max = JET_HEADROOM * sqrt(2.0 * lung.max(0.0) / RHO);
                nr[0] = self.below_glottis(area, lung, y[0]);
            }
        }

        let dt = self.dt;
        let mut u_nose = 0.0;
        let mut u_skin = 0.0;
        // The same, by port: each nostril, and the walls of the cheeks.
        let mut nostrils = [0.0f32; 2];
        let mut u_cheeks = 0.0;
        let (hiss, hiss_teeth) = if self.noisy {
            self.turbulence.tick(self.voiced)
        } else {
            (0.0, 0.0)
        };
        let mut jet = (0.0f32, 0usize);
        // The share of its noise the jet behind the teeth still makes when
        // it reaches them: the cube of the speed it has left, which follows
        // the gap it leaves tick by tick (a closing gap's jet dies away
        // before the teeth as the gap shuts).
        let reach = {
            let width = 2.0 * sqrt(1.0 / (PI * inv[self.jet_cell]));
            let speed = (JET_CORE * width / self.jet_travel.max(1e-6)).min(1.0);
            speed * speed * speed
        };
        for j in 1..N {
            let i = j - 1;
            if larynx_on && j > lar_low && j < lar_high {
                // Inside the larynx slide: no joint.
                continue;
            }
            // Each side of the joint is a cell or a slide's end.
            let (y1, inv1, from_left) = if larynx_on && j == lar_high {
                (lar.0, lar.1, lar_up)
            } else if horn_on && j == HORN_CELL + 1 {
                (y[i], inv[i], horn_up)
            } else {
                (y[i], inv[i], self.r[i])
            };
            let (y2, inv2, from_right) = if larynx_on && j == lar_low {
                (lar.0, lar.1, lar_down)
            } else if horn_on && j == HORN_CELL {
                (y[j], inv[j], horn_down)
            } else {
                (y[j], inv[j], self.l[j])
            };
            let mut shunt = self.wall_shunt(i, y[i] * RHO_C);
            if EXPANDING.contains(&i) {
                if larynx_on && j == lar_high {
                    // The shares of the cells inside the slide too.
                    shunt.add(0.0, widen * LARYNX_REST);
                } else {
                    shunt.add(0.0, widen);
                }
            }
            if larynx_on && j == lar_high {
                // The air displaced out of the whole slide, as long as it is.
                let all: f32 = LARYNX_CELLS.map(|k| self.displaced(k, y[k] * RHO_C)).sum();
                let ticks = self.larynx.length() / LARYNX_REST;
                shunt.add(0.0, all * ticks - self.displaced(i, y[i] * RHO_C));
            }
            if horn_on && j == HORN_CELL + 1 {
                let ticks = self.horn.length();
                shunt.add(0.0, self.displaced(i, y[i] * RHO_C) * (ticks - 1.0));
                shunt.add(0.0, piston[0]);
            }
            if larynx_on && j == lar_low {
                shunt.add(0.0, piston[1]);
            }
            let nose_here = j == NOSE_START;
            if nose_here {
                for b in &self.nose {
                    let s = b.shunt(lambda);
                    shunt.add(s.g, s.h);
                }
            }
            let piriform_here = j == piriform_at && self.side_cavities;
            if piriform_here {
                for b in &self.piriform {
                    let s = b.shunt(lambda);
                    shunt.add(s.g, s.h);
                }
            }
            let (k_fwd, k_rev) = expansion_loss(inv1, inv2);
            // A stream leaving a narrow cell breaks into turbulence; one
            // that strikes the teeth does so far more efficiently.
            let mut pn = 0.0;
            if self.noisy {
                let free = jet_noise(
                    self.flow[j],
                    inv1,
                    inv2,
                    self.turbulence.jet_eff,
                    self.jet_cap(i),
                );
                pn = hiss * free;
                self.jet_sum[j] += free;
                if free > jet.0 {
                    jet = (free, i);
                }
                // A jet from behind the teeth strikes the upper ones, or,
                // as the jaw closes the incisors, the gap between them.
                let at = match j {
                    TEETH_AT => 1.0 - teeth,
                    TEETH_GAP => teeth,
                    _ => 0.0,
                };
                if at > 0.0 && self.jet_cell + 1 < TEETH_AT {
                    let c = self.jet_cell;
                    let struck = jet_noise(
                        self.flow[c + 1],
                        inv[c],
                        inv[j],
                        self.turbulence.teeth_eff,
                        self.jet_cap(c),
                    ) * reach
                        * at;
                    pn += hiss_teeth * struck;
                    if struck > jet.0 {
                        jet = (struck, c);
                    }
                }
            }
            if j == self.probe.0 {
                pn += self.probe.1;
            }
            let s = solve(
                &Joint {
                    y1,
                    y2,
                    shunt,
                    k_fwd,
                    k_rev,
                    r: self.visc * inv2 * inv2,
                    pn,
                    sealed: false,
                },
                from_left,
                from_right,
            );
            if larynx_on && j == lar_high {
                sent[3] = s.out_left;
                self.slide_ends[1] = s.p1;
            } else if horn_on && j == HORN_CELL + 1 {
                sent[1] = s.out_left;
                self.slide_ends[3] = s.p1;
                piston_power += s.p1 * piston[0];
            } else {
                nl[i] = s.out_left;
            }
            if larynx_on && j == lar_low {
                sent[2] = s.out_right;
                self.slide_ends[0] = s.p1;
                piston_power += s.p1 * piston[1];
            } else if horn_on && j == HORN_CELL {
                sent[0] = s.out_right;
                self.slide_ends[2] = s.p1;
            } else {
                nr[j] = s.out_right;
            }
            self.flow[j] += (s.u - self.flow[j]) * self.flow_smooth;
            self.press[j] += (s.p1 - self.press[j]) * self.press_smooth;
            self.flow_sum[j] += s.u;
            if larynx_on && j == lar_high {
                // The stream through the slide is the stream at its ends.
                for k in lar_low + 1..lar_high {
                    self.flow[k] = self.flow[j];
                    self.press[k] = self.press[j];
                    self.flow_sum[k] += s.u;
                }
            }
            let wall = self.wall_commit(i, s.p1);
            u_skin += wall;
            if i >= CHEEKS_FROM {
                u_cheeks += wall;
            }
            if nose_here {
                for (b, side) in self.nose.iter_mut().zip(&mut nostrils) {
                    let u = b.tick(lambda, s.p1, dt);
                    *side += u;
                    u_nose += u;
                }
            }
            if piriform_here {
                for b in &mut self.piriform {
                    b.tick(lambda, s.p1, dt);
                }
            }
        }

        // Lips, protruding as far as their opening says (with the lip
        // horn on, the protrusion is the horn's).
        let i = N - 1;
        let shunt = self.wall_shunt(i, y[i] * RHO_C);
        let lip_area = y[i] * RHO_C;
        let series = if horn_on {
            0.0
        } else {
            self.lip_tube(lip_opening)
        };
        let mut u_lips = 0.0;
        let mut joint = Joint {
            y1: y[i],
            y2: 0.0,
            shunt,
            k_fwd: 0.0,
            k_rev: 0.0,
            r: 0.0,
            pn: 0.0,
            sealed: false,
        };
        if let Some(load) = self.lips.load_through(lip_area, series, dt) {
            joint.y2 = load.y;
            joint.k_fwd = load.k_out;
            if self.noisy {
                let free = jet_noise(
                    self.flow[N],
                    inv[i],
                    0.0,
                    self.turbulence.jet_eff,
                    self.jet_cap(i),
                );
                joint.pn = hiss * free;
                self.jet_sum[N] += free;
                if free > jet.0 {
                    jet = (free, i);
                }
            }
            let s = solve(&joint, self.r[i], load.b);
            nl[i] = s.out_left;
            self.lips
                .commit_through(&load, load.b + s.out_right, s.u, dt);
            let wall = self.wall_commit(i, s.p1);
            u_skin += wall;
            u_cheeks += wall;
            u_lips = s.u;
        } else {
            let s = solve(&joint, self.r[i], 0.0);
            nl[i] = s.out_left;
            let wall = self.wall_commit(i, s.p1);
            u_skin += wall;
            u_cheeks += wall;
        }
        self.probe.1 = 0.0;
        // The jet of /f v θ ð/ on the edges of the teeth and the lip: out
        // of the narrowest gap from the tongue tip's at the teeth to the
        // lips, radiated as the flow its force amounts to.
        let mut u_edge = 0.0;
        if self.noisy && edge > 0.0 {
            let mut c = EDGE_FROM;
            for k in EDGE_FROM + 1..N {
                if inv[k] > inv[c] {
                    c = k;
                }
            }
            let p = jet_noise(
                self.flow[c + 1],
                inv[c],
                0.0,
                self.turbulence.jet_eff,
                self.jet_cap(c),
            );
            if p > 0.0 {
                u_edge = self.turbulence.edge_hiss(self.voiced) * edge * p * INV_RHO_C / inv[c];
            }
        }
        self.flow[N] += (u_lips - self.flow[N]) * self.flow_smooth;
        self.flow_sum[N] += u_lips;
        self.summed += 1.0;
        let cheek = 0.5 * SKIN_RADIATION * u_cheeks;
        self.ports[radiation::LIPS] = u_lips + u_edge;
        self.ports[radiation::NOSTRIL_LEFT] = nostrils[0];
        self.ports[radiation::NOSTRIL_RIGHT] = nostrils[1];
        self.ports[radiation::THROAT] = SKIN_RADIATION * (u_skin - u_cheeks);
        self.ports[radiation::CHEEK_LEFT] = cheek;
        self.ports[radiation::CHEEK_RIGHT] = cheek;
        let u_skin = SKIN_RADIATION * u_skin;

        for i in 0..N {
            let att = self.att0[i] + (self.att[i] - self.att0[i]) * lambda;
            self.r[i] = nr[i] * att;
            self.l[i] = nl[i] * att;
        }
        if horn_on {
            let k = HORN_CELL;
            let att = self.att0[k] + (self.att[k] - self.att0[k]) * lambda;
            self.horn.push(sent[0], sent[1], att);
        }
        if larynx_on {
            let mean = LARYNX_CELLS
                .map(|k| self.att0[k] + (self.att[k] - self.att0[k]) * lambda)
                .sum::<f32>()
                / LARYNX_REST;
            self.larynx.push(sent[2], sent[3], mean);
        }
        self.gauge.piston = piston_power;
        self.gauge.u_lips = u_lips;
        self.gauge.u_nose = u_nose;
        self.gauge.u_skin = u_skin;
        self.gauge.u_edge = u_edge;
        self.gauge.jet = jet.0;
        self.gauge.jet_cell = jet.1;
        self.gauge.jet_speed = (self.flow[jet.1 + 1] * inv[jet.1]).max(0.0);
        u_lips + u_nose + u_skin + u_edge
    }

    /// Fastest a jet out of cell `i` can be: never more than lung pressure
    /// gives, and while the cell is shutting no more than what the pressure
    /// behind it drives against the open air in front, `√(2p/ρ)` with the
    /// headroom of a throat that wins some of it back (the pressure averaged
    /// over [`PRESSURE_AVERAGING`]). In a steady stream the flow over the
    /// area is the stream's speed already, and a fricative's hiss is left as
    /// it was (bounded so in its steady part too, the gap behind a narrow
    /// channel lost some of its drive: /f θ/ came within 10 dB of /s/; and
    /// bounded while a gap closes only to a fricative's, the /f/ of "this
    /// fact" lost 3.7 dB as the /s/ ran into it, so only a cell still
    /// closing faster than [`SHUTTING_SPEED`] is: an articulator aiming past
    /// contact meets it at speed, one settling into a gap slows down). As a
    /// gap shuts the averaged flow lags it:
    /// bounded by lung pressure alone, the jet of a closing tongue tip ran
    /// at its fastest for the last 5 to 8 ms before the closure, where the
    /// pressure behind a gap closing on a voice is a fraction of the lungs',
    /// and that was the loudest moment of an utterance (the tap of "a pot
    /// of": its loudest sample 8 dB over the vowel's; the /d/ of "slid": 6
    /// dB).
    fn jet_cap(&self, i: usize) -> f32 {
        if self.a1[i] >= self.a0[i] || self.speed[i] > -SHUTTING_SPEED {
            // Opening, held, or settling into a fricative's gap: the
            // averaged flow over the area is the stream's own speed (or lags
            // behind it, as a release opens).
            return self.v_max;
        }
        self.v_max
            .min(JET_HEADROOM * sqrt(2.0 * self.press[i].max(0.0) / RHO))
    }

    /// Advance the active widening of the pharynx by one tick; returns the
    /// flow each widening cell takes up, cm³/s.
    #[inline]
    fn widening(&mut self) -> f32 {
        let size = self.len * self.len * self.len;
        let target = if self.expansion_target > 0.0 && self.expanded < EXPANSION_LIMIT * size {
            self.expansion_target * size
        } else {
            0.0
        };
        if self.expansion_target <= 0.0 {
            self.expanded = 0.0;
        }
        self.expansion += (target - self.expansion) * (self.dt / EXPANSION_TIME).min(1.0);
        let p = self.pressure(EXPANSION_PROBE);
        self.pharynx_pressure +=
            (p - self.pharynx_pressure) * (self.dt / EXPANSION_PRESSURE_TIME).min(1.0);
        let yielding = (self.pharynx_pressure / EXPANSION_PRESSURE).clamp(0.0, 1.0);
        let rate = self.expansion * yielding;
        self.expanded += rate * self.dt;
        #[allow(clippy::cast_precision_loss)]
        let cells = EXPANDING.len() as f32;
        rate / cells
    }

    /// The lungs, trachea and glottis. Returns the wave sent up into the
    /// first tract cell (admittance `y0`).
    #[inline]
    #[allow(clippy::many_single_char_names)]
    fn below_glottis(&mut self, area: f32, lung: f32, y0: f32) -> f32 {
        let mut nsr = [0.0f32; NS];
        let mut nsl = [0.0f32; NS];

        // Lungs: a pressure reservoir behind a resistance.
        nsr[0] = (lung - (1.0 - LUNG_KAPPA) * self.sub_l[0]) / (1.0 + LUNG_KAPPA);

        for k in 1..NS {
            let (k_fwd, k_rev) = expansion_loss(self.sub_inv[k - 1], self.sub_inv[k]);
            let s = solve(
                &Joint {
                    y1: self.sub_y[k - 1],
                    y2: self.sub_y[k],
                    shunt: Shunt::NONE,
                    k_fwd,
                    k_rev,
                    r: 0.0,
                    pn: 0.0,
                    sealed: false,
                },
                self.sub_r[k - 1],
                self.sub_l[k],
            );
            nsl[k - 1] = s.out_left;
            nsr[k] = s.out_right;
        }

        // The glottis: the same joint, with a throat.
        let sealed = area < A_MIN;
        let inv_g = 1.0 / area.max(A_MIN);
        let k = 0.5 * RHO * GLOTTAL_KT * inv_g * inv_g;
        self.voiced += (self.voiced_target - self.voiced) * (self.dt / BREATH_SHARE_TIME).min(1.0);
        let pn = if self.noisy && !sealed {
            self.turbulence.breath(self.voiced)
                * glottal_noise(
                    self.flow[0],
                    inv_g,
                    FOLD_LENGTH * self.len,
                    self.turbulence.glottal_eff,
                    self.v_max,
                )
        } else {
            0.0
        };
        let s = solve(
            &Joint {
                y1: self.sub_y[NS - 1],
                y2: y0,
                shunt: Shunt::NONE,
                k_fwd: k,
                k_rev: k,
                r: self.glottal_visc * inv_g * inv_g * inv_g,
                pn,
                sealed,
            },
            self.sub_r[NS - 1],
            self.l[0],
        );
        nsl[NS - 1] = s.out_left;
        self.flow[0] += (s.u - self.flow[0]) * self.flow_smooth;
        self.gauge.u_glottis = s.u;
        self.gauge.p_sub = s.p1;
        self.gauge.p_supra = self.l[0] + s.out_right;
        self.press[0] += (self.gauge.p_supra - self.press[0]) * self.press_smooth;

        for k in 0..NS {
            self.sub_r[k] = nsr[k] * TRACHEA_LOSS;
            self.sub_l[k] = nsl[k] * TRACHEA_LOSS;
        }
        s.out_right
    }
}

#[cfg(test)]
mod tests;
