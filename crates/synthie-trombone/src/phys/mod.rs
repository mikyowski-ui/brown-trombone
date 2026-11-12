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
//! * [`radiation`]: a frequency-dependent load at the lips and nostrils.
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
pub(crate) mod source;

use crate::math::sqrt;
use crate::tract::{N, NOSE_LEN, NOSE_START};
use branch::{Branch, End, Resonator};
use core::f32::consts::PI;
use junction::{Joint, Shunt, Y_MIN, expansion_loss, solve};
use noise::{JET_HEADROOM, Turbulence, V_MAX, full_jet, glottal_noise, jet_noise, reynolds};
use radiation::Radiation;

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

/// Seconds over which displaced-air flow is smoothed (articulators are
/// updated in blocks; their velocity must not step).
const DISPLACEMENT_SMOOTHING: f32 = 0.004;

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

/// Joint in front of which the upper teeth stand in the air stream.
pub(crate) const TEETH_AT: usize = 40;
/// How far behind the teeth (cells) a jet can form and still strike them.
const JET_REACH: usize = 9;

/// When a closure opens while the voice is wanted, the air the walls took
/// up behind it is let go for this long, seconds, with this time constant,
/// instead of all being pushed back out through the opening. Otherwise the
/// mouth stays pressurised for some 20 to 40 ms and the folds cannot
/// restart until it has drained. It is not applied to voiceless releases,
/// whose burst and aspiration are that very air.
const BLEED_TIME: f32 = 0.012;
const BLEED_TAU: f32 = 0.002;

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
    /// Noise pressure of the strongest jet in the mouth, dyn/cm².
    pub jet: f32,
    /// Speed of that jet, cm/s, and the cell it leaves.
    pub jet_speed: f32,
    pub jet_cell: usize,
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
    /// Narrowest cell behind the teeth this block.
    jet_cell: usize,
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

    pub(crate) gauge: Gauge,
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
            nasal_port: crate::Tuning::DEFAULT.nasal_port,
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
            jet_cell: TEETH_AT - 1,
            turbulence: Turbulence::new(0x7B0E_5EED),
            noisy: true,
            v_max: V_MAX,
            flow_sum: [0.0; N + 1],
            jet_sum: [0.0; N + 1],
            summed: 0.0,
            flow_mean: [0.0; N + 1],
            jet_mean: [0.0; N + 1],
            gauge: Gauge::default(),
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
        self.flow_smooth = (dt / noise::FLOW_AVERAGING).min(1.0);
        // Areas depend on the scale: rebuild from the next block.
        self.primed = false;
    }

    /// Take the noise and nasal settings of a voice.
    pub(crate) fn set_tuning(&mut self, tuning: &crate::Tuning) {
        self.nasal_port = tuning.nasal_port;
        self.turbulence.set_tuning(tuning);
        self.turbulence.set_rate(self.dt, self.len);
    }

    /// Seed the turbulence. Equal seeds give equal sound.
    pub(crate) fn seed(&mut self, seed: u32) {
        let old = self.turbulence.clone();
        self.turbulence = Turbulence::new(seed);
        self.turbulence.jet_eff = old.jet_eff;
        self.turbulence.teeth_eff = old.teeth_eff;
        self.turbulence.glottal_eff = old.glottal_eff;
        self.turbulence.corner = old.corner;
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

    /// Silence everything; the shape is kept.
    pub(crate) fn reset(&mut self) {
        self.r = [0.0; N];
        self.l = [0.0; N];
        self.wall_v = [0.0; N];
        self.wall_y = [0.0; N];
        self.wall_w = [0.0; N];
        self.adot = [0.0; N];
        self.bleed = 0.0;
        self.sub_r = [0.0; NS];
        self.sub_l = [0.0; NS];
        for b in &mut self.nose {
            b.reset();
        }
        for b in &mut self.piriform {
            b.reset();
        }
        self.lips.reset();
        self.flow = [0.0; N + 1];
        self.flow_sum = [0.0; N + 1];
        self.jet_sum = [0.0; N + 1];
        self.summed = 0.0;
        self.flow_mean = [0.0; N + 1];
        self.jet_mean = [0.0; N + 1];
        self.turbulence.reset();
        self.gauge = Gauge::default();
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
        for (i, &d) in diameter.iter().enumerate() {
            self.a1[i] = self.area_of(d);
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
            for (nose, &(factor, _)) in self.nose.iter_mut().zip(&NASAL_SIDES) {
                nose.set_port(port * factor);
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
        self.wall_s0 = self.wall_s1;
        self.adot = [0.0; N];
        self.adot_target = [0.0; N];
    }

    /// Jump straight to a shape, with no motion (for measurements).
    pub(crate) fn snap(&mut self, diameter: &[f32; N], velum: f32) {
        self.primed = false;
        self.begin_block(diameter, velum, 1.0, false);
        self.adot = [0.0; N];
    }

    fn refresh(&mut self, block_dt: f32) {
        self.wall_s0 = self.wall_s1;
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
    }

    /// Pressure in cell `i`, dyn/cm².
    pub(crate) fn pressure(&self, i: usize) -> f32 {
        self.r[i] + self.l[i]
    }

    /// Acoustic energy stored in the main tube (arbitrary units).
    #[cfg(test)]
    pub(crate) fn energy(&self, lambda: f32) -> f32 {
        (0..N)
            .map(|i| {
                let a = self.a0[i] + (self.a1[i] - self.a0[i]) * lambda;
                a * (self.r[i] * self.r[i] + self.l[i] * self.l[i])
            })
            .sum()
    }

    /// The wall of cell `i` as a shunt on the node at its right end, plus
    /// the air the moving wall displaces.
    #[inline]
    fn wall_shunt(&self, i: usize) -> Shunt {
        let s = self.wall_s[i];
        if s == 0.0 {
            return Shunt::NONE;
        }
        let rest = self.wall_cv
            * (self.wall_v[i] - self.wall_ck * self.wall_y[i] + self.wall_cw * self.wall_w[i]);
        Shunt {
            g: s * self.wall_cg,
            h: s * rest + self.dx * self.adot[i],
        }
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
        for i in 0..N {
            self.wall_y[i] *= keep;
            let a = self.a0[i] + (self.a1[i] - self.a0[i]) * lambda;
            y[i] = a * INV_RHO_C;
            inv[i] = 1.0 / a.max(A_MIN);
            self.wall_s[i] = self.wall_s0[i] + (self.wall_s1[i] - self.wall_s0[i]) * lambda;
            self.adot[i] += (self.adot_target[i] - self.adot[i]) * self.adot_smooth;
        }
        let mut nr = [0.0f32; N];
        let mut nl = [0.0f32; N];

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
        let (hiss, hiss_teeth) = if self.noisy {
            self.turbulence.tick()
        } else {
            (0.0, 0.0)
        };
        let mut jet = (0.0f32, 0usize);
        for j in 1..N {
            let i = j - 1;
            let mut shunt = self.wall_shunt(i);
            let nose_here = j == NOSE_START;
            if nose_here {
                for b in &self.nose {
                    let s = b.shunt(lambda);
                    shunt.add(s.g, s.h);
                }
            }
            let piriform_here = j == PIRIFORM_AT && self.side_cavities;
            if piriform_here {
                for b in &self.piriform {
                    let s = b.shunt(lambda);
                    shunt.add(s.g, s.h);
                }
            }
            let (k_fwd, k_rev) = expansion_loss(inv[i], inv[j]);
            // A stream leaving a narrow cell breaks into turbulence; one
            // that strikes the teeth does so far more efficiently.
            let mut pn = 0.0;
            if self.noisy {
                let free = jet_noise(
                    self.flow[j],
                    inv[i],
                    inv[j],
                    self.turbulence.jet_eff,
                    self.v_max,
                );
                pn = hiss * free;
                self.jet_sum[j] += free;
                if free > jet.0 {
                    jet = (free, i);
                }
                if j == TEETH_AT && self.jet_cell + 1 < TEETH_AT {
                    let c = self.jet_cell;
                    let struck = jet_noise(
                        self.flow[c + 1],
                        inv[c],
                        inv[j],
                        self.turbulence.teeth_eff,
                        self.v_max,
                    );
                    pn += hiss_teeth * struck;
                    if struck > jet.0 {
                        jet = (struck, c);
                    }
                }
            }
            let s = solve(
                &Joint {
                    y1: y[i],
                    y2: y[j],
                    shunt,
                    k_fwd,
                    k_rev,
                    r: self.visc * inv[j] * inv[j],
                    pn,
                    sealed: false,
                },
                self.r[i],
                self.l[j],
            );
            nl[i] = s.out_left;
            nr[j] = s.out_right;
            self.flow[j] += (s.u - self.flow[j]) * self.flow_smooth;
            self.flow_sum[j] += s.u;
            u_skin += self.wall_commit(i, s.p1);
            if nose_here {
                for b in &mut self.nose {
                    u_nose += b.tick(lambda, s.p1, dt);
                }
            }
            if piriform_here {
                for b in &mut self.piriform {
                    b.tick(lambda, s.p1, dt);
                }
            }
        }

        // Lips.
        let i = N - 1;
        let shunt = self.wall_shunt(i);
        let lip_area = y[i] * RHO_C;
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
        if let Some(load) = self.lips.load(lip_area, dt) {
            joint.y2 = load.y;
            joint.k_fwd = load.k_out;
            if self.noisy {
                let free = jet_noise(
                    self.flow[N],
                    inv[i],
                    0.0,
                    self.turbulence.jet_eff,
                    self.v_max,
                );
                joint.pn = hiss * free;
                self.jet_sum[N] += free;
                if free > jet.0 {
                    jet = (free, i);
                }
            }
            let s = solve(&joint, self.r[i], load.b);
            nl[i] = s.out_left;
            self.lips.commit(load.b + s.out_right, dt);
            u_skin += self.wall_commit(i, s.p1);
            u_lips = s.u;
        } else {
            let s = solve(&joint, self.r[i], 0.0);
            nl[i] = s.out_left;
            u_skin += self.wall_commit(i, s.p1);
        }
        self.flow[N] += (u_lips - self.flow[N]) * self.flow_smooth;
        self.flow_sum[N] += u_lips;
        self.summed += 1.0;
        let u_skin = SKIN_RADIATION * u_skin;

        for i in 0..N {
            self.r[i] = nr[i] * self.att[i];
            self.l[i] = nl[i] * self.att[i];
        }
        self.gauge.u_lips = u_lips;
        self.gauge.u_nose = u_nose;
        self.gauge.u_skin = u_skin;
        self.gauge.jet = jet.0;
        self.gauge.jet_cell = jet.1;
        self.gauge.jet_speed = (self.flow[jet.1 + 1] * inv[jet.1]).max(0.0);
        u_lips + u_nose + u_skin
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
        let pn = if self.noisy && !sealed {
            self.turbulence.breath()
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

        for k in 0..NS {
            self.sub_r[k] = nsr[k] * TRACHEA_LOSS;
            self.sub_l[k] = nsl[k] * TRACHEA_LOSS;
        }
        s.out_right
    }
}

#[cfg(test)]
mod tests;
