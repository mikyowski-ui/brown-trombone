//! Turbulence: the hiss of air forced through a narrow place.
//!
//! The classic engine adds band-passed white noise wherever a finger
//! narrows the tract, whether or not any air is moving. Here the noise is a
//! property of the flow. A stream that leaves a narrow cell at speed `v`
//! forms a jet; once its Reynolds number passes a critical value the jet
//! breaks up, and the pressure it throws back into the tract grows with
//! the cube of its speed (Stevens 1971: source pressure ∝ ΔP^1.5). A jet
//! that strikes an obstacle — the teeth — makes far more noise than one
//! that dies in open space (Shadle 1985), which is why /s/ and /sh/ are
//! loud and /f/ and /th/ are faint.
//!
//! The efficiencies below are calibrated against published speech levels;
//! they are not derived from first principles.

use super::tilt::{Shelf, Tilt};
use super::{MU, RHO};
use crate::math::powf;
use crate::noise::Rng;
use crate::tuning::Tuning;
use core::f32::consts::PI;

/// Kinematic viscosity of air, cm²/s.
const NU: f32 = MU / RHO;
/// Reynolds number above which a jet in the tract is turbulent.
const RE_CRITICAL: f32 = 1700.0;
/// The same for the glottal jet (Samlan & Story 2011).
const RE_CRITICAL_GLOTTIS: f32 = 1200.0;
/// Jet speed, cm/s, at which the source pressure equals `efficiency` times
/// the jet's dynamic pressure: the speed driven by about 5 cmH2O.
const V_REF: f32 = 3000.0;
// How efficiently a jet turns into sound comes from the voice's
// [`Tuning`](crate::Tuning): `jet_efficiency` is the share of a free jet's
// dynamic pressure that returns as sound at `V_REF`; `teeth_gain` is the
// same share for a jet that strikes the teeth, relative to a free jet (the
// number looks small only because a free jet's share is further cut by how
// gently the tract widens after it, while the teeth take the whole jet at
// once: in effect a jet on the teeth is 15 to 20 dB louder than the same
// jet in the open, as measured by Shadle 1985); `glottal_efficiency` is the
// share for the jet between the folds; `hiss_corner` is the frequency above
// which turbulence has less and less energy (12 dB per octave here;
// radiation gives 6 of them back).
/// Fastest a jet can be, relative to the speed lung pressure alone would
/// give it (`√(2P/ρ)`). A throat that widens gently behind its narrowest
/// point wins back some pressure, so a steady jet runs up to a third faster
/// than that; beyond it the figure is an artefact. The averaged flow lags a
/// closure that is opening or shutting, and dividing it by the instantaneous
/// area would otherwise give speeds no breath could reach.
pub(crate) const JET_HEADROOM: f32 = 1.35;
/// The cap while there is no lung pressure to go by (an ideal flow source):
/// the one 8 cmH2O gives.
pub(crate) const V_MAX: f32 = 5000.0;
/// Seconds over which the flow that sets the noise level is averaged.
/// Turbulence follows the stream, not each sound wave riding on it; it
/// still follows the voice's own pulsing, as real voiced fricatives do.
pub(crate) const FLOW_AVERAGING: f32 = 0.001;
/// Mixed into the seed of the edge source's generator.
const EDGE_SEED: u32 = 0x00ED_6E5E;
/// Corner of the high-pass that removes rumble from the noise of free jets
/// and of the breath, Hz.
const RUMBLE_CORNER: f32 = 400.0;

/// High-pass coefficient, low-pass coefficient and the gain that brings
/// white noise through them back to unit variance, for a high-pass corner
/// `low` and a low-pass corner `high` (Hz) at tick `dt`.
fn shaping(low: f32, high: f32, dt: f32) -> (f32, f32, f32) {
    let r = 1.0 / (1.0 + 2.0 * PI * low * dt);
    let w = 2.0 * PI * high * dt;
    let k = w / (1.0 + w);
    // Variance left by two equal one-pole low-passes in a row, for a
    // white input: k⁴·(1 + a²)/(1 − a²)³ with a = 1 − k. Scaled back to
    // unit variance whatever the tick rate; the high-pass only takes away
    // what lies below its corner, so the noise above keeps its level.
    let a2 = (1.0 - k) * (1.0 - k);
    let k2 = k * k;
    let rest = 1.0 - a2;
    let kept = k2 * k2 * (1.0 + a2) / (rest * rest * rest);
    (r, k, 1.0 / crate::math::sqrt(kept))
}

/// Amplitude, dyn/cm², of the noise pressure at a joint where flow `u`
/// (cm³/s, rightward) leaves a cell of inverse area `inv_narrow` for one of
/// inverse area `inv_wide`, for a jet of the given `efficiency` that can be
/// no faster than `v_max`. Zero unless the stream is leaving a narrowing,
/// fast enough to be turbulent.
#[inline]
pub(crate) fn jet_noise(
    u: f32,
    inv_narrow: f32,
    inv_wide: f32,
    efficiency: f32,
    v_max: f32,
) -> f32 {
    if u <= 0.0 || inv_narrow <= inv_wide {
        return 0.0;
    }
    // Re = v·d/ν with d the diameter of a round lumen of the same area.
    let re2 = 4.0 * u * u * inv_narrow / (PI * NU * NU);
    if re2 <= RE_CRITICAL * RE_CRITICAL {
        return 0.0;
    }
    let onset = 1.0 - RE_CRITICAL * RE_CRITICAL / re2;
    let v = (u * inv_narrow).min(v_max);
    let widening = 1.0 - inv_wide / inv_narrow;
    efficiency * 0.5 * RHO * v * v * (v / V_REF) * widening * widening * onset
}

/// Reynolds number of flow `u` through a round lumen of inverse area
/// `inv_area`.
#[inline]
pub(crate) fn reynolds(u: f32, inv_area: f32) -> f32 {
    crate::math::sqrt(4.0 * u * u * inv_area / PI) / NU
}

/// Noise pressure of a free jet as fast as lung pressure alone would make
/// it (`speed`, cm/s), per unit efficiency: what jets are measured against.
#[inline]
pub(crate) fn full_jet(speed: f32) -> f32 {
    0.5 * RHO * speed * speed * (speed / V_REF)
}

/// Amplitude of the noise pressure of the glottal jet: flow `u` through a
/// slit of inverse area `inv_area` and length `fold_length`, no faster than
/// `v_max`.
#[inline]
pub(crate) fn glottal_noise(
    u: f32,
    inv_area: f32,
    fold_length: f32,
    efficiency: f32,
    v_max: f32,
) -> f32 {
    let u = u.abs();
    // A slit's hydraulic diameter is twice its width.
    let re = 2.0 * u / (fold_length * NU);
    if re <= RE_CRITICAL_GLOTTIS {
        return 0.0;
    }
    let onset = 1.0 - RE_CRITICAL_GLOTTIS * RE_CRITICAL_GLOTTIS / (re * re);
    let v = (u * inv_area).min(v_max);
    efficiency * 0.5 * RHO * v * v * (v / V_REF).min(2.0) * onset
}

/// Noise shaped like turbulence: no rumble at the bottom, falling away at
/// the top, about unit variance in between.
#[derive(Clone, Debug, Default)]
struct Hiss {
    x: f32,
    y: f32,
    a: f32,
    b: f32,
}

impl Hiss {
    #[inline]
    fn next(&mut self, rng: &mut Rng, r: f32, k: f32, gain: f32) -> f32 {
        // Uniform noise in [−1, 1) has variance 1/3.
        let x = rng.bipolar() * 1.732;
        self.y = r * (self.y + x - self.x);
        self.x = x;
        self.a += k * (self.y - self.a);
        self.b += k * (self.a - self.b);
        self.b * gain
    }
}

/// A one-pole high-pass's state.
#[derive(Clone, Copy, Debug, Default)]
struct HighPass {
    x: f32,
    y: f32,
}

impl HighPass {
    #[inline]
    fn high_pass(&mut self, r: f32, x: f32) -> f32 {
        self.y = r * (self.y + x - self.x);
        self.x = x;
        self.y
    }
}

/// The random part of every noise source in one tract.
#[derive(Clone, Debug)]
pub(crate) struct Turbulence {
    rng: Rng,
    /// The edge source's own generator, so the rest of the noise is as it
    /// would be without it.
    edge_rng: Rng,
    jets: Hiss,
    teeth: Hiss,
    glottis: Hiss,
    edge: Hiss,
    /// High-pass and low-pass coefficients and gain of the noise of free
    /// jets and of the breath, and of the jet that strikes the teeth.
    r: f32,
    k: f32,
    gain: f32,
    r_teeth: f32,
    k_teeth: f32,
    gain_teeth: f32,
    /// Corner above which free jets and the breath have less and less
    /// energy, Hz.
    pub(crate) corner: f32,
    /// Corners below and above which the jet on the teeth has less and
    /// less, Hz.
    pub(crate) teeth_floor: f32,
    pub(crate) teeth_corner: f32,
    /// Efficiency of a free jet, of a jet on the teeth, and of the jet
    /// between the folds.
    pub(crate) jet_eff: f32,
    pub(crate) teeth_eff: f32,
    pub(crate) glottal_eff: f32,
    /// The voice's source tilt (`Tuning::voice_tilt` at
    /// `Tuning::tilt_corner`): every noise source passes its inverse, as
    /// far as the voice is on, before it enters the tract, so that the
    /// tilt the voice takes as it leaves (`Phys::radiate`) leaves the noise
    /// as it was.
    pub(crate) tilt_depth: f32,
    pub(crate) tilt_corner: f32,
    /// Its second section (`Tuning::tilt_top` at `Tuning::tilt_top_corner`).
    pub(crate) tilt_top: f32,
    pub(crate) tilt_top_corner: f32,
    /// The high-passes of a voice's breath noise: their corner (Hz at
    /// native size, `0` for none), coefficient and states.
    pub(crate) slit_corner: f32,
    slit_r: f32,
    /// How much louder that noise is than an open glottis's would be at the
    /// same flow (`Tuning::voiced_breath_gain`, as a factor).
    pub(crate) slit_gain: f32,
    slit: [HighPass; 2],
    untilt: Shelf,
    /// The inverse's state for the jets, the teeth, the glottis and the
    /// edges.
    untilted: [Tilt; 4],
}

impl Turbulence {
    pub(crate) fn new(seed: u32) -> Self {
        Self {
            rng: Rng::new(seed),
            edge_rng: Rng::new(seed ^ EDGE_SEED),
            jets: Hiss::default(),
            teeth: Hiss::default(),
            glottis: Hiss::default(),
            edge: Hiss::default(),
            r: 1.0,
            k: 1.0,
            gain: 1.0,
            r_teeth: 1.0,
            k_teeth: 1.0,
            gain_teeth: 1.0,
            teeth_floor: Tuning::DEFAULT.anatomy.hiss_floor,
            teeth_corner: Tuning::DEFAULT.anatomy.teeth_corner,
            corner: Tuning::DEFAULT.anatomy.hiss_corner,
            jet_eff: Tuning::DEFAULT.anatomy.jet_efficiency,
            teeth_eff: Tuning::DEFAULT.anatomy.jet_efficiency * Tuning::DEFAULT.anatomy.teeth_gain,
            glottal_eff: Tuning::DEFAULT.anatomy.glottal_efficiency,
            tilt_depth: Tuning::DEFAULT.anatomy.voice_tilt,
            tilt_corner: Tuning::DEFAULT.anatomy.tilt_corner,
            tilt_top: Tuning::DEFAULT.anatomy.tilt_top,
            tilt_top_corner: Tuning::DEFAULT.anatomy.tilt_top_corner,
            slit_corner: Tuning::DEFAULT.anatomy.voiced_breath_corner,
            slit_r: 1.0,
            slit_gain: 1.0,
            slit: [HighPass::default(); 2],
            untilt: Shelf::FLAT,
            untilted: [Tilt::default(); 4],
        }
    }

    /// Take the noise settings of a voice. Call `set_rate` afterwards.
    pub(crate) fn set_tuning(&mut self, tuning: &Tuning) {
        self.corner = tuning.anatomy.hiss_corner;
        self.teeth_floor = tuning.anatomy.hiss_floor;
        self.teeth_corner = tuning.anatomy.teeth_corner;
        self.jet_eff = tuning.anatomy.jet_efficiency;
        self.teeth_eff = tuning.anatomy.jet_efficiency * tuning.anatomy.teeth_gain;
        self.glottal_eff = tuning.anatomy.glottal_efficiency;
        self.tilt_depth = tuning.anatomy.voice_tilt;
        self.tilt_corner = tuning.anatomy.tilt_corner;
        self.tilt_top = tuning.anatomy.tilt_top;
        self.tilt_top_corner = tuning.anatomy.tilt_top_corner;
        self.slit_corner = tuning.anatomy.voiced_breath_corner;
        self.slit_gain = powf(10.0, tuning.anatomy.voiced_breath_gain / 20.0);
    }

    /// Set the tick length and the speaker's relative size. A jet's hiss is
    /// pitched by its speed over its width (a fixed Strouhal number), and a
    /// small speaker's jets are as fast but narrower: the corner rises as
    /// the body shrinks.
    pub(crate) fn set_rate(&mut self, dt: f32, size: f32) {
        (self.r, self.k, self.gain) = shaping(RUMBLE_CORNER, self.corner / size, dt);
        (self.r_teeth, self.k_teeth, self.gain_teeth) =
            shaping(self.teeth_floor / size, self.teeth_corner / size, dt);
        self.untilt = self.shelf(dt, size, true);
        self.slit_r = if self.slit_corner > 0.0 {
            1.0 / (1.0 + 2.0 * PI * self.slit_corner / size * dt)
        } else {
            1.0
        };
    }

    /// The shelf the voice takes as it leaves (the inverse of the one the
    /// noise takes here), for ticks of `dt` at relative size `size`.
    pub(crate) fn tilt(&self, dt: f32, size: f32) -> Shelf {
        self.shelf(dt, size, false)
    }

    fn shelf(&self, dt: f32, size: f32, inverse: bool) -> Shelf {
        Shelf::new(
            self.tilt_corner / size,
            self.tilt_depth,
            self.tilt_top_corner / size,
            self.tilt_top,
            dt,
            inverse,
        )
    }

    pub(crate) fn reset(&mut self) {
        for h in [
            &mut self.jets,
            &mut self.teeth,
            &mut self.glottis,
            &mut self.edge,
        ] {
            *h = Hiss::default();
        }
        self.untilted = [Tilt::default(); 4];
        self.slit = [HighPass::default(); 2];
    }

    /// This tick's noise for free jets and for the teeth. A jet striking the
    /// teeth (/s ʃ/) makes its noise from small, fast eddies, reaching well
    /// past 5 kHz (Stevens 1971; Shadle 1985): its noise has a corner of
    /// its own, higher than the rest.
    /// `voiced` (`0‥1`) is how far the voice's tilt is on.
    #[inline]
    pub(crate) fn tick(&mut self, voiced: f32) -> (f32, f32) {
        let (r, k, gain) = (self.r, self.k, self.gain);
        let (rt, kt, gt) = (self.r_teeth, self.k_teeth, self.gain_teeth);
        let jets = self.jets.next(&mut self.rng, r, k, gain);
        let teeth = self.teeth.next(&mut self.rng, rt, kt, gt);
        let [a, b, _, _] = &mut self.untilted;
        (
            a.blend(&self.untilt, jets, voiced),
            b.blend(&self.untilt, teeth, voiced),
        )
    }

    /// This tick's noise for a jet striking the edges of the upper incisors
    /// and the lip in front of them (/f v θ ð/): an obstacle's noise, shaped
    /// like that of the jet on the teeth, from a generator of its own.
    #[inline]
    pub(crate) fn edge_hiss(&mut self, voiced: f32) -> f32 {
        let (r, k, gain) = (self.r_teeth, self.k_teeth, self.gain_teeth);
        let x = self.edge.next(&mut self.edge_rng, r, k, gain);
        self.untilted[3].blend(&self.untilt, x, voiced)
    }

    /// This tick's noise for the glottal jet, `voiced` (`0‥1`) of the way
    /// to that of the narrow slit of vibrating folds: the eddies of a jet
    /// are pitched by its speed over its width (a fixed Strouhal number),
    /// and the slit between vibrating folds is a tenth of a millimetre to a
    /// millimetre wide where an open glottis is several, so the breath
    /// noise of a voice has little below a few kHz (two more one-pole
    /// high-passes at `Tuning::voiced_breath_corner`).
    #[inline]
    pub(crate) fn breath(&mut self, voiced: f32) -> f32 {
        let (r, k, gain) = (self.r, self.k, self.gain);
        let mut x = self.glottis.next(&mut self.rng, r, k, gain);
        if self.slit_r < 1.0 {
            let once = self.slit[0].high_pass(self.slit_r, x);
            let twice = self.slit[1].high_pass(self.slit_r, once);
            x += (twice * self.slit_gain - x) * voiced;
        } else {
            x += (x * self.slit_gain - x) * voiced;
        }
        self.untilted[2].blend(&self.untilt, x, voiced)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slow_or_widening_free_flow_is_silent() {
        // 200 cm³/s through 3 cm²: a vowel. No hiss.
        assert_eq!(jet_noise(200.0, 1.0 / 3.0, 1.0 / 5.0, 1.0, V_MAX), 0.0);
        // Flow into a narrowing makes no jet.
        assert_eq!(jet_noise(400.0, 1.0, 10.0, 1.0, V_MAX), 0.0);
        // Inhaling makes none here either.
        assert_eq!(jet_noise(-400.0, 10.0, 1.0, 1.0, V_MAX), 0.0);
    }

    #[test]
    fn noise_grows_with_the_cube_of_jet_speed() {
        let a = jet_noise(200.0, 5.0, 0.5, 1.0, V_MAX);
        let b = jet_noise(400.0, 5.0, 0.5, 1.0, V_MAX);
        assert!(a > 0.0);
        let ratio = b / a;
        // Cubic, plus the onset just above the critical Reynolds number.
        assert!((8.0..14.0).contains(&ratio), "{ratio}");
    }

    #[test]
    fn hiss_has_unit_variance_and_no_offset() {
        let mut t = Turbulence::new(3);
        // The shaping alone: without the inverse of the voice's tilt.
        t.tilt_depth = 0.0;
        t.tilt_top = 0.0;
        // At any tick rate (the rate follows the tract length). The breath
        // and free jets have unit variance, and the jet on the teeth.
        for rate in [60_000.0, 96_000.0, 192_000.0] {
            t.set_rate(1.0 / rate, 1.0);
            let (mut sum, mut sq, mut breath) = (0.0f64, 0.0f64, 0.0f64);
            let n = 300_000;
            for _ in 0..n {
                let (_, x) = t.tick(0.0);
                sum += f64::from(x);
                sq += f64::from(x * x);
                breath += f64::from(t.breath(0.0).powi(2));
            }
            let mean = sum / f64::from(n);
            let var = sq / f64::from(n);
            let breath = breath / f64::from(n);
            assert!(mean.abs() < 0.01, "{rate}: mean {mean}");
            assert!((0.75..1.1).contains(&var), "{rate}: variance {var}");
            assert!((0.75..1.1).contains(&breath), "{rate}: breath {breath}");
        }
    }
}
