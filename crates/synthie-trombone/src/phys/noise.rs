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

use super::{MU, RHO};
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
/// Fastest jet the lungs can drive, cm/s (about 15 cmH2O). The averaged
/// flow lags a closure that is opening or shutting, and dividing it by the
/// instantaneous area would otherwise give speeds no breath could reach.
const V_MAX: f32 = 5000.0;
/// Seconds over which the flow that sets the noise level is averaged.
/// Turbulence follows the stream, not each sound wave riding on it; it
/// still follows the voice's own pulsing, as real voiced fricatives do.
pub(crate) const FLOW_AVERAGING: f32 = 0.001;
/// Corner of the high-pass that removes rumble from the noise, Hz.
const RUMBLE_CORNER: f32 = 400.0;

/// Amplitude, dyn/cm², of the noise pressure at a joint where flow `u`
/// (cm³/s, rightward) leaves a cell of inverse area `inv_narrow` for one of
/// inverse area `inv_wide`, for a jet of the given `efficiency`. Zero unless
/// the stream is leaving a narrowing, fast enough to be turbulent.
#[inline]
pub(crate) fn jet_noise(u: f32, inv_narrow: f32, inv_wide: f32, efficiency: f32) -> f32 {
    if u <= 0.0 || inv_narrow <= inv_wide {
        return 0.0;
    }
    // Re = v·d/ν with d the diameter of a round lumen of the same area.
    let re2 = 4.0 * u * u * inv_narrow / (PI * NU * NU);
    if re2 <= RE_CRITICAL * RE_CRITICAL {
        return 0.0;
    }
    let onset = 1.0 - RE_CRITICAL * RE_CRITICAL / re2;
    let v = (u * inv_narrow).min(V_MAX);
    let widening = 1.0 - inv_wide / inv_narrow;
    efficiency * 0.5 * RHO * v * v * (v / V_REF) * widening * widening * onset
}

/// Amplitude of the noise pressure of the glottal jet: flow `u` through a
/// slit of inverse area `inv_area` and length `fold_length`.
#[inline]
pub(crate) fn glottal_noise(u: f32, inv_area: f32, fold_length: f32, efficiency: f32) -> f32 {
    let u = u.abs();
    // A slit's hydraulic diameter is twice its width.
    let re = 2.0 * u / (fold_length * NU);
    if re <= RE_CRITICAL_GLOTTIS {
        return 0.0;
    }
    let onset = 1.0 - RE_CRITICAL_GLOTTIS * RE_CRITICAL_GLOTTIS / (re * re);
    let v = (u * inv_area).min(V_MAX);
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

/// The random part of every noise source in one tract.
#[derive(Clone, Debug)]
pub(crate) struct Turbulence {
    rng: Rng,
    jets: Hiss,
    teeth: Hiss,
    glottis: Hiss,
    r: f32,
    k: f32,
    gain: f32,
    corner: f32,
    /// Efficiency of a free jet, of a jet on the teeth, and of the jet
    /// between the folds.
    pub(crate) jet_eff: f32,
    pub(crate) teeth_eff: f32,
    pub(crate) glottal_eff: f32,
}

impl Turbulence {
    pub(crate) fn new(seed: u32) -> Self {
        Self {
            rng: Rng::new(seed),
            jets: Hiss::default(),
            teeth: Hiss::default(),
            glottis: Hiss::default(),
            r: 1.0,
            k: 1.0,
            gain: 1.0,
            corner: Tuning::DEFAULT.hiss_corner,
            jet_eff: Tuning::DEFAULT.jet_efficiency,
            teeth_eff: Tuning::DEFAULT.jet_efficiency * Tuning::DEFAULT.teeth_gain,
            glottal_eff: Tuning::DEFAULT.glottal_efficiency,
        }
    }

    /// Take the noise settings of a voice. Call `set_rate` afterwards.
    pub(crate) fn set_tuning(&mut self, tuning: &Tuning) {
        self.corner = tuning.hiss_corner;
        self.jet_eff = tuning.jet_efficiency;
        self.teeth_eff = tuning.jet_efficiency * tuning.teeth_gain;
        self.glottal_eff = tuning.glottal_efficiency;
    }

    pub(crate) fn set_rate(&mut self, dt: f32) {
        self.r = 1.0 / (1.0 + 2.0 * PI * RUMBLE_CORNER * dt);
        let w = 2.0 * PI * self.corner * dt;
        self.k = w / (1.0 + w);
        // Variance left by two equal one-pole low-passes in a row, for a
        // white input: k⁴·(1 + a²)/(1 − a²)³ with a = 1 − k. Scale the
        // result back to unit variance, whatever the tick rate.
        let a2 = (1.0 - self.k) * (1.0 - self.k);
        let k2 = self.k * self.k;
        let rest = 1.0 - a2;
        let kept = k2 * k2 * (1.0 + a2) / (rest * rest * rest);
        self.gain = 1.0 / crate::math::sqrt(kept);
    }

    pub(crate) fn reset(&mut self) {
        for h in [&mut self.jets, &mut self.teeth, &mut self.glottis] {
            *h = Hiss::default();
        }
    }

    /// This tick's noise for free jets and for the teeth.
    #[inline]
    pub(crate) fn tick(&mut self) -> (f32, f32) {
        let (r, k, gain) = (self.r, self.k, self.gain);
        (
            self.jets.next(&mut self.rng, r, k, gain),
            self.teeth.next(&mut self.rng, r, k, gain),
        )
    }

    /// This tick's noise for the glottal jet.
    #[inline]
    pub(crate) fn breath(&mut self) -> f32 {
        let (r, k, gain) = (self.r, self.k, self.gain);
        self.glottis.next(&mut self.rng, r, k, gain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slow_or_widening_free_flow_is_silent() {
        // 200 cm³/s through 3 cm²: a vowel. No hiss.
        assert_eq!(jet_noise(200.0, 1.0 / 3.0, 1.0 / 5.0, 1.0), 0.0);
        // Flow into a narrowing makes no jet.
        assert_eq!(jet_noise(400.0, 1.0, 10.0, 1.0), 0.0);
        // Inhaling makes none here either.
        assert_eq!(jet_noise(-400.0, 10.0, 1.0, 1.0), 0.0);
    }

    #[test]
    fn noise_grows_with_the_cube_of_jet_speed() {
        let a = jet_noise(200.0, 5.0, 0.5, 1.0);
        let b = jet_noise(400.0, 5.0, 0.5, 1.0);
        assert!(a > 0.0);
        let ratio = b / a;
        // Cubic, plus the onset just above the critical Reynolds number.
        assert!((8.0..14.0).contains(&ratio), "{ratio}");
    }

    #[test]
    fn hiss_has_unit_variance_and_no_offset() {
        let mut t = Turbulence::new(3);
        // At any tick rate (the rate follows the tract length).
        for rate in [60_000.0, 96_000.0, 192_000.0] {
            t.set_rate(1.0 / rate);
            let (mut sum, mut sq) = (0.0f64, 0.0f64);
            let n = 300_000;
            for _ in 0..n {
                let (x, _) = t.tick();
                sum += f64::from(x);
                sq += f64::from(x * x);
            }
            let mean = sum / f64::from(n);
            let var = sq / f64::from(n);
            assert!(mean.abs() < 0.01, "{rate}: mean {mean}");
            assert!((0.75..1.1).contains(&var), "{rate}: variance {var}");
        }
    }
}
