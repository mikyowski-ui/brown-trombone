#![allow(
    dead_code,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Transfer-function measurements for the probe and tuning examples: the
//! spectrum of a tract impulse response, and the resonances in it.

use synthie_trombone::Trombone;

/// A resonance of the tract.
#[derive(Clone, Copy, Debug)]
pub struct Resonance {
    /// Centre frequency, Hz.
    pub hz: f32,
    /// Half-power bandwidth, Hz.
    pub bandwidth: f32,
    /// Peak level, dB (arbitrary reference).
    pub level: f32,
}

/// Magnitude spectrum of `x` (sampled at `rate`) from 0 to `max_hz` in
/// steps of `step` Hz.
pub fn spectrum(x: &[f32], rate: f32, max_hz: f32, step: f32) -> Vec<f32> {
    let bins = (max_hz / step) as usize;
    (0..bins)
        .map(|k| {
            let w = 2.0 * std::f64::consts::PI * f64::from(k as f32 * step) / f64::from(rate);
            // Rotate a phasor instead of calling sin/cos per sample.
            let (rot_re, rot_im) = (w.cos(), -w.sin());
            let (mut ph_re, mut ph_im) = (1.0f64, 0.0f64);
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for &v in x {
                let v = f64::from(v);
                re += v * ph_re;
                im += v * ph_im;
                let t = ph_re * rot_re - ph_im * rot_im;
                ph_im = ph_re * rot_im + ph_im * rot_re;
                ph_re = t;
            }
            (re * re + im * im).sqrt() as f32
        })
        .collect()
}

/// Peaks of a magnitude spectrum above 80 Hz: the largest value within
/// ±40 Hz and at least 3 % of the tallest.
pub fn resonances(mag: &[f32], step: f32) -> Vec<Resonance> {
    let reach = (40.0 / step) as usize;
    let first = (80.0 / step) as usize;
    let n = mag.len();
    let tallest = mag[first..].iter().fold(0.0f32, |m, &v| m.max(v));
    let mut out: Vec<Resonance> = Vec::new();
    for k in first.max(reach)..n.saturating_sub(reach) {
        let window = &mag[k - reach..=k + reach];
        if mag[k] < 0.03 * tallest || window.iter().any(|&v| v > mag[k]) {
            continue;
        }
        let hz = k as f32 * step;
        if out.last().is_some_and(|r| hz - r.hz < 2.0 * step) {
            continue;
        }
        let half = mag[k] * std::f32::consts::FRAC_1_SQRT_2;
        let mut lo = k;
        while lo > 0 && mag[lo] > half {
            lo -= 1;
        }
        let mut hi = k;
        while hi < n - 1 && mag[hi] > half {
            hi += 1;
        }
        out.push(Resonance {
            hz,
            bandwidth: (hi - lo) as f32 * step,
            level: 20.0 * mag[k].max(1e-20).log10(),
        });
    }
    out
}

/// Transfer function magnitude of the voice's current, settled shape.
pub fn transfer(voice: &Trombone, seconds: f32, max_hz: f32, step: f32) -> Vec<f32> {
    let rate = voice.tick_rate();
    let mut x = vec![0.0f32; (seconds * rate) as usize];
    voice.impulse_response(&mut x);
    spectrum(&x, rate, max_hz, step)
}

/// Resonances of the voice's current, settled shape below `max_hz`.
pub fn formants(voice: &Trombone, max_hz: f32) -> Vec<Resonance> {
    let step = 2.0;
    resonances(&transfer(voice, 0.2, max_hz, step), step)
}
