#![allow(
    dead_code,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]
//! Formants of a vowel shape on the physical engine, measured as
//! `tune_physical` does: the peaks of the tract's transfer function.
//!
//! [`measure`] is the reference (it goes through `spectrum::transfer`).
//! [`Meter`] gives the same numbers several times faster, for surveys: it
//! low-passes and decimates the impulse response before taking the same
//! spectrum, and divides the filter's response back out.

use crate::spectrum;
use synthie_trombone::inverse::Shape;
use synthie_trombone::{Engine, Trombone};

/// Seconds of impulse response analysed at the native tract length.
const SECONDS: f32 = 0.08;
/// Highest frequency analysed, Hz, at the native tract length.
const MAX_HZ: f32 = 4200.0;
/// Spectrum step, Hz, at the native tract length.
const STEP: f32 = 4.0;
/// Tick rate of the native tract, Hz.
const NATIVE_RATE: f32 = 96_000.0;

/// A physical-engine voice holding `shape` at `tract_length`, settled; a
/// shape with the modal body's numbers ([`Shape::modes`]) on the modal body
/// (`tongue_model` 1), with [`base`] for the rest of its tuning.
pub fn voice(shape: &Shape, tract_length: f32) -> Trombone {
    let mut t = Trombone::new(48_000.0);
    t.set_engine(Engine::Physical);
    if shape.modes.is_some() {
        let mut tuning = base();
        tuning.anatomy.tongue_model = 1.0;
        t.set_tuning(&tuning);
    }
    t.set_tract_length(tract_length);
    t.set_shape(shape);
    t.settle();
    t
}

/// The tuning a modal shape is measured with: the default, or the profile
/// named by the environment variable `SURVEY_PROFILE` (so that the tables
/// can be solved again for a tract whose other numbers are switched on, such
/// as the lip horn or the epilarynx).
pub fn base() -> synthie_trombone::Tuning {
    std::env::var("SURVEY_PROFILE").map_or(synthie_trombone::Tuning::DEFAULT, |path| {
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        synthie_trombone::VoiceSettings::default()
            .with_profile(&text)
            .unwrap_or_else(|e| panic!("{path}: {e}"))
            .tuning
    })
}

/// The first three peaks of a transfer function magnitude sampled every
/// [`STEP`] Hz, each refined by a parabola through its neighbours.
fn peaks(mag: &[f32]) -> Option<[f32; 3]> {
    let r = spectrum::resonances(mag, STEP);
    let refine = |hz: f32| {
        let k = (hz / STEP) as usize;
        let (a, b, c) = (mag[k - 1].ln(), mag[k].ln(), mag[k + 1].ln());
        hz + STEP * 0.5 * (a - c) / (a - 2.0 * b + c)
    };
    Some([
        refine(r.first()?.hz),
        refine(r.get(1)?.hz),
        refine(r.get(2)?.hz),
    ])
}

/// F1, F2 and F3 of a settled voice, Hz; `None` when the transfer function
/// shows fewer than three peaks.
///
/// The analysis is done in the tract's own time steps, so a tract of any
/// length is measured with the same relative resolution, and the result is
/// then converted to Hz.
pub fn formants(voice: &Trombone) -> Option<[f32; 3]> {
    let mut x = vec![0.0f32; (SECONDS * NATIVE_RATE) as usize];
    voice.impulse_response(&mut x);
    let mag = spectrum::spectrum(&x, NATIVE_RATE, MAX_HZ, STEP);
    let to_hz = voice.tick_rate() / NATIVE_RATE;
    peaks(&mag).map(|f| f.map(|hz| hz * to_hz))
}

/// F1, F2 and F3 of `shape` at `tract_length`, Hz: the reference
/// measurement.
pub fn measure(shape: &Shape, tract_length: f32) -> Option<[f32; 3]> {
    formants(&voice(shape, tract_length))
}

/// Decimation factor of the fast measurement.
const DECIMATE: usize = 8;
/// Taps of its low-pass filter.
const TAPS: usize = 161;

/// The same measurement as [`measure`], faster. Keeps its buffers between
/// calls: use one per thread.
pub struct Meter {
    taps: Vec<f32>,
    /// `DECIMATE / |H|` of the low-pass at every spectrum bin.
    gain: Vec<f32>,
    x: Vec<f32>,
    y: Vec<f32>,
    mag: Vec<f32>,
}

impl Default for Meter {
    fn default() -> Self {
        Self::new()
    }
}

impl Meter {
    /// A meter that analyses the same [`SECONDS`] of impulse response as
    /// [`measure`].
    pub fn new() -> Self {
        Self::with_seconds(SECONDS)
    }

    /// A meter that analyses only the first `seconds` of the impulse
    /// response (at the native tract length). Vowel resonances have died
    /// away after 40 ms, so half the reference length gives nearly the
    /// same numbers in half the time.
    pub fn with_seconds(seconds: f32) -> Self {
        // Kaiser-windowed sinc: flat to MAX_HZ, down by the time the
        // decimated signal would alias back into that band.
        let nyquist = NATIVE_RATE / (2.0 * DECIMATE as f32);
        let cutoff = f64::from(nyquist) / f64::from(NATIVE_RATE);
        let beta = 9.0f64;
        let mid = (TAPS - 1) as f64 / 2.0;
        let bessel = |x: f64| {
            let (mut sum, mut term) = (1.0f64, 1.0f64);
            for k in 1..40 {
                term *= (x / (2.0 * f64::from(k))).powi(2);
                sum += term;
            }
            sum
        };
        let taps: Vec<f32> = (0..TAPS)
            .map(|n| {
                let t = n as f64 - mid;
                let sinc = if t == 0.0 {
                    2.0 * cutoff
                } else {
                    (2.0 * std::f64::consts::PI * cutoff * t).sin() / (std::f64::consts::PI * t)
                };
                let w = bessel(beta * (1.0 - (t / mid).powi(2)).max(0.0).sqrt()) / bessel(beta);
                (sinc * w) as f32
            })
            .collect();
        let bins = (MAX_HZ / STEP) as usize;
        let gain = (0..bins)
            .map(|k| {
                let w = 2.0 * std::f64::consts::PI * f64::from(k as f32 * STEP)
                    / f64::from(NATIVE_RATE);
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for (n, &h) in taps.iter().enumerate() {
                    re += f64::from(h) * (w * n as f64).cos();
                    im -= f64::from(h) * (w * n as f64).sin();
                }
                (DECIMATE as f64 / (re * re + im * im).sqrt()) as f32
            })
            .collect();
        Self {
            taps,
            gain,
            x: vec![0.0; (seconds * NATIVE_RATE) as usize],
            y: Vec::new(),
            mag: Vec::new(),
        }
    }

    /// F1, F2 and F3 of a settled voice, Hz.
    pub fn formants(&mut self, voice: &Trombone) -> Option<[f32; 3]> {
        voice.impulse_response(&mut self.x);
        let n = self.x.len();
        let out = (n + TAPS - 1).div_ceil(DECIMATE);
        self.y.clear();
        for m in 0..out {
            // y[m] = Σ taps[j] · x[m·DECIMATE − j]
            let at = m * DECIMATE;
            let lo = (at + 1).saturating_sub(n);
            let hi = TAPS.min(at + 1);
            let mut acc = 0.0f32;
            for j in lo..hi {
                acc += self.taps[j] * self.x[at - j];
            }
            self.y.push(acc);
        }
        spectrum_wide(&self.y, NATIVE_RATE / DECIMATE as f32, &mut self.mag);
        for (m, g) in self.mag.iter_mut().zip(&self.gain) {
            *m *= g;
        }
        let to_hz = voice.tick_rate() / NATIVE_RATE;
        peaks(&self.mag).map(|f| f.map(|hz| hz * to_hz))
    }

    /// F1, F2 and F3 of `shape` at `tract_length`, Hz.
    pub fn measure(&mut self, shape: &Shape, tract_length: f32) -> Option<[f32; 3]> {
        self.formants(&voice(shape, tract_length))
    }
}

/// `spectrum::spectrum(x, rate, MAX_HZ, STEP)`, the same arithmetic done
/// for several bins side by side.
fn spectrum_wide(x: &[f32], rate: f32, out: &mut Vec<f32>) {
    const LANES: usize = 8;
    let bins = (MAX_HZ / STEP) as usize;
    out.clear();
    for first in (0..bins).step_by(LANES) {
        let mut rot_re = [0.0f64; LANES];
        let mut rot_im = [0.0f64; LANES];
        for j in 0..LANES {
            let w =
                2.0 * std::f64::consts::PI * f64::from((first + j) as f32 * STEP) / f64::from(rate);
            rot_re[j] = w.cos();
            rot_im[j] = -w.sin();
        }
        let mut ph_re = [1.0f64; LANES];
        let mut ph_im = [0.0f64; LANES];
        let mut re = [0.0f64; LANES];
        let mut im = [0.0f64; LANES];
        for &v in x {
            let v = f64::from(v);
            for j in 0..LANES {
                re[j] += v * ph_re[j];
                im[j] += v * ph_im[j];
                let t = ph_re[j] * rot_re[j] - ph_im[j] * rot_im[j];
                ph_im[j] = ph_re[j] * rot_im[j] + ph_im[j] * rot_re[j];
                ph_re[j] = t;
            }
        }
        for j in 0..LANES.min(bins - first) {
            out.push((re[j] * re[j] + im[j] * im[j]).sqrt() as f32);
        }
    }
}

/// Small deterministic random generator.
pub struct Lcg(pub u64);

impl Lcg {
    /// Uniform in `0‥1`.
    pub fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in `lo‥hi`.
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}
