#![allow(
    dead_code,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! The hiss of a fricative said between two vowels, measured the way the
//! judge's naturalness measure does on recordings (`tools/judge/natural.py`):
//! its spectrum, how broad it is, and how it starts and stops.

use synthie_trombone::{Talker, VoiceSettings};

/// Output rate of the measurements, Hz.
pub const SR: f32 = 48_000.0;
/// Width and number of the spectrum's bands (500 Hz up to 8 kHz).
pub const BAND_HZ: f32 = 500.0;
pub const BANDS: usize = 16;
/// The envelope: 10 ms windows every 2.5 ms.
const HOP: usize = 120;
const WINDOW: usize = 480;
/// Edge of the band the hiss's envelope is measured in, Hz: above the
/// vowels' own highs, and below most of the hiss (/s z/ have little under
/// 3 kHz).
fn hiss_from(phoneme: &str) -> f32 {
    if matches!(phoneme, "S" | "Z") {
        3000.0
    } else {
        2000.0
    }
}

/// One fricative's hiss.
#[derive(Clone, Debug)]
pub struct Hiss {
    /// Level per band, dB under the loudest band.
    pub bands: Vec<f32>,
    /// Centroid of the power over 1 to 8 kHz, Hz.
    pub centroid: f32,
    /// Geometric over arithmetic mean of the power over 1 to 8 kHz (1 for
    /// white noise, near 0 for a single whistle).
    pub flatness: f32,
    /// Width of the band within 6 dB of the peak (100 Hz bins, smoothed
    /// over 300 Hz), Hz.
    pub width6: f32,
    /// Time the hiss takes from 15 dB under its peak to 3 dB under, and
    /// back down, ms.
    pub rise_ms: f32,
    pub fall_ms: f32,
    /// How long it stays within 10 dB of its peak, ms.
    pub dur_ms: f32,
    /// Peak level of the hiss against the loudest 10 ms of the vowels, dB.
    pub level: f32,
}

/// Power of `x` at `hz` (Hann window).
fn power(x: &[f32], hz: f32) -> f32 {
    let n = x.len() as f32;
    let (mut re, mut im) = (0.0f32, 0.0f32);
    for (k, &v) in x.iter().enumerate() {
        let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * k as f32 / n).cos();
        let ph = 2.0 * std::f32::consts::PI * hz * k as f32 / SR;
        re += w * v * ph.cos();
        im -= w * v * ph.sin();
    }
    re * re + im * im
}

/// Second-order high-pass at `hz` (Butterworth).
fn high_pass(x: &[f32], hz: f32) -> Vec<f32> {
    let w = 2.0 * std::f64::consts::PI * f64::from(hz) / f64::from(SR);
    let alpha = w.sin() / (2.0 * std::f64::consts::FRAC_1_SQRT_2);
    let a0 = 1.0 + alpha;
    let (b0, b1) = ((1.0 + w.cos()) / 2.0 / a0, -(1.0 + w.cos()) / a0);
    let (a1, a2) = (-2.0 * w.cos() / a0, (1.0 - alpha) / a0);
    let (mut x1, mut x2, mut y1, mut y2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    x.iter()
        .map(|&v| {
            let v = f64::from(v);
            let y = b0 * v + b1 * x1 + b0 * x2 - a1 * y1 - a2 * y2;
            x2 = x1;
            x1 = v;
            y2 = y1;
            y1 = y;
            y as f32
        })
        .collect()
}

/// Level in dB per hop (10 ms windows).
fn envelope(x: &[f32]) -> Vec<f32> {
    (0..x.len() / HOP)
        .map(|f| {
            let c = f * HOP + HOP / 2;
            let (a, b) = (c.saturating_sub(WINDOW / 2), (c + WINDOW / 2).min(x.len()));
            let ms = x[a..b].iter().map(|v| v * v).sum::<f32>() / (b - a).max(1) as f32;
            10.0 * ms.max(1e-15).log10()
        })
        .collect()
}

/// Say `AA <phoneme> AA` (ARPAbet name) with `settings` and measure the
/// fricative's hiss.
pub fn measure(settings: &VoiceSettings, phoneme: &str) -> Hiss {
    let mut talker = Talker::new(SR, *settings);
    let utterance = talker.compile(&format!("[AA1 {phoneme} AA]"));
    // Where the fricative is: its segments (a /ð/ may start with a closure).
    let mut t = 0.0;
    let (mut start, mut end) = (f32::MAX, 0.0f32);
    for s in utterance.segments() {
        if s.phoneme.to_string() == phoneme {
            start = start.min(t);
            end = t + s.duration;
        }
        t += s.duration;
    }
    let audio = talker.bake_utterance(utterance);
    let mid = (start + end) / 2.0;
    let half = (0.2 * (end - start)).max(0.02);
    let core = &audio[((mid - half) * SR) as usize..((mid + half) * SR) as usize];
    let fine: Vec<f32> = (0..80).map(|k| power(core, 50.0 + 100.0 * k as f32)).collect();
    let mut bands: Vec<f32> = (0..BANDS)
        .map(|b| fine[5 * b..5 * b + 5].iter().sum::<f32>())
        .collect();
    let top = bands.iter().fold(0.0f32, |m, &v| m.max(v));
    for v in &mut bands {
        *v = 10.0 * (*v / top).max(1e-12).log10();
    }
    let hi = &fine[10..80];
    let freq = |k: usize| 50.0 + 100.0 * (k + 10) as f32;
    let total: f32 = hi.iter().sum();
    let centroid = hi.iter().enumerate().map(|(k, &p)| freq(k) * p).sum::<f32>() / total;
    let log_mean = hi.iter().map(|&p| (p + 1e-20).ln()).sum::<f32>() / hi.len() as f32;
    let flatness = log_mean.exp() / (total / hi.len() as f32);
    let smooth: Vec<f32> = (0..hi.len())
        .map(|k| {
            let (a, b) = (k.saturating_sub(1), (k + 2).min(hi.len()));
            10.0 * (hi[a..b].iter().sum::<f32>() / (b - a) as f32 + 1e-20).log10()
        })
        .collect();
    let peak = (0..smooth.len())
        .max_by(|&a, &b| smooth[a].total_cmp(&smooth[b]))
        .unwrap_or(0);
    let (mut a, mut b) = (peak, peak);
    while a > 0 && smooth[a - 1] >= smooth[peak] - 6.0 {
        a -= 1;
    }
    while b + 1 < smooth.len() && smooth[b + 1] >= smooth[peak] - 6.0 {
        b += 1;
    }
    let width6 = 100.0 * (b - a + 1) as f32;

    // The hiss's envelope, around the fricative.
    let env = envelope(&high_pass(&audio, hiss_from(phoneme)));
    let frame = |t: f32| ((t * SR) as usize / HOP).min(env.len().saturating_sub(1));
    let (w0, w1) = (frame(start - 0.05), frame(end + 0.05));
    let k = (w0..=w1)
        .max_by(|&a, &b| env[a].total_cmp(&env[b]))
        .unwrap_or(w0);
    let pk = env[k];
    let hop_ms = HOP as f32 / SR * 1000.0;
    let walk = |step: isize| {
        let (mut t_hi, mut t_lo) = (None, None);
        let mut j = k as isize;
        while j >= w0 as isize && j <= w1 as isize {
            let v = env[j as usize];
            if t_hi.is_none() && v < pk - 3.0 {
                t_hi = Some(j);
            }
            if v < pk - 15.0 {
                t_lo = Some(j);
                break;
            }
            j += step;
        }
        match (t_hi, t_lo) {
            (Some(h), Some(l)) => (h - l).unsigned_abs() as f32 * hop_ms,
            _ => f32::NAN,
        }
    };
    let (mut a, mut b) = (k, k);
    while a > w0 && env[a - 1] >= pk - 10.0 {
        a -= 1;
    }
    while b < w1 && env[b + 1] >= pk - 10.0 {
        b += 1;
    }
    let loudest = envelope(&audio).iter().fold(f32::MIN, |m, &v| m.max(v));
    let full = envelope(&audio);
    Hiss {
        bands,
        centroid,
        flatness,
        width6,
        rise_ms: walk(-1),
        fall_ms: walk(1),
        dur_ms: (b - a + 1) as f32 * hop_ms,
        level: full[k] - loudest,
    }
}
