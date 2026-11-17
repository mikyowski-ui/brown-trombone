//! Measurements of the non-verbal vocalizations (wave 3, stage 3B), shared
//! by the `vocal` example and `tests/vocal_gate.rs`: a voice recorded sample
//! by sample (its sound, the lungs, the flows and the pitch of its folds),
//! and what a laugh's calls, a cough's burst and a sigh's breath are in it.

#![allow(
    clippy::many_single_char_names,
    dead_code,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use synthie_trombone::speech::Speaker;
use synthie_trombone::{Engine, Phoneme, Preset, Trombone, VoiceSettings};

/// Output rate of the measurements, Hz.
pub const SR: f32 = 48_000.0;

/// Everything recorded, one entry per sample.
#[derive(Default)]
pub struct Rec {
    pub out: Vec<f32>,
    /// Lungs and below the folds, cmH2O.
    pub lung: Vec<f32>,
    pub sub: Vec<f32>,
    /// Glottal flow and the nostrils' flow, cm³/s.
    pub flow: Vec<f32>,
    pub nose: Vec<f32>,
    /// Flow out of the lips and through the walls, cm³/s.
    pub lips: Vec<f32>,
    pub skin: Vec<f32>,
    /// Glottal area, and the part of it between the vibrating edges, cm².
    pub area: Vec<f32>,
    pub edge: Vec<f32>,
    /// The folds' measured pitch, Hz (0 while they do not vibrate).
    pub pitch: Vec<f32>,
    /// Lumen of the ventricular valve's cell (cell 1), cm².
    pub valve: Vec<f32>,
    pub finite: bool,
}

impl Rec {
    pub fn push(&mut self, out: f32, voice: &Trombone) {
        let g = voice.gauges();
        self.out.push(out);
        self.lung.push(g.lung_pressure);
        self.sub.push(g.subglottal_pressure);
        self.flow.push(g.glottal_flow);
        self.nose.push(g.nose_flow);
        self.lips.push(g.lip_flow);
        self.skin.push(g.skin_flow);
        self.area.push(g.glottal_area);
        self.edge.push(g.edge_area);
        self.pitch.push(g.fold_pitch);
        self.valve.push(g.valve_area);
        self.finite &= out.is_finite()
            && g.lung_pressure.is_finite()
            && g.subglottal_pressure.is_finite()
            && g.glottal_flow.is_finite();
    }

    pub fn len(&self) -> usize {
        self.out.len()
    }
}

/// A voice of `settings` on the physical engine, ready for speech and
/// silent, its mouth at rest.
pub fn voice(settings: &VoiceSettings) -> Trombone {
    let mut v = Trombone::new(SR);
    settings.with_engine(Engine::Physical).apply(&mut v);
    Speaker::prepare(&mut v);
    v.articulate_phoneme(Phoneme::Sil);
    v.settle();
    v
}

/// The deep voice's settings.
pub fn deep() -> VoiceSettings {
    Preset::Deep.settings()
}

/// Render `seconds` of `voice`, recording it.
pub fn record(voice: &mut Trombone, seconds: f32) -> Rec {
    let mut r = Rec {
        finite: true,
        ..Rec::default()
    };
    for _ in 0..(seconds * SR) as usize {
        let out = voice.process();
        r.push(out, voice);
    }
    r
}

pub fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

pub fn db(x: f32) -> f32 {
    20.0 * x.max(1e-12).log10()
}

/// RMS of `x` in frames of `ms` milliseconds every `hop` ms.
pub fn envelope(x: &[f32], ms: f32, hop: f32) -> Vec<f32> {
    let n = (ms * 1e-3 * SR) as usize;
    let h = (hop * 1e-3 * SR) as usize;
    (0..x.len().saturating_sub(n) / h)
        .map(|k| rms(&x[k * h..k * h + n]))
        .collect()
}

/// Whether the folds vibrate in each 10 ms frame of a recording: the area
/// between their edges swings by more than 0.01 cm² (0.005 in the frame
/// and under a third of its mean: breath through folds held apart moves
/// them less).
pub fn vibrating(r: &Rec) -> Vec<bool> {
    let n = (0.01 * SR) as usize;
    r.edge
        .chunks(n)
        .map(|c| {
            let lo = c.iter().copied().fold(f32::MAX, f32::min);
            let hi = c.iter().copied().fold(f32::MIN, f32::max);
            hi - lo > 0.01 && lo < 0.5 * hi
        })
        .collect()
}

/// The runs of voice in a recording: (first, last) sample of each stretch
/// where the folds vibrate (see [`vibrating`]), at least `ms` long, joined
/// across gaps of one frame.
pub fn voiced_runs(r: &Rec, ms: f32) -> Vec<(usize, usize)> {
    let n = (0.01 * SR) as usize;
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (f, &on) in vibrating(r).iter().enumerate() {
        if !on {
            continue;
        }
        let k = f * n;
        let gap = n + 1;
        let end = (k + n - 1).min(r.len() - 1);
        match runs.last_mut() {
            Some(last) if k - last.1 <= gap => last.1 = end,
            _ => runs.push((k, end)),
        }
    }
    runs.retain(|&(a, b)| (b - a) as f32 >= ms * 1e-3 * SR);
    runs
}

/// Median of the nonzero values of `x`.
pub fn median(x: &[f32]) -> f32 {
    let mut v: Vec<f32> = x.iter().copied().filter(|&p| p > 0.0).collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

/// A laugh's calls: (start s, length s, median pitch Hz, RMS dB) of each
/// voiced run.
pub fn calls(r: &Rec) -> Vec<(f32, f32, f32, f32)> {
    voiced_runs(r, 30.0)
        .into_iter()
        .map(|(a, b)| {
            (
                a as f32 / SR,
                (b - a) as f32 / SR,
                median(&r.pitch[a..b]),
                db(rms(&r.out[a..b])),
            )
        })
        .collect()
}

/// The share of noise in `x` (a stretch of voice at about `hz`): one less
/// the normalised autocorrelation at the best lag within a tenth of the
/// period, over 40 ms frames, averaged.
pub fn noise_share(x: &[f32], hz: f32) -> f32 {
    let n = (0.04 * SR) as usize;
    let period = SR / hz.max(30.0);
    let (lo, hi) = ((0.85 * period) as usize, (1.15 * period) as usize + 1);
    let mut shares = Vec::new();
    let mut k = 0;
    while k + n + hi < x.len() {
        let a = &x[k..k + n];
        let mut best = 0.0f32;
        for lag in lo..hi {
            let b = &x[k + lag..k + lag + n];
            let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
            for (p, q) in a.iter().zip(b) {
                ab += f64::from(*p) * f64::from(*q);
                aa += f64::from(*p) * f64::from(*p);
                bb += f64::from(*q) * f64::from(*q);
            }
            let r = (ab / (aa * bb).sqrt().max(1e-30)) as f32;
            best = best.max(r);
        }
        shares.push(1.0 - best.max(0.0));
        k += n;
    }
    if shares.is_empty() {
        return 1.0;
    }
    shares.iter().sum::<f32>() / shares.len() as f32
}

/// The largest step from one sample to the next in each 10 ms frame of
/// `x` against the frame's RMS, over the frames within `floor` dB of the
/// loudest frame: a click stands out as a step many times the sound
/// around it. (The voice's own closures make steps of a few times its RMS.)
pub fn click_ratio(x: &[f32], floor: f32) -> f32 {
    let n = (0.01 * SR) as usize;
    let loudest = x.chunks(n).map(rms).fold(0.0, f32::max);
    x.chunks(n)
        .filter(|c| c.len() > 1 && db(rms(c)) > db(loudest) - floor)
        .map(|c| largest_step(c) / rms(c).max(1e-9))
        .fold(0.0, f32::max)
}

/// Largest sample-to-sample step of `x`.
pub fn largest_step(x: &[f32]) -> f32 {
    x.windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0, f32::max)
}
