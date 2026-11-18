//! Measurements of the vocal effort states (hybrid roadmap, stage 3),
//! shared by the `effort` example and `tests/effort_gate.rs`: a sentence
//! recorded sample by sample with what the lungs and the folds do, and its
//! level, pitch, spectral balance, H1-H2 and tremor.

#![allow(
    clippy::many_single_char_names,
    dead_code,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use synthie_trombone::{Effort, Engine, Preset, Talker, VoiceSettings};

/// Output rate of the measurements, Hz.
pub const SR: f32 = 48_000.0;

/// A sentence, one entry per output sample.
#[derive(Default)]
pub struct Rec {
    pub out: Vec<f32>,
    /// Glottal flow, cm³/s.
    pub flow: Vec<f32>,
    /// The folds' measured pitch, Hz (0 while they do not vibrate).
    pub pitch: Vec<f32>,
    /// The lungs' pressure, cmH2O.
    pub lung: Vec<f32>,
    /// The vocal effort state the voice is in or gliding into.
    pub effort: Vec<Effort>,
    /// The jaw's opening, `0‥1`.
    pub jaw: Vec<f32>,
    /// The glottis's area, cm².
    pub area: Vec<f32>,
}

/// The deep voice on the physical engine in the state `effort`.
pub fn settings(effort: Effort) -> VoiceSettings {
    Preset::Deep
        .settings()
        .with_engine(Engine::Physical)
        .with_effort(effort)
}

/// `text` spoken by a voice of `settings`, recorded; `each` is called
/// before every block of 64 samples with the talker and the seconds so far.
pub fn speak(settings: &VoiceSettings, text: &str, mut each: impl FnMut(&mut Talker, f32)) -> Rec {
    let mut talker = Talker::new(SR, *settings);
    let u = talker.compile(text);
    let frames = ((u.duration() + 0.25) * SR) as usize;
    talker.say_utterance(u);
    let mut rec = Rec::default();
    for k in 0..frames {
        if k % 64 == 0 {
            each(&mut talker, k as f32 / SR);
        }
        let x = talker.process();
        let v = talker.voice_mut();
        let g = v.gauges();
        rec.out.push(x);
        rec.flow.push(g.glottal_flow);
        rec.pitch.push(g.fold_pitch);
        rec.lung.push(g.lung_pressure);
        rec.effort.push(v.effort());
        rec.jaw.push(v.articulators().jaw_open);
        rec.area.push(g.glottal_area);
    }
    rec
}

pub fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

pub fn db(x: f32) -> f32 {
    20.0 * x.max(1e-12).log10()
}

/// Level of the loud part of `x`, dB: the frames of 20 ms within 25 dB of
/// the loudest.
pub fn level(x: &[f32]) -> f32 {
    let n = (0.02 * SR) as usize;
    let frames: Vec<f32> = x.chunks(n).map(rms).collect();
    let top = frames.iter().fold(0.0f32, |m, &v| m.max(v));
    let loud: Vec<f32> = frames
        .into_iter()
        .filter(|&v| db(v) > db(top) - 25.0)
        .collect();
    db((loud.iter().map(|v| v * v).sum::<f32>() / loud.len().max(1) as f32).sqrt())
}

/// Median of the values of `x` above zero (0 if there are none).
pub fn median(x: &[f32]) -> f32 {
    let mut v: Vec<f32> = x.iter().copied().filter(|&p| p > 0.0).collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

/// Share of the samples at which the folds vibrate (the pitch gauge reads).
pub fn voiced_share(pitch: &[f32]) -> f32 {
    pitch.iter().filter(|&&p| p > 0.0).count() as f32 / pitch.len().max(1) as f32
}

/// Energy of `x` in bands, dB, from frames of 1024 samples within 25 dB of
/// the loudest (Hann window, a plain Fourier sum): 0 to 1 kHz, 1 to
/// 2.5 kHz, 2.5 to 5 kHz.
pub fn bands(x: &[f32]) -> [f32; 3] {
    const N: usize = 1024;
    let top = x.chunks(N).map(rms).fold(0.0f32, f32::max);
    let edges = [0.0f32, 1000.0, 2500.0, 5000.0];
    let mut sum = [0.0f64; 3];
    let bins = (5000.0 / (SR / N as f32)) as usize;
    let table: Vec<(f32, f32)> = (0..N)
        .map(|n| {
            let a = std::f32::consts::TAU * n as f32 / N as f32;
            (a.cos(), a.sin())
        })
        .collect();
    for c in x
        .chunks(N)
        .filter(|c| c.len() == N && db(rms(c)) > db(top) - 25.0)
    {
        let w: Vec<f32> = c
            .iter()
            .enumerate()
            .map(|(n, &v)| v * (0.5 - 0.5 * table[n].0))
            .collect();
        for k in 1..=bins {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for (n, &v) in w.iter().enumerate() {
                let (c, s) = table[(k * n) % N];
                re += v * c;
                im += v * s;
            }
            let hz = k as f32 * SR / N as f32;
            let band = edges[1..].iter().position(|&e| hz < e).unwrap_or(2);
            sum[band] += f64::from(re * re + im * im);
        }
    }
    sum.map(|s| 10.0 * (s.max(1e-30)).log10() as f32)
}

/// Magnitude of `x` (Hann window) at `hz`.
fn tone(x: &[f32], hz: f32) -> f32 {
    let n = x.len();
    let w = std::f64::consts::TAU * f64::from(hz) / f64::from(SR);
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (k, &v) in x.iter().enumerate() {
        let h = 0.5 - 0.5 * (std::f64::consts::TAU * k as f64 / n as f64).cos();
        re += f64::from(v) * h * (w * k as f64).cos();
        im += f64::from(v) * h * (w * k as f64).sin();
    }
    (re * re + im * im).sqrt() as f32
}

/// First harmonic over the second of the glottal flow, dB: the median over
/// stretches of 40 ms in which the folds vibrate at a steady pitch.
pub fn h1_h2(flow: &[f32], pitch: &[f32]) -> f32 {
    let n = (0.04 * SR) as usize;
    let mut all = Vec::new();
    for (f, p) in flow.chunks(n).zip(pitch.chunks(n)) {
        let hz = median(p);
        if f.len() < n || hz <= 0.0 || p.iter().any(|&v| v <= 0.0 || (v / hz - 1.0).abs() > 0.06) {
            continue;
        }
        let peak = |c: f32| {
            (-6..=6)
                .map(|k| tone(f, c * (1.0 + 0.01 * k as f32)))
                .fold(0.0, f32::max)
        };
        all.push(db(peak(hz)) - db(peak(2.0 * hz)));
    }
    if all.is_empty() {
        return f32::NAN;
    }
    all.sort_by(f32::total_cmp);
    all[all.len() / 2]
}

/// The tremor of a track (the pitch, the lungs' pressure) between 6 and
/// 8.5 Hz, above the rate at which a sentence's melody and its syllables
/// move it: its strongest rate there, Hz, and its depth as a share of the
/// track's mean (over the samples above zero, taken every 5 ms).
pub fn tremor(track: &[f32]) -> (f32, f32) {
    let step = (0.005 * SR) as usize;
    let v: Vec<f32> = track
        .iter()
        .step_by(step)
        .copied()
        .filter(|&p| p > 0.0)
        .collect();
    if v.len() < 100 {
        return (0.0, 0.0);
    }
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let mut best = (0.0f32, 0.0f32);
    let mut hz = 6.0f32;
    while hz <= 8.5 {
        let (mut re, mut im) = (0.0f32, 0.0f32);
        for (k, &x) in v.iter().enumerate() {
            let a = std::f32::consts::TAU * hz * k as f32 * 0.005;
            re += (x - mean) * a.cos();
            im += (x - mean) * a.sin();
        }
        let depth = 2.0 * (re * re + im * im).sqrt() / v.len() as f32 / mean;
        if depth > best.1 {
            best = (hz, depth);
        }
        hz += 0.25;
    }
    best
}

/// What a state sounds like in numbers.
#[derive(Clone, Copy, Debug)]
pub struct Signature {
    /// Level of the loud part, dB.
    pub level: f32,
    /// Median pitch of the folds, Hz (0: they do not vibrate).
    pub pitch: f32,
    /// Share of the time the folds vibrate.
    pub voiced: f32,
    /// 2.5 to 5 kHz against 0 to 1 kHz, dB.
    pub tilt: f32,
    /// 1 to 2.5 kHz against 0 to 1 kHz, dB.
    pub mid: f32,
    /// First harmonic over the second of the glottal flow, dB.
    pub h1_h2: f32,
    /// Median of the lungs' pressure while it pushes, cmH2O.
    pub lung: f32,
    /// The pitch's tremor: its rate, Hz, and its depth as a share.
    pub tremor: (f32, f32),
    /// Seconds the sentence takes.
    pub seconds: f32,
    /// The widest the jaw opens.
    pub jaw: f32,
}

pub fn signature(rec: &Rec) -> Signature {
    let b = bands(&rec.out);
    Signature {
        level: level(&rec.out),
        pitch: median(&rec.pitch),
        voiced: voiced_share(&rec.pitch),
        tilt: b[2] - b[0],
        mid: b[1] - b[0],
        h1_h2: h1_h2(&rec.flow, &rec.pitch),
        lung: median(&rec.lung),
        tremor: tremor(&rec.pitch),
        seconds: rec.out.len() as f32 / SR,
        jaw: rec.jaw.iter().fold(0.0f32, |m, &v| m.max(v)),
    }
}
