//! Measurements of the living voice (wave 3, stage 3A), shared by the
//! `breath` example and the breath gates: what the folds' pulses do (their
//! heights and spacing alternating, the energy at half the pitch), and a
//! text spoken with the lungs, the glottal flow and the radiated sound
//! recorded sample by sample, and the breaths in it.

#![allow(
    dead_code,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use synthie_trombone::speech::{Speaker, Utterance};
use synthie_trombone::{Engine, Manner, Phoneme, Trombone, VoiceSettings};

/// Output rate of the measurements, Hz.
pub const SR: f32 = 48_000.0;

/// The pulses of a stretch of glottal flow: (sample of each peak, its
/// height above the floor). A pulse is a run above a quarter of the swing.
pub fn pulses(flows: &[f32]) -> Vec<(usize, f32)> {
    let peak = flows.iter().fold(0.0f32, |m, &v| m.max(v));
    let floor = flows.iter().fold(f32::INFINITY, |m, &v| m.min(v));
    let swing = peak - floor;
    let mut out = Vec::new();
    if swing <= 1.0 {
        return out;
    }
    let lo = floor + 0.15 * swing;
    let mut k = 0;
    while k < flows.len() {
        if flows[k] > lo {
            let mut best = (k, flows[k]);
            while k < flows.len() && flows[k] > lo {
                if flows[k] > best.1 {
                    best = (k, flows[k]);
                }
                k += 1;
            }
            out.push((best.0, best.1 - floor));
        }
        k += 1;
    }
    out
}

/// Magnitude of x (Hann window) at hz.
pub fn tone(x: &[f32], hz: f32) -> f32 {
    let n = x.len();
    let w = 2.0 * std::f64::consts::PI * f64::from(hz) / f64::from(SR);
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (k, &v) in x.iter().enumerate() {
        let h = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * k as f64 / n as f64).cos();
        re += f64::from(v) * h * (w * k as f64).cos();
        im += f64::from(v) * h * (w * k as f64).sin();
    }
    (re * re + im * im).sqrt() as f32
}

pub fn peak_near(x: &[f32], hz: f32) -> f32 {
    (-6..=6)
        .map(|c| tone(x, hz * (1.0 + c as f32 * 0.01)))
        .fold(0.0f32, f32::max)
}

/// What a stretch of voice is: (pulse rate Hz, alternation of pulse
/// heights, alternation of pulse spacing, energy at half the pitch against
/// the pitch dB).
pub fn creak(out: &[f32], flows: &[f32]) -> (f32, f32, f32, f32) {
    let p = pulses(flows);
    if p.len() < 6 {
        return (0.0, 0.0, 0.0, -99.0);
    }
    let spacing: Vec<f32> = p.windows(2).map(|w| (w[1].0 - w[0].0) as f32).collect();
    let mean_space = spacing.iter().sum::<f32>() / spacing.len() as f32;
    let rate = SR / mean_space;
    let mean_h = p.iter().map(|q| q.1).sum::<f32>() / p.len() as f32;
    let alt_h =
        p.windows(2).map(|w| (w[1].1 - w[0].1).abs()).sum::<f32>() / (p.len() - 1) as f32 / mean_h;
    let alt_t = spacing.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f32>()
        / (spacing.len() - 1) as f32
        / mean_space;
    let sub = peak_near(out, 0.5 * rate);
    let f0 = peak_near(out, rate);
    (
        rate,
        alt_h,
        alt_t,
        20.0 * (sub.max(1e-12) / f0.max(1e-12)).log10(),
    )
}

/// Everything recorded while speaking a text, one entry per sample.
pub struct Take {
    pub out: Vec<f32>,
    pub lung: Vec<f32>,
    pub flow: Vec<f32>,
    pub area: Vec<f32>,
    pub sub: Vec<f32>,
    pub segment: Vec<usize>,
    pub phonemes: Vec<Phoneme>,
    pub voiced: Vec<bool>,
}

/// Speak `text` with `settings` on the physical engine, recording it.
pub fn take(settings: &VoiceSettings, text: &str) -> Take {
    let settings = settings.with_engine(Engine::Physical);
    let u = Utterance::from_text(text, &settings.prosody());
    let mut voice = Trombone::new(SR);
    settings.apply(&mut voice);
    Speaker::prepare(&mut voice);
    let phonemes = u.segments().iter().map(|s| s.phoneme).collect();
    let voiced = u.segments().iter().map(|s| s.voiced).collect();
    let mut speaker = Speaker::new();
    speaker.speak(u);
    let mut t = Take {
        out: Vec::new(),
        lung: Vec::new(),
        flow: Vec::new(),
        area: Vec::new(),
        sub: Vec::new(),
        segment: Vec::new(),
        phonemes,
        voiced,
    };
    let mut after = 0;
    while after < (0.2 * SR) as usize {
        let index = speaker.current().map(|(i, _)| i);
        if index.is_none() {
            after += 1;
        }
        t.out.push(speaker.process(&mut voice));
        t.segment.push(index.unwrap_or(usize::MAX));
        let g = voice.gauges();
        t.lung.push(g.lung_pressure);
        t.flow.push(g.glottal_flow);
        t.area.push(g.glottal_area);
        t.sub.push(g.subglottal_pressure);
    }
    t
}

pub fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

pub fn db(x: f32) -> f32 {
    20.0 * x.max(1e-12).log10()
}

/// Samples of segment `k`.
pub fn span(t: &Take, k: usize) -> (usize, usize) {
    let a = t.segment.iter().position(|&s| s == k).unwrap_or(0);
    let b = t.segment.iter().rposition(|&s| s == k).map_or(a, |e| e + 1);
    (a, b)
}

/// The breaths of a take: (start, end) samples where the lungs draw more
/// than half a cmH2O.
pub fn breaths(t: &Take) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut k = 0;
    while k < t.lung.len() {
        if t.lung[k] < -0.5 {
            let a = k;
            while k < t.lung.len() && t.lung[k] < -0.5 {
                k += 1;
            }
            out.push((a, k));
        }
        k += 1;
    }
    out
}

/// The first vowel's samples after sample `at`.
pub fn vowel_after(t: &Take, at: usize) -> Option<(usize, usize)> {
    let seg = t.segment[at..].iter().find(|&&s| {
        s != usize::MAX
            && t.phonemes
                .get(s)
                .is_some_and(|p| matches!(p.manner(), Manner::Vowel | Manner::Diphthong))
    })?;
    Some(span(t, *seg))
}

/// The vowels of the phrase that starts after sample `at`, end to end.
pub fn phrase_vowels(t: &Take, at: usize) -> Vec<f32> {
    let first = t.segment[at..]
        .iter()
        .copied()
        .find(|&s| s != usize::MAX && t.phonemes[s] != Phoneme::Sil)
        .unwrap_or(0);
    let mut out = Vec::new();
    for s in first..t.phonemes.len() {
        if t.phonemes[s] == Phoneme::Sil {
            break;
        }
        if matches!(t.phonemes[s].manner(), Manner::Vowel | Manner::Diphthong) {
            let (x, y) = span(t, s);
            out.extend_from_slice(&t.out[x..y]);
        }
    }
    out
}

/// Largest step between two samples of `x`.
pub fn largest_step(x: &[f32]) -> f32 {
    x.windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0f32, f32::max)
}

/// The last `seconds` of voice before each pause of a take that follows a
/// voiced sound: (start, end) samples and the segment of the last voiced
/// sound.
pub fn phrase_ends(t: &Take, seconds: f32) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    for k in 1..t.phonemes.len() {
        if t.phonemes[k] != Phoneme::Sil || t.phonemes[k - 1] == Phoneme::Sil {
            continue;
        }
        let Some(last) = (0..k).rev().find(|&i| t.voiced[i]) else {
            continue;
        };
        let (_, end) = span(t, last);
        out.push((end.saturating_sub((seconds * SR) as usize), end, last));
    }
    out
}
