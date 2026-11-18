//! Measurements of a voice interrupted or knocked in the middle of a word
//! (hybrid roadmap, stage 4), shared by the `interrupt` example and
//! `tests/interrupt_gate.rs`: the same sentence recorded sample by sample
//! with and without something happening at one sample, and what the
//! difference between the two shows (how soon the voice answers, whether
//! it clicks).

#![allow(
    clippy::many_single_char_names,
    dead_code,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use synthie_trombone::{Engine, Manner, Phoneme, Preset, Talker, VoiceSettings};

/// Output rate of the measurements, Hz.
pub const SR: f32 = 48_000.0;

/// A voice speaking, one entry per output sample.
#[derive(Default)]
pub struct Run {
    pub out: Vec<f32>,
    /// The glottis's area, cm².
    pub area: Vec<f32>,
    /// The lungs' pressure, cmH2O.
    pub lung: Vec<f32>,
    /// Glottal flow, cm³/s.
    pub flow: Vec<f32>,
    /// The lips' closing and the jaw's opening, `0‥1`.
    pub lips: Vec<f32>,
    pub jaw: Vec<f32>,
    /// Whether a vocalization is being made, and whether speech is going on.
    pub vocalizing: Vec<bool>,
    pub speaking: Vec<bool>,
}

/// The deep voice on the physical engine.
pub fn deep() -> VoiceSettings {
    Preset::Deep.settings().with_engine(Engine::Physical)
}

/// `text` spoken for `seconds` (or until it and what follows it are over),
/// with `act` called once before sample `at`.
pub fn run(
    settings: &VoiceSettings,
    text: &str,
    seconds: f32,
    at: Option<usize>,
    mut act: impl FnMut(&mut Talker),
) -> Run {
    let mut talker = Talker::new(SR, *settings);
    talker.say(text);
    let mut r = Run::default();
    for k in 0..(seconds * SR) as usize {
        if Some(k) == at {
            act(&mut talker);
        }
        let x = talker.process();
        let speaking = talker.is_speaking();
        let v = talker.voice_mut();
        let g = v.gauges();
        let a = v.articulators();
        r.out.push(x);
        r.area.push(g.glottal_area);
        r.lung.push(g.lung_pressure);
        r.flow.push(g.glottal_flow);
        r.lips.push(a.lip_close);
        r.jaw.push(a.jaw_open);
        r.vocalizing.push(v.is_vocalizing());
        r.speaking.push(speaking);
    }
    r
}

/// Where to break in: the sample in the middle of the first long vowel, of
/// the first stop's closure, of the first /s/ and of the first nasal of
/// `text`, each with the phone's name.
pub fn points(settings: &VoiceSettings, text: &str) -> Vec<(String, usize)> {
    let talker = Talker::new(SR, *settings);
    let u = talker.compile(text);
    let spans = u.phone_spans();
    let mut phones = vec![None; spans.len()];
    for (i, s) in u.segments().iter().enumerate() {
        if let Some(p) = u.phone_of(i)
            && phones[p].is_none()
        {
            phones[p] = Some(s.phoneme);
        }
    }
    let mut out = Vec::new();
    let mut find = |name: &str, at: f32, which: &dyn Fn(Phoneme, f32) -> bool| {
        for (k, &(a, b)) in spans.iter().enumerate() {
            if let Some(p) = phones[k]
                && which(p, b - a)
            {
                out.push((format!("{name} /{p}/"), ((a + at * (b - a)) * SR) as usize));
                return;
            }
        }
    };
    find("vowel", 0.5, &|p, d| {
        p.manner() == Manner::Vowel && d > 0.08
    });
    find("closure", 0.35, &|p, _| {
        matches!(p, Phoneme::P | Phoneme::K | Phoneme::B)
    });
    find("hiss", 0.5, &|p, _| p == Phoneme::S);
    find("nasal", 0.5, &|p, _| p.manner() == Manner::Nasal);
    out
}

pub fn peak(x: &[f32]) -> f32 {
    x.iter().fold(0.0f32, |m, &v| m.max(v.abs()))
}

pub fn db(x: f32) -> f32 {
    20.0 * x.max(1e-12).log10()
}

/// Largest sample-to-sample step of `x`.
pub fn step(x: &[f32]) -> f32 {
    x.windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0, f32::max)
}

/// The two runs' difference from sample `at` on.
pub fn difference(a: &Run, b: &Run, at: usize) -> Vec<f32> {
    a.out[at..]
        .iter()
        .zip(&b.out[at..])
        .map(|(x, y)| x - y)
        .collect()
}

/// Milliseconds from sample `at` until two tracks first differ by more
/// than `by` (`None`: they never do).
pub fn departs(a: &[f32], b: &[f32], at: usize, by: f32) -> Option<f32> {
    a[at..]
        .iter()
        .zip(&b[at..])
        .position(|(x, y)| (x - y).abs() > by)
        .map(|k| 1000.0 * k as f32 / SR)
}

/// What breaking in at one sample did, against the same run left alone.
#[derive(Clone, Copy, Debug)]
pub struct Cut {
    /// Milliseconds until the larynx or the lungs do something else than
    /// they would have (the glottis's area by 0.002 cm², the lungs'
    /// pressure by 0.05 cmH2O).
    pub body_ms: f32,
    /// Milliseconds until the sound differs by more than 60 dB under the
    /// speech's peak.
    pub sound_ms: f32,
    /// The largest sample-to-sample step of the difference between the two
    /// runs over the first 3 ms, dB against the speech's peak: a click
    /// made by the breaking-in itself would be a step here.
    pub step_db: f32,
    /// The largest step of the interrupted sound over the 60 ms after the
    /// cut, against the largest step anywhere in the speech left alone.
    pub after_step: f32,
    /// Whether speech has stopped at the cut's sample and a vocalization
    /// is being made.
    pub stopped: bool,
    pub vocalizing: bool,
}

pub fn cut(plain: &Run, with: &Run, at: usize) -> Cut {
    let top = peak(&plain.out);
    let d = difference(with, plain, at);
    let ms3 = (0.003 * SR) as usize;
    let ms60 = (0.06 * SR) as usize;
    let body = [
        departs(&with.area, &plain.area, at, 0.002),
        departs(&with.lung, &plain.lung, at, 0.05),
    ]
    .into_iter()
    .flatten()
    .fold(f32::MAX, f32::min);
    Cut {
        body_ms: body,
        sound_ms: departs(&with.out, &plain.out, at, 0.001 * top).unwrap_or(f32::MAX),
        step_db: db(step(&d[..ms3.min(d.len())]) / top),
        after_step: step(&with.out[at..(at + ms60).min(with.out.len())]) / step(&plain.out),
        stopped: !with.speaking[at],
        vocalizing: with.vocalizing[at],
    }
}
