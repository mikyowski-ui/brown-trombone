#![allow(
    dead_code,
    clippy::many_single_char_names,
    clippy::manual_midpoint,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Voicing measurements of stops and fricatives spoken by the physical
//! engine, shared by the `voicing` example and the physics gates: closure
//! length, voice bar, burst, voice onset time, mouth pressure at the
//! release, the pitch at the start of the vowel, and how long vowels are
//! before voiceless and voiced codas.

use synthie_trombone::speech::{Speaker, Utterance};
use synthie_trombone::{Engine, Manner, Phoneme, Preset, Trombone, VoiceSettings};

/// Output rate of the measurements, Hz.
pub const SR: f32 = 48_000.0;
/// Glottal area swing, cm², over a 10 ms window that counts as the folds
/// vibrating.
const SWING: f32 = 0.02;
/// Half the window over which the swing is taken, samples (6 ms).
const HALF_WINDOW: usize = 288;
/// Voicing has begun once it lasts this long, samples (20 ms).
const ONSET_HOLD: usize = 960;
/// The voice-bar band, Hz: below 300 Hz, above the slow swings of the
/// radiated flow as the glottis and the tract open and close.
const VOICE_BAR_HZ: f64 = 300.0;
const VOICE_BAR_FLOOR_HZ: f64 = 70.0;
/// Corner of the hiss band, Hz.
const HISS_HZ: f64 = 2500.0;

/// Everything recorded while an utterance was spoken, one entry per sample.
pub struct Recording {
    pub out: Vec<f32>,
    /// Index of the segment being spoken.
    pub segment: Vec<usize>,
    pub phonemes: Vec<Phoneme>,
    pub durations: Vec<f32>,
    pub glottal_area: Vec<f32>,
    /// The oral tract is shut somewhere in front of the larynx.
    pub sealed: Vec<bool>,
    /// Pressure, cmH2O, in the cell behind each place of closure (lips,
    /// tongue tip, tongue back) and below the glottis.
    pub behind: Vec<[f32; 3]>,
    pub subglottal: Vec<f32>,
    /// Pitch the folds are measured at (0 while not vibrating), Hz.
    pub fold_pitch: Vec<f32>,
}

/// Speak `text` with `settings` on the physical engine and record it.
#[must_use]
pub fn record(settings: &VoiceSettings, text: &str) -> Recording {
    let settings = settings.with_engine(Engine::Physical);
    let mut voice = Trombone::new(SR);
    settings.apply(&mut voice);
    Speaker::prepare(&mut voice);
    let utterance = Utterance::from_text(text, &settings.prosody());
    let phonemes = utterance.segments().iter().map(|s| s.phoneme).collect();
    let durations = utterance.segments().iter().map(|s| s.duration).collect();
    let mut speaker = Speaker::new();
    speaker.speak(utterance);
    let mut r = Recording {
        out: Vec::new(),
        segment: Vec::new(),
        phonemes,
        durations,
        glottal_area: Vec::new(),
        sealed: Vec::new(),
        behind: Vec::new(),
        subglottal: Vec::new(),
        fold_pitch: Vec::new(),
    };
    let tail = (0.15 * SR) as usize;
    let mut after = 0;
    while after < tail {
        let index = speaker.current().map(|(i, _)| i);
        if index.is_none() {
            after += 1;
        }
        r.out.push(speaker.process(&mut voice));
        r.segment.push(index.unwrap_or(usize::MAX));
        let g = voice.gauges();
        r.glottal_area.push(g.glottal_area);
        r.sealed
            .push(voice.diameters()[10..].iter().any(|&d| d <= 0.0));
        r.behind
            .push([voice.pressure(38), voice.pressure(31), voice.pressure(16)]);
        r.subglottal.push(g.subglottal_pressure);
        r.fold_pitch.push(g.fold_pitch);
    }
    r
}

/// Samples of segment `k`.
fn span(r: &Recording, k: usize) -> (usize, usize) {
    let start = r.segment.iter().position(|&s| s == k).unwrap_or(0);
    let end = r
        .segment
        .iter()
        .rposition(|&s| s == k)
        .map_or(start, |e| e + 1);
    (start, end)
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

/// Level in dB.
#[must_use]
pub fn db(x: f32) -> f32 {
    20.0 * x.max(1e-9).log10()
}

/// Second-order Butterworth low-pass (`high == false`) or high-pass.
fn filter(x: &[f32], hz: f64, high: bool) -> Vec<f32> {
    let w = 2.0 * std::f64::consts::PI * hz / f64::from(SR);
    let alpha = w.sin() / (2.0 * std::f64::consts::FRAC_1_SQRT_2);
    let a0 = 1.0 + alpha;
    let (b0, b1) = if high {
        ((1.0 + w.cos()) / 2.0 / a0, -(1.0 + w.cos()) / a0)
    } else {
        ((1.0 - w.cos()) / 2.0 / a0, (1.0 - w.cos()) / a0)
    };
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

/// Whether the folds vibrate around each sample: over the 10 ms around it
/// the glottal area swings by `SWING` and passes through its own mean both
/// ways (a glottis that is only opening or closing passes one way).
fn vibrating(r: &Recording) -> Vec<bool> {
    let n = r.glottal_area.len();
    (0..n)
        .map(|k| {
            let w = &r.glottal_area[k.saturating_sub(HALF_WINDOW)..(k + HALF_WINDOW).min(n)];
            let lo = w.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = w.iter().copied().fold(0.0f32, f32::max);
            if hi - lo <= SWING {
                return false;
            }
            let mean = w.iter().sum::<f32>() / w.len() as f32;
            let rises = w.windows(2).any(|p| p[0] <= mean && p[1] > mean);
            let falls = w.windows(2).any(|p| p[0] > mean && p[1] <= mean);
            rises && falls
        })
        .collect()
}

/// The stressed (or else the first) vowel segment of the recording.
fn main_vowel(r: &Recording, after: usize) -> usize {
    (after..r.phonemes.len())
        .find(|&k| matches!(r.phonemes[k].manner(), Manner::Vowel | Manner::Diphthong))
        .or_else(|| {
            (0..r.phonemes.len())
                .rev()
                .find(|&k| matches!(r.phonemes[k].manner(), Manner::Vowel | Manner::Diphthong))
        })
        .unwrap_or(0)
}

/// What one stop did.
#[derive(Clone, Copy, Debug, Default)]
pub struct StopMeasure {
    /// How long the tract was shut, ms.
    pub closure: f32,
    /// Share of the closure in which the folds vibrated.
    pub voiced_share: f32,
    /// Energy between 70 and 300 Hz during the closure, dB re the vowel's.
    pub voice_bar: f32,
    /// Level of the first 5 ms after the release, dB re the vowel.
    pub burst: f32,
    /// Voice onset time, ms after the release (0 when the voice carries
    /// through it); `NaN` if the voice never came.
    pub vot: f32,
    /// Mouth pressure just before the release over the pressure below the
    /// glottis.
    pub pressure: f32,
    /// Pitch of the first 25 ms of voicing in the vowel, and of the 25 ms
    /// after, minus that of the vowel's steady middle, Hz.
    pub f0_onset: f32,
    pub f0_next: f32,
    /// Steady pitch of the vowel, Hz.
    pub f0_steady: f32,
}

fn place(p: Phoneme) -> usize {
    match p {
        Phoneme::P | Phoneme::B | Phoneme::M => 0,
        Phoneme::K | Phoneme::G | Phoneme::Ng => 2,
        _ => 1,
    }
}

/// Mean pitch, Hz, of the glottal cycles that start in `from..to`: the
/// time between successive rises of the glottal area through the middle of
/// its swing over the vowel `vowel`. `NaN` without two cycles.
fn cycle_pitch(r: &Recording, from: usize, to: usize, vowel: (usize, usize)) -> f32 {
    let w = &r.glottal_area[vowel.0..vowel.1];
    let lo = w.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = w.iter().copied().fold(0.0f32, f32::max);
    let mid = 0.5 * (lo + hi);
    let to = to.min(r.glottal_area.len() - 1);
    let rises: Vec<usize> = (from.max(1)..to)
        .filter(|&k| r.glottal_area[k - 1] <= mid && r.glottal_area[k] > mid)
        .collect();
    if rises.len() < 2 {
        return f32::NAN;
    }
    let periods: Vec<f32> = rises
        .windows(2)
        .map(|p| (p[1] - p[0]) as f32 / SR)
        .filter(|&t| (0.002..0.02).contains(&t))
        .collect();
    if periods.is_empty() {
        return f32::NAN;
    }
    periods.len() as f32 / periods.iter().sum::<f32>()
}

/// Measure the stop `p` (the first segment that says it) in `text`.
#[must_use]
pub fn stop(settings: &VoiceSettings, text: &str, p: Phoneme) -> StopMeasure {
    let r = record(settings, text);
    let Some(k) = r.phonemes.iter().position(|&q| q == p) else {
        return StopMeasure::default();
    };
    let (seg_start, seg_end) = span(&r, k);
    let vibr = vibrating(&r);
    // The closure: the run of shut samples that overlaps the segment.
    let probe = (seg_start + seg_end) / 2;
    let mut lo = probe;
    let mut hi = probe;
    if !r.sealed[probe] {
        // Not shut at the middle: search the segment.
        if let Some(s) = (seg_start..seg_end).find(|&s| r.sealed[s]) {
            lo = s;
            hi = s;
        }
    }
    if r.sealed[lo] {
        while lo > 0 && r.sealed[lo - 1] {
            lo -= 1;
        }
        while hi + 1 < r.sealed.len() && r.sealed[hi + 1] {
            hi += 1;
        }
        hi += 1;
    } else {
        hi = lo;
    }
    let release = hi;
    let vowel_k = main_vowel(&r, k + 1);
    let (vs, ve) = span(&r, vowel_k);
    let mid = (vs + (ve - vs) / 4, ve - (ve - vs) / 4);
    let vowel_rms = rms(&r.out[mid.0..mid.1]);
    let low = filter(
        &filter(&r.out, VOICE_BAR_HZ, false),
        VOICE_BAR_FLOOR_HZ,
        true,
    );
    let bar = db(rms(&low[lo..hi])) - db(rms(&low[mid.0..mid.1]));
    let voiced = vibr[lo..hi].iter().filter(|&&v| v).count() as f32 / (hi - lo).max(1) as f32;
    let burst_end = (release + 240).min(r.out.len());
    let burst = db(rms(&r.out[release..burst_end])) - db(vowel_rms);
    // Onset: the first sample from which the voice lasts 20 ms; zero if the
    // voice was already running at the release.
    let onset = (release.saturating_sub(HALF_WINDOW)..r.out.len().saturating_sub(ONSET_HOLD))
        .find(|&s| vibr[s..s + ONSET_HOLD].iter().all(|&v| v));
    let vot = onset.map_or(f32::NAN, |s| {
        (s.saturating_sub(release) as f32 / SR * 1000.0).max(0.0)
    });
    let before = release.saturating_sub(96);
    let pressure = if release > 0 {
        let pl = place(p);
        let mouth: f32 = r.behind[before..release].iter().map(|b| b[pl]).sum::<f32>()
            / (release - before).max(1) as f32;
        let sub: f32 =
            r.subglottal[before..release].iter().sum::<f32>() / (release - before).max(1) as f32;
        mouth / sub.max(0.1)
    } else {
        f32::NAN
    };
    let steady = cycle_pitch(&r, mid.0, mid.1, (vs, ve));
    let (f0_onset, f0_next) = match onset {
        Some(s) if vs >= release => {
            let s = s.max(release);
            (
                cycle_pitch(&r, s, s + 1200, (vs, ve)) - steady,
                cycle_pitch(&r, s + 1200, s + 2400, (vs, ve)) - steady,
            )
        }
        _ => (f32::NAN, f32::NAN),
    };
    StopMeasure {
        closure: (hi - lo) as f32 / SR * 1000.0,
        voiced_share: voiced,
        voice_bar: bar,
        burst,
        vot,
        pressure,
        f0_onset,
        f0_next,
        f0_steady: steady,
    }
}

/// What one fricative did.
#[derive(Clone, Copy, Debug, Default)]
pub struct FricativeMeasure {
    /// Length of the segment, ms.
    pub length: f32,
    /// Share of the middle of the segment in which the folds vibrated.
    pub voiced_share: f32,
    /// Energy between 70 and 300 Hz over the middle of the segment, dB re
    /// the vowel's.
    pub voice_bar: f32,
    /// Energy above 2.5 kHz over the middle, dB re the whole vowel.
    pub hiss: f32,
}

/// Measure the fricative `p` in `text`: its last segment (for an
/// affricate, the frication after the closure).
#[must_use]
pub fn fricative(settings: &VoiceSettings, text: &str, p: Phoneme) -> FricativeMeasure {
    let r = record(settings, text);
    let Some(k) = r.phonemes.iter().rposition(|&q| q == p) else {
        return FricativeMeasure::default();
    };
    let (s, e) = span(&r, k);
    let mid = (s + (e - s) / 5, e - (e - s) / 5);
    let vowel_k = main_vowel(&r, 0);
    let (vs, ve) = span(&r, vowel_k);
    let vmid = (vs + (ve - vs) / 4, ve - (ve - vs) / 4);
    let vibr = vibrating(&r);
    let low = filter(
        &filter(&r.out, VOICE_BAR_HZ, false),
        VOICE_BAR_FLOOR_HZ,
        true,
    );
    let high = filter(&r.out, HISS_HZ, true);
    FricativeMeasure {
        length: (e - s) as f32 / SR * 1000.0,
        voiced_share: vibr[mid.0..mid.1].iter().filter(|&&v| v).count() as f32
            / (mid.1 - mid.0).max(1) as f32,
        voice_bar: db(rms(&low[mid.0..mid.1])) - db(rms(&low[vmid.0..vmid.1])),
        hiss: db(rms(&high[mid.0..mid.1])) - db(rms(&r.out[vmid.0..vmid.1])),
    }
}

/// Length, s, of the vowel segments (glides included) of `text` as
/// compiled for `settings` on the physical engine.
#[must_use]
pub fn vowel_length(settings: &VoiceSettings, text: &str) -> f32 {
    let settings = settings.with_engine(Engine::Physical);
    Utterance::from_text(text, &settings.prosody())
        .segments()
        .iter()
        .filter(|s| matches!(s.phoneme.manner(), Manner::Vowel | Manner::Diphthong))
        .map(|s| s.duration)
        .sum()
}

/// The voice the judge listens to.
#[must_use]
pub fn judge_voice() -> VoiceSettings {
    Preset::Deep.settings()
}

/// The six stops, voiceless and voiced in pairs.
pub const STOPS: [(Phoneme, Phoneme); 3] = [
    (Phoneme::P, Phoneme::B),
    (Phoneme::T, Phoneme::D),
    (Phoneme::K, Phoneme::G),
];

/// The fricative pairs, and the affricates (measured on their frication).
pub const FRICATIVES: [(Phoneme, Phoneme); 5] = [
    (Phoneme::S, Phoneme::Z),
    (Phoneme::F, Phoneme::V),
    (Phoneme::Th, Phoneme::Dh),
    (Phoneme::Sh, Phoneme::Zh),
    (Phoneme::Ch, Phoneme::Jh),
];

/// Text for a consonant in a position: `initial` before a stressed vowel,
/// `medial` between a schwa and a stressed vowel, `final` after one.
#[must_use]
pub fn context(p: Phoneme, position: &str) -> String {
    let c = p.symbol();
    match position {
        "initial" => format!("[{c} AA1]"),
        "medial" => format!("[AX {c} AA1]"),
        "final" => format!("[AA1 {c}]"),
        _ => format!("[AA1 {c} AA]"),
    }
}

/// Loudest sample of the middle halves of the vowels of a recording.
#[must_use]
pub fn vowel_peak(r: &Recording) -> f32 {
    let mut peak = 0.0f32;
    for k in 0..r.phonemes.len() {
        if matches!(r.phonemes[k].manner(), Manner::Vowel | Manner::Diphthong) {
            let (s, e) = span(r, k);
            for v in &r.out[s + (e - s) / 4..e - (e - s) / 4] {
                peak = peak.max(v.abs());
            }
        }
    }
    peak
}

/// Largest step between adjacent samples of a recording.
#[must_use]
pub fn largest_step(r: &Recording) -> f32 {
    r.out
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0f32, f32::max)
}

/// A stop or affricate release.
#[derive(Clone, Copy, Debug)]
pub struct Burst {
    pub phoneme: Phoneme,
    /// Loudest sample of the 10 ms after the release.
    pub peak: f32,
    /// Loudest sample of the vowel after it (before it, at the end of a
    /// phrase).
    pub vowel: f32,
    /// Largest step between adjacent samples from 5 ms before the release
    /// to 10 ms after it.
    pub step: f32,
}

/// Every stop and affricate release of a recording.
#[must_use]
pub fn bursts(r: &Recording) -> Vec<Burst> {
    let mut out = Vec::new();
    for k in 0..r.phonemes.len() {
        let p = r.phonemes[k];
        let first = k == 0 || r.phonemes[k - 1] != p;
        if !first || !matches!(p.manner(), Manner::Stop | Manner::Affricate) {
            continue;
        }
        let (s, e) = span(r, k);
        // The release: where the closure that overlaps the segment opens.
        let Some(shut) = (s..e).rev().find(|&i| r.sealed[i]) else {
            continue;
        };
        let mut release = shut;
        while release + 1 < r.sealed.len() && r.sealed[release + 1] {
            release += 1;
        }
        release += 1;
        let after = (k + 1..r.phonemes.len())
            .find(|&j| matches!(r.phonemes[j].manner(), Manner::Vowel | Manner::Diphthong))
            .or_else(|| {
                (0..k)
                    .rev()
                    .find(|&j| matches!(r.phonemes[j].manner(), Manner::Vowel | Manner::Diphthong))
            });
        let Some(v) = after else {
            continue;
        };
        let (vs, ve) = span(r, v);
        let vowel = r.out[vs..ve].iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let end = (release + 480).min(r.out.len());
        let burst = r.out[release.min(end)..end]
            .iter()
            .fold(0.0f32, |m, x| m.max(x.abs()));
        let step = r.out[release.saturating_sub(240).min(end)..end]
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        out.push(Burst {
            phoneme: p,
            peak: burst,
            vowel,
            step,
        });
    }
    out
}
