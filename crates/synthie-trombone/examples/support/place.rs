#![allow(
    dead_code,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]
//! Where the physical engine makes its consonants, measured from the tract
//! itself, shared by the `place` example and the physics gates: the second
//! and third formants on the way into a consonant and out of it (the peaks
//! of the tract's transfer function, frame by frame, never LPC on the
//! output), the antiresonance of a nasal murmur, how open the mouth gets
//! between the consonants of a cluster, and how much of each sound the
//! tract spends at its target.

use crate::spectrum;
use synthie_trombone::speech::{Speaker, Utterance};
use synthie_trombone::{Articulation, Engine, Manner, N, Phoneme, Trombone, VoiceSettings};

/// Output rate, Hz.
pub const SR: f32 = 48_000.0;
/// Samples per frame (5 ms).
pub const FRAME: usize = 240;
/// Seconds of impulse response a frame's formants are taken from.
const RESPONSE: f32 = 0.05;
/// Spectrum step for frame formants, Hz.
const STEP: f32 = 8.0;
/// First cell of the oral tract looked at for its narrowest opening (above
/// the larynx).
const ORAL_FROM: usize = 10;
/// Lowest frequency taken as a formant, Hz (below it: the walls' own
/// resonance).
const LOWEST_FORMANT: f32 = 180.0;

/// One 5 ms frame of an utterance.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    /// Segment being spoken (`usize::MAX` after the end).
    pub segment: usize,
    /// F1, F2, F3 of the tract as it is at the frame's start, Hz; `None`
    /// while the mouth is shut or the formants were not asked for.
    pub formants: Option<[f32; 3]>,
    /// Narrowest diameter of the oral tract (cells 10 on) and its cell.
    pub narrowest: f32,
    pub at: usize,
    /// Velum openness, `0..1`.
    pub velum: f32,
    /// RMS distance of the shape from the settled target of the segment
    /// being spoken, diameter units over cells 10 on.
    pub off_target: f32,
}

/// An utterance as it was spoken.
pub struct Spoken {
    pub phonemes: Vec<Phoneme>,
    /// Sample range of each segment.
    pub ranges: Vec<(usize, usize)>,
    pub frames: Vec<Frame>,
    pub audio: Vec<f32>,
}

/// The shape an articulation settles to on `settings`' voice.
fn settled(settings: &VoiceSettings, a: &Articulation) -> [f32; N] {
    let mut t = Trombone::new(SR);
    settings.apply(&mut t);
    t.articulate(a);
    t.settle();
    *t.diameters()
}

/// F1..F3 of the voice's present shape.
fn formants_now(voice: &Trombone) -> [f32; 3] {
    let mag = spectrum::transfer(voice, RESPONSE, 4200.0, STEP);
    let peaks: Vec<f32> = spectrum::resonances(&mag, STEP)
        .iter()
        .map(|r| r.hz)
        .filter(|&hz| hz >= LOWEST_FORMANT)
        .collect();
    [
        peaks.first().copied().unwrap_or(0.0),
        peaks.get(1).copied().unwrap_or(0.0),
        peaks.get(2).copied().unwrap_or(0.0),
    ]
}

/// Speak `text` on the physical engine; with `formants`, measure them in
/// every frame the mouth is open.
#[must_use]
pub fn speak(settings: &VoiceSettings, text: &str, formants: bool) -> Spoken {
    let settings = settings.with_engine(Engine::Physical);
    let mut voice = Trombone::new(SR);
    settings.apply(&mut voice);
    Speaker::prepare(&mut voice);
    let utterance = Utterance::from_text(text, &settings.prosody());
    let segments = utterance.segments().to_vec();
    let targets: Vec<[f32; N]> = segments
        .iter()
        .map(|s| settled(&settings, &s.articulation))
        .collect();
    let mut speaker = Speaker::new();
    speaker.speak(utterance);
    let mut s = Spoken {
        phonemes: segments.iter().map(|s| s.phoneme).collect(),
        ranges: vec![(usize::MAX, 0); segments.len()],
        frames: Vec::new(),
        audio: Vec::new(),
    };
    let tail = (0.12 * SR) as usize;
    let mut after = 0;
    while after < tail {
        let index = speaker.current().map_or(usize::MAX, |(i, _)| i);
        if index == usize::MAX {
            after += 1;
        } else {
            let r = &mut s.ranges[index];
            r.0 = r.0.min(s.audio.len());
            r.1 = s.audio.len() + 1;
        }
        if s.audio.len().is_multiple_of(FRAME) {
            let d = voice.diameters();
            let (at, narrowest) =
                (ORAL_FROM..N)
                    .map(|i| (i, d[i]))
                    .fold(
                        (ORAL_FROM, f32::INFINITY),
                        |m, x| if x.1 < m.1 { x } else { m },
                    );
            let off_target = targets.get(index).map_or(0.0, |t| {
                let n = (N - ORAL_FROM) as f32;
                ((ORAL_FROM..N).map(|i| (d[i] - t[i]).powi(2)).sum::<f32>() / n).sqrt()
            });
            s.frames.push(Frame {
                segment: index,
                formants: (formants && narrowest > 0.05).then(|| formants_now(&voice)),
                narrowest,
                at,
                velum: voice.velum(),
                off_target,
            });
        }
        s.audio.push(speaker.process(&mut voice));
    }
    s
}

impl Spoken {
    /// Frame index of output sample `n`.
    #[must_use]
    pub fn frame_of(n: usize) -> usize {
        n / FRAME
    }

    /// The formants of the frame `ms` milliseconds from sample `n`.
    #[must_use]
    pub fn formants_at(&self, n: usize, ms: f32) -> Option<[f32; 3]> {
        let at = n as f32 + ms * 1e-3 * SR;
        if at < 0.0 {
            return None;
        }
        self.frames.get(Self::frame_of(at as usize))?.formants
    }

    /// Index of the first segment of phoneme `p` from segment `from` on.
    #[must_use]
    pub fn find(&self, p: Phoneme, from: usize) -> Option<usize> {
        (from..self.phonemes.len()).find(|&k| self.phonemes[k] == p)
    }

    /// The consonant segments from `k` on that make one consonant (a
    /// stop's closure and its aspiration share the phoneme).
    #[must_use]
    pub fn consonant_span(&self, k: usize) -> (usize, usize) {
        let p = self.phonemes[k];
        let mut e = k;
        while e + 1 < self.phonemes.len() && self.phonemes[e + 1] == p {
            e += 1;
        }
        (k, e)
    }
}

/// F2 and F3 around one consonant in `[V C V1]`: 60, 30 and 10 ms before
/// the consonant starts, at the onset of the second vowel (10 ms in) and in
/// its middle.
#[derive(Clone, Copy, Debug, Default)]
pub struct Transition {
    pub before: [[f32; 2]; 3],
    pub onset: [f32; 2],
    pub mid: [f32; 2],
}

fn f23(f: Option<[f32; 3]>) -> [f32; 2] {
    f.map_or([0.0; 2], |f| [f[1], f[2]])
}

/// The transitions of consonant `c` between two vowels `v`.
#[must_use]
pub fn transition(settings: &VoiceSettings, c: Phoneme, v: Phoneme) -> Transition {
    let text = format!("[{v} {c} {v}1]");
    let s = speak(settings, &text, true);
    let Some(k) = (1..s.phonemes.len()).find(|&k| s.phonemes[k] == c) else {
        return Transition::default();
    };
    let (start, end) = s.consonant_span(k);
    let c0 = s.ranges[start].0;
    let v = end + 1;
    let (v0, v1) = s.ranges.get(v).copied().unwrap_or((0, 0));
    Transition {
        before: [-60.0, -30.0, -10.0].map(|ms| f23(s.formants_at(c0, ms))),
        onset: f23(s.formants_at(v0, 10.0)),
        mid: f23(s.formants_at(usize::midpoint(v0, v1), 0.0)),
    }
}

/// Locus of a consonant from the locus equation `F2onset = k·F2mid + c`
/// fitted over several vowels: the F2 at which onset and target agree,
/// `c / (1 - k)`. Also returns the slope `k` (1: the onset is the vowel's;
/// 0: the consonant's own).
#[must_use]
pub fn locus(points: &[(f32, f32)]) -> (f32, f32) {
    let n = points.len() as f32;
    let (sx, sy) = points
        .iter()
        .fold((0.0, 0.0), |(a, b), &(x, y)| (a + x, b + y));
    let (mx, my) = (sx / n, sy / n);
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for &(x, y) in points {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx) * (x - mx);
    }
    let k = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    let c = my - k * mx;
    let locus = if (1.0 - k).abs() > 0.05 {
        c / (1.0 - k)
    } else {
        f32::NAN
    };
    (locus, k)
}

/// A notch in a spectrum: frequency, Hz, and depth below the lower of the
/// highest peaks within 700 Hz on either side, dB.
#[derive(Clone, Copy, Debug)]
pub struct Notch {
    pub hz: f32,
    pub depth: f32,
}

/// The notches of the tract's transfer function while `nasal` is held
/// after vowel `v` (its tongue), deepest first. The closed mouth behind the
/// oral closure is a side branch of the nose: its own resonance is the
/// murmur's antiresonance.
#[must_use]
pub fn murmur(settings: &VoiceSettings, nasal: Phoneme, v: Phoneme) -> Vec<Notch> {
    let settings = settings.with_engine(Engine::Physical);
    let mut t = Trombone::new(SR);
    settings.apply(&mut t);
    let mut a = nasal.articulation();
    if a.tongue.is_none() {
        a.tongue = v.articulation().tongue;
    }
    t.articulate(&v.articulation());
    t.settle();
    t.articulate(&a);
    t.settle();
    let step = 5.0;
    let mag = spectrum::transfer(&t, 0.2, 5000.0, step);
    let db: Vec<f32> = mag.iter().map(|m| 20.0 * m.max(1e-20).log10()).collect();
    // Light smoothing over ±15 Hz.
    let smooth: Vec<f32> = (0..db.len())
        .map(|k| {
            let lo = k.saturating_sub(3);
            let hi = (k + 4).min(db.len());
            db[lo..hi].iter().sum::<f32>() / (hi - lo) as f32
        })
        .collect();
    let reach = (700.0 / step) as usize;
    let mut out = Vec::new();
    for k in (300.0 / step) as usize..smooth.len() - 1 {
        if smooth[k] > smooth[k - 1] || smooth[k] > smooth[k + 1] {
            continue;
        }
        let left = smooth[k.saturating_sub(reach)..k]
            .iter()
            .fold(f32::MIN, |m, &v| m.max(v));
        let right = smooth[k + 1..(k + reach).min(smooth.len())]
            .iter()
            .fold(f32::MIN, |m, &v| m.max(v));
        out.push(Notch {
            hz: k as f32 * step,
            depth: left.min(right) - smooth[k],
        });
    }
    out.sort_by(|a, b| b.depth.total_cmp(&a.depth));
    out
}

/// How far the mouth opens between two consonants that follow one another:
/// for each pair of neighbouring consonant segments of different phonemes,
/// the widest the narrowest oral opening gets from the middle of the first
/// to the middle of the second, and how long (ms) it stays wider than
/// `open` meanwhile. A cluster that goes straight from one constriction to
/// the next stays narrow; one that lets the mouth open in between has a
/// vowel in it nobody asked for.
#[must_use]
pub fn cluster_openings(s: &Spoken, open: f32) -> Vec<(Phoneme, Phoneme, f32, f32)> {
    let consonant = |p: Phoneme| {
        matches!(
            p.manner(),
            Manner::Stop
                | Manner::Nasal
                | Manner::Fricative
                | Manner::Affricate
                | Manner::Approximant
        )
    };
    let mut out = Vec::new();
    for k in 0..s.phonemes.len().saturating_sub(1) {
        let (a, b) = (s.phonemes[k], s.phonemes[k + 1]);
        if a == b || !consonant(a) || !consonant(b) {
            continue;
        }
        let (a0, a1) = s.ranges[k];
        let (b0, b1) = s.ranges[k + 1];
        if a0 == usize::MAX || b0 == usize::MAX {
            continue;
        }
        let from = Spoken::frame_of(usize::midpoint(a0, a1));
        let to = Spoken::frame_of(usize::midpoint(b0, b1)).min(s.frames.len());
        let frames = &s.frames[from..to.max(from)];
        let widest = frames.iter().fold(0.0f32, |m, f| m.max(f.narrowest));
        let wide = frames.iter().filter(|f| f.narrowest > open).count();
        out.push((a, b, widest, wide as f32 * FRAME as f32 / SR * 1e3));
    }
    out
}

/// Words with consonant clusters, for the check that nothing is heard
/// between their consonants that was not asked for.
pub const CLUSTER_WORDS: [&str; 10] = [
    "slid", "smooth", "planks", "glue", "blue", "flame", "street", "spring", "breeze", "crossed",
];

/// Vowel-like moments between two consonants that follow one another: for
/// each such pair, the frames from the middle of the first to the middle
/// of the second in which the tract's transfer function has its first
/// resonance above `f1` Hz, as a vowel's (or an /l/'s) is; a consonant's
/// constriction holds F1 low. `s` must have been spoken with formants.
#[must_use]
pub fn cluster_vowel_frames(s: &Spoken, f1: f32) -> Vec<(Phoneme, Phoneme, usize)> {
    let consonant = |p: Phoneme| {
        matches!(
            p.manner(),
            Manner::Stop
                | Manner::Nasal
                | Manner::Fricative
                | Manner::Affricate
                | Manner::Approximant
        )
    };
    let mut out = Vec::new();
    for k in 0..s.phonemes.len().saturating_sub(1) {
        let (a, b) = (s.phonemes[k], s.phonemes[k + 1]);
        if a == b || !consonant(a) || !consonant(b) {
            continue;
        }
        let (a0, a1) = s.ranges[k];
        let (b0, b1) = s.ranges[k + 1];
        if a0 == usize::MAX || b0 == usize::MAX {
            continue;
        }
        let from = Spoken::frame_of(usize::midpoint(a0, a1));
        let to = Spoken::frame_of(usize::midpoint(b0, b1)).min(s.frames.len());
        let n = s.frames[from..to.max(from)]
            .iter()
            .filter(|f| f.formants.is_some_and(|x| x[0] > f1))
            .count();
        out.push((a, b, n));
    }
    out
}

/// Share of the frames of each segment within `near` (RMS diameter) of its
/// target, summed over the utterance per manner: (manner, frames near,
/// frames).
#[must_use]
pub fn at_target(s: &Spoken, near: f32) -> Vec<(Manner, usize, usize)> {
    let mut out: Vec<(Manner, usize, usize)> = Vec::new();
    for f in &s.frames {
        let Some(&p) = s.phonemes.get(f.segment) else {
            continue;
        };
        let m = p.manner();
        if m == Manner::Pause {
            continue;
        }
        let i = out.iter().position(|e| e.0 == m).unwrap_or_else(|| {
            out.push((m, 0, 0));
            out.len() - 1
        });
        out[i].2 += 1;
        if f.off_target <= near {
            out[i].1 += 1;
        }
    }
    out
}

/// Level of `x` in dB.
#[must_use]
pub fn db(x: f32) -> f32 {
    20.0 * x.max(1e-9).log10()
}

/// RMS of `x`.
#[must_use]
pub fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}
