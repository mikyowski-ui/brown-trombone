//! Gates on the sibilants of the physical engine as a listener hears them:
//! at the full output rate. The judge's recognizer listens at 16 kHz and
//! never hears what lies above 8 kHz, where most of an /s/ is; it scored a
//! dull /s/, close to /f/, 97 % right.
//!
//! The human figures come from the same measurement in sentences (forced
//! alignment, the hiss core within 10 dB of its peak, Welch spectra): the
//! LibriSpeech reader the judge uses (16 kHz, 32 /s/) and the listener's
//! own recording (96 kHz). Here the engine's own segment times stand in for
//! the alignment, and the core is the middle half of each segment.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]

use std::sync::OnceLock;

use synthie_trombone::speech::{Speaker, Utterance};
use synthie_trombone::{Engine, Phoneme, Preset, Trombone, VoiceSettings};

const SR: f32 = 48_000.0;
/// Spectrum bins, Hz.
const STEP: f32 = 100.0;
const TOP: f32 = 16_000.0;

/// The tuning under test laid over `settings`: the default one, or, when
/// the environment variable `SYNTHIE_TROMBONE_PROFILE` names a profile
/// file, that profile's (as in `physical_gates`, so the tuner can hold a
/// candidate to these gates too).
fn tuned(settings: VoiceSettings) -> VoiceSettings {
    static PROFILE: OnceLock<Option<String>> = OnceLock::new();
    let profile = PROFILE.get_or_init(|| {
        let path = std::env::var("SYNTHIE_TROMBONE_PROFILE").ok()?;
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        println!("gates run with the profile {path}");
        Some(text)
    });
    let Some(profile) = profile else {
        return settings;
    };
    let laid = settings.with_profile(profile).expect("a voice profile");
    VoiceSettings {
        tuning: laid.tuning,
        ..settings
    }
}

/// A sentence said by the deep voice of the given size on the physical
/// engine: the audio, and each segment's phoneme with its start and end,
/// seconds.
fn say(text: &str, size: f32) -> (Vec<f32>, Vec<(Phoneme, f32, f32)>) {
    let mut settings = tuned(Preset::Deep.settings().with_engine(Engine::Physical));
    settings.tract_length = size;
    let mut voice = Trombone::new(SR);
    settings.apply(&mut voice);
    Speaker::prepare(&mut voice);
    let utterance = Utterance::from_text(text, &settings.prosody());
    let mut t = 0.0;
    let segments = utterance
        .segments()
        .iter()
        .map(|s| {
            t += s.duration;
            (s.phoneme, t - s.duration, t)
        })
        .collect();
    let mut speaker = Speaker::new();
    speaker.speak(utterance);
    let mut audio = Vec::new();
    while speaker.is_speaking() {
        audio.push(speaker.process(&mut voice));
    }
    (audio, segments)
}

/// Power per `STEP` from 0 to `TOP` of `x` under a Hann window (relative
/// units, the same for every call).
fn power(x: &[f32]) -> Vec<f64> {
    let n = x.len();
    let bins = (TOP / STEP) as usize;
    let w: Vec<f64> = (0..n)
        .map(|k| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * k as f64 / n as f64).cos())
        .collect();
    let norm: f64 = w.iter().map(|v| v * v).sum();
    (0..bins)
        .map(|b| {
            let ph = 2.0 * std::f64::consts::PI * f64::from(b as f32 * STEP) / f64::from(SR);
            let (cr, ci) = (ph.cos(), -ph.sin());
            let (mut pr, mut pi) = (1.0f64, 0.0f64);
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (v, wk) in x.iter().zip(&w) {
                let v = f64::from(*v) * wk;
                re += v * pr;
                im += v * pi;
                let t = pr * cr - pi * ci;
                pi = pr * ci + pi * cr;
                pr = t;
            }
            (re * re + im * im) / norm
        })
        .collect()
}

fn bin(hz: f32) -> usize {
    (hz / STEP) as usize
}

/// Centroid of `p` between `lo` and `hi`, Hz.
fn centroid(p: &[f64], lo: f32, hi: f32) -> f32 {
    let (mut num, mut den) = (0.0, 0.0);
    for (b, &v) in p.iter().enumerate().take(bin(hi)).skip(bin(lo)) {
        num += v * f64::from(b as f32 * STEP);
        den += v;
    }
    (num / den.max(1e-30)) as f32
}

/// Power between `lo` and `hi`, dB (relative units).
fn band(p: &[f64], lo: f32, hi: f32) -> f32 {
    let sum: f64 = p[bin(lo)..bin(hi)].iter().sum();
    (10.0 * sum.max(1e-30).log10()) as f32
}

/// The middle half of segment `seg`.
fn core<'a>(audio: &'a [f32], seg: &(Phoneme, f32, f32)) -> &'a [f32] {
    let quarter = 0.25 * (seg.2 - seg.1);
    let a = ((seg.1 + quarter) * SR) as usize;
    let b = (((seg.2 - quarter) * SR) as usize).min(audio.len());
    &audio[a.min(b)..b]
}

/// The power spectrum of every `p` in a sentence, summed over their cores
/// (each weighted by its length).
fn hiss_of(audio: &[f32], segments: &[(Phoneme, f32, f32)], p: Phoneme) -> Vec<f64> {
    let mut sum = vec![0.0; (TOP / STEP) as usize];
    for seg in segments.iter().filter(|s| s.0 == p) {
        let x = core(audio, seg);
        for (s, v) in sum.iter_mut().zip(power(x)) {
            *s += v * x.len() as f64;
        }
    }
    sum
}

/// Mean square of the 25 ms frames (hop 10 ms) of `x`, dB, the loudest.
fn peak_frame(x: &[f32]) -> f32 {
    let (win, hop) = ((0.025 * SR) as usize, (0.01 * SR) as usize);
    let mut best = f32::MIN;
    let mut a = 0;
    while a + win <= x.len() {
        let ms = x[a..a + win].iter().map(|v| v * v).sum::<f32>() / win as f32;
        best = best.max(10.0 * ms.max(1e-15).log10());
        a += hop;
    }
    best
}

fn is_vowel(p: Phoneme) -> bool {
    matches!(
        p.manner(),
        synthie_trombone::Manner::Vowel | synthie_trombone::Manner::Diphthong
    )
}

/// Level of each `p`'s core below 8 kHz against the peak frame of the
/// louder of the vowels next to it, dB.
fn levels(audio: &[f32], segments: &[(Phoneme, f32, f32)], p: Phoneme) -> Vec<f32> {
    let mut out = Vec::new();
    for (k, seg) in segments.iter().enumerate() {
        if seg.0 != p {
            continue;
        }
        let x = core(audio, seg);
        // Mean square below 8 kHz, from the (Hann-normalised) spectrum.
        let p = power(x);
        let below: f64 =
            p[..bin(8000.0)].iter().sum::<f64>() * f64::from(STEP) / f64::from(SR) * 2.0;
        let level = (10.0 * below.max(1e-30).log10()) as f32;
        let vowel = [k.checked_sub(1), Some(k + 1)]
            .into_iter()
            .flatten()
            .filter_map(|j| segments.get(j))
            .filter(|s| is_vowel(s.0))
            .map(|s| {
                let a = (s.1 * SR) as usize;
                let b = ((s.2 * SR) as usize).min(audio.len());
                peak_frame(&audio[a.min(b)..b])
            })
            .fold(f32::MIN, f32::max);
        if vowel > f32::MIN {
            out.push(level - vowel);
        }
    }
    out
}

const SEASHELLS: &str = "She sells sea shells by the sea shore.";

#[test]
fn s_peaks_where_a_speakers_does_and_sh_far_below() {
    // In sentences the reader's /s/ has its centroid (1 to 8 kHz) at 5.8
    // kHz and the listener's at 5.8 (7.4 over 1 to 16 kHz), their /ʃ/ at
    // 4.5 and 4.7; the listener's /s/ is 10 dB stronger at 5 to 8 kHz than
    // at 12 to 16. The engine's /s/ was dark and dull: centroid 4.4 kHz,
    // its 5 to 8 kHz band 20 dB under the reader's against the vowel, and
    // its strongest band at 12 to 16 kHz (8 dB over 5 to 8): the open mouth
    // in front of the tongue had no resonance below 12 kHz.
    let (audio, segments) = say(SEASHELLS, 1.0);
    let s = hiss_of(&audio, &segments, Phoneme::S);
    let sh = hiss_of(&audio, &segments, Phoneme::Sh);
    let (cs, csh) = (centroid(&s, 1000.0, 8000.0), centroid(&sh, 1000.0, 8000.0));
    let wide = centroid(&s, 1000.0, 16_000.0);
    let tilt = band(&s, 5000.0, 8000.0) - band(&s, 12_000.0, 16_000.0);
    println!(
        "s: centroid {cs:.0} Hz (1-16 kHz: {wide:.0}), 5-8 kHz {tilt:+.1} dB re 12-16 | sh: centroid {csh:.0} Hz"
    );
    assert!(cs > 5000.0, "s centroid {cs:.0} Hz");
    assert!(cs > csh + 1200.0, "s {cs:.0} Hz, sh {csh:.0} Hz");
    assert!(
        (6000.0..10_000.0).contains(&wide),
        "s centroid {wide:.0} Hz"
    );
    assert!(tilt > 4.0, "s: 5-8 kHz only {tilt:+.1} dB over 12-16 kHz");
}

#[test]
fn s_is_as_loud_as_a_speakers() {
    // Against the peak of the vowel next to it, the core of the reader's
    // /s/ is 4.2 dB down below 8 kHz and the listener's 13.8 dB down. The
    // engine's was 15 dB down in the same sentences, quieter than its /ʃ/
    // (7 dB down), where both speakers' /s/ and /ʃ/ are within 3 dB.
    let (audio, segments) = say(SEASHELLS, 1.0);
    let s = levels(&audio, &segments, Phoneme::S);
    let sh = levels(&audio, &segments, Phoneme::Sh);
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    println!(
        "re the vowel: s {s:.1?} (mean {:.1}), sh {sh:.1?}",
        mean(&s)
    );
    assert!(s.len() >= 3);
    assert!((-15.0..-4.0).contains(&mean(&s)), "s {:.1} dB", mean(&s));
    assert!(
        mean(&s) > mean(&sh) - 6.0,
        "s {:.1} dB, sh {:.1} dB",
        mean(&s),
        mean(&sh)
    );
}

#[test]
fn the_teeth_tune_the_cavity_in_front_of_s_at_every_size() {
    // The jaw brings the incisors together for /s z/: the jet strikes them
    // through a narrow gap, and the small cavity between the tongue tip and
    // the teeth resonates near 9.8 kHz at the native size, scaling with the
    // body, as a speaker's does (the source's own fall above 5.5 kHz puts
    // the hiss's peak at 7 to 8 kHz). Without the teeth the mouth before
    // the tongue was a horn with its first resonance at 13 to 15 kHz.
    for size in [0.7, 1.0, 1.3] {
        let mut t = Trombone::new(SR);
        t.set_engine(Engine::Physical);
        t.set_tract_length(size);
        t.articulate_phoneme(Phoneme::Aa);
        t.settle();
        t.articulate_phoneme(Phoneme::S);
        t.settle();
        let mut ir = vec![0.0f32; 8192];
        t.source_response(41, 0.37, 8.0, &mut ir);
        let rate = t.tick_rate();
        // Radiated: the flow's spectrum times frequency, peak 3 to 18 kHz.
        let (mut best, mut at) = (0.0f64, 0.0f32);
        let mut hz = 3000.0f32;
        while hz < 18_000.0 {
            let w = 2.0 * std::f64::consts::PI * f64::from(hz) / f64::from(rate);
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (n, &v) in ir.iter().enumerate() {
                re += f64::from(v) * (w * n as f64).cos();
                im -= f64::from(v) * (w * n as f64).sin();
            }
            let p = (re * re + im * im) * f64::from(hz) * f64::from(hz);
            if p > best {
                (best, at) = (p, hz);
            }
            hz += 100.0;
        }
        println!(
            "size {size}: front cavity at {at:.0} Hz ({:.0} Hz scaled)",
            at * size
        );
        assert!(
            (8000.0..11_000.0).contains(&(at * size)),
            "size {size}: peak at {at:.0} Hz"
        );
    }
}

#[test]
fn z_hisses_in_the_band_of_s_under_its_voice() {
    // A /z/ heard as /ð/ has its voice but not enough of the hiss of /s/.
    // In the reader's sentences /z/ is 3.9 dB under /s/ at 5 to 8 kHz, and
    // its centroid (1 to 8 kHz) as high (5.9 kHz).
    let (audio, segments) = say("His zoo has six zebras and seven seals.", 1.0);
    let s = hiss_of(&audio, &segments, Phoneme::S);
    let z = hiss_of(&audio, &segments, Phoneme::Z);
    let count = |p| segments.iter().filter(|s| s.0 == p).count() as f32;
    // Per segment: the sums are weighted by length.
    let length = |p| {
        segments
            .iter()
            .filter(|s| s.0 == p)
            .map(|s| s.2 - s.1)
            .sum::<f32>()
    };
    let per = |p: &[f64], ph| band(p, 5000.0, 8000.0) - 10.0 * (length(ph) / count(ph)).log10();
    let gap = per(&s, Phoneme::S) - per(&z, Phoneme::Z);
    let cz = centroid(&z, 1000.0, 8000.0);
    println!("z: 5-8 kHz {gap:.1} dB under s, centroid {cz:.0} Hz");
    assert!(gap < 9.0, "z {gap:.1} dB under s at 5-8 kHz");
    assert!(cz > 5000.0, "z centroid {cz:.0} Hz");
}

/// Each segment's phoneme with its start and end, seconds.
type Segments = Vec<(Phoneme, f32, f32)>;

/// [`say`] with the flow out of the lips, cm³/s, at every sample.
fn say_with_lips(text: &str) -> (Vec<f32>, Segments, Vec<f32>) {
    let settings = tuned(Preset::Deep.settings().with_engine(Engine::Physical));
    let mut voice = Trombone::new(SR);
    settings.apply(&mut voice);
    Speaker::prepare(&mut voice);
    let utterance = Utterance::from_text(text, &settings.prosody());
    let mut t = 0.0;
    let segments = utterance
        .segments()
        .iter()
        .map(|s| {
            t += s.duration;
            (s.phoneme, t - s.duration, t)
        })
        .collect();
    let mut speaker = Speaker::new();
    speaker.speak(utterance);
    let (mut audio, mut lips) = (Vec::new(), Vec::new());
    while speaker.is_speaking() {
        audio.push(speaker.process(&mut voice));
        lips.push(voice.gauges().lip_flow);
    }
    (audio, segments, lips)
}

/// Level of `audio` between `lo` and `hi` Hz in 10 ms frames every 2.5 ms
/// from `from` to `to` seconds, dB (the units of [`band`]).
fn frames(audio: &[f32], from: f32, to: f32, lo: f32, hi: f32) -> Vec<f32> {
    let (win, hop) = ((0.01 * SR) as usize, (0.0025 * SR) as usize);
    let (mut a, end) = ((from * SR) as usize, ((to * SR) as usize).min(audio.len()));
    let mut out = Vec::new();
    while a + win <= end {
        out.push(band(&power(&audio[a..a + win]), lo, hi));
        a += hop;
    }
    out
}

/// What the way out of each /s/ into an /f/ in `text` looks like: the
/// lowest 10 ms frame (below 8 kHz) between the middle of the /s/ and the
/// core of the /f/ against the frames of that core, how long the band of
/// the /s/ (2 to 8 kHz) takes from 3 to 10 dB under its level in the first
/// half of the /s/, and the largest flow out of the lips from the middle of
/// the /s/ to the end of the core of the /f/.
fn hand_overs(text: &str) -> Vec<(f32, f32, f32)> {
    let (audio, segments, lips) = say_with_lips(text);
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    let mut out = Vec::new();
    for w in segments.windows(2) {
        let (s, f) = (w[0], w[1]);
        if s.0 != Phoneme::S || f.0 != Phoneme::F {
            continue;
        }
        let mid = s.1 + 0.5 * (s.2 - s.1);
        let (fa, fb) = (f.1 + 0.25 * (f.2 - f.1), f.2 - 0.25 * (f.2 - f.1));
        let plateau =
            mean(&frames(&audio, f.1 - 0.0, fb, 0.0, 8000.0)[(((fa - f.1) / 0.0025) as usize)..]);
        let between = frames(&audio, mid, fa, 0.0, 8000.0);
        let dip = between.iter().fold(f32::MAX, |m, &v| m.min(v)) - plateau;
        let hiss = mean(&frames(
            &audio,
            s.1 + 0.2 * (s.2 - s.1),
            mid,
            2000.0,
            8000.0,
        ));
        let fall = frames(&audio, mid - 0.02, fb, 2000.0, 8000.0);
        let first = |db: f32| {
            fall.iter()
                .position(|&v| v < hiss - db)
                .unwrap_or(fall.len())
        };
        let fall_ms = 2.5 * (first(10.0) as f32 - first(3.0) as f32);
        let (a, b) = ((mid * SR) as usize, ((fb * SR) as usize).min(lips.len()));
        let surge = lips[a..b].iter().fold(0.0f32, |m, &v| m.max(v));
        out.push((dip, fall_ms, surge));
    }
    out
}

#[test]
fn s_fades_into_f_without_a_gap_or_a_puff() {
    // Into an /f/ the LibriSpeech reader's /s/ fades by 10 dB over some 25
    // ms, and the level between them never falls more than about 3 dB
    // under the /f/. The engine's tongue used to let go at once: its /s/
    // stopped within 4 ms, the air rushed out of the open mouth (lip flow
    // from about 120 to 950 cm³/s), and only then did the lip shut on a
    // faint /f/: a gap, a puff and a hiss, the pattern of /p/, and "makes
    // fine" was heard as "makes pine".
    for text in [
        "The juice of lemons makes fine punch.",
        "He makes fine tea.",
        "This fact is clear.",
    ] {
        let found = hand_overs(text);
        assert!(!found.is_empty(), "{text}: no /s f/");
        for (dip, fall, surge) in found {
            println!(
                "{text}: dip {dip:+.1} dB re /f/, /s/ falls 10 dB in {fall:.1} ms, lip flow up to {surge:.0} cm³/s"
            );
            // Let go at once: dips of 3 to 6.5 dB, falls of 0 to 2.5 ms,
            // 950 to 1090 cm³/s. Now some 0 to 2 dB, 18 to 20 ms, 460 to
            // 490 cm³/s (about 120 to 180 in the /s/ itself).
            assert!(dip > -2.5, "{text}: {dip:.1} dB under the /f/");
            assert!(fall >= 12.0, "{text}: /s/ falls 10 dB in {fall:.1} ms");
            assert!(surge < 650.0, "{text}: lip flow {surge:.0} cm³/s");
        }
    }
}

#[test]
fn f_and_v_are_as_loud_as_a_speakers() {
    // Between vowels the reader's /f/ is 16 dB under the louder vowel's
    // peak (below 8 kHz, n = 9) and his /v/, mostly voice, 19 dB under.
    // The engine's /f/ was 27 to 36 dB under, the free jet in the lip's
    // gap alone, and heard as /p/; the jet's force on the edges of the
    // teeth and the lip (`Tuning::labial_edge`) puts it 22 dB under (the
    // mean; 18 to 30).
    // It adds no air to the gap, so /v/ keeps its voice.
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    let (mut f, mut v) = (Vec::new(), Vec::new());
    for text in [
        "A cup of coffee.",
        "They often laugh.",
        "Five of them gave over.",
        "She was fond of a safe offer.",
    ] {
        let (audio, segments) = say(text, 1.0);
        f.extend(levels(&audio, &segments, Phoneme::F));
        v.extend(levels(&audio, &segments, Phoneme::V));
    }
    println!(
        "re the vowel: f {f:.1?} (mean {:.1}), v {v:.1?} (mean {:.1})",
        mean(&f),
        mean(&v)
    );
    assert!(f.len() >= 5 && v.len() >= 3);
    assert!((-24.0..-12.0).contains(&mean(&f)), "f {:.1} dB", mean(&f));
    assert!(f.iter().all(|&x| x < -10.0), "f {f:.1?}");
    assert!((-16.0..-4.0).contains(&mean(&v)), "v {:.1} dB", mean(&v));
}
