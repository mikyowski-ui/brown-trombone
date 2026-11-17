//! Gates on the affricates /tʃ dʒ/ of the physical engine: after its
//! closure each one hisses as its fricative does, on both tracts, before a
//! front, a back and a rounded vowel and at the end of a word.
//!
//! "The juice of lemons" was heard as "the dose of lemons": the frication
//! of /dʒ/ was made with the folds pressed as for its closure, and its
//! channel, as narrow as that of /ʒ/ (0.04 to 0.05 cm²), carried a mean
//! flow of some 20 cm³/s against the 90 of /ʒ/. The jet was turbulent for
//! 45 to 53 % of the frication between vowels (/ʒ/ 65 to 71 %), 16 to 24 %
//! at the end of a word, and the hiss of a final /dʒ/ was 12 to 15 dB under
//! that of a final /ʒ/. The tract made no difference: the original one
//! measured the same.
//!
//! A final /dʒ/ is told from a final /tʃ/ by the voice through its closure
//! and into its frication as well: "He wore a large orange badge." was
//! heard as "batch" while the voice died in its frication.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]

use synthie_trombone::speech::{Speaker, Utterance};
use synthie_trombone::{Engine, Manner, Phoneme, Preset, TractModel, Trombone, Tuning};

const SR: f32 = 48_000.0;

/// How a consonant's frication measured: the loudest 10 ms of its hiss
/// (2 to 8 kHz) against the loudest 10 ms of the vowel next to it, dB; the
/// mean square of the engine's jet noise over it, dB re 1 cmH2O; and its
/// length, seconds.
#[derive(Debug, Clone, Copy)]
struct Frication {
    hiss: f32,
    source: f32,
    length: f32,
}

/// Power of x near hz (a Hann-windowed DFT bin).
fn bin(x: &[f32], hz: f32) -> f32 {
    let n = x.len();
    let w = 2.0 * std::f32::consts::PI * hz / SR;
    let (mut re, mut im) = (0.0f32, 0.0f32);
    for (k, &v) in x.iter().enumerate() {
        let h = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * k as f32 / n as f32).cos();
        re += v * h * (w * k as f32).cos();
        im -= v * h * (w * k as f32).sin();
    }
    (re * re + im * im) / n as f32
}

/// The loudest 10 ms frame (hop 2.5 ms) of x between lo and hi Hz, in 100
/// Hz bins.
fn loudest(x: &[f32], lo: f32, hi: f32) -> f32 {
    let (win, hop) = ((0.01 * SR) as usize, (0.0025 * SR) as usize);
    let mut best = 0.0f32;
    let mut a = 0;
    while a + win <= x.len() {
        let mut sum = 0.0;
        let mut hz = lo + 50.0;
        while hz < hi {
            sum += bin(&x[a..a + win], hz);
            hz += 100.0;
        }
        best = best.max(sum);
        a += hop;
    }
    best
}

fn db(x: f32) -> f32 {
    10.0 * x.max(1e-20).log10()
}

fn is_vowel(p: Phoneme) -> bool {
    matches!(p.manner(), Manner::Vowel | Manner::Diphthong)
}

/// The frication of the first affricate, /ʃ/ or /ʒ/ of `text` (for an
/// affricate the second of its two segments), said by the deep voice on
/// `tract`.
fn frication(text: &str, tract: TractModel) -> Frication {
    let mut settings = Preset::Deep.settings().with_engine(Engine::Physical);
    settings.tuning = Tuning::DEFAULT.with_tract(tract);
    let mut voice = Trombone::new(SR);
    settings.apply(&mut voice);
    Speaker::prepare(&mut voice);
    let utterance = Utterance::from_text(text, &settings.prosody());
    let mut t = 0.0;
    let segments: Vec<(Phoneme, f32, f32)> = utterance
        .segments()
        .iter()
        .map(|s| {
            t += s.duration;
            (s.phoneme, t - s.duration, t)
        })
        .collect();
    let mut speaker = Speaker::new();
    speaker.speak(utterance);
    let (mut audio, mut jet) = (Vec::new(), Vec::new());
    while speaker.is_speaking() {
        audio.push(speaker.process(&mut voice));
        jet.push(voice.gauges().jet_noise);
    }
    let k = (1..segments.len())
        .find(|&k| {
            let p = segments[k].0;
            match p.manner() {
                Manner::Affricate => segments[k - 1].0 == p,
                Manner::Fricative => matches!(p, Phoneme::Sh | Phoneme::Zh),
                _ => false,
            }
        })
        .unwrap_or_else(|| panic!("{text}: no frication"));
    let s = segments[k];
    let vowel = segments
        .get(k + 1)
        .filter(|v| is_vowel(v.0))
        .or_else(|| segments[..k].iter().rev().find(|v| is_vowel(v.0)))
        .expect("a vowel");
    let span = |a: f32, b: f32| ((a * SR) as usize, ((b * SR) as usize).min(audio.len()));
    let (va, vb) = span(vowel.1, vowel.2);
    let (a, b) = span(s.1, s.2);
    let reference = loudest(&audio[va..vb], 0.0, 8000.0);
    let source = jet[a..b].iter().map(|x| x * x).sum::<f32>() / (b - a) as f32;
    Frication {
        hiss: db(loudest(&audio[a..b], 2000.0, 8000.0) / reference),
        source: db(source),
        length: s.2 - s.1,
    }
}

/// Each affricate in each context against its fricative in the same one,
/// on both tracts: (context, affricate, fricative).
fn pairs(affricate: &str, fricative: &str) -> Vec<(TractModel, String, Frication, Frication)> {
    let contexts = [
        ("[AX0 {} IY1 S]", "before /i/"),
        ("[AX0 {} AA1 S]", "before /ɑ/"),
        ("[AX0 {} UW1 S]", "before /u/"),
        ("[EH1 {}]", "after /ɛ/, final"),
        ("[UW1 {}]", "after /u/, final"),
    ];
    let mut out = Vec::new();
    for tract in [TractModel::Anatomical, TractModel::Original] {
        for (pattern, name) in contexts {
            let a = frication(&pattern.replace("{}", affricate), tract);
            let f = frication(&pattern.replace("{}", fricative), tract);
            println!(
                "{tract:?} {affricate} {name}: hiss {:.1} dB (/{fricative}/ {:.1}), source {:.1} dB (/{fricative}/ {:.1}), {:.0} ms ({:.0})",
                a.hiss,
                f.hiss,
                a.source,
                f.source,
                1000.0 * a.length,
                1000.0 * f.length
            );
            out.push((tract, name.to_string(), a, f));
        }
    }
    out
}

/// Whether the glottal flow of the 10 ms around sample `a` is a voice: it
/// swings by more than 50 cm³/s and repeats itself (normalised
/// autocorrelation over 0.5) at some lag between 2.5 and 12.5 ms. (The
/// flutter of folds held apart for a /tʃ/ swings by 20 to 40.)
fn periodic(flow: &[f32], a: usize) -> bool {
    let win = (0.01 * SR) as usize;
    let (lo, hi) = ((0.0025 * SR) as usize, (0.0125 * SR) as usize);
    let a = a.saturating_sub(hi / 2);
    if a + win + hi > flow.len() {
        return false;
    }
    let x = &flow[a..a + win + hi];
    let (min, max) = x[..win]
        .iter()
        .fold((f32::MAX, f32::MIN), |(l, h), &v| (l.min(v), h.max(v)));
    if max - min < 50.0 {
        return false;
    }
    let mean = |s: &[f32]| s.iter().sum::<f32>() / s.len() as f32;
    let m0 = mean(&x[..win]);
    (lo..=hi).step_by(12).any(|lag| {
        let m1 = mean(&x[lag..lag + win]);
        let (mut xy, mut xx, mut yy) = (0.0, 0.0, 0.0);
        for k in 0..win {
            let (p, q) = (x[k] - m0, x[k + lag] - m1);
            xy += p * q;
            xx += p * p;
            yy += q * q;
        }
        xy / (xx * yy).sqrt().max(1e-12) > 0.5
    })
}

/// Share of the closure and frication of the last affricate of `text`
/// (its two segments) over which the glottal flow is a voice, in 10 ms
/// frames every 2.5 ms, said by the deep voice on `tract`.
fn voiced_share(text: &str, tract: TractModel) -> f32 {
    let mut settings = Preset::Deep.settings().with_engine(Engine::Physical);
    settings.tuning = Tuning::DEFAULT.with_tract(tract);
    let mut voice = Trombone::new(SR);
    settings.apply(&mut voice);
    Speaker::prepare(&mut voice);
    let utterance = Utterance::from_text(text, &settings.prosody());
    let mut t = 0.0;
    let segments: Vec<(Phoneme, f32, f32)> = utterance
        .segments()
        .iter()
        .map(|s| {
            t += s.duration;
            (s.phoneme, t - s.duration, t)
        })
        .collect();
    let mut speaker = Speaker::new();
    speaker.speak(utterance);
    let mut flow = Vec::new();
    while speaker.is_speaking() {
        speaker.process(&mut voice);
        flow.push(voice.gauges().glottal_flow);
    }
    let k = (1..segments.len())
        .rev()
        .find(|&k| {
            segments[k].0.manner() == Manner::Affricate && segments[k - 1].0 == segments[k].0
        })
        .unwrap_or_else(|| panic!("{text}: no affricate"));
    let (a, b) = (
        (segments[k - 1].1 * SR) as usize,
        ((segments[k].2 * SR) as usize).min(flow.len()),
    );
    let hop = (0.0025 * SR) as usize;
    let frames: Vec<bool> = (a..b).step_by(hop).map(|i| periodic(&flow, i)).collect();
    frames.iter().filter(|&&v| v).count() as f32 / frames.len() as f32
}

#[test]
fn final_dzh_is_voiced_through_its_closure_and_frication_where_tsh_is_not() {
    // A final /dʒ/ is often partly devoiced in English; what keeps it /dʒ/
    // is, among the rest, the voice through its closure and into its
    // frication, where that of /tʃ/ stops with its closure. Measured with
    // the voice held through the release of a /dʒ/ before a pause
    // (`final_voicing_hold`): /dʒ/ 65 to 82 % of the two on both tracts,
    // /tʃ/ 0 to 9 %. Without the hold "badge" ending the sentence was
    // voiced for 56 % (and heard as "batch"), the words alone 72 to 82 %.
    for tract in [TractModel::Anatomical, TractModel::Original] {
        for (dzh, tsh) in [
            (
                "He wore a large orange badge.",
                "He wore a large orange batch.",
            ),
            ("[B AE1 JH]", "[B AE1 CH]"),
            ("[EH1 JH]", "[EH1 CH]"),
            ("[UW1 JH]", "[UW1 CH]"),
        ] {
            let (d, t) = (voiced_share(dzh, tract), voiced_share(tsh, tract));
            println!(
                "{tract:?} {dzh}: voiced {:.0} % of the closure and frication, {tsh} {:.0} %",
                100.0 * d,
                100.0 * t
            );
            assert!(d > 0.6, "{tract:?} {dzh}: voiced {:.0} %", 100.0 * d);
            assert!(
                d > t + 0.4,
                "{tract:?} {dzh}: voiced {:.0} %, {tsh} {:.0} %",
                100.0 * d,
                100.0 * t
            );
        }
    }
}

#[test]
fn dzh_hisses_after_its_release_as_zh_does() {
    // Now: the hiss of /dʒ/ is 0 to 7 dB under that of /ʒ/ (or louder)
    // and its jet noise up to 3.3 dB under, where it was 12 to 15 dB and
    // 12 to 19 dB under at the end of a word.
    for (tract, context, a, f) in pairs("JH", "ZH") {
        assert!(
            a.hiss > f.hiss - 9.0,
            "{tract:?} /dʒ/ {context}: hiss {:.1} dB, /ʒ/ {:.1}",
            a.hiss,
            f.hiss
        );
        assert!(
            a.source > f.source - 6.0,
            "{tract:?} /dʒ/ {context}: jet noise {:.1} dB, /ʒ/ {:.1}",
            a.source,
            f.source
        );
        // Not a /ʒ/ with a stop in front: its frication is the shorter.
        assert!(
            a.length < f.length,
            "{tract:?} /dʒ/ {context}: {:.0} ms",
            1000.0 * a.length
        );
    }
}

#[test]
fn tsh_hisses_after_its_release_as_sh_does() {
    // /tʃ/ has hissed as /ʃ/ all along (its glottis is spread): 0 to 7 dB
    // louder at its peak.
    for (tract, context, a, f) in pairs("CH", "SH") {
        assert!(
            a.hiss > f.hiss - 6.0,
            "{tract:?} /tʃ/ {context}: hiss {:.1} dB, /ʃ/ {:.1}",
            a.hiss,
            f.hiss
        );
        assert!(
            a.source > f.source - 6.0,
            "{tract:?} /tʃ/ {context}: jet noise {:.1} dB, /ʃ/ {:.1}",
            a.source,
            f.source
        );
        assert!(
            a.length < f.length,
            "{tract:?} /tʃ/ {context}: {:.0} ms",
            1000.0 * a.length
        );
    }
}
