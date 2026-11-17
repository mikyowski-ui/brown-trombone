//! The voice of a /dʒ/ before a pause runs through its release (physical
//! engine, `ArticulationConfig::final_voicing_hold`).
//!
//! "He wore a large orange badge." was heard as "... batch": at the end of
//! a statement the glottis opened for the hiss as the frication of the
//! final /dʒ/ began, and with the mouth's pressure at the release the folds
//! all but stopped. Now the folds stay together, the cover slack, for some
//! 20 ms after the mouth opens, and only then open for the hiss. These
//! tests measure that at the folds and in the sound: the voice's swing just
//! after the release, the glottal gap behind the folds through it, the
//! hiss after it, and that nothing else changes.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use synthie_trombone::speech::{Speaker, Utterance};
use synthie_trombone::{Engine, Manner, Phoneme, Preset, TractModel, Trombone, Tuning};

const SR: f32 = 48_000.0;

/// One output sample's readings.
#[derive(Clone, Copy)]
struct Reading {
    audio: f32,
    shut: bool,
    /// Flow through the glottis, cm³/s.
    glottal: f32,
    /// The glottal area not between the vibrating edges: the gap standing
    /// behind them, cm².
    standing: f32,
}

/// `text` said by the deep voice with `hold` as `final_voicing_hold`: the
/// readings and the segments (phoneme, start, end, seconds).
fn say(text: &str, hold: f32, tract: TractModel) -> (Vec<Reading>, Vec<(Phoneme, f32, f32)>) {
    let mut settings = Preset::Deep.settings().with_engine(Engine::Physical);
    settings.tuning = Tuning::DEFAULT.with_tract(tract);
    settings.tuning.articulation.final_voicing_hold = hold;
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
    let mut out = Vec::new();
    while speaker.is_speaking() {
        let audio = speaker.process(&mut voice);
        let g = voice.gauges();
        out.push(Reading {
            audio,
            shut: g.mouth_shut,
            glottal: g.glottal_flow,
            standing: g.glottal_area - g.edge_area,
        });
    }
    (out, segments)
}

/// The sample at which the mouth opens out of the closure of the last
/// affricate of the text.
fn release(r: &[Reading], segments: &[(Phoneme, f32, f32)]) -> usize {
    let k = (1..segments.len())
        .rev()
        .find(|&k| {
            segments[k].0.manner() == Manner::Affricate && segments[k - 1].0 == segments[k].0
        })
        .expect("an affricate")
        - 1;
    let from = ((segments[k].1 - 0.03) * SR) as usize;
    let to = ((segments[k + 1].2 * SR) as usize).min(r.len());
    let shut = (from..to).find(|&i| r[i].shut).expect("a closure");
    (shut..to).find(|&i| !r[i].shut).expect("a release")
}

/// Mean over the 10 ms frames from `a` to `b` seconds after sample `at` of
/// the glottal flow's range in each, cm³/s: the voice's swing.
fn swing(r: &[Reading], at: usize, a: f32, b: f32) -> f32 {
    let win = (0.01 * SR) as usize;
    let mut k = at + (a * SR) as usize;
    let (mut sum, mut n) = (0.0, 0);
    while k + win <= at + (b * SR) as usize {
        let (lo, hi) = r[k..k + win]
            .iter()
            .fold((f32::MAX, f32::MIN), |(l, h), q| {
                (l.min(q.glottal), h.max(q.glottal))
            });
        sum += hi - lo;
        n += 1;
        k += win;
    }
    sum / n as f32
}

/// Mean standing glottal gap from `a` to `b` seconds after sample `at`, cm².
fn standing(r: &[Reading], at: usize, a: f32, b: f32) -> f32 {
    let part = &r[at + (a * SR) as usize..at + (b * SR) as usize];
    part.iter().map(|q| q.standing).sum::<f32>() / part.len() as f32
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

/// The loudest 10 ms (hop 2.5 ms) of the 2 to 8 kHz hiss from `a` to `b`
/// seconds after sample `at`, dB.
fn hiss(r: &[Reading], at: usize, a: f32, b: f32) -> f32 {
    let x: Vec<f32> = r.iter().map(|q| q.audio).collect();
    let (win, hop) = ((0.01 * SR) as usize, (0.0025 * SR) as usize);
    let end = (at + (b * SR) as usize).min(x.len());
    let mut k = at + (a * SR) as usize;
    let mut best = 0.0f32;
    while k + win <= end {
        let mut sum = 0.0;
        let mut hz = 2050.0;
        while hz < 8000.0 {
            sum += bin(&x[k..k + win], hz);
            hz += 100.0;
        }
        best = best.max(sum);
        k += hop;
    }
    10.0 * best.max(1e-20).log10()
}

const HOLD: f32 = Tuning::DEFAULT.articulation.final_voicing_hold;

/// The four measurements, with the hold and without, on the anatomical
/// tract: (swing 10 to 40 ms after the release, standing gap 5 to 30 ms
/// after it, standing gap 60 to 80 ms after it, loudest hiss in the 150 ms
/// after it).
fn measure(text: &str, hold: f32) -> (f32, f32, f32, f32) {
    let (r, segments) = say(text, hold, TractModel::Anatomical);
    let at = release(&r, &segments);
    (
        swing(&r, at, 0.01, 0.04),
        standing(&r, at, 0.005, 0.03),
        standing(&r, at, 0.06, 0.08),
        hiss(&r, at, 0.0, 0.15),
    )
}

const FINALS: [&str; 3] = [
    "He wore a large orange badge.",
    "What is your age?",
    "Stand at the edge.",
];

#[test]
fn the_voice_runs_through_the_release() {
    for text in FINALS {
        let (on, ..) = measure(text, HOLD);
        let (off, ..) = measure(text, 0.0);
        println!(
            "{text}: the voice swings by {on:.0} cm³/s 10 to 40 ms after the release ({off:.0} without the hold)"
        );
        // Measured: 118 against 90, 56 against 24, 191 against 171.
        assert!(
            on > off + 15.0,
            "{text}: {on:.0} cm³/s, {off:.0} without the hold"
        );
    }
}

#[test]
fn the_glottis_opens_for_the_hiss_after_the_hold() {
    for text in FINALS {
        let (_, early, late, _) = measure(text, HOLD);
        let (_, early_off, late_off, _) = measure(text, 0.0);
        println!(
            "{text}: standing gap {early:.3} cm² 5 to 30 ms after the release ({early_off:.3} without), {late:.3} 60 to 80 ms after it ({late_off:.3})"
        );
        // Together through the release (the gap opens some 10 ms later:
        // measured 0.010 to 0.011 against 0.017 to 0.019 cm²)...
        assert!(
            early < early_off - 0.005,
            "{text}: {early:.3} cm², {early_off:.3} without"
        );
        // ...and apart for the hiss after it, as without the hold (0.047
        // and 0.048).
        assert!(
            (late - late_off).abs() < 0.005,
            "{text}: {late:.3} cm², {late_off:.3} without"
        );
    }
}

#[test]
fn the_hiss_after_it_is_as_loud() {
    for text in FINALS {
        let (.., on) = measure(text, HOLD);
        let (.., off) = measure(text, 0.0);
        println!("{text}: loudest hiss {on:.1} dB ({off:.1} without the hold)");
        // Measured: 0.3 to 0.5 dB under.
        assert!(on > off - 3.0, "{text}: {on:.1} dB, {off:.1} without");
    }
}

#[test]
fn nothing_else_changes() {
    // Before a vowel, a voiced or a voiceless sound, inside a word and at
    // the start of one, the /dʒ/ is said as without the hold, and so is
    // every other sound.
    for text in [
        "The juice of lemons makes fine punch.",
        "Wear the badge on your coat.",
        "A large orange cat.",
        "The huge river.",
        "A large size.",
        "The raging storm.",
        "He wore a large orange batch.",
    ] {
        for tract in [TractModel::Anatomical, TractModel::Original] {
            let (on, _) = say(text, HOLD, tract);
            let (off, _) = say(text, 0.0, tract);
            assert_eq!(on.len(), off.len(), "{text}");
            assert!(
                on.iter()
                    .zip(&off)
                    .all(|(a, b)| a.audio.to_bits() == b.audio.to_bits()),
                "{tract:?} {text}: the hold changed it"
            );
        }
    }
}
