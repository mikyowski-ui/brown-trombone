//! Gates on how natural the physical engine's speech is: how its phrases
//! start and end, that the level flows inside a sentence instead of falling
//! dead between sounds, how its hisses are made, and that a statement falls
//! at its end. Each bound is set from the judge's naturalness measure of
//! the LibriSpeech reader (`tools/judge/natural.py`, `report/natural.md`)
//! and stated with it.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

#[path = "../examples/support/hiss.rs"]
mod hiss;

use synthie_trombone::speech::{Speaker, Utterance};
use synthie_trombone::{Engine, Preset, Trombone};

const SR: f32 = 48_000.0;
/// Level frames: 25 ms windows every 10 ms, as the judge's.
const HOP: usize = 480;
const WINDOW: usize = 1200;

/// A sentence said by the default voice on the physical engine: the
/// audio, and the pitch the folds produced per sample (0 while not
/// vibrating).
fn say(text: &str) -> (Vec<f32>, Vec<f32>) {
    let settings = Preset::Deep.settings().with_engine(Engine::Physical);
    let mut voice = Trombone::new(SR);
    settings.apply(&mut voice);
    Speaker::prepare(&mut voice);
    let mut speaker = Speaker::new();
    speaker.speak(Utterance::from_text(text, &settings.prosody()));
    let (mut audio, mut pitch) = (Vec::new(), Vec::new());
    let mut tail = (0.3 * SR) as usize;
    while tail > 0 {
        if !speaker.is_speaking() {
            tail -= 1;
        }
        audio.push(speaker.process(&mut voice));
        pitch.push(voice.gauges().fold_pitch);
    }
    (audio, pitch)
}

/// Level per frame, dB under the loudest three frames' mean.
fn levels(x: &[f32]) -> Vec<f32> {
    let db: Vec<f32> = (0..x.len() / HOP)
        .map(|f| {
            let c = f * HOP + HOP / 2;
            let (a, b) = (c.saturating_sub(WINDOW / 2), (c + WINDOW / 2).min(x.len()));
            let ms = x[a..b].iter().map(|v| v * v).sum::<f32>() / (b - a).max(1) as f32;
            10.0 * ms.max(1e-15).log10()
        })
        .collect();
    let peak = db
        .windows(3)
        .map(|w| w.iter().sum::<f32>() / 3.0)
        .fold(f32::MIN, f32::max);
    db.iter().map(|v| v - peak).collect()
}

/// Seconds from where the level first reaches `lo` dB under the peak to
/// where it first reaches `hi`, and from where it last is at `hi` to where
/// it last is at `lo`.
fn edges(rel: &[f32], lo: f32, hi: f32) -> (f32, f32) {
    let first = |l: f32| rel.iter().position(|&v| v >= l).unwrap_or(0);
    let last = |l: f32| rel.iter().rposition(|&v| v >= l).unwrap_or(0);
    let frame = HOP as f32 / SR;
    (
        (first(hi) - first(lo)) as f32 * frame,
        (last(lo) - last(hi)) as f32 * frame,
    )
}

const SENTENCES: [&str; 3] = [
    "The birch canoe slid on the smooth planks.",
    "Mister quilter is the apostle of the middle classes.",
    "The little girl had been asleep but she heard the raps and opened the door.",
];

#[test]
fn phrases_swell_in_and_die_away() {
    // The reader's phrases rise from 30 to 6 dB under their peak in 145 ms
    // (44 to 406 across his sentences) and fall back in 710 ms (277 to
    // 1257): the last words are quieter, then the voice dies away. The
    // engine's fell in 125 ms, its last syllable as loud as any.
    let mut falls = 0.0;
    for text in SENTENCES {
        let (audio, _) = say(text);
        let (rise, fall) = edges(&levels(&audio), -30.0, -6.0);
        println!(
            "{text}: rise {:.0} ms, fall {:.0} ms",
            rise * 1e3,
            fall * 1e3
        );
        assert!((0.04..0.45).contains(&rise), "{text}: rise {rise}");
        falls += fall;
    }
    let fall = falls / SENTENCES.len() as f32;
    assert!(fall > 0.2, "mean fall {:.0} ms", fall * 1e3);
}

#[test]
fn nothing_is_heard_before_the_breath() {
    // Moving the tongue and lips to the first sound at speech speed pushed
    // a puff of air out of the silent tract, a thump 13 to 20 dB under the
    // sentence's peak before it began. The articulators are set silently;
    // what is left is the breath building behind a first closure, through
    // the walls (about 45 dB down).
    for text in ["Bob.", "The birch canoe.", "Sat.", "Glue the sheet."] {
        let (audio, _) = say(text);
        let rel = levels(&audio);
        let lead_in = (0.08 * SR) as usize / HOP;
        let loudest = rel[..lead_in].iter().fold(f32::MIN, |m, &v| m.max(v));
        println!("{text}: lead-in at most {loudest:.1} dB");
        assert!(loudest < -35.0, "{text}: {loudest:.1} dB before speech");
    }
}

#[test]
fn no_dead_silence_inside_a_phrase() {
    // Inside a phrase the reader's level never falls 60 dB under its peak;
    // his dips below 30 dB last 67 ms on average, 250 at most (a pause).
    // Without punctuation there is no pause: a voiceless closure is the
    // longest gap, some 75 ms.
    let text = "The birch canoe slid on the smooth planks";
    let (audio, _) = say(text);
    let rel = levels(&audio);
    let a = rel.iter().position(|&v| v >= -30.0).unwrap_or(0);
    let b = rel.iter().rposition(|&v| v >= -30.0).unwrap_or(0);
    let (mut dead, mut run, mut dip, mut drun) = (0, 0, 0, 0);
    for &v in &rel[a..b] {
        run = if v < -60.0 { run + 1 } else { 0 };
        drun = if v < -30.0 { drun + 1 } else { 0 };
        dead = dead.max(run);
        dip = dip.max(drun);
    }
    let frame_ms = HOP as f32 / SR * 1000.0;
    println!(
        "longest dead stretch {:.0} ms, longest dip {:.0} ms",
        dead as f32 * frame_ms,
        dip as f32 * frame_ms
    );
    assert!(dead as f32 * frame_ms <= 30.0, "dead {dead} frames");
    assert!(dip as f32 * frame_ms <= 160.0, "dip {dip} frames");
}

#[test]
fn a_statement_falls_at_its_end() {
    // The reader's last 150 ms of voice sit 2.8 semitones under his median
    // (-5.5 to +1.6 across sentences; the listener's -4.5), and his pitch
    // spans 10 semitones (5 to 95 %). The engine's ended 2 under, over 5.
    for text in SENTENCES {
        let (_, pitch) = say(text);
        let mut voiced: Vec<f32> = pitch.iter().copied().filter(|&f| f > 0.0).collect();
        let tail = voiced.len().saturating_sub((0.15 * SR) as usize);
        let end = voiced[tail..].iter().sum::<f32>() / (voiced.len() - tail).max(1) as f32;
        voiced.sort_by(f32::total_cmp);
        let median = voiced[voiced.len() / 2];
        let pct = |p: f32| voiced[((voiced.len() - 1) as f32 * p) as usize];
        let fall = 12.0 * (end / median).log2();
        let range = 12.0 * (pct(0.95) / pct(0.05)).log2();
        println!("{text}: end {fall:+.1} st re the median, range {range:.1} st");
        assert!(fall < -2.0, "{text}: ends {fall:+.1} st");
        assert!(range > 5.5, "{text}: range {range:.1} st");
    }
}

#[test]
fn hisses_are_broad_and_s_is_brighter_than_sh() {
    // The reader's /s/ has its centroid (1 to 8 kHz) at 5.8 kHz, his /ʃ/
    // at 4.5 kHz with the band within 6 dB of its peak 635 Hz wide and a
    // flatness of 0.39. The engine's /ʃ/ rang as one narrow, low resonance:
    // centroid 2.8 kHz, 250 Hz wide, flatness 0.31 in sentences, a peak at
    // 1.25 kHz alone.
    let settings = Preset::Deep.settings().with_engine(Engine::Physical);
    let s = hiss::measure(&settings, "S");
    let sh = hiss::measure(&settings, "SH");
    println!(
        "S: centroid {:.0} Hz, flatness {:.2}, width {:.0} Hz | SH: centroid {:.0} Hz, flatness {:.2}, width {:.0} Hz",
        s.centroid, s.flatness, s.width6, sh.centroid, sh.flatness, sh.width6
    );
    assert!(s.centroid > 5000.0, "S centroid {}", s.centroid);
    assert!(
        (3500.0..5000.0).contains(&sh.centroid),
        "SH centroid {}",
        sh.centroid
    );
    assert!(s.centroid > sh.centroid + 800.0);
    assert!(sh.flatness > 0.3 && sh.width6 >= 400.0, "SH {sh:?}");
    // Its loudest band lies above 1.5 kHz, not at the 1.25 kHz whistle.
    let top = (0..hiss::BANDS)
        .max_by(|&a, &b| sh.bands[a].total_cmp(&sh.bands[b]))
        .unwrap_or(0);
    assert!(
        top as f32 * hiss::BAND_HZ >= 1500.0,
        "SH peaks in band {top}"
    );
}

#[test]
fn a_word_without_a_pause_ends_in_silence() {
    // An utterance that ends without punctuation has no closing pause to
    // let the breath go in: the lungs went on pushing through its last
    // shape, a murmur 31 dB under the word's peak for as long as the voice
    // ran on, and isolated words were heard worse for it (the judge's word
    // set 52.8 → 56.8 % phoneme errors).
    for text in ["[N AH1 M B ER]", "[HH AW1 S]", "[W AO1 T ER]"] {
        let (audio, _) = say(text);
        let rel = levels(&audio);
        let tail = (0.12 * SR) as usize / HOP;
        let loudest = rel[rel.len() - tail..]
            .iter()
            .fold(f32::MIN, |m, &v| m.max(v));
        println!("{text}: tail at most {loudest:.1} dB");
        assert!(loudest < -55.0, "{text}: {loudest:.1} dB after the word");
    }
}
