//! Gates of the living voice's breath and creak (wave 3, stage 3A), on the
//! physical engine: the air goes in while a breath is drawn and out once
//! the voice speaks, the lungs reach the pressure asked for, a breath is
//! heard at the level speech breaths have, the turn from breathing in to
//! speaking does not click, statements end in creak (alternating strong
//! and weak pulses) and questions do not, and all of it is off by default.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

#[path = "../examples/support/breath.rs"]
mod breath;

use breath::{SR, breaths, creak, db, largest_step, phrase_ends, phrase_vowels, rms, take};
use synthie_trombone::speech::Utterance;
use synthie_trombone::{Engine, Phoneme, Preset, Trombone, Tuning, VoiceSettings};

const TEXT: &str =
    "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background.";

/// Lungs drawing this deep before a long phrase, cmH2O: the deepest the
/// owner asked for (3 to 5).
const DEPTH: f32 = 5.0;

/// The owner's voice with breaths and creak on.
fn living() -> VoiceSettings {
    let mut s = Preset::Deep.settings().with_engine(Engine::Physical);
    s.tuning.prosody.breath_intake = DEPTH;
    s.tuning.prosody.final_fry = 1.0;
    s
}

fn plain() -> VoiceSettings {
    Preset::Deep.settings().with_engine(Engine::Physical)
}

#[test]
fn breaths_and_creak_are_off_by_default() {
    assert_eq!(Tuning::DEFAULT.prosody.breath_intake, 0.0);
    assert_eq!(Tuning::DEFAULT.prosody.final_fry, 0.0);
    let u = Utterance::from_text(TEXT, &plain().prosody());
    assert!(
        u.segments()
            .iter()
            .all(|s| s.gesture.intake == 0.0 && s.gesture.creak == 0.0),
        "a breath or a creak in a default utterance"
    );
}

#[test]
fn probes_and_hand_made_segments_take_no_breath() {
    let prosody = living().prosody();
    let breathes = |u: &Utterance| u.segments().iter().any(|s| s.gesture.intake > 0.0);
    assert!(breathes(&Utterance::from_text(TEXT, &prosody)));
    // The judge's carriers: bracketed phonemes, a probe of the sounds.
    let probe = Utterance::from_text("[S EY1] word [AX0 G EH1 N].", &prosody);
    assert!(!breathes(&probe), "a probe took a breath");
}

#[test]
fn a_breath_adds_its_time_before_the_utterance_and_none_between_sentences() {
    // Before the utterance the breath and its turn are added; between the
    // sentences they are taken out of the pause, which keeps its length.
    let quiet = Utterance::from_text(TEXT, &plain().prosody());
    let breathing = Utterance::from_text(TEXT, &living().prosody());
    let t = living().tuning.prosody.breath_time;
    let added = breathing.duration() - quiet.duration();
    assert!(
        (added - (t + 0.1)).abs() < 1e-4,
        "{added} s added, not the breath and its turn"
    );
    let starts = |u: &Utterance| -> Vec<f32> { u.phone_spans().iter().map(|s| s.0).collect() };
    let (a, b) = (starts(&quiet), starts(&breathing));
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        assert!((y - x - added).abs() < 1e-3, "a phone moved by {} s", y - x);
    }
}

#[test]
fn the_air_goes_in_during_a_breath_and_out_after_it() {
    let t = take(&living(), TEXT);
    let found = breaths(&t);
    // One before the utterance, one between the sentences.
    assert_eq!(found.len(), 2, "{found:?}");
    for &(a, b) in &found {
        let low = t.lung[a..b].iter().copied().fold(0.0f32, f32::min);
        let deep: Vec<usize> = (a..b).filter(|&k| t.lung[k] < 0.5 * low).collect();
        let inflow = deep.iter().map(|&k| t.flow[k]).sum::<f32>() / deep.len() as f32;
        println!(
            "breath at {:.2} s: lungs {low:.2} cmH2O, glottal flow {inflow:.0} cm3/s",
            a as f32 / SR
        );
        assert!(inflow < -200.0, "{inflow} cm3/s through the glottis");
        // The voice after it: the air goes out.
        let after = phrase_vowels(&t, b);
        assert!(!after.is_empty());
        let (first, _) = breath::vowel_after(&t, b).expect("a vowel after the breath");
        let out = t.flow[first..first + (0.1 * SR) as usize]
            .iter()
            .sum::<f32>()
            / (0.1 * SR);
        assert!(out > 50.0, "{out} cm3/s in the vowel after the breath");
    }
}

#[test]
fn the_lungs_reach_the_pressure_asked() {
    let mut v = Trombone::new(SR);
    v.set_engine(Engine::Physical);
    v.articulate_phoneme(Phoneme::Aa);
    v.settle();
    v.set_inhalation(4.0);
    let mut buf = vec![0.0; (0.25 * SR) as usize];
    v.render(&mut buf);
    let g = v.gauges();
    println!(
        "drawing 4: lungs {:.3}, below the folds {:.2} cmH2O, flow {:.0} cm3/s",
        g.lung_pressure, g.subglottal_pressure, g.glottal_flow
    );
    assert!((g.lung_pressure + 4.0).abs() < 0.05, "{}", g.lung_pressure);
    assert!(g.subglottal_pressure < -1.0);
    assert!(g.glottal_flow < -100.0);
    // Let go and speak: the lungs push again and the air goes out.
    v.set_inhalation(0.0);
    v.set_frequency(100.0);
    v.note_on();
    v.render(&mut buf);
    let mut flow = 0.0;
    for s in buf.iter_mut() {
        *s = v.process();
        flow += v.gauges().glottal_flow;
    }
    let g = v.gauges();
    assert!(g.lung_pressure > 4.0, "{}", g.lung_pressure);
    assert!(flow / buf.len() as f32 > 50.0);
    assert!(g.fold_pitch > 80.0, "the folds at {} Hz", g.fold_pitch);
}

#[test]
fn a_breath_sits_25_to_40_db_under_the_speech() {
    // Speech breaths are heard some 20 to 40 dB under the speech around
    // them; held to 25 to 40 dB under the vowels of the phrase after the
    // breath, its level while the lungs draw more than half their depth.
    // Measured: 32 to 34 dB under at 5 cmH2O (36 at 4).
    let t = take(&living(), TEXT);
    for (a, b) in breaths(&t) {
        let low = t.lung[a..b].iter().copied().fold(0.0f32, f32::min);
        let deep: Vec<f32> = (a..b)
            .filter(|&k| t.lung[k] < 0.5 * low)
            .map(|k| t.out[k])
            .collect();
        let level = db(rms(&deep)) - db(rms(&phrase_vowels(&t, b)));
        println!(
            "breath at {:.2} s: {level:+.1} dB re the vowels after it",
            a as f32 / SR
        );
        assert!((-40.0..=-25.0).contains(&level), "{level:+.1} dB");
    }
}

#[test]
fn the_turn_from_breathing_in_to_speaking_does_not_click() {
    // The largest step between two samples within 40 ms of the lungs
    // crossing over from drawing to pushing, against the largest step in
    // the vowels of the phrase after. Measured: 0.06 to 0.11.
    let t = take(&living(), TEXT);
    for (_, b) in breaths(&t) {
        let turn = (b..t.lung.len()).find(|&k| t.lung[k] >= 0.0).unwrap_or(b);
        let w = (0.04 * SR) as usize;
        let step = largest_step(&t.out[turn - w..turn + w]);
        let vowel = largest_step(&phrase_vowels(&t, b));
        println!(
            "turn at {:.2} s: {:.2} x the vowels' largest step",
            turn as f32 / SR,
            step / vowel
        );
        assert!(step <= 0.5 * vowel, "{:.2}", step / vowel);
    }
}

/// The last 100 ms of the voice of each phrase of `text`: (alternation of
/// the pulse heights, energy at half the pitch against the pitch's, dB).
fn ends(settings: &VoiceSettings, text: &str) -> Vec<(f32, f32)> {
    let t = take(settings, text);
    phrase_ends(&t, 0.1)
        .iter()
        .map(|&(a, b, _)| {
            let (_, alternation, _, half) = creak(&t.out[a..b], &t.flow[a..b]);
            (alternation, half)
        })
        .collect()
}

#[test]
fn statements_end_in_creak_and_questions_do_not() {
    // In creak the folds alternate strong and weak pulses, and the voice
    // has energy at half its pitch. Measured over the last 100 ms of the
    // voice, with `final_fry` 1: the pulse heights alternate by 0.54 to
    // 0.57 of their mean and the half-pitch stands 0.3 to 9.5 dB under the
    // pitch on statements; 0.02 to 0.22 and 15 to 35 dB under without it;
    // the question 0.02 either way.
    let texts = ["She sells sea shells by the sea shore.", "It is mine."];
    for text in texts {
        for (on, off) in ends(&living(), text).iter().zip(ends(&plain(), text)) {
            println!(
                "{text}: alternation {:.2} ({:.2} off), half-pitch {:+.1} dB ({:+.1} off)",
                on.0, off.0, on.1, off.1
            );
            assert!(on.0 >= 0.35 && on.1 >= -15.0, "{text}: no creak, {on:?}");
            assert!(off.0 <= 0.25, "{text}: creak without asking, {off:?}");
        }
    }
    for (alternation, half) in ends(&living(), "Where is the car?") {
        println!("question: alternation {alternation:.2}, half-pitch {half:+.1} dB");
        assert!(alternation <= 0.1, "a question in creak: {alternation}");
    }
}

#[test]
fn a_breath_taken_with_the_api_comes_and_goes() {
    for block in [512u32, 128] {
        let mut v = Trombone::new(SR);
        v.set_engine(Engine::Physical);
        v.set_block(block);
        v.articulate_phoneme(Phoneme::Ax);
        v.settle();
        assert!(v.breathe_in(4.0, 0.15).is_some());
        let mut lowest = 0.0f32;
        let mut inflow = 0.0f32;
        for _ in 0..(0.15 * SR) as usize {
            v.process();
            lowest = lowest.min(v.gauges().lung_pressure);
            inflow = inflow.min(v.gauges().glottal_flow);
        }
        for _ in 0..(0.15 * SR) as usize {
            v.process();
        }
        let after = v.gauges().lung_pressure;
        println!(
            "block {block}: lungs down to {lowest:.2} cmH2O, flow {inflow:.0} cm3/s, {after:.3} 0.15 s after"
        );
        assert!((-4.0..=-3.0).contains(&lowest), "{lowest}");
        assert!(inflow < -100.0);
        assert!(after > -0.1, "still drawing at {after}");
        assert!(v.inhalation() == 0.0);
    }
}
