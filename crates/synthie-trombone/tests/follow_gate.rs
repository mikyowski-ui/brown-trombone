//! The gates of the physical engine, held for an utterance that follows a
//! speaker's schedule with everything at its limits: the tenseness of every
//! voiced phone at the edge of its range (by turns its lowest and its
//! highest share), the breath likewise, every vowel's formants moved as far
//! as they may go, and every phone at the shortest the schedule may make it
//! (so the articulators are driven at their fastest) — in the voices a
//! transfer gives: the deep and child presets and the baselines matched to
//! the two reference recordings (a man at 112 Hz; a woman at 204 Hz with a
//! loose voice, tenseness 0.45, the floor of the baseline match: at its
//! own match of 0.29 that voice's medial voiced stops were voiced through
//! 62 % of their closures by the schedule alone, one /b/ through 24 %).
//! No click stands out of the speech, the pulse source never sings a note
//! the folds could, and the voiced stops keep their voice bar.

#![allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use synthie_trombone::speech::{
    FollowOptions, PhoneSpan, PitchMode, Schedule, Speaker, Token, Utterance, phonemize,
};
use synthie_trombone::{Engine, GlottalSource, Phoneme, Preset, Trombone, Tuning, VoiceSettings};

#[path = "../examples/support/stops.rs"]
mod stops;

const SR: f32 = 48_000.0;
const TEXT: &str =
    "Glue the big dog bag to the dark blue background. The juice of lemons makes fine punch.";
/// Voiced stops between vowels, where a closure carries a voice bar (the
/// context of `voiced_closures_carry_a_voice_bar_and_voiceless_ones_do_not`).
/// Each of /b d g/ before /a/, /i/ and /u/.
const MEDIAL: &str = "[AX B AA1] [AX D AA1] [AX G AA1] [AX B IY1] [AX D IY1] [AX G IY1] [AX B UW1] [AX D UW1] [AX G UW1]";
/// The medial voiced stops of [`MEDIAL`]: the stop and which of its kind.
const MEDIAL_STOPS: [(Phoneme, usize); 9] = [
    (Phoneme::B, 0),
    (Phoneme::D, 0),
    (Phoneme::G, 0),
    (Phoneme::B, 1),
    (Phoneme::D, 1),
    (Phoneme::G, 1),
    (Phoneme::B, 2),
    (Phoneme::D, 2),
    (Phoneme::G, 2),
];

/// The voices a transfer gives.
fn voices() -> Vec<(&'static str, VoiceSettings)> {
    let deep = Preset::Deep.settings().with_engine(Engine::Physical);
    let child = Preset::Child.settings().with_engine(Engine::Physical);
    // As `prosody.py transfer` matched them to the reference recordings.
    let woman = VoiceSettings {
        pitch: 160.6,
        tract_length: 0.935,
        tenseness: 0.45,
        ..deep
    }
    .with_profile(r#"{"tuning": {"posterior_gap": 0.119, "lung_pressure": 8.0}}"#)
    .expect("profile");
    let man = VoiceSettings {
        pitch: 83.6,
        tract_length: 1.065,
        tenseness: 0.88,
        ..deep
    };
    vec![
        ("deep", deep),
        ("child", child),
        ("woman", woman),
        ("man", man),
    ]
}

/// `text` followed at its limits: every phone at the shortest share of its
/// rule length, tenseness and breath by turns at the bottom and the top of
/// their ranges, every vowel's F1 and F2 moved by the most allowed, in
/// opposite directions by turns; a pitch gliding around `hz`.
fn at_the_limits(settings: &VoiceSettings, text: &str, hz: f32) -> Utterance {
    at_limits_of(settings, text, hz, "tbfs", 0)
}

/// [`at_the_limits`] for the parts named (`t`enseness, `b`reath,
/// `f`ormants, `s`hort phones), the turns starting at the top
/// (`phase` 0) or at the bottom (1).
fn at_limits_of(
    settings: &VoiceSettings,
    text: &str,
    hz: f32,
    parts: &str,
    phase: usize,
) -> Utterance {
    let tokens = phonemize(text);
    let t = settings.tuning;
    let rule = Utterance::from_tokens(&tokens, &settings.prosody()).phone_spans();
    let n = tokens
        .iter()
        .filter(|t| matches!(t, Token::Phone { .. }))
        .count();
    assert_eq!(n, rule.len());
    let mut at = 0.3;
    let phones = rule
        .iter()
        .enumerate()
        .map(|(k, &(a, b))| {
            let d = (b - a)
                * if parts.contains('s') {
                    t.follow_shortest
                } else {
                    1.0
                };
            let mut s = PhoneSpan::new(None, at, at + d);
            at += d;
            let odd = (k + phase) % 2 == 1;
            if parts.contains('t') {
                s.tenseness = if odd { 0.0 } else { 2.0 };
            }
            if parts.contains('b') {
                s.breath = if odd { 0.0 } else { 2.0 };
            }
            let r = t.follow_formant_range + 0.1;
            if parts.contains('f') {
                s.formants = Some(if odd {
                    (1.0 + r, 1.0 - r)
                } else {
                    (1.0 - r, 1.0 + r)
                });
            }
            s
        })
        .collect();
    let steps = ((at + 1.0) / 0.01) as usize;
    let schedule = Schedule {
        phones,
        pitch_start: 0.0,
        pitch_hop: 0.01,
        // A phrase's fall, from a fifth above the voice's pitch to a
        // fourth below.
        pitch: (0..steps)
            .map(|q| hz * (1.5 - 0.75 * q as f32 / steps as f32))
            .collect(),
        median_hz: hz,
    };
    let mut u = Utterance::from_tokens(&schedule.phrase(&tokens), &settings.prosody());
    let options = FollowOptions::new(PitchMode::Speaker, &t, settings.tract_length);
    let r = u.follow_with(&schedule, &options).expect("follows");
    if parts == "tbfs" && text == TEXT {
        assert!(r.shaped >= 5 && r.sped >= 5, "{r:?}");
        assert!(r.tenseness_limited >= 10 && r.breath_limited >= 10, "{r:?}");
    }
    u
}

#[test]
fn a_followed_utterance_at_its_limits_keeps_the_gates() {
    for (name, settings) in voices() {
        let hz = settings.pitch;
        // No click stands out of the speech (the bound of
        // `stop_releases_are_bursts_not_clicks`: 2.5 vowel peaks).
        let r = stops::record_utterance(&settings, at_the_limits(&settings, TEXT, hz));
        let click = stops::largest_click(&r) / stops::vowel_peak(&r);
        println!("{name}: click {click:.2} vowel peaks");
        assert!(click <= 2.5, "{name}: a click of {click:.2} vowel peaks");

        // The voiced stops between vowels keep their voice bar (the
        // context of `voiced_closures_carry_a_voice_bar_and_voiceless_ones_do_not`,
        // which holds the default voice by the rules to 80 % of the
        // closure voiced). How much of a closure is voiced turns on a few
        // milliseconds near the folds' threshold in the child's and the
        // loose voices: a tenseness share of 1.08 in place of 1.15 before
        // one /g/ took it from 0.88 to 0.47. So the bound is on the mean
        // over /b d g/ before /a i u/ and both turns of the limits: within
        // 0.05 of the same schedule followed without the voice's parts,
        // and no closure voiced through less than a third.
        let (mut plain, mut all, mut least) = (0.0, 0.0, f32::INFINITY);
        for phase in 0..2 {
            let a =
                stops::record_utterance(&settings, at_limits_of(&settings, MEDIAL, hz, "", phase));
            let b = stops::record_utterance(
                &settings,
                at_limits_of(&settings, MEDIAL, hz, "tbfs", phase),
            );
            for (p, k) in MEDIAL_STOPS {
                let v = stops::stop_in(&b, p, k).voiced_share;
                plain += stops::stop_in(&a, p, k).voiced_share;
                all += v;
                least = least.min(v);
            }
        }
        let n = 2.0 * MEDIAL_STOPS.len() as f32;
        let (plain, all) = (plain / n, all / n);
        println!(
            "{name}: medial voiced stops voiced through {plain:.2} of their closures, at the limits {all:.2} (least {least:.2})"
        );
        assert!(
            all >= plain - 0.05,
            "{name}: closures voiced through {all:.2} ({plain:.2} without)"
        );
        assert!(
            least >= 1.0 / 3.0,
            "{name}: a closure voiced through {least:.2}"
        );

        // The pulse source sings no note inside the folds' range (as
        // `the_folds_carry_running_speech`).
        let mut t = Trombone::new(SR);
        settings.apply(&mut t);
        Speaker::prepare(&mut t);
        let mut speaker = Speaker::new();
        speaker.speak(at_the_limits(&settings, TEXT, hz));
        let (mut voiced, mut inside) = (0usize, 0usize);
        while speaker.is_speaking() {
            speaker.process(&mut t);
            match t.gauges().sounding {
                Some(GlottalSource::Pulse) => {
                    voiced += 1;
                    let (lo, hi) = t.fold_range();
                    inside += usize::from((lo..=hi).contains(&t.frequency()));
                }
                Some(GlottalSource::Folds) => voiced += 1,
                None => {}
            }
        }
        assert!(
            voiced as f32 > 0.5 * SR,
            "{name}: voiced only {voiced} samples"
        );
        assert_eq!(
            inside, 0,
            "{name}: the pulse sang {inside} samples inside the folds' range"
        );
    }
}

#[test]
fn the_limits_are_those_of_the_tuning() {
    let t = Tuning::DEFAULT;
    assert!(
        t.follow_tenseness_low >= 0.88,
        "round 8: 12 % less tenseness clicks"
    );
    assert!(
        t.follow_breath_low >= 0.5,
        "under half the breath the folds drop out"
    );
    assert!(t.follow_speed_max <= 1.6 + 1e-6);
}
