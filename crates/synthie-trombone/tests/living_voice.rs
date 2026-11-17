//! The living voice of wave 3 as a whole (`w3-switch`): the switches
//! that turn it on and off, where it breathes and where it does not, and
//! that its creak leaves a statement's last consonants as they were.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

#[path = "../examples/support/breath.rs"]
mod breath;

use breath::{span, take, tone};
use synthie_trombone::sing::{self, SingOptions, SungNote};
use synthie_trombone::speech::{Speaker, Utterance};
use synthie_trombone::stream::{Granularity, voice_stream};
use synthie_trombone::{Engine, Manner, Phoneme, Preset, Trombone, Tuning, VoiceSettings};

/// The deep voice on the physical engine with wave 3 `on` or off.
fn voice(on: bool) -> VoiceSettings {
    let mut s = Preset::Deep.settings().with_engine(Engine::Physical);
    s.tuning = s.tuning.with_living_voice(on);
    s
}

#[test]
fn the_living_voice_switches_five_numbers_and_nothing_else() {
    let on = Tuning::DEFAULT.with_living_voice(true);
    let off = Tuning::DEFAULT.with_living_voice(false);
    assert_eq!(on.anatomy.head_mixdown, 1.0);
    assert_eq!(on.prosody.breath_intake, 5.0);
    assert_eq!(on.prosody.breath_first, 0.0);
    assert_eq!(on.prosody.final_fry, 1.0);
    assert_eq!(on.prosody.fry_time, 0.08);
    assert_eq!(off.anatomy.head_mixdown, 0.0);
    assert_eq!(off.prosody.breath_intake, 0.0);
    assert_eq!(off.prosody.breath_first, 1.0);
    assert_eq!(off.prosody.final_fry, 0.0);
    assert_eq!(off.prosody.fry_time, 0.12);
    // Every other number is the default's.
    let mut back = on;
    back.anatomy.head_mixdown = off.anatomy.head_mixdown;
    back.prosody.breath_intake = off.prosody.breath_intake;
    back.prosody.breath_first = off.prosody.breath_first;
    back.prosody.final_fry = off.prosody.final_fry;
    back.prosody.fry_time = off.prosody.fry_time;
    assert_eq!(back, off);
    assert_eq!(off.with_living_voice(true), on);
}

const ONE: &str = "The birch canoe slid on the smooth planks.";
const TWO: &str =
    "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background.";

fn breaths(u: &Utterance) -> usize {
    u.segments()
        .iter()
        .filter(|s| s.gesture.intake > 0.0)
        .count()
}

#[test]
fn breaths_between_sentences_leave_a_single_sentence_as_it_was() {
    // With `breath_first` 0 a breath is drawn in the pause between two
    // sentences, out of the pause's own time, and none before the first:
    // one sentence is compiled segment for segment as without breaths, and
    // two take as long as without.
    let on = voice(true).prosody();
    let mut quiet = voice(true);
    quiet.tuning.prosody.breath_intake = 0.0;
    let quiet = quiet.prosody();
    let (a, b) = (
        Utterance::from_text(ONE, &on),
        Utterance::from_text(ONE, &quiet),
    );
    assert_eq!(breaths(&a), 0);
    assert_eq!(
        a, b,
        "a single sentence changed with breaths between sentences"
    );
    let (a, b) = (
        Utterance::from_text(TWO, &on),
        Utterance::from_text(TWO, &quiet),
    );
    assert_eq!(breaths(&a), 1, "one breath between the two sentences");
    assert!(
        (a.duration() - b.duration()).abs() < 1e-4,
        "{} s against {} s",
        a.duration(),
        b.duration()
    );
    // Asked for, the breath before the utterance comes as well.
    let mut first = voice(true);
    first.tuning.prosody.breath_first = 1.0;
    assert_eq!(breaths(&Utterance::from_text(TWO, &first.prosody())), 2);
}

#[test]
fn probes_and_songs_take_no_breath_and_no_creak() {
    let on = voice(true);
    // The judge's carriers: bracketed phonemes.
    let probe = Utterance::from_text(
        "[AX0 G EH1 N]. And [S EY1] again. [AA1 B AA0].",
        &on.prosody(),
    );
    assert_eq!(breaths(&probe), 0, "a probe took a breath");
    // A song: its segments are made by hand, with neither.
    let notes: Vec<SungNote> = ["la", "la", "la"]
        .iter()
        .enumerate()
        .map(|(k, s)| SungNote {
            start: 0.6 * k as f64,
            duration: 0.5,
            key: 50.0 + k as f32,
            syllable: Some((*s).to_string()),
        })
        .collect();
    let song = sing::compile(&notes, &on, &SingOptions::default());
    assert!(song.duration() > 1.0, "{} s", song.duration());
    assert!(
        song.segments()
            .iter()
            .all(|s| s.gesture.intake == 0.0 && s.gesture.creak == 0.0),
        "a song took a breath or creaked"
    );
}

/// Level of the hiss in the last segment of phoneme `p` in a take of `text`
/// spoken with `settings`: the mean magnitude from 3 to 8 kHz, dB.
fn hiss(settings: &VoiceSettings, text: &str, p: Phoneme) -> f32 {
    let spoken = take(settings, text);
    let last = spoken
        .phonemes
        .iter()
        .rposition(|&q| q == p)
        .expect("the phoneme");
    let (from, to) = span(&spoken, last);
    let x = &spoken.out[from..to];
    let bins: Vec<f32> = (0..=20)
        .map(|i| tone(x, 3000.0 + 250.0 * i as f32))
        .collect();
    let mean = bins.iter().sum::<f32>() / bins.len() as f32 / x.len() as f32;
    20.0 * mean.max(1e-12).log10()
}

#[test]
fn a_statement_s_last_voiced_fricative_hisses_in_creak_as_without() {
    // Pressed together, the folds pass too little air for a fricative: when
    // the creak ran on through the voiced consonants after the last
    // sonorant, a final /ʒ/ lost 30 dB of its jet noise, a final /dʒ/ all
    // of it, and the /z/ of "rose" 8.5 dB of its hiss (here: under 1 dB). The creak ends with the last sonorant and these are
    // spoken in modal voice.
    let mut creak = voice(false);
    creak.tuning.prosody.final_fry = 1.0;
    let plain = voice(false);
    for (text, p) in [
        ("The sun rose.", Phoneme::Z),
        ("We have five.", Phoneme::V),
        ("She sold the shells.", Phoneme::Z),
        ("He read the books.", Phoneme::S),
    ] {
        let (on, off) = (hiss(&creak, text, p), hiss(&plain, text, p));
        println!("{text} /{p:?}/: hiss {on:+.1} dB in creak, {off:+.1} dB without");
        assert!(on > off - 3.0, "{text}: {on:+.1} dB against {off:+.1}");
        let u = Utterance::from_text(text, &creak.prosody());
        let last = u
            .segments()
            .iter()
            .rposition(|s| {
                s.voiced
                    && matches!(
                        s.phoneme.manner(),
                        Manner::Vowel | Manner::Diphthong | Manner::Nasal | Manner::Approximant
                    )
            })
            .expect("a sonorant");
        assert!(u.segments()[last].gesture.creak > 0.0, "{text}: no creak");
        for s in &u.segments()[last + 1..] {
            if s.voiced {
                assert_eq!(s.gesture.creak, 0.0, "{text}: {:?} in creak", s.phoneme);
            }
        }
    }
}

/// Breaths (runs of the lungs drawing more than half a cmH2O) and the
/// seconds from the first sound to the last, of a voice spoken sample by
/// sample with `step`.
fn breaths_and_length(mut step: impl FnMut(&mut Trombone) -> Option<f32>) -> (usize, f32) {
    let mut voice = Trombone::new(48_000.0);
    voice.set_engine(Engine::Physical);
    let (mut count, mut drawing) = (0, false);
    let (mut first, mut last, mut n) = (None, 0, 0usize);
    while let Some(x) = step(&mut voice) {
        let lungs = voice.gauges().lung_pressure;
        if lungs < -0.5 && !drawing {
            count += 1;
        }
        drawing = lungs < -0.5;
        if x.abs() > 1e-4 {
            first.get_or_insert(n);
            last = n;
        }
        n += 1;
        assert!(n < 30 * 48_000, "never fell silent");
    }
    (count, (last - first.unwrap_or(0)) as f32 / 48_000.0)
}

#[test]
fn streamed_sentences_breathe_between_them_as_the_text_whole() {
    // The stream compiles a sentence at a time; with breaths between
    // sentences and none before the utterance each sentence after the
    // first opens with the breath the text's pause would have taken, out of
    // the pause's own time.
    let prosody = voice(true).prosody();
    let whole = {
        let u = Utterance::from_text(TWO, &prosody);
        let mut speaker = Speaker::new();
        let mut started = false;
        breaths_and_length(|v| {
            if !started {
                Speaker::prepare(v);
                speaker.speak(u.clone());
                started = true;
            }
            speaker.is_speaking().then(|| speaker.process(v))
        })
    };
    let streamed = {
        let (mut stream, mut player) = voice_stream(1024);
        stream.set_prosody(prosody);
        stream.set_granularity(Granularity::Sentence);
        let mut fed = false;
        breaths_and_length(|v| {
            if !fed {
                for word in TWO.split_inclusive(' ') {
                    stream.push_text(word);
                }
                stream.flush_text();
                fed = true;
                return Some(player.process(v));
            }
            (!player.is_idle()).then(|| player.process(v))
        })
    };
    println!(
        "the text whole: {} breaths, {:.3} s; streamed: {} breaths, {:.3} s",
        whole.0, whole.1, streamed.0, streamed.1
    );
    assert_eq!(whole.0, 1);
    assert_eq!(streamed.0, 1, "the stream's breaths");
    assert!(
        (streamed.1 - whole.1).abs() < 0.02,
        "{streamed:?} against {whole:?}"
    );
}
