//! Gate for mimic mode: a voice heard through the [`Listener`] is copied by
//! a second voice, vowel for vowel and hiss for hiss.
//!
//! ```sh
//! cargo test -p synthie-trombone --release --test mimic_gate -- --nocapture
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

#[path = "../examples/support/spectrum.rs"]
mod spectrum;
#[path = "../examples/support/survey.rs"]
mod survey;

use synthie_trombone::listen::Listener;
use synthie_trombone::mimic::Mimic;
use synthie_trombone::{Engine, Phoneme, Trombone};

const SR: f32 = 48_000.0;

/// The voice that is listened to: a steady sung note.
fn singer(hz: f32) -> Trombone {
    let mut t = Trombone::new(SR);
    t.set_engine(Engine::Physical);
    t.set_vibrato(0.0, 0.0);
    t.set_jitter(0.0);
    t.set_frequency(hz);
    t.articulate_phoneme(Phoneme::Ax);
    t.settle();
    t.note_on();
    t
}

/// The voice that copies. Its articulators are quick: the person it
/// follows has already taken the time to move theirs.
fn copier() -> Trombone {
    let mut t = Trombone::new(SR);
    t.set_engine(Engine::Physical);
    t.set_vibrato(0.0, 0.0);
    t.set_jitter(0.0);
    t.set_articulation_speed(8.0);
    t
}

fn off(got: f32, want: f32) -> f32 {
    100.0 * (got / want - 1.0)
}

fn within(f: Option<[f32; 3]>, truth: [f32; 3], percent: f32) -> bool {
    f.is_some_and(|f| off(f[0], truth[0]).abs() <= percent && off(f[1], truth[1]).abs() <= percent)
}

#[test]
fn a_sung_vowel_sequence_is_followed() {
    // A voice sings six vowels on one note, changing abruptly every half
    // second. The copy, fed only the sound, must make each vowel with F1
    // and F2 within 10% of the singer's, and be there within 60 ms of the
    // change: the shape it is told to take, and its own tract.
    let vowels = [
        Phoneme::Iy,
        Phoneme::Aa,
        Phoneme::Uw,
        Phoneme::Eh,
        Phoneme::Ao,
        Phoneme::Ae,
    ];
    let mut a = singer(110.0);
    let mut b = copier();
    let mut listener = Listener::new(SR);
    let mut mimic = Mimic::new(1.0);
    for _ in 0..(0.3 * SR) as usize {
        if let Some(h) = listener.push(a.process()) {
            mimic.follow(&h).apply(&mut b);
        }
        b.process();
    }
    let mut failures = Vec::new();
    for p in vowels {
        // The listener hears the physical /i/ and /u/ (F1 348 and 386 Hz,
        // Hillenbrand's, between the 3rd and 4th harmonic of 110 Hz) with
        // F1 15-17 % high, and /ɔ/ 7 % high (see `listen_gate`), and the
        // copy makes what it hears: those are held to 20 %.
        let tolerance = if matches!(p, Phoneme::Iy | Phoneme::Uw | Phoneme::Ao) {
            20.0
        } else {
            10.0
        };
        a.articulate_phoneme(p);
        a.settle();
        let truth = survey::formants(&a).expect("the singer's formants");
        let (mut told, mut there) = (None, None);
        let every = (0.0025 * SR) as usize;
        for n in 0..(0.5 * SR) as usize {
            if let Some(h) = listener.push(a.process()) {
                mimic.follow(&h).apply(&mut b);
            }
            b.process();
            if n % every == 0 && n < (0.2 * SR) as usize {
                let ms = 1000.0 * n as f32 / SR;
                // The time from which the copy stays within 10%.
                let asked = mimic.shape().and_then(|s| survey::measure(s, 1.0));
                if within(asked, truth, tolerance) {
                    told.get_or_insert(ms);
                } else {
                    told = None;
                }
                if within(survey::formants(&b), truth, tolerance) {
                    there.get_or_insert(ms);
                } else {
                    there = None;
                }
            }
        }
        let got = survey::formants(&b).expect("the copy's formants");
        let (e1, e2) = (off(got[0], truth[0]), off(got[1], truth[1]));
        println!(
            "{p}: singer {:.0}/{:.0} Hz, copy {:.0}/{:.0} Hz ({e1:+.1}%, {e2:+.1}%); told after {told:?} ms, tract there after {there:?} ms; voiced {}, pitch {:.1} Hz",
            truth[0],
            truth[1],
            got[0],
            got[1],
            b.is_voiced(),
            b.target_frequency()
        );
        if e1.abs() > tolerance || e2.abs() > tolerance {
            failures.push(format!("{p}: F1 {e1:+.1}%, F2 {e2:+.1}%"));
        }
        if told.is_none_or(|ms| ms > 60.0) {
            failures.push(format!("{p}: told after {told:?} ms"));
        }
        if there.is_none_or(|ms| ms > 60.0) {
            failures.push(format!("{p}: tract there after {there:?} ms"));
        }
        assert!(b.is_voiced());
        assert!((b.target_frequency() / 110.0 - 1.0).abs() < 0.03);
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn hiss_and_silence_are_followed() {
    // An /s/ and a /sh/ are copied as a gap at the right place, with the
    // voice off; silence stops everything.
    let mut b = copier();
    let mut listener = Listener::new(SR);
    let mut mimic = Mimic::new(1.0);
    let mut places = Vec::new();
    for p in [Phoneme::S, Phoneme::Sh] {
        let mut a = Trombone::new(SR);
        a.set_engine(Engine::Physical);
        a.articulate_phoneme(Phoneme::Aa);
        a.settle();
        a.articulate_phoneme(p);
        for _ in 0..(0.4 * SR) as usize {
            if let Some(h) = listener.push(a.process()) {
                mimic.follow(&h).apply(&mut b);
            }
            b.process();
        }
        let gap = b.constriction(synthie_trombone::SLOT_MAIN);
        println!("{p}: copied as {gap:?}, voiced {}", b.is_voiced());
        assert!(!b.is_voiced());
        assert_eq!(gap, p.articulation().constriction, "{p}");
        places.push(gap);
    }
    for _ in 0..(0.3 * SR) as usize {
        if let Some(h) = listener.push(0.0) {
            mimic.follow(&h).apply(&mut b);
        }
        b.process();
    }
    assert!(!b.is_voiced());
    assert_eq!(b.constriction(synthie_trombone::SLOT_MAIN), None);
}
