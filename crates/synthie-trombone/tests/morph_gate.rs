//! A voice morphing between bodies (hybrid roadmap, stage 1), measured as
//! what it is: the neutral morph changes nothing; a held vowel whose body
//! glides from three quarters to five quarters of its size and back makes
//! no click and its folds never stall; its formants follow the tract's
//! length; the four numbers of a morph move together; speech goes on
//! through a morph; a morph laid over the settings is the body a glide
//! arrives at; nothing allocates, and the glide takes the same course at
//! any block length.

#![allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp
)]

#[path = "../examples/support/morph.rs"]
mod morph;
#[path = "../examples/support/spectrum.rs"]
mod spectrum;

use morph::{Rec, SR, click_ratio, glide, held, run, weakest_swing};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use synthie_trombone::stream::{Event, voice_stream};
use synthie_trombone::{Engine, MorphTarget, Phoneme, Preset, Talker, Trombone, VoiceSettings};

struct Counting;

thread_local! {
    static WATCHED: Cell<bool> = const { Cell::new(false) };
}
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call is passed on to the system allocator unchanged; the
// counting reads a thread-local flag and bumps an atomic.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if WATCHED.try_with(Cell::get).unwrap_or(false) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if WATCHED.try_with(Cell::get).unwrap_or(false) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const VOICES: [(&str, Preset, f32); 2] = [
    ("deep", Preset::Deep, 105.0),
    ("bright", Preset::Bright, 210.0),
];

/// F1 and F2 of the voice's shape as it stands, Hz.
fn formants(v: &Trombone) -> (f32, f32) {
    let r = spectrum::formants(v, 4000.0);
    (r[0].hz, r[1].hz)
}

fn physical(preset: Preset) -> VoiceSettings {
    preset.settings().with_engine(Engine::Physical)
}

fn speak(settings: &VoiceSettings, text: &str, mut each: impl FnMut(&mut Talker, f32)) -> Vec<f32> {
    let mut talker = Talker::new(SR, *settings);
    let u = talker.compile(text);
    let frames = ((u.duration() + 0.25) * SR) as usize;
    talker.say_utterance(u);
    let mut out = vec![0.0; frames];
    for (k, block) in out.chunks_mut(64).enumerate() {
        each(&mut talker, (k * 64) as f32 / frames as f32);
        talker.render(block);
    }
    out
}

#[test]
fn the_neutral_morph_changes_nothing() {
    let text = "She sells sea shells by the sea shore. Glue the sheet.";
    for preset in [Preset::Deep, Preset::Bright] {
        let plain = physical(preset);
        assert_eq!(plain.with_morph(&MorphTarget::NEUTRAL), plain);
        let a = speak(&plain, text, |_, _| {});
        let b = speak(&plain.with_morph(&MorphTarget::sized(1.0)), text, |t, _| {
            t.set_morph(MorphTarget::NEUTRAL);
        });
        assert!(a == b, "{preset}: the neutral morph moved the sound");
    }
    // And the classic engine.
    let classic = Preset::Deep.settings();
    let a = speak(&classic, text, |_, _| {});
    let b = speak(&classic, text, |t, _| {
        t.set_morph(MorphTarget::matured(0.0));
    });
    assert!(a == b, "classic: the neutral morph moved the sound");
}

#[test]
fn a_body_glides_from_three_quarters_to_five_quarters_without_a_click_or_a_stall() {
    // Measured: the roughest 10 ms of a held vowel is as rough while the
    // body glides (0.8 to 1.2 times the steady vowel's), the glottal
    // flow's swing never falls under 0.86 of the swing around it, and the
    // pitch stays within 6 % of the pitch asked for.
    for (name, preset, hz) in VOICES {
        for p in [Phoneme::Aa, Phoneme::Iy, Phoneme::Uw, Phoneme::Ae] {
            let mut v = held(&preset.settings(), p, hz);
            let mut steady = Rec::default();
            run(&mut v, 1.0, &mut steady);
            glide(&mut v, MorphTarget::sized(0.75), 0.3, &mut Rec::default());
            let mut rec = Rec::default();
            glide(&mut v, MorphTarget::sized(1.25), 0.0, &mut rec);
            glide(&mut v, MorphTarget::sized(0.75), 0.0, &mut rec);
            glide(&mut v, MorphTarget::NEUTRAL, 0.2, &mut rec);
            assert!(rec.out.iter().all(|x| x.is_finite()), "{name} {p}");
            // There and back at the default speed: 0.5 + 0.5 + 0.25 s.
            let seconds = rec.out.len() as f32 / SR - 0.2;
            assert!((1.2..1.32).contains(&seconds), "{name} {p}: {seconds} s");
            let (still, moving) = (click_ratio(&steady.out), click_ratio(&rec.out));
            assert!(
                moving <= 1.35 * still + 0.2,
                "{name} {p}: a click: {moving:.2} against {still:.2} steady"
            );
            let weakest = weakest_swing(&rec.flow, hz);
            assert!(weakest > 0.6, "{name} {p}: the folds stalled: {weakest:.2}");
            for &f in &rec.pitch {
                assert!((f / hz - 1.0).abs() < 0.1, "{name} {p}: pitch {f:.1} Hz");
            }
        }
    }
}

#[test]
fn the_formants_follow_the_tract() {
    for (name, preset, hz) in VOICES {
        for p in [Phoneme::Aa, Phoneme::Iy, Phoneme::Uw] {
            let mut v = held(&preset.settings(), p, hz);
            let at_rest = formants(&v);
            for scale in [0.75f32, 1.25] {
                glide(&mut v, MorphTarget::sized(scale), 0.1, &mut Rec::default());
                let (f1, f2) = formants(&v);
                // The shape is the one held for the voice's own size: its
                // resonances scale with the tube (the yielding walls keep
                // F1 a little nearer; measured within 4 %).
                assert!(
                    (f1 * scale / at_rest.0 - 1.0).abs() < 0.07,
                    "{name} {p} at {scale}: F1 {f1:.0} against {:.0}",
                    at_rest.0
                );
                assert!(
                    (f2 * scale / at_rest.1 - 1.0).abs() < 0.05,
                    "{name} {p} at {scale}: F2 {f2:.0} against {:.0}",
                    at_rest.1
                );
            }
            // And back where it was.
            glide(&mut v, MorphTarget::NEUTRAL, 0.1, &mut Rec::default());
            let back = formants(&v);
            assert!((back.0 / at_rest.0 - 1.0).abs() < 0.01, "{name} {p}");
            assert!((back.1 / at_rest.1 - 1.0).abs() < 0.01, "{name} {p}");
            assert_eq!(v.tract_length(), preset.settings().tract_length);
        }
    }
}

#[test]
fn all_four_numbers_move_together_and_the_voice_goes_on() {
    let to = MorphTarget {
        tract_scale: 1.2,
        larynx_depth: 0.6,
        fold_thickness: 0.7,
        epilarynx_flare: 1.4,
    };
    let back = MorphTarget {
        tract_scale: 0.8,
        larynx_depth: -0.3,
        fold_thickness: 1.4,
        epilarynx_flare: 0.7,
    };
    for (name, preset, hz) in VOICES {
        let mut v = held(&preset.settings(), Phoneme::Aa, hz);
        let mut steady = Rec::default();
        run(&mut v, 0.5, &mut steady);
        let mut rec = Rec::default();
        v.set_morph(to);
        // Half way there every number is half way.
        while (v.morph().fold_thickness - 0.85).abs() > 0.006 {
            run(&mut v, 0.002, &mut rec);
            assert!(rec.out.len() < SR as usize, "{name}: never half way");
        }
        let m = v.morph();
        assert!((m.tract_scale - 1.1).abs() < 0.01, "{name}: {m:?}");
        assert!((m.larynx_depth - 0.3).abs() < 0.02, "{name}: {m:?}");
        assert!((m.epilarynx_flare - 1.2).abs() < 0.01, "{name}: {m:?}");
        glide(&mut v, to, 0.1, &mut rec);
        assert_eq!(v.morph(), to);
        glide(&mut v, back, 0.1, &mut rec);
        glide(&mut v, MorphTarget::NEUTRAL, 0.1, &mut rec);
        assert!(rec.out.iter().all(|x| x.is_finite()), "{name}");
        let (still, moving) = (click_ratio(&steady.out), click_ratio(&rec.out));
        assert!(
            moving <= 1.35 * still + 0.2,
            "{name}: {moving:.2} against {still:.2}"
        );
        let weakest = weakest_swing(&rec.flow, hz);
        assert!(weakest > 0.6, "{name}: the folds stalled: {weakest:.2}");
    }
}

#[test]
fn a_morph_over_the_settings_is_the_body_a_glide_arrives_at() {
    let target = MorphTarget::matured(1.0).then(MorphTarget::sized(1.1));
    let settings = Preset::Bright.settings();
    let made = held(&settings.with_morph(&target), Phoneme::Ae, 210.0);
    let mut glided = held(&settings, Phoneme::Ae, 210.0);
    glide(&mut glided, target, 0.3, &mut Rec::default());
    assert!((glided.tract_length() / made.tract_length() - 1.0).abs() < 1e-5);
    // The glided voice holds the shape it took at its old size, the made
    // one the shape for its new size: the same body, F2 within 4 %.
    let (a, b) = (formants(&made), formants(&glided));
    assert!((a.1 / b.1 - 1.0).abs() < 0.04, "{a:?} against {b:?}");
    // Spoken anew, the glided voice takes its shapes for the body it is in.
    glided.articulate_phoneme(Phoneme::Ae);
    glided.settle();
    let c = formants(&glided);
    assert!((a.0 / c.0 - 1.0).abs() < 0.02, "{a:?} against {c:?}");
    assert!((a.1 / c.1 - 1.0).abs() < 0.02, "{a:?} against {c:?}");
}

#[test]
fn the_matured_bright_voice_is_a_larger_womans() {
    // Every formant some 7 % lower (5 % of tract, 0.3 cm of larynx), the
    // folds still able to sing a woman's range.
    let plain = Preset::Bright.settings();
    let grown = plain.with_morph(&MorphTarget::matured(1.0));
    assert!((grown.tract_length / plain.tract_length - 1.05).abs() < 1e-5);
    assert_eq!(grown.pitch, plain.pitch);
    let mut share = Vec::new();
    for p in [
        Phoneme::Iy,
        Phoneme::Eh,
        Phoneme::Ae,
        Phoneme::Aa,
        Phoneme::Uw,
        Phoneme::Ah,
    ] {
        let a = formants(&held(&plain, p, 210.0));
        let b = formants(&held(&grown, p, 210.0));
        share.push(b.0 / a.0);
        share.push(b.1 / a.1);
    }
    let mean = share.iter().sum::<f32>() / share.len() as f32;
    assert!(
        (0.91..0.95).contains(&mean),
        "formants at {mean:.3} of the bright voice's"
    );
    let (lo, hi) = held(&grown, Phoneme::Aa, 185.0).fold_range();
    assert!(lo < 120.0 && hi > 320.0, "fold range {lo:.0} to {hi:.0} Hz");
    // And a juvenile the other way.
    let young = plain.with_morph(&MorphTarget::matured(-1.0));
    assert!((young.tract_length / plain.tract_length - 0.95).abs() < 1e-5);
}

#[test]
fn speech_goes_on_through_a_morph() {
    // A sentence of vowels, glides and nasals has no bursts of its own, so
    // its roughest 10 ms shows what the morph adds: nothing.
    let smooth = "We were away a year ago. Why worry Mary?";
    // With stops and hisses the roughest 10 ms is a burst, and bursts
    // differ from body to body (measured 4.7 to 8.1 over bodies of 0.8 to
    // 1.2 that do not change, 5.0 to 10.3 through glides between them).
    let bursts =
        "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background.";
    for preset in [Preset::Deep, Preset::Bright] {
        let settings = physical(preset);
        let glide = |t: &mut Talker, at: f32| t.set_morph(MorphTarget::sized(0.8 + 0.4 * at));
        let plain = speak(&settings, smooth, |_, _| {});
        let morphed = speak(&settings, smooth, glide);
        assert!(morphed.iter().all(|x| x.is_finite()));
        assert_eq!(plain.len(), morphed.len());
        let (a, b) = (morph::rms(&plain), morph::rms(&morphed));
        assert!(
            (0.5..2.0).contains(&(b / a)),
            "{preset}: level {b} against {a}"
        );
        let (still, moving) = (click_ratio(&plain), click_ratio(&morphed));
        assert!(
            moving <= 1.35 * still + 0.2,
            "{preset}: a click in speech: {moving:.2} against {still:.2}"
        );
        let fixed = [0.8, 0.9, 1.0, 1.1, 1.2]
            .map(|s| {
                click_ratio(&speak(
                    &settings.with_morph(&MorphTarget::sized(s)),
                    bursts,
                    |_, _| {},
                ))
            })
            .into_iter()
            .fold(0.0, f32::max);
        let morphed = speak(&settings, bursts, glide);
        assert!(morphed.iter().all(|x| x.is_finite()));
        let moving = click_ratio(&morphed);
        assert!(
            moving <= 1.6 * fixed,
            "{preset}: a click among the bursts: {moving:.2} against {fixed:.2}"
        );
    }
}

#[test]
fn the_glide_takes_the_same_course_at_any_block_length() {
    let target = MorphTarget::matured(1.0).then(MorphTarget::sized(0.8));
    let mut ends = Vec::new();
    for block in [128, 512] {
        let mut v = held(&Preset::Deep.settings(), Phoneme::Aa, 105.0);
        v.set_block(block);
        v.set_morph(target);
        let mut rec = Rec::default();
        run(&mut v, 0.1, &mut rec);
        ends.push((v.morph(), v.tract_length()));
        assert_ne!(v.morph(), target);
        run(&mut v, 0.2, &mut rec);
        assert_eq!(v.morph(), target);
    }
    let (a, b) = (ends[0], ends[1]);
    assert!(
        (a.0.tract_scale - b.0.tract_scale).abs() < 0.012,
        "{a:?} against {b:?}"
    );
    assert!((a.1 - b.1).abs() < 0.012, "{a:?} against {b:?}");
}

#[test]
fn a_morph_never_allocates_and_goes_through_a_stream() {
    let (mut stream, mut player) = voice_stream(1024);
    let mut v = held(&Preset::Deep.settings(), Phoneme::Aa, 105.0);
    let mut block = [0.0f32; 512];
    v.render(&mut block);
    ALLOCATIONS.store(0, Ordering::Relaxed);
    WATCHED.with(|w| w.set(true));
    for k in 0..60 {
        v.set_morph(
            MorphTarget::sized(0.75 + 0.5 * (k % 20) as f32 / 20.0).then(MorphTarget::matured(1.0)),
        );
        v.render(&mut block);
    }
    WATCHED.with(|w| w.set(false));
    let target = MorphTarget::matured(-1.0);
    stream.push(Event::Morph(target)).unwrap();
    for _ in 0..90 {
        WATCHED.with(|w| w.set(true));
        player.render(&mut v, &mut block);
        WATCHED.with(|w| w.set(false));
        assert!(block.iter().all(|x| x.is_finite()));
    }
    assert_eq!(v.morph_target(), target);
    assert_eq!(v.morph(), target);
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
}
