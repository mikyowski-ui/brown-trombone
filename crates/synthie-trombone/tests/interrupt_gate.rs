//! A voice interrupted or knocked in the middle of a word (hybrid
//! roadmap, stage 4), measured against the same voice left alone: it
//! answers within 10 ms; the sound is the same to the sample until it
//! does (nothing in the tube is reset) and takes no step from the breaking
//! in; what is said next comes out of a living voice; a knock swells,
//! empties or muffles the voice and dies away; nothing allocates, and a
//! stream carries both.

#![allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp
)]

#[path = "../examples/support/interrupt.rs"]
mod interrupt;

use interrupt::{Run, SR, cut, db, deep, peak, points, run, step};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use synthie_trombone::stream::{Event, When, voice_stream};
use synthie_trombone::{Engine, Preset, Talker, Trombone, Vocalization};

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

const TEXT: &str = "The birch canoe slid on the smooth planks.";

/// A voice with the deep voice's settings, and a stream that compiles text
/// for it.
fn streamed() -> (
    synthie_trombone::stream::VoiceStream<'static>,
    synthie_trombone::stream::VoicePlayer<'static>,
    Trombone,
) {
    let (mut stream, player) = voice_stream(1024);
    let mut voice = Trombone::new(SR);
    deep().apply(&mut voice);
    stream.set_prosody(deep().prosody());
    (stream, player, voice)
}

/// Level of a stretch, dB.
fn level(x: &[f32]) -> f32 {
    db((x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt())
}

#[test]
fn an_interruption_is_answered_within_ten_milliseconds_and_takes_no_step() {
    // Measured (examples/interrupt.rs), the deep voice broken in on in the
    // middle of a vowel, a stop's closure, an /s/ and a nasal, by every
    // vocalization: the larynx or the lungs do something else after 1.3 to
    // 3.5 ms, the sound after 1.1 to 2.4 ms; the largest step of the
    // difference to the voice left alone over the first 3 ms is 45 to 51 dB
    // under the speech's peak in the vowel, 50 to 60 in the closure and 32
    // to 40 in the nasal (whose closure opens into a mouth with a voice in
    // it); in the /s/ the two are different noises. Before the lungs and
    // the folds were made to land softly the vowel's was 30 dB under: a
    // step of 3 % of the peak where the lungs were told to stop.
    for preset in [Preset::Deep, Preset::Bright] {
        let settings = preset.settings().with_engine(Engine::Physical);
        let plain = run(&settings, TEXT, 3.0, None, |_| {});
        let where_ = points(&settings, TEXT);
        assert_eq!(where_.len(), 4, "{preset}: {where_:?}");
        for (name, at) in where_ {
            for v in Vocalization::ALL {
                let with = run(&settings, TEXT, 3.0, Some(at), |t| {
                    assert!(t.interrupt_with(v).is_some());
                });
                let c = cut(&plain, &with, at);
                let what = format!("{preset}, {name}, {}: {c:?}", v.name());
                assert!(with.out.iter().all(|x| x.is_finite()), "{what}");
                // The speech has stopped and the vocalization is being made
                // from the sample of the call.
                assert!(c.stopped && c.vocalizing, "{what}");
                // Within 10 ms the body and the sound are on another course.
                assert!(c.body_ms < 10.0 && c.sound_ms < 10.0, "{what}");
                // Nothing in the tube was reset: through the first
                // millisecond (in which the next control block may already
                // come) the sound is the voice's own to within 40 dB under
                // the speech's peak; a tube emptied would differ by the
                // whole sound.
                let same = (0.001 * SR) as usize;
                let apart = with.out[at..at + same]
                    .iter()
                    .zip(&plain.out[at..at + same])
                    .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
                // (In an /s/ the two runs are different noises from the
                // next control block on.)
                // (In an /s/ the two runs are different noises from the
                // next control block on, and a closure that opens lets its
                // air out: the vowel shows the cut alone.)
                assert!(
                    !name.starts_with("vowel") || apart <= 1e-2 * peak(&plain.out),
                    "{what}: {apart} apart in the first millisecond"
                );
                // No step from the breaking in: under -40 dB of the
                // speech's peak in a vowel, where the cut is all there is;
                // under -25 dB where a closure opens with the voice or the
                // pressure behind it (a stop, a nasal: 29 to 60 dB under),
                // which is that closure's own release and far under one in
                // speech (the last check). (In an /s/ the two runs are
                // different noises.)
                if name.starts_with("vowel") {
                    assert!(c.step_db < -40.0, "{what}");
                } else if !name.starts_with("hiss") {
                    assert!(c.step_db < -25.0, "{what}");
                }
                // And nothing after it steeper than the speech's own
                // steepest moment.
                assert!(c.after_step <= 1.0, "{what}");
            }
        }
    }
}

#[test]
fn the_answer_comes_at_the_next_control_block_whatever_its_length() {
    let settings = deep();
    for block in [64u32, 128, 512] {
        let speak = |at: Option<usize>| {
            let mut talker = Talker::new(SR, settings);
            talker.voice_mut().set_block(block);
            talker.say(TEXT);
            let mut out = Vec::new();
            let mut area = Vec::new();
            for k in 0..(1.0 * SR) as usize {
                if Some(k) == at {
                    talker.interrupt_with(Vocalization::Cough);
                }
                out.push(talker.process());
                area.push(talker.voice_mut().gauges().glottal_area);
            }
            (out, area)
        };
        let at = (0.25 * SR) as usize;
        let (plain, plain_area) = speak(None);
        let (with, with_area) = speak(Some(at));
        let first = (at..with.len())
            .position(|k| (with_area[k] - plain_area[k]).abs() > 0.002 || with[k] != plain[k])
            .unwrap();
        // A control block is that many of the model's samples, at most
        // 10.7 ms; the larynx answers at the next one.
        let ms = 1000.0 * first as f32 / SR;
        let block_ms = 1000.0 * block as f32 / SR;
        assert!(
            ms <= block_ms + 0.5,
            "block {block}: answered after {ms:.2} ms"
        );
        assert!(with.iter().all(|x| x.is_finite()));
    }
}

#[test]
fn what_is_said_next_comes_out_of_a_living_voice() {
    let settings = deep();
    let next = "Mary may win more money.";
    let fresh = Talker::new(SR, settings).bake(next);
    for (name, at) in points(&settings, TEXT) {
        for v in [Vocalization::Cough, Vocalization::Sigh, Vocalization::Laugh] {
            let mut talker = Talker::new(SR, settings);
            talker.say(TEXT);
            talker.say("This is never said.");
            let (mut out, mut lung, mut flow) = (Vec::new(), Vec::new(), Vec::new());
            let mut began = None;
            let mut seconds = 0.0;
            for k in 0..(9.0 * SR) as usize {
                if k == at {
                    seconds = talker.interrupt_with(v).unwrap();
                    talker.say(next);
                }
                out.push(talker.process());
                let voice = talker.voice_mut();
                let g = voice.gauges();
                lung.push(g.lung_pressure);
                flow.push(g.glottal_flow);
                if k > at && began.is_none() && !voice.is_vocalizing() {
                    began = Some(k);
                }
                if k > at && !talker.is_speaking() && began.is_some() {
                    break;
                }
            }
            let what = format!("{name}, {}", v.name());
            assert!(out.iter().all(|x| x.is_finite()), "{what}");
            // The vocalization was made, whole: its length, and what it is
            // (a cough's pressure behind the shut folds, air drawn in for a
            // sigh and a laugh).
            let began = began.unwrap();
            let took = (began - at) as f32 / SR;
            assert!(
                (took - seconds).abs() < 0.02,
                "{what}: {took} s for {seconds} s"
            );
            let most = lung[at..began].iter().fold(0.0f32, |m, &p| m.max(p));
            let least = flow[at..began].iter().fold(0.0f32, |m, &f| m.min(f));
            match v {
                Vocalization::Cough => assert!(most > 30.0, "{what}: {most:.1} cmH2O"),
                _ => assert!(least < -100.0, "{what}: flow {least:.0} cm³/s"),
            }
            // Then the next sentence, and not the one that was queued: as
            // long and as loud as from a voice that was never broken off.
            let said = &out[began..];
            let seconds = said.len() as f32 / SR;
            let expected = fresh.len() as f32 / SR - 0.25;
            assert!(
                (seconds - expected).abs() < 0.15,
                "{what}: {seconds} s against {expected}"
            );
            let (a, b) = (level(said), level(&fresh[..fresh.len() - 12_000]));
            assert!((a - b).abs() < 3.0, "{what}: {a:.1} dB against {b:.1}");
            // And it starts without a step of its own.
            assert!(step(&said[..4800]) <= step(&fresh), "{what}");
        }
    }
}

#[test]
fn a_knock_swells_empties_or_muffles_the_voice_and_dies_away() {
    // Measured: a push of 8 cmH2O raises the lungs by 3.7 to 4.5 (they
    // follow within 30 ms while it dies in 80) and the sound by 3.3 to
    // 5.7 dB; a gasp of 15 takes them 7.3 to 8.3 under and turns the air
    // inward; a jaw knocked shut brings the lips half way together.
    let settings = deep();
    let text = "We were away a year ago. Mary may win more money.";
    let plain = run(&settings, text, 4.0, None, |_| {});
    let top = peak(&plain.out);
    let at = points(&settings, text)[0].1;
    let window = (0.12 * SR) as usize;
    let later = (0.8 * SR) as usize;
    let most = |r: &Run, of: fn(&Run) -> &Vec<f32>| {
        of(r)[at..at + window]
            .iter()
            .zip(&of(&plain)[at..at + window])
            .map(|(a, b)| a - b)
            .fold(0.0f32, |m, d| if d.abs() > m.abs() { d } else { m })
    };
    let onset_step = |r: &Run| {
        let d: Vec<f32> = r.out[at..at + 144]
            .iter()
            .zip(&plain.out[at..at + 144])
            .map(|(a, b)| a - b)
            .collect();
        db(step(&d) / top)
    };

    // A bump to the chest.
    let bump = run(&settings, text, 4.0, Some(at), |t| t.perturb(8.0, 0.0));
    let rise = most(&bump, |r| &r.lung);
    assert!((3.0..8.0).contains(&rise), "lungs {rise:+.1} cmH2O");
    let louder = level(&bump.out[at..at + window]) - level(&plain.out[at..at + window]);
    assert!(louder > 2.0, "a bump made the voice {louder:+.1} dB");
    assert!(
        onset_step(&bump) < -40.0,
        "a bump clicked: {:.1} dB",
        onset_step(&bump)
    );
    // It dies away: 0.8 s later the lungs are the speech's again.
    let left = (bump.lung[at + later] - plain.lung[at + later]).abs();
    assert!(left < 0.2, "{left} cmH2O left of a bump");
    assert!(bump.speaking[at + window], "the speech went on");

    // A gasp.
    let gasp = run(&settings, text, 4.0, Some(at), |t| t.perturb(-15.0, 0.0));
    assert!(most(&gasp, |r| &r.lung) < -5.0);
    let inward = gasp.flow[at..at + window]
        .iter()
        .fold(0.0f32, |m, &f| m.min(f));
    assert!(inward < -20.0, "a gasp's flow {inward:.0} cm³/s");
    assert!(plain.flow[at..at + window].iter().all(|&f| f > -5.0));
    assert!(
        onset_step(&gasp) < -40.0,
        "a gasp clicked: {:.1} dB",
        onset_step(&gasp)
    );
    assert!((gasp.lung[at + later] - plain.lung[at + later]).abs() < 0.2);

    // The jaw knocked shut.
    let jaw = run(&settings, text, 4.0, Some(at), |t| t.perturb(0.0, 1.0));
    let shut = most(&jaw, |r| &r.lips);
    assert!(shut > 0.15, "the lips came {shut:.2} together");
    let muffled = level(&jaw.out[at..at + window]) - level(&plain.out[at..at + window]);
    assert!(
        muffled.abs() > 0.3,
        "the jaw changed the sound by {muffled:+.1} dB"
    );
    assert!(
        onset_step(&jaw) < -40.0,
        "the jaw clicked: {:.1} dB",
        onset_step(&jaw)
    );
    assert!((jaw.lips[at + later] - plain.lips[at + later]).abs() < 0.05);
    // The lungs were left alone by it, and the sound goes on.
    assert!(most(&jaw, |r| &r.lung).abs() < 0.5);
    for r in [&bump, &gasp, &jaw] {
        assert!(r.out.iter().all(|x| x.is_finite()));
        let after = level(&r.out[at + later..]) - level(&plain.out[at + later..]);
        assert!(after.abs() < 1.5, "the voice came back {after:+.1} dB off");
    }
}

#[test]
fn a_stream_carries_interruptions_and_knocks_without_allocating() {
    let (mut stream, mut player, mut voice) = streamed();
    stream.push_text("The birch canoe slid on the smooth planks. This is never said. ");
    let mut block = [0.0f32; 480];
    let mut heard = Vec::new();
    for _ in 0..40 {
        player.render(&mut voice, &mut block);
        heard.extend_from_slice(&block);
    }
    // 0.4 s in: a knock, then the speech broken off with a cough at a
    // sample of the player's clock, and other words after it.
    stream.perturb(6.0, 0.5, When::Now).unwrap();
    let at = player.now() + 2_400;
    stream
        .interrupt_with(Vocalization::Cough, When::At(at))
        .unwrap();
    stream.push_text("Mary may win. ");
    ALLOCATIONS.store(0, Ordering::Relaxed);
    let mut vocalizing_from = None;
    let mut speaking_again = None;
    for _ in 0..400 {
        let now = player.now();
        WATCHED.with(|w| w.set(true));
        player.render(&mut voice, &mut block);
        WATCHED.with(|w| w.set(false));
        heard.extend_from_slice(&block);
        if vocalizing_from.is_none() && voice.is_vocalizing() {
            vocalizing_from = Some(now);
        }
        if vocalizing_from.is_some()
            && speaking_again.is_none()
            && !voice.is_vocalizing()
            && player
                .speaker()
                .is_some_and(synthie_trombone::speech::Speaker::is_speaking)
        {
            speaking_again = Some(now);
        }
    }
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
    assert!(heard.iter().all(|x| x.is_finite()));
    // The cough began in the block of its sample, and the words pushed
    // after it were spoken after it: the sentence that was queued never.
    let from = vocalizing_from.expect("the cough was made");
    assert!(
        from <= at && at < from + 480,
        "cough from {from}, asked at {at}"
    );
    let again = speaking_again.expect("the next words were spoken");
    assert!(again > at + 9_600, "spoken at {again}");
    let total = heard.len() as f32 / SR;
    assert!(total < 6.0);
    // And the same through events pushed by hand, and through a talker.
    let (mut stream, mut player, mut voice) = streamed();
    stream.push_text("The birch canoe slid. ");
    for _ in 0..30 {
        player.render(&mut voice, &mut block);
    }
    stream
        .push(Event::Perturb {
            lung: -12.0,
            jaw: 0.0,
        })
        .unwrap();
    stream.interrupt_with_text("No. Mary may win.");
    let mut flows = Vec::new();
    for _ in 0..300 {
        player.render(&mut voice, &mut block);
        flows.push(voice.gauges().glottal_flow);
        assert!(block.iter().all(|x| x.is_finite()));
    }
    assert!(player.speaker().is_some_and(|s| !s.is_speaking()));
    let mut talker = Talker::new(SR, deep());
    talker.say(TEXT);
    let mut buf = vec![0.0f32; 9_600];
    talker.render(&mut buf);
    ALLOCATIONS.store(0, Ordering::Relaxed);
    WATCHED.with(|w| w.set(true));
    talker.perturb(8.0, 1.0);
    talker.render(&mut buf);
    WATCHED.with(|w| w.set(false));
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
    // The classic engine has no body to knock and no reflex to make: its
    // speech stops, and a knock changes nothing.
    let mut classic = Talker::new(SR, Preset::Deep.settings());
    classic.say(TEXT);
    classic.render(&mut buf);
    assert!(classic.interrupt_with(Vocalization::Cough).is_none());
    assert!(!classic.is_speaking());
}
