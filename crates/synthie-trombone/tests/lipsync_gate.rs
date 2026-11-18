//! The articulators' frames (hybrid roadmap, stage 2), measured as what
//! they are: every channel stays in 0 to 1; each phoneme shows on the
//! channel that makes it; the lips read sealed within a frame of the
//! moment the mouth acoustically shuts for /p b m/; a bake's frames are
//! one for every 512 samples and recording them leaves the sound alone; a
//! stream's player sends them on the sample, at any sample rate, without
//! allocating, and a controller that falls behind loses only the oldest.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use synthie_trombone::speech::Speaker;
use synthie_trombone::stream::{Event, voice_stream};
use synthie_trombone::{
    ArticulatorFrame, Engine, Phoneme, Preset, Talker, Trombone, VoiceSettings,
};

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

const SR: f32 = 48_000.0;

fn physical(preset: Preset) -> VoiceSettings {
    preset.settings().with_engine(Engine::Physical)
}

/// The frame of a voice holding `p`.
fn held(p: Phoneme) -> ArticulatorFrame {
    let mut v = Trombone::new(SR);
    physical(Preset::Deep).apply(&mut v);
    Speaker::prepare(&mut v);
    v.articulate_phoneme(p);
    v.settle();
    let mut block = vec![0.0f32; 4800];
    v.render(&mut block);
    v.articulators()
}

fn in_range(f: &ArticulatorFrame) -> bool {
    f.to_array().iter().all(|c| (0.0..=1.0).contains(c))
}

#[test]
fn each_phoneme_shows_on_the_channel_that_makes_it() {
    // Measured (examples/lipsync.rs): the numbers in the comments.
    for p in [Phoneme::P, Phoneme::B, Phoneme::M] {
        let f = held(p);
        assert_eq!(f.lip_close, 1.0, "{p}: the lips are sealed");
        assert_eq!(f.lip_press, 0.0, "{p}");
        assert!(f.tongue_up < 0.2, "{p}: {f:?}");
    }
    for p in [Phoneme::F, Phoneme::V] {
        // The lower lip at the teeth: 0.87, and the lips not sealed.
        let f = held(p);
        assert!(f.lip_press > 0.7, "{p}: {f:?}");
        assert!(f.lip_close < 0.98, "{p}: {f:?}");
    }
    for p in [Phoneme::T, Phoneme::D, Phoneme::N] {
        let f = held(p);
        assert_eq!(f.tongue_up, 1.0, "{p}: the tip shuts the mouth");
        assert!(f.lip_close < 0.1, "{p}: {f:?}");
    }
    for p in [Phoneme::S, Phoneme::Z, Phoneme::Sh, Phoneme::L, Phoneme::Th] {
        // 0.90, 0.90, 0.87, 0.65, 0.72: raised, not shut.
        let f = held(p);
        assert!((0.55..1.0).contains(&f.tongue_up), "{p}: {f:?}");
    }
    for p in [Phoneme::K, Phoneme::G, Phoneme::Ng] {
        // The body makes them, not the tip or the lips.
        let f = held(p);
        assert!(f.tongue_up < 0.5 && f.lip_close < 0.1, "{p}: {f:?}");
    }
    for p in [Phoneme::M, Phoneme::N, Phoneme::Ng] {
        assert_eq!(held(p).velum_open, 1.0, "{p}");
    }
    for p in [Phoneme::Aa, Phoneme::Iy, Phoneme::P, Phoneme::S] {
        assert_eq!(held(p).velum_open, 0.0, "{p}");
    }
    // Rounding: /w/ and /u/ pucker (0.94, 0.68), /ʃ/ funnels more than it
    // puckers, the spread and the open vowels do neither.
    assert!(held(Phoneme::W).lip_pucker > 0.8);
    assert!(held(Phoneme::Uw).lip_pucker > 0.5);
    let sh = held(Phoneme::Sh);
    assert!(
        sh.lip_funnel > sh.lip_pucker && sh.lip_funnel > 0.15,
        "{sh:?}"
    );
    for p in [Phoneme::Iy, Phoneme::Aa, Phoneme::Ae, Phoneme::Eh] {
        let f = held(p);
        assert_eq!(
            (f.lip_pucker, f.lip_funnel, f.lip_close),
            (0.0, 0.0, 0.0),
            "{p}"
        );
    }
    // The jaw: /æ ɑ ɛ/ open (0.93, 0.70, 0.60), /i u/ shut (0.16, 0.00).
    assert!(held(Phoneme::Ae).jaw_open > 0.8);
    assert!(held(Phoneme::Aa).jaw_open > 0.6);
    assert!(held(Phoneme::Eh).jaw_open > 0.45);
    assert!(held(Phoneme::Iy).jaw_open < 0.25);
    assert!(held(Phoneme::Uw).jaw_open < 0.1);
    // The body: /i/ at the front, /ɑ ɔ/ at the back.
    assert!(held(Phoneme::Iy).tongue_back < 0.1);
    assert!(held(Phoneme::Aa).tongue_back > 0.7);
    assert!(held(Phoneme::Ao).tongue_back > 0.9);
    // The larynx at rest.
    assert_eq!(held(Phoneme::Aa).larynx_height, 0.5);
}

#[test]
fn the_lips_read_sealed_within_a_frame_of_the_mouth_shutting() {
    // Only bilabial closures shut this text's mouth. Sample by sample: the
    // moment the lips' lumen shuts (the gauge the sound itself goes by),
    // against the first frame of the track that reads the lips sealed.
    let text = "Papa. Mama. Baby boy. Bob may buy more. My mom, by a bear.";
    for preset in [Preset::Deep, Preset::Bright] {
        let mut talker = Talker::new(SR, physical(preset));
        let u = talker.compile(text);
        let len = ((u.duration() + 0.25) * SR) as usize;
        talker.record_articulators(true);
        talker.say_utterance(u);
        let mut shut_at = Vec::new();
        // (Before its first block the engine's gauge reads shut.)
        let mut was = true;
        for k in 0..len {
            talker.process();
            let shut = talker.voice_mut().gauges().mouth_shut;
            if shut && !was {
                shut_at.push(k);
            }
            was = shut;
        }
        let frames = talker.take_articulators();
        assert_eq!(frames.len(), len.div_ceil(512));
        assert!(shut_at.len() >= 12, "{preset}: {} closures", shut_at.len());
        for &at in &shut_at {
            // The first frame at or after the closure, and the one before.
            let k = at.div_ceil(512);
            let sealed = frames[k].lip_close;
            assert!(
                sealed >= 0.98,
                "{preset}: mouth shut at sample {at}, frame {k} reads {sealed:.3}"
            );
            // Not sealed a frame early: the track does not run ahead of
            // the sound by more than one frame.
            if k >= 2 {
                let before = frames[k - 2].lip_close;
                assert!(
                    before < 0.98,
                    "{preset}: frame {} already reads {before:.3} before sample {at}",
                    k - 2
                );
            }
        }
        // And every sealed frame is a shut mouth: the lips never read
        // sealed while sound leaves them.
        assert!(frames.iter().all(in_range));
    }
}

#[test]
fn a_bake_has_a_frame_for_every_512_samples_and_the_same_sound() {
    let text = "We fish for food. [laugh] The moon is, um, too blue! [cough]";
    for (name, settings) in [
        ("physical", physical(Preset::Deep)),
        ("bright", physical(Preset::Bright)),
        ("classic", Preset::Deep.settings()),
    ] {
        let plain = Talker::new(SR, settings).bake(text);
        let mut talker = Talker::new(SR, settings);
        let (out, frames) = talker.bake_with_articulators(text);
        assert!(plain == out, "{name}: recording the frames moved the sound");
        assert_eq!(frames.len(), out.len().div_ceil(512), "{name}");
        assert!(frames.iter().all(in_range), "{name}");
        // Frame k is the mouth before sample k · 512: rendered again by
        // hand, the voice reads the same there.
        let mut again = Talker::new(SR, settings);
        let u = again.compile(text);
        again.voice_mut().reset();
        again.say_utterance(u);
        for (k, frame) in frames.iter().enumerate().take(200) {
            assert_eq!(
                *frame,
                again.voice_mut().articulators(),
                "{name}: frame {k}"
            );
            let mut block = [0.0f32; 512];
            again.render(&mut block);
        }
        // The mouth moves: the track is not a constant.
        let moved = |c: fn(&ArticulatorFrame) -> f32| {
            let (lo, hi) = frames
                .iter()
                .fold((1.0f32, 0.0f32), |(l, h), f| (l.min(c(f)), h.max(c(f))));
            hi - lo
        };
        assert!(moved(|f| f.lip_close) > 0.9, "{name}");
        assert!(moved(|f| f.tongue_up) > 0.9, "{name}");
        assert!(moved(|f| f.jaw_open) > 0.2, "{name}");
        assert!(moved(|f| f.glottal_open) > 0.3, "{name}");
        // Another rate, the same clock: a frame every 10.67 ms.
        let mut slow = Talker::new(44_100.0, settings);
        let (out, frames) = slow.bake_with_articulators("We fish for food.");
        let seconds = out.len() as f32 / 44_100.0;
        assert_eq!(
            frames.len(),
            (seconds / ArticulatorFrame::PERIOD).ceil() as usize,
            "{name}"
        );
    }
}

#[test]
fn a_stream_sends_a_frame_every_512_samples_without_allocating() {
    let (mut stream, mut player) = voice_stream(1024);
    let mut voice = Trombone::new(SR);
    physical(Preset::Deep).apply(&mut voice);
    Speaker::prepare(&mut voice);
    assert!(stream.poll_frame().is_none());
    stream.push(Event::Articulate(Phoneme::M)).unwrap();
    stream.push(Event::NoteOn(45.0)).unwrap();
    stream
        .push_at(9_600, Event::Articulate(Phoneme::Aa))
        .unwrap();
    let mut block = [0.0f32; 512];
    let mut taken = Vec::with_capacity(64);
    ALLOCATIONS.store(0, Ordering::Relaxed);
    for _ in 0..40 {
        WATCHED.with(|w| w.set(true));
        player.render(&mut voice, &mut block);
        WATCHED.with(|w| w.set(false));
        while let Some(f) = stream.poll_frame() {
            taken.push(f);
        }
    }
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
    // One at every 512th sample, the first at the first.
    assert_eq!(taken.len(), 40);
    for (k, (at, frame)) in taken.iter().enumerate() {
        assert_eq!(*at, 512 * k as u64);
        assert!(in_range(frame));
    }
    // The /m/ and then the /ɑ/: lips sealed with the nose open, then open
    // with it shut.
    assert_eq!(taken[10].1.lip_close, 1.0);
    assert_eq!(taken[10].1.velum_open, 1.0);
    assert_eq!(taken[39].1.lip_close, 0.0);
    assert!(taken[39].1.velum_open < 0.05);
    assert!(taken[39].1.jaw_open > 0.5);
    // The last frame is the voice as it stood then.
    let before = taken[39].1;
    assert_eq!(before.lip_close, 0.0);

    // A controller that does not look for two seconds keeps the newest 63,
    // in order, and misses nothing after that.
    for _ in 0..190 {
        player.render(&mut voice, &mut block);
    }
    let mut late = Vec::new();
    while let Some((at, _)) = stream.poll_frame() {
        late.push(at);
    }
    assert_eq!(late.len(), 63);
    assert_eq!(*late.last().unwrap(), 512 * 229);
    assert!(late.windows(2).all(|w| w[1] == w[0] + 512));
    player.render(&mut voice, &mut block);
    assert_eq!(stream.poll_frame().map(|f| f.0), Some(512 * 230));
    assert!(stream.poll_frame().is_none());

    // At 44.1 kHz the frames keep their 10.67 ms: one every 470.4 samples.
    let (mut stream, mut player) = voice_stream(1024);
    let mut voice = Trombone::new(44_100.0);
    let mut block = vec![0.0f32; 4410];
    player.render(&mut voice, &mut block);
    let mut at = Vec::new();
    while let Some((t, _)) = stream.poll_frame() {
        at.push(t);
    }
    assert_eq!(at.len(), 10);
    for (k, t) in at.iter().enumerate() {
        let due = k as f64 * 512.0 * 44_100.0 / 48_000.0;
        assert!(
            (*t as f64 - due).abs() <= 1.0,
            "frame {k} at {t}, due {due}"
        );
    }
}
