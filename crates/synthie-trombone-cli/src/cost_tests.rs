//! What the audio callback costs: the time it takes per block, measured on
//! the engine the sound device runs (no device), while the voice speaks and
//! while it holds a note. Run with
//! `cargo test --release -p synthie-trombone-cli callback_cost -- --ignored --nocapture`.

use crate::app::App;
use crate::audio::{output, wire};
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::time::Instant;
use synthie_trombone::{Engine, Stereo};
use synthie_ui::{Key, KeyCode};

const SR: f32 = 48_000.0;
const BLOCK: usize = 512;
const CHANNELS: usize = 2;

/// Seconds of sound rendered per measurement.
const SECONDS: f32 = 6.0;

/// Microseconds per 512-frame callback (median and 99th percentile) and
/// the share of one core the audio thread takes, for `engine`, while the
/// app speaks `text` (or holds a note when it is empty).
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn measure(engine: Engine, text: &str, stereo: Stereo) -> (f64, f64, f64) {
    let size = Arc::new(AtomicU32::new(1.0f32.to_bits()));
    let (mut core, link) = wire(SR, None, size, None);
    core.set_stereo(stereo);
    let mut app = App::new(link, true, engine);
    let mut callback = output(core, CHANNELS);
    let mut buf = vec![0.0f32; BLOCK * CHANNELS];
    if text.is_empty() {
        app.key(Key::press(KeyCode::Char('5')));
        app.key(Key::press(KeyCode::Char('i')));
    } else {
        app.paste(text);
        app.key(Key::press(KeyCode::Enter));
    }
    let blocks = (SECONDS * SR) as usize / BLOCK;
    let mut times = Vec::with_capacity(blocks);
    for _ in 0..blocks {
        let t0 = Instant::now();
        callback(&mut buf);
        times.push(t0.elapsed().as_secs_f64() * 1e6);
        app.tick();
    }
    let total: f64 = times.iter().sum();
    times.sort_by(f64::total_cmp);
    let median = times[times.len() / 2];
    let p99 = times[times.len() * 99 / 100];
    let budget = BLOCK as f64 / f64::from(SR) * 1e6;
    (median, p99, total / (blocks as f64 * budget))
}

#[test]
#[ignore = "a measurement: run it on its own, in release"]
#[allow(clippy::cast_precision_loss)]
fn callback_cost() {
    let text =
        "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background.";
    for (engine, stereo) in [
        (Engine::Classic, Stereo::Off),
        (Engine::Physical, Stereo::Off),
        (Engine::Physical, Stereo::Binaural),
    ] {
        for (what, t) in [("speaking", text), ("drone", "")] {
            // The first run warms caches and the allocator up.
            let _ = measure(engine, t, stereo);
            let (median, p99, share) = measure(engine, t, stereo);
            println!(
                "{engine:?} {stereo:?} {what}: {median:.0} µs median, {p99:.0} µs p99 per {BLOCK}-frame callback \
                 ({:.0} µs budget); {:.1} % of a core",
                BLOCK as f64 / f64::from(SR) * 1e6,
                share * 100.0
            );
        }
    }
}

/// Counts what the thread running the audio callback allocates or frees
/// while it is inside the callback.
struct Counting;

thread_local! {
    static IN_CALLBACK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

static ALLOCATED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn counted() {
    if IN_CALLBACK.try_with(std::cell::Cell::get).unwrap_or(false) {
        ALLOCATED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

// SAFETY: every call goes to the system allocator unchanged; the count
// reads a thread-local flag and bumps an atomic, neither of which
// allocates.
unsafe impl std::alloc::GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        counted();
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        counted();
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, size: usize) -> *mut u8 {
        counted();
        unsafe { std::alloc::System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// The audio callback neither allocates nor frees, with the instrument
/// panel's readings (the glottis watch, the tiers, the copies of the voice
/// for the formant meter) on, while the voice speaks and is played.
#[test]
fn the_callback_never_allocates() {
    for (engine, stereo) in [
        (Engine::Physical, Stereo::Off),
        (Engine::Physical, Stereo::Binaural),
        (Engine::Classic, Stereo::Binaural),
    ] {
        let size = Arc::new(AtomicU32::new(1.0f32.to_bits()));
        let (mut core, link) = wire(SR, None, size, None);
        core.set_stereo(stereo);
        let mut app = App::new(link, true, engine);
        let mut callback = output(core, CHANNELS);
        let mut buf = vec![0.0f32; BLOCK * CHANNELS];
        // The first block outside the count.
        callback(&mut buf);
        app.paste("The birch canoe slid on the smooth planks.");
        app.key(Key::press(KeyCode::Enter));
        let mut measured = 0;
        for k in 0..300 {
            if k == 200 {
                app.key(Key::press(KeyCode::Esc));
                app.key(Key::press(KeyCode::Tab));
                app.key(Key::press(KeyCode::Char('h')));
                app.key(Key::press(KeyCode::Char('n')));
            }
            IN_CALLBACK.with(|c| c.set(true));
            callback(&mut buf);
            IN_CALLBACK.with(|c| c.set(false));
            app.tick();
            if app.formants.is_some() {
                measured += 1;
            }
            std::thread::sleep(std::time::Duration::from_micros(300));
        }
        assert_eq!(
            ALLOCATED.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "{engine:?}: the callback allocated"
        );
        // The formants were measured meanwhile (off the audio thread).
        assert!(measured > 0, "{engine:?}");
    }
}
