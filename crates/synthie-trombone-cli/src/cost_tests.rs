//! What the audio callback costs: the time it takes per block, measured on
//! the engine the sound device runs (no device), while the voice speaks and
//! while it holds a note. Run with
//! `cargo test --release -p synthie-trombone-cli callback_cost -- --ignored --nocapture`.

use crate::app::App;
use crate::audio::{output, wire};
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::time::Instant;
use synthie_trombone::Engine;
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
fn measure(engine: Engine, text: &str) -> (f64, f64, f64) {
    let size = Arc::new(AtomicU32::new(1.0f32.to_bits()));
    let (core, link) = wire(SR, None, size, None);
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
fn callback_cost() {
    let text =
        "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background.";
    for engine in [Engine::Classic, Engine::Physical] {
        for (what, t) in [("speaking", text), ("drone", "")] {
            // The first run warms caches and the allocator up.
            let _ = measure(engine, t);
            let (median, p99, share) = measure(engine, t);
            println!(
                "{engine:?} {what}: {median:.0} µs median, {p99:.0} µs p99 per {BLOCK}-frame callback \
                 ({:.0} µs budget); {:.1} % of a core",
                f64::from(BLOCK as u32) / f64::from(SR) * 1e6,
                share * 100.0
            );
        }
    }
}
