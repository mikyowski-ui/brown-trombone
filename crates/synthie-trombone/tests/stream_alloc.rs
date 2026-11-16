//! The player's end of a voice stream never allocates or frees memory,
//! while events, gestures, compiled text and voice settings arrive from
//! another thread: counted by an allocator that watches the audio thread.

#![allow(clippy::cast_precision_loss)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use synthie_trombone::stream::standin::{Gesture, Tier};
use synthie_trombone::stream::{Event, When, voice_stream};
use synthie_trombone::{Engine, Phoneme, Preset, Trombone};

struct Counting;

thread_local! {
    /// Set on the thread that renders, while it is inside the player.
    static WATCHED: Cell<bool> = const { Cell::new(false) };
}

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static FREES: AtomicUsize = AtomicUsize::new(0);

fn watched() -> bool {
    WATCHED.try_with(Cell::get).unwrap_or(false)
}

// SAFETY: every call is passed on to the system allocator unchanged; the
// counting reads a thread-local flag and bumps atomics, neither of which
// allocates.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if watched() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if watched() {
            FREES.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if watched() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn the_player_never_allocates() {
    // The count sees what the watched thread allocates.
    WATCHED.with(|w| w.set(true));
    let probe = std::hint::black_box(vec![1u8; 64]);
    drop(probe);
    WATCHED.with(|w| w.set(false));
    assert_eq!(ALLOCATIONS.swap(0, Ordering::Relaxed), 1);
    assert_eq!(FREES.swap(0, Ordering::Relaxed), 1);

    let (mut stream, mut player) = voice_stream(2048);
    let settings = Preset::Deep.settings().with_engine(Engine::Physical);
    stream.set_prosody(settings.prosody());
    let done = AtomicBool::new(false);
    let rendered = AtomicUsize::new(0);
    std::thread::scope(|s| {
        // The controller: notes, tract events, gestures, a change of voice
        // and text in chunks, as fast as it can and timed ahead.
        s.spawn(|| {
            let text = "The birch canoe slid on the smooth planks. Glue the sheet to the dark \
                        blue background. It is easy to tell the depth of a well.";
            let words: Vec<&str> = text.split_inclusive(' ').collect();
            let mut k = 0usize;
            while k < 400 {
                let now = stream.now();
                let _ = stream.push_at(now + 500, Event::Pitch(90.0 + (k % 20) as f32));
                let _ = stream.push(Event::Articulate(Phoneme::ALL[k % Phoneme::ALL.len()]));
                let tier = Tier::ALL[k % Tier::ALL.len()];
                let _ = stream.push_gesture(
                    Gesture::new(tier, (k % 5) as f32 / 4.0),
                    When::At(now + 100),
                );
                if k == 50 {
                    let _ = stream.push_voice(settings, When::Now);
                }
                if let Some(w) = words.get(k) {
                    stream.push_text(w);
                }
                if k == words.len() {
                    stream.flush_text();
                }
                k += 1;
                std::thread::sleep(std::time::Duration::from_micros(500));
            }
            done.store(true, Ordering::Release);
        });
        // The audio thread.
        let mut voice = Trombone::new(48_000.0);
        voice.set_engine(Engine::Physical);
        let mut block = [0.0f32; 256];
        // Warm up outside the count (the first speech sets nothing up
        // lazily, but the thread's own first steps may).
        player.render(&mut voice, &mut block);
        while !done.load(Ordering::Acquire) || !player.is_idle() {
            WATCHED.with(|w| w.set(true));
            player.render(&mut voice, &mut block);
            WATCHED.with(|w| w.set(false));
            rendered.fetch_add(block.len(), Ordering::Relaxed);
            assert!(block.iter().all(|x| x.is_finite()));
            assert!(
                rendered.load(Ordering::Relaxed) < 48_000 * 60,
                "it never ended"
            );
        }
    });
    let seconds = rendered.load(Ordering::Relaxed) as f32 / 48_000.0;
    println!(
        "{seconds:.1} s rendered: {} allocations, {} frees in the player",
        ALLOCATIONS.load(Ordering::Relaxed),
        FREES.load(Ordering::Relaxed)
    );
    assert!(seconds > 3.0, "only {seconds} s: the speech did not play");
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
    assert_eq!(FREES.load(Ordering::Relaxed), 0);
}
