//! The gesture score on a playing voice: gestures pushed from outside move
//! the tract, the tiers report what they do, and nothing allocates while
//! they play.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use synthie_trombone::gesture::{Target, Window, var};
use synthie_trombone::{Engine, Gesture, Phoneme, Tier, Trombone};

/// Counts the allocations made on each thread.
struct Counting;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn allocations() -> usize {
    ALLOCATIONS.with(Cell::get)
}

// SAFETY: forwards to the system allocator; only counts.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
        // SAFETY: as the caller's contract.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: as the caller's contract.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const SR: f32 = 48_000.0;

fn voice() -> Trombone {
    let mut v = Trombone::new(SR);
    v.set_engine(Engine::Physical);
    v.articulate_phoneme(Phoneme::Aa);
    v.settle();
    v.set_frequency(110.0);
    v.note_on();
    v
}

fn run(v: &mut Trombone, seconds: f32) {
    let mut buf = [0.0f32; 256];
    for _ in 0..(seconds * SR / 256.0) as usize {
        v.render(&mut buf);
    }
}

fn closure() -> Gesture {
    Gesture::new(
        Tier::TongueTip,
        Target::NONE
            .with(var::tip::LOCATION, 36.0)
            .with(var::tip::DEGREE, 0.0),
    )
}

#[test]
fn a_tip_gesture_from_outside_shuts_and_opens_the_mouth() {
    let mut v = voice();
    run(&mut v, 0.1);
    assert!(!v.mouth_shut());
    assert!(v.push_gesture(closure().window(Window::between(0.0, 0.12))));
    let tip = Tier::TongueTip.index();
    let a = v.tier_activity()[tip];
    assert!(
        (a.activation - 1.0).abs() < 1e-6 && a.gestures == 1,
        "{a:?}"
    );
    assert!((a.target - 1.0).abs() < 1e-6, "{a:?}");
    run(&mut v, 0.06);
    assert!(v.mouth_shut(), "the tip closes within 60 ms");
    assert!(v.tier_activity()[tip].value > 0.9);
    run(&mut v, 0.12);
    assert!(!v.mouth_shut(), "and lets go when its window ends");
    let a = v.tier_activity()[tip];
    assert_eq!(a.gestures, 0);
    assert!(a.activation.abs() < 1e-6);
}

#[test]
fn a_velum_gesture_opens_the_nose_and_lets_it_shut() {
    let mut v = voice();
    let port = Target::NONE.with(var::velum::PORT, 1.0);
    let id = v
        .push_gesture_tracked(Gesture::new(Tier::Velum, port))
        .expect("taken");
    run(&mut v, 0.15);
    assert!(v.velum() > 0.9, "{}", v.velum());
    assert!(v.end_gesture(id, 0.02));
    run(&mut v, 0.2);
    assert!(v.velum() < 0.05, "{}", v.velum());
}

#[test]
fn playing_a_score_allocates_nothing() {
    let mut v = voice();
    run(&mut v, 0.05);
    let before = allocations();
    let mut buf = [0.0f32; 128];
    for k in 0..400 {
        if k % 10 == 0 {
            let start = 0.002 * (k % 7) as f32;
            v.push_gesture(
                closure()
                    .dominance(1.0 + (k % 3) as f32)
                    .window(Window::between(start, start + 0.03).ramps(0.01, 0.01)),
            );
            let pitch = Target::NONE.with(var::larynx::PITCH, 100.0 + k as f32 * 0.1);
            v.push_gesture(
                Gesture::new(Tier::LarynxAndPitch, pitch).window(Window::between(0.0, 0.02)),
            );
        }
        v.render(&mut buf);
        let _ = v.tier_activity();
    }
    let after = allocations();
    assert_eq!(after, before, "allocations while playing");
}
