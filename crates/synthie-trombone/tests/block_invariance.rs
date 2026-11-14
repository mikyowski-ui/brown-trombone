//! The articulators move the same way whatever the length of the control
//! block: the same gesture, brought up to date every 10.7 ms (512 internal
//! samples) or every 2.7 ms (128), gives the same trajectory.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

use synthie_trombone::{Engine, Phoneme, Trombone};

const SR: f32 = 48_000.0;
/// Trajectories are compared every this many samples (one long block).
const EVERY: usize = 512;

fn voice(block: u32) -> Trombone {
    let mut v = Trombone::new(SR);
    v.set_engine(Engine::Physical);
    v.set_block(block);
    v.set_frequency(110.0);
    v.set_voiced(true);
    v
}

/// Tract diameters every [`EVERY`] samples while `gesture` is carried out:
/// its steps are `(time in seconds, phoneme to articulate)`.
fn trajectory(
    block: u32,
    start: Phoneme,
    gesture: &[(f32, Phoneme)],
    seconds: f32,
) -> Vec<Vec<f32>> {
    let mut v = voice(block);
    v.articulate_phoneme(start);
    v.settle();
    let mut out = Vec::new();
    let mut buf = [0.0f32; EVERY];
    let mut next = 0;
    let steps = (seconds * SR) as usize / EVERY;
    for k in 0..steps {
        let t = (k * EVERY) as f32 / SR;
        while next < gesture.len() && gesture[next].0 <= t {
            v.articulate_phoneme(gesture[next].1);
            next += 1;
        }
        v.render(&mut buf);
        out.push(v.diameters().to_vec());
    }
    out
}

/// Largest difference between two trajectories, as a share of the largest
/// distance any cell travels.
fn difference(a: &[Vec<f32>], b: &[Vec<f32>]) -> (f32, usize, usize) {
    let mut travel = 0.0f32;
    for i in 0..a[0].len() {
        let (lo, hi) = a.iter().fold((f32::MAX, f32::MIN), |(lo, hi), row| {
            (lo.min(row[i]), hi.max(row[i]))
        });
        travel = travel.max(hi - lo);
    }
    let mut worst = (0.0f32, 0, 0);
    for (k, (ra, rb)) in a.iter().zip(b).enumerate() {
        for (i, (x, y)) in ra.iter().zip(rb).enumerate() {
            let d = (x - y).abs() / travel;
            if d > worst.0 {
                worst = (d, k, i);
            }
        }
    }
    worst
}

/// A name, the sound held at the start, and the gesture's steps.
type Case = (&'static str, Phoneme, Vec<(f32, Phoneme)>);

#[test]
fn articulators_move_alike_at_any_block_length() {
    use Phoneme as P;
    let cases: [Case; 5] = [
        ("a -> i", P::Aa, vec![(0.0, P::Iy)]),
        ("i -> u", P::Iy, vec![(0.0, P::Uw)]),
        ("a t a", P::Aa, vec![(0.0, P::T), (0.085, P::Aa)]),
        ("a n i", P::Aa, vec![(0.0, P::N), (0.085, P::Iy)]),
        ("u s a", P::Uw, vec![(0.0, P::S), (0.107, P::Aa)]),
    ];
    let mut worst = 0.0f32;
    for (name, start, gesture) in &cases {
        let long = trajectory(512, *start, gesture, 0.3);
        for block in [256u32, 128, 64] {
            let short = trajectory(block, *start, gesture, 0.3);
            let (d, k, i) = difference(&long, &short);
            println!(
                "{name}: block {block} differs from 512 by {:.2} % of the travel (at {:.0} ms, cell {i})",
                100.0 * d,
                1e3 * (k * EVERY) as f32 / SR
            );
            worst = worst.max(d);
        }
    }
    assert!(worst <= 0.01, "{:.2} % of the travel", 100.0 * worst);
}
