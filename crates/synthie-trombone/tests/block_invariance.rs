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

/// The lips' protrusion and the larynx's height every [`EVERY`] samples,
/// with the lip horn and the larynx slide on, while `gesture` is carried
/// out (as in [`trajectory`]); the larynx is lowered by 0.8 cm from the
/// start and let go at 0.1 s.
fn lengths(block: u32, start: Phoneme, gesture: &[(f32, Phoneme)], seconds: f32) -> Vec<Vec<f32>> {
    let mut v = voice(block);
    let mut tuning = synthie_trombone::Tuning::DEFAULT;
    tuning.anatomy.lip_horn = 1.2;
    tuning.anatomy.larynx_travel = 1.0;
    v.set_tuning(&tuning);
    v.articulate_phoneme(start);
    v.settle();
    v.set_larynx_height(-0.8);
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
        if t >= 0.1 {
            v.set_larynx_height(0.0);
        }
        v.render(&mut buf);
        let (protrusion, height) = v.tract_lengths();
        let mut row = v.diameters().to_vec();
        row.extend([protrusion, height]);
        out.push(row);
    }
    out
}

#[test]
fn the_tract_s_length_moves_alike_at_any_block_length() {
    // The lips' protrusion and the larynx's height are springs of the tract
    // like its cells: with both slides on, they and the cells follow the
    // same course whatever the control block.
    use Phoneme as P;
    let cases: [Case; 3] = [
        ("i -> u", P::Iy, vec![(0.0, P::Uw)]),
        ("u b a", P::Uw, vec![(0.0, P::B), (0.085, P::Aa)]),
        ("a sh a", P::Aa, vec![(0.0, P::Sh), (0.107, P::Aa)]),
    ];
    let mut worst = 0.0f32;
    for (name, start, gesture) in &cases {
        let long = lengths(512, *start, gesture, 0.3);
        for block in [256u32, 128, 64] {
            let short = lengths(block, *start, gesture, 0.3);
            // The two lengths, each against its own travel.
            for k in [synthie_trombone::N, synthie_trombone::N + 1] {
                let pick = |t: &[Vec<f32>]| t.iter().map(|r| vec![r[k]]).collect::<Vec<_>>();
                let (d, at, _) = difference(&pick(&long), &pick(&short));
                println!(
                    "{name}: block {block}, {} differs from 512 by {:.2} % of its travel (at {:.0} ms)",
                    if k == synthie_trombone::N {
                        "protrusion"
                    } else {
                        "larynx"
                    },
                    100.0 * d,
                    1e3 * (at * EVERY) as f32 / SR
                );
                worst = worst.max(d);
            }
        }
    }
    assert!(worst <= 0.01, "{:.2} % of the travel", 100.0 * worst);
}
