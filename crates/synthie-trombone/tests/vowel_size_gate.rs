//! Gates for vowels of every body size on the physical engine: a voice
//! with a shorter tract should have every formant raised by `1 / length`,
//! as a smaller person's are, and keep its vowels. Shapes are measured
//! through the real engine, after `articulate_phoneme` and `settle`.
//!
//! ```sh
//! cargo test -p synthie-trombone --release --test vowel_size_gate -- --nocapture
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

use survey::Meter;
use synthie_trombone::{Articulation, Engine, Phoneme, SLOT_LIPS, Trombone};

/// Every vowel the physical engine shapes, glides included.
const VOWELS: [Phoneme; 14] = [
    Phoneme::Iy,
    Phoneme::Ih,
    Phoneme::Eh,
    Phoneme::Ae,
    Phoneme::Ah,
    Phoneme::Aa,
    Phoneme::Ao,
    Phoneme::Uh,
    Phoneme::Uw,
    Phoneme::Ow,
    Phoneme::Er,
    Phoneme::Ax,
    Phoneme::W,
    Phoneme::Y,
];

/// F1 and F2 must be within this of `native / length`.
const LIMIT: f32 = 0.03;

/// Where 3 % cannot be met, and what is met instead: (vowel, length,
/// largest F1 error). In a long tract (a large body) the close rounded
/// vowels /u/ and /w/ cannot lower F1 far enough: their tongue, lips and
/// narrowing are already as tight as any vowel may be without hissing
/// (tongue 2.05, lips 0.65, narrowing 0.7), and the yielding walls hold F1
/// up. A search over every tongue position (12 to 29) and narrowing place
/// (10 to 40) at those openings found nothing lower than +6.5 % at length
/// 1.4. F2 stays within 3 %. Everything else in 0.6‥1.4, here and between,
/// is within 3 %.
const KNOWN_MISSES: &[(Phoneme, f32, f32)] = &[
    (Phoneme::Uw, 1.25, 0.045),
    (Phoneme::W, 1.25, 0.045),
    (Phoneme::Uw, 1.3, 0.052),
    (Phoneme::W, 1.3, 0.052),
    (Phoneme::Uw, 1.35, 0.06),
    (Phoneme::W, 1.35, 0.06),
    (Phoneme::Uw, 1.4, 0.067),
    (Phoneme::W, 1.4, 0.067),
];

/// A physical voice of `length` holding phoneme `p`, settled.
fn holding(p: Phoneme, length: f32) -> Trombone {
    let mut t = Trombone::new(48_000.0);
    t.set_engine(Engine::Physical);
    t.set_tract_length(length);
    t.articulate_phoneme(p);
    t.settle();
    t
}

/// F1..F3 of every `(vowel, length)`, measured on all threads.
fn measure_all(jobs: &[(Phoneme, f32)]) -> Vec<[f32; 3]> {
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    let chunk = jobs.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        let handles: Vec<_> = jobs
            .chunks(chunk)
            .map(|part| {
                s.spawn(move || {
                    let mut m = Meter::new();
                    part.iter()
                        .map(|&(p, l)| {
                            m.formants(&holding(p, l))
                                .unwrap_or_else(|| panic!("{p} at {l}: no three formants"))
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("worker panicked"))
            .collect()
    })
}

/// Measure every vowel at `lengths` and hold F1 and F2 to [`LIMIT`] of
/// `native / length` (or a [known miss](KNOWN_MISSES) to its figure).
fn gate(name: &str, lengths: &[f32]) {
    let natives = measure_all(&VOWELS.map(|p| (p, 1.0)));
    let jobs: Vec<(Phoneme, f32)> = lengths
        .iter()
        .flat_map(|&l| VOWELS.map(|p| (p, l)))
        .collect();
    let got = measure_all(&jobs);
    let mut failures = Vec::new();
    println!("{name}: error against native / length, F1 / F2 / F3 (worst vowel)");
    for &l in lengths {
        let mut worst = [(0.0f32, Phoneme::Aa); 3];
        for ((&(p, jl), f), _) in jobs.iter().zip(&got).zip(0..) {
            if jl.to_bits() != l.to_bits() {
                continue;
            }
            let v = VOWELS.iter().position(|&q| q == p).expect("a vowel");
            let e = [0, 1, 2].map(|k| f[k] / (natives[v][k] / l) - 1.0);
            for k in 0..3 {
                if e[k].abs() > worst[k].0 {
                    worst[k] = (e[k].abs(), p);
                }
            }
            let known = KNOWN_MISSES
                .iter()
                .find(|&&(q, ml, _)| q == p && (ml - l).abs() < 1e-4);
            let (f1_limit, f2_limit) = known.map_or((LIMIT, LIMIT), |&(_, _, m)| (m, LIMIT));
            if e[0].abs() > f1_limit || e[1].abs() > f2_limit {
                failures.push(format!(
                    "{p} at length {l}: F1 {:+.2} %, F2 {:+.2} % (F3 {:+.2} %)",
                    100.0 * e[0],
                    100.0 * e[1],
                    100.0 * e[2]
                ));
            }
            if let Some(&(_, _, m)) = known {
                println!(
                    "  known miss {p} at {l}: F1 {:+.2} % (allowed {:.1} %), F2 {:+.2} %",
                    100.0 * e[0],
                    100.0 * m,
                    100.0 * e[1]
                );
            }
        }
        println!(
            "  length {l:.2}: {:.2} % ({}) / {:.2} % ({}) / {:.2} % ({})",
            100.0 * worst[0].0,
            worst[0].1,
            100.0 * worst[1].0,
            worst[1].1,
            100.0 * worst[2].0,
            worst[2].1,
        );
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn every_vowel_scales_with_the_tract_length() {
    gate("table lengths", &[0.6, 0.7, 0.85, 1.0, 1.15, 1.3, 1.4]);
}

#[test]
fn lengths_between_table_rows_scale_too() {
    // The table has a row every 0.1; these fall between rows.
    gate(
        "between rows",
        &[0.55, 0.65, 0.75, 0.93, 0.97, 1.03, 1.25, 1.35],
    );
}

#[test]
fn the_native_length_is_unchanged() {
    // A voice that was never resized and one resized back to 1.0 hold
    // every phoneme in exactly the same tract, which is exactly the
    // fitted vowel shape.
    for &p in Phoneme::ALL {
        let mut a = Trombone::new(48_000.0);
        a.set_engine(Engine::Physical);
        a.articulate_phoneme(p);
        a.settle();
        let mut b = Trombone::new(48_000.0);
        b.set_engine(Engine::Physical);
        b.set_tract_length(0.7);
        b.set_tract_length(1.0);
        b.articulate_phoneme(p);
        b.settle();
        let bits = |t: &Trombone| t.target_diameters().map(f32::to_bits);
        assert_eq!(bits(&a), bits(&b), "{p}");
        assert_eq!(
            p.tongue(Engine::Physical)
                .map(|(i, d)| (i.to_bits(), d.to_bits())),
            p.tongue_for(Engine::Physical, 1.0)
                .map(|(i, d)| (i.to_bits(), d.to_bits())),
            "{p}"
        );
        if let Some(shape) = p.physical_shape(1.0) {
            let mut c = Trombone::new(48_000.0);
            c.set_engine(Engine::Physical);
            c.set_shape(&shape);
            c.settle();
            assert_eq!(bits(&a), bits(&c), "{p}");
        }
    }
    // And the default voice still sounds the same, sample for sample.
    let speak = |resized: bool| {
        let mut t = Trombone::with_seed(48_000.0, 7);
        t.set_engine(Engine::Physical);
        if resized {
            t.set_tract_length(1.3);
            t.set_tract_length(1.0);
        }
        t.note_on();
        let mut out = Vec::new();
        for p in [Phoneme::Iy, Phoneme::Aa, Phoneme::Uw, Phoneme::Ay] {
            t.articulate_phoneme(p);
            let mut buf = vec![0.0; 4800];
            t.render(&mut buf);
            out.extend(buf);
        }
        out
    };
    assert_eq!(speak(false), speak(true));
}

/// Settled tract of `p` at `length`.
fn tract(p: Phoneme, length: f32) -> Vec<f32> {
    holding(p, length).target_diameters().to_vec()
}

#[test]
fn shapes_vary_smoothly_with_length() {
    // Step the length by 0.01 from 0.5 to 1.6. A smooth path changes the
    // tract a little per step, never by a jump: the cells move by at most
    // 0.025 per step (about 0.016 is the most they do), where a different
    // vowel moves some by 1 or more.
    let lengths: Vec<f32> = (0..=110).map(|k| 0.5 + 0.01 * k as f32).collect();
    let mut worst = (0.0f32, Phoneme::Aa, 0.0f32);
    for p in VOWELS {
        let tracts: Vec<Vec<f32>> = lengths.iter().map(|&l| tract(p, l)).collect();
        let steps: Vec<f32> = tracts
            .windows(2)
            .map(|w| {
                w[0].iter()
                    .zip(&w[1])
                    .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
            })
            .collect();
        for (k, &s) in steps.iter().enumerate() {
            if s > worst.0 {
                worst = (s, p, lengths[k]);
            }
            assert!(
                s < 0.025,
                "{p}: the tract moves by {s} between lengths {} and {}",
                lengths[k],
                lengths[k + 1]
            );
        }
        // Across the native length too: the table's first rows join the
        // fitted shape without a jump.
        let shape = |l: f32| p.physical_shape(l).expect("a vowel shape");
        let (below, at, above) = (shape(0.999), shape(1.0), shape(1.001));
        for s in [below, above] {
            assert!(
                (s.tongue_index - at.tongue_index).abs() < 0.02,
                "{p}: {s:?}"
            );
            assert!(
                (s.tongue_diameter - at.tongue_diameter).abs() < 0.005,
                "{p}: {s:?}"
            );
        }
    }
    println!(
        "largest tract change per 0.01 of length: {:.4} ({} at {:.2})",
        worst.0, worst.1, worst.2
    );
}

#[test]
fn no_shape_hisses_or_leaves_the_engine_range() {
    // Never tighter than the tightest native vowels: lips 0.65 (/u/) and
    // narrowing 0.7 (/i/, /u/).
    for p in VOWELS {
        for k in 0..=110 {
            let l = 0.5 + 0.01 * k as f32;
            let s = p.physical_shape(l).expect("a vowel shape");
            assert!(
                synthie_trombone::TONGUE_INDEX.contains(&s.tongue_index),
                "{p} at {l}: {s:?}"
            );
            assert!(
                synthie_trombone::TONGUE_DIAMETER.contains(&s.tongue_diameter),
                "{p} at {l}: {s:?}"
            );
            assert!(s.lips.is_none_or(|d| d >= 0.65), "{p} at {l}: {s:?}");
            let (at, d) = s.narrowing.expect("a narrowing");
            assert!(d >= 0.7 && (2.0..39.0).contains(&at), "{p} at {l}: {s:?}");
            // Spread vowels stay spread, rounded ones rounded.
            assert_eq!(
                s.lips.is_some(),
                p.physical_shape(1.0).expect("a vowel shape").lips.is_some(),
                "{p}"
            );
        }
    }
}

#[test]
fn consonants_diphthongs_and_the_tongue_handle_agree() {
    for l in [0.6, 0.85, 1.0, 1.3] {
        for p in VOWELS {
            let shape = p.physical_shape(l).expect("a vowel shape");
            let tongue = Some((shape.tongue_index, shape.tongue_diameter));
            // Where the app puts the tongue handle.
            assert_eq!(p.tongue_for(Engine::Physical, l), tongue, "{p} at {l}");
            // What the voice does.
            let t = holding(p, l);
            assert_eq!(Some(t.tongue()), tongue, "{p} at {l}");
            assert_eq!(
                t.constriction(SLOT_LIPS).map(|c| c.diameter),
                shape.lips,
                "{p} at {l}"
            );
            // A consonant spoken with this vowel's tongue (as the speech
            // front end fills it in) gets the same tongue.
            let mut s = Phoneme::S.articulation();
            s.tongue = p.articulation().tongue;
            let mut c = Trombone::new(48_000.0);
            c.set_engine(Engine::Physical);
            c.set_tract_length(l);
            c.articulate(&s);
            assert_eq!(Some(c.tongue()), tongue, "{p} with /s/ at {l}");
        }
        // Diphthongs glide between the shapes of their two vowels.
        for p in [
            Phoneme::Ay,
            Phoneme::Aw,
            Phoneme::Ey,
            Phoneme::Ow,
            Phoneme::Oy,
        ] {
            let (from, to) = p.glide().expect("a diphthong");
            let start = p.physical_shape(l).expect("a start shape");
            if p != Phoneme::Ow {
                // OW starts on its own /o/; the others on their first vowel.
                assert_eq!(Some(start), from.physical_shape(l), "{p} at {l}");
            }
            let end: Articulation = to.articulation();
            let mut v = Trombone::new(48_000.0);
            v.set_engine(Engine::Physical);
            v.set_tract_length(l);
            v.articulate(&end);
            let shape = to.physical_shape(l).expect("an end shape");
            assert_eq!(
                v.tongue(),
                (shape.tongue_index, shape.tongue_diameter),
                "{p} at {l}"
            );
        }
        // /r/ borrows the tongue of /er/.
        let er = Phoneme::Er.physical_shape(l).expect("ER");
        assert_eq!(
            Phoneme::R.tongue_for(Engine::Physical, l),
            Some((er.tongue_index, er.tongue_diameter))
        );
    }
}
