//! Gates for vowels of every body size on the physical engine: every
//! vowel, glide and diphthong end must land on the formants of its target
//! (`src/vowel_targets.rs`), scaled to the voice's size as a smaller or
//! larger person's are. Shapes are measured through the real engine, after
//! `articulate` and `settle`.
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
use synthie_trombone::vowel_targets::{self, Lips, TARGETS, VowelKey};
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

/// F1 and F2 must be within this of the target for the voice's size.
/// No vowel, glide or diphthong end is exempt at any length in 0.6‥1.4.
const LIMIT: f32 = 0.03;

/// The articulation that stands for `key`.
fn articulation(key: VowelKey) -> Articulation {
    match key {
        VowelKey::Vowel(p) | VowelKey::Start(p) => p.articulation(),
        VowelKey::End(p) => p.glide_end().expect("a diphthong"),
    }
}

/// A physical voice of `length` holding `a`, settled.
fn holding_articulation(a: &Articulation, length: f32) -> Trombone {
    let mut t = Trombone::new(48_000.0);
    t.set_engine(Engine::Physical);
    t.set_tract_length(length);
    t.articulate(a);
    t.settle();
    t
}

/// A physical voice of `length` holding phoneme `p`, settled.
fn holding(p: Phoneme, length: f32) -> Trombone {
    holding_articulation(&p.articulation(), length)
}

/// F1..F3 of every `(shape, length)`, measured on all threads.
fn measure_all(jobs: &[(VowelKey, f32)]) -> Vec<[f32; 3]> {
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    let chunk = jobs.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        let handles: Vec<_> = jobs
            .chunks(chunk)
            .map(|part| {
                s.spawn(move || {
                    let mut m = Meter::new();
                    part.iter()
                        .map(|&(k, l)| {
                            m.formants(&holding_articulation(&articulation(k), l))
                                .unwrap_or_else(|| panic!("{k} at {l}: no three formants"))
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

/// Measure every vowel shape at `lengths` and hold F1 and F2 to [`LIMIT`]
/// of its target for that size; report F3 too.
fn gate(name: &str, lengths: &[f32]) {
    let jobs: Vec<(VowelKey, f32)> = lengths
        .iter()
        .flat_map(|&l| TARGETS.iter().map(move |t| (t.key, l)))
        .collect();
    let got = measure_all(&jobs);
    let mut failures = Vec::new();
    println!("{name}: error against the target, F1 / F2 / F3 (worst shape)");
    for &l in lengths {
        let mut worst = [(0.0f32, TARGETS[0].key); 3];
        for (&(k, jl), f) in jobs.iter().zip(&got) {
            if jl.to_bits() != l.to_bits() {
                continue;
            }
            let t = vowel_targets::formants(k, l).expect("a target");
            let e = [0, 1, 2].map(|i| f[i] / t[i] - 1.0);
            for i in 0..3 {
                if e[i].abs() > worst[i].0 {
                    worst[i] = (e[i].abs(), k);
                }
            }
            if e[0].abs() > LIMIT || e[1].abs() > LIMIT {
                failures.push(format!(
                    "{k} at length {l}: F1 {:+.2} %, F2 {:+.2} % (F3 {:+.2} %)",
                    100.0 * e[0],
                    100.0 * e[1],
                    100.0 * e[2]
                ));
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
fn every_vowel_meets_its_target_at_every_size() {
    gate(
        "table lengths",
        &[0.6, 0.7, 0.8, 0.85, 1.0, 1.15, 1.2, 1.3, 1.4],
    );
}

#[test]
fn lengths_between_table_rows_meet_it_too() {
    // The table has a row every 0.1; these fall between rows.
    gate(
        "between rows",
        &[0.55, 0.65, 0.75, 0.93, 0.97, 1.03, 1.25, 1.35],
    );
}

#[test]
fn lax_and_tense_vowels_stay_apart() {
    // The pairs a listener most easily confuses must be as far apart as
    // their targets are (distance in ln F1 and ln F2), at every size.
    let pairs = [
        (Phoneme::Ih, Phoneme::Iy),
        (Phoneme::Uh, Phoneme::Uw),
        (Phoneme::Eh, Phoneme::Ae),
        (Phoneme::Ah, Phoneme::Aa),
    ];
    let lengths = [0.6, 1.0, 1.4];
    let jobs: Vec<(VowelKey, f32)> = lengths
        .iter()
        .flat_map(|&l| {
            pairs
                .iter()
                .flat_map(move |&(a, b)| [(VowelKey::Vowel(a), l), (VowelKey::Vowel(b), l)])
        })
        .collect();
    let got = measure_all(&jobs);
    let distance = |a: [f32; 3], b: [f32; 3]| {
        let d1 = (b[0] / a[0]).ln();
        let d2 = (b[1] / a[1]).ln();
        (d1 * d1 + d2 * d2).sqrt()
    };
    for (n, pair) in got.chunks(2).enumerate() {
        let (ka, l) = jobs[2 * n];
        let kb = jobs[2 * n + 1].0;
        let want = distance(
            vowel_targets::formants(ka, l).expect("a target"),
            vowel_targets::formants(kb, l).expect("a target"),
        );
        let have = distance(pair[0], pair[1]);
        println!(
            "{ka}-{kb} at {l}: ΔF1 {:+.0} Hz, ΔF2 {:+.0} Hz, distance {:.3} (target {want:.3})",
            pair[1][0] - pair[0][0],
            pair[1][1] - pair[0][1],
            have
        );
        // Each vowel may be up to 3 % off its target: together they may come
        // that much closer than the targets are, no more.
        assert!(have > want - LIMIT, "{ka}-{kb} at {l}: {have} vs {want}");
    }
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

/// Settled tract of `key` at `length`.
fn tract(key: VowelKey, length: f32) -> Vec<f32> {
    holding_articulation(&articulation(key), length)
        .target_diameters()
        .to_vec()
}

#[test]
fn shapes_vary_smoothly_with_length() {
    // Step the length by 0.01 from 0.5 to 1.6. A smooth path changes the
    // tract a little per step, never by a jump: the cells move by at most
    // 0.025 per step (about 0.016 is the most they do), where a different
    // vowel moves some by 1 or more.
    let lengths: Vec<f32> = (0..=110).map(|k| 0.5 + 0.01 * k as f32).collect();
    let mut worst = (0.0f32, TARGETS[0].key, 0.0f32);
    for p in TARGETS.iter().map(|t| t.key) {
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
        // Across the native length too.
        let shape = |l: f32| articulation(p).physical_shape(l).expect("a vowel shape");
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
    // Never tighter than the tightest vowels may be without hissing: lips
    // 0.65 and narrowing 0.7.
    for t in TARGETS {
        for k in 0..=110 {
            let l = 0.5 + 0.01 * k as f32;
            let p = t.key;
            let s = articulation(p).physical_shape(l).expect("a vowel shape");
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
            assert_eq!(s.lips.is_some(), t.lips == Lips::Rounded, "{p} at {l}");
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
            // front end fills it in) gets the same tongue. Not /w/'s: its
            // classic tongue is that of /k g ŋ/, which have their own.
            if p == Phoneme::W {
                continue;
            }
            let leaning = |q: Phoneme| {
                let mut a = q.articulation();
                a.tongue = p.articulation().tongue;
                let mut c = Trombone::new(48_000.0);
                c.set_engine(Engine::Physical);
                c.set_tract_length(l);
                c.articulate(&a);
                c.tongue()
            };
            // The lips leave the body exactly where the vowel has it...
            assert_eq!(Some(leaning(Phoneme::P)), tongue, "{p} with /p/ at {l}");
            // ...the tongue tip of /s/ draws it a little forward and up,
            // never further than the tongue's own reach.
            let (index, diameter) = tongue.expect("a tongue");
            let (si, sd) = leaning(Phoneme::S);
            assert!(
                (si - index).abs() <= 0.35 * (26.5 - index).abs() + 1e-3
                    && (sd - diameter).abs() <= 0.35 * (2.5 - diameter).abs() + 1e-3
                    && (si - index) * (26.5 - index) >= 0.0,
                "{p} with /s/ at {l}: {si} {sd} from {index} {diameter}"
            );
        }
        // Diphthongs glide between their own two shapes.
        for p in [
            Phoneme::Ay,
            Phoneme::Aw,
            Phoneme::Ey,
            Phoneme::Ow,
            Phoneme::Oy,
        ] {
            let start = p.physical_shape(l).expect("a start shape");
            assert_eq!(
                Some(start),
                articulation(VowelKey::Start(p)).physical_shape(l),
                "{p} at {l}"
            );
            let end = articulation(VowelKey::End(p));
            let v = holding_articulation(&end, l);
            let shape = end.physical_shape(l).expect("an end shape");
            assert_eq!(
                v.tongue(),
                (shape.tongue_index, shape.tongue_diameter),
                "{p} at {l}"
            );
            assert_eq!(
                v.constriction(SLOT_LIPS).map(|c| c.diameter),
                shape.lips,
                "{p} at {l}"
            );
        }
        // /w/ is rounded, and so protrudes, as /u/ does.
        for p in [Phoneme::W, Phoneme::Uw] {
            let lips = p.physical_shape(l).and_then(|s| s.lips).expect("rounded");
            assert!(lips < 1.0, "{p} at {l}: {lips}");
        }
        // /r/ borrows the tongue of /er/.
        let er = Phoneme::Er.physical_shape(l).expect("ER");
        assert_eq!(
            Phoneme::R.tongue_for(Engine::Physical, l),
            Some((er.tongue_index, er.tongue_diameter))
        );
    }
}
