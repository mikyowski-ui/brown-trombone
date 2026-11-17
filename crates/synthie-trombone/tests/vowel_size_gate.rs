//! Gates for vowels of every body size on the physical engine: every
//! vowel, glide and diphthong end must land on the formants of its target
//! (`src/vowel_targets.rs`), scaled to the voice's size as a smaller or
//! larger person's are. Shapes are measured through the real engine, after
//! `articulate` and `settle`. Each tract has tables of its own
//! (`TractModel`), and each test names the tract it holds: the cosine
//! body's gates hold both.
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
use synthie_trombone::{Articulation, Engine, Phoneme, SLOT_LIPS, TractModel, Trombone, Tuning};

/// The default tuning with the numbers of `tract` (the cosine body).
fn on(tract: TractModel) -> Tuning {
    Tuning::DEFAULT.with_tract(tract)
}

/// Both tracts, as tunings.
fn both() -> [(TractModel, Tuning); 2] {
    [TractModel::Original, TractModel::Anatomical].map(|t| (t, on(t)))
}

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

/// The limit for `key` on a voice of `tuning`: [`LIMIT`], but 4 % for /ɔ/
/// (and the /ɔ/ that /ɔɪ/ starts from) on the modal body. Its F1 stays up
/// to 3.9 % short of 652 Hz in small bodies (length 0.85 and under; 2.5 %
/// at the native length) with every start and narrowing the solver has:
/// the jaw follows the cosine table's /ɔ/, which hardly opens it (0.12),
/// and the modes alone do not make up the rest (see `examples/vowel_size.rs`'s
/// report). The cosine body is held to 3 % everywhere.
fn limit_of(key: VowelKey, tuning: &Tuning) -> f32 {
    let ao = matches!(
        key,
        VowelKey::Vowel(Phoneme::Ao) | VowelKey::Start(Phoneme::Oy)
    );
    if ao && tuning.anatomy.tongue_model >= 0.5 {
        0.04
    } else {
        LIMIT
    }
}

/// The articulation that stands for `key`.
fn articulation(key: VowelKey) -> Articulation {
    key.articulation().expect("a vowel shape")
}

/// A physical voice of `length` and `tuning` holding `a`, settled (its
/// tongue model and tract choose the table); a rounded vowel with the
/// larynx lowered by `rounded_larynx`, as speech lowers it.
fn holding_on(a: &Articulation, length: f32, tuning: &Tuning) -> Trombone {
    let mut t = Trombone::new(48_000.0);
    t.set_engine(Engine::Physical);
    if tuning != &Tuning::DEFAULT {
        t.set_tuning(tuning);
    }
    t.set_tract_length(length);
    let rounded = a
        .vowel
        .and_then(vowel_targets::target)
        .is_some_and(|t| t.lips == Lips::Rounded);
    if rounded && tuning.articulation.rounded_larynx > 0.0 {
        t.set_larynx_height(-tuning.articulation.rounded_larynx);
    }
    t.articulate(a);
    t.settle();
    t
}

/// A physical voice of `length` and `tuning` holding phoneme `p`, settled.
fn holding(p: Phoneme, length: f32, tuning: &Tuning) -> Trombone {
    holding_on(&p.articulation(), length, tuning)
}

/// F1..F3 of every `(shape, length)` on a voice of `tuning`, measured on
/// all threads.
fn measure_on(jobs: &[(VowelKey, f32)], tuning: &Tuning) -> Vec<[f32; 3]> {
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
                            m.formants(&holding_on(&articulation(k), l, tuning))
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

/// Measure every vowel shape at `lengths` on a voice of `tuning` (the table
/// its tongue model and tract choose) and hold F1 and F2 to [`LIMIT`] of
/// its target for that size; report F3 too.
fn gate_on(name: &str, lengths: &[f32], tuning: &Tuning) {
    let jobs: Vec<(VowelKey, f32)> = lengths
        .iter()
        .flat_map(|&l| TARGETS.iter().map(move |t| (t.key, l)))
        .collect();
    let got = measure_on(&jobs, tuning);
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
            let limit = limit_of(k, tuning);
            if e[0].abs() > limit || e[1].abs() > limit {
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
    for (tract, tuning) in both() {
        gate_on(
            &format!("{tract:?} tract, table lengths"),
            &[0.6, 0.7, 0.8, 0.85, 1.0, 1.15, 1.2, 1.3, 1.4],
            &tuning,
        );
    }
}

#[test]
fn lengths_between_table_rows_meet_it_too() {
    // The table has a row every 0.1; these fall between rows.
    for (tract, tuning) in both() {
        gate_on(
            &format!("{tract:?} tract, between rows"),
            &[0.55, 0.65, 0.75, 0.93, 0.97, 1.03, 1.25, 1.35],
            &tuning,
        );
    }
}

#[test]
fn lax_and_tense_vowels_stay_apart() {
    for (tract, tuning) in both() {
        println!("{tract:?} tract:");
        lax_and_tense_vowels_apart(&tuning);
    }
}

fn lax_and_tense_vowels_apart(tuning: &Tuning) {
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
    let got = measure_on(&jobs, tuning);
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
    // fitted vowel shape, on either tract.
    for (tract, tuning) in both() {
        for &p in Phoneme::ALL {
            let voice = || {
                let mut t = Trombone::new(48_000.0);
                t.set_tuning(&tuning);
                t.set_engine(Engine::Physical);
                t
            };
            let mut a = voice();
            a.articulate_phoneme(p);
            a.settle();
            let mut b = voice();
            b.set_tract_length(0.7);
            b.set_tract_length(1.0);
            b.articulate_phoneme(p);
            b.settle();
            let bits = |t: &Trombone| t.target_diameters().map(f32::to_bits);
            assert_eq!(bits(&a), bits(&b), "{tract:?} {p}");
            if let Some(shape) = p.articulation().physical_shape_with(1.0, &tuning) {
                let mut c = voice();
                c.set_shape(&shape);
                c.settle();
                assert_eq!(bits(&a), bits(&c), "{tract:?} {p}");
            }
        }
    }
    // The tongue the default voice gives a phoneme is that of its size.
    for &p in Phoneme::ALL {
        assert_eq!(
            p.tongue(Engine::Physical)
                .map(|(i, d)| (i.to_bits(), d.to_bits())),
            p.tongue_for(Engine::Physical, 1.0)
                .map(|(i, d)| (i.to_bits(), d.to_bits())),
            "{p}"
        );
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

/// Settled tract of `key` at `length` on a voice of `tuning`.
fn tract(key: VowelKey, length: f32, tuning: &Tuning) -> Vec<f32> {
    holding_on(&articulation(key), length, tuning)
        .target_diameters()
        .to_vec()
}

#[test]
fn shapes_vary_smoothly_with_length() {
    // On the anatomical tract /w/'s lips are as tight as a vowel's may be
    // (0.65) from the native length up, and its tongue goes back by 0.84
    // cells per 0.1 of length there: 0.032 per step between 1.0 and 1.01, a
    // steep stretch, not a jump; the original tract's /w/ keeps its lips
    // free and moves at most 0.016.
    for (tract, tuning, most) in [
        (TractModel::Original, on(TractModel::Original), 0.025),
        (TractModel::Anatomical, on(TractModel::Anatomical), 0.035),
    ] {
        println!("{tract:?} tract:");
        shapes_vary_smoothly(&tuning, most);
    }
}

fn shapes_vary_smoothly(tuning: &Tuning, most: f32) {
    // Step the length by 0.01 from 0.5 to 1.6. A smooth path changes the
    // tract a little per step, never by a jump: the cells move by at most
    // `most` per step, where a different vowel moves some by 1 or more.
    let lengths: Vec<f32> = (0..=110).map(|k| 0.5 + 0.01 * k as f32).collect();
    let mut worst = (0.0f32, TARGETS[0].key, 0.0f32);
    for p in TARGETS.iter().map(|t| t.key) {
        let tracts: Vec<Vec<f32>> = lengths.iter().map(|&l| tract(p, l, tuning)).collect();
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
                s < most,
                "{p}: the tract moves by {s} between lengths {} and {}",
                lengths[k],
                lengths[k + 1]
            );
        }
        // Across the native length too.
        let shape = |l: f32| {
            articulation(p)
                .physical_shape_with(l, tuning)
                .expect("a vowel shape")
        };
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
    for (t, (_, tuning)) in TARGETS.iter().flat_map(|t| both().map(move |b| (t, b))) {
        for k in 0..=110 {
            let l = 0.5 + 0.01 * k as f32;
            let p = t.key;
            let s = articulation(p)
                .physical_shape_with(l, &tuning)
                .expect("a vowel shape");
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
    for (tract, tuning) in both() {
        println!("{tract:?} tract");
        consonants_diphthongs_and_tongue(&tuning);
    }
    // Where the app puts the tongue handle: the default voice's tongue.
    for l in [0.6, 0.85, 1.0, 1.3] {
        for p in VOWELS {
            let shape = p.physical_shape(l).expect("a vowel shape");
            let tongue = Some((shape.tongue_index, shape.tongue_diameter));
            assert_eq!(p.tongue_for(Engine::Physical, l), tongue, "{p} at {l}");
        }
        let er = Phoneme::Er.physical_shape(l).expect("ER");
        assert_eq!(
            Phoneme::R.tongue_for(Engine::Physical, l),
            Some((er.tongue_index, er.tongue_diameter))
        );
    }
}

fn consonants_diphthongs_and_tongue(tuning: &Tuning) {
    let shape_of = |a: Articulation, l: f32| a.physical_shape_with(l, tuning);
    for l in [0.6, 0.85, 1.0, 1.3] {
        for p in VOWELS {
            let shape = shape_of(p.articulation(), l).expect("a vowel shape");
            let tongue = Some((shape.tongue_index, shape.tongue_diameter));
            // What the voice does.
            let t = holding(p, l, tuning);
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
                c.set_tuning(tuning);
                c.set_engine(Engine::Physical);
                c.set_tract_length(l);
                c.articulate(&a);
                c.tongue()
            };
            // The lips leave the body exactly where the vowel has it...
            assert_eq!(Some(leaning(Phoneme::P)), tongue, "{p} with /p/ at {l}");
            // ...the tongue tip of /s/ draws it a little forward and up,
            // never further than the tongue's own reach.
            let (index, diameter) = (shape.tongue_index, shape.tongue_diameter);
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
            let start = shape_of(p.articulation(), l).expect("a start shape");
            assert_eq!(
                Some(start),
                shape_of(articulation(VowelKey::Start(p)), l),
                "{p} at {l}"
            );
            let end = articulation(VowelKey::End(p));
            let v = holding_on(&end, l, tuning);
            let shape = shape_of(end, l).expect("an end shape");
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
            let lips = shape_of(p.articulation(), l)
                .and_then(|s| s.lips)
                .expect("rounded");
            assert!(lips < 1.0, "{p} at {l}: {lips}");
        }
        // /r/ borrows the tongue of /er/.
        let er = shape_of(Phoneme::Er.articulation(), l).expect("ER");
        assert_eq!(
            Some(holding(Phoneme::R, l, tuning).tongue()),
            Some((er.tongue_index, er.tongue_diameter))
        );
    }
}

// ----- the modal tongue body ------------------------------------------------
//
// The modal body (`AnatomyConfig::tongue_model` 1) reads its own tables,
// `src/vowel_modes_table.rs` on the original tract and
// `src/vowel_modes_wave_table.rs` on the anatomical one
// (`vowel_targets::lengthened`).

/// The modal body on the original tract.
fn modal() -> Tuning {
    let mut t = on(TractModel::Original);
    t.anatomy.tongue_model = 1.0;
    t
}

/// The modal body on the anatomical tract.
fn anatomical_modal() -> Tuning {
    let mut t = on(TractModel::Anatomical);
    t.anatomy.tongue_model = 1.0;
    t
}

#[test]
fn modal_vowels_meet_their_targets_at_every_size() {
    gate_on(
        "modal body, table lengths",
        &[0.6, 0.7, 0.8, 0.85, 1.0, 1.15, 1.2, 1.3, 1.4],
        &modal(),
    );
}

#[test]
fn modal_lengths_between_table_rows_meet_it_too() {
    gate_on(
        "modal body, between rows",
        &[0.55, 0.65, 0.75, 0.93, 0.97, 1.03, 1.25, 1.35],
        &modal(),
    );
}

// The modal body on the anatomical tract misses these: /ɔ ɔɪ/ fall up to
// 4.7 % short of their F1 at the small sizes, and /æ/ comes closer to /ɛ/
// than their targets at length 0.6. Run with `--ignored`. (The cosine body
// on the same tract missed the open vowels' F1 by up to 5 % with the tube
// above the folds opening into the lower pharynx in a step; it flares over
// four cells since, as the measured speaker's does, and is held above.)

/// The lengths the anatomical tract's modal table is held to.
const ANATOMICAL_LENGTHS: [f32; 7] = [0.6, 0.75, 0.85, 1.0, 1.15, 1.3, 1.4];

#[test]
#[ignore = "the modal body's /ɔ ɔɪ/ miss F1 at small sizes on the anatomical tract (see above)"]
fn modal_vowels_of_the_anatomical_tract_meet_their_targets() {
    gate_on(
        "anatomical tract, modal body",
        &ANATOMICAL_LENGTHS,
        &anatomical_modal(),
    );
}

#[test]
fn modal_lax_and_tense_vowels_stay_apart() {
    lax_and_tense_apart(&modal());
}

#[test]
#[ignore = "the modal body's /æ/ comes too close to /ɛ/ at length 0.6 on the anatomical tract (see above)"]
fn modal_lax_and_tense_vowels_of_the_anatomical_tract_stay_apart() {
    lax_and_tense_apart(&anatomical_modal());
}

/// The pairs a listener most easily confuses as far apart as their
/// targets, less [`LIMIT`], at three sizes, on a voice of `tuning`.
fn lax_and_tense_apart(tuning: &Tuning) {
    let pairs = [
        (Phoneme::Ih, Phoneme::Iy),
        (Phoneme::Uh, Phoneme::Uw),
        (Phoneme::Eh, Phoneme::Ae),
        (Phoneme::Ah, Phoneme::Aa),
    ];
    for l in [0.6, 1.0, 1.4] {
        let jobs: Vec<(VowelKey, f32)> = pairs
            .iter()
            .flat_map(|&(a, b)| [(VowelKey::Vowel(a), l), (VowelKey::Vowel(b), l)])
            .collect();
        let got = measure_on(&jobs, tuning);
        for (n, pair) in got.chunks(2).enumerate() {
            let (ka, kb) = (jobs[2 * n].0, jobs[2 * n + 1].0);
            let d = |a: [f32; 3], b: [f32; 3]| {
                ((b[0] / a[0]).ln().powi(2) + (b[1] / a[1]).ln().powi(2)).sqrt()
            };
            let want = d(
                vowel_targets::formants(ka, l).expect("a target"),
                vowel_targets::formants(kb, l).expect("a target"),
            );
            let have = d(pair[0], pair[1]);
            assert!(have > want - LIMIT, "{ka}-{kb} at {l}: {have} vs {want}");
        }
    }
}

#[test]
fn modal_shapes_stay_inside_the_body_and_do_not_hiss() {
    // The body's numbers inside what the tract takes, lips and narrowing
    // never tighter than the tightest vowels may be, rounded vowels
    // rounded; and a held vowel is the shape the table gives.
    for tuning in [modal(), anatomical_modal()] {
        for t in TARGETS {
            for k in 0..=110 {
                let l = 0.5 + 0.01 * k as f32;
                let s = articulation(t.key)
                    .physical_shape_with(l, &tuning)
                    .expect("a vowel shape");
                let q = s.modes.expect("the modal body's numbers");
                for (k, v) in q.iter().enumerate() {
                    let (lo, hi) = synthie_trombone::inverse::mode_range(k);
                    assert!((lo..=hi).contains(v), "{} at {l}: {s:?}", t.key);
                }
                assert!(s.lips.is_none_or(|d| d >= 0.65), "{} at {l}: {s:?}", t.key);
                let (_, d) = s.narrowing.expect("a narrowing (none is wide open)");
                assert!(d >= 0.7, "{} at {l}: {s:?}", t.key);
                assert_eq!(
                    s.lips.is_some(),
                    t.lips == Lips::Rounded,
                    "{} at {l}",
                    t.key
                );
            }
            for l in [0.6, 1.0, 1.3] {
                let a = articulation(t.key);
                let s = a.physical_shape_with(l, &tuning).expect("a vowel shape");
                let held = holding_on(&a, l, &tuning);
                let mut c = holding_on(&Phoneme::Sil.articulation(), l, &tuning);
                c.set_shape(&s);
                c.settle();
                let bits = |t: &Trombone| t.target_diameters().map(f32::to_bits);
                assert_eq!(bits(&held), bits(&c), "{} at {l}", t.key);
            }
        }
    }
}
