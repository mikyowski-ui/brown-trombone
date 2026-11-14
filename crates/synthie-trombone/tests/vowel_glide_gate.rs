//! The physical engine's lax vowels in speech: shorter than the tense
//! ones, and gliding towards schwa as they end, while the tense vowels hold
//! their shape; and the tongue root that the vowel shapes carry.
//!
//! The recognizer of `tools/judge` heard the lax vowels of /hɪd hɛd hʌd
//! hʊd/ merge into a few categories in sentences: they were nearly as long
//! as the tense vowels and held their shape to the end. These gates keep
//! the length contrast and the glide in place.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]

#[path = "../examples/support/spectrum.rs"]
mod spectrum;

use std::sync::OnceLock;

use synthie_trombone::speech::{Speaker, Utterance};
use synthie_trombone::vowel_targets::{self, VowelKey};
use synthie_trombone::{Engine, Phoneme, Preset, Trombone, VoiceSettings};

const SR: f32 = 48_000.0;

/// The tuning under test laid over `settings`: the default one, or, when
/// the environment variable `SYNTHIE_TROMBONE_PROFILE` names a profile
/// file, that profile's (as in `physical_gates`, so the tuner can hold a
/// candidate to these gates too).
fn tuned(settings: VoiceSettings) -> VoiceSettings {
    static PROFILE: OnceLock<Option<String>> = OnceLock::new();
    let profile = PROFILE.get_or_init(|| {
        let path = std::env::var("SYNTHIE_TROMBONE_PROFILE").ok()?;
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        println!("gates run with the profile {path}");
        Some(text)
    });
    let Some(profile) = profile else {
        return settings;
    };
    let laid = settings.with_profile(profile).expect("a voice profile");
    VoiceSettings {
        tuning: laid.tuning,
        ..settings
    }
}

/// The deep voice on the physical engine with the tuning under test.
fn physical() -> VoiceSettings {
    tuned(Preset::Deep.settings().with_engine(Engine::Physical))
}

/// The physical voice with its lax vowels as much shorter than its tense
/// ones as a speaker's (the default; set here so the gate does not move with
/// it; see `Tuning::lax_length`).
fn short_lax() -> VoiceSettings {
    let mut v = physical();
    v.tuning.lax_length = 0.62;
    v.tuning.unstressed_lax_length = 0.62;
    v
}

/// Seconds of `p` in `text` (every segment of it).
fn length(text: &str, p: Phoneme) -> f32 {
    Utterance::from_text(text, &short_lax().prosody())
        .segments()
        .iter()
        .filter(|s| s.phoneme == p)
        .map(|s| s.duration)
        .sum()
}

#[test]
fn lax_vowels_can_be_35_to_40_percent_shorter_than_tense_ones() {
    // In the same frame, stressed and unstressed...
    for (lax, tense) in [(Phoneme::Ih, Phoneme::Iy), (Phoneme::Uh, Phoneme::Uw)] {
        for stress in ["1", "0"] {
            let text = |v: Phoneme| format!("[S {}{stress} L] [AX0]", v.symbol());
            let ratio = length(&text(lax), lax) / length(&text(tense), tense);
            println!("{lax}/{tense} (stress {stress}): {ratio:.3}");
            assert!((0.60..=0.65).contains(&ratio), "{lax}/{tense}: {ratio}");
        }
    }
    // ...and over whole sentences, against the tense vowels and the
    // diphthongs alike.
    let sentences = [
        "The birch canoe slid on the smooth planks.",
        "Glue the sheet to the dark blue background.",
        "It's easy to tell the depth of a well.",
        "The juice of lemons makes fine punch.",
        "The fish twisted and turned on the bent hook.",
        "Press the pants and sew a button on the vest.",
        "The small pup gnawed a hole in the sock.",
        "She sells sea shells by the sea shore.",
    ];
    let lax = [Phoneme::Ih, Phoneme::Eh, Phoneme::Ah, Phoneme::Uh];
    let tense = [Phoneme::Iy, Phoneme::Uw, Phoneme::Ey, Phoneme::Ow];
    let (mut l, mut t) = (Vec::new(), Vec::new());
    for s in sentences {
        let u = Utterance::from_text(s, &short_lax().prosody());
        // Whole vowels: a lax vowel's offglide and a diphthong's glide
        // belong to the segment before them.
        let segs = u.segments();
        let mut k = 0;
        while k < segs.len() {
            let p = segs[k].phoneme;
            let mut d = segs[k].duration;
            if segs
                .get(k + 1)
                .is_some_and(|n| n.phoneme == p && n.articulation.vowel == Some(VowelKey::End(p)))
            {
                d += segs[k + 1].duration;
                k += 1;
            }
            if lax.contains(&p) {
                l.push(d);
            } else if tense.contains(&p) {
                t.push(d);
            }
            k += 1;
        }
    }
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let ratio = mean(&l) / mean(&t);
    println!(
        "sentences: {} lax vowels {:.0} ms, {} tense {:.0} ms, ratio {ratio:.2}",
        l.len(),
        1000.0 * mean(&l),
        t.len(),
        1000.0 * mean(&t)
    );
    assert!(ratio < 0.65, "{ratio}");
    // At the defaults a lax vowel is still the shorter.
    let default = |v: Phoneme| {
        Utterance::from_text(
            &format!("[S {}1 L] [AX0]", v.symbol()),
            &physical().prosody(),
        )
        .segments()
        .iter()
        .filter(|s| s.phoneme == v)
        .map(|s| s.duration)
        .sum::<f32>()
    };
    assert!(default(Phoneme::Ih) < default(Phoneme::Iy));
    // The classic engine keeps them alike.
    let classic = |text: &str, p| {
        Utterance::from_text(text, &Preset::Deep.settings().prosody())
            .segments()
            .iter()
            .filter(|s| s.phoneme == p)
            .map(|s| s.duration)
            .sum::<f32>()
    };
    let (bit, beat) = (
        classic("[S IH1 L] [AX0]", Phoneme::Ih),
        classic("[S IY1 L] [AX0]", Phoneme::Iy),
    );
    assert!((bit - beat).abs() < 1e-6);
}

/// F1 and F2 of the tract as it is now: the first two peaks of the
/// closed-glottis transfer function within 40 dB of its tallest (a rounded
/// vowel's F2 is weak beside its F1).
fn formants(voice: &Trombone) -> (f32, f32) {
    let step = 5.0;
    let rate = voice.tick_rate();
    let mut x = vec![0.0f32; (0.08 * rate) as usize];
    voice.impulse_response(&mut x);
    let mag: Vec<f32> = spectrum::spectrum(&x, rate, 3500.0, step)
        .iter()
        .map(|m| 20.0 * m.max(1e-20).log10())
        .collect();
    let top = mag.iter().copied().fold(f32::MIN, f32::max);
    let reach = (40.0 / step) as usize;
    let mut peaks = (reach.max((150.0 / step) as usize)..mag.len() - reach)
        .filter(|&k| mag[k] > top - 40.0 && mag[k - reach..=k + reach].iter().all(|&v| v <= mag[k]))
        .map(|k| k as f32 * step);
    (peaks.next().unwrap_or(0.0), peaks.next().unwrap_or(0.0))
}

/// F1 and F2 at 40 % and at 90 % of the vowel `p` said as `[HH p1]` at the
/// end of an utterance (nothing after it pulls the tongue elsewhere), with
/// `nucleus` of a lax vowel held on its own shape.
fn trajectory_with(p: Phoneme, nucleus: f32) -> ((f32, f32), (f32, f32)) {
    let mut settings = physical();
    settings.tuning.lax_nucleus = nucleus;
    trajectory_of(p, &settings)
}

/// F1 and F2 at 40 % and at 90 % of `[HH p1]` said by `settings`.
fn trajectory_of(p: Phoneme, settings: &VoiceSettings) -> ((f32, f32), (f32, f32)) {
    let mut voice = Trombone::new(SR);
    settings.apply(&mut voice);
    Speaker::prepare(&mut voice);
    let u = Utterance::from_text(&format!("[HH {}1]", p.symbol()), &settings.prosody());
    let segs = u.segments().to_vec();
    let first = segs.iter().position(|s| s.phoneme == p).expect("the vowel");
    let start: f32 = segs[..first].iter().map(|s| s.duration).sum();
    let len: f32 = segs[first..]
        .iter()
        .take_while(|s| s.phoneme == p)
        .map(|s| s.duration)
        .sum();
    let at = [start + 0.4 * len, start + 0.9 * len].map(|t| (t * SR) as usize);
    let mut sp = Speaker::new();
    sp.speak(u);
    let mut got = Vec::new();
    let mut n = 0;
    while sp.is_speaking() && got.len() < 2 {
        sp.process(&mut voice);
        n += 1;
        if at.contains(&n) {
            got.push(formants(&voice));
        }
    }
    (got[0], got[1])
}

/// [`trajectory_with`] the voice's own nucleus.
fn trajectory(p: Phoneme) -> ((f32, f32), (f32, f32)) {
    trajectory_with(p, physical().tuning.lax_nucleus)
}

/// [`trajectory_with`] for a voice of its own.
fn trajectory_in(p: Phoneme, settings: &VoiceSettings) -> ((f32, f32), (f32, f32)) {
    trajectory_of(p, settings)
}

/// How far a vowel has moved from `a` to `b`, as a share of the way from
/// `from` to `goal`, in `ln` F1 and F2.
fn share(from: (f32, f32), a: (f32, f32), b: (f32, f32), goal: [f32; 3]) -> f32 {
    let d = ((b.0 / a.0).ln(), (b.1 / a.1).ln());
    let g = ((goal[0] / from.0).ln(), (goal[1] / from.1).ln());
    (d.0 * g.0 + d.1 * g.1) / (g.0 * g.0 + g.1 * g.1)
}

#[test]
fn lax_vowels_glide_towards_schwa_and_tense_ones_hold() {
    // Near its end a lax vowel has moved towards schwa against the same
    // vowel held to the end (the same utterance with no glide, so that
    // what the end of the utterance does to both cancels).
    let schwa = vowel_targets::formants(VowelKey::Vowel(Phoneme::Ax), 1.0).expect("schwa");
    for p in [Phoneme::Ih, Phoneme::Eh, Phoneme::Ah, Phoneme::Uh] {
        let (mid, glided) = trajectory(p);
        let (_, held) = trajectory_with(p, 1.0);
        let s = share(mid, held, glided, schwa);
        println!(
            "{p}: F1 {:.0}, F2 {:.0} at 40 %; at 90 % {:.0}, {:.0} held, {:.0}, {:.0} gliding: {:.0} % of the way to schwa",
            mid.0,
            mid.1,
            held.0,
            held.1,
            glided.0,
            glided.1,
            100.0 * s
        );
        // /ɛ/ moves least: 9 % at the default length, 22 % at 0.82 of a
        // tense vowel's (where /ʊ/ moves most, 55 %). /ɪ/ does not open
        // towards schwa: the LibriSpeech reader's closes a little over its
        // second half (F1 -19 Hz, F2 -34 Hz from 40 to 80 % of the vowel
        // in his sentences), the owner's in "slid" too (F1 -18, F2 +31),
        // and the engine's, gliding to schwa (F1 +17, F2 -82), started
        // open, closed and opened again, which is /eɪ/ ("slid" heard as
        // "slayed"); its glide end is a little closer than its nucleus
        // since round 9 (-2 %; 0.07 to 0.6 until then).
        let range = if p == Phoneme::Ih {
            -0.1..=0.1
        } else {
            0.07..=0.6
        };
        assert!(range.contains(&s), "{p}: {s}");
    }
    // The tense vowels have no glide of their own...
    for p in [Phoneme::Iy, Phoneme::Uw, Phoneme::Aa] {
        let u = Utterance::from_text(&format!("[HH {}1]", p.symbol()), &physical().prosody());
        let n = u.segments().iter().filter(|s| s.phoneme == p).count();
        assert_eq!(n, 1, "{p}");
    }
    // ...nor does /æ/ by default; with `Tuning::ae_nucleus` at 0.55 a
    // stressed one glides about a fifth of the way to schwa, an unstressed
    // one not at all.
    let held = physical();
    let u = Utterance::from_text("[HH AE1]", &held.prosody());
    assert_eq!(
        u.segments()
            .iter()
            .filter(|s| s.phoneme == Phoneme::Ae)
            .count(),
        1
    );
    let mut gliding = physical();
    gliding.tuning.ae_nucleus = 0.55;
    let (mid, glided) = trajectory_in(Phoneme::Ae, &gliding);
    let (_, still) = trajectory_in(Phoneme::Ae, &held);
    let s = share(mid, still, glided, schwa);
    println!("AE: {:.0} % of the way to schwa", 100.0 * s);
    assert!((0.15..=0.3).contains(&s), "AE: {s}");
    let u = Utterance::from_text("[HH AE0 D]", &gliding.prosody());
    assert_eq!(
        u.segments()
            .iter()
            .filter(|s| s.phoneme == Phoneme::Ae)
            .count(),
        1
    );
}

#[test]
fn stressed_lax_vowels_have_their_own_length_and_are_never_too_short() {
    // A stressed lax vowel is 18 % shorter than a tense one, an unstressed
    // one 38 % ("sells" against "of"); until round 6 both were 38 %. A
    // stressed /ɪ/ keeps the short length (`Tuning::ih_length`: "slid" was
    // heard as "slayed" at the longer one). The held part of a stressed one
    // never falls under `Tuning::lax_floor`, however fast the speech.
    let ratio = |lax: Phoneme, stress: &str, settings: &VoiceSettings| {
        let len = |v: Phoneme| {
            Utterance::from_text(
                &format!("[S {}{stress} L] [AX0]", v.symbol()),
                &settings.prosody(),
            )
            .segments()
            .iter()
            .filter(|s| s.phoneme == v)
            .map(|s| s.duration)
            .sum::<f32>()
        };
        len(lax) / len(Phoneme::Iy)
    };
    let mut v = physical();
    let (stressed, unstressed) = (ratio(Phoneme::Eh, "1", &v), ratio(Phoneme::Eh, "0", &v));
    let ih = ratio(Phoneme::Ih, "1", &v);
    v.tuning.lax_length = 0.62;
    let as_short = ratio(Phoneme::Eh, "1", &v);
    println!(
        "EH/IY stressed {stressed:.3} (at 0.62: {as_short:.3}), unstressed {unstressed:.3}; IH/IY stressed {ih:.3}"
    );
    assert!((0.80..=0.84).contains(&stressed), "{stressed}");
    assert!((0.60..=0.64).contains(&unstressed), "{unstressed}");
    assert!((0.60..=0.64).contains(&as_short), "{as_short}");
    assert!((0.63..=0.67).contains(&ih), "{ih}");
    // Fast speech: the floor holds every stressed lax nucleus.
    for rate in [1.0f32, 1.4, 2.0] {
        let mut fast = physical();
        fast.rate *= rate;
        let u = Utterance::from_text("[P AH1 N CH] [S EH1 L Z] [K IH1 T]", &fast.prosody());
        let segs = u.segments();
        let heads: Vec<f32> = segs
            .iter()
            .filter(|s| {
                matches!(s.phoneme, Phoneme::Ih | Phoneme::Eh | Phoneme::Ah)
                    && s.articulation.vowel != Some(VowelKey::End(s.phoneme))
            })
            .map(|s| s.duration)
            .collect();
        println!("rate x{rate}: nuclei {heads:.3?} s");
        assert_eq!(heads.len(), 3);
        assert!(
            heads.iter().all(|&d| d >= fast.tuning.lax_floor - 1e-6),
            "{heads:?}"
        );
    }
}

#[test]
fn ey_and_ow_glide_the_whole_way() {
    // /eɪ/ closes and fronts, /oʊ/ closes a little and moves towards /ʊ/.
    let (a, b) = trajectory(Phoneme::Ey);
    println!(
        "EY: F1 {:.0} -> {:.0}, F2 {:.0} -> {:.0}",
        a.0, b.0, a.1, b.1
    );
    assert!(a.0 - b.0 > 60.0 && b.1 - a.1 > 100.0, "EY: {a:?} -> {b:?}");
    let (a, b) = trajectory(Phoneme::Ow);
    println!(
        "OW: F1 {:.0} -> {:.0}, F2 {:.0} -> {:.0}",
        a.0, b.0, a.1, b.1
    );
    assert!(a.0 - b.0 > 25.0 && b.1 - a.1 > 100.0, "OW: {a:?} -> {b:?}");
}

#[test]
fn the_tongue_root_shapes_the_pharynx_and_stays_with_the_body() {
    // Every built-in vowel keeps its root at rest (`LAX_ROOTS` in
    // `examples/vowel_size.rs` says why): the root is the profile's or the
    // caller's to set, as `Shape::pharynx`.
    for t in vowel_targets::TARGETS {
        let a = t.key.articulation().expect("a vowel");
        let root = a.physical_shape(1.0).expect("a shape").pharynx;
        assert!(root.abs() < 1e-6, "{}: {root}", t.key);
    }
    // A lax vowel's root, pulled back as round 6 solved it for /ɪ/ (-0.2),
    // stays under a consonant made on the vowel's body: the pharynx under
    // an /s/ leaning on it is narrower than with the root at rest.
    let mid_s_pharynx = |root: bool| {
        let mut t = Trombone::new(SR);
        t.set_engine(Engine::Physical);
        if root {
            let mut shape = Phoneme::Ih.physical_shape(1.0).expect("a shape");
            shape.pharynx = -0.2;
            t.set_vowel_shape(VowelKey::Vowel(Phoneme::Ih), Some(&shape));
        }
        let mut s = Phoneme::S.articulation();
        s.vowel = Some(VowelKey::Vowel(Phoneme::Ih));
        s.tongue = Phoneme::Ih.articulation().tongue;
        t.articulate(&s);
        t.settle();
        t.target_diameters()[13]
    };
    let (with, without) = (mid_s_pharynx(true), mid_s_pharynx(false));
    println!("pharynx (cell 13) under /s/ after IH: {with:.3} with its root, {without:.3} at rest");
    assert!(with < 0.95 * without);
    // A widened pharynx lowers F1 and a narrowed one raises it, as an
    // advanced and a retracted tongue root do.
    let f1 = |pharynx: f32| {
        let mut shape = Phoneme::Ih.physical_shape(1.0).expect("a shape");
        shape.pharynx = pharynx;
        let mut t = Trombone::new(SR);
        t.set_engine(Engine::Physical);
        t.set_shape(&shape);
        t.settle();
        formants(&t).0
    };
    let (wide, rest, narrow) = (f1(0.4), f1(0.0), f1(-0.4));
    println!(
        "IH with its root drawn forward, at rest, pulled back: F1 {wide:.0}, {rest:.0}, {narrow:.0} Hz"
    );
    assert!(wide < rest - 40.0 && narrow > rest + 40.0);
    // A consonant made on a vowel keeps that vowel's root (the body glides
    // from vowel to vowel under it); the classic engine has none.
    let pharynx_under_s = |engine: Engine, pharynx: f32| {
        let mut t = Trombone::new(SR);
        t.set_engine(engine);
        let mut shape = Phoneme::Uh.physical_shape(1.0).expect("a shape");
        shape.pharynx = pharynx;
        t.set_vowel_shape(VowelKey::Vowel(Phoneme::Uh), Some(&shape));
        let mut s = Phoneme::S.articulation();
        s.vowel = Some(VowelKey::Vowel(Phoneme::Uh));
        s.tongue = Phoneme::Uh.articulation().tongue;
        t.articulate(&s);
        t.settle();
        t.target_diameters()[13]
    };
    assert!(pharynx_under_s(Engine::Physical, -0.8) < 0.8 * pharynx_under_s(Engine::Physical, 0.0));
    assert_eq!(
        pharynx_under_s(Engine::Classic, -0.8).to_bits(),
        pharynx_under_s(Engine::Classic, 0.0).to_bits()
    );
}
