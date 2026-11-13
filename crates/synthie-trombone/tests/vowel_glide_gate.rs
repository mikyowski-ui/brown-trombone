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
    clippy::cast_sign_loss
)]

#[path = "../examples/support/spectrum.rs"]
mod spectrum;

use synthie_trombone::speech::{Speaker, Utterance};
use synthie_trombone::vowel_targets::{self, VowelKey};
use synthie_trombone::{Engine, Phoneme, Preset, Trombone, VoiceSettings};

const SR: f32 = 48_000.0;

fn physical() -> VoiceSettings {
    Preset::Deep.settings().with_engine(Engine::Physical)
}

/// Seconds of `p` in `text` (every segment of it).
fn length(text: &str, p: Phoneme) -> f32 {
    Utterance::from_text(text, &physical().prosody())
        .segments()
        .iter()
        .filter(|s| s.phoneme == p)
        .map(|s| s.duration)
        .sum()
}

#[test]
fn lax_vowels_are_35_to_40_percent_shorter_than_tense_ones() {
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
        let u = Utterance::from_text(s, &physical().prosody());
        // Whole vowels: a lax vowel's offglide and a diphthong's glide
        // belong to the segment before them.
        let segs = u.segments();
        let mut k = 0;
        while k < segs.len() {
            let p = segs[k].phoneme;
            let mut d = segs[k].duration;
            if segs.get(k + 1).is_some_and(|n| {
                n.phoneme == p && n.articulation.vowel == Some(VowelKey::End(p))
            }) {
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
    // Schwa is the shortest vowel of all.
    assert!(length("[S AX0 L]", Phoneme::Ax) < 0.95 * length("[S IH0 L]", Phoneme::Ih));
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

/// F1 and F2 of the tract as it is now (closed-glottis transfer function).
fn formants(voice: &Trombone) -> (f32, f32) {
    let f = spectrum::formants(voice, 3500.0);
    let mut hz = f.iter().map(|r| r.hz).filter(|&hz| hz > 150.0);
    (hz.next().unwrap_or(0.0), hz.next().unwrap_or(0.0))
}

/// F1 and F2 at 40 % and at 95 % of the vowel `p` said as `[HH p1]` at the
/// end of an utterance (nothing after it pulls the tongue elsewhere).
fn trajectory(p: Phoneme) -> ((f32, f32), (f32, f32)) {
    let settings = physical();
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
    let at = [start + 0.4 * len, start + 0.95 * len].map(|t| (t * SR) as usize);
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

/// How far a vowel has moved from `a` to `b`, as a share of the way from
/// `a` to `goal`, in `ln` F1 and F2.
fn share(a: (f32, f32), b: (f32, f32), goal: [f32; 3]) -> f32 {
    let d = ((b.0 / a.0).ln(), (b.1 / a.1).ln());
    let g = ((goal[0] / a.0).ln(), (goal[1] / a.1).ln());
    (d.0 * g.0 + d.1 * g.1) / (g.0 * g.0 + g.1 * g.1)
}

#[test]
fn lax_vowels_glide_towards_schwa_and_tense_ones_hold() {
    let schwa = vowel_targets::formants(VowelKey::Vowel(Phoneme::Ax), 1.0).expect("schwa");
    for p in [Phoneme::Ih, Phoneme::Eh, Phoneme::Ah, Phoneme::Uh] {
        let (a, b) = trajectory(p);
        let s = share(a, b, schwa);
        println!(
            "{p}: F1 {:.0} -> {:.0}, F2 {:.0} -> {:.0}: {:.0} % of the way to schwa",
            a.0,
            b.0,
            a.1,
            b.1,
            100.0 * s
        );
        assert!((0.1..=0.35).contains(&s), "{p}: {s}");
    }
    for p in [Phoneme::Iy, Phoneme::Uw] {
        let (a, b) = trajectory(p);
        let s = share(a, b, schwa);
        println!("{p}: {:.0} % of the way to schwa", 100.0 * s);
        assert!(s.abs() < 0.06, "{p}: {s}");
    }
}

#[test]
fn ey_and_ow_glide_the_whole_way() {
    for (p, f1_fall, f2_change) in [(Phoneme::Ey, 60.0, 100.0), (Phoneme::Ow, 50.0, -50.0)] {
        let (a, b) = trajectory(p);
        println!("{p}: F1 {:.0} -> {:.0}, F2 {:.0} -> {:.0}", a.0, b.0, a.1, b.1);
        assert!(a.0 - b.0 > f1_fall, "{p}: F1 {} -> {}", a.0, b.0);
        if f2_change > 0.0 {
            assert!(b.1 - a.1 > f2_change, "{p}: F2 {} -> {}", a.1, b.1);
        } else {
            assert!(b.1 - a.1 < f2_change, "{p}: F2 {} -> {}", a.1, b.1);
        }
    }
}

#[test]
fn tense_vowels_draw_the_tongue_root_forward_and_lax_ones_do_not() {
    let root = |p: Phoneme| p.physical_shape(1.0).expect("a vowel").pharynx;
    for p in [Phoneme::Iy, Phoneme::Ey] {
        assert!(root(p) > 0.2, "{p}: {}", root(p));
    }
    for p in [Phoneme::Ih, Phoneme::Eh, Phoneme::Ah, Phoneme::Uh] {
        assert!(root(p) < 0.15, "{p}: {}", root(p));
    }
    // A consonant made on a vowel keeps that vowel's root: the classic
    // engine, which has none, is left alone.
    let mut a = Trombone::new(SR);
    a.set_engine(Engine::Physical);
    a.articulate_phoneme(Phoneme::Iy);
    a.settle();
    let wide = a.target_diameters()[14];
    a.articulate_phoneme(Phoneme::Ih);
    a.settle();
    assert!(a.target_diameters()[14] < wide);
}
