//! Where the physical engine makes its consonants, and what happens
//! between them: the measurements behind the place-of-articulation gates.
//!
//! ```sh
//! cargo run -p synthie-trombone --example place --release
//! cargo run -p synthie-trombone --example place --release -- loci murmur
//! cargo run -p synthie-trombone --example place --release -- steps --voice elder
//! ```
//!
//! Sections (all by default, or name some):
//!
//! * `loci`: F2 and F3 of the tract's transfer function 60, 30 and 10 ms
//!   before each consonant and at the onset and middle of the vowel after
//!   it, in `[V C V1]` for five vowels, and the locus equation fitted over
//!   them. Published F2 loci: bilabial about 700 to 1100 Hz, alveolar 1700
//!   to 1800 Hz, velar following the vowel with F2 and F3 close together.
//! * `murmur`: the antiresonance of the mouth behind each nasal's closure
//!   in the middle of `[V C V1]` for ten vowels (from the tract's shape as
//!   a closed side branch of the nose, and the depth of the notch the
//!   transfer function has there), then the deepest notches of the
//!   transfer function while a nasal is held. The closed mouth behind the
//!   closure is a side branch, whose resonance is an antiresonance: about
//!   800 to 1000 Hz for /m/, 1500 to 2000 Hz for /n/, above 3 kHz for /ŋ/.
//! * `hiss`: level of each fricative above 2.5 kHz and between 70 and 300
//!   Hz, against the vowel, and how much of it is voiced.
//! * `final`: whether word-final stops are released: the loudest sample in
//!   the 10 ms after the closure opens, against the vowel's.
//! * `steps`: the largest step between two output samples in the click
//!   gate's sentences, and the sound it falls in.
//! * `tongues`: where each vowel holds the tongue body.
//! * `clusters`: in the judge's sentences, how wide the mouth gets between
//!   two consonants that follow one another (a wide one is a vowel nobody
//!   asked for), and how much of each kind of sound the tract spends near
//!   its target.

#![allow(
    clippy::format_push_string,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]

#[path = "support/place.rs"]
mod place;
#[path = "support/spectrum.rs"]
mod spectrum;
#[path = "support/stops.rs"]
mod stops;

use synthie_trombone::{Manner, Phoneme, VoiceSettings};

const VOWELS: [Phoneme; 5] = [
    Phoneme::Iy,
    Phoneme::Eh,
    Phoneme::Aa,
    Phoneme::Ao,
    Phoneme::Uw,
];

const CONSONANTS: [Phoneme; 20] = [
    Phoneme::B,
    Phoneme::P,
    Phoneme::M,
    Phoneme::F,
    Phoneme::V,
    Phoneme::Th,
    Phoneme::Dh,
    Phoneme::D,
    Phoneme::T,
    Phoneme::N,
    Phoneme::S,
    Phoneme::Z,
    Phoneme::Sh,
    Phoneme::Ch,
    Phoneme::L,
    Phoneme::R,
    Phoneme::Y,
    Phoneme::G,
    Phoneme::K,
    Phoneme::Ng,
];

/// The judge's tuning and held-out sentences.
const SENTENCES: &[&str] = &[
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It's easy to tell the depth of a well.",
    "These days a chicken leg is a rare dish.",
    "Rice is often served in round bowls.",
    "The juice of lemons makes fine punch.",
    "The box was thrown beside the parked truck.",
    "The hogs were fed chopped corn and garbage.",
    "Four hours of steady work faced us.",
    "A large size in stockings is hard to sell.",
    "The boy was there when the sun rose.",
    "A rod is used to catch pink salmon.",
    "The source of the huge river is the clear spring.",
    "Kick the ball straight and follow through.",
    "Help the woman get back to her feet.",
    "A pot of tea helps to pass the evening.",
    "Smoky fires lack flame and heat.",
    "The soft cushion broke the man's fall.",
    "The salt breeze came across from the sea.",
    "The girl at the booth sold fifty bonds.",
];

/// Run `f` over `items` on every core, keeping the order.
fn parallel<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);
    let chunk = items.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        let handles: Vec<_> = items
            .chunks(chunk)
            .map(|c| s.spawn(|| c.iter().map(&f).collect::<Vec<_>>()))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a worker"))
            .collect()
    })
}

fn loci(settings: &VoiceSettings) {
    println!("== loci: F2 (F3) in Hz; 'before' is 60/30/10 ms before the consonant");
    let pairs: Vec<(Phoneme, Phoneme)> = CONSONANTS
        .iter()
        .flat_map(|&c| VOWELS.iter().map(move |&v| (c, v)))
        .collect();
    let all = parallel(&pairs, |&(c, v)| place::transition(settings, c, v));
    for (ci, &c) in CONSONANTS.iter().enumerate() {
        let rows = &all[ci * VOWELS.len()..(ci + 1) * VOWELS.len()];
        let points: Vec<(f32, f32)> = rows
            .iter()
            .filter(|t| t.onset[0] > 0.0 && t.mid[0] > 0.0)
            .map(|t| (t.mid[0], t.onset[0]))
            .collect();
        let (locus, k) = place::locus(&points);
        let mut line = format!("{c:>3}: locus {locus:6.0} slope {k:4.2} |");
        for (v, t) in VOWELS.iter().zip(rows) {
            line += &format!(
                " {v}: onset {:4.0}({:4.0}) mid {:4.0} pre {:4.0}/{:4.0}/{:4.0} |",
                t.onset[0], t.onset[1], t.mid[0], t.before[0][0], t.before[1][0], t.before[2][0]
            );
        }
        println!("{line}");
    }
}

fn murmur(settings: &VoiceSettings) {
    println!(
        "== murmur: the mouth's antiresonance as spoken in [V C V1], Hz (notch depth in the transfer function, dB)"
    );
    for nasal in [Phoneme::M, Phoneme::N, Phoneme::Ng] {
        let mut line = format!("{nasal:>3}:");
        for v in [
            Phoneme::Iy,
            Phoneme::Ih,
            Phoneme::Eh,
            Phoneme::Ae,
            Phoneme::Aa,
            Phoneme::Ah,
            Phoneme::Ao,
            Phoneme::Ow,
            Phoneme::Uw,
            Phoneme::Ax,
        ] {
            let t = place::spoken_nasal(settings, nasal, v);
            let zero = place::branch_zeros(t.diameters(), t.tick_rate())
                .first()
                .copied();
            let notches = place::notches(&t);
            match zero {
                Some(z) => {
                    let depth = notches
                        .iter()
                        .filter(|n| (n.hz - z).abs() <= 0.1 * z)
                        .map(|n| n.depth)
                        .fold(0.0f32, f32::max);
                    line += &format!(" {v} {z:5.0} ({depth:4.1})");
                }
                None => line += &format!(" {v}  none      "),
            }
        }
        println!("{line}");
    }
    println!("== murmur: deepest notches of the transfer function, Hz (depth dB)");
    for nasal in [Phoneme::M, Phoneme::N, Phoneme::Ng] {
        for v in [Phoneme::Iy, Phoneme::Aa, Phoneme::Uw] {
            let notches = place::murmur(settings, nasal, v);
            let mut line = format!("{nasal:>3} after {v}:");
            for n in notches.iter().filter(|n| n.depth >= 8.0).take(8) {
                line += &format!(" {:5.0} ({:4.1})", n.hz, n.depth);
            }
            println!("{line}");
        }
    }
}

fn hiss(settings: &VoiceSettings) {
    println!("== hiss in [AA1 C AA]: dB re the vowel");
    let fr = [
        Phoneme::F,
        Phoneme::Th,
        Phoneme::S,
        Phoneme::Sh,
        Phoneme::V,
        Phoneme::Dh,
        Phoneme::Z,
        Phoneme::Zh,
        Phoneme::Hh,
    ];
    let all = parallel(&fr, |&p| {
        stops::fricative(settings, &format!("[AA1 {p} AA]"), p)
    });
    for (p, m) in fr.iter().zip(all) {
        println!(
            "{p:>3}: hiss {:6.1}  voice bar {:6.1}  voiced {:3.0}%  length {:3.0} ms",
            m.hiss,
            m.voice_bar,
            100.0 * m.voiced_share,
            m.length
        );
    }
}

fn finals(settings: &VoiceSettings) {
    println!("== final stops: burst peak re vowel peak, dB");
    let texts: Vec<String> = ["P", "B", "T", "D", "K", "G"]
        .iter()
        .map(|c| format!("[AA1 {c}]"))
        .chain(
            [
                "dark.",
                "bed.",
                "the parked truck.",
                "round bowls.",
                "bonds.",
            ]
            .map(String::from),
        )
        .collect();
    let all = parallel(&texts, |t| {
        let r = stops::record(settings, t);
        stops::bursts(&r)
    });
    for (t, bursts) in texts.iter().zip(all) {
        let mut line = format!("{t:>22}:");
        for b in bursts {
            line += &format!(
                " {} {:5.1}",
                b.phoneme,
                stops::db(b.peak) - stops::db(b.vowel)
            );
        }
        println!("{line}");
    }
}

fn clusters(settings: &VoiceSettings) {
    println!("== clusters: widest opening between two consonants (diameter), ms wider than 0.8");
    let spoken = parallel(SENTENCES, |t| place::speak(settings, t, false));
    let mut wide = 0;
    let mut total = 0;
    for (t, s) in SENTENCES.iter().zip(&spoken) {
        let mut line = format!("{:>48}:", &t[..t.len().min(48)]);
        for (a, b, widest, ms) in place::cluster_openings(s, 0.8) {
            total += 1;
            if ms > 0.0 {
                wide += 1;
            }
            line += &format!(" {a}-{b} {widest:.2}/{ms:.0}");
        }
        println!("{line}");
    }
    println!("clusters that open wider than 0.8: {wide} of {total}");
    println!("== cluster words: frames between two consonants with F1 above 400 Hz");
    let words = parallel(&place::CLUSTER_WORDS, |w| place::speak(settings, w, true));
    for (w, s) in place::CLUSTER_WORDS.iter().zip(&words) {
        let mut line = format!("{w:>8}:");
        for (a, b, n) in place::cluster_vowel_frames(s, 400.0) {
            line += &format!(" {a}-{b} {n}");
        }
        let f1: Vec<String> = s
            .frames
            .iter()
            .map(|f| f.formants.map_or("-".into(), |x| format!("{:.0}", x[0])))
            .collect();
        line += &format!("   F1: {}", f1.join(" "));
        println!("{line}");
    }
    println!("== time near target (RMS shape error <= 0.25), by manner");
    let mut sums: Vec<(Manner, usize, usize)> = Vec::new();
    for s in &spoken {
        for (m, near, n) in place::at_target(s, 0.25) {
            match sums.iter_mut().find(|e| e.0 == m) {
                Some(e) => {
                    e.1 += near;
                    e.2 += n;
                }
                None => sums.push((m, near, n)),
            }
        }
    }
    for (m, near, n) in sums {
        println!(
            "{m:?}: {:.0}% of {} frames",
            100.0 * near as f32 / n.max(1) as f32,
            n
        );
    }
}

fn tongues() {
    println!("== physical tongue body (index, diameter) of each vowel");
    for &p in Phoneme::ALL {
        if let Some((i, d)) = p.tongue(synthie_trombone::Engine::Physical)
            && matches!(p.manner(), Manner::Vowel | Manner::Diphthong)
        {
            println!("{p:>3}: {i:5.2} {d:5.2}");
        }
    }
}

fn steps(settings: &VoiceSettings) {
    println!("== largest step between two samples, x the loudest vowel sample, and where");
    for text in [
        "As for etchings they are of two kinds British and foreign.",
        "The judge checked the big black bag at the gate.",
        "Pick a cup and dig deep to get the chips.",
    ] {
        let r = stops::record(settings, text);
        let vowel = stops::vowel_peak(&r);
        let (n, step) = r
            .out
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .enumerate()
            .fold((0, 0.0f32), |m, (n, x)| if x > m.1 { (n, x) } else { m });
        let k = r.segment[n];
        let at = |k: usize| r.phonemes.get(k).map_or("-", |p| p.symbol());
        let start = r.segment.iter().position(|&s| s == k).unwrap_or(n);
        println!(
            "{:.2} at {:.3} s, {:.1} ms into {} (glottis {:.3} cm², mouth shut {}; after {}, before {}): {text}",
            step / vowel,
            n as f32 / stops::SR,
            (n - start) as f32 / stops::SR * 1e3,
            at(k),
            r.glottal_area[n],
            r.sealed[n],
            at(k.wrapping_sub(1)),
            at(k + 1)
        );
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // `--voice NAME`: another preset than the judge's.
    let settings = match args.iter().position(|a| a == "--voice") {
        Some(k) if k + 1 < args.len() => {
            let name = args.remove(k + 1);
            args.remove(k);
            synthie_trombone::Preset::from_name(&name)
                .unwrap_or_else(|| panic!("no voice {name}"))
                .settings()
        }
        _ => stops::judge_voice(),
    };
    let wants = |s: &str| args.is_empty() || args.iter().any(|a| a == s);
    if wants("steps") {
        steps(&settings);
    }
    if wants("tongues") {
        tongues();
    }
    if wants("loci") {
        loci(&settings);
    }
    if wants("murmur") {
        murmur(&settings);
    }
    if wants("hiss") {
        hiss(&settings);
    }
    if wants("final") {
        finals(&settings);
    }
    if wants("clusters") {
        clusters(&settings);
    }
}
