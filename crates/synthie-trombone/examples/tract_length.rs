//! What a tract that changes length does (wave 2A): the lip horn
//! (`AnatomyConfig::lip_horn`) and the larynx slide with its piston
//! (`AnatomyConfig::larynx_travel`, `ArticulationConfig::voiced_stop_larynx`).
//!
//! ```sh
//! cargo run -p synthie-trombone --example tract_length --release -- formants
//! cargo run -p synthie-trombone --example tract_length --release -- solve
//! cargo run -p synthie-trombone --example tract_length --release -- voicing
//! cargo run -p synthie-trombone --example tract_length --release -- cost
//! ```
//!
//! * `formants`: F1 to F3 of the rounded sounds as the vowel tables shape
//!   them now, with the lips not pushed forward at all, with today's lumped
//!   lip tube, and with the horn at 0.73 cm (the lumped tube's length as a
//!   real tube), 0.95 and 1.2 cm; and with the larynx lowered by 0.4 and
//!   0.8 cm.
//! * `solve`: how much wider the lips of /u/ and of the fronted /u/ can be,
//!   or how much lower the tongue, for the F1 and F2 they have today, with
//!   the horn on.
//! * `voicing`: voiced stops between vowels and before unstressed ones,
//!   today, with the larynx lowered through the closure, and without the
//!   pharynx's widening: how long the folds vibrate into the closure, the
//!   voice bar, the pressure across the glottis, the mouth's pressure at the
//!   release and the burst.
//! * `cost`: the time one 512-frame callback takes, off and on.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names
)]

#[path = "support/spectrum.rs"]
mod spectrum;
#[path = "support/stops.rs"]
mod stops;
#[path = "support/survey.rs"]
mod survey;

use synthie_trombone::inverse::Shape;
use synthie_trombone::speech::{Speaker, Utterance};
use synthie_trombone::{Engine, Phoneme, Trombone, Tuning, VoiceSettings};

/// The tuning of the judge's voice with the lip horn at `horn` cm and the
/// larynx slide at `travel`, the larynx lowered by `stop` cm in voiced
/// closures.
fn tuning(horn: f32, travel: f32, stop: f32) -> Tuning {
    let mut t = stops::judge_voice().tuning;
    t.anatomy.lip_horn = horn;
    t.anatomy.larynx_travel = travel;
    t.articulation.voiced_stop_larynx = stop;
    t
}

/// A settled physical voice of `tuning` holding `p` (or `shape`), the lips
/// pushed forward by `protrusion` cm if given, the larynx at `height` cm.
fn held(
    tuning: &Tuning,
    p: Option<Phoneme>,
    shape: Option<&Shape>,
    protrusion: Option<f32>,
    height: f32,
) -> Trombone {
    let mut t = Trombone::new(48_000.0);
    t.set_tuning(tuning);
    t.set_engine(Engine::Physical);
    if let Some(p) = p {
        t.articulate_phoneme(p);
    }
    if let Some(s) = shape {
        t.set_shape(s);
    }
    t.set_lip_protrusion(protrusion);
    t.set_larynx_height(height);
    t.settle();
    t
}

fn show(f: Option<[f32; 3]>) -> String {
    f.map_or_else(
        || "   -    -    -".to_string(),
        |f| format!("{:4.0} {:4.0} {:4.0}", f[0], f[1], f[2]),
    )
}

fn formants() {
    use Phoneme as P;
    let sounds = [P::Uw, P::Uh, P::Ow, P::Ao, P::W, P::R, P::Sh, P::Iy, P::Aa];
    let off = tuning(0.0, 0.0, 0.0);
    let cases: Vec<(&str, Tuning, Option<f32>, f32)> = vec![
        ("no protrusion", tuning(1.2, 0.0, 0.0), Some(0.0), 0.0),
        ("today (lumped, up to 0.73)", off, None, 0.0),
        ("horn 0.73", tuning(0.73, 0.0, 0.0), None, 0.0),
        ("horn 0.95", tuning(0.95, 0.0, 0.0), None, 0.0),
        ("horn 1.2", tuning(1.2, 0.0, 0.0), None, 0.0),
        ("larynx slide at rest", tuning(0.0, 1.0, 0.0), None, 0.0),
        ("larynx -0.4", tuning(0.0, 1.0, 0.0), None, -0.4),
        ("larynx -0.8", tuning(0.0, 1.0, 0.0), None, -0.8),
        ("horn 1.2, larynx -0.8", tuning(1.2, 1.0, 0.0), None, -0.8),
    ];
    print!("{:28}", "F1 F2 F3, Hz");
    for p in sounds {
        print!(" | {:^14}", p.symbol());
    }
    println!();
    let mut today = Vec::new();
    for (name, t, protrusion, height) in &cases {
        print!("{name:28}");
        let mut row = Vec::new();
        for p in sounds {
            let v = held(t, Some(p), None, *protrusion, *height);
            let f = survey::formants(&v);
            row.push(f);
            print!(" | {}", show(f));
        }
        println!();
        if name.starts_with("today") {
            today.clone_from(&row);
        }
        // The lips' protrusion each sound settles at.
        print!("{:28}", "  lips forward, cm");
        for p in sounds {
            let v = held(t, Some(p), None, *protrusion, *height);
            print!(" | {:^14.2}", v.tract_lengths().0);
        }
        println!();
    }
    println!("\nagainst today, per cent:");
    for (name, t, protrusion, height) in &cases {
        print!("{name:28}");
        for (k, p) in sounds.into_iter().enumerate() {
            let f = survey::formants(&held(t, Some(p), None, *protrusion, *height));
            match (f, today[k]) {
                (Some(f), Some(o)) => print!(
                    " | {:+4.1} {:+4.1} {:+4.1}",
                    100.0 * (f[0] / o[0] - 1.0),
                    100.0 * (f[1] / o[1] - 1.0),
                    100.0 * (f[2] / o[2] - 1.0)
                ),
                _ => print!(" | {:^14}", "-"),
            }
        }
        println!();
    }
}

/// The native rows of the vowel table (`src/vowel_size_table.rs`) for /u/
/// and the fronted /u/ of "canoe".
fn table_shape(fronted: bool) -> Shape {
    if fronted {
        Shape {
            tongue_index: 29.0,
            tongue_diameter: 2.05,
            lips: Some(0.867),
            narrowing: Some((14.0, 1.775)),
            pharynx: 0.0,
            modes: None,
        }
    } else {
        Shape {
            tongue_index: 26.181,
            tongue_diameter: 2.05,
            lips: Some(0.733),
            narrowing: Some((14.0, 1.203)),
            pharynx: 0.0,
            modes: None,
        }
    }
}

/// F1, F2 of `shape` with `tuning`.
fn f12(tuning: &Tuning, shape: &Shape) -> Option<[f32; 2]> {
    survey::formants(&held(tuning, None, Some(shape), None, 0.0)).map(|f| [f[0], f[1]])
}

/// Move `shape`'s variables `vars` (0 the lips, 1 the tongue's height) so
/// that its F1 and F2 with `tuning` come to `goal` (the first or both):
/// damped Newton on finite differences.
fn solve_for(tuning: &Tuning, shape: &Shape, goal: [f32; 2], vars: &[usize]) -> Shape {
    let get = |s: &Shape, v: usize| {
        if v == 0 {
            s.lips.unwrap_or(1.5)
        } else {
            s.tongue_diameter
        }
    };
    let set = |s: &mut Shape, v: usize, x: f32| {
        if v == 0 {
            s.lips = Some(x.clamp(0.4, 1.6));
        } else {
            s.tongue_diameter = x.clamp(2.05, 3.5);
        }
    };
    let mut s = *shape;
    for _ in 0..12 {
        let Some(f) = f12(tuning, &s) else { break };
        // Residuals, relative: F2 alone for one variable, F1 and F2 for two.
        let r: Vec<f32> = if vars.len() == 1 {
            vec![f[1] / goal[1] - 1.0]
        } else {
            vec![f[0] / goal[0] - 1.0, f[1] / goal[1] - 1.0]
        };
        if r.iter().all(|e| e.abs() < 0.002) {
            break;
        }
        let h = 0.01;
        let mut jac = Vec::new();
        for &v in vars {
            let mut t = s;
            set(&mut t, v, get(&s, v) + h);
            let Some(g) = f12(tuning, &t) else { return s };
            let col: Vec<f32> = if vars.len() == 1 {
                vec![(g[1] - f[1]) / goal[1] / h]
            } else {
                vec![(g[0] - f[0]) / goal[0] / h, (g[1] - f[1]) / goal[1] / h]
            };
            jac.push(col);
        }
        let step: Vec<f32> = if vars.len() == 1 {
            vec![-r[0] / jac[0][0]]
        } else {
            let (a, b, c, d) = (jac[0][0], jac[1][0], jac[0][1], jac[1][1]);
            let det = a * d - b * c;
            vec![-(d * r[0] - b * r[1]) / det, -(-c * r[0] + a * r[1]) / det]
        };
        for (k, &v) in vars.iter().enumerate() {
            let x = get(&s, v) + step[k].clamp(-0.15, 0.15);
            set(&mut s, v, x);
        }
    }
    s
}

fn solve() {
    let off = tuning(0.0, 0.0, 0.0);
    for horn in [0.73f32, 1.2] {
        let on = tuning(horn, 0.0, 0.0);
        for fronted in [false, true] {
            let name = if fronted { "fronted /u/" } else { "/u/" };
            let s = table_shape(fronted);
            let Some(goal) = f12(&off, &s) else { continue };
            let Some(now) = f12(&on, &s) else { continue };
            println!(
                "{name}, horn {horn}: today {:.0} {:.0} Hz; with the horn {:.0} {:.0} ({:+.1} %, {:+.1} %)",
                goal[0],
                goal[1],
                now[0],
                now[1],
                100.0 * (now[0] / goal[0] - 1.0),
                100.0 * (now[1] / goal[1] - 1.0)
            );
            let lips = solve_for(&on, &s, goal, &[0]);
            let f = f12(&on, &lips).unwrap_or([0.0; 2]);
            let forward = held(&on, None, Some(&lips), None, 0.0).tract_lengths().0;
            println!(
                "  lips alone for today's F2: {:.3} (from {:.3}; the lips' cell {:.2} to {:.2} cm²), F1 F2 {:.0} {:.0}, lips forward {forward:.2} cm",
                lips.lips.unwrap_or(0.0),
                s.lips.unwrap_or(0.0),
                area(s.lips.unwrap_or(0.0)),
                area(lips.lips.unwrap_or(0.0)),
                f[0],
                f[1]
            );
            let tongue = solve_for(&on, &s, goal, &[1]);
            let f = f12(&on, &tongue).unwrap_or([0.0; 2]);
            println!(
                "  tongue height alone for today's F2: {:.3} (from {:.3}), F1 F2 {:.0} {:.0}",
                tongue.tongue_diameter, s.tongue_diameter, f[0], f[1]
            );
            let both = solve_for(&on, &s, goal, &[0, 1]);
            let f = f12(&on, &both).unwrap_or([0.0; 2]);
            println!(
                "  lips and tongue height for today's F1 and F2: lips {:.3} ({:.2} cm²), tongue {:.3} (from {:.3}), F1 F2 {:.0} {:.0}",
                both.lips.unwrap_or(0.0),
                area(both.lips.unwrap_or(0.0)),
                both.tongue_diameter,
                s.tongue_diameter,
                f[0],
                f[1]
            );
        }
    }
}

/// Area, cm², of the lips' cell for a rounding of touch diameter `d` at the
/// native size (the cell is `d − 0.3` across, read by `AREA_SCALE` 1.6).
fn area(d: f32) -> f32 {
    let w = (d - 0.3).max(0.0);
    1.6 * std::f32::consts::FRAC_PI_4 * w * w
}

/// The voiced stops of each phrase, in order.
const PHRASES: [(&str, &[Phoneme]); 6] = [
    ("again", &[Phoneme::G]),
    ("begin", &[Phoneme::B, Phoneme::G]),
    ("a good", &[Phoneme::G, Phoneme::D]),
    ("the dog", &[Phoneme::D, Phoneme::G]),
    ("rugby", &[Phoneme::G, Phoneme::B]),
    ("bigger", &[Phoneme::B, Phoneme::G]),
];

/// The run of shut samples of the recording around the `nth` segment that
/// says `p`: `(closure start, release)`.
fn closure(r: &stops::Recording, p: Phoneme, nth: usize) -> Option<(usize, usize)> {
    let k = (0..r.phonemes.len())
        .filter(|&k| r.phonemes[k] == p && (k == 0 || r.phonemes[k - 1] != p))
        .nth(nth)?;
    let samples: Vec<usize> = (0..r.segment.len())
        .filter(|&s| r.segment[s] == k)
        .collect();
    let s = *samples.iter().find(|&&s| r.sealed[s])?;
    let (mut lo, mut hi) = (s, s);
    while lo > 0 && r.sealed[lo - 1] {
        lo -= 1;
    }
    while hi + 1 < r.sealed.len() && r.sealed[hi + 1] {
        hi += 1;
    }
    Some((lo, hi + 1))
}

fn voicing() {
    let base = stops::judge_voice();
    let mut no_widening = base;
    no_widening.tuning.anatomy.cavity_expansion = 0.0;
    let with = |t: Tuning| VoiceSettings { tuning: t, ..base };
    let cases: Vec<(&str, VoiceSettings)> = vec![
        ("today", base),
        ("larynx slide, not lowered", with(tuning(0.0, 1.0, 0.0))),
        ("larynx lowered 0.4", with(tuning(0.0, 1.0, 0.4))),
        ("larynx lowered 0.8", with(tuning(0.0, 1.0, 0.8))),
        ("no widening", no_widening),
        ("no widening, lowered 0.8", {
            let mut s = with(tuning(0.0, 1.0, 0.8));
            s.tuning.anatomy.cavity_expansion = 0.0;
            s
        }),
    ];
    // The volumes at stake, for the judge's voice.
    let mut v = Trombone::new(48_000.0);
    base.apply(&mut v);
    let size = v.tract_length();
    let d = v.diameters();
    let area = |k: usize| 1.6 * std::f32::consts::FRAC_PI_4 * (d[k] * size).powi(2);
    let slide = (area(7) + area(8) + area(9)) / 3.0;
    println!(
        "voice size {size:.2}: the larynx slide's area {slide:.2} cm², so {:.2} cm³ per cm lowered ({:.2} cm³ at 0.8 cm); the widening takes up to {:.1} cm³ at {:.0} cm³/s ({:.1} cm³ in a 58 ms closure)",
        slide * size,
        slide * size * 0.8,
        15.0 * size.powi(3),
        base.tuning.anatomy.cavity_expansion * size.powi(3),
        base.tuning.anatomy.cavity_expansion * size.powi(3) * 0.058
    );
    println!(
        "closure; folds vibrating from its start, and the share of it; voice bar dB re vowel; across the glottis 20 ms in and at the release, cmH2O; behind the closure at the release, cmH2O and share of below the glottis; burst dB re vowel"
    );
    for (text, stops_in) in PHRASES {
        for (name, settings) in &cases {
            let r = stops::record(settings, text);
            for (n, &p) in stops_in.iter().enumerate() {
                let nth = stops_in[..n].iter().filter(|&&q| q == p).count();
                let m = stops::stop_in(&r, p, nth);
                let Some((lo, hi)) = closure(&r, p, nth) else {
                    println!("{text:8} {name:26} {}: no closure", p.symbol());
                    continue;
                };
                let place = match p {
                    Phoneme::B => 0,
                    Phoneme::G => 2,
                    _ => 1,
                };
                // Pressures over the 10 ms before sample `s` (a voice's
                // cycle or more), cmH2O: behind the closure and below the
                // glottis.
                let at = |s: usize| {
                    let w = 480.min(s);
                    let span = s - w..s.max(1);
                    let n = span.len().max(1) as f32;
                    let mouth: f32 =
                        r.behind[span.clone()].iter().map(|b| b[place]).sum::<f32>() / n;
                    let sub: f32 = r.subglottal[span].iter().sum::<f32>() / n;
                    (mouth, sub)
                };
                let (m20, s20) = at((lo + 960).min(hi));
                let (mr, sr) = at(hi);
                // How long the folds kept vibrating into the closure, ms:
                // the glottal area swinging by 0.02 cm² over 12 ms.
                let swinging = |s: usize| {
                    let w =
                        &r.glottal_area[s.saturating_sub(288)..(s + 288).min(r.glottal_area.len())];
                    let lo = w.iter().copied().fold(f32::INFINITY, f32::min);
                    let hi = w.iter().copied().fold(0.0f32, f32::max);
                    hi - lo > 0.02
                };
                let vibrating =
                    (lo..hi).take_while(|&s| swinging(s)).count() as f32 / stops::SR * 1000.0;
                println!(
                    "{text:8} {name:26} {}: closure {:3.0} ms, vibrating {:3.0} ms ({:3.0} %), bar {:+5.1}, across {:4.1} / {:4.1}, behind {:4.1} ({:.2}), burst {:+5.1}",
                    p.symbol(),
                    m.closure,
                    vibrating,
                    100.0 * m.voiced_share,
                    m.voice_bar,
                    s20 - m20,
                    sr - mr,
                    mr,
                    m.pressure,
                    m.burst
                );
            }
        }
        println!();
    }
}

fn cost() {
    let text =
        "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background.";
    for (name, t) in [
        ("off", tuning(0.0, 0.0, 0.0)),
        ("lip horn 1.2", tuning(1.2, 0.0, 0.0)),
        ("larynx slide", tuning(0.0, 1.0, 0.8)),
        ("both", tuning(1.2, 1.0, 0.8)),
    ] {
        let settings = VoiceSettings {
            tuning: t,
            ..stops::judge_voice()
        };
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let mut voice = Trombone::new(48_000.0);
            settings.apply(&mut voice);
            voice.set_engine(Engine::Physical);
            Speaker::prepare(&mut voice);
            let mut speaker = Speaker::new();
            speaker.speak(Utterance::from_text(text, &settings.prosody()));
            let mut blocks = 0usize;
            let start = std::time::Instant::now();
            while speaker.is_speaking() {
                for _ in 0..512 {
                    std::hint::black_box(speaker.process(&mut voice));
                }
                blocks += 1;
            }
            let per = start.elapsed().as_secs_f64() / blocks as f64;
            best = best.min(per);
        }
        println!(
            "{name:14}: {:.3} ms per 512-frame callback ({:.1} % of its 10.7 ms)",
            1000.0 * best,
            100.0 * best / (512.0 / 48_000.0)
        );
    }
}

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("formants") => formants(),
        Some("solve") => solve(),
        Some("voicing") => voicing(),
        Some("cost") => cost(),
        _ => {
            formants();
            solve();
            voicing();
            cost();
        }
    }
}
