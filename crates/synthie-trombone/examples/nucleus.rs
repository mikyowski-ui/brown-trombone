//! What the physical engine's vowels are in running speech: for every vowel
//! of every line of text, its length and the tract's F1, F2 and F3 at five
//! points through it (10, 30, 50, 70 and 90 % of the vowel, diphthongs
//! whole), measured from the transfer function of the tract as it was at
//! that moment, beside the formants the vowel's shape settles to.
//!
//! ```sh
//! cargo run -p synthie-trombone --example nucleus --release -- lines.txt > vowels.tsv
//! cargo run -p synthie-trombone --example nucleus --release -- --profile p.json lines.txt
//! ```
//!
//! One line of text per line of the file (bracketed ARPAbet works); the
//! output is one tab-separated row per vowel: line, phoneme, stressed,
//! the phonemes before and after, length (ms), the settled F1 F2 F3, then
//! F1 F2 F3 at each of the five points.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

#[path = "support/spectrum.rs"]
mod spectrum;

use std::collections::HashMap;
use std::fmt::Write as _;
use synthie_trombone::speech::{Speaker, Token, Utterance, phonemize};
use synthie_trombone::vowel_targets::VowelKey;
use synthie_trombone::{Articulation, Engine, Manner, Phoneme, Preset, Trombone, VoiceSettings};

const SR: f32 = 48_000.0;
const POINTS: [f32; 5] = [0.1, 0.3, 0.5, 0.7, 0.9];

fn is_vowel(p: Phoneme) -> bool {
    matches!(p.manner(), Manner::Vowel | Manner::Diphthong)
}

/// F1, F2 and F3 of the voice's tract as it is now.
fn formants(voice: &Trombone) -> [f32; 3] {
    let step = 6.0;
    let rate = voice.tick_rate();
    let mut x = vec![0.0f32; (0.06 * rate) as usize];
    voice.impulse_response(&mut x);
    let mag = spectrum::spectrum(&x, rate, 4000.0, step);
    let r = spectrum::resonances(&mag, step);
    let mut f = r.iter().map(|r| r.hz).filter(|&hz| hz > 150.0);
    [
        f.next().unwrap_or(0.0),
        f.next().unwrap_or(0.0),
        f.next().unwrap_or(0.0),
    ]
}

fn settled(settings: &VoiceSettings, a: &Articulation) -> [f32; 3] {
    let mut t = Trombone::new(SR);
    settings.apply(&mut t);
    t.articulate(a);
    t.settle();
    formants(&t)
}

fn line(settings: &VoiceSettings, k: usize, text: &str) -> String {
    let tokens = phonemize(text);
    let phones: Vec<(Phoneme, bool)> = tokens
        .iter()
        .filter_map(|t| match *t {
            Token::Phone { phoneme, stressed } => Some((phoneme, stressed)),
            _ => None,
        })
        .collect();
    let mut voice = Trombone::new(SR);
    settings.apply(&mut voice);
    Speaker::prepare(&mut voice);
    let utterance = Utterance::from_text(text, &settings.prosody());
    let segments = utterance.segments().to_vec();
    // Each vowel: its segments (two for a diphthong).
    let mut vowels: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < segments.len() {
        let p = segments[i].phoneme;
        if is_vowel(p) {
            // A diphthong, or a vowel that glides to an end of its own.
            let glides = segments
                .get(i + 1)
                .is_some_and(|s| s.phoneme == p && s.articulation.vowel == Some(VowelKey::End(p)));
            let n = if glides { 2 } else { 1 };
            vowels.push((i, (i + n).min(segments.len())));
            i += n;
        } else {
            i += 1;
        }
    }
    let starts: Vec<usize> = segments
        .iter()
        .scan(0usize, |t, s| {
            let at = *t;
            *t += (s.duration * SR).round() as usize;
            Some(at)
        })
        .collect();
    let mut speaker = Speaker::new();
    speaker.speak(utterance);
    let mut out = String::new();
    let mut goal_cache: HashMap<String, [f32; 3]> = HashMap::new();
    let mut tracks: Vec<Vec<[f32; 3]>> = vec![Vec::new(); vowels.len()];
    let times: Vec<Vec<usize>> = vowels
        .iter()
        .map(|&(a, b)| {
            let from = starts[a];
            let len: f32 = segments[a..b].iter().map(|s| s.duration).sum();
            POINTS
                .iter()
                .map(|f| from + (f * len * SR) as usize)
                .collect()
        })
        .collect();
    let mut n = 0usize;
    while speaker.is_speaking() {
        speaker.process(&mut voice);
        n += 1;
        for (v, ts) in times.iter().enumerate() {
            if ts.contains(&n) {
                tracks[v].push(formants(&voice));
            }
        }
    }
    let phone_of = |seg: usize| segments[seg].phoneme;
    let mut vowel_no = 0usize;
    let mut phone_at = Vec::new();
    for (j, &(p, _)) in phones.iter().enumerate() {
        if is_vowel(p) {
            phone_at.push(j);
        }
    }
    for (v, &(a, b)) in vowels.iter().enumerate() {
        let p = phone_of(a);
        let (stressed, before, after) = phone_at.get(vowel_no).map_or((false, "-", "-"), |&j| {
            let before = j.checked_sub(1).map_or("-", |q| phones[q].0.symbol());
            let after = phones.get(j + 1).map_or("-", |q| q.0.symbol());
            (phones[j].1, before, after)
        });
        vowel_no += 1;
        let key = format!("{:?}", segments[a].articulation);
        let goal = *goal_cache
            .entry(key)
            .or_insert_with(|| settled(settings, &segments[a].articulation));
        let len: f32 = segments[a..b].iter().map(|s| s.duration).sum();
        let _ = write!(
            out,
            "{k}\t{}\t{}\t{before}\t{after}\t{:.0}\t{:.0}\t{:.0}\t{:.0}",
            p.symbol(),
            u8::from(stressed),
            1000.0 * len,
            goal[0],
            goal[1],
            goal[2]
        );
        for f in &tracks[v] {
            let _ = write!(out, "\t{:.0}\t{:.0}\t{:.0}", f[0], f[1], f[2]);
        }
        out.push('\n');
    }
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut settings = Preset::Deep.settings().with_engine(Engine::Physical);
    let mut file = None;
    while let Some(a) = args.next() {
        if a == "--profile" {
            let path = args.next().expect("--profile FILE");
            let text = std::fs::read_to_string(&path).expect("profile");
            settings = settings.with_profile(&text).expect("profile");
        } else {
            file = Some(a);
        }
    }
    let text = std::fs::read_to_string(file.expect("a file of lines")).expect("lines");
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let threads = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);
    let chunks: Vec<Vec<(usize, &str)>> = (0..threads)
        .map(|t| {
            lines
                .iter()
                .enumerate()
                .filter(|(k, _)| k % threads == t)
                .map(|(k, l)| (k, *l))
                .collect()
        })
        .collect();
    let mut rows: Vec<(usize, String)> = std::thread::scope(|s| {
        let handles: Vec<_> = chunks
            .iter()
            .map(|c| {
                s.spawn(|| {
                    c.iter()
                        .map(|&(k, l)| (k, line(&settings, k, l)))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("thread"))
            .collect()
    });
    rows.sort_by_key(|r| r.0);
    for (_, r) in rows {
        print!("{r}");
    }
}
