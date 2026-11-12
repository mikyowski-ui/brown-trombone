//! Build the text front end's dictionary (`src/speech/core.dict` and
//! `extra.dict`) from the CMU Pronouncing Dictionary.
//!
//! ```sh
//! cargo run -p synthie-trombone --example lexicon --release -- \
//!     --cmudict cmudict.dict --ranked words-by-frequency.txt
//! ```
//!
//! * `--cmudict`: the CMU Pronouncing Dictionary (`cmudict.dict` from
//!   <https://github.com/cmusphinx/cmudict>, BSD licence; its notice is in
//!   `src/speech/CMUDICT-LICENSE`).
//! * `--ranked`: English words, most frequent first, one per line. The
//!   shipped files used the whole-word tokens of the GPT-2 byte-pair
//!   vocabulary (MIT licence) in merge order, a rough frequency ranking.
//! * `--top N` (default 20000): how many ranked words are considered.
//! * `--core-bytes N` (default 32000): size of the always-present part.
//!
//! A word is added when the front end, given the entries chosen so far,
//! says it differently from `CMUdict` (ignoring unaccented-vowel quality and
//! other notation that does not change the word). Passes repeat until
//! nothing changes, since a new entry can change how other words are taken
//! apart. The hand-written `src/speech/manual.txt` comes first and wins.
//! The most frequent entries go to `core.dict` (always built in), the rest
//! to `extra.dict` (cargo feature `dictionary`).

use std::collections::HashMap;
use std::fmt::Write as _;
use synthie_trombone::speech::{Token, phonemize_with};

/// Same table as the front end's `CODES`.
const CODES: [(char, &str); 40] = [
    ('A', "AA"),
    ('a', "AE"),
    ('V', "AH"),
    ('O', "AO"),
    ('@', "AX"),
    ('E', "EH"),
    ('3', "ER"),
    ('I', "IH"),
    ('i', "IY"),
    ('U', "UH"),
    ('u', "UW"),
    ('Y', "AY"),
    ('W', "AW"),
    ('e', "EY"),
    ('o', "OW"),
    ('y', "OY"),
    ('p', "P"),
    ('b', "B"),
    ('t', "T"),
    ('d', "D"),
    ('k', "K"),
    ('g', "G"),
    ('m', "M"),
    ('n', "N"),
    ('N', "NG"),
    ('f', "F"),
    ('v', "V"),
    ('T', "TH"),
    ('D', "DH"),
    ('s', "S"),
    ('z', "Z"),
    ('S', "SH"),
    ('Z', "ZH"),
    ('C', "CH"),
    ('J', "JH"),
    ('l', "L"),
    ('r', "R"),
    ('w', "W"),
    ('j', "Y"),
    ('h', "HH"),
];

/// A phoneme with its CMU stress digit (`None` for consonants).
type Sym = (String, Option<u8>);

fn main() {
    let mut cmu_path = None;
    let mut ranked_path = None;
    let mut top = 20_000usize;
    let mut core_bytes = 32_000usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--cmudict" => cmu_path = Some(value()),
            "--ranked" => ranked_path = Some(value()),
            "--top" => top = value().parse().expect("--top needs a number"),
            "--core-bytes" => core_bytes = value().parse().expect("--core-bytes needs a number"),
            _ => panic!("unknown argument {a}"),
        }
    }
    let cmu_path = cmu_path.expect("--cmudict is required");
    let ranked_path = ranked_path.expect("--ranked is required");
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/speech");

    let manual = read_manual(&format!("{dir}/manual.txt"));
    let cmu = read_cmudict(&cmu_path);
    let ranked: Vec<String> = std::fs::read_to_string(&ranked_path)
        .expect("cannot read --ranked")
        .lines()
        .map(str::trim)
        .filter(|w| cmu.contains_key(*w))
        .take(top)
        .map(String::from)
        .collect();
    let rank: HashMap<&str, usize> = ranked
        .iter()
        .enumerate()
        .map(|(k, w)| (w.as_str(), k))
        .collect();

    // Chosen entries, as ARPAbet with 1 on the accent.
    let mut chosen: HashMap<String, String> = manual.iter().cloned().collect();
    for pass in 1..=6 {
        let mut added = 0;
        let mut removed = 0;
        for w in &ranked {
            let in_manual = manual.iter().any(|(m, _)| m == w);
            if in_manual {
                continue;
            }
            let reference = &cmu[w];
            // How the front end says it without this word's own entry.
            let had = chosen.remove(w);
            let said = say(w, &chosen);
            let wrong = differs(reference, &said);
            if wrong {
                chosen.insert(w.clone(), arpabet(reference));
                added += usize::from(had.is_none());
            } else if had.is_some() {
                removed += 1;
            }
        }
        println!(
            "pass {pass}: {} entries (+{added} -{removed})",
            chosen.len()
        );
        if added == 0 && removed == 0 {
            break;
        }
    }

    // Split by frequency: manual and the most frequent entries are core.
    let mut entries: Vec<(&String, &String)> = chosen.iter().collect();
    entries.sort_by_key(|(w, _)| {
        let manual_first = !manual.iter().any(|(m, _)| m == *w);
        (
            manual_first,
            rank.get(w.as_str()).copied().unwrap_or(usize::MAX),
        )
    });
    let mut core = Vec::new();
    let mut extra = Vec::new();
    let mut size = 0;
    for (w, a) in entries {
        let line = format!("{w} {}\n", encode(a));
        if size + line.len() <= core_bytes {
            size += line.len();
            core.push(line);
        } else {
            extra.push(line);
        }
    }
    for (name, mut lines) in [("core", core), ("extra", extra)] {
        lines.sort_by(|a, b| a.split(' ').next().cmp(&b.split(' ').next()));
        let text: String = lines.concat();
        let path = format!("{dir}/{name}.dict");
        std::fs::write(&path, &text).expect("cannot write dictionary");
        println!("{path}: {} words, {} bytes", lines.len(), text.len());
    }
}

fn read_manual(path: &str) -> Vec<(String, String)> {
    std::fs::read_to_string(path)
        .expect("cannot read manual.txt")
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let (w, a) = l.split_once(' ').expect("word and phonemes");
            (w.to_string(), a.trim().to_string())
        })
        .collect()
}

/// First pronunciation of each word, `AH0` as `AX`.
fn read_cmudict(path: &str) -> HashMap<String, Vec<Sym>> {
    let text = std::fs::read_to_string(path).expect("cannot read --cmudict");
    let mut out = HashMap::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("");
        let mut it = line.split_whitespace();
        let Some(word) = it.next() else { continue };
        if word.contains('(') {
            continue;
        }
        let syms: Vec<Sym> = it
            .map(|p| {
                let digit = p
                    .bytes()
                    .last()
                    .filter(u8::is_ascii_digit)
                    .map(|d| d - b'0');
                let base = p.trim_end_matches(|c: char| c.is_ascii_digit());
                let base = if base == "AH" && digit == Some(0) {
                    "AX"
                } else {
                    base
                };
                (base.to_string(), digit)
            })
            .collect();
        out.entry(word.to_string()).or_insert(syms);
    }
    out
}

/// ARPAbet with `1` on the primary accent only.
fn arpabet(syms: &[Sym]) -> String {
    syms.iter()
        .map(|(p, d)| {
            if *d == Some(1) {
                format!("{p}1")
            } else {
                p.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn encode(arpabet: &str) -> String {
    let mut out = String::new();
    for sym in arpabet.split_whitespace() {
        let base = sym.trim_end_matches('1');
        if sym.ends_with('1') {
            out.push('\'');
        }
        let c = CODES
            .iter()
            .find(|(_, s)| *s == base)
            .unwrap_or_else(|| panic!("no code for {sym}"))
            .0;
        let _ = write!(out, "{c}");
    }
    out
}

/// The front end's phonemes for `word`, given `entries` as its dictionary.
fn say(word: &str, entries: &HashMap<String, String>) -> Vec<Sym> {
    phonemize_with(word, &|w| entries.get(w).map(String::as_str))
        .into_iter()
        .filter_map(|t| match t {
            Token::Phone { phoneme, stressed } => {
                let vowel = matches!(
                    phoneme.manner(),
                    synthie_trombone::Manner::Vowel | synthie_trombone::Manner::Diphthong
                );
                Some((
                    phoneme.symbol().to_string(),
                    vowel.then_some(u8::from(stressed)),
                ))
            }
            _ => None,
        })
        .collect()
}

/// Notation that does not change a word: r-coloured schwa written two
/// ways, tense and lax vowels before r, n before k and g.
fn canonical(syms: &[Sym]) -> Vec<Sym> {
    let mut out: Vec<Sym> = Vec::new();
    for (k, s) in syms.iter().enumerate() {
        let next = syms.get(k + 1).map(|(p, _)| p.as_str());
        let mut s = s.clone();
        if s.0 == "R"
            && out
                .last()
                .is_some_and(|(p, _)| matches!(p.as_str(), "ER" | "AX" | "AH"))
        {
            out.last_mut().unwrap().0 = "ER".into();
            continue;
        }
        if next == Some("R") {
            let lax = match s.0.as_str() {
                "IY" => Some("IH"),
                "UW" => Some("UH"),
                "OW" => Some("AO"),
                "EY" => Some("EH"),
                _ => None,
            };
            if let Some(l) = lax {
                s.0 = l.into();
            }
        }
        if s.0 == "N" && matches!(next, Some("K" | "G")) {
            s.0 = "NG".into();
        }
        out.push(s);
    }
    out
}

fn same(a: &Sym, b: &Sym) -> bool {
    if a.0 == b.0 {
        return true;
    }
    let pair = |x: &str, y: &str| (a.0 == x && b.0 == y) || (a.0 == y && b.0 == x);
    let reduced = a.1 == Some(0) || b.1 == Some(0);
    pair("AX", "AH")
        || pair("AX", "IH")
        || pair("AH", "IH")
        || pair("IY", "IH")
        || pair("AA", "AO")
        // Unaccented vowel quality varies between dictionaries.
        || reduced && (pair("EH", "IH") || pair("EH", "AX") || pair("UH", "AX") || pair("UH", "UW") || pair("IY", "AX"))
}

/// Whether the front end's `said` is a different word from `reference`.
fn differs(reference: &[Sym], said: &[Sym]) -> bool {
    let (refs, hyps) = (canonical(reference), canonical(said));
    // Edit distance, one row at a time.
    let mut prev: Vec<usize> = (0..=hyps.len()).collect();
    for (row, r) in refs.iter().enumerate() {
        let mut cur = vec![row + 1; hyps.len() + 1];
        for (col, h) in hyps.iter().enumerate() {
            let cost = usize::from(!same(r, h));
            cur[col + 1] = (prev[col] + cost).min(prev[col + 1] + 1).min(cur[col] + 1);
        }
        prev = cur;
    }
    prev[hyps.len()] > 0
}
