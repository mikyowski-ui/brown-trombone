//! Print the phonemes the text front end gives each line of standard input.
//!
//! ```sh
//! printf 'canoe\nfire\n' | cargo run -p synthie-trombone --example phonemize --release
//! ```
//!
//! Each output line is the input line, a tab, and the ARPAbet phonemes with
//! `1` after the stressed vowel: `canoe\tK AX N UW1`. Useful for checking the
//! front end on long word lists without rendering any sound. With `--rules`
//! the built-in dictionary is left out, to check the spelling rules alone.

use std::io::{BufRead, Write};
use synthie_trombone::speech::{Token, phonemize, phonemize_with};

fn main() -> std::io::Result<()> {
    let rules_only = std::env::args().any(|a| a == "--rules");
    let stdin = std::io::stdin();
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for line in stdin.lock().lines() {
        let line = line?;
        let mut phones = Vec::new();
        let tokens = if rules_only {
            phonemize_with(&line, &|_| None)
        } else {
            phonemize(&line)
        };
        for t in tokens {
            if let Token::Phone { phoneme, stressed } = t {
                phones.push(format!("{phoneme}{}", if stressed { "1" } else { "" }));
            }
        }
        writeln!(out, "{line}\t{}", phones.join(" "))?;
    }
    out.flush()
}
