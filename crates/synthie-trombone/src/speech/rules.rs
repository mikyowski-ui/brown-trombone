//! English spelling rules: the sounds of a word the dictionary does not know.
//!
//! A word goes through three passes, in the spirit of the NRL rules
//! (Elovitz et al., 1976) but with stress, which those rules lack:
//!
//! 1. **Spellings:** the letters are cut into consonant spellings (`ch`,
//!    `tion`, `kn`, soft `c` ...) whose sounds are fixed at once, and vowel
//!    nuclei (`ea`, `ir`, magic *e* ...) that keep a stressed and an
//!    unstressed sound.
//! 2. **Stress:** one nucleus takes the accent, chosen from the word's
//!    endings (`-tion` stresses the syllable before it), prefixes (`be-`,
//!    `con-`), and syllable weight (the Latin rule for long words). Every
//!    other syllable before it takes a secondary accent.
//! 3. **Vowels:** accented vowels get their full sound, long or short by
//!    the syllable they sit in; unaccented ones reduce, mostly to schwa.
//!
//! The rules are tuned on frequent English words, not on any list the
//! front end is checked against.

use crate::phoneme::Phoneme;
use alloc::vec::Vec;
use Phoneme as P;

/// Accent of a nucleus.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stress {
    None,
    Secondary,
    Primary,
}

/// How a single vowel letter's accented sound is chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Length {
    /// Short (closed syllable).
    Short,
    /// Long (digraph, magic *e*, end of word).
    Long,
    /// One consonant before the next vowel: long or short by position.
    Open,
}

/// What a nucleus does to the accent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Hint {
    Normal,
    /// Takes the accent (`-ee`, `-oon`, `-ique`).
    Takes,
    /// Never accented (syllabic `-le`, `-ism`).
    Never,
}

#[derive(Clone, Copy, Debug)]
struct Vowel {
    short: &'static [Phoneme],
    long: &'static [Phoneme],
    weak: &'static [Phoneme],
    length: Length,
    hint: Hint,
    stress: Stress,
    /// The vowel letter, for the long-or-short choice of open syllables.
    letter: u8,
    /// Spelled with a long vowel or closed by its own consonant (`ar`).
    heavy: bool,
}

impl Vowel {
    const fn new(short: &'static [Phoneme], long: &'static [Phoneme], weak: &'static [Phoneme]) -> Self {
        Self {
            short,
            long,
            weak,
            length: Length::Short,
            hint: Hint::Normal,
            stress: Stress::None,
            letter: 0,
            heavy: false,
        }
    }

    /// A long vowel (digraph): the same sound accented or not unless `weak`
    /// says otherwise.
    const fn long(full: &'static [Phoneme], weak: &'static [Phoneme]) -> Self {
        let mut v = Self::new(full, full, weak);
        v.length = Length::Long;
        v.heavy = true;
        v
    }

    const fn fixed(ph: &'static [Phoneme]) -> Self {
        Self::long(ph, ph)
    }

    const fn hint(mut self, hint: Hint) -> Self {
        self.hint = hint;
        self
    }

    const fn light(mut self) -> Self {
        self.heavy = false;
        self
    }
}

#[derive(Clone, Copy, Debug)]
enum Unit {
    Cons(&'static [Phoneme]),
    Vowel(Vowel),
}

#[derive(Clone, Copy, Debug)]
struct Piece {
    unit: Unit,
    /// First letter of the spelling.
    start: usize,
    /// One past its last letter.
    end: usize,
}

/// Endings that do not move the accent of what they are added to.
const NEUTRAL: &[&str] = &[
    "ness", "less", "fully", "ful", "ments", "ment", "ably", "able", "ism", "ists", "ist", "ly", "ings",
    "ing", "ship", "hood",
];

/// Endings that put the accent on the syllable just before them.
const BEFORE_ONE: &[&str] = &[
    "tion", "sion", "cion", "xion", "cian", "tian", "sian", "tial", "cial", "tious", "cious", "xious",
    "geous", "gious", "gion", "ient", "ience", "iency", "ial", "ially", "ian", "ion", "ious", "eous",
    "uous", "ual", "ually", "uent", "uence", "ic", "ics", "ical", "ically", "ity", "ety", "ify", "ible",
    "ibly", "itive", "ative", "itude", "ular", "ulate", "ulous", "ia", "ium", "ior", "iour", "io",
    "logy", "graphy", "nomy", "metry", "pathy", "phony", "sophy", "meter",
];

/// Endings that put the accent two syllables before them.
const BEFORE_TWO: &[&str] = &[
    "ate", "ates", "ize", "izes", "ise", "ary", "ery", "ory", "ite", "ude", "ute", "ency", "ancy",
];

/// Endings that take the accent themselves.
const SELF: &[&str] = &[
    "ee", "eer", "oon", "ique", "esque", "ette", "ese", "aire", "oo", "self", "selves", "eur", "euse",
    "ade", "iety",
];

/// Prefixes that do not take the accent of a word of two or three syllables.
const PREFIXES: &[&str] = &[
    "be", "de", "re", "pre", "pro", "con", "com", "col", "cor", "dis", "ex", "em", "en", "im", "in",
    "mis", "ob", "per", "sub", "suc", "suf", "sup", "sus", "un", "ad", "ab", "sur", "trans", "es",
];

/// The sounds of `word` (lower-case ASCII letters), each with whether it
/// carries the word's accent.
///
/// `verb` says the word is known to be a verb (it had `-ed` or `-ing`): its
/// `-ate` keeps the full vowel (`separate` the verb, not the adjective).
pub(super) fn spell(word: &[u8], verb: bool) -> Vec<(Phoneme, bool)> {
    let w: Vec<u8> = word.iter().copied().filter(u8::is_ascii_lowercase).collect();
    if w.is_empty() {
        return Vec::new();
    }
    let mut spelling = Spelling::new(&w);
    spelling.verb = verb;
    let mut pieces = spelling.cut();
    accent(&w, &mut pieces);
    sound(&w, &pieces)
}

// ===== 1. spellings ========================================================

fn is_vowel_letter(c: u8) -> bool {
    matches!(c, b'a' | b'e' | b'i' | b'o' | b'u' | b'y')
}

struct Spelling<'a> {
    w: &'a [u8],
    n: usize,
    out: Vec<Piece>,
    verb: bool,
}

impl<'a> Spelling<'a> {
    fn new(w: &'a [u8]) -> Self {
        Self {
            w,
            n: w.len(),
            out: Vec::new(),
            verb: false,
        }
    }

    fn at(&self, i: usize) -> u8 {
        self.w.get(i).copied().unwrap_or(0)
    }

    fn before(&self, i: usize) -> u8 {
        if i == 0 { 0 } else { self.w[i - 1] }
    }

    fn starts(&self, i: usize, s: &str) -> bool {
        self.w.get(i..).is_some_and(|r| r.starts_with(s.as_bytes()))
    }

    /// The word from `i` on is exactly `s`.
    fn ends(&self, i: usize, s: &str) -> bool {
        self.w.get(i..).is_some_and(|r| r == s.as_bytes())
    }

    fn rest(&self, i: usize) -> &[u8] {
        self.w.get(i..).unwrap_or(&[])
    }

    /// Whether the letter at `i` is used as a vowel.
    fn vowel(&self, i: usize) -> bool {
        let c = self.at(i);
        match c {
            b'a' | b'e' | b'i' | b'o' => true,
            // u after q (and before a vowel after g) is part of the consonant.
            b'u' => !(self.before(i) == b'q' || (self.before(i) == b'g' && is_vowel_letter(self.at(i + 1)))),
            // y before a vowel is a consonant: yes, beyond, canyon.
            b'y' => !is_vowel_letter(self.at(i + 1)) || (i > 0 && !is_vowel_letter(self.before(i)) && self.at(i + 1) == b'e' && i + 2 == self.n),
            _ => false,
        }
    }

    /// Any vowel letter before `i`.
    fn vowel_before(&self, i: usize) -> bool {
        (0..i).any(|k| self.vowel(k))
    }

    /// The final `e` at `i` is silent.
    fn silent_e(&self, i: usize) -> bool {
        i + 1 == self.n && self.at(i) == b'e' && self.vowel_before(i) && !is_vowel_letter(self.before(i))
    }

    /// Consonant letters from `i` up to the next vowel letter.
    fn consonants_from(&self, i: usize) -> usize {
        (i..self.n).take_while(|&k| !self.vowel(k)).count()
    }

    /// A magic `e`: after the vowel at `i` come one consonant spelling and a
    /// silent final `e`. Returns the letters of the consonant spelling.
    fn magic(&self, i: usize) -> Option<usize> {
        let c = self.at(i + 1);
        let len = if self.starts(i + 1, "th") || self.starts(i + 1, "ch") {
            2
        } else if (self.at(i) == b'a' && (self.starts(i + 1, "st") || self.starts(i + 1, "ng")))
            || (self.at(i) == b'a' && self.starts(i + 1, "nge"))
        {
            2
        } else if matches!(c, b'b' | b'c' | b'd' | b'f' | b'g' | b'k' | b'l' | b'm' | b'n' | b'p' | b's' | b't' | b'v' | b'z') {
            1
        } else {
            return None;
        };
        (self.silent_e(i + 1 + len)).then_some(len)
    }

    fn push(&mut self, unit: Unit, start: usize, end: usize) {
        self.out.push(Piece { unit, start, end });
    }

    fn cut(mut self) -> Vec<Piece> {
        let mut i = 0;
        // un- and non- before a consonant: an unaccented full vowel.
        // over-, under-, inter- before a consonant keep their own sounds.
        for (pre, ph) in [
            ("over", &[P::Ow, P::V, P::Er][..]),
            ("under", &[P::Ah, P::N, P::D, P::Er][..]),
            ("inter", &[P::Ih, P::N, P::T, P::Er][..]),
        ] {
            let k = pre.len();
            if self.starts(0, pre) && self.n >= k + 3 && !self.vowel(k) && self.rest(k).iter().any(|&c| is_vowel_letter(c)) && !self.starts(0, "interest") && !self.starts(0, "interpret") {
                let (head, tail) = ph.split_at(ph.len() - 1);
                for (j, &p) in head.iter().enumerate() {
                    let unit = if matches!(p, P::Ow | P::Ah | P::Ih) { Unit::Vowel(Vowel::fixed(&ph[j..=j]).hint(Hint::Never)) } else { Unit::Cons(&ph[j..=j]) };
                    self.push(unit, j.min(k - 1), (j + 1).min(k - 1));
                }
                self.push(Unit::Vowel(Vowel::fixed(tail).hint(Hint::Never)), k - 2, k);
                i = k;
            }
        }
        // un-, non-, up- before a consonant: an unaccented full vowel.
        for (pre, v, c) in [
            ("non", Vowel::fixed(&[P::Aa]), &[P::N][..]),
            ("un", Vowel::fixed(&[P::Ah]), &[P::N][..]),
            ("up", Vowel::fixed(&[P::Ah]), &[P::P][..]),
        ] {
            let k = pre.len();
            if i == 0 && self.starts(0, pre) && self.n > k + 2 && (!self.vowel(k) || pre == "un" && self.at(k) != b'i') && self.at(k) != self.at(k - 1) && !self.starts(0, "uncle") && !self.starts(0, "under") && self.rest(k).iter().any(|&c| is_vowel_letter(c)) {
                let first = k - 2;
                if first == 1 {
                    self.push(Unit::Cons(&[P::N]), 0, 1);
                }
                self.push(Unit::Vowel(v.hint(Hint::Never)), first, first + 1);
                // n before a k or hard g sound: unconscious, nonconformity.
                let velar = matches!(self.at(k), b'k' | b'q' | b'x') && !self.starts(k, "kn") || matches!(self.at(k), b'c' | b'g') && !matches!(self.at(k + 1), b'e' | b'i' | b'y');
                let c: &'static [Phoneme] = if c == [P::N] && velar { &[P::Ng] } else { c };
                self.push(Unit::Cons(c), first + 1, k);
                i = k;
            }
        }
        while i < self.n {
            let len = if self.vowel(i) {
                self.vowel_spelling(i)
            } else {
                self.consonant_spelling(i)
            };
            i += len.max(1);
        }
        self.out
    }

    // ---- vowels ----------------------------------------------------------

    /// Push the vowel spelling at `i`; returns the letters used.
    #[allow(clippy::too_many_lines)] // one rule table
    fn vowel_spelling(&mut self, i: usize) -> usize {
        let c = self.at(i);
        let next = self.at(i + 1);
        let after = self.at(i + 2);
        let end_or_cons = |k: usize| k >= self.n || !self.vowel(k);
        // r closes the syllable: not followed by a vowel or another r.
        let r_closes = |k: usize| self.at(k) == b'r' && self.at(k + 1) != b'r' && end_or_cons(k + 1);
        let (v, len): (Vowel, usize) = match c {
            // ---- final silent e and syllabic endings ----
            b'e' if self.silent_e(i) => return 1,
            b'e' if i + 2 == self.n && next == b's' && i > 1 && self.vowel_before(i) => {
                let b = self.before(i);
                if matches!(b, b's' | b'z' | b'x' | b'c' | b'g') || self.starts(i - 2, "ch") || self.starts(i - 2, "sh") {
                    (Vowel::fixed(&[P::Ih]).hint(Hint::Never).light(), 1)
                } else {
                    return 1;
                }
            }
            // ---- three and four letter spellings ----
            b'e' if self.starts(i, "eau") => (Vowel::long(&[P::Ow], &[P::Ow]), 3),
            b'e' if self.starts(i, "eigh") => (Vowel::long(&[P::Ey], &[P::Ey]), 4),
            b'a' if self.starts(i, "aigh") => (Vowel::long(&[P::Ey], &[P::Ey]), 4),
            b'i' if self.starts(i, "igh") => (Vowel::long(&[P::Ay], &[P::Ay]), 3),
            b'o' if self.starts(i, "ought") => (Vowel::long(&[P::Ao], &[P::Ao]), 4),
            b'o' if self.starts(i, "ough") && self.before(i) == b'r' && i >= 2 && self.vowel_before(i - 1) => {
                (Vowel::long(&[P::Ow], &[P::Ow]), 4)
            }
            b'o' if self.starts(i, "ough") => (Vowel::long(&[P::Ah, P::F], &[P::Ax, P::F]), 4),
            b'a' if self.starts(i, "augh") && self.at(i + 4) == b't' => (Vowel::long(&[P::Ao], &[P::Ao]), 4),
            b'a' if self.starts(i, "augh") => (Vowel::long(&[P::Ae, P::F], &[P::Ae, P::F]), 4),
            b'y' if self.ends(i, "ye") && i > 0 => (Vowel::long(&[P::Ay], &[P::Ay]), 2),
            // ---- r-coloured ----
            b'a' if self.starts(i, "air") => (Vowel::long(&[P::Eh, P::R], &[P::Er]), 3),
            b'e' if self.starts(i, "ear") && r_closes(i + 2) => {
                if self.ends(i + 3, "") || self.ends(i + 3, "s") || matches!(self.at(i + 3), b'd' if self.ends(i + 3, "d") && self.before(i) == b'b') {
                    (Vowel::long(&[P::Ih, P::R], &[P::Ih, P::R]), 3)
                } else if self.at(i + 3) == b't' && self.at(i + 4) != b'h' {
                    (Vowel::long(&[P::Aa, P::R], &[P::Aa, P::R]), 3)
                } else {
                    (Vowel::long(&[P::Er], &[P::Er]), 3)
                }
            }
            b'e' if self.starts(i, "eer") || (self.starts(i, "ier") && i + 3 == self.n && i > 1 && self.before(i - 1) == b'h') => {
                (Vowel::long(&[P::Ih, P::R], &[P::Ih, P::R]).hint(if self.starts(i, "eer") && i > 1 { Hint::Takes } else { Hint::Normal }), 3)
            }
            b'e' if self.starts(i, "eir") => (Vowel::long(&[P::Eh, P::R], &[P::Er]), 3),
            b'o' if self.starts(i, "oar") || (self.starts(i, "oor") && r_closes(i + 2)) => (Vowel::long(&[P::Ao, P::R], &[P::Er]), 3),
            b'o' if self.starts(i, "our") && r_closes(i + 2) => {
                if i + 3 == self.n {
                    (Vowel::long(&[P::Aw, P::Er], &[P::Er]), 3)
                } else if matches!(self.at(i + 3), b'n' | b'j') && self.at(i + 4) != b'e' {
                    (Vowel::long(&[P::Er], &[P::Er]), 3)
                } else {
                    (Vowel::long(&[P::Ao, P::R], &[P::Er]), 3)
                }
            }
            b'a' if next == b'r' && self.ends(i + 2, "e") => (Vowel::long(&[P::Eh, P::R], &[P::Er]), 3),
            b'i' | b'y' if next == b'r' && self.ends(i + 2, "e") => (Vowel::long(&[P::Ay, P::Er], &[P::Ay, P::Er]), 3),
            b'o' if next == b'r' && self.ends(i + 2, "e") => (Vowel::long(&[P::Ao, P::R], &[P::Ao, P::R]), 3),
            b'e' if next == b'r' && self.ends(i + 2, "e") && i > 0 => (Vowel::long(&[P::Ih, P::R], &[P::Ih, P::R]), 3),
            b'u' if next == b'r' && self.ends(i + 2, "e") => {
                let b = self.before(i);
                if b == b't' || b == b's' || b == b'z' || b == b'd' {
                    // Already palatalised by the consonant: nature, measure.
                    (Vowel::long(&[P::Uh, P::R], &[P::Er]), 3)
                } else {
                    (Vowel::long(&[P::Y, P::Uh, P::R], &[P::Y, P::Er]), 3)
                }
            }
            b'a' if r_closes(i + 1) => {
                let w = self.before(i) == b'w' || (self.before(i) == b'u' && i > 1 && self.at(i - 2) == b'q');
                if w {
                    (Vowel::long(&[P::Ao, P::R], &[P::Er]), 2)
                } else {
                    (Vowel::long(&[P::Aa, P::R], &[P::Er]), 2)
                }
            }
            b'o' if r_closes(i + 1) => {
                if self.before(i) == b'w' && !(self.ends(i + 2, "n") || self.ends(i + 2, "d") && i == 2) {
                    (Vowel::long(&[P::Er], &[P::Er]), 2)
                } else {
                    (Vowel::long(&[P::Ao, P::R], &[P::Er]), 2)
                }
            }
            b'e' | b'i' | b'u' | b'y' if r_closes(i + 1) => (Vowel::long(&[P::Er], &[P::Er]), 2),
            // ---- digraphs ----
            b'a' if next == b'i' || (next == b'y' && !(self.ends(i + 2, "") && self.vowel(i + 2))) => {
                (Vowel::long(&[P::Ey], if next == b'i' && !self.ends(i + 2, "") && i > 0 { &[P::Ih] } else { &[P::Ey] }), 2)
            }
            b'a' if next == b'u' => (Vowel::long(&[P::Ao], &[P::Ao]), 2),
            b'a' if next == b'w' && !self.vowel(i + 2) => (Vowel::long(&[P::Ao], &[P::Ao]), 2),
            b'e' if next == b'a' && (self.starts(0, "rea") && i == 1 && (self.starts(3, "ct") || self.starts(3, "l") && self.vowel(4) || self.starts(3, "lm") && false) || (self.starts(0, "crea") && i == 2 || self.starts(0, "thea") && i == 2) && !self.starts(i + 2, "tu")) => {
                // react, reality, create, theater: two syllables.
                (Vowel::long(&[P::Iy], &[P::Iy]).light(), 1)
            }
            b'e' if next == b'a' => self.ea(i),
            b'e' if next == b'e' => {
                let takes = i + 2 == self.n && self.vowel_before(i);
                (Vowel::long(&[P::Iy], &[P::Iy]).hint(if takes { Hint::Takes } else { Hint::Normal }), 2)
            }
            b'e' if next == b'i' => {
                if self.before(i) == b'c' || self.starts(i, "eiz") {
                    (Vowel::long(&[P::Iy], &[P::Iy]), 2)
                } else if matches!(after, b'n' | b'l' | b'g') && !self.ends(i + 2, "n") {
                    (Vowel::long(&[P::Ey], &[P::Ey]), 2)
                } else {
                    (Vowel::long(&[P::Iy], &[P::Iy]), 2)
                }
            }
            b'e' if next == b'u' || next == b'w' && !self.vowel(i + 2) || next == b'w' && after == b'e' => {
                if i == 0 || matches!(self.before(i), b'f' | b'v' | b'p' | b'b' | b'm' | b'h' | b'k' | b'c') {
                    (Vowel::long(&[P::Y, P::Uw], &[P::Y, P::Uw]), 2)
                } else {
                    (Vowel::long(&[P::Uw], &[P::Uw]), 2)
                }
            }
            b'e' if next == b'y' => (Vowel::long(&[P::Ey], &[P::Iy]), 2),
            b'i' if next == b'e' => self.ie(i),
            b'o' if next == b'a' => (Vowel::long(&[P::Ow], &[P::Ow]), 2),
            b'o' if next == b'e' && i + 2 == self.n => (Vowel::long(&[P::Ow], &[P::Ow]), 2),
            b'o' if next == b'i' || next == b'y' => (Vowel::long(&[P::Oy], &[P::Oy]), 2),
            b'o' if next == b'o' => {
                let takes = i + 2 >= self.n - 1 && self.vowel_before(i);
                let v = if self.starts(i + 2, "k") || (self.starts(i + 2, "d") && !self.starts(i - 1, "f") && !self.starts(i - 1, "m")) || self.ends(i + 2, "t") && self.before(i) == b'f' || self.starts(i + 2, "l") && self.before(i) == b'w' {
                    Vowel::long(&[P::Uh], &[P::Uh])
                } else {
                    Vowel::long(&[P::Uw], &[P::Uw])
                };
                (v.hint(if takes { Hint::Takes } else { Hint::Normal }), 2)
            }
            b'o' if next == b'u' => self.ou(i),
            b'o' if next == b'w' => {
                let b = self.before(i);
                let end = i + 2 == self.n || (self.ends(i + 2, "s"));
                let ow = if end && i > 0 && !self.vowel_before(i) {
                    // One syllable: how, now; but low, show, know, grow.
                    if matches!(b, b'l' | b'r' | b'n' | b'h' | b'w') && i > 1 && !(b == b'h' && self.at(i - 2) == b's') {
                        P::Ow
                    } else if b == b'h' && i > 1 && self.at(i - 2) == b's' {
                        P::Ow
                    } else {
                        P::Aw
                    }
                } else if end {
                    // Unaccented final -ow: window, follow; accented: allow.
                    P::Ow
                } else if self.starts(i + 2, "n") && matches!(b, b'l' | b'r' | b'h') && !(i > 1 && matches!(self.at(i - 2), b'd' | b'b' | b'c' | b'g' | b't')) {
                    P::Ow
                } else if self.starts(i + 2, "l") && b != b'h' && !self.starts(i + 2, "ly") {
                    P::Ow
                } else {
                    P::Aw
                };
                let weak: &'static [Phoneme] = if ow == P::Ow { &[P::Ow] } else { &[P::Aw] };
                (Vowel::long(if ow == P::Ow { &[P::Ow] } else { &[P::Aw] }, weak), 2)
            }
            b'u' if next == b'e' && i + 2 == self.n => {
                if matches!(self.before(i), b'r' | b'l' | b'j' | b's' | b't' | b'd' | b'n' | b'z') && !(self.before(i) == b'l' && i > 1 && is_vowel_letter(self.at(i - 2))) {
                    (Vowel::long(&[P::Uw], &[P::Uw]), 2)
                } else {
                    (Vowel::long(&[P::Y, P::Uw], &[P::Y, P::Uw]), 2)
                }
            }
            b'u' if next == b'i' && self.before(i) == b'b' => (Vowel::new(&[P::Ih], &[P::Ih], &[P::Ih]), 2),
            b'u' if next == b'i' && (matches!(after, b't' | b'c' | b's' | b'z') && !self.starts(i + 3, "ion") || self.ends(i + 2, "")) => (Vowel::long(&[P::Uw], &[P::Uw]), 2),
            b'u' if next == b'y' => (Vowel::long(&[P::Ay], &[P::Ay]), 2),
            // ---- single vowels ----
            _ => return self.single(i),
        };
        self.push(Unit::Vowel(v), i, i + len);
        len
    }

    fn ea(&self, i: usize) -> (Vowel, usize) {
        let r = self.rest(i + 2);
        let short = r.starts_with(b"d") && !r.starts_with(b"de") && !(self.before(i) == b'l' || self.before(i) == b'b' || self.before(i) == b'n' || (self.before(i) == b'r' && i == 1))
            || r.starts_with(b"lth")
            || r.starts_with(b"th") && !r.starts_with(b"the") && r.len() <= 2 && self.before(i) != b'h'
            || r.starts_with(b"ther")
            || r.starts_with(b"vy")
            || r.starts_with(b"ven")
            || r.starts_with(b"sure")
            || r.starts_with(b"nt")
            || r.starts_with(b"pon")
            || r.starts_with(b"lous")
            || r.starts_with(b"sant")
            || r.starts_with(b"fer")
            || r.starts_with(b"st") && self.before(i) == b'r';
        if short {
            (Vowel::long(&[P::Eh], &[P::Eh]).light(), 2)
        } else if i + 2 == self.n && self.vowel_before(i) {
            (Vowel::long(&[P::Iy, P::Ax], &[P::Iy, P::Ax]), 2)
        } else if r.starts_with(b"l") && r.len() == 1 && self.vowel_before(i) {
            // -eal after another syllable: ideal, cereal.
            (Vowel::long(&[P::Iy, P::Ax], &[P::Iy, P::Ax]), 2)
        } else {
            (Vowel::long(&[P::Iy], &[P::Iy]), 2)
        }
    }

    fn ie(&self, i: usize) -> (Vowel, usize) {
        let r = self.rest(i + 2);
        if i + 2 == self.n {
            // die, lie, tie; cookie.
            return if self.vowel_before(i) {
                (Vowel::long(&[P::Iy], &[P::Iy]), 2)
            } else {
                (Vowel::long(&[P::Ay], &[P::Ay]), 2)
            };
        }
        if r == b"r" && self.vowel_before(i) {
            return (Vowel::long(&[P::Iy, P::Er], &[P::Iy, P::Er]), 3);
        }
        if r.starts_with(b"t") || (r.starts_with(b"nc") && self.before(i) == b'c') || r.starts_with(b"nt") && self.starts(i.saturating_sub(2), "cl") {
            // diet, quiet, society, science: two syllables.
            return (Vowel::long(&[P::Ay, P::Ax], &[P::Ay, P::Ax]), 2);
        }
        if r.starts_with(b"nt") || r.starts_with(b"nc") || r.starts_with(b"s") && self.vowel_before(i) && r.len() > 1 {
            return (Vowel::long(&[P::Iy, P::Ax], &[P::Iy, P::Ax]).light(), 2);
        }
        (Vowel::long(&[P::Iy], &[P::Iy]), 2)
    }

    fn ou(&self, i: usize) -> (Vowel, usize) {
        let r = self.rest(i + 2);
        let r_closes = r.first() == Some(&b'r') && r.get(1) != Some(&b'r');
        if r == b"s" || r.starts_with(b"sly") || r.starts_with(b"sness") {
            // -ous: famous.
            return (Vowel::new(&[P::Ah], &[P::Ah], &[P::Ax]).light(), 2);
        }
        if r_closes {
            return (Vowel::new(&[P::Er], &[P::Er], &[P::Er]), 2);
        }
        let v = if r.starts_with(b"ld") && self.before(i) != b'b' && self.before(i) != b'm' {
            Vowel::long(&[P::Uh], &[P::Uh])
        } else if r.starts_with(b"l") {
            Vowel::long(&[P::Ow], &[P::Ow])
        } else if r.starts_with(b"p") && !r.starts_with(b"pl") || r.starts_with(b"th") && i == 1 && self.before(i) == b'y' || r.starts_with(b"t") && self.ends(i + 2, "te") {
            Vowel::long(&[P::Uw], &[P::Uw])
        } else if r.starts_with(b"bl") || r.starts_with(b"pl") || r.starts_with(b"ntr") || r.starts_with(b"ng") || r.starts_with(b"ch") && self.before(i) == b't' || r.starts_with(b"sin") || r.starts_with(b"rage") {
            Vowel::new(&[P::Ah], &[P::Ah], &[P::Ax])
        } else if i + 2 == self.n {
            Vowel::long(&[P::Uw], &[P::Uw])
        } else {
            Vowel::long(&[P::Aw], &[P::Aw])
        };
        (v, 2)
    }

    /// A single vowel letter.
    #[allow(clippy::too_many_lines)] // one rule table
    fn single(&mut self, i: usize) -> usize {
        let c = self.at(i);
        let next = self.at(i + 1);
        let b = self.before(i);
        let n = self.n;
        // Directly before another vowel nucleus (radio, poem, diet).
        let hiatus = i + 1 < n && self.vowel(i + 1);
        let w_before = b == b'w' || (b == b'u' && i > 1 && self.at(i - 2) == b'q');
        let mut v = match c {
            b'a' => Vowel::new(&[P::Ae], &[P::Ey], &[P::Ax]),
            b'e' => Vowel::new(&[P::Eh], &[P::Iy], &[P::Ih]),
            b'i' => Vowel::new(&[P::Ih], &[P::Ay], &[P::Ih]),
            b'o' => Vowel::new(&[P::Aa], &[P::Ow], &[P::Ax]),
            b'u' => Vowel::new(&[P::Ah], &[P::Uw], &[P::Ax]),
            _ => Vowel::new(&[P::Ih], &[P::Ay], &[P::Ih]),
        };
        v.letter = c;
        let mut len = 1;
        if let Some(_m) = self.magic(i) {
            v.length = Length::Long;
            v.heavy = true;
            // Endings whose unaccented vowel reduces: -ace, -age, -ate, -ice,
            // -ile, -ine, -ite, -ive.
            let tail = self.rest(i);
            v.weak = match tail {
                [b'a', b'c', b'e'] | [b'i', b'c', b'e'] => &[P::Ih],
                [b'a', b'g', b'e'] => &[P::Ih],
                [b'a', b't', b'e'] if !self.verb && (matches!(b, b'i' | b'u') || self.starts(i.saturating_sub(2), "im") || self.starts(i.saturating_sub(3), "ur") || self.starts(i.saturating_sub(4), "orpor") || self.starts(i.saturating_sub(3), "par")) => &[P::Ax],
                [b'i', b'l', b'e'] => &[P::Ax],
                [b'i', b'n', b'e'] => &[P::Ih],
                [b'i', b'v', b'e'] => &[P::Ih],
                [b'i', b't', b'e'] if matches!(b, b'n' | b's' | b'r') => &[P::Ih],
                [b'o', b's', b'e'] if b == b'p' && i > 1 && self.at(i - 2) == b'r' => &[P::Ax],
                [b'o', b'm', b'e'] if b == b's' => &[P::Ax],
                [b'o', b'm', b'e'] if b == b'c' => &[P::Ah],
                [b'u', b'r', b'e'] => &[P::Er],
                _ => v.long,
            };
            if c == b'u' && !matches!(b, b'r' | b'l' | b'j' | b's' | b't' | b'd' | b'n' | b'z') || c == b'u' && b == b'l' && i > 1 && !is_vowel_letter(self.at(i - 2)) {
                v.long = &[P::Y, P::Uw];
                if v.weak == [P::Uw] {
                    v.weak = &[P::Y, P::Uw];
                }
            }
            if c == b'o' && self.starts(i + 1, "ve") && self.ends(i + 1, "ve") && matches!(b, b'l' | b'b' | b'h') {
                // love, above, glove.
                v.long = &[P::Ah];
            }
            if c == b'i' && self.ends(i, "ine") && self.out.iter().filter(|p| matches!(p.unit, Unit::Vowel(_))).count() > 0 && matches!(b, b'z' | b't' | b'r' | b'd' | b'c' | b's') && !self.starts(i.saturating_sub(4), "com") {
                // French -ine: machine, routine, magazine.
                v.long = &[P::Iy];
                v.weak = &[P::Iy];
            }
        } else if i + 1 == n {
            // Word-final single vowel.
            v.length = Length::Long;
            v.heavy = true;
            match c {
                b'a' => {
                    v.long = &[P::Aa];
                    v.weak = &[P::Ax];
                }
                b'i' => {
                    v.weak = &[P::Iy];
                    if self.vowel_before(i) {
                        v.long = &[P::Iy];
                    }
                }
                b'e' => v.weak = &[P::Iy],
                b'o' => v.weak = &[P::Ow],
                b'u' => v.weak = &[P::Uw],
                _ => {
                    v.weak = &[P::Iy];
                    if self.fy_verb(i) {
                        v.weak = &[P::Ay];
                        if self.before(i) == b'l' {
                            v.hint = Hint::Takes;
                        }
                    } else if self.vowel_before(i) {
                        v.long = &[P::Iy];
                    }
                }
            }
            if c == b'u' && !matches!(b, b'r' | b'l' | b'j' | b's' | b't' | b'd' | b'n' | b'z') {
                v.long = &[P::Y, P::Uw];
                v.weak = &[P::Y, P::Uw];
            }
        } else if hiatus {
            v.length = Length::Long;
            v.heavy = false;
            let first = !self.vowel_before(i);
            v.weak = match c {
                b'i' | b'y' if first && i > 0 => &[P::Ay],
                b'i' | b'e' | b'y' => &[P::Iy],
                b'o' => &[P::Ow],
                b'u' => &[P::Uw],
                _ => &[P::Ax],
            };
            if c == b'u' && !matches!(b, b'r' | b'l' | b'j' | b's' | b't' | b'd' | b'n' | b'z') {
                v.long = &[P::Y, P::Uw];
                v.weak = &[P::Y, P::Uw];
            }
        } else {
            let cons = self.consonants_from(i + 1);
            // Special long spellings before two consonants.
            let tail = self.rest(i + 1);
            if c == b'i' && (tail.starts_with(b"nd") && !tail.starts_with(b"nde") || tail.starts_with(b"ld") || tail.starts_with(b"gn") && (i + 3 == n || !self.vowel(i + 3))) {
                v.length = Length::Long;
                v.heavy = true;
                if tail.starts_with(b"gn") {
                    self.push(Unit::Vowel(v), i, i + 1);
                    self.push(Unit::Cons(&[P::N]), i + 1, i + 3);
                    return 3;
                }
            } else if c == b'o' && (tail.starts_with(b"ld") || tail.starts_with(b"lt") || tail.starts_with(b"ll") && (i + 3 == n || !self.vowel(i + 3)) && !matches!(b, b'd' | b'm')) {
                v.length = Length::Long;
                v.heavy = true;
            } else if c == b'o' && (tail == b"l" || tail == b"ls") && i >= 2 && self.starts(i - 2, "tr") {
                // control, patrol.
                v.length = Length::Long;
                v.heavy = true;
            } else if c == b'o' && tail.starts_with(b"st") && matches!(b, b'm' | b'p' | b'h') && (i + 3 == n || !self.vowel(i + 3)) {
                v.length = Length::Long;
                v.heavy = true;
            } else if c == b'o' && tail.starts_with(b"lk") {
                // folk, yolk: the l is silent.
                v.length = Length::Long;
                v.heavy = true;
                self.push(Unit::Vowel(v), i, i + 2);
                return 2;
            } else if c == b'o' && tail.starts_with(b"ther") {
                v.short = &[P::Ah];
                v.long = &[P::Ah];
            } else if c == b'a' && !w_before && (tail.starts_with(b"lk")) {
                v.short = &[P::Ao];
                v.long = &[P::Ao];
                self.push(Unit::Vowel(v), i, i + 2);
                return 2;
            } else if c == b'a' && tail.starts_with(b"lm") && (i + 3 == n || !self.vowel(i + 3)) {
                v.short = &[P::Aa];
                v.long = &[P::Aa];
                v.heavy = true;
                self.push(Unit::Vowel(v), i, i + 2);
                return 2;
            } else if c == b'a' && (tail.starts_with(b"ll") && (i + 3 == n || !self.vowel(i + 3) && self.at(i + 3) != b'y') || tail.starts_with(b"lt") || tail.starts_with(b"ls") || tail.starts_with(b"ld") || tail.starts_with(b"lso") || tail.starts_with(b"lw") || tail.starts_with(b"lth") || i == 0 && (tail.starts_with(b"lmo") || tail.starts_with(b"lr"))) {
                v.short = &[P::Ao];
                v.heavy = true;
            } else if c == b'a' && w_before && !matches!(next, b'k' | b'g' | b'x') && !tail.starts_with(b"ck") && !tail.starts_with(b"ng") {
                v.short = &[P::Aa];
            } else if c == b'a' && (tail.starts_with(b"nge") || tail.starts_with(b"ngi") || tail.starts_with(b"nger")) && i > 0 {
                v.length = Length::Long;
                v.heavy = true;
            } else if c == b'u' && matches!(b, b'p' | b'b' | b'f') && (tail.starts_with(b"ll") || tail.starts_with(b"sh") || tail == b"t" || tail.starts_with(b"tch")) {
                v.short = &[P::Uh];
            }
            if v.length != Length::Long {
                let doubled = cons >= 2 && self.at(i + 1) == self.at(i + 2);
                let digraph = cons == 2 && matches!(&tail[..2], b"th" | b"sh" | b"ch" | b"ph" | b"gh" | b"wh");
                let liquid = cons == 2 && matches!(tail[0], b'p' | b'b' | b't' | b'd' | b'k' | b'c' | b'g' | b'f') && (tail[1] == b'r' || tail[1] == b'l' && &tail[2..] == b"e");
                let single = cons == 1 && next != b'x' || digraph || liquid;
                if single && i + 1 + cons < n {
                    v.length = Length::Open;
                    if c == b'u' && !matches!(b, b'r' | b'l' | b'j' | b's' | b't' | b'd' | b'n' | b'z') || c == b'u' && b == b'l' && i > 1 && !is_vowel_letter(self.at(i - 2)) && self.at(i - 2) != 0 {
                        v.long = &[P::Y, P::Uw];
                    }
                    if c == b'u' && !matches!(b, b'r' | b'j' | b's' | b't' | b'd' | b'z' | b'n') && i > 0 && matches!(next, b'l') {
                        // -ular, -ulate: popular, regular.
                        v.weak = &[P::Y, P::Uh];
                    } else if c == b'u' && matches!(b, b'p' | b'b' | b'c' | b'k' | b'g' | b'm' | b'f' | b'v' | b'h' | b'n' | b'l') && i > 0 && next != b'r' {
                        // document, executive, manufacture.
                        v.weak = &[P::Y, P::Uw];
                    } else if c == b'u' && matches!(b, b'd' | b't') && self.vowel(i + 2) && i > 1 && next != b'r' {
                        v.weak = &[P::Uw];
                    }
                } else {
                    v.length = Length::Short;
                    v.heavy = cons >= 2 || doubled;
                }
                if c == b'e' && i == 0 && next == b'x' {
                    v.weak = &[P::Eh];
                }
                if c == b'i' && i == 0 && v.length == Length::Open {
                    // idea, identify.
                    v.weak = &[P::Ay];
                }
                if c == b'y' && v.length == Length::Open && !self.vowel_before(i) {
                    // hypothesis, dynamic.
                    v.weak = &[P::Ay];
                }
                if c == b'u' && i == 0 && v.length == Length::Open {
                    // unique, united.
                    v.weak = &[P::Y, P::Uw];
                }
                if c == b'e' && i == 0 && matches!(next, b'n' | b'm') && v.length == Length::Short {
                    // enjoy, employ.
                    v.weak = &[P::Eh];
                }
            }
            if c == b'e' && next == b'r' && !self.at(i + 2).eq(&b'r') {
                // er + vowel: hero, serious, period long; very, merit short.
                let r = self.rest(i + 2);
                // serious, material: ih-r; hero, theory, era: iy-uh-r.
                let near = r.first() == Some(&b'i') && r.get(1).is_some_and(|&x| is_vowel_letter(x));
                let long = near || r == b"o" || r == b"os" || r == b"a" || r == b"y" && self.vowel_before(i) && false;
                v.long = if near {
                    &[P::Ih]
                } else if long {
                    &[P::Iy, P::Ax]
                } else {
                    &[P::Eh]
                };
                v.weak = &[P::Ax];
                if long {
                    v.length = Length::Long;
                }
            } else if c == b'a' && next == b'r' && self.at(i + 2) != b'r' {
                v.long = &[P::Eh];
            } else if c == b'o' && next == b'r' && b == b'w' {
                // worry.
                v.short = &[P::Er];
                v.long = &[P::Er];
                v.weak = &[P::Er];
            } else if c == b'o' && next == b'r' {
                v.short = &[P::Ao];
                v.long = &[P::Ao];
            } else if c == b'u' && next == b'r' && self.at(i + 2) != b'r' {
                v.length = Length::Long;
                if matches!(v.long, [P::Uw]) {
                    v.long = &[P::Uh];
                } else {
                    v.long = &[P::Y, P::Uh];
                }
            } else if c == b'u' && next == b'r' {
                v.short = &[P::Er];
                v.long = &[P::Er];
            }
        }
        // Final -le after a consonant is its own syllable: table, little.
        if i + 2 == n && c == b'e' && next == 0 {
            len = 1;
        }
        self.push(Unit::Vowel(v), i, i + len);
        len
    }

    /// Final `y` of a verb in `-fy` or `-ply` (modify, apply) is long.
    fn fy_verb(&self, i: usize) -> bool {
        let b = self.before(i);
        (b == b'f' && i > 1 && is_vowel_letter(self.at(i - 2)))
            || (b == b'l' && i > 2 && self.at(i - 2) == b'p' && (self.at(i - 3) == b'p' || self.starts(0, "re") || self.starts(0, "com") || self.starts(0, "im") || self.starts(0, "multi")))
    }

    // ---- consonants ------------------------------------------------------

    /// Push the consonant spelling at `i`; returns the letters used.
    #[allow(clippy::too_many_lines)] // one rule table
    fn consonant_spelling(&mut self, i: usize) -> usize {
        let c = self.at(i);
        let next = self.at(i + 1);
        let after = self.at(i + 2);
        let n = self.n;
        let front = |x: u8| matches!(x, b'e' | b'i' | b'y');
        let vowel_next = self.vowel(i + 1);
        // ti, ci, si, xi, gi before an unaccented ending sound like sh/zh/j.
        let absorbed = |k: usize| -> bool {
            let r = self.rest(k);
            matches!(r.first(), Some(b'i') | Some(b'e'))
                && (r[1..].starts_with(b"on")
                    || r[1..].starts_with(b"al")
                    || r[1..].starts_with(b"ous")
                    || r[1..].starts_with(b"ent")
                    || r[1..].starts_with(b"enc")
                    || r[1..].starts_with(b"an")
                    || r[1..] == *b"a"
                    || r[1..] == *b"as")
                && !(r[0] == b'e' && (r[1..].starts_with(b"ent") || r[1..].starts_with(b"enc") || r[1..].starts_with(b"al")))
        };
        let (ph, len): (&'static [Phoneme], usize) = match c {
            b'b' if next == b'b' => (&[P::B], 2),
            b'b' if next == b't' && (i + 2 == n || self.ends(i + 2, "s") || self.starts(i + 2, "le")) => (&[], 1),
            b'b' => (&[P::B], 1),
            b'c' if next == b'h' => {
                let r = self.rest(i + 2);
                let k = matches!(after, b'r' | b'l')
                    || (i > 0 && self.before(i) == b's')
                    || self.starts(i, "chem")
                    || self.starts(i, "chor")
                    || self.starts(i, "chao")
                    || self.starts(i, "chara")
                    || self.starts(i, "chiro")
                    || self.starts(i, "chasm")
                    || (i > 0 && self.starts(i - 1, "ech") && matches!(after, b'o' | b'a' | b'n'))
                    || (i > 1 && self.starts(i - 2, "arch") && matches!(after, b'i' | b'a' | b'e') && after != 0 && self.at(i + 3) != b'r')
                    || (i > 0 && self.starts(i - 1, "ache"))
                    || self.ends(i, "ch") && i > 2 && self.starts(i - 3, "mach")
                    || (i > 0 && self.before(i) == b'y')
                    || (i > 1 && self.starts(i - 2, "tech"));
                if k {
                    (&[P::K], 2)
                } else if r == b"e" && i > 0 || self.starts(0, "chef") || self.starts(i, "chute") || self.starts(i, "chand") || self.starts(i, "champa") || self.starts(i, "chauf") || self.starts(i, "chic") && !self.starts(i, "chick") || (i > 0 && self.starts(i, "chine")) {
                    (&[P::Sh], 2)
                } else {
                    (&[P::Ch], 2)
                }
            }
            b'c' if next == b'k' => (&[P::K], 2),
            b'c' if next == b'c' && front(after) => (&[P::K, P::S], 2),
            b'c' if next == b'c' => (&[P::K], 2),
            b'c' if i > 0 && absorbed(i + 1) => {
                // social, ocean, musician: the i or e joins the consonant.
                (&[P::Sh], 2)
            }
            b'c' if front(next) => (&[P::S], 1),
            b'c' => (&[P::K], 1),
            b'd' if next == b'd' => (&[P::D], 2),
            b'd' if next == b'g' => (&[P::Jh], 2),
            b'd' if next == b'j' => (&[P::Jh], 2),
            b'd' if i > 0 && self.vowel_before(i) && next == b'u' && (self.vowel(i + 2) || self.starts(i + 2, "le") || self.starts(i + 2, "la") || self.starts(i + 2, "re") && self.ends(i + 2, "re") || self.starts(i + 2, "cat") || self.starts(i + 2, "cation")) => (&[P::Jh], 1),
            b'd' => (&[P::D], 1),
            b'f' if next == b'f' => (&[P::F], 2),
            b'f' => (&[P::F], 1),
            b'g' if next == b'g' && front(after) && i > 0 && matches!(self.before(i), b'u' | b'a') && !self.starts(i + 2, "er") && !self.starts(i + 2, "y") && !self.starts(i + 2, "ing") && !self.starts(i + 2, "ed") => (&[P::Jh], 2),
            b'g' if next == b'g' => (&[P::G], 2),
            b'g' if next == b'h' => (if i == 0 { &[P::G] } else { &[] }, 2),
            b'g' if next == b'n' && (i == 0 || i + 2 == n || self.ends(i + 2, "s")) => (&[P::N], 2),
            b'g' if self.ends(i, "gue") && i > 0 => (&[P::G], 3),
            b'g' if next == b'u' && self.vowel(i + 2) => {
                if self.before(i) == b'n' {
                    (&[P::G, P::W], 2)
                } else if front(after) || after == b'a' && self.at(i + 3) == b'r' {
                    (&[P::G], 2)
                } else {
                    (&[P::G, P::W], 2)
                }
            }
            b'g' if i > 0 && absorbed(i + 1) => (&[P::Jh], 2),
            b'g' if front(next) && !self.hard_g(i) => (&[P::Jh], 1),
            b'g' => (&[P::G], 1),
            b'h' => {
                let silent = (i == 0 && (self.starts(0, "hour") || self.starts(0, "honest") || self.starts(0, "honor") || self.starts(0, "honour") || self.starts(0, "heir")))
                    || (i == 2 && self.starts(0, "exh"))
                    || !(self.vowel(i + 1) || next == b'y' || next == b'u');
                (if silent { &[] } else { &[P::Hh] }, 1)
            }
            b'j' => (&[P::Jh], 1),
            b'k' if next == b'n' && (i == 0 || i == 2 && self.starts(0, "un")) => (&[P::N], 2),
            b'k' if next == b'k' => (&[P::K], 2),
            b'k' => (&[P::K], 1),
            b'l' if next == b'l' => (&[P::L], 2),
            b'l' if next == b'e' && i + 2 == n && i > 0 && !is_vowel_letter(self.before(i)) && self.vowel_before(i) => {
                self.push(Unit::Vowel(Vowel::fixed(&[P::Ax]).hint(Hint::Never).light()), i, i + 1);
                (&[P::L], 2)
            }
            b'l' => (&[P::L], 1),
            b'm' if next == b'm' => (&[P::M], 2),
            b'm' if (next == b'b' || next == b'n') && i + 2 == n => (&[P::M], 2),
            b'm' => (&[P::M], 1),
            b'n' if next == b'n' => (&[P::N], 2),
            b'n' if next == b'g' => {
                if (front(after) && !self.hard_g(i + 1)) || absorbed(i + 2) && after == b'i' {
                    (&[P::N], 1)
                } else if self.vowel(i + 2) || after == b'l' || after == b'r' || self.starts(i + 2, "u") {
                    (&[P::Ng, P::G], 2)
                } else {
                    (&[P::Ng], 2)
                }
            }
            b'n' if matches!(next, b'k' | b'q' | b'x') || (next == b'c' && !front(after) && after != b'h') => (&[P::Ng], 1),
            b'n' => (&[P::N], 1),
            b'p' if next == b'h' => (&[P::F], 2),
            b'p' if next == b'p' => (&[P::P], 2),
            b'p' if i == 0 && matches!(next, b's' | b'n' | b't') => (&[], 1),
            b'p' => (&[P::P], 1),
            b'q' if next == b'u' && self.ends(i, "que") && i > 0 => (&[P::K], 3),
            b'q' if next == b'u' => (&[P::K, P::W], 2),
            b'q' => (&[P::K], 1),
            b'r' if next == b'r' || next == b'h' => (&[P::R], 2),
            b'r' => (&[P::R], 1),
            b's' if next == b'h' => (&[P::Sh], 2),
            b's' if self.starts(i, "sch") => (&[P::S, P::K], 3),
            b's' if next == b'c' && front(after) => {
                if i > 0 && absorbed(i + 2) && after == b'i' && !self.starts(i, "scien") {
                    (&[P::Sh], 3)
                } else {
                    (&[P::S], 2)
                }
            }
            b's' if next == b's' => {
                if absorbed(i + 2) {
                    (&[P::Sh], 3)
                } else if self.starts(i + 2, "u") && (self.starts(i + 2, "ue") || self.starts(i + 2, "ure") || self.starts(i + 2, "ua")) {
                    (&[P::Sh], 2)
                } else {
                    (&[P::S], 2)
                }
            }
            b's' if i > 0 && absorbed(i + 1) && next == b'i' => {
                let voiced = self.vowel(i - 1) || self.before(i) == b'r';
                (if voiced { &[P::Zh] } else { &[P::Sh] }, 2)
            }
            b's' if i > 0 && next == b'u' && (self.ends(i + 1, "ure") || self.starts(i + 1, "ual")) && self.vowel_before(i) => {
                let voiced = self.vowel(i - 1);
                (if voiced { &[P::Zh] } else { &[P::Sh] }, 1)
            }
            b's' if next == b'm' && i + 2 == n && i > 0 => {
                // -ism, -asm: a syllabic m.
                self.push(Unit::Cons(&[P::Z]), i, i + 1);
                self.push(Unit::Vowel(Vowel::fixed(&[P::Ax]).hint(Hint::Never).light()), i + 1, i + 1);
                return 1;
            }
            b's' => (if self.voiced_s(i) { &[P::Z] } else { &[P::S] }, 1),
            b't' if self.starts(i, "tch") => (&[P::Ch], 3),
            b't' if next == b'h' => (if self.voiced_th(i) { &[P::Dh] } else { &[P::Th] }, 2),
            b't' if i > 0 && absorbed(i + 1) && next == b'i' => {
                (if self.before(i) == b's' { &[P::Ch] } else { &[P::Sh] }, 2)
            }
            b't' if i > 0 && next == b'u' && self.vowel_before(i) && (self.vowel(i + 2) || self.ends(i + 2, "re") || self.starts(i + 2, "r") && self.vowel(i + 3)) => (&[P::Ch], 1),
            b't' if next == b't' => (&[P::T], 2),
            b't' if self.before(i) == b's' && (self.ends(i + 1, "le") || self.ends(i + 1, "en")) => (&[], 1),
            b't' if next == b'z' => (&[P::T, P::S], 2),
            b't' => (&[P::T], 1),
            b'v' if next == b'v' => (&[P::V], 2),
            b'v' => (&[P::V], 1),
            b'w' if next == b'r' && i == 0 => (&[P::R], 2),
            b'w' if next == b'h' => (if after == b'o' && self.at(i + 3) != b'a' { &[P::Hh] } else { &[P::W] }, 2),
            b'w' => (&[P::W], 1),
            b'x' if i == 0 => (&[P::Z], 1),
            b'x' if i > 0 && absorbed(i + 1) => (&[P::K, P::Sh], 2),
            b'x' if i > 0 && self.starts(i + 1, "ua") => (&[P::K, P::Sh], 1),
            b'x' if i == 1 && self.at(0) == b'e' && (vowel_next || next == b'h') => (&[P::G, P::Z], 1),
            b'x' => (&[P::K, P::S], 1),
            b'y' => (&[P::Y], 1),
            b'z' if next == b'z' => (&[P::Z], 2),
            b'z' if next == b'u' && self.ends(i + 1, "ure") => (&[P::Zh], 1),
            b'z' => (&[P::Z], 1),
            _ => (&[], 1),
        };
        self.push(Unit::Cons(ph), i, i + len);
        len
    }

    /// `g` before e, i or y that stays hard: get, give, girl, tiger, anger.
    fn hard_g(&self, i: usize) -> bool {
        let r = self.rest(i);
        (i == 0 && (r.starts_with(b"get") || r.starts_with(b"gea") || r.starts_with(b"gee") || r.starts_with(b"gif") || r.starts_with(b"giv") || r.starts_with(b"gir") || r.starts_with(b"gig") || r.starts_with(b"gild") || r.starts_with(b"gim")))
            || r == b"get" || r == b"gets"
            || (r.starts_with(b"ger") && i > 0 && (matches!(self.before(i), b'i' | b'u' | b'a') && !self.starts(i.saturating_sub(2), "an") || self.before(i) == b'n' && i > 1 && (self.at(i - 2) != b'a' || i == 2)))
            || (self.starts(i.saturating_sub(2), "begi"))
            || r.starts_with(b"giv")
    }

    /// Whether the `s` at `i` sounds like z.
    fn voiced_s(&self, i: usize) -> bool {
        let b = self.before(i);
        let next = self.at(i + 1);
        if i == 0 || !self.vowel(i - 1) && b != b'n' && b != b'l' && b != b'r' {
            return false;
        }
        let rest = self.rest(i);
        if !self.vowel(i - 1) {
            // After a consonant: mostly s (person, else); -nse is s.
            return false;
        }
        if !self.vowel(i + 1) && !(next == b'm' && i + 2 == self.n) {
            return false;
        }
        // dis- before a vowel: disagree, disappear.
        if i == 2 && self.starts(0, "dis") {
            return false;
        }
        // research, resource; use, abuse, close.
        if i == 2 && self.starts(0, "res") && (self.starts(3, "ear") || self.starts(3, "our")) {
            return false;
        }
        if self.w == b"close" || self.w.ends_with(b"rpose") {
            return false;
        }
        // Final -se after a vowel.
        if rest == b"se" {
            let r2 = self.rest(i.saturating_sub(2));
            return !(r2 == b"ouse" || b == b'a' && i > 0 || r2 == b"ease" && matches!(self.at(i.wrapping_sub(3)), b'r' | b'c' | b'l') && !(i >= 4 && self.at(i - 4) == b'p') || self.starts(i.saturating_sub(4), "purpose") || r2 == b"ose" && self.at(i.saturating_sub(3)) == b'd');
        }
        // -sis, -sy after a vowel, iso-.
        if rest.starts_with(b"sis") || rest.starts_with(b"sic") && b == b'a' || (i == 1 && b == b'i') || rest.starts_with(b"so") && self.starts(i.saturating_sub(4), "philo") {
            return false;
        }
        true
    }

    /// Whether the `th` at `i` is voiced.
    fn voiced_th(&self, i: usize) -> bool {
        let r = self.rest(i + 2);
        let b = self.before(i);
        (i == 0 && (self.starts(0, "there") || self.starts(0, "thus") || self.starts(0, "thy") || self.starts(0, "thou") || self.starts(0, "thee") || self.starts(0, "thine") || self.starts(0, "thence")))
            || (r.starts_with(b"er") && i > 0 && (is_vowel_letter(b) || b == b'r') && !(b == b'e' && i == 1))
            || (r == b"e" && i > 0)
            || (r == b"m" && i > 0)
            || (r == b"y" && b == b'r')
            || (r.starts_with(b"in") && i == 2 && self.starts(0, "with"))
    }
}

// ===== 2. stress ===========================================================

fn ends_with(w: &[u8], end: usize, s: &str) -> bool {
    end >= s.len() && &w[end - s.len()..end] == s.as_bytes()
}

/// Where the part of the word that takes the accent ends: before neutral
/// endings like `-ness` and `-ment`.
fn stem_end(w: &[u8]) -> usize {
    let mut end = w.len();
    loop {
        let Some(s) = NEUTRAL.iter().find(|s| ends_with(w, end, s) && end - s.len() >= 3) else {
            break;
        };
        let stem = end - s.len();
        if *s == "ly" && (stem < 4 || matches!(w[stem - 1], b'a' | b'i' | b'o' | b'u')) {
            break;
        }
        if !w[..stem].iter().any(|&c| is_vowel_letter(c)) {
            break;
        }
        end = stem;
    }
    end
}

/// Mark the accented nucleus (and secondary accents) of a word.
fn accent(w: &[u8], pieces: &mut [Piece]) {
    let nuclei: Vec<usize> = pieces
        .iter()
        .enumerate()
        .filter(|(_, p)| matches!(p.unit, Unit::Vowel(_)))
        .map(|(k, _)| k)
        .collect();
    let m = nuclei.len();
    if m == 0 {
        return;
    }
    let vowel = |pieces: &[Piece], k: usize| -> Vowel {
        match pieces[nuclei[k]].unit {
            Unit::Vowel(v) => v,
            Unit::Cons(_) => unreachable!(),
        }
    };
    let can = |pieces: &[Piece], k: usize| vowel(pieces, k).hint != Hint::Never;
    // The nucleus that holds letter `pos` (the first one starting at or after it).
    let nucleus_at = |pieces: &[Piece], pos: usize| -> usize {
        (0..m).find(|&k| pieces[nuclei[k]].end > pos).unwrap_or(m)
    };

    let end = stem_end(w);
    // Number of stressable nuclei in the stem.
    let stem_m = nucleus_at(pieces, end).min(m);
    let stem_m = (0..stem_m).filter(|&k| can(pieces, k)).count().max(1).min(m);
    let stressable: Vec<usize> = (0..m).filter(|&k| can(pieces, k)).collect();
    let pick = |idx: isize| -> Option<usize> {
        if idx >= 0 {
            stressable.get(idx as usize).copied()
        } else {
            None
        }
    };

    let mut primary: Option<usize> = None;
    let mut ary: Option<usize> = None;
    // 1. A nucleus that takes stress.
    if let Some(k) = (0..m).rev().find(|&k| vowel(pieces, k).hint == Hint::Takes) {
        primary = Some(k);
    }
    // 2. Endings.
    if primary.is_none() {
        for (table, back) in [(SELF, 0isize), (BEFORE_ONE, 1), (BEFORE_TWO, 2)] {
            let Some(s) = table.iter().filter(|s| ends_with(w, end, s)).max_by_key(|s| s.len()) else {
                continue;
            };
            let start = end - s.len();
            if start == 0 {
                continue;
            }
            // Index (in stressable nuclei) of the ending's first nucleus.
            let first = nucleus_at(pieces, start);
            let Some(si) = stressable.iter().position(|&k| k >= first) else {
                continue;
            };
            let si = si as isize;
            let target = if back == 2 {
                if si >= 2 {
                    si - 2
                } else if si == 1 && !starts_with_prefix(w, pieces, &nuclei) {
                    0
                } else {
                    si
                }
            } else {
                si - back
            };
            if let Some(k) = pick(target) {
                primary = Some(k);
                // -ary, -ory, -ery keep a secondary accent: military, territory.
                if back == 2 && si >= 2 && (s.starts_with("ar") || s.starts_with("or") || s.starts_with("er")) {
                    ary = stressable.get(si as usize).copied();
                }
                break;
            }
            if back == 1 {
                // -ic in a one-syllable stem: the stem.
                primary = pick(0);
                break;
            }
        }
    }
    // 3. Prefixes, syllable weight.
    if primary.is_none() {
        let heavy = |k: usize| -> bool { weight(w, pieces, &nuclei, k) };
        let sm = stem_m.min(stressable.len());
        primary = Some(if sm <= 1 {
            stressable[0]
        } else if sm == 2 {
            let (a, b) = (stressable[0], stressable[1]);
            let tail = &w[pieces[nuclei[b]].start..end.max(pieces[nuclei[b]].start)];
            let ends = |list: &[&str]| list.iter().any(|s| tail == s.as_bytes());
            let vowel_initial = matches!(w[0], b'a' | b'e' | b'o') && pieces[nuclei[a]].end == 1;
            let doubled = vowel_initial && w.len() > 3 && w[1] == w[2] && !is_vowel_letter(w[1]);
            let single = vowel_initial && w.len() > 2 && !is_vowel_letter(w[1]) && (is_vowel_letter(w[2]) || matches!(w[2], b'r' | b'l') && !matches!(w[1], b'r' | b'l' | b'n' | b'm' | b's'));
            let latin = ["con", "com", "pro", "ob", "sub", "ab"].iter().any(|p| w.starts_with(p.as_bytes()));
            let soft_ent = tail == b"ent" && matches!(w[pieces[nuclei[b]].start - 1], b'c' | b'g');
            let onset = pieces[nuclei[a]].end..pieces[nuclei[b]].start;
            let single_onset = w[onset].iter().filter(|&&c| !is_vowel_letter(c)).count() <= 1;
            if prefix_strong(w) && (!ends(STRONG_KEEP) || tail == b"ent" && !soft_ent || !single_onset) {
                b
            } else if starts_with_prefix(w, pieces, &nuclei) && !ends(NOUNISH) && !(latin && ends(NOUN_CLUSTER)) {
                b
            } else if doubled && !ends(DOUBLED_KEEP) {
                b
            } else if single && (heavy(b) || w[0] == b'a') && !ends(NOUNISH) {
                b
            } else {
                // The root after a prefix keeps a secondary accent: content, contact.
                if starts_with_prefix(w, pieces, &nuclei) && !ends(NOUNISH) {
                    ary = Some(b);
                }
                a
            }
        } else {
            let pen = stressable[sm - 2];
            let ante = stressable[sm - 3];
            let noun_end = noun_ending(w, end) || prefix_strong(w) && ["it", "ice", "ipe"].iter().any(|s| ends_with(w, end, s));
            if heavy(pen) {
                pen
            } else if ante == stressable[0] && starts_with_prefix(w, pieces, &nuclei) && !noun_end {
                pen
            } else {
                ante
            }
        });
    }
    let p = primary.unwrap_or(stressable[0]);
    // Secondary accents: every second syllable before the primary.
    let pi = stressable.iter().position(|&k| k == p).unwrap_or(0);
    let mut sec: Vec<usize> = ary.into_iter().collect();
    if pi >= 2 {
        let first = usize::from(pi >= 3 && starts_with_prefix(w, pieces, &nuclei));
        let mut k = first;
        while k + 1 < pi {
            sec.push(stressable[k]);
            k += 2;
        }
    }
    for (k, &ix) in nuclei.iter().enumerate() {
        if let Unit::Vowel(v) = &mut pieces[ix].unit {
            v.stress = if k == p {
                Stress::Primary
            } else if sec.contains(&k) {
                Stress::Secondary
            } else {
                Stress::None
            };
        }
    }
}

/// The word starts with an unaccented prefix.
fn starts_with_prefix(w: &[u8], pieces: &[Piece], nuclei: &[usize]) -> bool {
    let first_end = pieces[nuclei[0]].end;
    if matches!(pieces[nuclei[0]].unit, Unit::Vowel(v) if v.hint == Hint::Never) {
        // un- and non- are handled already.
        return false;
    }
    PREFIXES.iter().any(|p| {
        let p = p.as_bytes();
        w.starts_with(p)
            && p.len() + 2 <= w.len()
            && first_end <= p.len()
            && (!is_vowel_letter(w[p.len()]) || p == b"ex")
            && w[p.len()] != w[p.len() - 1]
            && w[p.len()] != w[p.len() + 1]
            && w[p.len()..].iter().any(|&c| is_vowel_letter(c))
    }) || (w.len() > 5 && w.starts_with(b"co") && w[2] == w[3] && matches!(w[2], b'm' | b'n' | b'l' | b'r') && first_end <= 2)
}

/// Prefixes that are nearly always unaccented.
fn prefix_strong(w: &[u8]) -> bool {
    ["pre", "be", "de", "re"].iter().any(|p| {
        let k = p.len();
        w.starts_with(p.as_bytes()) && w.len() > k + 1 && !is_vowel_letter(w[k]) && w[k] != w[k + 1] && w[k] != b'y'
    })
}

/// Final syllables of two-syllable words that keep the accent on the first
/// syllable even after a prefix: concert, absent, envy.
const NOUNISH: &[&str] = &[
    "er", "or", "ar", "y", "ow", "en", "on", "el", "et", "ic", "ish", "ent", "ant", "ence", "ance",
    "ism", "age", "ive", "ous", "us", "um", "a", "al", "ey", "ly", "ine", "ite", "ile", "o", "id", "il",
    "op", "om", "em", "ate", "and", "ond", "ort",
];

/// Noun endings after con-, com-, pro-, ob-, sub-, ab-: contact, project,
/// object, process, content.
const NOUN_CLUSTER: &[&str] = &[
    "act", "ect", "uct", "ept", "ex", "ext", "ict", "ast", "est", "ess", "ent", "ost", "ound", "ort",
];

/// The same after be-, de-, re-, pre- (recent, demon, relic).
const STRONG_KEEP: &[&str] = &[
    "ent", "al", "ant", "ic", "ish", "ism", "age", "ous", "um", "a", "o", "ey", "ly", "ine", "ile", "id",
    "il", "op", "om", "em", "et", "it", "el", "en", "on", "us", "ence",
];

/// The same after a vowel and a doubled consonant (attic, office, apple).
const DOUBLED_KEEP: &[&str] = &[
    "er", "or", "ar", "y", "ic", "et", "it", "ish", "ey", "o", "a", "age", "ice", "um", "us", "ent", "ant",
    "ort", "en", "on", "ess",
];

/// Whether a nucleus counts as heavy for the Latin rule.
fn weight(w: &[u8], pieces: &[Piece], nuclei: &[usize], k: usize) -> bool {
    let Unit::Vowel(v) = pieces[nuclei[k]].unit else {
        return false;
    };
    if v.heavy && v.length == Length::Long {
        return true;
    }
    let from = pieces[nuclei[k]].end;
    let to = nuclei.get(k + 1).map_or(w.len(), |&j| pieces[j].start);
    let cons: usize = pieces[nuclei[k] + 1..]
        .iter()
        .take_while(|p| p.start < to)
        .map(|p| match p.unit {
            Unit::Cons(ph) => ph.len().max(usize::from(p.end - p.start >= 2 && w[p.start] == w[p.start + 1])),
            Unit::Vowel(_) => 0,
        })
        .sum();
    let _ = from;
    if cons >= 2 {
        // A stop or f with r or l starts the next syllable.
        let letters = &w[from..to];
        !(letters.len() == 2 && matches!(letters[0], b'p' | b'b' | b't' | b'd' | b'k' | b'c' | b'g' | b'f') && matches!(letters[1], b'r' | b'l'))
    } else {
        v.heavy
    }
}

/// Endings of nouns and adjectives, which take the accent early.
fn noun_ending(w: &[u8], end: usize) -> bool {
    ["y", "ence", "ance", "ent", "ant", "ency", "ancy", "ess", "est", "ous", "um", "us", "a", "ics", "o", "sis"]
        .iter()
        .any(|s| ends_with(w, end, s))
}

// ===== 3. vowels ===========================================================

/// Whether an accented single vowel in an open syllable is long.
fn open_long(w: &[u8], pieces: &[Piece], k: usize, v: &Vowel) -> bool {
    // Nuclei after this one, ignoring syllabic endings.
    let end = stem_end(w);
    let later: Vec<&Piece> = pieces[k + 1..]
        .iter()
        .filter(|p| p.start < end && matches!(p.unit, Unit::Vowel(x) if x.hint != Hint::Never))
        .collect();
    let syllabic_le = pieces[k + 1..].iter().any(|p| matches!(p.unit, Unit::Vowel(x) if x.hint == Hint::Never)) && w.ends_with(b"le");
    // Consonant spelling right after.
    let cons = pieces.get(k + 1);
    let palatal = cons.is_some_and(|p| matches!(p.unit, Unit::Cons(ph) if matches!(ph, [P::Sh] | [P::Zh]) && p.end - p.start >= 2 && matches!(w[p.end - 1], b'i' | b'e')));
    if palatal {
        // nation, motion; but vision, special.
        return v.letter != b'i' && !(v.letter == b'e' && w[pieces[k + 1].start] == b'c');
    }
    if later.is_empty() {
        return syllabic_le || true;
    }
    let next = later[0];
    let tail = &w[pieces[k].end..];
    let after_cons = &w[next.start..];
    // A vowel before i or e plus a vowel (radio, media): long.
    if matches!(after_cons.first(), Some(b'i' | b'e')) && after_cons.get(1).is_some_and(|&c| is_vowel_letter(c)) && after_cons != b"ie" && after_cons != b"ies" {
        return v.letter != b'i';
    }
    if later.len() >= 2 {
        // Trisyllabic laxing: family, animal; but u stays long (humanity).
        let ary = ["ary", "ory", "ery"].iter().any(|s| after_cons == s.as_bytes() || after_cons.ends_with(s.as_bytes()) && after_cons.len() == 3);
        return v.letter == b'u' || ary;
    }
    let _ = tail;
    if syllabic_le {
        return true;
    }
    let r = &after_cons[..after_cons.len().min(end.saturating_sub(next.start))];
    let short_end = ["ic", "ics", "id", "it", "ish", "ice", "il", "el", "et", "ow", "age", "ine", "ey", "in", "ive", "ite", "ance", "ue", "ess", "ect", "uct", "ise"];
    let is = |list: &[&str]| list.iter().any(|s| r == s.as_bytes());
    match v.letter {
        b'u' => !(r == b"ish"),
        b'a' => {
            if is(&short_end) {
                r == b"ive"
            } else {
                !(r == b"en" && false) && !(r == b"on" && false)
            }
        }
        b'e' => is(&["al", "us", "a", "o", "ent", "um", "is"]),
        b'i' => {
            if r == b"er" {
                !(pieces[k + 1].start < w.len() && w[pieces[k + 1].start] == b'v')
            } else {
                is(&["or", "al", "us", "a", "o", "ent", "ant", "ot", "ate", "ar", "an", "em", "is"])
            }
        }
        b'o' => {
            if is(&short_end) {
                r == b"ice" || r == b"ive"
            } else if r == b"y" {
                false
            } else if r == b"er" {
                w[pieces[k + 1].start] != b'v' && w[pieces[k + 1].start] != b'p'
            } else {
                true
            }
        }
        _ => !is(&["ic", "ical"]),
    }
}

/// An unaccented final syllable that keeps its full vowel: object,
/// contact, program, import.
fn keeps_full(w: &[u8], pieces: &[Piece], k: usize) -> bool {
    let p = &pieces[k];
    let Unit::Vowel(v) = p.unit else { return false };
    if pieces[k + 1..].iter().any(|q| matches!(q.unit, Unit::Vowel(x) if x.hint != Hint::Never)) {
        return false;
    }
    let tail = &w[p.start..];
    if v.letter == 0 {
        // or + consonant: import, perform; not effort, comfort.
        return tail.len() >= 3 && tail.starts_with(b"or") && !is_vowel_letter(tail[2]) && tail[2] != b'r' && !(p.start > 0 && w[p.start - 1] == b'f' && p.start > 1 && !is_vowel_letter(w[p.start - 2])) && !w[..p.start].ends_with(b"w");
    }
    if v.length != Length::Short || !matches!(v.letter, b'a' | b'e' | b'o') {
        return false;
    }
    ["ct", "pt", "x", "mp", "ck"].iter().any(|e| &tail[1..] == e.as_bytes()) && !(tail == b"ock" && p.start > 2)
        || w.ends_with(b"gram") && tail == b"am"
        || tail == b"ac" && p.start > 0 && w[p.start - 1] == b'i'
}

/// An unaccented first syllable closed by two consonants keeps its full
/// vowel when it is not a prefix: activity, technology, fantastic.
fn first_closed(w: &[u8], pieces: &[Piece], k: usize, prefixed: bool) -> bool {
    let p = &pieces[k];
    let Unit::Vowel(v) = p.unit else { return false };
    if prefixed || pieces[..k].iter().any(|q| matches!(q.unit, Unit::Vowel(_))) {
        return false;
    }
    if v.letter == 0 && (w[p.start..].starts_with(b"ar") || w[p.start..].starts_with(b"or")) && p.end - p.start == 2 {
        // barbaric, organic.
        return !w[p.end..].first().is_some_and(|&c| is_vowel_letter(c));
    }
    if v.length != Length::Short || !matches!(v.letter, b'a' | b'e' | b'o') {
        return false;
    }
    let cons = &w[p.end..];
    let n = cons.iter().take_while(|&&c| !is_vowel_letter(c)).count();
    n >= 2 && cons[0] != cons[1] && !(n == 2 && matches!(cons[1], b'r' | b'l') && !matches!(cons[0], b'r' | b'l' | b'n' | b'm' | b's')) && !(n == 2 && matches!(&cons[..2], b"th" | b"sh" | b"ch" | b"ph" | b"ck"))
}

/// Put the sounds together.
fn sound(w: &[u8], pieces: &[Piece]) -> Vec<(Phoneme, bool)> {
    let mut out: Vec<(Phoneme, bool)> = Vec::new();
    let nuclei: Vec<usize> = (0..pieces.len()).filter(|&k| matches!(pieces[k].unit, Unit::Vowel(_))).collect();
    let prefixed = !nuclei.is_empty() && starts_with_prefix(w, pieces, &nuclei);
    for (k, piece) in pieces.iter().enumerate() {
        match piece.unit {
            Unit::Cons(ph) => {
                // ex- before a vowel is egz unless it carries the accent: exam, exercise.
                let ph: &[Phoneme] = if ph == [P::G, P::Z] && pieces[..k].iter().any(|p| matches!(p.unit, Unit::Vowel(v) if v.stress == Stress::Primary)) {
                    &[P::K, P::S]
                } else {
                    ph
                };
                for &p in ph {
                    if out.last().is_some_and(|&(q, _)| q == p) {
                        continue;
                    }
                    out.push((p, false));
                }
            }
            Unit::Vowel(v) => {
                let ph: &[Phoneme] = match v.stress {
                    Stress::None if first_closed(w, pieces, k, prefixed) => {
                        if v.letter == 0 { v.long } else { v.short }
                    }
                    Stress::None if keeps_full(w, pieces, k) => {
                        if w[piece.start] == b'o' && v.letter == 0 {
                            &[P::Ao, P::R]
                        } else {
                            v.short
                        }
                    }
                    Stress::None => v.weak,
                    _ => match v.length {
                        Length::Long => v.long,
                        Length::Short => v.short,
                        Length::Open => {
                            if open_long(w, pieces, k, &v) {
                                v.long
                            } else {
                                v.short
                            }
                        }
                    },
                };
                let mut marked = v.stress != Stress::Primary;
                for &p in ph {
                    let vowel = matches!(p.manner(), crate::phoneme::Manner::Vowel | crate::phoneme::Manner::Diphthong);
                    // Y after a consonant that already palatalises (j, sh, y), and
                    // after n or l before the accent (numerical, lunar).
                    if p == P::Y && out.last().is_some_and(|&(q, _)| matches!(q, P::Jh | P::Ch | P::Sh | P::Zh | P::Y | P::R)) {
                        continue;
                    }
                    if p == P::Y && out.last().is_some_and(|&(q, _)| matches!(q, P::N | P::L)) && (v.stress == Stress::Primary || !out.iter().any(|&(_, s)| s)) {
                        continue;
                    }
                    if !vowel && out.last().is_some_and(|&(q, _)| q == p) {
                        continue;
                    }
                    let stressed = !marked && vowel;
                    if stressed {
                        marked = true;
                    }
                    out.push((p, stressed));
                }
            }
        }
    }
    out
}
