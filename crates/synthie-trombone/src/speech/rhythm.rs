//! Where each phone stands in its word and syllable, for the physical
//! engine's rhythm (round 11): how long the rules make it
//! ([`stretch`]), and which phones keep a floor on their length when a
//! speaker's schedule is followed ([`held_in_transfer`]).
//!
//! Measured phone by phone in the LibriSpeech reader's 16 sentences, the
//! two reference recordings and the owner's recording against the rules
//! saying the same text (each speaker's durations over the rules', at the
//! speaker's own pace, the phrase's last syllable left out; a fit of the
//! log ratio on the phone's class and its place): the consonants that open
//! a stressed syllable are 1.30 to 1.37 times the rules' for all four
//! speakers (the reader's halves 1.35 and 1.27), where no other place in
//! the syllable is; each consonant of a cluster is 0.81 of a single one's
//! for the reader (0.94 for the references; the rules made them alike);
//! /ə/ is 0.69 to 0.76 of the rules' everywhere; the consonants of
//! function words other than stops 0.78 for the reader ("the", "of"), and
//! the diphthongs 0.67 (the references have one, at a phrase end). The
//! stressed vowels themselves were not short (tense 0.89 to 0.99). With
//! these the rules' vowels were also too even against people's: the
//! normalised pairwise variability of successive vowel intervals (nPVI-V)
//! was 30 to 43 for the rules and 57 to 58 for every speaker.

use super::Token;
use crate::{Manner, Phoneme, Tuning};
use alloc::vec::Vec;

/// Where a phone stands (see the module).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)] // independent flags
pub(crate) struct Role {
    /// Its word has no stressed vowel of its own ("the", "of", a weak form).
    pub(crate) function: bool,
    /// The syllable it belongs to is stressed (a vowel: it is stressed).
    pub(crate) stressed: bool,
    /// A consonant before its syllable's vowel.
    pub(crate) onset: bool,
    /// One of two or more consonants side by side before or after the same
    /// vowel in its word ("planks", "blue", "sixth").
    pub(crate) cluster: bool,
    /// In the last syllable of its phrase, which the rules lengthen on
    /// their own (`Tuning::final_coda_stretch` and the last vowel's
    /// lengthening); [`stretch`] leaves it alone.
    pub(crate) last: bool,
    /// In the first syllable of its phrase, through which the lungs are
    /// still building up their pressure (`Tuning::phrase_onset`);
    /// [`stretch`] leaves it alone.
    pub(crate) first: bool,
}

fn is_vowel(p: Phoneme) -> bool {
    matches!(p.manner(), Manner::Vowel | Manner::Diphthong)
}

/// Whether `a b` may open an English syllable.
fn onset_pair(a: Phoneme, b: Phoneme) -> bool {
    use Phoneme as P;
    match b {
        P::L => matches!(a, P::P | P::B | P::K | P::G | P::F | P::S),
        P::R => matches!(
            a,
            P::P | P::B | P::K | P::G | P::F | P::T | P::D | P::Th | P::Sh
        ),
        P::W => matches!(a, P::T | P::D | P::K | P::G | P::S | P::Th),
        P::Y => matches!(a, P::P | P::B | P::K | P::M | P::F | P::V | P::Hh),
        P::P | P::T | P::K | P::M | P::N | P::F => a == P::S,
        _ => false,
    }
}

/// How many of the consonants `between` (all of them in one word, between
/// two vowels) open the second syllable: the longest run at their end that
/// may open an English syllable, at least the last.
fn onset_of(between: &[Phoneme]) -> usize {
    let n = between.len();
    if n >= 3
        && between[n - 3] == Phoneme::S
        && matches!(between[n - 2], Phoneme::P | Phoneme::T | Phoneme::K)
        && onset_pair(between[n - 2], between[n - 1])
    {
        3
    } else if n >= 2 && onset_pair(between[n - 2], between[n - 1]) {
        2
    } else {
        n.min(1)
    }
}

/// The [`Role`] of every phone of `tokens`, in order.
pub(crate) fn roles(tokens: &[Token]) -> Vec<Role> {
    let mut out = Vec::new();
    // Each phone's syllable: the place of its vowel among the text's phones.
    let mut nuclei: Vec<Option<usize>> = Vec::new();
    let mut vowels: Vec<bool> = Vec::new();
    let mut word: Vec<(Phoneme, bool)> = Vec::new();
    let mut phrase = 0;
    for t in tokens {
        match *t {
            Token::Phone { phoneme, stressed } => {
                word.push((phoneme, stressed));
                vowels.push(is_vowel(phoneme));
            }
            Token::WordBreak => {
                roles_of_word(&word, &mut out, &mut nuclei);
                word.clear();
            }
            Token::Break(_) => {
                roles_of_word(&word, &mut out, &mut nuclei);
                word.clear();
                mark_last(&mut out[phrase..], &nuclei[phrase..], &vowels, phrase);
                phrase = out.len();
            }
        }
    }
    roles_of_word(&word, &mut out, &mut nuclei);
    mark_last(&mut out[phrase..], &nuclei[phrase..], &vowels, phrase);
    out
}

/// Mark the phones of a phrase (its roles and syllables, starting at phone
/// `from` of `vowels`) that are in its first and in its last syllable.
fn mark_last(roles: &mut [Role], nuclei: &[Option<usize>], vowels: &[bool], from: usize) {
    let span = from..from + roles.len();
    let last = span.clone().rev().find(|&k| vowels[k]);
    let first = span.clone().find(|&k| vowels[k]);
    for (k, (r, n)) in span.zip(roles.iter_mut().zip(nuclei)) {
        r.last = last.is_some() && *n == last;
        // (A consonant of no syllable before the first vowel counts too.)
        r.first = first.is_some() && (*n == first || (n.is_none() && first.is_some_and(|f| k < f)));
    }
}

fn roles_of_word(word: &[(Phoneme, bool)], out: &mut Vec<Role>, nuclei: &mut Vec<Option<usize>>) {
    let function = !word.iter().any(|&(_, s)| s);
    let start = out.len();
    out.extend(word.iter().map(|_| Role {
        function,
        ..Role::default()
    }));
    let roles = &mut out[start..];
    let vowels: Vec<usize> = (0..word.len()).filter(|&k| is_vowel(word[k].0)).collect();
    let (Some(&first), Some(&last)) = (vowels.first(), vowels.last()) else {
        // No vowel: all onset, of nothing stressed.
        for r in roles.iter_mut() {
            r.onset = true;
            r.cluster = word.len() > 1;
        }
        nuclei.extend(word.iter().map(|_| None));
        return;
    };
    // Each phone's vowel (its syllable's) and whether it is an onset.
    let mut nucleus = alloc::vec![0usize; word.len()];
    for (k, r) in roles.iter_mut().enumerate() {
        if is_vowel(word[k].0) {
            nucleus[k] = k;
        } else if k < first {
            nucleus[k] = first;
            r.onset = true;
        } else if k > last {
            nucleus[k] = last;
        }
    }
    for pair in vowels.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let between: Vec<Phoneme> = word[a + 1..b].iter().map(|&(p, _)| p).collect();
        let on = onset_of(&between);
        for (i, k) in (a + 1..b).enumerate() {
            if i + on >= between.len() {
                nucleus[k] = b;
                roles[k].onset = true;
            } else {
                nucleus[k] = a;
            }
        }
    }
    for k in 0..word.len() {
        roles[k].stressed = word[nucleus[k]].1;
    }
    nuclei.extend(nucleus.iter().map(|&k| Some(start + k)));
    // Clusters: runs of consonants on one side of one vowel.
    let mut k = 0;
    while k < word.len() {
        if is_vowel(word[k].0) {
            k += 1;
            continue;
        }
        let mut j = k + 1;
        while j < word.len()
            && !is_vowel(word[j].0)
            && nucleus[j] == nucleus[k]
            && roles[j].onset == roles[k].onset
        {
            j += 1;
        }
        if j - k > 1 {
            for r in &mut roles[k..j] {
                r.cluster = true;
            }
        }
        k = j;
    }
}

/// How much longer the physical engine's rules make phone `p` where it
/// stands than its own length (see the module and the `Tuning` fields of
/// round 11); `1` for a pause or a phone of no word.
pub(crate) fn stretch(p: Phoneme, role: Role, timing: &Tuning) -> f32 {
    // A phrase's last syllable keeps the rules' own lengthening (the last
    // vowel's and `Tuning::final_coda_stretch`); the fit above leaves it
    // out. Made like the rest, the tap of "water" lost its voice
    // (`a_t_or_d_between_vowels_is_a_tap`). Its first syllable keeps them
    // too: the lungs are still building up their pressure there, and with
    // the /ə/ of "a" shorter the voice after the /p/ of "a pa" did not
    // start and the /d/ of "a da" was voiced through 0.74 of its closure.
    if p == Phoneme::Sil || role.last || role.first {
        return 1.0;
    }
    let pace = timing.rhythm_pace;
    if is_vowel(p) {
        return pace
            * match p {
                Phoneme::Ax => timing.schwa_length,
                _ if p.manner() == Manner::Diphthong => timing.diphthong_hold,
                _ => 1.0,
            };
    }
    // The weak fricatives /f v θ ð h/ are not drawn out at the start of a
    // stressed syllable: the speakers' are shorter than the rules' already
    // (0.68 to 0.85 of them, wherever they stand), and a longer /f/ after
    // /s/ ("makes fine") stood out of the hand-over between them
    // (`tests/sibilant_gate.rs`).
    let weak = matches!(p.manner(), Manner::Fricative | Manner::Aspirate)
        && !matches!(p, Phoneme::S | Phoneme::Z | Phoneme::Sh | Phoneme::Zh);
    let mut f = pace;
    if role.function && p.manner() != Manner::Stop {
        f *= timing.function_consonant_length;
    }
    if role.onset && role.stressed && !role.function && !weak {
        f *= timing.stressed_onset_length;
    }
    // Not a sibilant closing a cluster ("sells", "planks"): the reader's
    // are 1.11 of the rules' (13 of them), where his other consonants in
    // clusters are shorter, and the /z/ of "sells" made shorter before the
    // /s/ of "sea" was a gap the voice had to widen to hiss
    // (`the_regulator_barely_touches_the_default_voice`).
    let sibilant_coda =
        !role.onset && matches!(p, Phoneme::S | Phoneme::Z | Phoneme::Sh | Phoneme::Zh);
    if role.cluster && !sibilant_coda {
        f *= timing.cluster_length;
    }
    f
}

/// Whether phone `p` keeps the transfer's floor on its length
/// ([`Tuning::follow_shortest`]) when a speaker's schedule makes it short,
/// rather than the light floor of the rest ([`Tuning::follow_shortest_light`]):
/// the stressed vowels of content words, the stops (their closure and
/// release, whatever the word), the sibilants and affricates, and the
/// consonants of clusters in content words.
pub(crate) fn held_in_transfer(p: Phoneme, role: Role) -> bool {
    if is_vowel(p) {
        return role.stressed && !role.function;
    }
    matches!(p.manner(), Manner::Stop | Manner::Affricate)
        || matches!(p, Phoneme::S | Phoneme::Z | Phoneme::Sh | Phoneme::Zh)
        || (role.cluster && !role.function)
}

/// EXPERIMENT: the classes of a phone for leaving the floor off one at a
/// time, as bits.
pub(crate) fn classes(p: Phoneme, role: Role) -> u32 {
    let mut b = 0;
    if is_vowel(p) {
        b |= if role.function {
            4
        } else if role.stressed {
            1
        } else {
            2
        };
        return b;
    }
    b |= match p.manner() {
        Manner::Stop => 8,
        Manner::Affricate => 16,
        Manner::Fricative if matches!(p, Phoneme::S | Phoneme::Z | Phoneme::Sh | Phoneme::Zh) => 16,
        Manner::Fricative | Manner::Aspirate => 32,
        Manner::Nasal => 64,
        _ => 128,
    };
    if role.function {
        b |= 256;
    }
    if role.cluster && !role.function {
        b |= 512;
    }
    b
}

#[cfg(test)]
mod tests {
    use super::super::phonemize;
    use super::*;

    fn of(text: &str) -> Vec<Role> {
        roles(&phonemize(text))
    }

    #[test]
    fn onsets_codas_and_clusters() {
        // DH AX | P L AE NG K S
        let r = of("the planks");
        assert!(r[0].function && r[1].function && !r[2].function);
        assert!(r[2].onset && r[2].cluster && r[3].onset && r[3].cluster && r[3].stressed);
        assert!(!r[5].onset && r[5].cluster && r[7].cluster && r[7].stressed);
        // K AX N UW: the /n/ opens the stressed syllable, the /k/ the
        // unstressed one.
        let r = of("canoe");
        assert!(r[0].onset && !r[0].stressed);
        assert!(r[2].onset && r[2].stressed && !r[2].cluster);
    }

    #[test]
    fn stressed_onsets_are_drawn_out_and_function_words_brisk() {
        let t = Tuning::DEFAULT;
        let toks = phonemize("in the dark blue sea");
        let r = roles(&toks);
        let p: Vec<Phoneme> = toks
            .iter()
            .filter_map(|t| match t {
                Token::Phone { phoneme, .. } => Some(*phoneme),
                _ => None,
            })
            .collect();
        // IH N | DH AX | D AA R K | B L UW | S IY
        assert_eq!(p.len(), 13, "{p:?}");
        let f = |k: usize| stretch(p[k], r[k], &t);
        // The first and the last syllable keep the rules' own lengths.
        assert!(r[0].first && r[1].first && !r[2].first);
        assert!((f(0) - 1.0).abs() < 1e-6 && (f(1) - 1.0).abs() < 1e-6);
        assert!(r[11].last && r[12].last && !r[10].last);
        assert!((f(11) - 1.0).abs() < 1e-6 && (f(12) - 1.0).abs() < 1e-6);
        assert!(f(2) < 1.0 && f(3) < 1.0, "the: {} {}", f(2), f(3));
        assert!(f(4) > 1.15, "the /d/ of dark: {}", f(4));
        assert!((f(5) - t.rhythm_pace).abs() < 1e-6);
        assert!(f(8) > f(5) && f(8) < f(4), "the /b/ of blue: {}", f(8));
        assert!(held_in_transfer(p[4], r[4]) && held_in_transfer(p[5], r[5]));
        assert!(!held_in_transfer(p[2], r[2]) && !held_in_transfer(p[3], r[3]));
    }
}
