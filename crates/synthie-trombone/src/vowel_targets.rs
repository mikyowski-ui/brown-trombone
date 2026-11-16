//! The formants every vowel of the physical engine is shaped to: one table
//! of F1, F2 and F3 per vowel, glide and diphthong end, for an adult man.
//!
//! The shapes themselves (tongue, lips, narrowing, for every body size) are
//! solved from this table offline, through the babble map
//! ([`inverse`](crate::inverse)) and then against the engine itself, by
//! `examples/vowel_size.rs`, which writes `src/vowel_size_table.rs`. To
//! move a vowel, change its row here and regenerate:
//!
//! ```sh
//! cargo run -p synthie-trombone --example vowel_size --release
//! ```
//!
//! `tests/vowel_size_gate.rs` then checks that every vowel lands within
//! 3 % of its F1 and F2 at every size.
//!
//! # Sources
//!
//! Vowels are the averages for the 45 men of Hillenbrand, Getty, Clark &
//! Wheeler (1995), "Acoustic characteristics of American English vowels",
//! JASA 97, 3099–3111, measured at the steadiest point of each vowel. The
//! men of Peterson & Barney (1952), JASA 24, 175–184, are given beside them
//! as a cross-check; their /i/ and /u/ are closer, their /æ/ and /ɑ/ more
//! open and further back. Neither study recorded schwa, glides or the ends
//! of diphthongs; how those rows were made is said at each.
//!
//! # Tuned rows
//!
//! Rows marked `tuned` were moved from their published value against the
//! phoneme recognizer of `tools/judge` (`vowel_tune.py`, on its tuning
//! material only), by at most 12 % in F1 and 10 % in F2, keeping the
//! vowels' order in F1 and F2. The moves bring the system closer to
//! General American as the recognizer knows it: /ʌ/ further front and
//! closer, so that it parts from /ɑ/ (whose Michigan F2 is as high), /æ/
//! and /ɛ/ more open, the /a/ of /aɪ/ further back and that of /aʊ/ more
//! open, /eɪ/ starting further front. /ɑ ɔ u/, the vowels the engine's
//! other gates are measured on, were tuned too (/ɑ/ and /ɔ/ further back,
//! /u/ a little front) but are kept at their published values here. The
//! published value is in the comment beside each tuned row.
//!
//! # Size
//!
//! The native tract (tract length `1.0`) is 44 cells of 0.365 cm, 16.0 cm,
//! and the radiation at rest lips adds an end correction of about 0.8 cm:
//! acoustically a 16.8 cm tube. A uniform tube of length `L` resonates at
//! `(2n − 1)·c / 4L`; the mean of `Fn / (2n − 1)` over Hillenbrand's twelve
//! vowels for men (523, 504 and 502 Hz for F1, F2 and F3) makes their
//! average man a 17.2 cm tube (c = 350 m/s). So he is tract length
//! [`ADULT_MALE_LENGTH`] here, and a voice of length `l` aims for the
//! table's formants times `ADULT_MALE_LENGTH / l` ([`formants`]).

use crate::phoneme::Phoneme;

/// The tract length, relative to the native one, of the average man of
/// Hillenbrand et al. (1995): 17.2 cm against the native 16.8 cm.
pub const ADULT_MALE_LENGTH: f32 = 1.02;

/// One vowel shape of the physical engine: a vowel or glide, or one end
/// of a diphthong (which the physical engine shapes for that diphthong,
/// not as the vowel it is spelled with).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VowelKey {
    /// A vowel or the glides `W` and `Y`.
    Vowel(Phoneme),
    /// Where a diphthong starts.
    Start(Phoneme),
    /// Where a diphthong ends, or where a vowel that glides on its own
    /// glides to (the lax vowels towards schwa, the fronted /u/ back
    /// towards the plain one).
    End(Phoneme),
    /// A vowel made further front after a consonant of the tongue's front:
    /// the /u/ of "two", "do", "soon", "canoe" (see `speech::fronts_u`);
    /// or, for /ə/, coloured by the high front vowel after it (see
    /// `speech::schwa_colour`).
    Fronted(Phoneme),
}

impl VowelKey {
    /// The phoneme this belongs to.
    #[must_use]
    pub const fn phoneme(self) -> Phoneme {
        match self {
            Self::Vowel(p) | Self::Start(p) | Self::End(p) | Self::Fronted(p) => p,
        }
    }
}

impl core::fmt::Display for VowelKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Vowel(p) => write!(f, "{p}"),
            Self::Start(p) => write!(f, "{p}<"),
            Self::End(p) => write!(f, "{p}>"),
            Self::Fronted(p) => write!(f, "{p}+"),
        }
    }
}

/// Lips of a vowel shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lips {
    /// Spread or neutral: the lips are left at rest (and part with the jaw).
    Spread,
    /// Rounded, and so pushed forward: the solver must round them.
    Rounded,
}

/// What one vowel shape aims for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Target {
    /// The shape.
    pub key: VowelKey,
    /// F1, F2 and F3 of an adult man, Hz.
    pub formants: [f32; 3],
    /// Rounded or spread.
    pub lips: Lips,
}

const fn vowel(p: Phoneme, formants: [f32; 3], lips: Lips) -> Target {
    Target {
        key: VowelKey::Vowel(p),
        formants,
        lips,
    }
}

const fn start(p: Phoneme, formants: [f32; 3], lips: Lips) -> Target {
    Target {
        key: VowelKey::Start(p),
        formants,
        lips,
    }
}

const fn end(p: Phoneme, formants: [f32; 3], lips: Lips) -> Target {
    Target {
        key: VowelKey::End(p),
        formants,
        lips,
    }
}

const fn fronted(p: Phoneme, formants: [f32; 3], lips: Lips) -> Target {
    Target {
        key: VowelKey::Fronted(p),
        formants,
        lips,
    }
}

/// Where the front offglides of `EY`, `AY` and `OY` head: half-way from
/// Hillenbrand's /ɪ/ (427, 2034, 2684) to his /i/ (342, 2322, 3000). A
/// diphthong's tail moves towards the tense vowel but seldom reaches it.
const FRONT_OFFGLIDE: [f32; 3] = [385.0, 2178.0, 2842.0];
/// Where the back offglides of `OW` and `AW` head: half-way from /ʊ/
/// (469, 1122, 2434) to /u/ (378, 997, 2343).
const BACK_OFFGLIDE: [f32; 3] = [424.0, 1060.0, 2389.0];

/// Every vowel shape's target, adult man, Hz. Vowels and glides first:
/// a consonant that borrows a vowel's tongue takes the first row whose
/// classic tongue position it has.
#[rustfmt::skip]
pub const TARGETS: &[Target] = &[
    //                              F1      F2      F3      Peterson & Barney
    vowel(Phoneme::Iy, [ 342.0, 2322.0, 3000.0], Lips::Spread),  // 270 2290 3010
    // /ɪ/ closer and further back than Hillenbrand's, away from the start
    // of /eɪ/ (480 1950): in sentences the engine's stood at 448 / 1998 Hz
    // (LPC, middle of the vowel) next to its /eɪ/ start at 510 / 1930,
    // where the LibriSpeech reader's /ɪ/ is 385 / 1747 and his /eɪ/ starts
    // at 422 / 1892 (round 9; F2 1850 alone, in round 6, made it central).
    vowel(Phoneme::Ih, [ 400.0, 1900.0, 2650.0], Lips::Spread), // moved; published 427 2034 2684; P&B 390 1990 2550
    vowel(Phoneme::Eh, [ 609.0, 1799.0, 2605.0], Lips::Spread), // tuned; published 580 1799 2605; P&B 530 1840 2480
    vowel(Phoneme::Ae, [ 630.0, 1880.0, 2601.0], Lips::Spread), // tuned; published 588 1952 2601; P&B 660 1720 2410
    vowel(Phoneme::Ah, [ 593.0, 1260.0, 2550.0], Lips::Spread), // tuned; published 623 1200 2550; P&B 640 1190 2390
    vowel(Phoneme::Aa, [ 768.0, 1333.0, 2522.0], Lips::Spread),  // 730 1090 2440
    // American /ɔ/ is rounded only a little, if at all.
    vowel(Phoneme::Ao, [ 652.0,  997.0, 2538.0], Lips::Spread),  // 570  840 2410
    vowel(Phoneme::Uh, [ 469.0, 1122.0, 2434.0], Lips::Rounded), // 440 1020 2240
    vowel(Phoneme::Uw, [ 378.0,  997.0, 2343.0], Lips::Rounded), // 300  870 2240
    vowel(Phoneme::Er, [ 474.0, 1379.0, 1710.0], Lips::Rounded), // 490 1350 1690
    // Schwa: the uniform tube of the same man, (2n − 1)·c / 4L.
    vowel(Phoneme::Ax, [ 510.0, 1530.0, 2550.0], Lips::Spread),
    // The glides are their vowels made closer: F1 about 10 % under their
    // vowel's, /w/'s F2 lower than /u/'s, /j/'s F2 a little under /i/'s.
    // (Much closer still and a long tract cannot follow: its yielding
    // walls hold F1 up.)
    vowel(Phoneme::W,  [ 340.0,  800.0, 2300.0], Lips::Rounded),
    vowel(Phoneme::Y,  [ 310.0, 2250.0, 3000.0], Lips::Spread),
    // Diphthongs: where they start...
    start(Phoneme::Ey, [ 480.0, 1950.0, 2620.0], Lips::Spread), // tuned; published 476 2089 2691; Hillenbrand's /e/
    // ...the /a/ of "bite", a quarter of the way from /ɑ/ to /æ/...
    start(Phoneme::Ay, [ 723.0, 1417.0, 2542.0], Lips::Spread), // tuned; published 723 1488 2542
    start(Phoneme::Ow, [ 497.0,  910.0, 2459.0], Lips::Rounded), // Hillenbrand's /o/
    // ...the /a/ of "bout", half-way from /ɑ/ to /æ/...
    start(Phoneme::Aw, [ 712.0, 1643.0, 2562.0], Lips::Spread), // tuned; published 678 1643 2562
    start(Phoneme::Oy, [ 652.0,  997.0, 2538.0], Lips::Spread),  // /ɔ/
    // ...and where they end: /eɪ/ glides the whole way, as a speaker's
    // does, F1 falling by some 120 Hz and F2 rising by some 200.
    end(Phoneme::Ey, [ 360.0, 2150.0, 2800.0], Lips::Spread), // tuned; published FRONT_OFFGLIDE
    end(Phoneme::Ay, FRONT_OFFGLIDE, Lips::Spread),
    end(Phoneme::Ow, BACK_OFFGLIDE, Lips::Rounded),
    end(Phoneme::Aw, BACK_OFFGLIDE, Lips::Rounded),
    end(Phoneme::Oy, FRONT_OFFGLIDE, Lips::Spread),
    // Where the lax vowels glide to over their last third: 60 % of the
    // way from the vowel to schwa (AX above). The heavy tongue body gets
    // a little over a third of the way there before the next sound takes
    // it, so the vowel ends some 15 % of the way to schwa, as English lax
    // vowels do (they centralise; the tense ones do not). Not /ɪ/: the
    // reader's closes a little over its second half (F1 -19 Hz, F2 -34
    // from 40 to 80 % of it), the owner's "slid" too (F1 -18), and the
    // engine's, opening towards schwa there (F1 +17, F2 -82) after a
    // body arriving from the sound before (F1 -80 over its first 40 %),
    // was open, closed, open: /eɪ/. Its glide end is a little closer.
    end(Phoneme::Ih, [ 385.0, 1860.0, 2650.0], Lips::Spread),
    end(Phoneme::Eh, [ 550.0, 1638.0, 2572.0], Lips::Spread),
    end(Phoneme::Ah, [ 543.0, 1422.0, 2550.0], Lips::Spread),
    end(Phoneme::Uh, [ 494.0, 1367.0, 2504.0], Lips::Rounded),
    // A stressed /æ/ glides the same way over its end ("planks").
    end(Phoneme::Ae, [ 558.0, 1670.0, 2570.0], Lips::Spread),
    // After /t d s z/ the /u/ is made further front and closer, still
    // rounded ("two", "do", "soon"; see `speech::fronts_u`): the
    // LibriSpeech reader's starts at F2 1700 to 2200 Hz with F1 250 to
    // 310 after /t d n/, the owner's "canoe" at 1860 to 1940 with F1 340
    // to 360; both glide back over a long vowel, to 1300 to 1400 at its
    // end, so the fronted /u/ glides to its end shape below. The owner
    // asked for 1600 to 2000 Hz: at 1600 and at 1800 the recognizer heard
    // 17 and 18 of 20 /u/ after coronals right, so the lower end is used.
    fronted(Phoneme::Uw, [ 320.0, 1600.0, 2300.0], Lips::Rounded),
    // Where the fronted /u/ glides back to: part of the way to the plain
    // /u/. The reader's ends at 1250 to 1650 Hz, the owner's "canoe" at
    // 1330; gliding to the plain /u/ (997 Hz), the engine's "do" fell from
    // 1860 to 1140.
    end(Phoneme::Uw, [ 350.0, 1250.0, 2340.0], Lips::Rounded),
    // An unstressed /ə/ before a high front or fronted vowel takes its
    // colour, higher and further front than the plain one (see
    // `speech::schwa_colour`): the LibriSpeech reader's /ə/ before a front
    // vowel stands at F1 372, F2 1604 Hz (median of 20, LPC at its middle;
    // his others 460 and 1614, the engine's 484 to 510 and 1516 to 1547
    // everywhere), the owner's in "canoe" near 300 and 1800 to 1890.
    fronted(Phoneme::Ax, [ 400.0, 1750.0, 2600.0], Lips::Spread),
];

/// Whether a voice of `tuning` speaks with the vowel shapes solved for the
/// tract whose length changes (wave 2A: `AnatomyConfig::lip_horn` or
/// `larynx_travel` above `0`; `src/vowel_size_wave_table.rs` and
/// `src/vowel_modes_wave_table.rs`, solved for the values written in their
/// headers) rather than for the default tract. A voice with only some of
/// those numbers on, or other values, reads the same lengthened tables.
#[must_use]
pub fn lengthened(tuning: &crate::Tuning) -> bool {
    tuning.anatomy.lip_horn > 0.0 || tuning.anatomy.larynx_travel > 0.0
}

/// How many vowel shapes there are (the length of [`TARGETS`]).
pub const TARGET_COUNT: usize = TARGETS.len();

/// The target of `key`, if it has one.
#[must_use]
pub fn target(key: VowelKey) -> Option<&'static Target> {
    TARGETS.iter().find(|t| t.key == key)
}

/// Where `key` is in [`TARGETS`], if it is there.
#[must_use]
pub fn position(key: VowelKey) -> Option<usize> {
    TARGETS.iter().position(|t| t.key == key)
}

/// F1, F2 and F3, Hz, that `key` aims for in a voice of `tract_length`
/// (as given to [`Trombone::set_tract_length`](crate::Trombone::set_tract_length)):
/// the table's, scaled from the adult man's length to this one.
#[must_use]
pub fn formants(key: VowelKey, tract_length: f32) -> Option<[f32; 3]> {
    let scale = ADULT_MALE_LENGTH / tract_length;
    target(key).map(|t| t.formants.map(|f| f * scale))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phoneme::Manner;

    #[test]
    fn every_vowel_glide_and_diphthong_end_has_one_target() {
        for &p in Phoneme::ALL {
            let keys: &[VowelKey] = match p.manner() {
                Manner::Vowel => &[VowelKey::Vowel(p)],
                Manner::Diphthong => &[VowelKey::Start(p), VowelKey::End(p)],
                _ if matches!(p, Phoneme::W | Phoneme::Y) => &[VowelKey::Vowel(p)],
                _ => &[],
            };
            for &k in keys {
                assert_eq!(TARGETS.iter().filter(|t| t.key == k).count(), 1, "{k}");
            }
        }
        for t in TARGETS {
            assert!(t.formants[0] < t.formants[1] && t.formants[1] < t.formants[2]);
        }
    }

    #[test]
    fn targets_scale_with_the_tract() {
        let k = VowelKey::Vowel(Phoneme::Aa);
        let man = formants(k, ADULT_MALE_LENGTH).unwrap();
        assert!((man[0] - 768.0).abs() < 1e-3);
        let child = formants(k, 0.6).unwrap();
        assert!((child[0] / man[0] - ADULT_MALE_LENGTH / 0.6).abs() < 1e-4);
    }
}
