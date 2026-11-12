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
    /// Where a diphthong ends.
    End(Phoneme),
}

impl VowelKey {
    /// The phoneme this belongs to.
    #[must_use]
    pub const fn phoneme(self) -> Phoneme {
        match self {
            Self::Vowel(p) | Self::Start(p) | Self::End(p) => p,
        }
    }
}

impl core::fmt::Display for VowelKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Vowel(p) => write!(f, "{p}"),
            Self::Start(p) => write!(f, "{p}<"),
            Self::End(p) => write!(f, "{p}>"),
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
    vowel(Phoneme::Ih, [ 427.0, 2034.0, 2684.0], Lips::Spread),  // 390 1990 2550
    vowel(Phoneme::Eh, [ 580.0, 1799.0, 2605.0], Lips::Spread),  // 530 1840 2480
    vowel(Phoneme::Ae, [ 588.0, 1952.0, 2601.0], Lips::Spread),  // 660 1720 2410
    vowel(Phoneme::Ah, [ 623.0, 1200.0, 2550.0], Lips::Spread),  // 640 1190 2390
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
    start(Phoneme::Ey, [ 476.0, 2089.0, 2691.0], Lips::Spread),  // Hillenbrand's /e/
    // ...the /a/ of "bite", a quarter of the way from /ɑ/ to /æ/...
    start(Phoneme::Ay, [ 723.0, 1488.0, 2542.0], Lips::Spread),
    start(Phoneme::Ow, [ 497.0,  910.0, 2459.0], Lips::Rounded), // Hillenbrand's /o/
    // ...the /a/ of "bout", half-way from /ɑ/ to /æ/...
    start(Phoneme::Aw, [ 678.0, 1643.0, 2562.0], Lips::Spread),
    start(Phoneme::Oy, [ 652.0,  997.0, 2538.0], Lips::Spread),  // /ɔ/
    // ...and where they end.
    end(Phoneme::Ey, FRONT_OFFGLIDE, Lips::Spread),
    end(Phoneme::Ay, FRONT_OFFGLIDE, Lips::Spread),
    end(Phoneme::Ow, BACK_OFFGLIDE, Lips::Rounded),
    end(Phoneme::Aw, BACK_OFFGLIDE, Lips::Rounded),
    end(Phoneme::Oy, FRONT_OFFGLIDE, Lips::Spread),
];

/// The target of `key`, if it has one.
#[must_use]
pub fn target(key: VowelKey) -> Option<&'static Target> {
    TARGETS.iter().find(|t| t.key == key)
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
