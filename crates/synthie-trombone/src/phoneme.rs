//! Phonemes as articulations: where the tongue sits, what closes the tract,
//! whether the velum is open and the folds vibrate.
//!
//! The set follows ARPAbet. Vowel positions of the classic engine were
//! fitted by grid search to Peterson & Barney's (1952) male F1/F2 averages
//! (see the `tune_vowels` example); consonants follow the original's tract
//! diagram. The physical engine shapes its vowels from the formant targets
//! in [`vowel_targets`](crate::vowel_targets) instead.

use crate::tract::Constriction;
use crate::voice::Engine;
use crate::vowel_size;
use crate::vowel_targets::{self, VowelKey};

/// How a phoneme is produced; drives timing in the speech front end.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Manner {
    /// Open tract, voiced.
    Vowel,
    /// Vowel that glides from one target to another.
    Diphthong,
    /// Full closure, released with a burst.
    Stop,
    /// Full oral closure, velum open.
    Nasal,
    /// Narrow constriction, turbulent noise.
    Fricative,
    /// Stop released into a fricative.
    Affricate,
    /// Partial narrowing without turbulence.
    Approximant,
    /// Breathy glottal noise (`h`).
    Aspirate,
    /// Silence.
    Pause,
}

/// A target vocal-tract configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Articulation {
    /// Tongue body `(index, diameter)`, or `None` to keep the current one
    /// (the speech front end fills it from the neighbouring vowel).
    pub tongue: Option<(f32, f32)>,
    /// Main constriction (tongue tip, lips, or tongue back).
    pub constriction: Option<Constriction>,
    /// Lip rounding.
    pub lips: Option<Constriction>,
    /// Velum openness, `0` (oral) to `1` (nasal).
    pub velum: f32,
    /// Whether the vocal folds vibrate.
    pub voiced: bool,
    /// Glottal tenseness override; low values make breathy noise.
    pub tenseness: Option<f32>,
    /// The vowel shape of the physical engine this articulation stands for
    /// (see [`vowel_targets`](crate::vowel_targets)): set on vowels, the
    /// glides `W` and `Y`, and both ends of a diphthong, which the physical
    /// engine shapes for that diphthong rather than as the vowel the
    /// classic engine uses there. The classic engine ignores it.
    pub vowel: Option<VowelKey>,
}

impl Articulation {
    const fn vowel(index: f32, diameter: f32) -> Self {
        Self {
            tongue: Some((index, diameter)),
            constriction: None,
            lips: None,
            velum: 0.0,
            voiced: true,
            tenseness: None,
            vowel: None,
        }
    }

    /// The same articulation, standing for the physical vowel shape `key`.
    const fn standing_for(mut self, key: VowelKey) -> Self {
        self.vowel = Some(key);
        self
    }

    const fn rounded(mut self, lips: f32) -> Self {
        self.lips = Some(Constriction::new(LIPS, lips));
        self
    }

    const fn consonant(index: f32, diameter: f32, voiced: bool) -> Self {
        Self {
            tongue: None,
            constriction: Some(Constriction::new(index, diameter)),
            lips: None,
            velum: 0.0,
            voiced,
            tenseness: None,
            vowel: None,
        }
    }

    const fn nasal(mut self) -> Self {
        self.velum = 1.0;
        self
    }

    const fn with_tongue(mut self, index: f32, diameter: f32) -> Self {
        self.tongue = Some((index, diameter));
        self
    }
}

/// Lip cell for rounding and bilabials.
const LIPS: f32 = 41.0;
/// Upper teeth, for labiodentals and dentals.
const TEETH: f32 = 39.5;
/// Alveolar ridge.
const ALVEOLAR: f32 = 36.0;
/// Hard palate behind the ridge.
const POSTALVEOLAR: f32 = 33.0;
/// Soft palate.
const VELAR: f32 = 21.0;
/// Closed.
const SHUT: f32 = 0.0;
/// Tongue body of /k g ŋ/ (classic engine).
const VELAR_TONGUE: (f32, f32) = (VELAR, 2.1);
/// Tongue body of /k g ŋ/ on the physical engine: forward of the closure,
/// leaving the throat behind it open, so that the walls there can take up
/// the air of a voiced /g/ and its voice lasts into the closure. (Where
/// the classic body sits, right on the closure, the cavity behind it is so
/// small that /g/ goes voiceless within 10 ms.)
const PHYSICAL_VELAR_TONGUE: (f32, f32) = (25.0, 2.05);
/// Narrow enough to hiss.
const HISS: f32 = 0.45;
macro_rules! phonemes {
    ($( $(#[$doc:meta])* $name:ident = $arpa:literal, $manner:ident; )*) => {
        /// An English phoneme (ARPAbet names).
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[allow(missing_docs)]
        pub enum Phoneme {
            $( $(#[$doc])* $name, )*
        }

        impl Phoneme {
            /// Every phoneme.
            pub const ALL: &'static [Phoneme] = &[$(Phoneme::$name,)*];

            /// ARPAbet symbol, e.g. `"AA"`.
            #[must_use]
            pub const fn symbol(self) -> &'static str {
                match self { $(Phoneme::$name => $arpa,)* }
            }

            /// Manner of articulation.
            #[must_use]
            pub const fn manner(self) -> Manner {
                match self { $(Phoneme::$name => Manner::$manner,)* }
            }

            /// Parse an ARPAbet symbol (case-insensitive, stress digits ignored).
            #[must_use]
            pub fn from_symbol(s: &str) -> Option<Self> {
                let s = s.trim_end_matches(|c: char| c.is_ascii_digit());
                Self::ALL.iter().copied().find(|p| p.symbol().eq_ignore_ascii_case(s))
            }
        }
    };
}

phonemes! {
    /// f**a**ther
    Aa = "AA", Vowel;
    /// c**a**t
    Ae = "AE", Vowel;
    /// b**u**t
    Ah = "AH", Vowel;
    /// th**ou**ght
    Ao = "AO", Vowel;
    /// **a**bout (schwa)
    Ax = "AX", Vowel;
    /// b**e**d
    Eh = "EH", Vowel;
    /// b**ir**d
    Er = "ER", Vowel;
    /// b**i**t
    Ih = "IH", Vowel;
    /// b**ee**t
    Iy = "IY", Vowel;
    /// b**oo**k
    Uh = "UH", Vowel;
    /// b**oo**t
    Uw = "UW", Vowel;
    /// b**i**te
    Ay = "AY", Diphthong;
    /// b**ou**t
    Aw = "AW", Diphthong;
    /// b**ai**t
    Ey = "EY", Diphthong;
    /// b**oa**t
    Ow = "OW", Diphthong;
    /// b**oy**
    Oy = "OY", Diphthong;
    P = "P", Stop;
    B = "B", Stop;
    T = "T", Stop;
    D = "D", Stop;
    K = "K", Stop;
    G = "G", Stop;
    M = "M", Nasal;
    N = "N", Nasal;
    /// si**ng**
    Ng = "NG", Nasal;
    F = "F", Fricative;
    V = "V", Fricative;
    /// **th**in
    Th = "TH", Fricative;
    /// **th**is
    Dh = "DH", Fricative;
    S = "S", Fricative;
    Z = "Z", Fricative;
    /// **sh**e
    Sh = "SH", Fricative;
    /// mea**s**ure
    Zh = "ZH", Fricative;
    /// **ch**in
    Ch = "CH", Affricate;
    /// **j**oy
    Jh = "JH", Affricate;
    L = "L", Approximant;
    R = "R", Approximant;
    W = "W", Approximant;
    Y = "Y", Approximant;
    Hh = "HH", Aspirate;
    /// Silence.
    Sil = "SIL", Pause;
}

impl Phoneme {
    /// The vowel targets of a diphthong, `None` for everything else.
    #[must_use]
    pub const fn glide(self) -> Option<(Phoneme, Phoneme)> {
        use Phoneme::{Aa, Ae, Ao, Aw, Ay, Eh, Ey, Ih, Iy, Ow, Oy, Uh};
        match self {
            Ay => Some((Aa, Iy)),
            Aw => Some((Ae, Uh)),
            Ey => Some((Eh, Ih)),
            Ow => Some((Ao, Uh)),
            Oy => Some((Ao, Iy)),
            _ => None,
        }
    }

    /// Where the tongue body sits for this sound on `engine`, as
    /// `(index, diameter)`: the two engines shape vowels differently.
    /// `None` for sounds that leave the tongue where it is. This is the
    /// position for a voice of the native size; see
    /// [`tongue_for`](Self::tongue_for) for other sizes.
    #[must_use]
    pub fn tongue(self, engine: Engine) -> Option<(f32, f32)> {
        self.tongue_for(engine, 1.0)
    }

    /// [`tongue`](Self::tongue) for a voice whose tract length is
    /// `tract_length` (as given to
    /// [`Trombone::set_tract_length`](crate::Trombone::set_tract_length)).
    /// The physical engine shapes each vowel a little differently for
    /// each size, so that its formants follow the size exactly; the
    /// classic engine does not depend on it.
    #[must_use]
    pub fn tongue_for(self, engine: Engine, tract_length: f32) -> Option<(f32, f32)> {
        match engine {
            Engine::Classic => self.articulation().tongue,
            Engine::Physical => self.articulation().physical(tract_length).0.tongue,
        }
    }

    /// The vowel shape the physical engine holds for this sound at
    /// `tract_length` (as given to
    /// [`Trombone::set_tract_length`](crate::Trombone::set_tract_length)):
    /// tongue body, lip rounding and narrowing along the tongue, solved so
    /// that its formants are its [target's](crate::vowel_targets) for that
    /// size. Diphthongs give the shape they start from (see
    /// [`glide_end`](Self::glide_end) for the other); `None` for sounds that
    /// are not vowels or glides.
    #[must_use]
    pub fn physical_shape(self, tract_length: f32) -> Option<crate::inverse::Shape> {
        self.articulation().physical_shape(tract_length)
    }

    /// Where a diphthong glides to: the classic engine's articulation of
    /// its second vowel (see [`glide`](Self::glide)), standing for the
    /// diphthong's own end shape on the physical engine. `None` for
    /// everything that is not a diphthong.
    #[must_use]
    pub const fn glide_end(self) -> Option<Articulation> {
        match self.glide() {
            Some((_, to)) => Some(to.classic().standing_for(VowelKey::End(self))),
            None => None,
        }
    }

    /// Where a vowel drifts to by its end: its own articulation on the
    /// classic engine, standing for the vowel's end shape on the physical
    /// engine (see [`VowelKey::End`]). `None` for everything that is not a
    /// vowel (diphthongs glide instead; see [`glide_end`](Self::glide_end)).
    #[must_use]
    pub const fn drift_end(self) -> Option<Articulation> {
        match self.manner() {
            Manner::Vowel => Some(self.classic().standing_for(VowelKey::End(self))),
            _ => None,
        }
    }

    /// Whether the vocal folds vibrate.
    #[must_use]
    pub const fn is_voiced(self) -> bool {
        self.articulation().voiced
    }

    /// Target tract configuration. Diphthongs return the shape they start
    /// from (for `OW` a pure /o/, which no ARPAbet monophthong covers).
    #[must_use]
    pub const fn articulation(self) -> Articulation {
        let a = self.classic();
        match self.manner() {
            Manner::Vowel => a.standing_for(VowelKey::Vowel(self)),
            Manner::Diphthong => a.standing_for(VowelKey::Start(self)),
            _ if matches!(self, Phoneme::W | Phoneme::Y) => a.standing_for(VowelKey::Vowel(self)),
            _ => a,
        }
    }

    /// [`articulation`](Self::articulation) without the
    /// [vowel shape](Articulation::vowel) it stands for: what the classic
    /// engine plays.
    #[allow(clippy::match_same_arms)] // a lookup table
    const fn classic(self) -> Articulation {
        use Phoneme as P;
        match self {
            P::Iy => Articulation::vowel(28.0, 2.15).rounded(1.3),
            P::Ih => Articulation::vowel(29.0, 2.8).rounded(1.1),
            P::Eh => Articulation::vowel(28.0, 3.5).rounded(1.3),
            P::Ae => Articulation::vowel(22.0, 2.95),
            P::Aa => Articulation::vowel(16.5, 2.25),
            P::Ao => Articulation::vowel(16.0, 2.15).rounded(1.1),
            P::Ah => Articulation::vowel(18.0, 2.15),
            P::Ax => Articulation::vowel(20.0, 3.2),
            P::Uh => Articulation::vowel(22.0, 2.3).rounded(0.95),
            P::Uw => Articulation::vowel(23.0, 2.05).rounded(0.8),
            P::Er => Articulation::vowel(19.0, 2.6)
                .rounded(1.1)
                .with_constriction(30.0, 0.75),
            P::Ay => Phoneme::Aa.classic(),
            P::Aw => Phoneme::Ae.classic(),
            P::Ey => Phoneme::Eh.classic(),
            // /o/, fitted to Hillenbrand et al. (1995): the "oh" in "go".
            P::Ow => Articulation::vowel(16.0, 2.25).rounded(0.95),
            P::Oy => Phoneme::Ao.classic(),

            P::P => Articulation::consonant(LIPS, SHUT, false),
            P::B => Articulation::consonant(LIPS, SHUT, true),
            P::T => Articulation::consonant(ALVEOLAR, SHUT, false),
            P::D => Articulation::consonant(ALVEOLAR, SHUT, true),
            P::K => Articulation::consonant(VELAR, SHUT, false)
                .with_tongue(VELAR_TONGUE.0, VELAR_TONGUE.1),
            P::G => Articulation::consonant(VELAR, SHUT, true)
                .with_tongue(VELAR_TONGUE.0, VELAR_TONGUE.1),
            P::M => Articulation::consonant(LIPS, SHUT, true).nasal(),
            P::N => Articulation::consonant(ALVEOLAR, SHUT, true).nasal(),
            P::Ng => Articulation::consonant(VELAR, SHUT, true)
                .with_tongue(VELAR_TONGUE.0, VELAR_TONGUE.1)
                .nasal(),
            P::F => Articulation::consonant(TEETH + 1.5, HISS + 0.05, false),
            P::V => Articulation::consonant(TEETH + 1.5, HISS + 0.05, true),
            P::Th => Articulation::consonant(TEETH, HISS + 0.1, false),
            P::Dh => Articulation::consonant(TEETH, HISS + 0.1, true),
            P::S => Articulation::consonant(ALVEOLAR + 1.0, HISS, false),
            P::Z => Articulation::consonant(ALVEOLAR + 1.0, HISS, true),
            P::Sh => Articulation::consonant(POSTALVEOLAR, HISS + 0.05, false).rounded(1.0),
            P::Zh => Articulation::consonant(POSTALVEOLAR, HISS + 0.05, true).rounded(1.0),
            P::Ch => Articulation::consonant(POSTALVEOLAR + 1.0, SHUT, false).rounded(1.0),
            P::Jh => Articulation::consonant(POSTALVEOLAR + 1.0, SHUT, true).rounded(1.0),
            P::L => Articulation::consonant(ALVEOLAR - 1.0, 0.9, true).with_tongue(16.0, 2.6),
            P::R => Articulation::consonant(31.5, 0.85, true)
                .with_tongue(19.0, 2.6)
                .rounded(1.1),
            P::W => Articulation::vowel(21.0, 2.1).rounded(0.75),
            P::Y => Articulation::vowel(27.5, 2.05).with_constriction(29.0, 0.95),
            P::Hh => Articulation {
                tongue: None,
                constriction: None,
                lips: None,
                velum: 0.0,
                voiced: true,
                tenseness: Some(0.0),
                vowel: None,
            },
            P::Sil => Articulation {
                tongue: None,
                constriction: None,
                lips: None,
                velum: 0.0,
                voiced: false,
                tenseness: None,
                vowel: None,
            },
        }
    }
}

impl VowelKey {
    /// The articulation that stands for this shape: the phoneme's own for a
    /// vowel, glide or diphthong start, its [`glide_end`](Phoneme::glide_end)
    /// or [`drift_end`](Phoneme::drift_end) for an end.
    #[must_use]
    pub const fn articulation(self) -> Option<Articulation> {
        match self {
            VowelKey::Vowel(p) | VowelKey::Start(p) => Some(p.articulation()),
            VowelKey::End(p) => match p.glide_end() {
                Some(a) => Some(a),
                None => p.drift_end(),
            },
        }
    }
}

impl Articulation {
    /// The same articulation as the physical engine needs it, plus the
    /// narrowing along the tongue that belongs to its vowel shape.
    /// Articulations are stored for the classic engine; here a vowel's
    /// tongue and lips are replaced by the shape solved for its
    /// [target](crate::vowel_targets), and a consonant that borrowed a
    /// vowel's tongue position gets that vowel's physical tongue position.
    ///
    /// The shapes are those for a voice of `tract_length` (see
    /// [`vowel_size`](crate::vowel_size)), so the formants follow the size.
    pub(crate) fn physical(self, tract_length: f32) -> (Self, Option<Constriction>) {
        self.physical_in(tract_length, &vowel_size::OwnShapes::NONE)
    }

    /// [`physical`](Self::physical) for a voice that holds some vowel
    /// shapes of its own (`held`) in place of the table's.
    pub(crate) fn physical_in(
        self,
        tract_length: f32,
        held: &vowel_size::OwnShapes,
    ) -> (Self, Option<Constriction>) {
        // The vowel this stands for, or else the first one whose tongue it
        // borrowed. Not /w/'s: its classic tongue is also that of /k g ŋ/,
        // which get their own (see `PHYSICAL_VELAR_TONGUE`).
        let key = self.vowel.or_else(|| {
            vowel_targets::TARGETS
                .iter()
                .map(|t| t.key)
                .filter(|k| k.phoneme() != Phoneme::W)
                .find(|k| k.articulation().is_some_and(|a| a.tongue == self.tongue))
        });
        let Some((key, own)) = key.and_then(|k| Some((k, k.articulation()?))) else {
            if self.tongue == Some(VELAR_TONGUE) {
                let mut a = self;
                a.tongue = Some(PHYSICAL_VELAR_TONGUE);
                return (a, None);
            }
            return (self, None);
        };
        let Some((tongue, lips, narrow)) = vowel_size::shape_in(key, tract_length, held) else {
            return (self, None);
        };
        let mut a = self;
        a.tongue = Some(tongue);
        if self.tongue == own.tongue
            && self.constriction == own.constriction
            && self.lips == own.lips
        {
            // The vowel itself, not a consonant leaning on it.
            a.constriction = None;
            a.lips = lips.map(|d| Constriction::new(LIPS, d));
            return (a, Some(Constriction::new(narrow.0, narrow.1)));
        }
        (a, None)
    }

    /// The vowel shape the physical engine holds for this articulation in
    /// a voice of `tract_length` (see [`Phoneme::physical_shape`]), or
    /// `None` if it is not a vowel's.
    #[must_use]
    pub fn physical_shape(self, tract_length: f32) -> Option<crate::inverse::Shape> {
        let (a, narrowing) = self.physical(tract_length);
        let ((tongue_index, tongue_diameter), narrowing) = (a.tongue?, narrowing?);
        Some(crate::inverse::Shape {
            tongue_index,
            tongue_diameter,
            lips: a.lips.map(|c| c.diameter),
            narrowing: Some((narrowing.index, narrowing.diameter)),
        })
    }

    const fn with_constriction(mut self, index: f32, diameter: f32) -> Self {
        self.constriction = Some(Constriction::new(index, diameter));
        self
    }
}

impl core::fmt::Display for Phoneme {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.symbol())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbols_round_trip() {
        for &p in Phoneme::ALL {
            assert_eq!(Phoneme::from_symbol(p.symbol()), Some(p));
            assert_eq!(
                Phoneme::from_symbol(&p.symbol().to_ascii_lowercase()),
                Some(p)
            );
        }
        assert_eq!(Phoneme::from_symbol("AA1"), Some(Phoneme::Aa));
        assert_eq!(Phoneme::from_symbol("XX"), None);
    }

    #[test]
    fn diphthongs_glide_between_vowels() {
        for &p in Phoneme::ALL {
            match p.manner() {
                Manner::Diphthong => {
                    let (a, b) = p.glide().unwrap();
                    assert_eq!(a.manner(), Manner::Vowel);
                    assert_eq!(b.manner(), Manner::Vowel);
                }
                _ => assert!(p.glide().is_none()),
            }
        }
    }

    #[test]
    fn every_vowel_target_has_a_physical_shape_for_every_size() {
        for t in vowel_targets::TARGETS {
            let a = t.key.articulation().unwrap();
            let native = a.physical_shape(1.0).unwrap_or_else(|| panic!("{}", t.key));
            assert_eq!(
                native.lips.is_some(),
                t.lips == vowel_targets::Lips::Rounded,
                "{}",
                t.key
            );
            // Another shape for a child.
            assert_ne!(a.physical_shape(0.6), Some(native), "{}", t.key);
        }
    }

    #[test]
    fn diphthongs_start_and_end_on_their_own_shapes() {
        for &p in Phoneme::ALL {
            let Some(end) = p.glide_end() else { continue };
            let (from, to) = p.glide().unwrap();
            // The classic engine plays the vowels it is spelled with...
            assert_eq!(end.tongue, to.articulation().tongue, "{p}");
            if p != Phoneme::Ow {
                assert_eq!(p.articulation().tongue, from.articulation().tongue, "{p}");
            }
            // ...the physical engine the diphthong's own ends.
            assert_eq!(end.vowel, Some(VowelKey::End(p)));
            assert_eq!(p.articulation().vowel, Some(VowelKey::Start(p)));
            assert!(end.physical_shape(1.0).is_some(), "{p}");
        }
    }

    #[test]
    fn stops_and_nasals_close_the_tract() {
        for &p in Phoneme::ALL {
            if matches!(p.manner(), Manner::Stop | Manner::Nasal | Manner::Affricate) {
                let c = p.articulation().constriction.unwrap();
                assert!(c.diameter <= 0.3, "{p}");
                assert_eq!(
                    p.articulation().velum > 0.0,
                    p.manner() == Manner::Nasal,
                    "{p}"
                );
            }
        }
    }
}
