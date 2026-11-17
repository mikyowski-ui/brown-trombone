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
/// Where the tip of an /l/ touches, just behind the alveolars'.
const LATERAL_PLACE: f32 = ALVEOLAR - 1.0;
/// Hard palate behind the ridge.
const POSTALVEOLAR: f32 = 33.0;
/// Soft palate.
const VELAR: f32 = 21.0;
/// Closed.
const SHUT: f32 = 0.0;
/// Widest touch diameter that still closes the tract.
const SHUT_WIDEST: f32 = 0.3;
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

// ----- consonants on the vowel's body (physical engine) ------------------

/// Tongue body a tongue-tip consonant drags the body towards: forward and
/// fairly high, where the blade behind a raised tip holds it.
const TIP_BODY: (f32, f32) = (26.5, 2.5);
/// How much narrower the physical engine makes /l r/ (touch diameter).
const APPROXIMANT_NARROWING: f32 = 0.08;
/// Widest consonant constriction that is an approximant's (touch
/// diameter); wider ones only shade a vowel.
const APPROXIMANT_WIDEST: f32 = 1.3;
/// Share of the way to [`TIP_BODY`] that alveolars (/t d n s z/) and
/// postalveolars (/ʃ ʒ tʃ dʒ/) take the body: the alveolar F2 locus near
/// 1800 Hz comes from a body held forward by the tip.
const ALVEOLAR_PULL: f32 = 0.3;
const POSTALVEOLAR_PULL: f32 = 0.45;
/// The same for /θ ð/: a tip held against the teeth draws the body
/// forward too, and with it the F2 at the gap up to the dental locus near
/// 1400 to 1600 Hz (on the vowel's body alone it had a labial's).
const DENTAL_PULL: f32 = 0.5;

/// Where /k g ŋ/ close before a vowel whose body is at tongue index
/// `index`: at the classic closure ([`VELAR`]) before back vowels, further
/// forward before front ones. Never further back: the throat behind a /g/
/// must stay large enough to take up the voice's air.
fn velar_place(index: f32) -> f32 {
    (0.3 * index + 14.5).clamp(VELAR, VELAR + 2.0)
}
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
            P::L => Articulation::consonant(LATERAL_PLACE, 0.9, true).with_tongue(16.0, 2.6),
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
    /// for a diphthong's end, and the vowel's own (standing for its end or
    /// its fronted shape) for the end of a vowel that glides on its own (a
    /// lax vowel's offglide, the glide back of a fronted /u/) and for a
    /// fronted vowel.
    #[must_use]
    pub const fn articulation(self) -> Option<Articulation> {
        match self {
            VowelKey::Vowel(p) | VowelKey::Start(p) => Some(p.articulation()),
            VowelKey::Fronted(p) if matches!(p.manner(), Manner::Vowel) => {
                Some(p.classic().standing_for(VowelKey::Fronted(p)))
            }
            VowelKey::Fronted(_) => None,
            VowelKey::End(p) => match p.glide_end() {
                Some(a) => Some(a),
                None if matches!(p.manner(), Manner::Vowel) => {
                    Some(p.classic().standing_for(VowelKey::End(p)))
                }
                None => None,
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
        let (mut a, narrowing) = self.physical_shape_of(tract_length, held);
        // /l r/ are narrower on the physical engine: with the classic
        // engine's gap their F1 stood near 550 Hz, a vowel's, where an /l/
        // has 350 to 400.
        if narrowing.is_none()
            && let Some(c) = a.constriction
            && c.diameter > HISS + 0.3
            && c.diameter < APPROXIMANT_WIDEST
        {
            a.constriction = Some(Constriction::new(
                c.index,
                c.diameter - APPROXIMANT_NARROWING,
            ));
        }
        (a, narrowing)
    }

    /// The vowel shape this stands for, or else the first one whose tongue
    /// it borrowed. Not /w/'s: its classic tongue is also that of /k g ŋ/,
    /// which get their own (see `PHYSICAL_VELAR_TONGUE`).
    fn vowel_shape_key(self) -> Option<VowelKey> {
        self.vowel.or_else(|| {
            vowel_targets::TARGETS
                .iter()
                .map(|t| t.key)
                .filter(|k| k.phoneme() != Phoneme::W)
                .find(|k| k.articulation().is_some_and(|a| a.tongue == self.tongue))
        })
    }

    /// The tongue root the physical engine holds for this articulation (see
    /// [`Shape::pharynx`](crate::inverse::Shape::pharynx)): a vowel's own,
    /// and a consonant's that of the vowel it is made on, since the root
    /// belongs to the tongue body that glides from vowel to vowel under the
    /// consonants. `None` (leave it where it is) for what has no vowel.
    pub(crate) fn physical_root(
        self,
        tract_length: f32,
        held: &vowel_size::OwnShapes,
    ) -> Option<f32> {
        let key = self.vowel_shape_key()?;
        vowel_size::shape_in(key, tract_length, held).map(|s| s.3)
    }

    /// The modal body's three numbers the physical engine holds for this
    /// articulation (`AnatomyConfig::tongue_model` 1, `held` on the modal
    /// table), or `None`, which leaves the tract to project the cosine
    /// tongue [`physical_in`](Self::physical_in) gives onto the modes: a
    /// vowel's own; a consonant's that of the vowel it is made on, moved by
    /// as much as the vowel's cosine tongue changes shape when the tip
    /// draws it towards [`TIP_BODY`] ([`leaning_on`](Self::leaning_on);
    /// `tract::drawn_modes`); /ɚ/'s for
    /// /r/; `None` for the velars and /l/, whose own tongues are their
    /// bodies, and for what has no vowel.
    pub(crate) fn physical_modes(
        self,
        tract_length: f32,
        held: &vowel_size::OwnShapes,
    ) -> Option<[f32; 3]> {
        if !held.1.modal() {
            return None;
        }
        // A vowel or glide itself (/w/ has the velars' classic tongue).
        if let Some(key) = self.vowel
            && let Some(own) = key.articulation()
            && self.tongue == own.tongue
            && self.constriction == own.constriction
            && self.lips == own.lips
        {
            return vowel_size::shape_in(key, tract_length, held).and_then(|s| s.4);
        }
        if self.tongue == Some(VELAR_TONGUE) {
            return None;
        }
        if self.own_body() {
            return if self.tongue == Phoneme::R.classic().tongue {
                vowel_size::shape_in(VowelKey::Vowel(Phoneme::Er), tract_length, held)
                    .and_then(|s| s.4)
            } else {
                None
            };
        }
        let key = self.vowel_shape_key()?;
        let shape = vowel_size::shape_in(key, tract_length, held)?;
        let q = shape.4?;
        let own = key.articulation()?;
        let itself = self.tongue == own.tongue
            && self.constriction == own.constriction
            && self.lips == own.lips;
        if itself {
            return Some(q);
        }
        Some(match self.body_pull() {
            Some((tip, w)) => crate::tract::drawn_modes(q, shape.0, tip, w),
            None => q,
        })
    }

    /// [`physical_in`](Self::physical_in) before the approximants are
    /// narrowed.
    ///
    /// A consonant made on a vowel's body (see [`leaning_on`](Self::leaning_on))
    /// holds that vowel's lips too, unless it shapes the lips itself (/ʃ ʒ
    /// tʃ dʒ r w/), holds a gap meant to hiss, or is made by the back of the
    /// tongue: a speaker rounds the lips for /u/ through the stops, nasals
    /// and liquids before it ("canoe", "blue", "moon", "to") and keeps them
    /// rounded after it ("food").
    /// Without them the lips of /u/ set off only as the vowel began, over a
    /// tongue body already in place for it: the second formant fell from
    /// about 1800 Hz to 1000 over the first 150 ms of the vowel after an
    /// /n/, which is /ju/, and "canoe" was heard as "can you". Under /s z/
    /// the rounded lips darkened the hiss ("rose" was heard as "rode"), and
    /// at /f v θ ð/ they would shape the gap itself. Under /k g ŋ/ they
    /// took the velar pinch out of the release (the second formant 10 ms
    /// into the /u/ of "[u g u]" fell to 820 Hz, under the vowel's own,
    /// where a velar's stands above it). The vowel's narrowing
    /// along the tongue stays the vowel's: held under the consonants as
    /// well, it cost 3 to 4 points of phoneme error in the held-out Harvard
    /// sentences (schwa after a consonant heard as /ɚ/).
    fn physical_shape_of(
        self,
        tract_length: f32,
        held: &vowel_size::OwnShapes,
    ) -> (Self, Option<Constriction>) {
        let Some((key, own)) = self
            .vowel_shape_key()
            .and_then(|k| Some((k, k.articulation()?)))
        else {
            if self.tongue == Some(VELAR_TONGUE) {
                let mut a = self;
                a.tongue = Some(PHYSICAL_VELAR_TONGUE);
                return (a, None);
            }
            return (self, None);
        };
        let Some((tongue, lips, narrow, _, _)) = vowel_size::shape_in(key, tract_length, held)
        else {
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
        let mut a = self.leaning_on(tongue, tract_length, held);
        let hisses = a
            .constriction
            .is_some_and(|c| c.diameter > SHUT_WIDEST && c.diameter <= crate::tract::HISS_WIDEST);
        let velar = self.tongue == Some(VELAR_TONGUE);
        if a.lips.is_none() && !hisses && !velar {
            a.lips = lips.map(|d| Constriction::new(LIPS, d));
        }
        (a, None)
    }

    /// A consonant made on top of a vowel's tongue body `body` (physical
    /// engine): the heavy body stays the vowel's and glides from vowel to
    /// vowel underneath, and the consonant is a gesture of the tip or the
    /// lips added to it, whose release comes out of a body still on its
    /// way: the locus transitions. Dental, alveolar and postalveolar
    /// tongue-tip sounds draw the body a little forward and up with the tip.
    ///
    /// Consonants with a tongue body of their own (/k g ŋ l r/; /r/ that of
    /// /ɚ/; see
    /// [`own_body`](Self::own_body)) keep it: the back of the tongue is
    /// their articulator. /k g ŋ/ close a little further forward before
    /// front vowels and further back before back ones ([`velar_place`]).
    fn leaning_on(self, body: (f32, f32), tract_length: f32, held: &vowel_size::OwnShapes) -> Self {
        let mut a = self;
        let (index, diameter) = body;
        // `w` of the way from the vowel's body to `(i0, d0)`.
        let pull = |(i0, d0): (f32, f32), w: f32| {
            (index + (i0 - index) * w, diameter + (d0 - diameter) * w)
        };
        let Some(c) = self.constriction else {
            a.tongue = Some(body);
            return a;
        };
        if self.tongue == Some(VELAR_TONGUE) {
            a.tongue = Some(PHYSICAL_VELAR_TONGUE);
            a.constriction = Some(Constriction::new(velar_place(index), c.diameter));
        } else if self.own_body() {
            // /l r/: their own body. /r/'s is that of /ɚ/ (the classic
            // engine gives both the same), as the physical engine shapes it.
            if self.tongue == Phoneme::R.classic().tongue
                && let Some((er, _, _, _, _)) =
                    vowel_size::shape_from(held.1, VowelKey::Vowel(Phoneme::Er), tract_length)
            {
                a.tongue = Some(er);
            }
        } else if (POSTALVEOLAR - 0.5..=ALVEOLAR + 1.5).contains(&c.index) && self.velum <= 0.0
            || c.index == TEETH
        {
            // (An /n/ leaves the body be: drawn up and forward behind a
            // nasal closure it is heard as /ŋ/.)
            let w = if c.index < ALVEOLAR - 1.0 {
                POSTALVEOLAR_PULL
            } else if c.index == TEETH {
                DENTAL_PULL
            } else {
                ALVEOLAR_PULL
            };
            a.tongue = Some(pull(TIP_BODY, w));
        } else {
            a.tongue = Some(body);
        }
        a
    }

    /// The vowel whose tongue body and lips this consonant is made on (see
    /// [`physical_shape_of`](Self::physical_shape_of)), if it is a
    /// consonant made on one and has no body of its own (/k g ŋ l r/ have):
    /// on the gesture path its body and lips are that vowel's gestures,
    /// spanning it, and the consonant adds only its own.
    pub(crate) fn leans_on(self) -> Option<VowelKey> {
        let key = self.vowel?;
        let own = key.articulation()?;
        let itself = self.tongue == own.tongue
            && self.constriction == own.constriction
            && self.lips == own.lips;
        (!itself && !self.own_body() && self.tongue != Some(VELAR_TONGUE)).then_some(key)
    }

    /// The tongue body a consonant's tip draws the vowel's body towards, and
    /// how far, `0‥1` (see [`leaning_on`](Self::leaning_on)): the dentals,
    /// alveolars and postalveolars but /n/.
    pub(crate) fn body_pull(self) -> Option<((f32, f32), f32)> {
        let c = self.constriction?;
        if self.tongue == Some(VELAR_TONGUE) || self.own_body() {
            return None;
        }
        let tip = (POSTALVEOLAR - 0.5..=ALVEOLAR + 1.5).contains(&c.index) && self.velum <= 0.0;
        if !(tip || c.index == TEETH) {
            return None;
        }
        let w = if c.index < ALVEOLAR - 1.0 {
            POSTALVEOLAR_PULL
        } else if c.index == TEETH {
            DENTAL_PULL
        } else {
            ALVEOLAR_PULL
        };
        Some((TIP_BODY, w))
    }

    /// Whether this consonant's gap is meant to hiss: a speaker's lips stay
    /// as they are under it, whatever the vowel's.
    pub(crate) fn hisses(self) -> bool {
        self.constriction
            .is_some_and(|c| c.diameter > SHUT_WIDEST && c.diameter <= crate::tract::HISS_WIDEST)
    }

    /// Whether this is a lateral, an /l/ (physical engine): its tip at the
    /// lateral's place and not shut, on its own body or the next vowel's
    /// (see `speech::lateral_onto_vowels`).
    #[must_use]
    pub(crate) fn lateral(self) -> bool {
        self.velum <= 0.0
            && self.constriction.is_some_and(|c| {
                c.index == LATERAL_PLACE
                    && c.diameter > SHUT_WIDEST
                    && c.diameter < APPROXIMANT_WIDEST
            })
    }

    /// Whether this is a consonant with a tongue body of its own (/k g ŋ l
    /// r/): on the physical engine the back of the tongue is then the
    /// articulator making the consonant, and moves at its speed (see
    /// `Tuning::dorsum_rate`), not at the vowel shape's.
    #[must_use]
    pub(crate) fn own_body(self) -> bool {
        // By the tongue alone: a consonant before /ɚ/ (whose classic tongue
        // is /r/'s) is drawn into its r-colouring by the back of the
        // tongue too.
        let own = |p: Phoneme| self.tongue == p.classic().tongue;
        self.constriction.is_some() && (own(Phoneme::K) || own(Phoneme::L) || own(Phoneme::R))
    }

    /// The vowel shape the physical engine holds for this articulation in
    /// a voice of `tract_length` (see [`Phoneme::physical_shape`]), or
    /// `None` if it is not a vowel's.
    #[must_use]
    pub fn physical_shape(self, tract_length: f32) -> Option<crate::inverse::Shape> {
        self.physical_shape_with(tract_length, &crate::Tuning::DEFAULT)
    }

    /// [`physical_shape`](Self::physical_shape) for a voice of `tuning`: from
    /// the vowel table its tongue model (`AnatomyConfig::tongue_model`; the
    /// modal body's three numbers, lips and narrowing) and its tract
    /// ([`vowel_targets::lengthened`](crate::vowel_targets::lengthened)) read.
    #[must_use]
    pub fn physical_shape_with(
        self,
        tract_length: f32,
        tuning: &crate::Tuning,
    ) -> Option<crate::inverse::Shape> {
        let table = vowel_size::Table::of(tuning, tuning.anatomy.tongue_model >= 0.5);
        let held = vowel_size::OwnShapes::NONE.on(table);
        let (a, narrowing) = self.physical_in(tract_length, &held);
        let ((tongue_index, tongue_diameter), narrowing) = (a.tongue?, narrowing?);
        Some(crate::inverse::Shape {
            tongue_index,
            tongue_diameter,
            lips: a.lips.map(|c| c.diameter),
            narrowing: Some((narrowing.index, narrowing.diameter)),
            pharynx: self.physical_root(tract_length, &held).unwrap_or(0.0),
            modes: self.physical_modes(tract_length, &held),
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
