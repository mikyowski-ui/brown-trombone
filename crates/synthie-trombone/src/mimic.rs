//! Mimic: make the physical voice copy what a [`Listener`] hears.
//!
//! ```text
//! microphone → Listener → Heard → Mimic::follow → Gesture → Gesture::apply → Trombone
//! ```
//!
//! [`Mimic::follow`] turns one measurement into a [`Gesture`]: whether to
//! voice and at what pitch, the tract shape that gives the heard formants
//! (through the babble map, [`inverse`]), and a gap to hiss through when
//! the sound is a fricative. It is the slow half: finding a shape takes
//! about a millisecond, so run it where the measurements arrive, not in the
//! audio callback. [`Gesture::apply`] is the fast half: a few setters on the
//! voice, safe in the callback.
//!
//! The formants are copied as they are heard, in hertz, on a tract of the
//! voice's own length. A speaker much smaller or larger than the voice
//! makes vowels the voice cannot reach; those are pulled to the nearest
//! vowel it can make, so pick a voice of about the speaker's size.
//!
//! [`Listener`]: crate::listen::Listener

use crate::Phoneme;
use crate::inverse::{self, Shape};
use crate::listen::Heard;
use crate::tract::Constriction;
use crate::voice::{SLOT_MAIN, Trombone};

/// Share of the sound that must be hiss for it to count as a fricative.
const HISS: f32 = 0.5;
/// Brightness, Hz, above which a hiss is taken for /s/; below it, /sh/.
const SIBILANT_SPLIT: f32 = 3800.0;
/// A hiss this much quieter than the loudest heard lately is taken for the
/// faint /f/ (a level ratio).
const FAINT: f32 = 0.25;
/// How fast the memory of the loudest hiss fades, per measurement.
const FADE: f32 = 0.999;
/// Pitches outside this range are not believed.
const PITCH: (f32, f32) = (50.0, 600.0);
/// How far a formant must move, as a ratio, before a new shape is sought.
const MOVED: f32 = 0.01;

/// What the voice should do next, as decided from one measurement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gesture {
    /// Whether the folds should vibrate.
    pub voiced: bool,
    /// Their pitch, Hz (`0` to leave it where it is).
    pub pitch: f32,
    /// The vowel shape to move to, or `None` to stay.
    pub shape: Option<Shape>,
    /// A gap to hiss through, or `None` for no consonant.
    pub hiss: Option<Constriction>,
}

impl Gesture {
    /// Rest: no voice, no consonant, the shape left alone.
    pub const REST: Self = Self {
        voiced: false,
        pitch: 0.0,
        shape: None,
        hiss: None,
    };

    /// Carry the gesture out on `voice` (physical engine). Cheap: safe in
    /// an audio callback.
    pub fn apply(&self, voice: &mut Trombone) {
        if self.pitch > 0.0 {
            voice.set_frequency(self.pitch);
        }
        if let Some(shape) = &self.shape {
            voice.set_shape(shape);
        }
        voice.set_constriction(SLOT_MAIN, self.hiss);
        voice.set_voiced(self.voiced);
    }
}

/// Turns what is heard into what the voice should do (see the
/// [module](self)).
#[derive(Clone, Debug)]
pub struct Mimic {
    tract_length: f32,
    /// Level below which the input counts as silence.
    floor: f32,
    shape: Option<Shape>,
    /// The formants the current shape was solved for.
    solved: [f32; 2],
    /// Loudest hiss heard lately.
    loudest_hiss: f32,
    /// Whether the voice has the modal tongue body (shapes are solved on its
    /// babble map, `inverse::modal`).
    modal: bool,
}

impl Mimic {
    /// A mimic for a voice of the given tract length (see
    /// [`Trombone::set_tract_length`]).
    #[must_use]
    pub fn new(tract_length: f32) -> Self {
        Self {
            tract_length,
            floor: 0.003,
            shape: None,
            solved: [0.0; 2],
            loudest_hiss: 0.0,
            modal: false,
        }
    }

    /// Solve shapes for a voice with the modal tongue body
    /// ([`AnatomyConfig::tongue_model`](crate::AnatomyConfig::tongue_model) 1,
    /// on the modal babble map) or the cosine one.
    pub fn set_modal(&mut self, modal: bool) {
        if modal != self.modal {
            self.modal = modal;
            self.shape = None;
        }
    }

    /// Follow a voice of another size.
    pub fn set_tract_length(&mut self, tract_length: f32) {
        if tract_length.is_finite() && tract_length.to_bits() != self.tract_length.to_bits() {
            self.tract_length = tract_length;
            self.shape = None;
        }
    }

    /// Input level (RMS) below which nothing is copied: set it just above
    /// the room's noise. The default suits a quiet room and a close
    /// microphone.
    pub fn set_floor(&mut self, level: f32) {
        if level.is_finite() {
            self.floor = level.max(0.0);
        }
    }

    /// The shape last sent to the voice, if any.
    #[must_use]
    pub fn shape(&self) -> Option<&Shape> {
        self.shape.as_ref()
    }

    /// Decide what the voice should do about one measurement. Takes about
    /// a millisecond when the vowel has moved (a search of the babble
    /// map); do not call it from an audio callback.
    pub fn follow(&mut self, heard: &Heard) -> Gesture {
        self.loudest_hiss *= FADE;
        if heard.level.is_nan() || heard.level < self.floor {
            return Gesture::REST;
        }
        if !heard.voiced {
            if heard.hiss < HISS {
                return Gesture::REST;
            }
            self.loudest_hiss = self.loudest_hiss.max(heard.level);
            let like = if heard.level < FAINT * self.loudest_hiss {
                Phoneme::F
            } else if heard.brightness >= SIBILANT_SPLIT {
                Phoneme::S
            } else {
                Phoneme::Sh
            };
            return Gesture {
                hiss: like.articulation().constriction,
                ..Gesture::REST
            };
        }

        let [f1, f2, f3] = heard.formants;
        let moved = |now: f32, was: f32| was <= 0.0 || (now / was - 1.0).abs() > MOVED;
        let mut shape = None;
        if f1 > 0.0
            && f2 > f1
            && (self.shape.is_none() || moved(f1, self.solved[0]) || moved(f2, self.solved[1]))
        {
            let f3 = (f3 > f2).then_some(f3);
            let found = if self.modal {
                inverse::modal::solve_for(f1, f2, f3, self.shape.as_ref(), self.tract_length)
            } else {
                inverse::solve_for(f1, f2, f3, self.shape.as_ref(), self.tract_length)
            };
            self.shape = Some(found);
            self.solved = [f1, f2];
            shape = Some(found);
        }
        Gesture {
            voiced: true,
            pitch: if (PITCH.0..=PITCH.1).contains(&heard.pitch) {
                heard.pitch
            } else {
                0.0
            },
            shape,
            hiss: None,
        }
    }
}
