//! Vocal effort states: the same sentence whispered, shouted, spoken by
//! someone exhausted or by someone whose voice shakes.
//!
//! An [`Effort`] is a regime of the body, not a way of pronouncing: the
//! lungs' pressure, how the folds are held (apart for a whisper, pressed
//! for a shout, slack when tired), the tube above them, the pitch the
//! effort lifts or lets sink, the jaw, how fast the articulators move, and
//! a tremor of the pitch and the breath. The words, their timing and their
//! melody stay whoever's they are (the rules', or a neural prosody's), so
//! any sentence can be spoken through any of them.
//!
//! A voice is put into one with
//! [`VoiceSettings::with_effort`](crate::VoiceSettings::with_effort) (it is
//! a number of the voice's tuning, `prosody.effort`, so profiles and
//! streams carry it), a voice that is sounding glides into one with
//! [`Trombone::set_effort`](crate::Trombone::set_effort), and in text a tag
//! switches from there on: `"Wait. [whisper] I heard something. [shout]
//! Run!"` (`[normal]` goes back to the voice's own).

use crate::tuning::Tuning;

/// A vocal effort state (see the [module](self)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Effort {
    /// The voice as it is.
    #[default]
    Normal,
    /// A whisper: the folds held apart behind an open chink so that they
    /// do not meet, the sound made by the air rushing through.
    Whisper,
    /// Shouting or projecting: the lungs push with 14 to 18 cmH2O, the
    /// folds press together, the tube above them narrows, the pitch rises
    /// by four semitones and the jaw opens wider.
    Shout,
    /// Exhausted, tired: the lungs push with 5 to 6 cmH2O, the folds'
    /// cover is slack, the pitch sinks, the articulators are slow, a
    /// breath is taken first and statements end in a long creak.
    Exhausted,
    /// Nervous, close to a sob: pitch and breath tremble at about 7 Hz,
    /// the voice catches as it starts, and the breaths tremble too.
    Nervous,
}

impl Effort {
    /// Every effort state, in the order of their codes.
    pub const ALL: [Effort; 5] = [
        Effort::Normal,
        Effort::Whisper,
        Effort::Shout,
        Effort::Exhausted,
        Effort::Nervous,
    ];

    /// The state's name, as a flag and a tag take it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Effort::Normal => "normal",
            Effort::Whisper => "whisper",
            Effort::Shout => "shout",
            Effort::Exhausted => "exhausted",
            Effort::Nervous => "nervous",
        }
    }

    /// The state of that name: `normal`, `whisper`, `shout` (or
    /// `projected`), `exhausted` (or `tired`), `nervous` (or `sob`).
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "normal" => Effort::Normal,
            "whisper" => Effort::Whisper,
            "shout" | "projected" => Effort::Shout,
            "exhausted" | "tired" => Effort::Exhausted,
            "nervous" | "sob" => Effort::Nervous,
            _ => return None,
        })
    }

    /// The state as the tuning carries it (`ProsodyConfig::effort`): `0`
    /// normal, `1` whisper, `2` shout, `3` exhausted, `4` nervous.
    #[must_use]
    pub const fn code(self) -> f32 {
        self as u8 as f32
    }

    /// The state of a tuning's number (the nearest code).
    #[must_use]
    pub fn from_code(code: f32) -> Self {
        if code >= 3.5 {
            Effort::Nervous
        } else if code >= 2.5 {
            Effort::Exhausted
        } else if code >= 1.5 {
            Effort::Shout
        } else if code >= 0.5 {
            Effort::Whisper
        } else {
            Effort::Normal
        }
    }

    /// The state a tuning asks for.
    #[must_use]
    pub fn of(tuning: &Tuning) -> Self {
        Self::from_code(tuning.prosody.effort)
    }

    /// The body's regime in this state.
    pub(crate) const fn regime(self) -> Regime {
        match self {
            Effort::Normal => Regime::NEUTRAL,
            Effort::Whisper => Regime {
                whisper: 1.0,
                lung: WHISPER_LUNG,
                ..Regime::NEUTRAL
            },
            Effort::Shout => Regime {
                lung: SHOUT_LUNG,
                tenseness: SHOUT_TENSENESS,
                squeeze: SHOUT_SQUEEZE,
                pitch: SHOUT_PITCH,
                jaw: SHOUT_JAW,
                ..Regime::NEUTRAL
            },
            Effort::Exhausted => Regime {
                lung: TIRED_LUNG,
                tenseness: TIRED_TENSENESS,
                slack: TIRED_SLACK,
                pitch: TIRED_PITCH,
                speed: TIRED_SPEED,
                ..Regime::NEUTRAL
            },
            Effort::Nervous => Regime {
                tremor: NERVOUS_TREMOR,
                catch: 1.0,
                pitch: NERVOUS_PITCH,
                ..Regime::NEUTRAL
            },
        }
    }

    /// How the state changes what speech is compiled with: the pace, the
    /// breath before the utterance and the creak at a statement's end.
    pub(crate) fn shape(self, tuning: &mut Tuning, rate: &mut f32) {
        let p = &mut tuning.prosody;
        match self {
            Effort::Normal => {}
            Effort::Whisper => {
                // No voice to creak with.
                p.final_fry = 0.0;
            }
            Effort::Shout => {
                // A shout is held up to its end, on a full breath.
                p.final_fry = 0.0;
                p.breath_intake = p.breath_intake.max(6.0);
                *rate *= 0.94;
            }
            Effort::Exhausted => {
                // The sigh of a breath first, and the voice gives out early.
                p.breath_first = 1.0;
                p.breath_intake = p.breath_intake.max(6.0);
                p.breath_time = p.breath_time.max(0.25);
                p.final_fry = 1.0;
                p.fry_time = p.fry_time.max(0.2);
                *rate *= 0.88;
            }
            Effort::Nervous => {
                p.breath_first = 1.0;
                p.breath_intake = p.breath_intake.max(5.0);
                *rate *= 1.06;
            }
        }
    }
}

impl core::fmt::Display for Effort {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

// The regimes' numbers (measured in `examples/effort.rs`; see the README,
// "Vocal effort").
/// Lungs of a whisper against the voice's own: the open glottis lets the
/// air out, so a whisperer pushes a little more.
const WHISPER_LUNG: f32 = 1.25;
/// Lungs of a shout: 16 cmH2O on the voice's 8.
const SHOUT_LUNG: f32 = 2.0;
/// Tenseness a shout adds (the folds pressed, the body stiffened).
const SHOUT_TENSENESS: f32 = 0.2;
/// Share of the epilaryngeal tube's area a shout takes away.
const SHOUT_SQUEEZE: f32 = 0.4;
/// Semitones a shout lifts the pitch.
const SHOUT_PITCH: f32 = 4.0;
/// The jaw's opening in a shout against the voice's own.
const SHOUT_JAW: f32 = 1.3;
/// Lungs of an exhausted speaker: 6 cmH2O on the voice's 8 (5.4 at a
/// phrase's fullest breath).
const TIRED_LUNG: f32 = 0.75;
const TIRED_TENSENESS: f32 = -0.1;
/// Share of the folds' tension given up (a slack cover).
const TIRED_SLACK: f32 = 0.2;
const TIRED_PITCH: f32 = -1.5;
/// The articulators' speed when exhausted against the voice's own.
const TIRED_SPEED: f32 = 0.8;
/// Depth of a nervous voice's tremor in pitch (a share of it; the lungs'
/// pressure trembles [`TREMOR_LUNG`] times as deep).
const NERVOUS_TREMOR: f32 = 0.05;
const NERVOUS_PITCH: f32 = 1.5;
/// The tremor's rate, Hz (physiological tremor of the voice: 6 to 8 Hz).
pub(crate) const TREMOR_RATE: f32 = 7.0;
/// The lungs' tremor against the pitch's.
pub(crate) const TREMOR_LUNG: f32 = 3.4;
/// The tenseness a whisper holds the folds at: under the 0.12 below which
/// they let go, and near it, where the edges stand a slit apart and not a
/// breath's width (see `phys::source`: the gap goes from 0.03 cm² at 0.12
/// to the breathing glottis's 0.25 at 0).
pub(crate) const WHISPER_HOLD: f32 = 0.11;
/// Seconds a voice takes to glide from one regime into another.
pub(crate) const GLIDE: f32 = 0.12;

/// What a vocal effort does to the body, against the voice as it is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Regime {
    /// The lungs' pressure, times the voice's.
    pub lung: f32,
    /// Added to the tenseness asked for.
    pub tenseness: f32,
    /// How far the voice is a whisper, `0‥1`: the tenseness asked for is
    /// brought to [`WHISPER_HOLD`] by it, under which the folds part and
    /// do not meet.
    pub whisper: f32,
    /// Share of the epilaryngeal tube's area taken away.
    pub squeeze: f32,
    /// Semitones on the pitch asked for.
    pub pitch: f32,
    /// The jaw's opening, times the voice's.
    pub jaw: f32,
    /// Added to the fold slack asked for.
    pub slack: f32,
    /// The articulators' speed, times what is asked for.
    pub speed: f32,
    /// Depth of the tremor of pitch (a share of it) and, deeper, of the
    /// lungs' pressure.
    pub tremor: f32,
    /// How hard and how unevenly the voice catches as it starts, `0‥1`.
    pub catch: f32,
}

impl Regime {
    pub(crate) const NEUTRAL: Self = Self {
        lung: 1.0,
        tenseness: 0.0,
        whisper: 0.0,
        squeeze: 0.0,
        pitch: 0.0,
        jaw: 1.0,
        slack: 0.0,
        speed: 1.0,
        tremor: 0.0,
        catch: 0.0,
    };

    /// `t` of the way from `self` to `to`.
    pub(crate) fn mix(&self, to: &Self, t: f32) -> Self {
        let l = |a: f32, b: f32| a + (b - a) * t;
        Self {
            lung: l(self.lung, to.lung),
            tenseness: l(self.tenseness, to.tenseness),
            whisper: l(self.whisper, to.whisper),
            squeeze: l(self.squeeze, to.squeeze),
            pitch: l(self.pitch, to.pitch),
            jaw: l(self.jaw, to.jaw),
            slack: l(self.slack, to.slack),
            speed: l(self.speed, to.speed),
            tremor: l(self.tremor, to.tremor),
            catch: l(self.catch, to.catch),
        }
    }

    /// A tuning with the regime's larynx laid over it: the tube above the
    /// folds drawn in, and the voice's attack deeper, longer held and less
    /// even.
    pub(crate) fn over(&self, tuning: &Tuning) -> Tuning {
        let mut t = *tuning;
        let a = &mut t.anatomy;
        if self.squeeze > 0.0 {
            a.epilarynx_diameter =
                (a.epilarynx_diameter * crate::math::sqrt(1.0 - self.squeeze.min(0.8))).max(0.35);
        }
        if self.catch > 0.0 {
            a.attack_depth = (a.attack_depth * (1.0 + 0.5 * self.catch)).min(0.06);
            a.attack_hold = (a.attack_hold * (1.0 + 2.0 * self.catch)).min(0.01);
            a.cycle_jitter = (a.cycle_jitter * (1.0 + self.catch)).min(2.0);
        }
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_codes_go_round() {
        for e in Effort::ALL {
            assert_eq!(Effort::from_name(e.name()), Some(e));
            assert_eq!(Effort::from_code(e.code()), e);
        }
        assert_eq!(Effort::from_name("tired"), Some(Effort::Exhausted));
        assert_eq!(Effort::from_name("projected"), Some(Effort::Shout));
        assert_eq!(Effort::from_name("sob"), Some(Effort::Nervous));
        assert_eq!(Effort::from_name("Whisper"), None);
        assert_eq!(Effort::Normal.regime(), Regime::NEUTRAL);
        assert_eq!(Effort::of(&Tuning::DEFAULT), Effort::Normal);
    }

    #[test]
    fn the_neutral_regime_changes_no_tuning() {
        assert_eq!(Regime::NEUTRAL.over(&Tuning::DEFAULT), Tuning::DEFAULT);
        let half = Regime::NEUTRAL.mix(&Effort::Shout.regime(), 0.5);
        assert!((half.lung - 1.5).abs() < 1e-6 && (half.pitch - 2.0).abs() < 1e-6);
    }
}
