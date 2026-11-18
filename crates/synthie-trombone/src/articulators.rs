//! What the articulators are doing, as numbers a face can be drawn from.
//!
//! The sound comes from articulators that really move (a jaw, lips that
//! close, round and protrude, a tongue tip, a tongue body, a velum, a
//! larynx, a glottis), so a character's mouth does not have to be guessed
//! from the audio afterwards: [`Trombone::articulators`](crate::Trombone::articulators)
//! reads them off the voice as an [`ArticulatorFrame`] of ten channels,
//! each `0‥1`, at any moment and at no cost to the audio. A
//! [`VoicePlayer`](crate::stream::VoicePlayer) sends one every
//! [`ArticulatorFrame::PERIOD`] (512 samples at 48 kHz, 10.67 ms) to its
//! [`VoiceStream`](crate::stream::VoiceStream), and `say --export-lipsync`
//! writes them beside the WAV, frame for frame.

/// The articulators at one moment, every channel `0‥1` (see the
/// [module](self)).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ArticulatorFrame {
    /// How far the jaw is lowered: `0` shut (a close vowel, a consonant
    /// made with the teeth together), `1` as for an open /ɑ/.
    pub jaw_open: f32,
    /// How far the lips are brought together: `0` at rest, exactly `1`
    /// while they are sealed (/p b m/).
    pub lip_close: f32,
    /// Lips pushed forward and open, as for /ʃ/ and /ɝ/ (the lip horn's
    /// length, by how open the lips are).
    pub lip_funnel: f32,
    /// Lips pushed forward and drawn narrow, as for /u/ and /w/ (the lip
    /// horn's length, by how narrow the lips are).
    pub lip_pucker: f32,
    /// The lower lip against the upper teeth (/f v/).
    pub lip_press: f32,
    /// The tongue tip or blade raised to the teeth or the ridge behind
    /// them: `1` while it shuts the mouth there (/t d n/), high for
    /// /s z l θ/.
    pub tongue_up: f32,
    /// How far back the tongue body is: `0` for /i/, `1` for /ɑ ɔ/.
    pub tongue_back: f32,
    /// How open the way into the nose is: `0` shut, `1` as for /m n ŋ/.
    pub velum_open: f32,
    /// Where the larynx stands: `0.5` at rest, towards `0` lowered (a
    /// rounded vowel, a mature body), towards `1` raised.
    pub larynx_height: f32,
    /// How far apart the folds are held: near `0` while they vibrate,
    /// about `0.5` in a voiceless consonant, `1` breathing.
    pub glottal_open: f32,
}

impl ArticulatorFrame {
    /// Seconds between the frames a stream sends and `say --export-lipsync`
    /// writes: 512 samples at 48 kHz, the engine's control block.
    pub const PERIOD: f32 = 512.0 / 48_000.0;

    /// The channels' names, in the order of [`to_array`](Self::to_array).
    pub const CHANNELS: [&'static str; 10] = [
        "jaw_open",
        "lip_close",
        "lip_funnel",
        "lip_pucker",
        "lip_press",
        "tongue_up",
        "tongue_back",
        "velum_open",
        "larynx_height",
        "glottal_open",
    ];

    /// The ten channels in the order of [`CHANNELS`](Self::CHANNELS).
    #[must_use]
    pub fn to_array(&self) -> [f32; 10] {
        [
            self.jaw_open,
            self.lip_close,
            self.lip_funnel,
            self.lip_pucker,
            self.lip_press,
            self.tongue_up,
            self.tongue_back,
            self.velum_open,
            self.larynx_height,
            self.glottal_open,
        ]
    }

    /// A frame from its ten channels (see [`to_array`](Self::to_array)).
    #[must_use]
    pub fn from_array(c: [f32; 10]) -> Self {
        Self {
            jaw_open: c[0],
            lip_close: c[1],
            lip_funnel: c[2],
            lip_pucker: c[3],
            lip_press: c[4],
            tongue_up: c[5],
            tongue_back: c[6],
            velum_open: c[7],
            larynx_height: c[8],
            glottal_open: c[9],
        }
    }
}
