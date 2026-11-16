//! Stand-ins for the gesture engine's tiers and gestures, with the shapes
//! the gesture engine (`gesture.rs`, being written alongside) will have, so
//! that the instrument's tier panel and the stream's gesture lane can be
//! built and tested before it lands. When it does, [`Tier`],
//! [`TierActivity`] and [`Gesture`] give way to its own, and
//! [`tier_activity`] and [`push_gesture`] to `Trombone::tier_activity` and
//! `Trombone::push_gesture`.
//!
//! What they read and drive here is what the voice already has when it is
//! played by hand: the voicing gate, the pitch, the velum, the tongue body
//! and the constrictions of the tip and the lips.

use crate::tract::{BLADE_START, Constriction, LIP_START, MAX_CONSTRICTIONS, N, TIP_START};
use crate::{SLOT_LIPS, Trombone};

/// One tier of the gesture score: a set of articulators that move together.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tier {
    /// The folds brought together for voice, or apart for breath.
    GlottalAdduction,
    /// The larynx's height and the folds' tension: pitch.
    LarynxAndPitch,
    /// The soft palate: the door to the nose.
    Velum,
    /// The body of the tongue: vowels and the back consonants.
    TongueBody,
    /// The tip and blade of the tongue.
    TongueTip,
    /// The lips and the jaw.
    LipsAndJaw,
}

impl Tier {
    /// Every tier, in score order.
    pub const ALL: [Tier; 6] = [
        Tier::GlottalAdduction,
        Tier::LarynxAndPitch,
        Tier::Velum,
        Tier::TongueBody,
        Tier::TongueTip,
        Tier::LipsAndJaw,
    ];

    /// A short lowercase name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Tier::GlottalAdduction => "glottis",
            Tier::LarynxAndPitch => "larynx",
            Tier::Velum => "velum",
            Tier::TongueBody => "body",
            Tier::TongueTip => "tip",
            Tier::LipsAndJaw => "lips",
        }
    }

    /// Place of the tier in [`Tier::ALL`].
    #[must_use]
    pub fn index(self) -> usize {
        self as usize
    }
}

/// What one tier is doing right now.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TierActivity {
    /// How strongly the tier's gestures drive it, `0‥1`.
    pub activation: f32,
    /// The tier's main variable in the blended target, `0‥1`.
    pub target: f32,
    /// Where the tier is now, on the same scale.
    pub value: f32,
    /// Gestures active on the tier.
    pub gestures: u8,
}

/// A gesture for one tier (stand-in): move the tier's main variable to
/// `target` (`0‥1`, the scale of [`TierActivity::target`]); for the tongue
/// body, tip and lips at `place` along the tract (cell index; not finite:
/// where the tier is now, or its usual place). The stand-in holds a target
/// until the next gesture of the same tier; `duration` (seconds, `0`: hold)
/// is carried for the gesture engine, which times its gestures.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gesture {
    /// The tier it moves.
    pub tier: Tier,
    /// Where it moves it to, `0‥1`.
    pub target: f32,
    /// Where along the tract (tongue and lips), cell index.
    pub place: f32,
    /// How long it lasts, seconds (`0`: until the next).
    pub duration: f32,
}

impl Gesture {
    /// A gesture moving `tier` to `target` at its usual place, held.
    #[must_use]
    pub fn new(tier: Tier, target: f32) -> Self {
        Self {
            tier,
            target,
            place: f32::NAN,
            duration: 0.0,
        }
    }

    /// The same gesture at `place` (cell index).
    #[must_use]
    pub fn at(self, place: f32) -> Self {
        Self { place, ..self }
    }
}

/// Lowest and highest pitch of the larynx tier's scale, Hz (logarithmic).
pub const PITCH_SCALE: (f32, f32) = (50.0, 800.0);
/// Opening, in touch-diameter units, at which a tract tier counts as fully
/// open (`0` on its scale); shut is `1`.
pub const OPEN_DIAMETER: f32 = 2.0;
/// Constriction slot the tongue tip's stand-in gestures use.
pub const TIP_SLOT: usize = crate::SLOT_MAIN;
/// The tip's usual place (the alveolar ridge) and the lips', cell index.
pub const TIP_PLACE: f32 = 36.0;
/// The lips' usual place, cell index.
pub const LIP_PLACE: f32 = 41.0;

/// Pitch on the larynx tier's scale.
#[must_use]
pub fn pitch_unit(hz: f32) -> f32 {
    let (lo, hi) = PITCH_SCALE;
    (crate::math::ln(hz.max(1.0) / lo) / crate::math::ln(hi / lo)).clamp(0.0, 1.0)
}

/// How shut an opening of diameter `d` is, on a tract tier's scale.
#[must_use]
pub fn closure_unit(d: f32) -> f32 {
    (1.0 - d / OPEN_DIAMETER).clamp(0.0, 1.0)
}

fn narrowest(d: &[f32; N], cells: core::ops::Range<usize>) -> f32 {
    d[cells].iter().copied().fold(f32::INFINITY, f32::min)
}

/// What each tier of `voice` is doing, in [`Tier::ALL`] order (stand-in for
/// `Trombone::tier_activity`). Allocation-free and cheap: it reads the
/// voice's controls and its tract.
#[must_use]
pub fn tier_activity(voice: &Trombone) -> [TierActivity; 6] {
    let gate = voice.is_voiced() || voice.always_voice();
    let intensity = voice.intensity().clamp(0.0, 1.0);
    let (target, now) = (voice.target_diameters(), voice.diameters());
    let regions = [
        (BLADE_START, TIP_START),
        (TIP_START, LIP_START),
        (LIP_START, N),
    ];
    let mut held = [0u8; 3];
    #[allow(clippy::cast_precision_loss)] // cell indices
    for c in (0..MAX_CONSTRICTIONS).filter_map(|k| voice.constriction(k)) {
        // A constriction belongs to the region of the cell it presses on.
        if let Some(r) = regions
            .iter()
            .position(|&(a, b)| (a as f32 - 0.5..b as f32 - 0.5).contains(&c.index))
        {
            held[r] = held[r].saturating_add(1);
        }
    }
    let tract = |r: usize| {
        let (a, b) = regions[r];
        TierActivity {
            activation: if r == 0 || held[r] > 0 { 1.0 } else { 0.0 },
            target: closure_unit(narrowest(target, a..b)),
            value: closure_unit(narrowest(now, a..b)),
            // The tongue body always holds a shape.
            gestures: held[r] + u8::from(r == 0),
        }
    };
    let velum = voice.velum_target();
    let moving_velum = (voice.velum() - velum).abs() > 0.02;
    [
        TierActivity {
            activation: if gate { 1.0 } else { intensity },
            target: if gate { 1.0 } else { 0.0 },
            value: intensity,
            gestures: u8::from(gate),
        },
        TierActivity {
            activation: intensity,
            target: pitch_unit(voice.target_frequency()),
            value: pitch_unit(voice.frequency()),
            gestures: u8::from(intensity > 0.05),
        },
        TierActivity {
            activation: if velum > 0.01 || moving_velum {
                1.0
            } else {
                0.0
            },
            target: velum,
            value: voice.velum(),
            gestures: u8::from(velum > 0.01),
        },
        tract(0),
        tract(1),
        tract(2),
    ]
}

/// Carry out a stand-in gesture on `voice` (stand-in for
/// `Trombone::push_gesture`): each tier is driven through the control the
/// voice has for it, at once.
pub fn push_gesture(voice: &mut Trombone, g: &Gesture) {
    if !g.target.is_finite() {
        return;
    }
    let target = g.target.clamp(0.0, 1.0);
    let opening = OPEN_DIAMETER * (1.0 - target);
    let place = |usual: f32| if g.place.is_finite() { g.place } else { usual };
    match g.tier {
        Tier::GlottalAdduction => voice.set_voiced(target >= 0.5),
        Tier::LarynxAndPitch => {
            let (lo, hi) = PITCH_SCALE;
            let hz = lo * crate::math::powf(hi / lo, target);
            voice.set_frequency(hz);
        }
        Tier::Velum => voice.set_velum(target),
        Tier::TongueBody => {
            // Shut is the tongue body at its highest, open at its lowest.
            let (hi, lo) = (
                *crate::TONGUE_DIAMETER.start(),
                *crate::TONGUE_DIAMETER.end(),
            );
            let (index, _) = voice.tongue();
            voice.set_tongue(place(index), lo + (hi - lo) * target);
        }
        Tier::TongueTip => voice.set_constriction(
            TIP_SLOT,
            (target > 0.0).then(|| Constriction::new(place(TIP_PLACE), opening)),
        ),
        Tier::LipsAndJaw => voice.set_constriction(
            SLOT_LIPS,
            (target > 0.0).then(|| Constriction::new(place(LIP_PLACE), opening)),
        ),
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::Phoneme;

    #[test]
    fn tiers_follow_the_hand() {
        let mut v = Trombone::new(48_000.0);
        let quiet = tier_activity(&v);
        assert_eq!(quiet[0].target, 0.0);
        assert_eq!(quiet[2].activation, 0.0);
        assert_eq!(quiet[4].activation, 0.0);
        v.articulate_phoneme(Phoneme::Aa);
        v.set_velum(0.8);
        v.set_constriction(TIP_SLOT, Some(Constriction::new(36.0, 0.0)));
        v.set_frequency(200.0);
        v.note_on();
        let mut buf = std::vec![0.0f32; 4800];
        for _ in 0..10 {
            v.render(&mut buf);
        }
        let t = tier_activity(&v);
        assert_eq!(t[0].target, 1.0);
        assert!(t[0].value > 0.5, "{:?}", t[0]);
        assert!((t[1].target - pitch_unit(200.0)).abs() < 1e-6);
        assert!(
            (t[2].target - 0.8).abs() < 1e-3 && t[2].value > 0.7,
            "{:?}",
            t[2]
        );
        assert_eq!(t[4].gestures, 1);
        assert!(t[4].target > 0.99 && t[4].value > 0.9, "{:?}", t[4]);
        assert_eq!(t[5].activation, 0.0);
    }

    #[test]
    fn gestures_drive_their_tiers() {
        let mut v = Trombone::new(48_000.0);
        for (tier, target) in [
            (Tier::GlottalAdduction, 1.0),
            (Tier::LarynxAndPitch, 0.5),
            (Tier::Velum, 0.6),
            (Tier::TongueBody, 0.7),
            (Tier::TongueTip, 1.0),
            (Tier::LipsAndJaw, 0.75),
        ] {
            push_gesture(&mut v, &Gesture::new(tier, target));
        }
        v.settle();
        let t = tier_activity(&v);
        for (k, want) in [(0, 1.0), (1, 0.5), (2, 0.6), (4, 1.0)] {
            assert!((t[k].target - want).abs() < 0.02, "{k}: {:?}", t[k]);
        }
        // The tongue body and the lips shape the tract their own way: the
        // tier moves, if not to the number asked.
        assert!(t[3].target > 0.3, "{:?}", t[3]);
        assert!(t[5].target > 0.5 && t[5].activation > 0.99, "{:?}", t[5]);
        push_gesture(&mut v, &Gesture::new(Tier::TongueTip, 0.0));
        assert_eq!(v.constriction(TIP_SLOT), None);
    }
}
