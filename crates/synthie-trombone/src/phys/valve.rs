//! The ventricular valve (close-out of waves 3 and 4, c3-valve): the false
//! folds just above the true folds, and the aryepiglottic folds that close
//! over them, as a valve of the tube above the folds with its own state
//! and its own speed.
//!
//! In a cough people shut the false folds and the aryepiglottic folds over
//! the true folds while the lungs build their pressure (the compressive
//! phase), then throw the glottis and the supraglottic larynx wide open
//! (the expulsive phase); a throat clearing and an "ahem" release the air
//! through the narrowed supraglottic larynx, the aryepiglottic folds
//! fluttering in it at some 50 to 100 Hz. The articulation model's
//! epilaryngeal tube moves with the tongue body, far too slowly for that:
//! the valve acts on the physical tract's areas tick by tick, after the
//! tract's own shape, and the flow through it, its jet's loss and its
//! noise are those of any narrowing (`junction::solve`, `noise::jet_noise`).
//!
//! Where it sits: the true folds are the joint below cell 0, the
//! ventricle between them and the false folds some 3 to 5 mm long, and the
//! false folds 0.4 to 0.7 cm above the glottis, the lower end of the
//! epilaryngeal tube (cells 0 to 3, 1.46 cm at the native size, whose top is
//! the aryepiglottic folds' rim). Cell 1 ([`VALVE_CELL`], 0.36 to 0.73 cm
//! above the glottis) is where it closes; its flutter is a modulation of
//! the same cell's area (one valve stands for both folds). Opened wide, the
//! whole tube and its flare (cells 0 to 7, below the larynx slide) widen
//! to the valve's open area.
//!
//! The valve's position is one number, `−1‥1`: `0` at rest (the tract as
//! the articulation model has it, byte for byte: nothing here is run),
//! `1` shut, between them the share of cell 1's area it takes away (shut
//! from 0.95 on), and below `0` the share of the way from the tube's own
//! area to the valve's open area (`AnatomyConfig::valve_open`) that every
//! cell of the tube and the flare is widened by. It follows its target with
//! a first-order lag, slowly towards closure (`ArticulationConfig::valve_closing`)
//! and fast away from it (`ArticulationConfig::valve_opening`): held shut by
//! muscle it holds against any pressure below it; let go, the pressure
//! behind it throws it open in a few milliseconds.

use core::f32::consts::PI;

/// The tract cell the valve closes: the false folds, 0.36 to 0.73 cm above
/// the true folds at the native size.
pub(crate) const VALVE_CELL: usize = 1;
/// Cells (from the glottis) the valve widens when it is thrown open: the
/// epilaryngeal tube and its flare, up to the larynx slide's first cell.
pub(crate) const WIDENS: usize = 8;
/// Position from which the valve is shut: its lag reaches it in three time
/// constants and the last of the cell's area goes with it.
const SHUT_AT: f32 = 0.95;
/// Pressure across the valve, dyn/cm² (5 cmH2O), from which it is thrown
/// open at its opening speed; with less it opens only as much faster than
/// it closes as the pressure says, and with none at its closing speed (the
/// muscle letting go): let go at once with half a cmH2O behind it at the
/// end of a throat clearing, it puffed (a click within 50 dB of the
/// loudest 10 ms, 8.5 times the RMS of its 10 ms).
const THROWN_OPEN: f32 = 5.0 * super::CM_H2O;

/// The valve's state.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Valve {
    /// Where the valve is going (see the module's docs), `−1‥1`.
    pub(crate) target: f32,
    /// How deep the aryepiglottic flutter is to be, `0‥1` of the valve's
    /// open area taken away at the bottom of each cycle.
    pub(crate) flutter_target: f32,
    /// Where it is, and the flutter's depth now.
    x: f32,
    depth: f32,
    /// The flutter's phase, cycles.
    phase: f32,
    /// Seconds to shut (from rest), seconds from shut to 90 % open, the
    /// flutter's rate (Hz) and the open area (cm² at the native size).
    closing: f32,
    opening: f32,
    rate: f32,
    open_area: f32,
    /// The same as coefficients for one tick of `dt` at size `len`.
    k_close: f32,
    k_open: f32,
    step: f32,
    open: f32,
}

impl Valve {
    pub(crate) fn new() -> Self {
        let t = crate::Tuning::DEFAULT;
        let mut v = Self {
            target: 0.0,
            flutter_target: 0.0,
            x: 0.0,
            depth: 0.0,
            phase: 0.0,
            closing: t.articulation.valve_closing,
            opening: t.articulation.valve_opening,
            rate: t.anatomy.flutter_rate,
            open_area: t.anatomy.valve_open,
            k_close: 0.0,
            k_open: 0.0,
            step: 0.0,
            open: 0.0,
        };
        v.set_rate(1.0 / (2.0 * crate::INTERNAL_RATE), 1.0);
        v
    }

    /// Take the valve's numbers from a voice's tuning (call `set_rate`
    /// after).
    pub(crate) fn set_tuning(&mut self, tuning: &crate::Tuning) {
        self.closing = tuning.articulation.valve_closing.max(1e-4);
        self.opening = tuning.articulation.valve_opening.max(1e-4);
        self.rate = tuning.anatomy.flutter_rate.max(0.0);
        self.open_area = tuning.anatomy.valve_open.max(0.0);
    }

    /// Set the tick `dt` and the speaker's size `len`.
    pub(crate) fn set_rate(&mut self, dt: f32, len: f32) {
        // Shut in `closing` from rest: three time constants (95 %).
        self.k_close = (3.0 * dt / self.closing).min(1.0);
        // 10 to 90 % of the way in `opening`: 2.2 time constants.
        self.k_open = (2.2 * dt / self.opening).min(1.0);
        self.step = self.rate * dt;
        self.open = self.open_area * len * len;
    }

    /// Whether the valve does anything this tick: off its rest, going
    /// somewhere, or fluttering.
    #[inline]
    pub(crate) fn engaged(&self) -> bool {
        self.x != 0.0 || self.target != 0.0 || self.depth != 0.0 || self.flutter_target != 0.0
    }

    /// Move the valve one tick, `across` (dyn/cm²) the pressure below it
    /// over that above it.
    #[inline]
    pub(crate) fn advance(&mut self, across: f32) {
        let k = if self.target > self.x {
            self.k_close
        } else {
            let thrown = (across / THROWN_OPEN).clamp(0.0, 1.0);
            self.k_close + (self.k_open - self.k_close) * thrown
        };
        self.x += (self.target - self.x) * k;
        // The flutter grows and dies away over the valve's closing time
        // (stopped at once, the opening it leaves jumps open: a click).
        self.depth += (self.flutter_target - self.depth) * self.k_close;
        // Back at rest: exactly so, and nothing runs any more.
        if self.target == 0.0 && self.x.abs() < 1e-5 {
            self.x = 0.0;
        }
        if self.flutter_target == 0.0 && self.depth < 1e-5 {
            self.depth = 0.0;
        }
        if self.depth > 0.0 {
            self.phase += self.step;
            if self.phase >= 1.0 {
                self.phase -= 1.0;
            }
        } else {
            self.phase = 0.0;
        }
    }

    /// The area of cell `i` with the valve where it is, for the area `a`
    /// (cm²) the tract gives it.
    #[inline]
    pub(crate) fn area(&self, i: usize, a: f32) -> f32 {
        if i >= WIDENS {
            return a;
        }
        let mut a = a;
        if self.x < 0.0 && self.open > a {
            a += (self.open - a) * (-self.x).min(1.0);
        }
        if i == VALVE_CELL {
            if self.x > 0.0 {
                a *= 1.0 - (self.x / SHUT_AT).min(1.0);
            }
            if self.depth > 0.0 {
                // The aryepiglottic folds flutter across the stream: at the
                // bottom of each cycle they take `depth` of the opening away.
                let swing = 0.5 - 0.5 * crate::math::cos(2.0 * PI * self.phase);
                a *= 1.0 - self.depth.min(1.0) * swing;
            }
        }
        a
    }

    /// Where it is, `−1‥1`.
    #[cfg(test)]
    pub(crate) fn position(&self) -> f32 {
        self.x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 96_000.0;

    /// Ticks until `done`, at most a second's.
    fn ticks(v: &mut Valve, across: f32, done: impl Fn(&Valve) -> bool) -> usize {
        (0..96_000)
            .find(|_| {
                v.advance(across);
                done(v)
            })
            .unwrap_or(96_000)
    }

    #[test]
    fn it_shuts_in_its_closing_time_and_is_thrown_open_in_its_opening_time() {
        let mut v = Valve::new();
        v.set_rate(DT, 1.0);
        assert!(!v.engaged());
        v.target = 1.0;
        let shut = ticks(&mut v, 0.0, |v| v.area(VALVE_CELL, 0.45) == 0.0);
        #[allow(clippy::cast_precision_loss)]
        let t = shut as f32 * DT;
        let want = crate::Tuning::DEFAULT.articulation.valve_closing;
        assert!((0.9 * want..=1.1 * want).contains(&t), "shut in {t} s");
        // The cells above it are the tract's.
        assert_eq!(v.area(VALVE_CELL + 1, 0.45), 0.45);
        // Let go with 40 cmH2O behind it: 90 % open in the opening time.
        v.target = 0.0;
        let open = ticks(&mut v, 40.0 * super::super::CM_H2O, |v| {
            v.area(VALVE_CELL, 0.45) >= 0.9 * 0.45
        });
        #[allow(clippy::cast_precision_loss)]
        let t = open as f32 * DT;
        let want = crate::Tuning::DEFAULT.articulation.valve_opening;
        assert!(t <= 1.1 * want, "open in {t} s");
        // And at rest again exactly, and still.
        ticks(&mut v, 0.0, |v| !v.engaged());
        assert!(!v.engaged());
        assert_eq!(v.position(), 0.0);
        assert_eq!(v.area(0, 0.45), 0.45);
    }

    #[test]
    fn thrown_wide_it_widens_the_tube_and_its_flare_to_the_open_area() {
        let mut v = Valve::new();
        v.set_rate(DT, 1.0);
        v.target = -1.0;
        ticks(&mut v, 40.0 * super::super::CM_H2O, |v| {
            v.position() < -0.999
        });
        let open = crate::Tuning::DEFAULT.anatomy.valve_open;
        for i in 0..WIDENS {
            assert!((v.area(i, 0.45) - open).abs() < 0.01, "cell {i}");
        }
        // Not wider than a cell already is, nor past the flare.
        assert_eq!(v.area(3, 3.0), 3.0);
        assert_eq!(v.area(WIDENS, 0.45), 0.45);
    }

    #[test]
    fn the_flutter_swings_the_valve_at_its_rate() {
        let mut v = Valve::new();
        v.set_rate(DT, 1.0);
        v.flutter_target = 1.0;
        ticks(&mut v, 0.0, |v| v.depth > 0.999);
        let areas: Vec<f32> = (0..96_000)
            .map(|_| {
                v.advance(0.0);
                v.area(VALVE_CELL, 0.45)
            })
            .collect();
        let troughs = (1..areas.len() - 1)
            .filter(|&k| areas[k] < areas[k - 1] && areas[k] <= areas[k + 1])
            .count();
        #[allow(clippy::cast_precision_loss)]
        let hz = troughs as f32;
        let want = crate::Tuning::DEFAULT.anatomy.flutter_rate;
        assert!((hz - want).abs() <= 1.0, "{hz} Hz");
        let lo = areas.iter().copied().fold(f32::MAX, f32::min);
        assert!(lo < 0.01, "{lo}");
    }
}
