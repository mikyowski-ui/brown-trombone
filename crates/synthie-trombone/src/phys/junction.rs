//! The one scattering rule of the physical engine.
//!
//! Every joint in the model — two tube cells, the glottis, the lips, a side
//! branch port — is the same picture:
//!
//! ```text
//!   left cell ──●── series element ── right side
//!               │
//!             shunt
//! ```
//!
//! * `a` is the pressure wave arriving from the left cell (admittance `y1`),
//!   `b` the wave arriving from the right (admittance `y2`).
//! * The **shunt** hangs on the left node and takes a flow `g·p1 + h`: a
//!   yielding wall, a side branch, a sinus, displaced air. `g` is its
//!   instantaneous admittance, `h` everything already decided by its state.
//! * The **series element** drops `k·u·|u| + r·u − pn`: kinetic loss of a
//!   jet, viscous resistance, a noise source.
//!
//! Flow and pressure follow from one quadratic. It is solved in the
//! rationalised, admittance-multiplied form
//!
//! ```text
//!   n = (2·a·y1 − h) − (2·b − pn)·ys          ys = y1 + g
//!   d = ys + y2 + r·ys·y2
//!   w = 2·n / (d + sqrt(d² + 4·k·|n|·ys·y2²))  u = w·y2
//! ```
//!
//! which has no cancellation as `k → 0` (it becomes Kelly–Lochbaum exactly)
//! and no division by an area: a closed cell has `y = 0` and simply carries
//! no flow.

use super::{A_MIN, RHO, RHO_C};
use crate::math::sqrt;

/// Admittance below which a cell counts as sealed.
pub(crate) const Y_MIN: f32 = A_MIN / RHO_C;

/// Flow leaving the left node sideways: `u_s = g·p1 + h`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Shunt {
    pub g: f32,
    pub h: f32,
}

impl Shunt {
    pub(crate) const NONE: Shunt = Shunt { g: 0.0, h: 0.0 };

    /// Add another branch to the same node.
    #[inline]
    pub(crate) fn add(&mut self, g: f32, h: f32) {
        self.g += g;
        self.h += h;
    }
}

/// One joint's parameters for one tick.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Joint {
    /// Admittance `A/ρc` of the left cell.
    pub y1: f32,
    /// Admittance of the right side (a cell, or a radiation load).
    pub y2: f32,
    pub shunt: Shunt,
    /// Kinetic loss coefficient for rightward flow, `Δp = k·u²`.
    pub k_fwd: f32,
    /// Kinetic loss coefficient for leftward flow.
    pub k_rev: f32,
    /// Series resistance.
    pub r: f32,
    /// Series pressure source (pushes flow rightward when positive).
    pub pn: f32,
    /// The series element is shut (a closed glottis).
    pub sealed: bool,
}

/// What a joint sends back out.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Solved {
    /// Wave returned into the left cell.
    pub out_left: f32,
    /// Wave sent into the right side.
    pub out_right: f32,
    /// Flow through the series element, rightward positive.
    pub u: f32,
    /// Pressure at the left node (what the shunt feels).
    pub p1: f32,
}

/// Solve one joint.
#[inline]
#[allow(clippy::many_single_char_names)]
pub(crate) fn solve(j: &Joint, a: f32, b: f32) -> Solved {
    let ys = j.y1 + j.shunt.g;
    if ys < Y_MIN {
        // Nothing on the left to carry flow: both sides see a wall.
        return Solved {
            out_left: a,
            out_right: b,
            u: 0.0,
            p1: a + a,
        };
    }
    let src = 2.0 * a * j.y1 - j.shunt.h;
    if j.sealed || j.y2 < Y_MIN {
        // Sealed valve: the left node only feeds its shunt.
        let p1 = src / ys;
        return Solved {
            out_left: p1 - a,
            out_right: b,
            u: 0.0,
            p1,
        };
    }
    let n = src - (2.0 * b - j.pn) * ys;
    let k = if n >= 0.0 { j.k_fwd } else { j.k_rev };
    let d = ys + j.y2 + j.r * ys * j.y2;
    let w = 2.0 * n / (d + sqrt(d * d + 4.0 * k * n.abs() * ys * j.y2 * j.y2));
    let u = w * j.y2;
    let p1 = (src - u) / ys;
    Solved {
        out_left: p1 - a,
        out_right: b + w,
        u,
        p1,
    }
}

/// Kinetic (Borda–Carnot) loss where a stream leaves a narrow cell for a
/// wider one: `Δp = ρ/2 · u² · (1/A_narrow − 1/A_wide)²`. Returns
/// `(k_fwd, k_rev)` for a joint whose cells have inverse areas `inv1`
/// (left) and `inv2` (right).
#[inline]
pub(crate) fn expansion_loss(inv1: f32, inv2: f32) -> (f32, f32) {
    let d = inv1 - inv2;
    let k = 0.5 * RHO * d * d;
    if d > 0.0 { (k, 0.0) } else { (0.0, k) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(y1: f32, y2: f32) -> Joint {
        Joint {
            y1,
            y2,
            shunt: Shunt::NONE,
            k_fwd: 0.0,
            k_rev: 0.0,
            r: 0.0,
            pn: 0.0,
            sealed: false,
        }
    }

    #[test]
    fn reduces_to_kelly_lochbaum() {
        // Pressure-wave Kelly–Lochbaum with r = (A1 − A2)/(A1 + A2).
        for &(a1, a2) in &[(1.0f32, 1.0f32), (0.3, 4.0), (5.0, 0.2), (2.0, 2.5)] {
            let (a, b) = (3.0f32, -1.25f32);
            let s = solve(&plain(a1 / RHO_C, a2 / RHO_C), a, b);
            let r = (a1 - a2) / (a1 + a2);
            let right = (1.0 + r) * a - r * b;
            let left = r * a + (1.0 - r) * b;
            assert!((s.out_right - right).abs() < 1e-5, "{a1} {a2}");
            assert!((s.out_left - left).abs() < 1e-5, "{a1} {a2}");
        }
    }

    #[test]
    fn sealed_sides_never_divide_by_zero() {
        let s = solve(&plain(1.0 / RHO_C, 0.0), 2.0, 7.0);
        assert_eq!((s.out_left, s.out_right, s.u), (2.0, 7.0, 0.0));
        let s = solve(&plain(0.0, 1.0 / RHO_C), 2.0, 7.0);
        assert_eq!((s.out_left, s.out_right, s.u), (2.0, 7.0, 0.0));
        let s = solve(&plain(0.0, 0.0), 2.0, 7.0);
        assert!(s.out_left.is_finite() && s.out_right.is_finite());
        let mut j = plain(1.0 / RHO_C, 1.0 / RHO_C);
        j.sealed = true;
        let s = solve(&j, 2.0, 7.0);
        assert_eq!((s.out_left, s.out_right, s.u), (2.0, 7.0, 0.0));
    }

    #[test]
    fn kinetic_loss_matches_bernoulli() {
        // A wide-open reservoir at pressure P emptying through a throat:
        // u = a·sqrt(2P/ρ) when the tube impedances are negligible.
        let big = 1.0e4 / RHO_C;
        let throat = 0.1f32;
        let mut j = plain(big, big);
        j.k_fwd = 0.5 * RHO / (throat * throat);
        j.k_rev = j.k_fwd;
        let p = 8000.0f32;
        let s = solve(&j, 0.5 * p, 0.0);
        let ideal = throat * sqrt(2.0 * p / RHO);
        assert!((s.u - ideal).abs() / ideal < 0.02, "{} vs {ideal}", s.u);
        // Reversed pressure, reversed flow.
        let s2 = solve(&j, 0.0, 0.5 * p);
        assert!((s2.u + s.u).abs() / s.u < 1e-4);
    }

    #[test]
    fn tiny_loss_is_continuous_with_none() {
        let mut j = plain(0.02, 0.05);
        let base = solve(&j, 100.0, -40.0);
        j.k_fwd = 1e-12;
        j.k_rev = 1e-12;
        let near = solve(&j, 100.0, -40.0);
        assert!((base.u - near.u).abs() < 1e-4);
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn shunt_takes_its_share_of_the_flow() {
        // Linear case against the direct node equation.
        let (y1, y2, g, h) = (0.03f32, 0.05f32, 0.01f32, 0.2f32);
        let (a, b) = (50.0f32, 10.0f32);
        let mut j = plain(y1, y2);
        j.shunt = Shunt { g, h };
        let s = solve(&j, a, b);
        let p = (2.0 * a * y1 + 2.0 * b * y2 - h) / (y1 + y2 + g);
        assert!((s.p1 - p).abs() < 1e-3, "{} vs {p}", s.p1);
        let into_left_node = (2.0 * a - s.p1) * y1;
        let shunt_flow = g * s.p1 + h;
        assert!((into_left_node - shunt_flow - s.u).abs() < 1e-4);
    }

    #[test]
    fn passive_joint_never_creates_power() {
        // Power in = a²·y1 + b²·y2, power out = out_left²·y1 + out_right²·y2.
        let mut seed = 0x1234_5678u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            #[allow(clippy::cast_precision_loss)]
            let x = (seed >> 8) as f32 / 16_777_216.0;
            x
        };
        for _ in 0..20_000 {
            let y1 = rnd() * 0.2;
            let y2 = rnd() * 0.2;
            let mut j = plain(y1, y2);
            j.k_fwd = rnd() * 10.0;
            j.k_rev = rnd() * 10.0;
            j.r = rnd() * 50.0;
            let a = (rnd() - 0.5) * 2.0e4;
            let b = (rnd() - 0.5) * 2.0e4;
            let s = solve(&j, a, b);
            let p_in = a * a * y1 + b * b * y2;
            let p_out = s.out_left * s.out_left * y1 + s.out_right * s.out_right * y2;
            assert!(p_out <= p_in * 1.0001 + 1e-3, "{p_out} > {p_in}");
        }
    }
}
