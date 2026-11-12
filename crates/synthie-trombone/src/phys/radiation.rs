//! Sound leaving an opening (lips, a nostril) into open air.
//!
//! The classic engine reflects a fixed 85 % at the lips at every frequency.
//! A real opening behaves like a piston in a baffle: low frequencies are
//! reflected almost completely, high ones escape, and the air just outside
//! the opening adds a little length to the tube. Flanagan's network for
//! that is a resistance in parallel with an inertance,
//!
//! ```text
//!   R = 128·ρc / (9π²·A)      L = 8·ρ / (3π·sqrt(π·A))
//! ```
//!
//! Both scale with the opening, so the state kept here is the inertance's
//! momentum `φ = L·u_L`, which stays meaningful while the lips move and
//! falls to nothing as they shut.

use super::junction::Y_MIN;
use super::{RHO, RHO_C};
use crate::math::sqrt;
use core::f32::consts::PI;

/// Radiation load state for one opening.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Radiation {
    phi: f32,
}

/// The load as a joint's right side for this tick.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Load {
    /// Admittance presented to the tube.
    pub y: f32,
    /// The equivalent incoming wave.
    pub b: f32,
    /// Kinetic loss of the jet leaving the opening, `Δp = k·u²`.
    pub k_out: f32,
}

impl Radiation {
    pub(crate) fn reset(&mut self) {
        self.phi = 0.0;
    }

    /// The load an opening of `area` presents this tick, or `None` if shut.
    #[inline]
    pub(crate) fn load(&mut self, area: f32, dt: f32) -> Option<Load> {
        let g = 9.0 * PI * PI / (128.0 * RHO_C) * area;
        let inv_l = 3.0 * PI / (8.0 * RHO) * sqrt(PI * area.max(0.0));
        let y = g + dt * inv_l;
        if y < Y_MIN {
            self.phi = 0.0;
            return None;
        }
        // Backward Euler on the inertance: u_L = (φ + dt·p)/L.
        let q = inv_l * self.phi;
        Some(Load {
            y,
            b: -0.5 * q / y,
            k_out: 0.5 * RHO / (area * area),
        })
    }

    /// Advance the state with the pressure just outside the tube end.
    #[inline]
    pub(crate) fn commit(&mut self, p: f32, dt: f32) {
        self.phi += dt * p;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shut_opening_presents_no_load() {
        let mut r = Radiation::default();
        r.commit(100.0, 1e-5);
        assert!(r.load(0.0, 1e-5).is_none());
        // And forgets its momentum.
        let l = r.load(1.0, 1e-5).unwrap();
        assert!(l.b.abs() < 1e-12);
    }

    #[test]
    fn steady_flow_meets_no_resistance() {
        // At DC the inertance is a short: constant flow, pressure → 0.
        let mut r = Radiation::default();
        let dt = 1.0 / 96_000.0;
        let u = 200.0f32;
        let mut p = 0.0;
        for _ in 0..20_000 {
            let l = r.load(1.5, dt).unwrap();
            // p = (u − q)/y with q = −2·b·y.
            p = (u + 2.0 * l.b * l.y) / l.y;
            r.commit(p, dt);
        }
        assert!(p.abs() < 1.0, "{p}");
    }
}
