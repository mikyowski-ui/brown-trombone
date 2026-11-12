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
//!
//! Rounded lips are also pushed forward, which lengthens the tube by a
//! short, narrow piece in front of the opening (see
//! [`protrusion`](super::protrusion)). A tube that short and narrow is an
//! inertance, `ρ·ℓ/A`, in series with the load: [`Radiation::load_through`]
//! takes it, and the flow through it is kept as a second state.

use super::junction::Y_MIN;
use super::{RHO, RHO_C};
use crate::math::sqrt;
use core::f32::consts::PI;

/// Radiation load state for one opening.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Radiation {
    phi: f32,
    /// Flow through the series inertance last tick, cm³/s.
    u: f32,
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
    /// The radiation network alone (without the series inertance): its
    /// admittance and the flow already decided by its state.
    y_rad: f32,
    q_rad: f32,
    /// The series inertance, g/cm⁴.
    series: f32,
}

impl Radiation {
    pub(crate) fn reset(&mut self) {
        self.phi = 0.0;
        self.u = 0.0;
    }

    /// The load an opening of `area` presents this tick, or `None` if shut.
    #[inline]
    pub(crate) fn load(&mut self, area: f32, dt: f32) -> Option<Load> {
        self.load_through(area, 0.0, dt)
    }

    /// [`load`](Self::load) behind a series inertance `series`, g/cm⁴ (a
    /// short tube in front of the opening).
    #[inline]
    pub(crate) fn load_through(&mut self, area: f32, series: f32, dt: f32) -> Option<Load> {
        let g = 9.0 * PI * PI / (128.0 * RHO_C) * area;
        let inv_l = 3.0 * PI / (8.0 * RHO) * sqrt(PI * area.max(0.0));
        let y_rad = g + dt * inv_l;
        if y_rad < Y_MIN {
            self.reset();
            return None;
        }
        // Backward Euler on the inertance: u_L = (φ + dt·p)/L.
        let q_rad = inv_l * self.phi;
        // And on the series one: u = u_prev + dt·(p − p_rad)/L_s, with
        // u = y_rad·p_rad + q_rad, gives u = y·p + q.
        let (y, q) = if series > 0.0 {
            let k = y_rad * series / dt;
            (y_rad / (1.0 + k), (k * self.u + q_rad) / (1.0 + k))
        } else {
            (y_rad, q_rad)
        };
        Some(Load {
            y,
            b: -0.5 * q / y,
            k_out: 0.5 * RHO / (area * area),
            y_rad,
            q_rad,
            series,
        })
    }

    /// Advance the state with the pressure just outside the tube end.
    #[inline]
    pub(crate) fn commit(&mut self, p: f32, dt: f32) {
        self.phi += dt * p;
    }

    /// Advance the state of a [`load_through`](Self::load_through) load
    /// with the pressure `p` just outside the tube end and the flow `u`
    /// into the load.
    #[inline]
    pub(crate) fn commit_through(&mut self, load: &Load, p: f32, u: f32, dt: f32) {
        // The radiation network sees what the series inertance leaves.
        let p_rad = if load.series > 0.0 {
            (u - load.q_rad) / load.y_rad
        } else {
            p
        };
        self.phi += dt * p_rad;
        self.u = u;
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

    /// Drive a load by the flow `flow(n)` at tick `n` (a flow source) for
    /// `ticks` ticks; returns the pressure across it at each.
    fn drive(
        rad: &mut Radiation,
        series: f32,
        dt: f32,
        ticks: u16,
        flow: impl Fn(u16) -> f32,
    ) -> Vec<f32> {
        let mut out = Vec::new();
        for n in 0..ticks {
            let load = rad.load_through(0.5, series, dt).unwrap();
            // u = y·p + q, with q = −2·b·y.
            let u = flow(n);
            let p = (u + 2.0 * load.b * load.y) / load.y;
            rad.commit_through(&load, p, u, dt);
            out.push(p);
        }
        out
    }

    #[test]
    fn protrusion_adds_its_inertance() {
        // A slowly rising flow: the pressure across the load is (almost)
        // L·du/dt, and the series inertance adds its own L to the
        // radiation's.
        let dt = 1.0 / 96_000.0;
        let series = RHO * 0.7 / 0.5;
        let slope = 1.0e5f32;
        let ramp = |n: u16| slope * dt * f32::from(n);
        let plain = drive(&mut Radiation::default(), 0.0, dt, 2000, ramp);
        let pushed = drive(&mut Radiation::default(), series, dt, 2000, ramp);
        let l_rad = 8.0 * RHO / (3.0 * PI * sqrt(PI * 0.5));
        let (p0, p1) = (plain[1999], pushed[1999]);
        assert!((p0 / (l_rad * slope) - 1.0).abs() < 0.05, "{p0}");
        assert!(
            (p1 / ((l_rad + series) * slope) - 1.0).abs() < 0.05,
            "{p1} vs {}",
            (l_rad + series) * slope
        );
        // And it holds no pressure against steady flow.
        let steady = drive(&mut Radiation::default(), series, dt, 20_000, |_| 200.0);
        assert!(steady[19_999].abs() < 1.0, "{}", steady[19_999]);
    }
}
