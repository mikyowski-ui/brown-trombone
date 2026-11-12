//! Self-oscillating vocal folds.
//!
//! The kinematic source prescribes the glottal area. Here nothing is
//! prescribed: each fold is three small masses on springs — a lower and an
//! upper piece of the loose cover, and the muscle body they ride on — and
//! the air pressure between them does the rest. Lung pressure pushes the
//! folds apart; the fast jet between them, and their own elasticity, pull
//! them back; the lower edge leads the upper, so the channel is convergent
//! while opening and divergent while closing, and that asymmetry is what
//! feeds energy into the motion (the mucosal wave).
//!
//! One solver serves two parameter sets:
//!
//! * [`FoldParams::steinecke_herzel`]: the two-mass model of Steinecke &
//!   Herzel (1995, JASA 97) with its published constants. The body is
//!   locked, which makes the three-mass topology exactly their model.
//! * A body–cover set with a free body (Story & Titze 1995), to be driven
//!   by muscle activations.
//!
//! Because the folds feel the pressure on both sides, they react to the
//! tract: voicing weakens and dies as the mouth fills behind a closure,
//! starts late after a voiceless stop, and stops below a threshold lung
//! pressure. None of that is scripted.

use super::{Gauge, RHO};

/// Mechanical description of one fold (the two are assumed alike).
/// Centimetres, grams, seconds.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FoldParams {
    /// Mass of the lower and upper cover pieces and of the body.
    pub m1: f32,
    pub m2: f32,
    pub mb: f32,
    /// Springs: lower cover–body, upper cover–body, between the cover
    /// pieces, body–cartilage.
    pub k1: f32,
    pub k2: f32,
    pub kc: f32,
    pub kb: f32,
    /// Dampers alongside `k1`, `k2`, `kb`.
    pub r1: f32,
    pub r2: f32,
    pub rb: f32,
    /// Extra stiffness while the folds are pressed together.
    pub c1: f32,
    pub c2: f32,
    /// Thickness of the lower and upper pieces.
    pub d1: f32,
    pub d2: f32,
    /// Length of the vibrating edge.
    pub length: f32,
    /// A locked body does not move (the two-mass model).
    pub body_locked: bool,
}

impl FoldParams {
    /// Steinecke & Herzel (1995), Table I, in CGS units.
    pub(crate) const fn steinecke_herzel() -> Self {
        Self {
            m1: 0.125,
            m2: 0.025,
            mb: 0.05,
            k1: 80_000.0,
            k2: 8_000.0,
            kc: 25_000.0,
            kb: 100_000.0,
            r1: 20.0,
            r2: 20.0,
            rb: 100.0,
            c1: 240_000.0,
            c2: 24_000.0,
            d1: 0.25,
            d2: 0.05,
            length: 1.4,
            body_locked: true,
        }
    }

    /// The same folds in a larger or smaller larynx: lengths scale with
    /// `size`, masses with its cube, stiffness with `size` itself, so the
    /// natural pitch goes as `1/size`.
    pub(crate) fn sized(mut self, size: f32) -> Self {
        let s3 = size * size * size;
        self.m1 *= s3;
        self.m2 *= s3;
        self.mb *= s3;
        for k in [
            &mut self.k1,
            &mut self.k2,
            &mut self.kc,
            &mut self.kb,
            &mut self.c1,
            &mut self.c2,
        ] {
            *k *= size;
        }
        // Damping ratio is kept: r ∝ sqrt(k·m).
        let r = size * size;
        self.r1 *= r;
        self.r2 *= r;
        self.rb *= r;
        self.d1 *= size;
        self.d2 *= size;
        self.length *= size;
        self
    }
}

/// The moving state of the folds.
#[derive(Clone, Debug)]
pub(crate) struct Folds {
    params: FoldParams,
    /// Displacement from rest and velocity of each mass (outward positive).
    x1: f32,
    v1: f32,
    x2: f32,
    v2: f32,
    xb: f32,
    vb: f32,
    /// Rest half-width of the glottis: how far apart the folds are held.
    pub(crate) rest: f32,
    /// How fast the cartilages were moving the folds on the last tick.
    rest_speed: f32,
    /// Tension: stiffness is multiplied and mass divided by this, which
    /// moves every natural frequency in proportion.
    pub(crate) tension: f32,
}

impl Folds {
    pub(crate) fn new(params: FoldParams) -> Self {
        Self {
            params,
            x1: 0.0,
            v1: 0.0,
            x2: 0.0,
            v2: 0.0,
            xb: 0.0,
            vb: 0.0,
            rest: 0.0179,
            rest_speed: 0.0,
            tension: 1.0,
        }
    }

    pub(crate) fn set_params(&mut self, params: FoldParams) {
        self.params = params;
    }

    pub(crate) fn reset(&mut self) {
        self.x1 = 0.0;
        self.v1 = 0.0;
        self.x2 = 0.0;
        self.v2 = 0.0;
        self.xb = 0.0;
        self.vb = 0.0;
        self.rest_speed = 0.0;
    }

    /// Bring the folds together or apart: `rest` is the new half-width
    /// the cartilages hold them at. The masses are carried along, and like
    /// anything carried they lag when the carrier speeds up and run on when
    /// it stops; that is the push that starts them vibrating when the
    /// folds are brought together.
    #[inline]
    pub(crate) fn hold_at(&mut self, rest: f32, dt: f32) {
        let speed = (rest - self.rest) / dt;
        let change = speed - self.rest_speed;
        self.v1 -= change;
        self.v2 -= change;
        if !self.params.body_locked {
            self.vb -= change;
        }
        self.rest_speed = speed;
        self.rest = rest;
    }

    /// Areas under the lower and upper cover pieces, cm² (negative while
    /// the folds are pressed together).
    #[inline]
    fn areas(&self) -> (f32, f32) {
        let l2 = 2.0 * self.params.length;
        (l2 * (self.rest + self.x1), l2 * (self.rest + self.x2))
    }

    /// Area open to the air stream, cm².
    #[inline]
    pub(crate) fn area(&self) -> f32 {
        let (a1, a2) = self.areas();
        a1.min(a2).max(0.0)
    }

    /// Displacement of the lower edge, for pitch tracking.
    #[inline]
    pub(crate) fn lower_edge(&self) -> f32 {
        self.x1
    }

    /// Move the folds one tick under the pressures and flow the tract
    /// solver just found. `share` is the part of that flow that passes
    /// between the vibrating edges (the rest leaks behind them).
    #[inline]
    pub(crate) fn step(&mut self, g: &Gauge, share: f32, dt: f32) {
        let p = &self.params;
        let q = self.tension;
        let (a1, a2) = self.areas();
        let (ps, pe, u) = (g.p_sub, g.p_supra, g.u_glottis * share);

        // Pressure on each piece. Upstream of the narrowest point the air
        // has sped up and its pressure has dropped (Bernoulli); past it the
        // jet has left the wall and the pressure is that of the tract.
        let (p1, p2) = if a1 <= 0.0 {
            (0.0, pe)
        } else if a2 <= 0.0 {
            (ps, pe)
        } else {
            let head = 0.5 * RHO * u * u;
            let p1 = (ps - head / (a1 * a1)).max(pe.min(ps));
            let p2 = if a2 <= a1 {
                (ps - head / (a2 * a2)).max(pe.min(ps))
            } else {
                pe
            };
            (p1, p2)
        };
        let f1 = p.length * p.d1 * p1;
        let f2 = p.length * p.d2 * p2;

        // Springs to the body, between the pieces, and against the other
        // fold while in contact.
        let l2 = 2.0 * p.length;
        let s1 = self.x1 - self.xb;
        let s2 = self.x2 - self.xb;
        let mut e1 = -q * p.k1 * s1 - q * p.kc * (self.x1 - self.x2);
        let mut e2 = -q * p.k2 * s2 - q * p.kc * (self.x2 - self.x1);
        if a1 < 0.0 {
            e1 -= q * p.c1 * a1 / l2;
        }
        if a2 < 0.0 {
            e2 -= q * p.c2 * a2 / l2;
        }

        // Semi-implicit step: damping taken at the new velocity.
        let (m1, m2) = (p.m1 / q, p.m2 / q);
        self.v1 = (self.v1 + dt * (f1 + e1 + p.r1 * self.vb) / m1) / (1.0 + dt * p.r1 / m1);
        self.v2 = (self.v2 + dt * (f2 + e2 + p.r2 * self.vb) / m2) / (1.0 + dt * p.r2 / m2);
        self.x1 += dt * self.v1;
        self.x2 += dt * self.v2;

        if p.body_locked {
            self.xb = 0.0;
            self.vb = 0.0;
        } else {
            let mb = p.mb / q;
            let eb = q * p.k1 * s1 + q * p.k2 * s2 - q * p.kb * self.xb
                + p.r1 * (self.v1 - self.vb)
                + p.r2 * (self.v2 - self.vb);
            self.vb = (self.vb + dt * eb / mb) / (1.0 + dt * p.rb / mb);
            self.xb += dt * self.vb;
        }
    }
}
