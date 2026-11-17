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

use super::{Gauge, MU, RHO};
use crate::math::{exp, sqrt};
use core::f32::consts::PI;

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

    /// Body–cover folds set by three laryngeal muscles, each `0‥1`: the
    /// cricothyroid (`ct`, stretches and thins the folds: higher), the
    /// thyroarytenoid (`ta`, the body of the fold itself: shortens and
    /// thickens them, stiffens the body) and the lateral cricoarytenoid
    /// (`lc`, brings them together, shortening them slightly).
    ///
    /// These are the rules of Titze & Story (2002, JASA 112, "Rules for
    /// controlling low-dimensional vocal fold models with muscle
    /// activation"): strain from the muscle balance, length, thickness and
    /// layer depths from strain, fibre stress in mucosa, ligament and
    /// muscle from strain, and from those the masses and stiffnesses of
    /// the cover and body. The tissue constants are the ones in Brad
    /// Story's `LeTalker` 1.22, which implements that paper.
    pub(crate) fn from_muscles(ct: f32, ta: f32, lc: f32) -> Self {
        /// Shear modulus of cover and body, dyn/cm².
        const SHEAR: f32 = 5000.0;
        /// Tissue density, g/cm³.
        const DENSITY: f32 = 1.04;
        /// Torque ratio, elongation gain, adductory strain factor.
        const R: f32 = 3.0;
        const G: f32 = 0.2;
        const H: f32 = 0.2;
        /// Resting length and thickness, cm (male).
        const LENGTH: f32 = 1.6;
        const THICKNESS: f32 = 0.3;
        /// Resting depth of muscle, ligament and mucosa, cm.
        const MUSCLE: f32 = 0.4;
        const LIGAMENT: f32 = 0.2;
        const MUCOSA: f32 = 0.2;
        /// Largest active stress of the muscle, dyn/cm².
        const ACTIVE_STRESS: f32 = 1.05e6;

        let strain = G * (R * ct - ta) - H * lc;
        let length = LENGTH * (1.0 + strain);
        let thick = THICKNESS / (1.0 + 0.8 * strain);
        let body_depth = (ta * MUSCLE + 0.5 * LIGAMENT) / (1.0 + 0.2 * strain);
        let cover_depth = (MUCOSA + 0.5 * LIGAMENT) / (1.0 + 0.2 * strain);
        // Where the cover pivots: higher with an active muscle.
        let node = thick * (1.0 + ta) / 3.0;

        // Passive fibre stress: linear from 50 % shortening, exponential
        // past a knee.
        let passive = |at_rest: f32, scale: f32, rate: f32, knee: f32| {
            let linear = 2.0 * at_rest * (strain + 0.5);
            let over = strain - knee;
            if over > 0.0 {
                linear + scale * (exp(rate * over) - 1.0 - rate * over)
            } else {
                linear
            }
        };
        let mucosa = passive(5000.0, 300_000.0, 4.4, -0.35);
        let ligament = passive(4000.0, 13_930.0, 17.0, 0.0);
        let muscle_passive = passive(10_000.0, 15_000.0, 6.5, -0.05);
        let stretch = strain - 0.4;
        let muscle =
            ta * ACTIVE_STRESS * (1.0 - 1.07 * stretch * stretch).max(0.0) + muscle_passive;
        let body_stress = (0.5 * ligament * LIGAMENT + muscle * MUSCLE) / body_depth;
        let cover_stress = (mucosa * MUCOSA + 0.5 * ligament * LIGAMENT) / cover_depth;

        let share = node / thick;
        let pi2 = PI * PI;
        let shear_cover = 2.0 * SHEAR * length * thick / cover_depth;
        let mb = DENSITY * length * thick * body_depth;
        let kb = 2.0 * SHEAR * length * thick / body_depth
            + pi2 * body_stress * body_depth * thick / length;
        let k1 = shear_cover * share + pi2 * cover_stress * (cover_depth / length) * node;
        let k2 = shear_cover * (1.0 - share)
            + pi2 * cover_stress * (cover_depth / length) * thick * (1.0 - share);
        let cover_mass = DENSITY * length * thick * cover_depth;
        let m1 = cover_mass * share;
        let m2 = cover_mass * (1.0 - share);
        let lever = share * (1.0 - share);
        let kc = (0.5 * SHEAR * (length * cover_depth / thick) / (1.0 / 3.0 - lever) - shear_cover)
            * lever;
        Self {
            m1,
            m2,
            mb,
            k1,
            k2,
            kc: kc.max(0.0),
            kb,
            r1: 0.2 * sqrt(m1 * k1),
            r2: 1.2 * sqrt(m2 * k2),
            rb: 0.2 * sqrt(mb * kb),
            // Contact stiffens the edge threefold, as in the two-mass set.
            c1: 3.0 * k1,
            c2: 3.0 * k2,
            d1: node,
            d2: thick - node,
            length,
            body_locked: false,
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
    /// Share of the coupling between the two cover pieces (`kc`) left:
    /// `1` as the parameters have it, less in creak (see
    /// `Source::creak`). A thick, slack fold couples its lower and upper
    /// margins far more loosely than the standard folds do: the rules of
    /// Titze & Story (2002) give `kc` 0.065 of `k1` at rest (6.9 against
    /// 106 kN/m), Steinecke & Herzel's set 0.31 (25 against 80).
    pub(crate) coupling: f32,
    /// The air is going in (the lungs are drawing): the folds' upper
    /// margin meets the stream first.
    pub(crate) inward: bool,
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
            coupling: 1.0,
            inward: false,
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

    /// Pressure on the lower and upper piece while the air goes in, from
    /// the tract above to the lungs below: the same rule as the air going
    /// out, mirrored (the upper margin is the entrance, the jet leaves
    /// past the narrowest point into the trachea). `u` is negative.
    fn inward_pressures(
        p: &FoldParams,
        (a1, a2): (f32, f32),
        (ps, pe): (f32, f32),
        u: f32,
    ) -> (f32, f32) {
        if a2 <= 0.0 {
            return (ps, 0.0);
        }
        if a1 <= 0.0 {
            return (ps, pe);
        }
        let head = 0.5 * RHO * u * u;
        let slit = -12.0 * MU * p.length * p.length * u;
        let (cube1, cube2) = (a1 * a1 * a1, a2 * a2 * a2);
        let drop2 = slit * 0.5 * p.d2 / cube2;
        let drop1 = slit * (p.d2 / cube2 + 0.5 * p.d1 / cube1);
        let p2 = (pe - head / (a2 * a2) - drop2).max(ps.min(pe));
        let p1 = if a1 <= a2 {
            (pe - head / (a1 * a1) - drop1).max(ps.min(pe))
        } else {
            ps
        };
        (p1, p2)
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
        //
        // In a narrow slit the air also loses pressure to viscosity on its
        // way through (Poiseuille: 12·μ·thickness·u / (length·width³)),
        // and the tract solver charges the glottis with that loss, so the
        // flow `u` it hands over is less than Bernoulli alone would pass.
        // The folds must feel the same loss: each piece sees the pressure
        // at its middle, after the drop along the slit up to there. Without
        // it a nearly shut glottis, where viscosity is most of the drop,
        // felt almost the whole lung pressure pushing it open, and under
        // vowels that load the folds heavily (a close vowel, a pitch near
        // F1, a breathy or a soft voice) the edges stopped meeting.
        let (p1, p2) = if self.inward {
            Self::inward_pressures(p, (a1, a2), (ps, pe), u)
        } else if a1 <= 0.0 {
            (0.0, pe)
        } else if a2 <= 0.0 {
            (ps, pe)
        } else {
            let head = 0.5 * RHO * u * u;
            let slit = 12.0 * MU * p.length * p.length * u;
            let (cube1, cube2) = (a1 * a1 * a1, a2 * a2 * a2);
            let drop1 = slit * 0.5 * p.d1 / cube1;
            let drop2 = slit * (p.d1 / cube1 + 0.5 * p.d2 / cube2);
            let p1 = (ps - head / (a1 * a1) - drop1).max(pe.min(ps));
            let p2 = if a2 <= a1 {
                (ps - head / (a2 * a2) - drop2).max(pe.min(ps))
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
        let qk = q * self.coupling;
        let mut e1 = -q * p.k1 * s1 - qk * p.kc * (self.x1 - self.x2);
        let mut e2 = -q * p.k2 * s2 - qk * p.kc * (self.x2 - self.x1);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, tolerance: f32) -> bool {
        (a - b).abs() <= tolerance * b.abs()
    }

    #[test]
    fn muscle_rules_match_a_hand_calculation() {
        // Cricothyroid 0.25, thyroarytenoid 0.25, lateral cricoarytenoid
        // 0.5: zero strain, so the folds are at their resting 1.6 cm and
        // 0.3 cm, and every rule can be followed with a pocket calculator.
        let p = FoldParams::from_muscles(0.25, 0.25, 0.5);
        assert!(close(p.length, 1.6, 1e-5));
        assert!(close(p.d1 + p.d2, 0.3, 1e-5));
        assert!(close(p.d1, 0.125, 1e-4));
        assert!(close(p.m1, 0.0624, 0.01), "m1 {}", p.m1);
        assert!(close(p.m2, 0.0874, 0.01), "m2 {}", p.m2);
        assert!(close(p.mb, 0.0998, 0.01), "mb {}", p.mb);
        assert!(close(p.k1, 106_000.0, 0.02), "k1 {}", p.k1);
        assert!(close(p.k2, 148_400.0, 0.02), "k2 {}", p.k2);
        assert!(close(p.kb, 193_800.0, 0.02), "kb {}", p.kb);
        assert!(close(p.kc, 6880.0, 0.03), "kc {}", p.kc);
        assert!(!p.body_locked);
    }

    #[test]
    fn muscles_do_what_their_names_say() {
        let rest = FoldParams::from_muscles(0.25, 0.25, 0.5);
        // The cricothyroid stretches: longer, thinner, stiffer cover.
        let stretched = FoldParams::from_muscles(0.75, 0.25, 0.5);
        assert!(stretched.length > rest.length);
        assert!(stretched.d1 + stretched.d2 < rest.d1 + rest.d2);
        assert!(stretched.k1 / stretched.m1 > rest.k1 / rest.m1);
        // The thyroarytenoid bulks and stiffens the body and shortens the
        // fold.
        let bulked = FoldParams::from_muscles(0.25, 0.75, 0.5);
        assert!(bulked.length < rest.length);
        assert!(bulked.mb > rest.mb);
        assert!(bulked.kb > rest.kb);
        // Every setting gives a physical fold.
        for ct in [0.0, 0.5, 1.0] {
            for ta in [0.0, 0.5, 1.0] {
                for lc in [0.0, 0.5, 1.0] {
                    let p = FoldParams::from_muscles(ct, ta, lc);
                    for v in [p.m1, p.m2, p.mb, p.k1, p.k2, p.kb, p.r1, p.r2, p.d1, p.d2] {
                        assert!(v.is_finite() && v > 0.0, "{ct} {ta} {lc}: {p:?}");
                    }
                    assert!(p.kc >= 0.0);
                }
            }
        }
    }

    #[test]
    fn a_larger_larynx_sings_lower() {
        let p = FoldParams::steinecke_herzel();
        let big = p.sized(1.4);
        let ratio = sqrt(big.k1 / big.m1) / sqrt(p.k1 / p.m1);
        assert!(close(ratio, 1.0 / 1.4, 1e-4), "{ratio}");
        // Damping ratio unchanged.
        let zeta = |q: &FoldParams| q.r1 / (2.0 * sqrt(q.k1 * q.m1));
        assert!(close(zeta(&big), zeta(&p), 1e-4));
    }
}
