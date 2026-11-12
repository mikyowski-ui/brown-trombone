//! Side tubes that hang off the main tract: the two nasal passages (open
//! at the far end) and the two piriform sinuses (blind pockets beside the
//! larynx).
//!
//! Cell `0` is the port onto the main tube; its area can move (the velum).
//! "Right" inside a branch means away from the main tube. A branch can
//! carry one [`Resonator`]: a cavity behind a narrow neck, which is how a
//! paranasal sinus hangs off a nasal passage.

use super::junction::{Joint, Shunt, Y_MIN, expansion_loss, solve};
use super::radiation::Radiation;
use super::{A_MIN, C, INV_RHO_C, RHO, cell_loss, viscous_resistance};
use crate::math::sqrt;
use core::f32::consts::PI;

/// A Helmholtz resonator as a shunt: a cavity of some volume reached
/// through a short neck. It swallows sound around its own resonance and so
/// carves a dip into whatever passes its mouth.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Resonator {
    /// Flow into the cavity, cm³/s.
    flow: f32,
    /// Extra air held in the cavity, cm³.
    volume: f32,
    /// Coefficients of the implicit step.
    damp: f32,
    gain: f32,
    spring: f32,
    dt: f32,
}

impl Resonator {
    /// `volume` cm³ behind a neck of `area` cm² and `length` cm, with the
    /// given quality factor.
    pub(crate) fn new(volume: f32, area: f32, length: f32, q: f32, dt: f32) -> Self {
        // The air just outside each end of the neck moves with it.
        let effective = length + 1.7 * sqrt(area / PI);
        let inertance = RHO * effective / area;
        let compliance = volume / (RHO * C * C);
        let resistance = sqrt(inertance / compliance) / q;
        let damp = 1.0 / (1.0 + dt * resistance / inertance);
        Self {
            flow: 0.0,
            volume: 0.0,
            damp,
            gain: damp * dt / inertance,
            spring: dt / (inertance * compliance),
            dt,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.flow = 0.0;
        self.volume = 0.0;
    }

    #[inline]
    fn shunt(&self) -> Shunt {
        Shunt {
            g: self.gain,
            h: self.damp * (self.flow - self.spring * self.volume),
        }
    }

    #[inline]
    fn commit(&mut self, p: f32) {
        self.flow = self.damp * (self.flow - self.spring * self.volume) + self.gain * p;
        self.volume += self.dt * self.flow;
    }
}

/// What sits at the far end of a branch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum End {
    /// A blind pocket: everything comes back.
    Closed,
    /// An opening into free air.
    Open,
}

/// A fixed-shape side tube of up to `M` cells.
#[derive(Clone, Debug)]
pub(crate) struct Branch<const M: usize> {
    len: usize,
    r: [f32; M],
    l: [f32; M],
    y: [f32; M],
    inv: [f32; M],
    att: [f32; M],
    visc: [f32; M],
    port0: f32,
    port1: f32,
    port_loss: f32,
    dx: f32,
    end: End,
    end_area: f32,
    rad: Radiation,
    /// A cavity hanging on the node before cell `.0`.
    sinus: Option<(usize, Resonator)>,
}

impl<const M: usize> Branch<M> {
    pub(crate) fn new(end: End) -> Self {
        Self {
            len: 0,
            r: [0.0; M],
            l: [0.0; M],
            y: [0.0; M],
            inv: [0.0; M],
            att: [1.0; M],
            visc: [0.0; M],
            port0: 0.0,
            port1: 0.0,
            port_loss: 1.0,
            dx: 0.0,
            end,
            end_area: 0.0,
            rad: Radiation::default(),
            sinus: None,
        }
    }

    /// Set the shape: `areas[0]` is the port, the rest is fixed. `loss`
    /// scales the wall losses (a nose is far lossier than a smooth pipe of
    /// the same area).
    pub(crate) fn configure(&mut self, areas: &[f32], loss: f32, dx: f32) {
        self.len = areas.len().min(M);
        self.dx = dx;
        self.port_loss = loss;
        for (k, &a) in areas.iter().take(M).enumerate() {
            self.y[k] = a * INV_RHO_C;
            self.inv[k] = 1.0 / a.max(A_MIN);
            self.att[k] = cell_loss(a, dx, loss);
            self.visc[k] = viscous_resistance(self.inv[k], dx);
        }
        if let Some(&a) = areas.first() {
            self.port0 = a;
            self.port1 = a;
        }
        self.end_area = areas.last().copied().unwrap_or(0.0);
    }

    /// Hang a cavity on the node before cell `at`.
    pub(crate) fn set_sinus(&mut self, at: usize, resonator: Resonator) {
        self.sinus = Some((at.clamp(1, M - 1), resonator));
    }

    /// New port area for the coming block.
    pub(crate) fn set_port(&mut self, area: f32) {
        self.port0 = self.port1;
        self.port1 = area;
    }

    pub(crate) fn reset(&mut self) {
        self.r = [0.0; M];
        self.l = [0.0; M];
        self.rad.reset();
        if let Some((_, r)) = &mut self.sinus {
            r.reset();
        }
    }

    #[inline]
    fn port_area(&self, lambda: f32) -> f32 {
        self.port0 + (self.port1 - self.port0) * lambda
    }

    /// The branch as a shunt on a main-tube node: the flow it draws is
    /// `(p − 2c)·y`, with `c` the wave coming back out of the branch.
    #[inline]
    pub(crate) fn shunt(&self, lambda: f32) -> Shunt {
        if self.len == 0 {
            return Shunt::NONE;
        }
        let y = self.port_area(lambda) * INV_RHO_C;
        if y < Y_MIN {
            return Shunt::NONE;
        }
        Shunt {
            g: y,
            h: -2.0 * self.l[0] * y,
        }
    }

    /// Advance one tick, given the pressure at the main-tube node. Returns
    /// the flow radiated from the far end (zero for a closed branch).
    #[inline]
    #[allow(clippy::many_single_char_names)]
    pub(crate) fn tick(&mut self, lambda: f32, node_pressure: f32, dt: f32) -> f32 {
        let n = self.len;
        if n == 0 {
            return 0.0;
        }
        let port = self.port_area(lambda);
        let y0 = port * INV_RHO_C;
        let inv0 = 1.0 / port.max(A_MIN);
        let mut nr = [0.0f32; M];
        let mut nl = [0.0f32; M];

        // Into the port from the main tube, or a wall if the port is shut.
        nr[0] = if y0 < Y_MIN {
            self.l[0]
        } else {
            node_pressure - self.l[0]
        };

        for k in 1..n {
            let (y1, inv1) = if k == 1 {
                (y0, inv0)
            } else {
                (self.y[k - 1], self.inv[k - 1])
            };
            let (k_fwd, k_rev) = expansion_loss(inv1, self.inv[k]);
            let sinus_here = self.sinus.as_ref().filter(|(at, _)| *at == k);
            let shunt = sinus_here.map_or(Shunt::NONE, |(_, r)| r.shunt());
            let s = solve(
                &Joint {
                    y1,
                    y2: self.y[k],
                    shunt,
                    k_fwd,
                    k_rev,
                    r: self.visc[k],
                    pn: 0.0,
                    sealed: false,
                },
                self.r[k - 1],
                self.l[k],
            );
            nl[k - 1] = s.out_left;
            nr[k] = s.out_right;
            if let Some((at, r)) = &mut self.sinus
                && *at == k
            {
                r.commit(s.p1);
            }
        }

        let mut radiated = 0.0;
        let last_y = if n == 1 { y0 } else { self.y[n - 1] };
        match self.end {
            End::Closed => nl[n - 1] = self.r[n - 1],
            End::Open => match self.rad.load(self.end_area, dt) {
                Some(load) => {
                    let s = solve(
                        &Joint {
                            y1: last_y,
                            y2: load.y,
                            shunt: Shunt::NONE,
                            k_fwd: load.k_out,
                            k_rev: 0.0,
                            r: 0.0,
                            pn: 0.0,
                            sealed: false,
                        },
                        self.r[n - 1],
                        load.b,
                    );
                    nl[n - 1] = s.out_left;
                    self.rad.commit(load.b + s.out_right, dt);
                    radiated = s.u;
                }
                None => nl[n - 1] = self.r[n - 1],
            },
        }

        let port_att = cell_loss(port, self.dx, self.port_loss);
        self.r[0] = nr[0] * port_att;
        self.l[0] = nl[0] * port_att;
        for k in 1..n {
            self.r[k] = nr[k] * self.att[k];
            self.l[k] = nl[k] * self.att[k];
        }
        radiated
    }
}
