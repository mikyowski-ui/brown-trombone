//! A uniform tube whose length changes: the lips' tube pushed forward and
//! the lower pharynx stretched by a lowered larynx.
//!
//! Every cell of the tract is one tick of travel long, so a tube that is
//! not a whole number of cells, or that grows and shrinks, is a pair of
//! delay lines, one per direction, read at a fractional delay. The joints
//! at its two ends are ordinary joints; between them the waves travel for
//! `length` ticks.
//!
//! The joints of the tract are all solved at once from the waves that
//! arrived since the last tick, so nothing can cross a slide in less than
//! one tick: a slide is at least one tick long (a slide of exactly one tick
//! is a cell, bit for bit), and a slide that replaces `n` cells can be made
//! up to `n − 1` ticks shorter.
//!
//! The reading interpolates linearly between neighbouring samples. Between
//! two samples that is a convex mix, so it never makes a wave larger than
//! the samples it is read from: passive at every fraction. At the half
//! sample its loss is `cos(πf/f_tick)`, at the 96 kHz tick of the native
//! size 0.04 dB at 3 kHz, 0.12 dB at 5 kHz and 0.30 dB at 8 kHz per pass
//! (a wave crosses the slide once each way per round trip); a cell's own
//! wall loss at the same frequencies (`cell_loss` and the yielding walls)
//! is larger, and the fraction moves on as the length does, so no
//! frequency sits at the worst of it for long.
//!
//! A delay line whose length grows reads some samples twice, and stretched
//! so a wave would leave with more energy than it brought; one that
//! shrinks skips samples, and loses some. Growing, the samples read are
//! scaled by `1 − dℓ`, `dℓ` the growth in this tick (a Doppler stretch of
//! about 10⁻³ per tick at the fastest an articulator moves, 1.5 cm in 40
//! ms at the native size), so the
//! slide never gives back more than it took in at any speed. The work a
//! moving wall does on the air (the volume it sweeps) is not the delay
//! lines': [`super::Phys`] adds it as a flow at the moving end, the piston.

/// Samples each delay line holds: the longest slide is two fewer ticks.
pub(crate) const SLIDE_CAP: usize = 12;

/// A tube of variable length between two joints (see the module).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Slide {
    /// Waves travelling up (away from the glottis), written at the lower
    /// end, and down, written at the upper end; `at` is the newest.
    up: [f32; SLIDE_CAP],
    down: [f32; SLIDE_CAP],
    at: usize,
    /// Length this tick and the last, ticks.
    length: f32,
    last: f32,
}

impl Slide {
    /// A silent slide one tick long.
    pub(crate) const fn new() -> Self {
        Self {
            up: [0.0; SLIDE_CAP],
            down: [0.0; SLIDE_CAP],
            at: 0,
            length: 1.0,
            last: 1.0,
        }
    }

    /// Silence it; the length is kept.
    pub(crate) fn reset(&mut self) {
        self.up = [0.0; SLIDE_CAP];
        self.down = [0.0; SLIDE_CAP];
        self.last = self.length;
    }

    /// The longest a slide can be, ticks.
    #[allow(clippy::cast_precision_loss)]
    pub(crate) const MAX: f32 = (SLIDE_CAP - 2) as f32;

    /// This tick's length, ticks (`1‥MAX`).
    pub(crate) fn set_length(&mut self, ticks: f32) {
        self.last = self.length;
        self.length = ticks.clamp(1.0, Self::MAX);
    }

    /// Jump to a length, with no motion.
    pub(crate) fn snap_length(&mut self, ticks: f32) {
        self.length = ticks.clamp(1.0, Self::MAX);
        self.last = self.length;
    }

    /// The length this tick, ticks.
    pub(crate) fn length(&self) -> f32 {
        self.length
    }

    /// How much it grew since the last tick, ticks (negative shrinking).
    pub(crate) fn growth(&self) -> f32 {
        self.length - self.last
    }

    /// Sample `k` ticks older than the newest of `line`.
    #[inline]
    fn sample(&self, line: &[f32; SLIDE_CAP], k: usize) -> f32 {
        line[(self.at + SLIDE_CAP - k) % SLIDE_CAP]
    }

    /// Read `line` at the slide's length.
    #[inline]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn read(&self, line: &[f32; SLIDE_CAP]) -> f32 {
        // The newest sample was sent in one tick ago.
        let back = self.length - 1.0;
        let k = back as usize;
        #[allow(clippy::cast_precision_loss)]
        let f = back - k as f32;
        let x0 = self.sample(line, k);
        let x = if f > 0.0 {
            x0 + (self.sample(line, k + 1) - x0) * f
        } else {
            x0
        };
        let grew = self.length - self.last;
        if grew > 0.0 { x * (1.0 - grew) } else { x }
    }

    /// The waves arriving at its ends this tick: `(at the upper end,
    /// travelling up; at the lower end, travelling down)`.
    #[inline]
    pub(crate) fn arrivals(&self) -> (f32, f32) {
        (self.read(&self.up), self.read(&self.down))
    }

    /// Take the waves sent into it this tick at its lower end (travelling
    /// up) and its upper end (travelling down). `att` is a cell's loss per
    /// tick of travel; the slide's is that over its length.
    #[inline]
    pub(crate) fn push(&mut self, up: f32, down: f32, att: f32) {
        // `att` over the first tick exactly (a slide one tick long is a
        // cell), and to first order over the rest.
        let loss = att * (1.0 - (self.length - 1.0) * (1.0 - att));
        self.at = (self.at + 1) % SLIDE_CAP;
        self.up[self.at] = up * loss;
        self.down[self.at] = down * loss;
    }

    /// Pressure, dyn/cm², at the share `x` (`0` the lower end, `1` the
    /// upper) of its length, as the waves in it now make it.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub(crate) fn pressure_at(&self, x: f32) -> f32 {
        // A wave sent up `t` ticks ago is `t` ticks up the slide; one sent
        // down `t` ticks ago is `t` ticks down from its upper end.
        let at = |line: &[f32; SLIDE_CAP], t: f32| {
            let t = t.clamp(0.0, self.length - 1.0);
            let k = t as usize;
            #[allow(clippy::cast_precision_loss)]
            let f = t - k as f32;
            let x0 = self.sample(line, k);
            x0 + (self.sample(line, (k + 1).min(SLIDE_CAP - 1)) - x0) * f
        };
        let x = x.clamp(0.0, 1.0);
        at(&self.up, x * self.length - 1.0) + at(&self.down, (1.0 - x) * self.length - 1.0)
    }

    /// Acoustic energy held in it, in the units of `Phys::energy` for a
    /// tube of `area` cm²: each tick of its length holds one sample of each
    /// line.
    #[cfg(test)]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub(crate) fn energy(&self, area: f32) -> f32 {
        let whole = self.length as usize;
        #[allow(clippy::cast_precision_loss)]
        let part = self.length - whole as f32;
        let mut e = 0.0;
        for k in 0..=whole.min(SLIDE_CAP - 1) {
            let w = if k < whole { 1.0 } else { part };
            let (u, d) = (self.sample(&self.up, k), self.sample(&self.down, k));
            e += w * (u * u + d * d);
        }
        area * e
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
mod tests {
    use super::*;

    extern crate std;
    use std::vec::Vec;

    /// Feed `x` in at the lower end of a slide of `length` ticks; what
    /// arrives at the upper end, tick by tick.
    fn through(length: impl Fn(usize) -> f32, x: &[f32]) -> Vec<f32> {
        let mut s = Slide::new();
        s.snap_length(length(0));
        let mut out = Vec::new();
        for (n, &v) in x.iter().enumerate() {
            s.set_length(length(n));
            let (up, _) = s.arrivals();
            out.push(up);
            s.push(v, 0.0, 1.0);
        }
        out
    }

    #[test]
    fn whole_lengths_are_exact_delays() {
        let x: Vec<f32> = (0..64)
            .map(|n| ((n * 37 % 11) as f32 - 5.0) * 0.37)
            .collect();
        for d in 1..=(Slide::MAX as usize) {
            let y = through(|_| d as f32, &x);
            for n in 0..x.len() {
                let want = if n >= d { x[n - d] } else { 0.0 };
                assert_eq!(y[n].to_bits(), want.to_bits(), "length {d}, tick {n}");
            }
        }
    }

    #[test]
    fn one_tick_is_a_cell() {
        // A cell: the wave sent in this tick, times its loss, arrives at the
        // next joint on the next tick.
        let mut s = Slide::new();
        let att = 0.998_73_f32;
        let mut cell = 0.0f32;
        for n in 0..50 {
            let v = (n as f32 * 0.7).sin() * 1234.5;
            let (up, _) = s.arrivals();
            assert_eq!(up.to_bits(), cell.to_bits(), "tick {n}");
            s.set_length(1.0);
            s.push(v, 0.0, att);
            cell = v * att;
        }
    }

    #[test]
    fn half_sample_reading_is_passive_and_costs_what_theory_says() {
        // A sinusoid through a slide 2.5 ticks long comes out at
        // cos(π f / f_tick) of its amplitude.
        let tick = 96_000.0f32;
        for hz in [3000.0f32, 5000.0, 8000.0] {
            let x: Vec<f32> = (0..4000)
                .map(|n| (2.0 * core::f32::consts::PI * hz * n as f32 / tick).sin())
                .collect();
            let y = through(|_| 2.5, &x);
            let rms = |v: &[f32]| (v.iter().map(|a| a * a).sum::<f32>() / v.len() as f32).sqrt();
            let gain = rms(&y[100..]) / rms(&x[100..]);
            let theory = (core::f32::consts::PI * hz / tick).cos();
            std::println!(
                "{hz} Hz: {:.3} dB (theory {:.3})",
                20.0 * gain.log10(),
                20.0 * theory.log10()
            );
            assert!((gain - theory).abs() < 2e-3, "{hz}: {gain} vs {theory}");
            assert!(gain <= 1.0);
        }
    }

    #[test]
    fn a_moving_slide_never_gives_back_more_than_it_took() {
        // Noise through a slide swept between one and eight ticks and back
        // at speeds from five to five hundred times the quickest
        // articulator's (10⁻³ ticks per tick): what leaves never
        // carries more energy than what went in.
        let mut seed = 99u32;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed >> 8) as f32 / 16_777_216.0 - 0.5
        };
        for &period in &[40.0f32, 400.0, 4000.0] {
            let x: Vec<f32> = (0..20_000).map(|_| rnd()).collect();
            let length = |n: usize| 4.5 + 3.5 * (n as f32 * core::f32::consts::TAU / period).sin();
            let y = through(length, &x);
            let e_in: f32 = x.iter().map(|v| v * v).sum();
            let e_out: f32 = y.iter().map(|v| v * v).sum();
            std::println!("period {period} ticks: out/in {:.5}", e_out / e_in);
            assert!(e_out <= e_in, "{e_out} > {e_in} at period {period}");
        }
    }
}
