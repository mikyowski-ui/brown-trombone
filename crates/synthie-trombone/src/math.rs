//! Float backend: `std` intrinsics or `libm`.

#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!("synthie-trombone needs either the `std` or the `libm` feature");

#[cfg(feature = "std")]
mod imp {
    #[inline]
    pub fn sin(x: f32) -> f32 {
        x.sin()
    }
    #[inline]
    pub fn cos(x: f32) -> f32 {
        x.cos()
    }
    #[inline]
    pub fn exp(x: f32) -> f32 {
        x.exp()
    }
    #[inline]
    pub fn ln(x: f32) -> f32 {
        x.ln()
    }
    #[inline]
    pub fn powf(x: f32, y: f32) -> f32 {
        x.powf(y)
    }
    #[inline]
    pub fn sqrt(x: f32) -> f32 {
        x.sqrt()
    }
    #[inline]
    pub fn acos(x: f32) -> f32 {
        x.acos()
    }
    #[inline]
    pub fn atan2(y: f32, x: f32) -> f32 {
        y.atan2(x)
    }
}

#[cfg(all(not(feature = "std"), feature = "libm"))]
mod imp {
    #[inline]
    pub fn sin(x: f32) -> f32 {
        libm::sinf(x)
    }
    #[inline]
    pub fn cos(x: f32) -> f32 {
        libm::cosf(x)
    }
    #[inline]
    pub fn exp(x: f32) -> f32 {
        libm::expf(x)
    }
    #[inline]
    pub fn ln(x: f32) -> f32 {
        libm::logf(x)
    }
    #[inline]
    pub fn powf(x: f32, y: f32) -> f32 {
        libm::powf(x, y)
    }
    #[inline]
    pub fn sqrt(x: f32) -> f32 {
        libm::sqrtf(x)
    }
    #[inline]
    pub fn acos(x: f32) -> f32 {
        libm::acosf(x)
    }
    #[inline]
    pub fn atan2(y: f32, x: f32) -> f32 {
        libm::atan2f(y, x)
    }
}

pub(crate) use imp::*;

/// Linear interpolation between `a` and `b`.
#[inline]
pub(crate) fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Move `current` towards `target`, rising by at most `up` and falling by at
/// most `down`.
#[inline]
pub(crate) fn move_towards(current: f32, target: f32, up: f32, down: f32) -> f32 {
    if current < target {
        (current + up).min(target)
    } else {
        (current - down).max(target)
    }
}

/// `floor` for `f64` without `std`.
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub(crate) fn floor_i64(x: f64) -> i64 {
    let i = x as i64;
    if x < i as f64 { i - 1 } else { i }
}

/// Frequency in Hz of a (fractional) MIDI note number.
#[must_use]
#[inline]
pub fn midi_to_hz(note: f32) -> f32 {
    440.0 * powf(2.0, (note - 69.0) / 12.0)
}

/// One exact step of `dt` seconds of a spring of natural frequency `w`
/// (rad/s) and damping ratio `zeta` (`1` critical; below it a little
/// overshoot and a shorter tail) from displacement `e0` and velocity `v0`:
/// the displacement and velocity after it. Exact for any `dt`, so a move
/// does not depend on how the control blocks fall.
pub(crate) fn spring_step(e0: f32, v0: f32, w: f32, zeta: f32, dt: f32) -> (f32, f32) {
    if zeta >= 0.999 {
        let decay = exp(-w * dt);
        let c = v0 + w * e0;
        return ((e0 + c * dt) * decay, (v0 - w * c * dt) * decay);
    }
    let wd = w * sqrt(1.0 - zeta * zeta);
    let decay = exp(-zeta * w * dt);
    let (sine, cosine) = (sin(wd * dt), cos(wd * dt));
    (
        decay * (e0 * cosine + (v0 + zeta * w * e0) / wd * sine),
        decay * (v0 * cosine - (w * w * e0 + zeta * w * v0) / wd * sine),
    )
}

/// Share of the way to its target that a first-order lag of time constant
/// `time` covers in a block of `dt` seconds, when a block is `share` of the
/// long block these lags were set at (512 internal samples). At `share` 1
/// it is the plain `dt / time` they have always used; a shorter block
/// covers the same ground over the same time, in more and smaller steps.
pub(crate) fn approach(dt: f32, time: f32, share: f32) -> f32 {
    let long = (dt / share / time).min(1.0);
    #[allow(clippy::float_cmp)]
    if share == 1.0 {
        long
    } else {
        1.0 - powf(1.0 - long, share)
    }
}
