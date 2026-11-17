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
//!
//! Once out, the sound of every port goes round a head ([`Head`], wave 3):
//! the lips', the nostrils' and the walls' flows each from their own place
//! on a rigid sphere to the two ears of a listener, and off the chest; and,
//! for the mono output, the ports as heard straight ahead, summed without
//! the paths' delays (the mixdown).

use super::junction::Y_MIN;
use super::{RHO, RHO_C};
use crate::math::{acos, atan2, cos, exp, sin, sqrt};
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

// ----- the head (wave 3, stage 3C) ---------------------------------------------

/// Speed of sound in the room, m/s (air at 20 °C; the tract's warm, moist
/// air is faster).
const C_AIR: f32 = 343.0;
/// The head shadow of a rigid sphere as one pole and one zero (Brown &
/// Duda 1998, fitted to Rayleigh's solution for a point on a sphere):
/// `H(ω, θ) = (1 + jαω/2ω₀) / (1 + jω/2ω₀)` with `ω₀ = c/a` and
/// `α(θ) = (1 + α_min/2) + (1 − α_min/2)·cos(θ/θ_min·π)`, `θ` the angle
/// between the point's outward normal and the direction of the far
/// listener (or source). Flat below about `c/(πa)` (620 Hz for 8.75 cm),
/// `α` above it: 2 (+6 dB) facing the listener, 0.76 (−2.4 dB) at 90°,
/// 0.1 at 150° and 0.28 straight behind (the bright spot).
const ALPHA_MIN: f32 = 0.1;
const THETA_MIN: f32 = 150.0 * PI / 180.0;
/// The listener's head radius, m (Brown & Duda's 8.75 cm; the talker's own
/// is `AnatomyConfig::head_radius` and grows with the speaker).
const LISTENER_RADIUS: f32 = 0.0875;
/// The plain stereo pair ([`Stereo::Pair`]): two cardioid microphones
/// 17 cm apart, pointing 55° to either side (ORTF), with no head between.
const PAIR_SPACING: f32 = 0.17;
const PAIR_ANGLE: f32 = 55.0 * PI / 180.0;
/// Ticks in each source's delay line (a power of two): the longest path
/// difference a listener can hear, round both heads and down to the chest,
/// is under 2.5 ms, 240 ticks at the native 96 kHz (more for the smallest
/// speaker, whose ticks are twice as short while the listener's head stays
/// as large).
const DELAY: usize = 512;
/// Seconds over which a moved listener's paths glide to their new lengths,
/// levels and colours (a step in a path's length is a click).
const GLIDE: f32 = 0.03;
/// Nearest an ear counts as being to a port in the paths' 1/r, m.
const NEAREST: f32 = 0.05;

/// Ports the tract's sound leaves by, as [`Head::tick`] takes them.
pub(crate) const LIPS: usize = 0;
pub(crate) const NOSTRIL_LEFT: usize = 1;
pub(crate) const NOSTRIL_RIGHT: usize = 2;
pub(crate) const THROAT: usize = 3;
pub(crate) const CHEEK_LEFT: usize = 4;
pub(crate) const CHEEK_RIGHT: usize = 5;
pub(crate) const PORTS: usize = 6;
/// The ports and the chest's image of the mouth.
const SOURCES: usize = PORTS + 1;
const CHEST: usize = PORTS;

/// Where each port sits on the talker's head, as (azimuth, elevation) of
/// its place on the sphere from the head's centre, degrees (positive to
/// the talker's right and up; the lips define straight ahead): the lips;
/// the nostrils 2 cm above them and 2.4 cm apart; the throat's skin below
/// the chin; the cheeks either side of the mouth. A path's length round
/// the head is measured from here.
const PORT_AT: [[f32; 2]; PORTS] = [
    [0.0, 0.0],
    [-8.0, 13.0],
    [8.0, 13.0],
    [0.0, -60.0],
    [-70.0, -10.0],
    [70.0, -10.0],
];
/// Where each port points, same angles: its high frequencies beam this way.
/// The nostrils open down and forward under the nose; the throat faces
/// forward and down.
const PORT_AXIS: [[f32; 2]; PORTS] = [
    [0.0, 0.0],
    [-10.0, -40.0],
    [10.0, -40.0],
    [0.0, -30.0],
    [-70.0, -10.0],
    [70.0, -10.0],
];
/// The chest's surface faces up and forward at this angle above straight
/// ahead, degrees: the mouth's sound that goes down meets it and comes back
/// up and forward.
const CHEST_FACING: f32 = 45.0;

/// How a voice is heard in two channels (see
/// [`Trombone::set_stereo`](crate::Trombone::set_stereo)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Stereo {
    /// Both channels carry the mono output.
    #[default]
    Off,
    /// The two ears of a listener (a rigid sphere of 8.75 cm): each port's
    /// own path round the talker's head and the listener's to each ear,
    /// with the level and arrival-time differences the listener's head
    /// makes. For headphones.
    Binaural,
    /// A plain stereo pair: two cardioid microphones 17 cm apart, pointing
    /// 55° to either side (ORTF), where the listener is. Level and time
    /// differences of a microphone pair, for loudspeakers.
    Pair,
}

/// Where a listener is, as the talker's head sees it, and how the
/// listener's own head is turned (see
/// [`Trombone::set_listener`](crate::Trombone::set_listener)).
///
/// Angles are degrees; azimuths are positive to the right, elevations up.
/// The talker's head is turned away from the listener by moving
/// `azimuth` (180 is behind the talker's head); the listener's head is
/// turned by moving `heard_azimuth`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Listener {
    /// Direction of the listener from the talker's face: 0 straight in
    /// front of the mouth, 90 at the talker's right, ±180 behind the head.
    pub azimuth: f32,
    /// Height of that direction above the mouth's level, −90 to 90.
    pub elevation: f32,
    /// Metres from the centre of the talker's head to the centre of the
    /// listener's, 0.3 to 1000.
    pub distance: f32,
    /// Where the listener hears the talker, in the listener's own head:
    /// 0 straight ahead (the listener faces the talker), 90 to the
    /// listener's right.
    pub heard_azimuth: f32,
    /// Height of that direction in the listener's head, −90 to 90.
    pub heard_elevation: f32,
}

impl Listener {
    /// Straight in front of the talker at a metre, facing the talker: the
    /// reference the mono mixdown is heard from.
    pub const FRONT: Self = Self {
        azimuth: 0.0,
        elevation: 0.0,
        distance: 1.0,
        heard_azimuth: 0.0,
        heard_elevation: 0.0,
    };

    /// The listener of two people on a map, as a game has them: positions
    /// in metres (`[east, north, up]`), headings in degrees clockwise from
    /// north (where each one's face points; heads level).
    #[must_use]
    pub fn on_map(
        talker: [f32; 3],
        talker_heading: f32,
        listener: [f32; 3],
        listener_heading: f32,
    ) -> Self {
        let d = [
            listener[0] - talker[0],
            listener[1] - talker[1],
            listener[2] - talker[2],
        ];
        let flat = sqrt(d[0] * d[0] + d[1] * d[1]);
        let distance = sqrt(flat * flat + d[2] * d[2]);
        let deg = 180.0 / PI;
        // Bearing of the listener from the talker, clockwise from north.
        let bearing = atan2(d[0], d[1]) * deg;
        let elevation = atan2(d[2], flat) * deg;
        Self {
            azimuth: wrap(bearing - talker_heading),
            elevation,
            distance,
            heard_azimuth: wrap(bearing + 180.0 - listener_heading),
            heard_elevation: -elevation,
        }
    }

    fn clamped(self) -> Self {
        let finite = |x: f32, or: f32| if x.is_finite() { x } else { or };
        Self {
            azimuth: wrap(finite(self.azimuth, 0.0)),
            elevation: finite(self.elevation, 0.0).clamp(-90.0, 90.0),
            distance: finite(self.distance, 1.0).clamp(0.3, 1000.0),
            heard_azimuth: wrap(finite(self.heard_azimuth, 0.0)),
            heard_elevation: finite(self.heard_elevation, 0.0).clamp(-90.0, 90.0),
        }
    }
}

impl Default for Listener {
    fn default() -> Self {
        Self::FRONT
    }
}

/// An angle in degrees brought into −180‥180.
fn wrap(deg: f32) -> f32 {
    let d = deg % 360.0;
    if d > 180.0 {
        d - 360.0
    } else if d < -180.0 {
        d + 360.0
    } else {
        d
    }
}

type Vec3 = [f32; 3];

fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: Vec3, k: f32) -> Vec3 {
    [a[0] * k, a[1] * k, a[2] * k]
}

fn norm(a: Vec3) -> f32 {
    sqrt(dot(a, a))
}

fn unit(a: Vec3) -> Vec3 {
    let n = norm(a);
    if n > 0.0 {
        scale(a, 1.0 / n)
    } else {
        [1.0, 0.0, 0.0]
    }
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// A direction in a head's frame (x forward, y left, z up) from azimuth
/// (positive right) and elevation, degrees.
fn direction(azimuth: f32, elevation: f32) -> Vec3 {
    let (a, e) = (azimuth * PI / 180.0, elevation * PI / 180.0);
    [cos(e) * cos(a), -cos(e) * sin(a), sin(e)]
}

fn angle(a: Vec3, b: Vec3) -> f32 {
    acos(dot(a, b).clamp(-1.0, 1.0))
}

/// Brown & Duda's `α` at `theta` radians from a sphere's normal.
fn alpha(theta: f32) -> f32 {
    (1.0 + 0.5 * ALPHA_MIN) + (1.0 - 0.5 * ALPHA_MIN) * cos(theta / THETA_MIN * PI)
}

/// How much further than the straight line a path from a point on a
/// sphere of `radius` goes, round it, to leave towards a direction at
/// `theta` from the point's normal (Woodworth's: the path hugs the sphere
/// past 90°): `a·(θ − π/2 + cos θ)` there, nothing before.
fn wrap_excess(radius: f32, theta: f32) -> f32 {
    radius * (theta - 0.5 * PI + cos(theta)).max(0.0)
}

/// Coefficients `[b0, b1, a1]` of `(1 + s·α/2ω₀) / (1 + s·β/2ω₀)` by the
/// bilinear transform, `k = a/(c·dt)`: unity at 0 Hz, `α/β` at the top.
fn shelf(alpha: f32, beta: f32, k: f32) -> [f32; 3] {
    let n = 1.0 / (1.0 + k * beta);
    [
        (1.0 + k * alpha) * n,
        (1.0 - k * alpha) * n,
        (1.0 - k * beta) * n,
    ]
}

/// The state of one first-order section, `y = b0·x + b1·x' − a1·y'`.
#[derive(Clone, Copy, Debug, Default)]
struct Section {
    x: f32,
    y: f32,
}

impl Section {
    #[inline]
    fn run(&mut self, c: &[f32; 3], x: f32) -> f32 {
        let y = c[0] * x + c[1] * self.x - c[2] * self.y;
        self.x = x;
        self.y = y;
        y
    }
}

/// The head: the sound of the tract's ports as a listener hears it, and
/// as the mono mixdown has it.
///
/// Each port (the lips, the two nostrils, the throat's and the cheeks'
/// walls) sits at its own place on a rigid sphere and points its own way.
/// Its sound beams as Brown & Duda's sphere says from where it points (the
/// lips' highs go forward; the walls carry only low frequencies, which a
/// head this size does not shadow), and travels to each ear of the
/// listener on its own path: the straight line from the port to the ear,
/// plus the arc round the talker's head where the port is turned away,
/// plus the arc round the listener's head to an ear turned away; its level
/// falls as 1/r along it. The mouth's sound also comes back off the chest:
/// an image of the lips mirrored in a plane facing up and forward,
/// low-passed (a chest is soft and smaller than the wavelengths it
/// returns), heard only from the side it faces. The listener's head then
/// shadows each ear as the sphere does, seen from the direction the
/// talker is heard from.
///
/// Everything is relative to the mouth heard straight ahead: the engine's
/// level and source tilt were fitted to recordings made in front of the
/// speaker, so what it put out before is the mouth on its axis. A port's
/// beaming here is its sphere response over the lips' own on the axis, an
/// ear's shadow its response over that of a sound from straight ahead.
///
/// The mono mixdown is the ports as heard straight ahead on the axis, each
/// through its own beaming and nothing else: no path lengths, no chest.
/// Its filters are first-order shelves that differ in phase by at most 16°
/// (the lips' and a cheek's, near 820 Hz), so no two ports can cancel: the
/// deepest notch two equal ports make in it below 8 kHz is 0.08 dB, where
/// a listener a metre in front hears one of about 1.5 dB near 800 Hz from
/// the chest's return.
#[derive(Clone, Debug)]
pub(crate) struct Head {
    stereo: Stereo,
    mixdown: bool,
    listener: Listener,
    dt: f32,
    /// The speaker's size, and the tuning's head radius (cm), chest share,
    /// chest distance (cm) and chest corner (Hz) at the native size.
    len: f32,
    anatomy: [f32; 4],
    /// The talker's head radius, m, and the chest's return: its share, its
    /// distance from the lips along its normal (m) and its low-pass corner
    /// (Hz).
    radius: f32,
    chest: [f32; 3],
    /// `a/(c·dt)` of the talker's head and of the listener's.
    k_talker: f32,
    k_listener: f32,
    lines: [[f32; DELAY]; SOURCES],
    at: usize,
    /// Each port's beaming towards the listener: `α` now and where it
    /// goes, its coefficients and its state.
    beam: [f32; PORTS],
    beam_to: [f32; PORTS],
    beam_c: [[f32; 3]; PORTS],
    beam_s: [Section; PORTS],
    /// Each ear's shadow: `α` now and where it goes, the `α` of a sound
    /// from straight ahead, the coefficients, the state; and a gain (the
    /// pair's cardioids).
    ear: [f32; 2],
    ear_to: [f32; 2],
    ear_ref: f32,
    ear_c: [[f32; 3]; 2],
    ear_s: [Section; 2],
    ear_gain: [f32; 2],
    ear_gain_to: [f32; 2],
    /// Each source's path to each ear: delay in ticks and level.
    delay: [[f32; 2]; SOURCES],
    delay_to: [[f32; 2]; SOURCES],
    gain: [[f32; 2]; SOURCES],
    gain_to: [[f32; 2]; SOURCES],
    /// The chest's low-pass: pole and state.
    chest_pole: f32,
    chest_s: f32,
    /// The mixdown's beaming of each port.
    mix_c: [[f32; 3]; PORTS],
    mix_s: [Section; PORTS],
    /// Ticks left of a glide, and its share per tick.
    glide: u32,
    glide_k: f32,
}

impl Head {
    pub(crate) fn new() -> Self {
        let mut h = Self {
            stereo: Stereo::Off,
            mixdown: false,
            listener: Listener::FRONT,
            dt: 1.0 / (2.0 * crate::INTERNAL_RATE),
            len: 1.0,
            anatomy: [8.75, 0.0, 15.0, 1500.0],
            radius: 0.0875,
            chest: [0.0, 0.15, 1500.0],
            k_talker: 0.0,
            k_listener: 0.0,
            lines: [[0.0; DELAY]; SOURCES],
            at: 0,
            beam: [2.0; PORTS],
            beam_to: [2.0; PORTS],
            beam_c: [[1.0, 0.0, 0.0]; PORTS],
            beam_s: [Section::default(); PORTS],
            ear: [1.0; 2],
            ear_to: [1.0; 2],
            ear_ref: 1.0,
            ear_c: [[1.0, 0.0, 0.0]; 2],
            ear_s: [Section::default(); 2],
            ear_gain: [1.0; 2],
            ear_gain_to: [1.0; 2],
            delay: [[0.0; 2]; SOURCES],
            delay_to: [[0.0; 2]; SOURCES],
            gain: [[0.0; 2]; SOURCES],
            gain_to: [[0.0; 2]; SOURCES],
            chest_pole: 0.0,
            chest_s: 0.0,
            mix_c: [[1.0, 0.0, 0.0]; PORTS],
            mix_s: [Section::default(); PORTS],
            glide: 0,
            glide_k: 0.0,
        };
        h.set_tuning(&crate::Tuning::DEFAULT);
        h
    }

    /// Whether anything of the head is to be computed.
    #[inline]
    pub(crate) fn on(&self) -> bool {
        self.mixdown || self.stereo != Stereo::Off
    }

    pub(crate) fn stereo(&self) -> Stereo {
        self.stereo
    }

    /// Whether the mixdown and the ears are being computed.
    #[inline]
    pub(crate) fn parts(&self) -> (bool, bool) {
        (self.mixdown, self.stereo != Stereo::Off)
    }

    pub(crate) fn listener(&self) -> Listener {
        self.listener
    }

    /// Set the tick length and the speaker's size `len` (the talker's head
    /// and chest grow with the tract, so their paths are as many ticks long
    /// at every size).
    pub(crate) fn set_rate(&mut self, dt: f32, len: f32) {
        self.dt = dt;
        self.len = len;
        self.configure();
    }

    /// Take the head's numbers from a tuning.
    pub(crate) fn set_tuning(&mut self, tuning: &crate::Tuning) {
        let a = &tuning.anatomy;
        self.anatomy = [
            a.head_radius,
            a.chest_reflection,
            a.chest_distance,
            a.chest_corner,
        ];
        let mixdown = a.head_mixdown > 0.0;
        if mixdown && !self.mixdown {
            self.mix_s = [Section::default(); PORTS];
        }
        self.mixdown = mixdown;
        self.configure();
    }

    /// Choose the two channels' mode; the paths start from silence.
    pub(crate) fn set_stereo(&mut self, stereo: Stereo) {
        if stereo != self.stereo {
            self.stereo = stereo;
            self.clear_paths();
            self.aim(true);
        }
    }

    /// Move the listener: the paths glide there over [`GLIDE`].
    pub(crate) fn set_listener(&mut self, listener: Listener) {
        self.listener = listener.clamped();
        self.aim(false);
    }

    /// Silence: the delay lines and every filter's state.
    pub(crate) fn reset(&mut self) {
        self.clear_paths();
        self.mix_s = [Section::default(); PORTS];
    }

    fn clear_paths(&mut self) {
        self.lines = [[0.0; DELAY]; SOURCES];
        self.beam_s = [Section::default(); PORTS];
        self.ear_s = [Section::default(); 2];
        self.chest_s = 0.0;
    }

    /// Everything that follows from the rate and the anatomy; the listener's
    /// paths are set at once.
    fn configure(&mut self) {
        let [radius, share, distance, corner] = self.anatomy;
        self.radius = 0.01 * radius * self.len;
        self.chest = [share, 0.01 * distance * self.len, corner / self.len];
        self.k_talker = self.radius / (C_AIR * self.dt);
        self.k_listener = LISTENER_RADIUS / (C_AIR * self.dt);
        self.chest_pole = exp(-2.0 * PI * self.chest[2] * self.dt);
        self.glide_k = (self.dt / GLIDE).min(1.0);
        let ahead = [1.0, 0.0, 0.0];
        for (c, [az, el]) in self.mix_c.iter_mut().zip(PORT_AXIS) {
            *c = shelf(alpha(angle(direction(az, el), ahead)), 2.0, self.k_talker);
        }
        self.ear_ref = alpha(0.5 * PI);
        self.aim(true);
    }

    /// Work out where every path is going for the listener; `snap` puts
    /// them there at once, else they glide.
    #[allow(
        clippy::many_single_char_names,
        clippy::too_many_lines,
        clippy::needless_range_loop
    )]
    fn aim(&mut self, snap: bool) {
        let l = self.listener;
        let pair = self.stereo == Stereo::Pair;
        let a_t = self.radius;
        let a_l = if pair {
            0.5 * PAIR_SPACING
        } else {
            LISTENER_RADIUS
        };
        // Directions in the talker's frame: x forward, y left, z up.
        let toward = direction(l.azimuth, l.elevation);
        let centre = scale(toward, l.distance.max(a_t + a_l + NEAREST));
        // The listener's frame, first facing the talker with the head level,
        // then turned so that the talker is heard where `heard_*` says.
        let f0 = scale(toward, -1.0);
        let up = sub([0.0, 0.0, 1.0], scale(f0, f0[2]));
        let u0 = if norm(up) > 1e-3 {
            unit(up)
        } else {
            unit(sub([1.0, 0.0, 0.0], scale(f0, f0[0])))
        };
        let l0 = cross(u0, f0);
        let (ha, he) = (l.heard_azimuth * PI / 180.0, l.heard_elevation * PI / 180.0);
        let (ca, sa, ce, se) = (cos(ha), sin(ha), cos(he), sin(he));
        let to_listener = |v: Vec3| -> Vec3 {
            let (x, y, z) = (dot(v, f0), dot(v, l0), dot(v, u0));
            let (x, z) = (x * ce - z * se, x * se + z * ce);
            [x * ca + y * sa, -x * sa + y * ca, z]
        };
        let to_world = |v: Vec3| -> Vec3 {
            let (x, y, z) = (v[0] * ca - v[1] * sa, v[0] * sa + v[1] * ca, v[2]);
            let (x, z) = (x * ce + z * se, -x * se + z * ce);
            add(add(scale(f0, x), scale(l0, y)), scale(u0, z))
        };
        let ears: [Vec3; 2] = [[0.0, a_l, 0.0], [0.0, -a_l, 0.0]];
        let ear_at = [
            add(centre, to_world(ears[0])),
            add(centre, to_world(ears[1])),
        ];
        // Path lengths are kept relative to the shortest any path can be.
        let common = norm(centre) - a_t - a_l;
        // Levels relative to the lips' path to an ear straight ahead at a
        // metre.
        let reference = {
            let r = 1.0 - self.radius;
            sqrt(r * r + LISTENER_RADIUS * LISTENER_RADIUS)
        };
        let lips_at = scale(direction(PORT_AT[LIPS][0], PORT_AT[LIPS][1]), a_t);
        let chest_normal = direction(0.0, CHEST_FACING);
        let image = sub(lips_at, scale(chest_normal, 2.0 * self.chest[1]));
        #[allow(clippy::cast_precision_loss)]
        let longest = (DELAY - 2) as f32;
        for s in 0..SOURCES {
            let (place, normal) = if s == CHEST {
                (image, None)
            } else {
                let n = direction(PORT_AT[s][0], PORT_AT[s][1]);
                (scale(n, a_t), Some(n))
            };
            for e in 0..2 {
                let q = sub(ear_at[e], place);
                let r = norm(q);
                let dir = unit(q);
                let mut path = r;
                let mut level = reference / r.max(NEAREST);
                match normal {
                    Some(n) => path += wrap_excess(a_t, angle(n, dir)),
                    None => level *= self.chest[0] * dot(chest_normal, dir).max(0.0),
                }
                if !pair {
                    // Round the listener's head to an ear turned away.
                    let arriving = scale(to_listener(dir), -1.0);
                    path += wrap_excess(a_l, angle(unit(ears[e]), arriving));
                }
                let ticks = (path - common) / (C_AIR * self.dt);
                self.delay_to[s][e] = ticks.clamp(0.0, longest);
                self.gain_to[s][e] = level;
            }
        }
        for p in 0..PORTS {
            let [az, el] = PORT_AXIS[p];
            self.beam_to[p] = alpha(angle(direction(az, el), toward));
        }
        // The talker as the listener's head has it.
        let heard = direction(l.heard_azimuth, l.heard_elevation);
        for e in 0..2 {
            if pair {
                let side = if e == 0 { -1.0 } else { 1.0 };
                let mic = direction(side * PAIR_ANGLE * 180.0 / PI, 0.0);
                self.ear_to[e] = self.ear_ref;
                self.ear_gain_to[e] = (1.0 + dot(mic, heard)) / (1.0 + cos(PAIR_ANGLE));
            } else {
                self.ear_to[e] = alpha(angle(unit(ears[e]), heard));
                self.ear_gain_to[e] = 1.0;
            }
        }
        if snap {
            self.delay = self.delay_to;
            self.gain = self.gain_to;
            self.beam = self.beam_to;
            self.ear = self.ear_to;
            self.ear_gain = self.ear_gain_to;
            self.glide = 0;
            self.coefficients();
        } else {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            {
                self.glide = ((8.0 * GLIDE / self.dt) as u32).max(1);
            }
        }
    }

    fn coefficients(&mut self) {
        for p in 0..PORTS {
            self.beam_c[p] = shelf(self.beam[p], 2.0, self.k_talker);
        }
        for e in 0..2 {
            self.ear_c[e] = shelf(self.ear[e], self.ear_ref, self.k_listener);
        }
    }

    /// One step of a glide towards where the listener's paths are going.
    fn step_glide(&mut self) {
        self.glide -= 1;
        let k = if self.glide == 0 { 1.0 } else { self.glide_k };
        let go = |x: &mut f32, to: f32| *x += (to - *x) * k;
        for s in 0..SOURCES {
            for e in 0..2 {
                go(&mut self.delay[s][e], self.delay_to[s][e]);
                go(&mut self.gain[s][e], self.gain_to[s][e]);
            }
        }
        for p in 0..PORTS {
            go(&mut self.beam[p], self.beam_to[p]);
        }
        for e in 0..2 {
            go(&mut self.ear[e], self.ear_to[e]);
            go(&mut self.ear_gain[e], self.ear_gain_to[e]);
        }
        self.coefficients();
    }

    /// One tick: the ports' flows in, `[mixdown, left ear, right ear]` out
    /// (a part not asked for is `0`).
    #[inline]
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn tick(&mut self, ports: &[f32; PORTS]) -> [f32; 3] {
        let mut out = [0.0f32; 3];
        if self.mixdown {
            let mut m = ports[LIPS];
            for p in 1..PORTS {
                m += self.mix_s[p].run(&self.mix_c[p], ports[p]);
            }
            out[0] = m;
        }
        if self.stereo == Stereo::Off {
            return out;
        }
        if self.glide > 0 {
            self.step_glide();
        }
        let at = self.at;
        for p in 0..PORTS {
            self.lines[p][at] = self.beam_s[p].run(&self.beam_c[p], ports[p]);
        }
        self.chest_s = ports[LIPS] + (self.chest_s - ports[LIPS]) * self.chest_pole;
        self.lines[CHEST][at] = self.chest_s;
        for e in 0..2 {
            let mut sum = 0.0;
            for s in 0..SOURCES {
                let d = self.delay[s][e];
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let whole = d as usize;
                #[allow(clippy::cast_precision_loss)]
                let frac = d - whole as f32;
                let line = &self.lines[s];
                let a = line[(at + DELAY - whole) & (DELAY - 1)];
                let b = line[(at + DELAY - whole - 1) & (DELAY - 1)];
                sum += self.gain[s][e] * (a + (b - a) * frac);
            }
            out[1 + e] = self.ear_gain[e] * self.ear_s[e].run(&self.ear_c[e], sum);
        }
        self.at = (at + 1) & (DELAY - 1);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The response of `[b0, b1, a1]` at `hz` for ticks of `dt`: (re, im).
    fn response(c: &[f32; 3], hz: f32, dt: f32) -> (f32, f32) {
        let w = 2.0 * PI * hz * dt;
        let (zr, zi) = (cos(w), -sin(w));
        let (nr, ni) = (c[0] + c[1] * zr, c[1] * zi);
        let (dr, di) = (1.0 + c[2] * zr, c[2] * zi);
        let d = dr * dr + di * di;
        ((nr * dr + ni * di) / d, (ni * dr - nr * di) / d)
    }

    #[test]
    fn the_mixdown_makes_no_notch_between_ports() {
        // Two ports of equal level summed in the mixdown, each through its
        // own beaming: the sum never falls more than a tenth of a dB under
        // the sum of their levels below 8 kHz (the worst pair is the lips
        // and a cheek, whose shelves differ most in phase: 16° at 820 Hz).
        // A listener's paths are not phase-clean: straight ahead the
        // chest's return alone, 0.62 ms late at 0.18 of the lips' level,
        // puts a notch of about 1.5 dB near 800 Hz.
        let head = Head::new();
        let flat = [1.0, 0.0, 0.0];
        let mut worst = (0.0f32, 0, 0, 0.0);
        for p in 0..PORTS {
            for q in p + 1..PORTS {
                let (cp, cq) = (if p == LIPS { flat } else { head.mix_c[p] }, head.mix_c[q]);
                let mut hz = 20.0;
                while hz <= 8000.0 {
                    let (a, b) = (response(&cp, hz, head.dt), response(&cq, hz, head.dt));
                    let together = sqrt((a.0 + b.0).powi(2) + (a.1 + b.1).powi(2));
                    let apart = sqrt(a.0 * a.0 + a.1 * a.1) + sqrt(b.0 * b.0 + b.1 * b.1);
                    let db = 20.0 * (together / apart).log10();
                    if db < worst.0 {
                        worst = (db, p, q, hz);
                    }
                    hz += 10.0;
                }
            }
        }
        std::println!("worst notch {worst:?}");
        assert!(worst.0 > -0.1, "{worst:?}");
        assert!(worst.0 < -0.01, "{worst:?}");
    }

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
