//! The head (wave 3, stage 3C): the tract's ports heard from their own
//! places on a head by the two ears of a listener, and the mono mixdown,
//! measured through the public API.
//!
//! The head is Brown & Duda's rigid sphere (8.75 cm): a port's highs beam
//! the way it points, its path to each ear goes round both heads, and the
//! listener's head shadows the ear turned away. Each test prints what it
//! measured.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use synthie_trombone::stream::{Event, voice_stream};
use synthie_trombone::{Engine, Listener, Phoneme, Preset, Stereo, Talker, Trombone, Tuning};

#[path = "../examples/support/spectrum.rs"]
mod spectrum;

const SR: f32 = 48_000.0;

/// A voice holding `phoneme` on the physical engine at 110 Hz, heard as
/// `stereo` by `listener`, with `Tuning::head_mixdown` at `mixdown`.
fn voice(phoneme: Phoneme, stereo: Stereo, listener: Listener, mixdown: f32) -> Trombone {
    let mut t = Trombone::new(SR);
    let mut tuning = Tuning::DEFAULT;
    tuning.anatomy.head_mixdown = mixdown;
    t.set_tuning(&tuning);
    t.set_engine(Engine::Physical);
    t.set_stereo(stereo);
    t.set_listener(listener);
    t.articulate_phoneme(phoneme);
    t.settle();
    t.set_vibrato(0.0, 0.0);
    t.set_jitter(0.0);
    t.set_glide(f32::INFINITY);
    t.set_frequency(110.0);
    t.note_on();
    t
}

/// `seconds` of the voice: left and right.
fn record(t: &mut Trombone, seconds: f32) -> [Vec<f32>; 2] {
    let n = (seconds * SR) as usize;
    let (mut l, mut r) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n {
        let [a, b] = t.process_stereo();
        l.push(a);
        r.push(b);
    }
    [l, r]
}

/// A held sound once it has settled: 0.25 s from 0.4 s on.
fn held(phoneme: Phoneme, stereo: Stereo, listener: Listener) -> [Vec<f32>; 2] {
    let mut t = voice(phoneme, stereo, listener, 0.0);
    record(&mut t, 0.4);
    record(&mut t, 0.25)
}

/// The same, mono.
fn mono(phoneme: Phoneme, mixdown: f32) -> Vec<f32> {
    let mut t = voice(phoneme, Stereo::Off, Listener::FRONT, mixdown);
    let mut x = vec![0.0f32; (0.4 * SR) as usize];
    t.render(&mut x);
    let mut x = vec![0.0f32; (0.25 * SR) as usize];
    t.render(&mut x);
    x
}

const STEP: f32 = 20.0;

/// Energy of `x` between `lo` and `hi` Hz, dB.
fn band(x: &[f32], lo: f32, hi: f32) -> f32 {
    let mag = spectrum::spectrum(x, SR, hi, STEP);
    let first = (lo / STEP) as usize;
    let e: f32 = mag[first..].iter().map(|m| m * m).sum();
    10.0 * e.max(1e-30).log10()
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

/// How much later `b` is than `a`, ms: the peak of their
/// cross-correlation, between samples by a parabola.
fn delay(a: &[f32], b: &[f32]) -> f32 {
    let corr = |k: i32| -> f32 {
        let mut s = 0.0f64;
        for n in 64..a.len() - 64 {
            let m = (n as i32 + k) as usize;
            s += f64::from(a[n]) * f64::from(b[m]);
        }
        s as f32
    };
    let (mut best, mut at) = (f32::MIN, 0);
    for k in -60..=60 {
        let c = corr(k);
        if c > best {
            best = c;
            at = k;
        }
    }
    let (y0, y1, y2) = (corr(at - 1), best, corr(at + 1));
    let frac = 0.5 * (y0 - y2) / (y0 - 2.0 * y1 + y2);
    1e3 * (at as f32 + frac) / SR
}

/// The listener in front at a metre, their head turned so that the talker
/// is heard at `heard_azimuth`.
fn turned(heard_azimuth: f32) -> Listener {
    Listener {
        heard_azimuth,
        ..Listener::FRONT
    }
}

/// The listener at a metre in the direction (`azimuth`, `elevation`) from
/// the talker's face, facing the talker.
fn from(azimuth: f32, elevation: f32) -> Listener {
    Listener {
        azimuth,
        elevation,
        ..Listener::FRONT
    }
}

#[test]
fn straight_ahead_the_two_ears_are_alike() {
    // Everything on the head is symmetric but the nostrils: the left has
    // 1.15 times the area of the right and a cell more length, so each
    // ear hears the nostril on its side a little nearer. A vowel (the
    // velum shut) comes to both ears alike; a nasal differs by the
    // nostrils' own asymmetry, some 30 dB under the sound.
    for (p, under) in [(Phoneme::Aa, 80.0), (Phoneme::Iy, 80.0), (Phoneme::M, 25.0)] {
        let [l, r] = held(p, Stereo::Binaural, Listener::FRONT);
        let d: Vec<f32> = l.iter().zip(&r).map(|(a, b)| a - b).collect();
        let db = 20.0 * (rms(&l) / rms(&d).max(1e-20)).log10();
        println!("{p}: left minus right {db:.1} dB under the left ear");
        assert!(db > under, "{p}: {db:.1} dB");
    }
}

#[test]
fn to_one_side_the_far_ear_is_later_and_duller() {
    // The talker heard at the listener's right: the left ear is later by
    // the sphere's delay (Woodworth's path round it, a·(1 + π/2)/c =
    // 0.66 ms, is the high frequencies'; the low frequencies that carry
    // most of a voice's energy arrive later still, Kuhn's 3a/c = 0.77 ms,
    // from the shadow's own phase) and duller above 2 kHz by its shadow
    // (the sphere's 0.28 against 2 at the top, 17 dB) while the lows go
    // round.
    for p in [Phoneme::Aa, Phoneme::Iy, Phoneme::M] {
        let [l, r] = held(p, Stereo::Binaural, turned(90.0));
        let late = delay(&r, &l);
        let dull = band(&r, 2000.0, 6000.0) - band(&l, 2000.0, 6000.0);
        let low = band(&r, 0.0, 500.0) - band(&l, 0.0, 500.0);
        println!(
            "{p}: the far ear {late:.3} ms later, {dull:.1} dB duller at 2 to 6 kHz, {low:.1} dB below 500 Hz"
        );
        assert!((0.6..=0.9).contains(&late), "{p}: {late} ms");
        assert!((10.0..=20.0).contains(&dull), "{p}: {dull} dB");
        assert!(low.abs() <= 4.0, "{p}: {low} dB");
    }
    // The plain pair: time and level differences of two cardioids 17 cm
    // apart (0.17/343 s = 0.50 ms; 1 + cos 35° against 1 + cos 145°).
    let [l, r] = held(Phoneme::Aa, Stereo::Pair, turned(90.0));
    let late = delay(&r, &l);
    let level = 20.0 * (rms(&r) / rms(&l)).log10();
    println!("pair: the far microphone {late:.3} ms later, {level:.1} dB lower");
    assert!((late - 0.496).abs() < 0.02, "{late} ms");
    assert!((level - 21.7).abs() < 1.0, "{level} dB");
}

#[test]
fn behind_the_head_the_lips_highs_fall_and_the_walls_lows_stay() {
    // A vowel from beside and behind the talker against from in front:
    // the lips' highs lose the sphere's beaming (2 at the top in front,
    // 0.76 at the side, 0.28 behind; measured speech loses 5 to 15 dB
    // there as well). Below 300 Hz the vowel loses only the chest's return
    // (+1.4 dB in front, none behind) and a longer path's 1/r (1.5 dB).
    for p in [Phoneme::Aa, Phoneme::Iy] {
        let [front, _] = held(p, Stereo::Binaural, Listener::FRONT);
        for (az, highs) in [(90.0, 6.0..=11.0), (180.0, 10.0..=16.0)] {
            let [heard, _] = held(p, Stereo::Binaural, from(az, 0.0));
            let fall = band(&front, 2000.0, 6000.0) - band(&heard, 2000.0, 6000.0);
            let low = band(&front, 0.0, 300.0) - band(&heard, 0.0, 300.0);
            println!(
                "{p} from {az}°: {fall:.1} dB fewer highs (2 to 6 kHz), {low:.1} dB fewer lows (0 to 300 Hz)"
            );
            assert!(highs.contains(&fall), "{p} at {az}: {fall} dB");
            assert!((0.0..=3.5).contains(&low), "{p} at {az}: {low} dB");
        }
    }
    // The walls alone: the voice bar of a closed /b/, which leaves by the
    // throat and the cheeks only. Their sound is low and goes round the
    // head: from behind it is within 3 dB of what it is in front (it loses
    // 2.2: at 300 Hz a sphere this size already shadows its back by a dB,
    // and the throat's path round to the back is longer), where the lips'
    // highs lose 12 to 14.
    let bar = |listener: Listener| {
        let mut t = voice(Phoneme::Aa, Stereo::Binaural, listener, 0.0);
        record(&mut t, 0.4);
        t.articulate_phoneme(Phoneme::B);
        record(&mut t, 0.03);
        record(&mut t, 0.06)
    };
    let [front, _] = bar(Listener::FRONT);
    let [behind, _] = bar(from(180.0, 0.0));
    let low = band(&front, 0.0, 400.0) - band(&behind, 0.0, 400.0);
    let all = 20.0 * (rms(&front) / rms(&behind)).log10();
    println!("voice bar from behind: {low:.1} dB fewer lows, {all:.1} dB in all");
    assert!(low.abs() <= 3.0 && all.abs() <= 3.0, "{low} {all}");
}

#[test]
fn a_nasal_comes_from_the_nostrils_and_a_vowel_from_the_lips() {
    // The nostrils open down and forward, the lips straight ahead: from
    // 60° below the face against 60° above, the highs of /m/ (1.5 to
    // 5 kHz, its murmur and the nostrils' share of it) are several dB
    // stronger; those of a vowel, from the lips, are not.
    let lift = |p: Phoneme| {
        let [below, _] = held(p, Stereo::Binaural, from(0.0, -60.0));
        let [above, _] = held(p, Stereo::Binaural, from(0.0, 60.0));
        band(&below, 1500.0, 5000.0) - band(&above, 1500.0, 5000.0)
    };
    let (m, n, a, i) = (
        lift(Phoneme::M),
        lift(Phoneme::N),
        lift(Phoneme::Aa),
        lift(Phoneme::Iy),
    );
    println!(
        "below against above, 1.5 to 5 kHz: /m/ {m:.1} dB, /n/ {n:.1}, /a/ {a:.1}, /i/ {i:.1}"
    );
    assert!(m >= 4.0 && n >= 4.0, "{m} {n}");
    assert!(a.abs() <= 1.5 && i.abs() <= 1.5, "{a} {i}");
}

/// The largest second difference of `x` in each window of `w` samples.
fn roughness(x: &[f32], w: usize) -> Vec<f32> {
    x.chunks(w)
        .map(|c| {
            c.windows(3)
                .map(|s| (s[0] - 2.0 * s[1] + s[2]).abs())
                .fold(0.0, f32::max)
        })
        .collect()
}

#[test]
fn moving_the_listener_makes_no_click() {
    // Turn the talker's head away and the listener's head aside at once,
    // and walk the listener off to 4 m, while a hummed /m/ plays (smooth,
    // so that a click would stand out): the paths glide, and nothing in
    // the move is rougher than the sound on either side of it. (Put there
    // at once instead, the step in the paths' lengths makes the move 37
    // times rougher than the hum.)
    for stereo in [Stereo::Binaural, Stereo::Pair] {
        let mut t = voice(Phoneme::M, stereo, Listener::FRONT, 0.0);
        record(&mut t, 0.4);
        let before = record(&mut t, 0.1);
        t.set_listener(Listener {
            azimuth: 150.0,
            elevation: 20.0,
            distance: 4.0,
            heard_azimuth: -80.0,
            heard_elevation: 0.0,
        });
        let during = record(&mut t, 0.15);
        record(&mut t, 0.2);
        let after = record(&mut t, 0.1);
        for k in 0..2 {
            let w = 96;
            let calm = roughness(&before[k], w)
                .into_iter()
                .chain(roughness(&after[k], w))
                .fold(0.0, f32::max);
            let moving = roughness(&during[k], w).into_iter().fold(0.0, f32::max);
            println!("{stereo:?} ear {k}: roughest {moving:.2e} while moving, {calm:.2e} still");
            assert!(
                moving <= 1.2 * calm,
                "{stereo:?} ear {k}: {moving} against {calm}"
            );
        }
    }
}

#[test]
fn the_mixdown_keeps_the_old_sum_s_spectrum() {
    // The mixdown is the ports as heard straight ahead, each through its
    // own beaming: the lips as they were, the nostrils (pointing 40° down)
    // a little duller above 1 kHz, the walls' lows as they were. In 500 Hz
    // bands up to 5 kHz a vowel stays within 0.25 dB of the plain sum, a
    // nasal within 1.5 dB under it.
    for (p, worst) in [
        (Phoneme::Aa, 0.25),
        (Phoneme::Iy, 0.25),
        (Phoneme::M, 1.5),
        (Phoneme::N, 1.5),
    ] {
        let old = mono(p, 0.0);
        let mix = mono(p, 1.0);
        let mut bands = Vec::new();
        for k in 0..10 {
            let lo = 500.0 * k as f32;
            bands.push(band(&mix, lo, lo + 500.0) - band(&old, lo, lo + 500.0));
        }
        let text: Vec<String> = bands.iter().map(|d| format!("{d:+.2}")).collect();
        println!(
            "{p}: mixdown against the sum, by 500 Hz bands to 5 kHz: {}",
            text.join(" ")
        );
        for d in bands {
            assert!(d <= 0.25 && d >= -worst, "{p}: {d} dB");
        }
    }
}

#[test]
fn the_mono_output_does_not_depend_on_the_stereo() {
    // With the mixdown off, the mono samples are the old sum's whatever
    // the two channels do (the coordinator's renders compare files with
    // `main` byte for byte).
    let speak = |stereo: Stereo| {
        let mut t = Talker::new(SR, Preset::Deep.settings().with_engine(Engine::Physical));
        t.set_stereo(stereo);
        t.set_listener(Listener {
            azimuth: 120.0,
            ..Listener::FRONT
        });
        t.bake("Many men mumbling.")
    };
    let off = speak(Stereo::Off);
    for stereo in [Stereo::Binaural, Stereo::Pair] {
        let on = speak(stereo);
        assert_eq!(off.len(), on.len());
        assert!(
            off.iter().zip(&on).all(|(a, b)| a.to_bits() == b.to_bits()),
            "{stereo:?}"
        );
    }
}

#[test]
fn the_classic_engine_is_untouched() {
    // The classic engine has no ports: in stereo both channels carry its
    // mono samples, bit for bit.
    let m = Talker::new(SR, Preset::Deep).bake("Hello there.");
    let mut t = Talker::new(SR, Preset::Deep);
    t.set_stereo(Stereo::Binaural);
    let s = t.bake_stereo("Hello there.");
    assert_eq!(2 * m.len(), s.len());
    for (k, x) in m.iter().enumerate() {
        assert_eq!(x.to_bits(), s[2 * k].to_bits());
        assert_eq!(x.to_bits(), s[2 * k + 1].to_bits());
    }
}

#[test]
fn the_host_s_buffer_length_does_not_matter() {
    // The same speech, and the same move of the listener at the same
    // frame, rendered in buffers of 1, 37 and 480 frames: the same bits.
    let render = |chunk: usize| {
        let mut t = Talker::new(SR, Preset::Deep.settings().with_engine(Engine::Physical));
        t.set_stereo(Stereo::Binaural);
        t.say("Papa sees a man.");
        let mut out = vec![0.0f32; 2 * 48_000];
        let mut done = 0;
        for block in out.chunks_mut(2 * chunk) {
            if done <= 20_000 && done + block.len() / 2 > 20_000 {
                // The move lands on frame 20 000 whatever the chunks.
                let (a, b) = block.split_at_mut(2 * (20_000 - done));
                t.render_stereo(a);
                t.set_listener(Listener {
                    azimuth: 90.0,
                    heard_azimuth: 40.0,
                    ..Listener::FRONT
                });
                t.render_stereo(b);
            } else {
                t.render_stereo(block);
            }
            done += block.len() / 2;
        }
        out
    };
    let one = render(1);
    for chunk in [37, 480] {
        let other = render(chunk);
        assert!(
            one.iter()
                .zip(&other)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{chunk}"
        );
    }
}

struct Counting;

thread_local! {
    /// Set on the thread that renders, while it is inside the voice.
    static WATCHED: Cell<bool> = const { Cell::new(false) };
}

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call is passed on to the system allocator unchanged; the
// counting reads a thread-local flag and bumps an atomic, neither of which
// allocates.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if WATCHED.try_with(Cell::get).unwrap_or(false) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if WATCHED.try_with(Cell::get).unwrap_or(false) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn the_stereo_path_never_allocates() {
    // Rendering in stereo, moving the listener and switching the mode, on
    // the voice and through a stream's player.
    let (mut stream, mut player) = voice_stream(1024);
    let mut t = voice(Phoneme::Aa, Stereo::Binaural, Listener::FRONT, 1.0);
    let mut block = [0.0f32; 512];
    t.render_stereo(&mut block);
    WATCHED.with(|w| w.set(true));
    for k in 0..40 {
        t.set_listener(Listener {
            azimuth: 9.0 * k as f32,
            heard_azimuth: -4.0 * k as f32,
            ..Listener::FRONT
        });
        if k == 20 {
            t.set_stereo(Stereo::Pair);
        }
        t.render_stereo(&mut block);
    }
    WATCHED.with(|w| w.set(false));
    for k in 0..20 {
        stream
            .push(Event::Listener(Listener {
                azimuth: 18.0 * k as f32,
                ..Listener::FRONT
            }))
            .unwrap();
        stream
            .push(Event::Stereo(if k % 2 == 0 {
                Stereo::Binaural
            } else {
                Stereo::Pair
            }))
            .unwrap();
        WATCHED.with(|w| w.set(true));
        player.render_stereo(&mut t, &mut block);
        WATCHED.with(|w| w.set(false));
        assert!(block.iter().all(|x| x.is_finite()));
    }
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
}
