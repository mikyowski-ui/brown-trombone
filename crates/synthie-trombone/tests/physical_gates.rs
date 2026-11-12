//! Stage gates of the physical engine, measured through the public API.
//!
//! Each test states the human-voice figure it is held to. The tube itself
//! (resonances, losses, walls, side cavities) is tested next to the code in
//! `src/phys/tests.rs`.

#![allow(
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use synthie_trombone::{
    Constriction, Engine, GlottalSource, Phoneme, Preset, SLOT_MAIN, Talker, Trombone,
};

const SR: f32 = 48_000.0;

fn voice(hz: f32, tenseness: f32, length: f32) -> Trombone {
    let mut t = Trombone::new(SR);
    t.set_engine(Engine::Physical);
    t.set_tract_length(length);
    t.articulate_phoneme(Phoneme::Aa);
    t.settle();
    t.set_vibrato(0.0, 0.0);
    t.set_jitter(0.0);
    t.set_glide(f32::INFINITY);
    t.set_articulation_speed(2.2);
    t.set_voicing_rates(40.0, 40.0);
    t.set_frequency(hz);
    t.set_tenseness(tenseness);
    t.note_on();
    t
}

fn render(t: &mut Trombone, seconds: f32) -> Vec<f32> {
    let mut x = vec![0.0f32; (seconds * SR) as usize];
    t.render(&mut x);
    x
}

/// Render while recording the glottal flow.
fn flows(t: &mut Trombone, seconds: f32) -> Vec<f32> {
    (0..(seconds * SR) as usize)
        .map(|_| {
            t.process();
            t.gauges().glottal_flow
        })
        .collect()
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

fn db(x: f32) -> f32 {
    20.0 * x.max(1e-12).log10()
}

/// Pitch from the glottal flow: each rise through the middle of its swing
/// counts once.
fn pitch(flow: &[f32]) -> f32 {
    let peak = flow.iter().fold(0.0f32, |m, &v| m.max(v));
    let floor = flow.iter().fold(f32::INFINITY, |m, &v| m.min(v));
    let (hi, lo) = (floor + 0.5 * (peak - floor), floor + 0.25 * (peak - floor));
    let mut rises = Vec::new();
    let mut armed = false;
    for k in 1..flow.len() {
        if flow[k] < lo {
            armed = true;
        }
        if armed && flow[k - 1] <= hi && flow[k] > hi {
            rises.push(k as f32 - 1.0 + (hi - flow[k - 1]) / (flow[k] - flow[k - 1]));
            armed = false;
        }
    }
    if rises.len() < 3 {
        return 0.0;
    }
    (rises.len() - 1) as f32 * SR / (rises[rises.len() - 1] - rises[0])
}

/// Magnitude of `x` at `hz`.
fn tone(x: &[f32], hz: f32) -> f32 {
    let w = 2.0 * std::f64::consts::PI * f64::from(hz) / f64::from(SR);
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (n, &v) in x.iter().enumerate() {
        re += f64::from(v) * (w * n as f64).cos();
        im += f64::from(v) * (w * n as f64).sin();
    }
    (re * re + im * im).sqrt() as f32
}

/// Spectral centre of gravity between 500 Hz and 12 kHz.
fn centroid(x: &[f32]) -> f32 {
    let (mut num, mut den) = (0.0f32, 0.0f32);
    let mut hz = 500.0;
    while hz < 12_000.0 {
        let p = tone(x, hz).powi(2);
        num += p * hz;
        den += p;
        hz += 100.0;
    }
    num / den.max(1e-20)
}

#[test]
fn folds_sing_the_requested_note() {
    // Every preset's pitch and tract length, plus notes around them:
    // within 10 cents once settled.
    for (hz, length) in [
        (62.0, 1.4),
        (95.0, 0.95),
        (105.0, 1.0),
        (118.0, 1.02),
        (140.0, 0.9),
        (165.0, 1.0),
        (210.0, 0.86),
        (220.0, 0.84),
        (290.0, 0.72),
        (360.0, 0.6),
    ] {
        let mut t = voice(hz, 0.65, length);
        assert_eq!(t.glottal_source(), GlottalSource::Folds);
        render(&mut t, 0.8);
        let f = pitch(&flows(&mut t, 0.5));
        let cents = 1200.0 * (f / hz).log2();
        println!("{hz} Hz at length {length}: {f:.1} Hz ({cents:+.1} cents)");
        assert!(cents.abs() < 10.0, "{hz} Hz came out as {f} Hz");
    }
}

#[test]
fn folds_start_quickly_and_near_the_note() {
    // Voice onset in people takes a few cycles. By 80 ms after the gate
    // the folds must be vibrating fully and within a semitone.
    for hz in [105.0, 160.0, 220.0] {
        let mut t = voice(hz, 0.65, 1.0);
        render(&mut t, 0.08);
        let f = pitch(&flows(&mut t, 0.06));
        let cents = 1200.0 * (f / hz).log2();
        println!("{hz} Hz: {f:.1} Hz 80-140 ms after onset ({cents:+.0} cents)");
        assert!(f > 0.0, "{hz} Hz: not vibrating 80 ms after onset");
        assert!(cents.abs() < 100.0, "{hz} Hz starts {cents} cents off");
    }
}

#[test]
fn flow_through_the_glottis_is_humanlike() {
    // Modal voice in men: mean flow 100-300 cm³/s, peak a few hundred,
    // the folds closed for part of each cycle.
    let mut t = voice(110.0, 0.65, 1.0);
    render(&mut t, 0.6);
    let f = flows(&mut t, 0.5);
    let mean = f.iter().sum::<f32>() / f.len() as f32;
    let peak = f.iter().fold(0.0f32, |m, &v| m.max(v));
    let floor = f.iter().fold(f32::INFINITY, |m, &v| m.min(v));
    let open = f
        .iter()
        .filter(|&&v| v > floor + 0.1 * (peak - floor))
        .count() as f32
        / f.len() as f32;
    println!("mean {mean:.0}, peak {peak:.0}, floor {floor:.0} cm3/s, open quotient {open:.2}");
    assert!((100.0..300.0).contains(&mean), "mean flow {mean}");
    assert!((250.0..900.0).contains(&peak), "peak flow {peak}");
    assert!(floor < 20.0, "folds never close: {floor}");
    assert!((0.4..0.8).contains(&open), "open quotient {open}");
}

#[test]
fn folds_stop_below_a_threshold_pressure() {
    // People need 2-5 cmH2O of lung pressure to keep the folds going.
    let vibrating = |pressure: f32| {
        let mut t = voice(110.0, 0.65, 1.0);
        t.set_lung_pressure(pressure);
        render(&mut t, 0.8);
        let f = flows(&mut t, 0.3);
        let peak = f.iter().fold(0.0f32, |m, &v| m.max(v));
        let floor = f.iter().fold(f32::INFINITY, |m, &v| m.min(v));
        peak - floor > 0.5 * peak
    };
    let (mut lo, mut hi) = (1.0f32, 8.0f32);
    assert!(!vibrating(lo), "vibrating at {lo} cmH2O");
    assert!(vibrating(hi), "not vibrating at {hi} cmH2O");
    for _ in 0..6 {
        let mid = f32::midpoint(lo, hi);
        if vibrating(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    println!("phonation threshold pressure about {hi:.1} cmH2O");
    assert!((2.0..5.5).contains(&hi), "threshold {hi} cmH2O");
}

#[test]
fn slacker_folds_sound_breathier() {
    // The first harmonic stands further above the second as the voice
    // goes from pressed to breathy (H1-H2), and more air escapes.
    let reading = |tense: f32| {
        let mut t = voice(110.0, tense, 1.0);
        render(&mut t, 0.7);
        let f = flows(&mut t, 0.5);
        let mean = f.iter().sum::<f32>() / f.len() as f32;
        let f0 = pitch(&f);
        (db(tone(&f, f0)) - db(tone(&f, 2.0 * f0)), mean)
    };
    let pressed = reading(0.95);
    let modal = reading(0.65);
    let breathy = reading(0.4);
    println!("H1-H2 / mean flow: pressed {pressed:?}, modal {modal:?}, breathy {breathy:?}");
    assert!(modal.0 > pressed.0 && breathy.0 > modal.0);
    assert!(modal.1 > pressed.1 && breathy.1 > modal.1);
}

#[test]
fn a_whisper_has_no_voice() {
    let mut t = voice(110.0, 0.0, 1.0);
    render(&mut t, 0.5);
    let f = flows(&mut t, 0.3);
    let peak = f.iter().fold(0.0f32, |m, &v| m.max(v));
    let floor = f.iter().fold(f32::INFINITY, |m, &v| m.min(v));
    assert!(floor > 200.0, "little air in a whisper: {floor}");
    assert!(peak - floor < 0.2 * peak, "a whisper is vibrating");
    assert!(rms(&render(&mut t, 0.3)) > 1e-4, "a whisper is silent");
}

/// Level and centroid of a sustained voiceless fricative.
fn fricative(p: Phoneme) -> (f32, f32) {
    let mut t = Trombone::new(SR);
    t.set_engine(Engine::Physical);
    t.articulate_phoneme(Phoneme::Aa);
    t.articulate_phoneme(p);
    render(&mut t, 0.3);
    let x = render(&mut t, 0.3);
    (rms(&x), centroid(&x))
}

#[test]
fn sibilants_are_loud_and_the_others_faint() {
    // /s/ and /sh/ sit 10-20 dB under a vowel; /f/ and /th/ another
    // 10-20 dB down. /s/ is brighter than /sh/.
    let vowel = {
        let mut t = voice(110.0, 0.65, 1.0);
        render(&mut t, 0.6);
        rms(&render(&mut t, 0.4))
    };
    let (s, s_hz) = fricative(Phoneme::S);
    let (sh, sh_hz) = fricative(Phoneme::Sh);
    let (f, _) = fricative(Phoneme::F);
    let (th, _) = fricative(Phoneme::Th);
    println!(
        "re vowel: s {:.1} dB ({s_hz:.0} Hz), sh {:.1} dB ({sh_hz:.0} Hz), f {:.1} dB, th {:.1} dB",
        db(s / vowel),
        db(sh / vowel),
        db(f / vowel),
        db(th / vowel)
    );
    assert!(
        (-22.0..-4.0).contains(&db(s / vowel)),
        "s {}",
        db(s / vowel)
    );
    assert!(
        (-22.0..-4.0).contains(&db(sh / vowel)),
        "sh {}",
        db(sh / vowel)
    );
    assert!(db(s / f) >= 10.0 && db(s / th) >= 8.0);
    assert!(db(f / vowel) < -20.0 && db(th / vowel) < -18.0);
    assert!(s_hz > sh_hz + 1500.0, "s {s_hz} Hz, sh {sh_hz} Hz");
}

#[test]
fn nothing_hisses_without_air() {
    // The classic engine hisses wherever a finger narrows the tract. Here
    // a narrowing with no breath behind it is silent.
    let mut t = Trombone::new(SR);
    t.set_engine(Engine::Physical);
    t.set_constriction(SLOT_MAIN, Some(Constriction::new(37.0, 0.45)));
    render(&mut t, 0.3);
    t.set_constriction(SLOT_MAIN, None);
    render(&mut t, 0.5);
    assert!(rms(&render(&mut t, 0.2)) < 1e-4);
}

/// What happens through a stop between two vowels.
struct Stop {
    /// Peak pressure behind the closure, cmH2O.
    pressure: f32,
    /// How long the folds keep vibrating once the mouth is shut, ms.
    voicing: f32,
    /// Loudest sample in the 30 ms after the release.
    burst: f32,
    /// Level just before the release.
    silence: f32,
    /// When the voice is back at half the vowel's level, ms after release.
    onset: f32,
}

fn stop(p: Phoneme, voiced: bool) -> Stop {
    let mut t = voice(110.0, 0.65, 1.0);
    render(&mut t, 0.5);
    let vowel = rms(&render(&mut t, 0.1));
    t.articulate_phoneme(p);
    t.set_voiced(voiced);
    let mut shut_at = None;
    let (mut voicing, mut pressure) = (0.0f32, 0.0f32);
    let mut last = Vec::new();
    for k in 0..24 {
        let (mut lo, mut hi, mut lips) = (f32::INFINITY, 0.0f32, 0.0f32);
        last.clear();
        for _ in 0..240 {
            last.push(t.process());
            let g = t.gauges();
            lo = lo.min(g.glottal_area);
            hi = hi.max(g.glottal_area);
            lips = lips.max(g.lip_flow.abs());
        }
        let ms = (k + 1) as f32 * 5.0;
        if shut_at.is_none() && lips < 20.0 {
            shut_at = Some(ms);
        }
        if let Some(at) = shut_at
            && hi - lo > 0.02
        {
            voicing = ms - at;
        }
        pressure = pressure.max(t.pressure(30));
    }
    let silence = rms(&last);
    t.articulate_phoneme(Phoneme::Aa);
    t.set_voiced(true);
    let after = render(&mut t, 0.2);
    let burst = after[..1440].iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let onset = after
        .chunks(240)
        .position(|c| rms(c) > 0.5 * vowel)
        .map_or(f32::INFINITY, |k| k as f32 * 5.0);
    Stop {
        pressure,
        voicing,
        burst,
        silence,
        onset,
    }
}

#[test]
fn stops_build_pressure_and_burst_by_themselves() {
    let p = stop(Phoneme::P, false);
    let b = stop(Phoneme::B, true);
    let d = stop(Phoneme::D, true);
    for (name, s) in [("p", &p), ("b", &b), ("d", &d)] {
        println!(
            "{name}: pressure {:.1} cmH2O, voicing {:.0} ms into closure, burst {:.3} over {:.5}, voice back after {:.0} ms",
            s.pressure, s.voicing, s.burst, s.silence, s.onset
        );
    }
    // Mouth pressure reaches at least 80 % of the 8 cmH2O in the lungs.
    assert!(p.pressure > 6.4, "/p/ mouth pressure {}", p.pressure);
    // Nothing is scripted, yet the release makes a burst.
    assert!(
        p.burst > 20.0 * p.silence.max(1e-5),
        "/p/ burst {}",
        p.burst
    );
    // A voiceless stop has no voicing in the closure; a voiced one keeps
    // it for 30 ms or more before the mouth fills.
    assert!(p.voicing < 15.0, "/p/ voiced for {} ms", p.voicing);
    assert!(b.voicing >= 30.0, "/b/ voiced for only {} ms", b.voicing);
    // Less room behind the tongue tip than behind the lips: /d/ loses its
    // voice sooner than /b/.
    assert!(d.voicing <= b.voicing);
    // The voice comes back later after /p/ than after /b/.
    assert!(p.onset > b.onset, "/p/ {} ms, /b/ {} ms", p.onset, b.onset);
}

#[test]
fn every_preset_speaks_on_the_physical_engine() {
    let text = "Hello there, how are you? Pink trombone, 42.";
    for p in Preset::ALL {
        let settings = p.settings().with_engine(Engine::Physical);
        let out = Talker::new(22_050.0, settings).bake(text);
        let peak = out.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        println!("{p}: rms {:.4}, peak {peak:.3}", rms(&out));
        assert!(out.iter().all(|s| s.is_finite()), "{p}");
        assert!(rms(&out) > 0.002, "{p} too quiet: {}", rms(&out));
        assert!(peak < 2.0, "{p} too loud: {peak}");
    }
}

#[test]
fn speech_ends_in_silence() {
    // Breath stops with the sentence: no hiss in the tail.
    let settings = Preset::Deep.settings().with_engine(Engine::Physical);
    let out = Talker::new(SR, settings).bake("Stay a while, and listen.");
    let tail = &out[out.len() - (0.1 * SR) as usize..];
    let body = rms(&out[..out.len() / 2]);
    println!("tail {:.1} dB re speech", db(rms(tail) / body));
    assert!(rms(tail) < 0.003 * body, "tail {} vs {}", rms(tail), body);
}

#[test]
fn same_seed_same_audio_on_the_physical_engine() {
    let make = || {
        let mut t = Trombone::with_seed(44_100.0, 42);
        t.set_engine(Engine::Physical);
        t.articulate_phoneme(Phoneme::S);
        let mut x = render(&mut t, 0.2);
        t.articulate_phoneme(Phoneme::Iy);
        t.note_on();
        x.extend(render(&mut t, 0.3));
        x
    };
    assert_eq!(make(), make());
}

#[test]
fn garbage_and_abuse_never_break_the_physical_engine() {
    let mut seed = 0x00C0_FFEEu32;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed >> 8) as f32 / 16_777_216.0
    };
    let mut t = Trombone::new(SR);
    t.set_engine(Engine::Physical);
    t.set_frequency(f32::NAN);
    t.set_tongue(f32::INFINITY, 2.0);
    t.set_constriction(0, Some(Constriction::new(f32::NAN, f32::NAN)));
    t.set_lung_pressure(f32::NAN);
    let mut buf = vec![0.0f32; 2400];
    for step in 0..400 {
        match (rnd() * 9.0) as u32 {
            0 => t.set_frequency(30.0 + 1500.0 * rnd()),
            1 => t.set_tenseness(rnd()),
            2 => t.set_tongue(12.0 + 17.0 * rnd(), 2.05 + 1.45 * rnd()),
            3 => {
                let slot = (rnd() * 4.0) as usize;
                let c = (rnd() > 0.3).then(|| Constriction::new(2.0 + 42.0 * rnd(), 1.5 * rnd()));
                t.set_constriction(slot, c);
            }
            4 => t.set_velum(rnd()),
            5 => t.set_voiced(rnd() > 0.4),
            6 => t.set_tract_length(0.5 + 1.1 * rnd()),
            7 => t.set_lung_pressure(1.0 + 30.0 * rnd()),
            _ => t.articulate_phoneme(Phoneme::ALL[(rnd() * Phoneme::ALL.len() as f32) as usize]),
        }
        t.render(&mut buf);
        let peak = buf.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(buf.iter().all(|s| s.is_finite()), "step {step}");
        assert!(peak < 8.0, "step {step}: peak {peak}");
    }
}
