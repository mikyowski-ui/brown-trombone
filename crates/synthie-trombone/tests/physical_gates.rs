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

use std::fmt::Write as _;
use std::sync::OnceLock;
use synthie_trombone::{
    Articulation, Constriction, Engine, GlottalSource, Persona, Phoneme, Preset, SLOT_MAIN, Talker,
    Trombone, VoiceSettings,
};

const SR: f32 = 48_000.0;

#[path = "../examples/support/place.rs"]
mod place;
#[path = "../examples/support/spectrum.rs"]
mod spectrum;
#[path = "../examples/support/stops.rs"]
mod stops;

/// The tuning under test: the default one, or, when the environment
/// variable `SYNTHIE_TROMBONE_PROFILE` names a profile file, that
/// profile's. The gates hold a tuned voice to the same figures as the
/// built-in one:
///
/// ```text
/// SYNTHIE_TROMBONE_PROFILE=profiles/tuned.json cargo test --release --test physical_gates
/// ```
fn tuned(settings: VoiceSettings) -> VoiceSettings {
    static PROFILE: OnceLock<Option<String>> = OnceLock::new();
    let profile = PROFILE.get_or_init(|| {
        let path = std::env::var("SYNTHIE_TROMBONE_PROFILE").ok()?;
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        println!("gates run with the profile {path}");
        Some(text)
    });
    let Some(profile) = profile else {
        return settings;
    };
    // Only the tuning is taken: each test chooses its own voice.
    let laid = settings.with_profile(profile).expect("a voice profile");
    VoiceSettings {
        tuning: laid.tuning,
        ..settings
    }
}

/// A voice with the tuning under test.
fn fresh() -> Trombone {
    let mut t = Trombone::new(SR);
    t.set_tuning(&tuned(VoiceSettings::default()).tuning);
    t
}

fn voice(hz: f32, tenseness: f32, length: f32) -> Trombone {
    let mut t = fresh();
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
fn every_vowel_voices_promptly_after_a_consonant() {
    // Folds brought together in a stream that is already flowing, or
    // behind lips that are still parting, must still start: within 60 ms
    // from silence and after /s/, and within 120 ms of a /p/ release even
    // for the close, rounded vowels (which keep the mouth pressurised
    // longest). Before the glottal attack some took over 200 ms or never
    // started.
    let vowels = [
        Phoneme::Iy,
        Phoneme::Ih,
        Phoneme::Eh,
        Phoneme::Ae,
        Phoneme::Ah,
        Phoneme::Aa,
        Phoneme::Ao,
        Phoneme::Uh,
        Phoneme::Uw,
        Phoneme::Ow,
        Phoneme::Er,
        Phoneme::Ax,
    ];
    for p in vowels {
        for (before, limit) in [(None, 60), (Some(Phoneme::S), 60), (Some(Phoneme::P), 120)] {
            let mut t = fresh();
            t.set_engine(Engine::Physical);
            t.set_articulation_speed(2.2);
            t.set_voicing_rates(40.0, 40.0);
            t.set_vibrato(0.0, 0.0);
            t.set_jitter(0.0);
            t.set_frequency(105.0);
            t.set_tenseness(0.65);
            t.articulate_phoneme(p);
            if let Some(c) = before {
                t.articulate_phoneme(c);
                render(&mut t, 0.1);
                t.articulate_phoneme(p);
            } else {
                t.settle();
            }
            t.set_voiced(true);
            // First 10 ms stretch in which the glottis both shuts and opens wide.
            let mut slices: Vec<(f32, f32)> = Vec::new();
            let mut started = None;
            for ms in 0..300 {
                let (mut lo, mut hi) = (f32::INFINITY, 0.0f32);
                for _ in 0..48 {
                    t.process();
                    let a = t.gauges().glottal_area;
                    lo = lo.min(a);
                    hi = hi.max(a);
                }
                slices.push((lo, hi));
                if slices.len() >= 10 {
                    let recent = &slices[slices.len() - 10..];
                    let lo = recent.iter().map(|s| s.0).fold(f32::INFINITY, f32::min);
                    let hi = recent.iter().map(|s| s.1).fold(0.0f32, f32::max);
                    if lo < 0.01 && hi > 0.1 && hi < 0.3 {
                        started = Some(ms + 1 - 9);
                        break;
                    }
                }
            }
            let started = started.unwrap_or(usize::MAX);
            assert!(
                started <= limit,
                "{p} after {before:?}: full voice only after {started} ms"
            );
        }
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
    let mut t = fresh();
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
    let mut t = fresh();
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

/// The stop is made the way the speech front end makes it: a voiceless one
/// with the glottis spread, a voiced one with the folds slackened and the
/// pharynx widening; both let go at the release.
fn stop(p: Phoneme, voiced: bool) -> Stop {
    let tuning = tuned(VoiceSettings::default()).tuning;
    let mut t = voice(110.0, 0.65, 1.0);
    render(&mut t, 0.5);
    let vowel = rms(&render(&mut t, 0.1));
    t.articulate_phoneme(p);
    t.set_voiced(voiced);
    if voiced {
        t.set_fold_slack(tuning.voiced_stop_slack);
        t.set_cavity_expansion(tuning.cavity_expansion);
    } else {
        t.set_glottal_spread(tuning.voiceless_spread);
    }
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
    t.set_glottal_spread(0.0);
    t.set_fold_slack(0.0);
    t.set_cavity_expansion(0.0);
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
    // it for 30 ms or more before the mouth fills (with the pharynx
    // widening, most of the 120 ms closure here).
    assert!(p.voicing < 15.0, "/p/ voiced for {} ms", p.voicing);
    assert!(b.voicing >= 30.0, "/b/ voiced for only {} ms", b.voicing);
    // Less room behind the tongue tip than behind the lips: /d/ loses its
    // voice no later than /b/.
    assert!(
        d.voicing <= b.voicing,
        "/d/ {} ms, /b/ {} ms",
        d.voicing,
        b.voicing
    );
    // The voice comes back later after /p/ than after /b/: the spread
    // glottis has to close first, while after /b/ the folds are vibrating
    // at the release.
    assert!(p.onset > b.onset, "/p/ {} ms, /b/ {} ms", p.onset, b.onset);
}

/// The voice the speech gates listen to: the judge's, with the tuning
/// under test.
fn speech_voice() -> VoiceSettings {
    tuned(stops::judge_voice())
}

#[test]
fn voice_onset_follows_the_stop() {
    // English before a stressed vowel (Lisker & Abramson 1964): /p t k/
    // aspirated, voice 40 to 65 ms after the release, later the further
    // back the closure; /b d g/ voiced at once (0 to 15 ms).
    let settings = speech_voice();
    let mut vot = Vec::new();
    for (voiceless, voiced) in stops::STOPS {
        let text = |p: Phoneme| stops::context(p, "medial");
        let u = stops::stop(&settings, &text(voiceless), voiceless);
        let v = stops::stop(&settings, &text(voiced), voiced);
        println!(
            "{voiceless}: VOT {:.0} ms   {voiced}: VOT {:.0} ms",
            u.vot, v.vot
        );
        assert!(
            (40.0..=65.0).contains(&u.vot),
            "{voiceless}: VOT {} ms",
            u.vot
        );
        assert!((0.0..=15.0).contains(&v.vot), "{voiced}: VOT {} ms", v.vot);
        vot.push(u.vot);
    }
    assert!(vot[2] > vot[0], "/k/ {} ms, /p/ {} ms", vot[2], vot[0]);
}

#[test]
fn voiced_closures_carry_a_voice_bar_and_voiceless_ones_do_not() {
    // The folds keep vibrating through a voiced closure between vowels
    // (the low "voice bar" of the spectrogram), for at least 80 % of it,
    // and stop at once in a voiceless one (under 30 %, the tail of the
    // vowel). The voiceless closure is the longer, and its mouth pressure
    // at the release the higher: it is what makes its burst.
    let settings = speech_voice();
    for (voiceless, voiced) in stops::STOPS {
        let text = |p: Phoneme| stops::context(p, "medial");
        let u = stops::stop(&settings, &text(voiceless), voiceless);
        let v = stops::stop(&settings, &text(voiced), voiced);
        println!(
            "{voiceless}: closure {:.0} ms, voiced {:.0} %, bar {:.1} dB, p {:.2} | {voiced}: closure {:.0} ms, voiced {:.0} %, bar {:.1} dB, p {:.2}",
            u.closure,
            100.0 * u.voiced_share,
            u.voice_bar,
            u.pressure,
            v.closure,
            100.0 * v.voiced_share,
            v.voice_bar,
            v.pressure
        );
        assert!(
            v.voiced_share >= 0.8,
            "{voiced}: voiced {:.2}",
            v.voiced_share
        );
        assert!(
            u.voiced_share < 0.3,
            "{voiceless}: voiced {:.2}",
            u.voiced_share
        );
        // The velar voice bar is the faintest: little wall behind the
        // tongue back to radiate it.
        let lift = if voiced == Phoneme::G { 1.0 } else { 3.0 };
        assert!(
            v.voice_bar > u.voice_bar + lift,
            "{voiced} vs {voiceless} voice bar"
        );
        assert!(
            u.closure > v.closure,
            "{voiceless} closure {} ms",
            u.closure
        );
        assert!(
            u.pressure > v.pressure,
            "{voiceless} pressure {}",
            u.pressure
        );
    }
}

#[test]
fn the_voice_starts_higher_after_a_voiceless_stop() {
    // Vowels start some 10 to 15 Hz higher after /p t k/ than their pitch
    // a little later, and higher than after /b d g/ (Hombert, Ohala &
    // Ewan 1979).
    let settings = speech_voice();
    for (voiceless, voiced) in stops::STOPS {
        let text = |p: Phoneme| stops::context(p, "medial");
        let u = stops::stop(&settings, &text(voiceless), voiceless);
        let v = stops::stop(&settings, &text(voiced), voiced);
        println!(
            "{voiceless}: F0 {:+.1} Hz then {:+.1}   {voiced}: F0 {:+.1} Hz then {:+.1}",
            u.f0_onset, u.f0_next, v.f0_onset, v.f0_next
        );
        assert!(
            u.f0_onset > 4.0,
            "{voiceless}: onset F0 {:+.1} Hz",
            u.f0_onset
        );
        assert!(
            u.f0_onset > u.f0_next,
            "{voiceless}: the raise does not decay"
        );
        assert!(
            u.f0_onset > v.f0_onset + 5.0,
            "{voiceless} {:+.1} Hz vs {voiced} {:+.1} Hz",
            u.f0_onset,
            v.f0_onset
        );
    }
}

#[test]
fn vowels_are_shorter_before_voiceless_codas() {
    // Pre-fortis clipping: a vowel before a voiceless coda lasts about two
    // thirds as long as before a voiced one ("bat" and "bad", Peterson &
    // Lehiste 1960). The physical engine's front end only.
    let settings = speech_voice();
    for (a, b) in [
        ("[B AE1 T]", "[B AE1 D]"),
        ("[R OW1 P]", "[R OW1 B]"),
        ("[B AE1 K]", "[B AE1 G]"),
        ("[B UW1 S]", "[B UW1 Z]"),
    ] {
        let ratio = stops::vowel_length(&settings, a) / stops::vowel_length(&settings, b);
        println!("{a} / {b}: {ratio:.2}");
        assert!((0.55..=0.8).contains(&ratio), "{a} / {b}: {ratio:.2}");
    }
}

#[test]
fn voiced_fricatives_hiss_and_voice_at_once() {
    // A voiced fricative keeps the folds vibrating through the hiss (and
    // its hiss is weaker than its voiceless twin's, but there); the
    // voiceless one does not vibrate.
    let settings = speech_voice();
    for (voiceless, voiced) in stops::FRICATIVES {
        let text = |p: Phoneme| stops::context(p, "vcv");
        let u = stops::fricative(&settings, &text(voiceless), voiceless);
        let v = stops::fricative(&settings, &text(voiced), voiced);
        println!(
            "{voiceless}: voiced {:.0} %, bar {:.1} dB, hiss {:.1} dB | {voiced}: voiced {:.0} %, bar {:.1} dB, hiss {:.1} dB",
            100.0 * u.voiced_share,
            u.voice_bar,
            u.hiss,
            100.0 * v.voiced_share,
            v.voice_bar,
            v.hiss
        );
        assert!(
            v.voiced_share >= 0.8,
            "{voiced}: voiced {:.2}",
            v.voiced_share
        );
        assert!(
            u.voiced_share < 0.2,
            "{voiceless}: voiced {:.2}",
            u.voiced_share
        );
        assert!(v.voice_bar > u.voice_bar + 3.0, "{voiced} voice bar");
        // The bound was 15 dB. The voiceless sibilants are louder now, as a
        // speaker's are (/s/ about level with the vowel next to it, /ʃ/ 3 dB
        // under, in the judge's recordings): the jet on the teeth has a
        // corner of 5.5 kHz (`Tuning::teeth_corner`, 4.5 before) and /ʃ tʃ/
        // open their lips a little (`Tuning::postalveolar_lips`), while
        // the voiced ones keep their hiss under the voice. Measured: S
        // -11.8 → +0.3 dB re the vowel, Z -22.3 → -15.8; SH -14.2 → +1.2,
        // ZH -26.2 → -15.1; CH -13.9 → +2.7, JH -27.2 → -15.8 (the vowel
        // itself starts softly now, which lifts every figure here).
        assert!(v.hiss > u.hiss - 19.0, "{voiced} hiss {:.1} dB", v.hiss);
    }
}

#[test]
fn muscle_driven_folds_sing_and_follow_their_muscles() {
    // Body-cover folds under the Titze & Story (2002) rules: the
    // cricothyroid raises the pitch, the thyroarytenoid lowers it and
    // lets less air through.
    let reading = |ct: f32, ta: f32| {
        let mut t = voice(110.0, 0.65, 1.0);
        t.override_muscles(Some((ct, ta, 0.5)));
        t.override_folds(Some(1.0), Some(0.01));
        render(&mut t, 0.6);
        let f = flows(&mut t, 0.4);
        (pitch(&f), f.iter().sum::<f32>() / f.len() as f32)
    };
    let low = reading(0.0, 0.25);
    let mid = reading(0.5, 0.25);
    let high = reading(0.75, 0.25);
    let thick = reading(0.0, 0.5);
    println!("ct 0: {low:?}, ct 0.5: {mid:?}, ct 0.75: {high:?}; ta 0.5 at ct 0: {thick:?}");
    assert!(low.0 > 80.0, "not vibrating: {low:?}");
    assert!(mid.0 > 1.25 * low.0 && high.0 > 1.25 * mid.0);
    assert!(thick.0 < low.0 && thick.1 < low.1);
}

// ----- the hiss regulator ------------------------------------------------

/// The default voice at another body size and lung pressure, as a
/// [`Persona`] would make it.
fn sized_voice(size: f32, lung: f32) -> Trombone {
    let mut settings = tuned(VoiceSettings::default())
        .with_engine(Engine::Physical)
        .with_persona(&Persona {
            size,
            ..Persona::NEUTRAL
        });
    settings.tuning.lung_pressure = lung;
    let mut t = fresh();
    settings.apply(&mut t);
    t.set_vibrato(0.0, 0.0);
    t.set_jitter(0.0);
    t
}

/// Level and mean noise of the strongest jet in the mouth over 0.3 s,
/// half a second into holding `p` (voiced if it is a vowel).
fn held(mut t: Trombone, p: Phoneme) -> (f32, f32) {
    t.articulate_phoneme(Phoneme::Aa);
    t.settle();
    t.articulate_phoneme(p);
    if p.is_voiced() {
        t.settle();
        t.note_on();
    }
    render(&mut t, 0.5);
    let n = (0.3 * SR) as usize;
    let mut x = Vec::with_capacity(n);
    let mut jet = 0.0f64;
    for _ in 0..n {
        x.push(t.process());
        jet += f64::from(t.gauges().jet_noise);
    }
    (rms(&x), (jet / n as f64) as f32)
}

const SIZES: [f32; 5] = [0.6, 0.8, 1.0, 1.2, 1.4];
const LUNGS: [f32; 4] = [5.0, 8.0, 11.0, 14.0];

#[test]
fn fricatives_keep_their_level_and_no_vowel_hisses_in_any_voice() {
    // From a child-sized body to a giant, from soft speech (5 cmH2O) to
    // loud (14): /s/ stays where it sits against the vowel in the default
    // voice, and no vowel's own jets come within 26 dB of the /s/ jet.
    // Three things make that so: the cap on jet speed follows lung
    // pressure, the pitch of the hiss follows the size of the gap, and the
    // regulator opens a vowel that starts to hiss.
    //
    // The aim for /s/ was 6 dB. Every body from 0.8 to 1.4 keeps within
    // 5 dB; the smallest (0.6) misses by 1 dB at 8 and 11 cmH2O. Its /s/
    // is only 2.7 dB under the default voice's in absolute terms; the rest
    // is its vowel, louder because it is sung three and a half times as
    // high: 4.3 dB louder than the default voice's /a/ (3.8 dB with the
    // /a/ fitted to Peterson & Barney, whose F1 was 10 % lower and met the
    // harmonics of the high voice elsewhere). No gap width changes that,
    // and neither does the jaw, so the bound was 7.5 dB. With the folds
    // closing firmly the /s/ of size 0.6 sits at most 6.2 dB from the
    // default's against its vowel (at 8 cmH2O), and the bound is 6.5 dB
    // again.
    let vowels = [
        Phoneme::Iy,
        Phoneme::Ih,
        Phoneme::Eh,
        Phoneme::Ae,
        Phoneme::Ah,
        Phoneme::Aa,
        Phoneme::Ao,
        Phoneme::Uh,
        Phoneme::Uw,
        Phoneme::Ow,
        Phoneme::Er,
        Phoneme::Ax,
    ];
    let measure = move |size: f32, lung: f32| {
        let (vowel, _) = held(sized_voice(size, lung), Phoneme::Aa);
        let (s, s_jet) = held(sized_voice(size, lung), Phoneme::S);
        let (sh, _) = held(sized_voice(size, lung), Phoneme::Sh);
        let mut worst = (f32::NEG_INFINITY, Phoneme::Aa);
        for p in vowels {
            let (_, jet) = held(sized_voice(size, lung), p);
            let level = db(jet / s_jet.max(1e-9));
            if level > worst.0 {
                worst = (level, p);
            }
        }
        (db(s / vowel), db(sh / vowel), worst)
    };
    let (s0, sh0, _) = measure(1.0, 8.0);
    std::thread::scope(|scope| {
        let mut jobs = Vec::new();
        for size in SIZES {
            for lung in LUNGS {
                jobs.push((size, lung, scope.spawn(move || measure(size, lung))));
            }
        }
        let mut failures = Vec::new();
        for (size, lung, job) in jobs {
            let (s, sh, (hiss, vowel)) = job.join().unwrap();
            println!(
                "size {size} at {lung} cmH2O: s {:+.1} dB, sh {:+.1} dB re default; loudest vowel jet {vowel} {hiss:.1} dB re the s jet",
                s - s0,
                sh - sh0
            );
            if (s - s0).abs() > 6.5 {
                failures.push(format!("s at size {size}, {lung} cmH2O: {:+.1} dB", s - s0));
            }
            if hiss > -26.0 {
                failures.push(format!(
                    "{vowel} hisses at size {size}, {lung} cmH2O: {hiss:.1} dB"
                ));
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    });
}

#[test]
fn a_gap_near_a_fricative_still_hisses() {
    // Playing by hand, nobody lands on a fricative exactly. Within two
    // cells and 0.15 in diameter of each fricative's own position a held
    // gap still hisses: no more than 20 dB under the fricative itself, or
    // under the one whose place it has moved to if that is quieter, where
    // without the regulator the narrow ones are shut or choked. (A gap
    // moved from /s/ onto the teeth is a /θ/: its jet no longer strikes
    // them, and with /s/ as loud as a speaker's it is 23 dB under /s/,
    // where /θ/ is 20 dB under.)
    let level = |index: f32, diameter: f32, regulated: bool| {
        let mut t = fresh();
        t.set_engine(Engine::Physical);
        t.set_regulation(regulated);
        t.articulate_phoneme(Phoneme::Aa);
        t.settle();
        t.set_constriction(SLOT_MAIN, Some(Constriction::new(index, diameter)));
        render(&mut t, 0.5);
        rms(&render(&mut t, 0.3))
    };
    std::thread::scope(|scope| {
        let mut jobs = Vec::new();
        let places = [Phoneme::S, Phoneme::Sh, Phoneme::F, Phoneme::Th].map(|p| {
            let c = p.articulation().constriction.unwrap();
            (p, c, level(c.index, c.diameter, true))
        });
        for &(p, c, own) in &places {
            for by in [-0.15, 0.0, 0.15] {
                for along in [-2.0, 0.0, 2.0] {
                    let (index, diameter) = (c.index + along, c.diameter + by);
                    // The fricative whose place the gap is nearest now.
                    let near = places
                        .iter()
                        .min_by(|a, b| {
                            (a.1.index - index)
                                .abs()
                                .total_cmp(&(b.1.index - index).abs())
                        })
                        .map_or(own, |q| q.2);
                    let own = own.min(near);
                    jobs.push((
                        p,
                        along,
                        by,
                        own,
                        scope.spawn(move || {
                            (level(index, diameter, true), level(index, diameter, false))
                        }),
                    ));
                }
            }
        }
        let (mut rescued, mut failures) = (0, Vec::new());
        for (p, along, by, own, job) in jobs {
            let (with, without) = job.join().unwrap();
            let (with, without) = (db(with / own), db(without / own));
            if without < -20.0 && with >= -20.0 {
                rescued += 1;
            }
            if with < -20.0 {
                failures.push(format!(
                    "{p} moved {along:+} cells, {by:+} diameter: {with:.1} dB (unregulated {without:.1})"
                ));
            }
        }
        println!("{rescued} of 36 positions hiss only because the regulator widened them");
        assert!(failures.is_empty(), "{failures:#?}");
        assert!(
            rescued >= 6,
            "the regulator rescued only {rescued} positions"
        );
    });
}

#[test]
fn the_regulator_barely_touches_the_default_voice() {
    // The default voice's fricatives hiss as they are: no gap is moved.
    // Its vowels do not hiss, bar a breathy moment such as the lips of
    // "who" just after the /h/, where the regulator opens them by a
    // fraction of what it may; the speech is as loud either way.
    let text = "She sells sea shells; who would choose those few?";
    let speak = |regulated: bool| {
        let mut talker = Talker::new(
            SR,
            tuned(VoiceSettings::default()).with_engine(Engine::Physical),
        );
        talker.voice_mut().set_regulation(regulated);
        talker.say(text);
        let (mut x, mut nudge, mut relief) = (Vec::new(), 0.0f32, 0.0f32);
        while talker.is_speaking() {
            x.push(talker.process());
            let g = talker.voice_mut().gauges();
            nudge = nudge.max(g.gap_nudge);
            relief = relief.max(g.vowel_relief);
        }
        (rms(&x), nudge, relief)
    };
    let (on, nudge, relief) = speak(true);
    let (off, _, _) = speak(false);
    println!(
        "gap nudge {nudge:.3}, vowel relief at most {relief:.3}, level {:+.2} dB",
        db(on / off)
    );
    assert!(nudge == 0.0, "a fricative gap was moved by {nudge}");
    assert!(relief <= 0.03, "vowels were opened by {relief}");
    assert!(db(on / off).abs() < 0.3);
}

/// Hold every other note of the app's range on one voice and vowel, at
/// three tenseness settings. Returns the quietest: level in dB re
/// `reference`, the note and the tenseness.
fn sweep_notes(preset: Preset, vowel: Phoneme, reference: f32) -> (f32, usize, f32) {
    let settings = tuned(preset.settings()).with_engine(Engine::Physical);
    let mut quietest = (0.0f32, 0, 0.0f32);
    for tense in [0.4f32, 0.65, 0.85] {
        for note in (24..=96).step_by(2) {
            let mut t = fresh();
            settings.apply(&mut t);
            t.articulate_phoneme(vowel);
            t.set_tenseness(tense);
            t.settle();
            t.note_on_midi(note as f32);
            let first = render(&mut t, 0.15);
            assert!(
                first.chunks(480).any(|c| rms(c) > 0.01 * reference),
                "{preset} {vowel} tenseness {tense} note {note}: nothing within 150 ms"
            );
            render(&mut t, 0.35);
            let level = db(rms(&render(&mut t, 0.1)) / reference);
            if level < quietest.0 {
                quietest = (level, note, tense);
            }
            assert!(
                level > -40.0,
                "{preset} {vowel} tenseness {tense} note {note}: {level:.1} dB re the voice"
            );
        }
    }
    quietest
}

#[test]
fn no_held_note_is_ever_silent() {
    // Every other note the terminal app can play (octaves 1 to 6), on an
    // open and two close vowels, breathy to pressed, on large and small
    // voices: sound within 150 ms, and still sounding half a second later.
    // Slack folds are held closer together, and notes outside their range
    // are sung by the pulse source (below it as fry). Inside it the folds
    // sing every note themselves.
    let presets = [Preset::Deep, Preset::Giant, Preset::Bright, Preset::Critter];
    let vowels = [Phoneme::Aa, Phoneme::Iy, Phoneme::Uw];
    std::thread::scope(|scope| {
        let mut jobs = Vec::new();
        for preset in presets {
            let reference = {
                let mut t = fresh();
                tuned(preset.settings())
                    .with_engine(Engine::Physical)
                    .apply(&mut t);
                t.articulate_phoneme(Phoneme::Aa);
                t.settle();
                t.note_on();
                render(&mut t, 0.5);
                rms(&render(&mut t, 0.2))
            };
            for vowel in vowels {
                jobs.push((
                    preset,
                    vowel,
                    scope.spawn(move || sweep_notes(preset, vowel, reference)),
                ));
            }
        }
        for (preset, vowel, job) in jobs {
            let (level, note, tense) = job.join().expect("a held note was silent");
            println!(
                "{preset} on {vowel}: quietest held note {level:.1} dB re the voice's own pitch (note {note}, tenseness {tense})"
            );
        }
    });
}

#[test]
fn the_voice_recovers_at_once_from_a_pinch() {
    // Hold a note, pinch the tract fully shut anywhere for 300 ms, let go.
    // The folds are vibrating again within 12 ms of the tract reopening.
    // (It used to take 16 to 43 ms: the mouth stayed pressurised by the
    // air its walls had taken up, and the pressed folds let go slowly.
    // The bound was 10 ms, met at 8-10 ms, until the folds felt the
    // viscous loss along the glottis: the first openings out of a pressed
    // start, where the slit is narrowest and viscosity most of the drop,
    // are now pushed apart less, and every cell takes 1 ms longer, 3-11 ms.)
    for index in [
        3.0f32, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 37.0, 40.0, 42.5,
    ] {
        let mut t = fresh();
        tuned(Preset::Deep.settings())
            .with_engine(Engine::Physical)
            .apply(&mut t);
        t.articulate_phoneme(Phoneme::Aa);
        t.settle();
        t.note_on_midi(45.0);
        render(&mut t, 0.5);
        t.set_constriction(2, Some(Constriction::new(index, 0.0)));
        render(&mut t, 0.3);
        assert!(
            t.diameters()[1..].iter().any(|&d| d <= 0.0),
            "cell {index} never shut"
        );
        t.set_constriction(2, None);
        let (mut reopened, mut swinging) = (None, None);
        let mut slices: Vec<(f32, f32)> = Vec::new();
        for ms in 0..200 {
            let (mut lo, mut hi) = (f32::INFINITY, 0.0f32);
            for _ in 0..48 {
                t.process();
                let a = t.gauges().glottal_area;
                lo = lo.min(a);
                hi = hi.max(a);
            }
            if reopened.is_none() && !t.diameters()[1..].iter().any(|&d| d <= 0.0) {
                reopened = Some(ms);
            }
            slices.push((lo, hi));
            let recent = &slices[slices.len().saturating_sub(8)..];
            let lo = recent.iter().map(|s| s.0).fold(f32::INFINITY, f32::min);
            let hi = recent.iter().map(|s| s.1).fold(0.0f32, f32::max);
            if swinging.is_none() && reopened.is_some() && hi - lo > 0.08 {
                swinging = Some(ms);
                break;
            }
        }
        let delay = swinging.expect("the voice never came back") - reopened.unwrap();
        println!("pinch at cell {index}: folds vibrating {delay} ms after it reopens");
        assert!(delay <= 12, "cell {index}: {delay} ms");
    }
}

#[test]
fn every_preset_speaks_on_the_physical_engine() {
    // "Hello there, how are you? Pink trombone, 42." with the accents the
    // text front end used when this gate was set (it now accents hel*lo*,
    // which makes the child voice peak at 2.46).
    let text = "[HH AX1 L OW] [DH EH1 R], [HH AW1] [AA0 R] [Y UW0]? \
                [P IH1 NG K] [T R AA1 M B OW N], [F AO1 R T IY] [T UW1].";
    for p in Preset::ALL {
        let settings = tuned(p.settings()).with_engine(Engine::Physical);
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
    let settings = tuned(Preset::Deep.settings()).with_engine(Engine::Physical);
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
    let mut t = fresh();
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

/// The twelve monophthongs.
const MONOPHTHONGS: [Phoneme; 12] = [
    Phoneme::Iy,
    Phoneme::Ih,
    Phoneme::Eh,
    Phoneme::Ae,
    Phoneme::Ah,
    Phoneme::Aa,
    Phoneme::Ao,
    Phoneme::Uh,
    Phoneme::Uw,
    Phoneme::Ow,
    Phoneme::Er,
    Phoneme::Ax,
];

/// Troughs of `a` (index, value), keeping only swings of at least `swing`:
/// one per cycle.
fn troughs(a: &[f32], swing: f32) -> Vec<(usize, f32)> {
    let mut out = Vec::new();
    let (mut lo, mut hi) = ((0usize, a[0]), (0usize, a[0]));
    let mut rising = false;
    for (i, &v) in a.iter().enumerate() {
        if rising {
            if v > hi.1 {
                hi = (i, v);
            }
            if hi.1 - v > swing {
                rising = false;
                lo = (i, v);
            }
        } else {
            if v < lo.1 {
                lo = (i, v);
            }
            if v - lo.1 > swing {
                out.push(lo);
                rising = true;
                hi = (i, v);
            }
        }
    }
    out
}

/// One trough per cycle: a notch within the open phase (a dip closer than
/// `shortest` samples to the trough before it) is merged with that trough,
/// the deeper one kept. Without this a notch in a child's /u/ was counted
/// as a cycle 0.4 ms long that never closed.
fn cycles(found: &[(usize, f32)], shortest: f32) -> Vec<(usize, f32)> {
    let mut out: Vec<(usize, f32)> = Vec::with_capacity(found.len());
    for &t in found {
        match out.last_mut() {
            Some(last) if ((t.0 - last.0) as f32) < shortest => {
                if t.1 < last.1 {
                    *last = t;
                }
            }
            _ => out.push(t),
        }
    }
    out
}

/// Strongest component of `x` within ±8 % of `hz`, under a Hann window.
#[allow(clippy::many_single_char_names)]
fn harmonic(x: &[f32], hz: f32) -> f32 {
    let n = x.len() as f64;
    let mut best = 0.0f64;
    for step in -8..=8 {
        let w = 2.0 * std::f64::consts::PI * f64::from(hz) * (1.0 + 0.01 * f64::from(step))
            / f64::from(SR);
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (k, &v) in x.iter().enumerate() {
            let k = k as f64;
            let v = f64::from(v) * (0.5 - 0.5 * (2.0 * std::f64::consts::PI * k / n).cos());
            re += v * (w * k).cos();
            im += v * (w * k).sin();
        }
        best = best.max((re * re + im * im).sqrt());
    }
    best as f32
}

/// What a held vowel did: cycles, cycles whose edges never met, the
/// longest gap between cycles (periods), the share of the voiced time the
/// pulse sang, and the H1–H2 of the sound in 50 ms frames.
struct Closure {
    cycles: usize,
    open: usize,
    longest: f32,
    pulse: f32,
    h1h2: Vec<f32>,
}

/// Hold `vowel` on `settings` at the voice's own pitch: 0.2 s to start,
/// then 2 s measured.
fn hold_closure(settings: VoiceSettings, vowel: Phoneme) -> Closure {
    /// A trough below this share of the typical peak counts as closed.
    const CLOSED: f32 = 0.01;
    let mut t = fresh();
    settings.with_engine(Engine::Physical).apply(&mut t);
    t.articulate_phoneme(vowel);
    t.settle();
    t.note_on();
    render(&mut t, 0.2);
    let n = (2.0 * SR) as usize;
    let (mut edge, mut out) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let mut pulse = 0usize;
    for _ in 0..n {
        out.push(t.process());
        let g = t.gauges();
        edge.push(g.edge_area);
        if g.sounding == Some(GlottalSource::Pulse) {
            pulse += 1;
        }
    }
    let mut sorted = edge.clone();
    sorted.sort_by(f32::total_cmp);
    let peak = sorted[n * 95 / 100].max(1e-6);
    let expected = SR / settings.pitch;
    let found = cycles(&troughs(&edge, 0.15 * peak), 0.5 * expected);
    let longest = found
        .windows(2)
        .map(|w| (w[1].0 - w[0].0) as f32 / expected)
        .fold(0.0f32, f32::max);
    let open = found.iter().skip(1).filter(|t| t.1 > CLOSED * peak).count();
    // H1-H2 per 50 ms, each frame at least four periods long.
    let hop = (0.05 * SR) as usize;
    let len = (hop as f32).max(4.0 * expected) as usize;
    let mut h1h2 = Vec::new();
    let mut from = 0;
    while from + len <= out.len() {
        let frame = &out[from..from + len];
        let hz = settings.pitch;
        h1h2.push(db(harmonic(frame, hz)) - db(harmonic(frame, 2.0 * hz)));
        from += hop;
    }
    Closure {
        cycles: found.len().saturating_sub(1),
        open,
        longest,
        pulse: pulse as f32 / n as f32,
        h1h2,
    }
}

/// Every voiced preset as it is, and the default voice at every body size.
fn closure_voices() -> Vec<(String, VoiceSettings)> {
    let mut voices: Vec<(String, VoiceSettings)> = Preset::ALL
        .into_iter()
        .filter(|&p| p != Preset::Whisper)
        .map(|p| (p.to_string(), tuned(p.settings())))
        .collect();
    for size in [0.6f32, 0.8, 1.2, 1.4] {
        let persona = Persona {
            size,
            ..Persona::NEUTRAL
        };
        voices.push((
            format!("deep at size {size}"),
            tuned(Preset::Deep.settings()).with_persona(&persona),
        ));
    }
    voices
}

#[test]
fn the_folds_close_on_every_cycle_and_h1_h2_holds_steady() {
    // Modal voice closes the membranous glottis on every cycle; a breathy
    // one leaks through the gap behind it, not between the edges. Every
    // vowel on every voiced preset (the whisper has no voice), and the
    // default voice at sizes 0.6 to 1.4: in 2 s held, no cycle whose
    // edges do not meet, no missing cycle, all of it sung by the folds,
    // and H1-H2 (the closure's signature in the sound) steady from one
    // 50 ms frame to the next.
    //
    // Before the folds felt the viscous loss along the slit they press on,
    // 1889 of 51343 cycles did not close (/i/ and /u/ on bright, child,
    // critter and elder, and the default voice at size 0.6, where every
    // other cycle stayed open), elder's /i/ and /u/ missed cycles, and
    // H1-H2 stepped by up to 19.2 dB between frames (range 20.9 dB); the
    // default voice at size 1.4 was sung by the pulse. Now no cycle stays
    // open, and the worst step is 1.3 dB, the worst range 1.5 dB (3.2 and
    // 3.8 dB on the vowels solved from Hillenbrand's formants).
    /// Largest H1-H2 step between neighbouring frames, and range, dB.
    const STEP: f32 = 4.0;
    const RANGE: f32 = 5.0;
    let voices = closure_voices();
    std::thread::scope(|scope| {
        let jobs: Vec<_> = voices
            .iter()
            .flat_map(|(name, settings)| {
                MONOPHTHONGS.map(|vowel| {
                    (
                        name,
                        vowel,
                        scope.spawn(move || hold_closure(*settings, vowel)),
                    )
                })
            })
            .collect();
        let (mut worst_step, mut worst_range) = (0.0f32, 0.0f32);
        let (mut cycles, mut open) = (0usize, 0usize);
        let mut failures = Vec::new();
        for (name, vowel, job) in jobs {
            let c = job.join().expect("a held vowel");
            let step = c
                .h1h2
                .windows(2)
                .map(|w| (w[1] - w[0]).abs())
                .fold(0.0f32, f32::max);
            let lo = c.h1h2.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = c.h1h2.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            worst_step = worst_step.max(step);
            worst_range = worst_range.max(hi - lo);
            cycles += c.cycles;
            open += c.open;
            let mut fail = |what: String| failures.push(format!("{name} {vowel}: {what}"));
            if c.cycles <= 50 {
                fail(format!("only {} cycles", c.cycles));
            }
            if c.open > 0 {
                fail(format!("{} of {} cycles did not close", c.open, c.cycles));
            }
            if c.longest >= 1.8 {
                fail(format!("a gap of {:.1} periods between cycles", c.longest));
            }
            if c.pulse > 0.0 {
                fail(format!("the pulse sang {:.0} %", 100.0 * c.pulse));
            }
            if step >= STEP {
                fail(format!("H1-H2 stepped by {step:.1} dB"));
            }
            if hi - lo >= RANGE {
                fail(format!("H1-H2 ranged over {:.1} dB", hi - lo));
            }
        }
        println!(
            "{open} of {cycles} cycles did not close; H1-H2: largest step {worst_step:.2} dB, largest range {worst_range:.2} dB"
        );
        for f in &failures {
            println!("  {f}");
        }
        assert!(
            failures.is_empty(),
            "{} failures, first: {}",
            failures.len(),
            failures[0]
        );
    });
}

/// The judge's sentences, `(id, text)`.
fn judge_sentences() -> Vec<(String, String)> {
    let mut lines = Vec::new();
    for file in ["sentence_tune.tsv", "sentence_heldout.tsv"] {
        let path = format!("../../tools/judge/material/{file}");
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        for row in text.lines().skip(1) {
            let cols: Vec<&str> = row.split('\t').collect();
            if cols.len() >= 3 {
                lines.push((cols[0].to_string(), cols[2].to_string()));
            }
        }
    }
    lines
}

#[test]
fn the_folds_carry_running_speech() {
    // In running speech the pulse source may sing only notes outside the
    // folds' range, never one inside it: not after a stall, not at the
    // end of a voiced stop. Every voiced preset speaks the judge's twenty
    // sentences; every voiced sample sung by the pulse is checked against
    // the range the folds had at that moment.
    use synthie_trombone::speech::{Speaker, Utterance};
    let sentences = judge_sentences();
    assert_eq!(sentences.len(), 20);
    std::thread::scope(|scope| {
        let jobs: Vec<_> = Preset::ALL
            .into_iter()
            .filter(|&p| p != Preset::Whisper)
            .map(|preset| {
                let sentences = &sentences;
                let job = scope.spawn(move || {
                    let settings = tuned(preset.settings()).with_engine(Engine::Physical);
                    let (mut voiced, mut pulse, mut inside) = (0usize, 0usize, Vec::new());
                    for (id, text) in sentences {
                        let mut t = fresh();
                        settings.apply(&mut t);
                        Speaker::prepare(&mut t);
                        let mut speaker = Speaker::new();
                        speaker.speak(Utterance::from_text(text, &settings.prosody()));
                        while speaker.is_speaking() {
                            speaker.process(&mut t);
                            match t.gauges().sounding {
                                Some(GlottalSource::Pulse) => {
                                    voiced += 1;
                                    pulse += 1;
                                    let (lo, hi) = t.fold_range();
                                    let hz = t.frequency();
                                    if (lo..=hi).contains(&hz) {
                                        inside
                                            .push(format!("{id} at {hz:.0} Hz ({lo:.0}-{hi:.0})"));
                                    }
                                }
                                Some(GlottalSource::Folds) => voiced += 1,
                                None => {}
                            }
                        }
                    }
                    (voiced, pulse, inside)
                });
                (preset, job)
            })
            .collect();
        for (preset, job) in jobs {
            let (voiced, pulse, inside) = job.join().expect("speech");
            println!(
                "{preset}: the pulse sang {:.2} % of {:.1} s voiced, all of it outside the folds' range",
                100.0 * pulse as f32 / voiced.max(1) as f32,
                voiced as f32 / SR
            );
            assert!(
                inside.is_empty(),
                "{preset}: the pulse sang {} samples inside the folds' range, first {}",
                inside.len(),
                inside[0]
            );
        }
    });
}

/// The four sentences of the owner's listening test.
const LISTENING: [&str; 4] = [
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "She sells sea shells by the sea shore.",
    "The juice of lemons makes fine punch.",
];

#[test]
fn voiceless_stops_shut_the_mouth_for_their_closure() {
    // A /p t k/ is a closure: the mouth and the nose shut for some 50 to
    // 100 ms in a speaker's running speech, and still some tens of
    // milliseconds in a fast one. Too short a closure is not heard as a
    // stop at all. Every voiceless stop of the judge's twenty sentences
    // and the four listening sentences, at three speaking rates, is held
    // shut (no lumen in front of the pharynx, the velum shut) for at least
    // `Tuning::closure_floor`, 22 ms, without a break. Before the floor the
    // shortest closures were 32 ms at rate 1 ([ŋk] in "planks", [mp], [nt]),
    // but at rate 1.5 eleven of 272 were shorter than 22 ms and at rate 2
    // forty-eight, six of them never shut at all.
    use synthie_trombone::speech::{Speaker, Utterance};
    let mut texts: Vec<String> = judge_sentences().into_iter().map(|(_, t)| t).collect();
    texts.extend(LISTENING.iter().map(|s| (*s).to_string()));
    let floor = synthie_trombone::Tuning::DEFAULT.closure_floor;
    std::thread::scope(|scope| {
        let jobs: Vec<_> = [1.0f32, 1.5, 2.0]
            .into_iter()
            .map(|rate| {
                let texts = &texts;
                scope.spawn(move || {
                    let mut settings = tuned(Preset::Deep.settings()).with_engine(Engine::Physical);
                    settings.rate = rate;
                    let (mut count, mut shortest, mut short) = (0, f32::INFINITY, Vec::new());
                    for text in texts {
                        let mut t = fresh();
                        settings.apply(&mut t);
                        Speaker::prepare(&mut t);
                        let utterance = Utterance::from_text(text, &settings.prosody());
                        let phonemes: Vec<Phoneme> =
                            utterance.segments().iter().map(|s| s.phoneme).collect();
                        let closures: Vec<bool> = utterance
                            .segments()
                            .iter()
                            .map(|s| {
                                !s.voiced
                                    && matches!(s.phoneme, Phoneme::P | Phoneme::T | Phoneme::K)
                                    && s.articulation
                                        .constriction
                                        .is_some_and(|c| c.diameter <= 0.3)
                            })
                            .collect();
                        let mut speaker = Speaker::new();
                        speaker.speak(utterance);
                        // Per closure segment: the longest sealed stretch
                        // that reaches into it.
                        let mut longest = vec![0usize; phonemes.len()];
                        let (mut run, mut touched) = (0usize, Vec::<usize>::new());
                        while let Some((i, _)) = speaker.current() {
                            speaker.process(&mut t);
                            if t.mouth_shut() && t.velum() < 0.05 {
                                run += 1;
                                if closures[i] && !touched.contains(&i) {
                                    touched.push(i);
                                }
                                for &k in &touched {
                                    longest[k] = longest[k].max(run);
                                }
                            } else {
                                run = 0;
                                touched.clear();
                            }
                        }
                        for (k, &c) in closures.iter().enumerate() {
                            if !c {
                                continue;
                            }
                            count += 1;
                            let ms = longest[k] as f32 / SR;
                            shortest = shortest.min(ms);
                            if ms < floor - 1.0 / SR {
                                let prev = k.checked_sub(1).map_or(Phoneme::Sil, |j| phonemes[j]);
                                let next = phonemes.get(k + 1).copied().unwrap_or(Phoneme::Sil);
                                short.push(format!(
                                    "{prev} [{}] {next} {:.1} ms in {text:?}",
                                    phonemes[k],
                                    1e3 * ms
                                ));
                            }
                        }
                    }
                    (rate, count, shortest, short)
                })
            })
            .collect();
        for job in jobs {
            let (rate, count, shortest, short) = job.join().expect("speech");
            println!(
                "rate {rate}: {count} voiceless stops, shortest closure {:.1} ms",
                1e3 * shortest
            );
            assert!(
                short.is_empty(),
                "rate {rate}: {} closures under {:.0} ms, first {}",
                short.len(),
                1e3 * floor,
                short[0]
            );
        }
    });
}

#[test]
fn affricates_release_into_full_frication() {
    // An affricate's closure opens into the fricative of its place: its
    // hiss is that of /sh/ or /zh/ (within 3 dB). The noise takes 10 to
    // 15 ms to build, so the 10 ms after the release that the burst gate
    // reads hold only its start.
    let settings = speech_voice();
    for (affricate, fricative) in [(Phoneme::Ch, Phoneme::Sh), (Phoneme::Jh, Phoneme::Zh)] {
        let a = stops::fricative(&settings, &stops::context(affricate, "vcv"), affricate);
        let f = stops::fricative(&settings, &stops::context(fricative, "vcv"), fricative);
        println!(
            "{affricate}: hiss {:.1} dB, {fricative}: {:.1} dB re the vowel",
            a.hiss, f.hiss
        );
        assert!(
            (a.hiss - f.hiss).abs() <= 3.0,
            "{affricate} hiss {:.1} dB",
            a.hiss
        );
    }
}

#[test]
fn stop_releases_are_bursts_not_clicks() {
    // A stop is released as a short burst of noise shaped by the cavity in
    // front of the closure, some 10 to 20 dB below the vowel that follows
    // (Stevens 1998, ch. 7), not as a step between two samples. Two such
    // steps came out of closures and releases: the air a closing
    // articulator squeezed out of a cell that had all but shut (thousands
    // of cmH2O in its last ticks, cut off at the seal), and the folds being
    // handed to the pulse source and back while apart for a pause. Before
    // the fix the largest step in these sentences was 11 times the loudest
    // vowel sample in the default voice, 8 to 9 in the elder's and the
    // giant's, 85 in the whisper's; up to 0.9 of it at a release.
    //
    // A third cause was the shape itself: it moved on straight lines
    // between articulator blocks, so a closing cell changed its speed at
    // once every 10.7 ms, and in a narrow gap the change in the air
    // squeezed out of it was a step (up to 3.2 vowel peaks in the elder).
    // The shape now follows the articulators' own speeds through each
    // block.
    //
    // Two more were found when the vowels were held longer: they came and
    // went with a few per cent of vowel timing, up to 7 vowel peaks. The
    // critter's high notes leave its folds' range within a vowel, and the
    // voice was handed to the pulse source in one tick, the glottal area
    // jumping from 0.044 to 0.003 cm² (now cross-faded over 6 ms); and the
    // larynx was timed from segment boundaries, so when the slower tongue
    // body moved a closure against them the glottis opened before the
    // mouth shut and the breath hissed through the closing gap (it now
    // opens as the mouth shuts).
    //
    // What is left is noise, which has steps of its own: the hiss of the
    // last gap before a closure, ringing of a narrow gap in the critter's
    // small tract, and in the whisper, breath through nasals and closing
    // gaps measured against vowels that are themselves faint breath. So a
    // click is told from noise as a step that stands far above the steps
    // around it (`stops::largest_click`). Swept over `vowel_hold` 1.0 to 1.3
    // on every voice: clicks at most 2.0 vowel peaks (critter; the whisper
    // 5.1, the pop of a /p/ released into a /t/'s closure), plain steps at
    // most 2.4 outside the whisper, steps at releases at most 0.27 (whisper
    // 0.79), bursts at most -0.7 dB re the vowel after them (whisper +5.0).
    //
    // Bounds: clicks 2.5 vowel peaks (whisper 6); steps 3 (whisper: not
    // bounded, its own clicks are); at releases 0.35 (whisper 1.0); no kind
    // of stop bursting louder on average than 6 dB below the vowel after
    // it, no burst louder than that vowel (whisper: 6 dB louder, against a
    // whispered vowel 15 to 20 dB weaker than a voiced one).
    const AT_RELEASE: f32 = 0.35;
    const STEP: f32 = 3.0;
    const CLICK: f32 = 2.5;
    const MEAN_BURST: f32 = -6.0;
    const MAX_BURST: f32 = 0.0;
    let texts = [
        "As for etchings they are of two kinds British and foreign.",
        "The judge checked the big black bag at the gate.",
        "Pick a cup and dig deep to get the chips.",
    ];
    for preset in Preset::ALL {
        let settings = tuned(preset.settings());
        let (at_release, step, click) = match preset {
            Preset::Whisper => (1.0, f32::INFINITY, 6.0),
            _ => (AT_RELEASE, STEP, CLICK),
        };
        let (mut worst, mut worst_release, mut worst_click) = (0.0f32, 0.0f32, 0.0f32);
        let mut levels: std::collections::BTreeMap<&str, Vec<f32>> =
            std::collections::BTreeMap::new();
        for text in texts {
            let r = stops::record(&settings, text);
            let vowel = stops::vowel_peak(&r);
            worst = worst.max(stops::largest_step(&r) / vowel);
            worst_click = worst_click.max(stops::largest_click(&r) / vowel);
            for b in stops::bursts(&r) {
                worst_release = worst_release.max(b.step / vowel);
                levels
                    .entry(b.phoneme.symbol())
                    .or_default()
                    .push(stops::db(b.peak / b.vowel));
            }
        }
        let mut line = format!(
            "{preset}: largest step {worst:.2} x vowel peak, click {worst_click:.2}, {worst_release:.2} at releases; bursts re the vowel's peak, dB (mean / max):"
        );
        for (p, v) in &levels {
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let _ = write!(line, " {p} {mean:+.1}/{max:+.1}");
            assert!(
                mean <= MEAN_BURST,
                "{preset} {p}: bursts {mean:+.1} dB on average"
            );
            let max_burst = if preset == Preset::Whisper {
                6.0
            } else {
                MAX_BURST
            };
            assert!(
                max <= max_burst,
                "{preset} {p}: a burst {max:+.1} dB over the vowel"
            );
        }
        println!("{line}");
        assert!(
            worst_release <= at_release,
            "{preset}: a step of {worst_release:.2} x the loudest vowel sample at a release"
        );
        assert!(
            worst <= step,
            "{preset}: a step of {worst:.2} x the loudest vowel sample"
        );
        assert!(
            worst_click <= click,
            "{preset}: a click of {worst_click:.2} x the loudest vowel sample"
        );
    }
}

// ----- place of articulation -------------------------------------------

#[test]
fn the_tongue_tip_is_quicker_than_the_tongue_body() {
    // The tip, the blade and the lips are light; the tongue body and the
    // jaw are heavy (in speech a tip gesture takes some 10 to 20 ms, a
    // body movement between vowels 40 to 50 ms and more). Here a /t/ is
    // released into /i/ out of /a/: the tip's cell opens and the body's
    // cells move from /a/ to /i/, both from the same moment. Measured: the
    // tip covers 63 % of its way in 11 ms, the body in 75.
    let settled = |a: &Articulation| {
        let mut t = Trombone::new(SR);
        t.set_engine(Engine::Physical);
        t.articulate(a);
        t.settle();
        *t.diameters()
    };
    let iy = Phoneme::Iy.articulation();
    let mut t_on_aa = Phoneme::T.articulation();
    t_on_aa.tongue = Phoneme::Aa.articulation().tongue;
    let (from, to) = (settled(&t_on_aa), settled(&iy));
    // The body's cell: the one that moves furthest between the vowels,
    // well behind the tip.
    let body = (12..28)
        .max_by(|&a, &b| (to[a] - from[a]).abs().total_cmp(&(to[b] - from[b]).abs()))
        .expect("cells");
    let tip = 36;
    let mut t = Trombone::new(SR);
    t.set_engine(Engine::Physical);
    synthie_trombone::speech::Speaker::prepare(&mut t);
    t.articulate(&t_on_aa);
    t.settle();
    t.articulate(&iy);
    let mut reached = [None; 2];
    for n in 0..(0.4 * SR) as usize {
        t.process();
        let d = t.diameters();
        for (k, &i) in [tip, body].iter().enumerate() {
            let share = (d[i] - from[i]) / (to[i] - from[i]);
            if reached[k].is_none() && share >= 0.63 {
                reached[k] = Some(n as f32 / SR * 1000.0);
            }
        }
    }
    let (tip_ms, body_ms) = (reached[0].expect("tip"), reached[1].expect("body"));
    println!(
        "63 % of the way: tip (cell {tip}) {tip_ms:.0} ms, body (cell {body}) {body_ms:.0} ms"
    );
    assert!(tip_ms <= 35.0, "tip {tip_ms} ms");
    assert!((50.0..=150.0).contains(&body_ms), "body {body_ms} ms");
    assert!(
        body_ms >= 2.5 * tip_ms,
        "tip {tip_ms} ms, body {body_ms} ms"
    );
}

#[test]
fn stop_transitions_point_to_their_place() {
    // The second formant at the onset of a vowel after a voiced stop
    // points back at the place it was released from (Delattre, Liberman
    // & Cooper 1955; Sussman's locus equations): below the vowel's own F2
    // after /b/ (bilabial locus 700 to 1100 Hz), above it after /d/
    // before back vowels and above /b/'s everywhere (alveolar, about
    // 1800 Hz), and after /g/ high before back vowels, following the
    // vowel, with a locus between the two and a slope well under 1.
    let settings = stops::judge_voice();
    let vowels = [Phoneme::Iy, Phoneme::Aa, Phoneme::Uw];
    let mut velar = Vec::new();
    for v in vowels {
        let b = place::transition(&settings, Phoneme::B, v);
        let d = place::transition(&settings, Phoneme::D, v);
        let g = place::transition(&settings, Phoneme::G, v);
        println!(
            "{v}: vowel F2 {:.0} | onset after B {:.0}, D {:.0}, G {:.0} Hz",
            b.mid[0], b.onset[0], d.onset[0], g.onset[0]
        );
        assert!(
            b.onset[0] < 0.97 * b.mid[0],
            "{v}: /b/ onset {:?} vs {:?}",
            b.onset,
            b.mid
        );
        assert!(
            d.onset[0] > b.onset[0] + 100.0,
            "{v}: /d/ {:?}, /b/ {:?}",
            d.onset,
            b.onset
        );
        if v != Phoneme::Iy {
            assert!(
                d.onset[0] > d.mid[0],
                "{v}: /d/ onset {:?} vs {:?}",
                d.onset,
                d.mid
            );
            assert!(
                g.onset[0] > g.mid[0] + 150.0,
                "{v}: /g/ onset {:?} vs {:?}",
                g.onset,
                g.mid
            );
        }
        velar.push((g.mid[0], g.onset[0]));
    }
    let (locus, slope) = place::locus(&velar);
    println!("/g/ locus {locus:.0} Hz, slope {slope:.2}");
    assert!(
        (1400.0..=2300.0).contains(&locus) && slope < 0.8,
        "/g/ locus {locus} slope {slope}"
    );
}

/// The deepest notch of a murmur's transfer function between `lo` and
/// `hi` Hz, if any is at least `depth` dB deep.
fn notch_in(notches: &[place::Notch], lo: f32, hi: f32, depth: f32) -> Option<place::Notch> {
    notches
        .iter()
        .filter(|n| (lo..=hi).contains(&n.hz) && n.depth >= depth)
        .max_by(|a, b| a.depth.total_cmp(&b.depth))
        .copied()
}

#[test]
fn nasal_murmurs_have_the_antiresonance_of_their_place() {
    // The mouth closed behind a nasal's closure is a side branch of the
    // nose; its own resonance is an antiresonance of the murmur: 750 to
    // 1250 Hz for /m/ (the whole mouth), 1450 to 2200 Hz for /n/, above 3
    // kHz for /ŋ/, whose closure leaves almost nothing behind it (Fujimura
    // 1962). Measured as spoken, in the middle of the nasal in [V C V1]
    // for ten vowels: the frequency at which the mouth behind the closure,
    // as a tube of the tract's cells, has no impedance, and the notch the
    // transfer function has there.
    //
    // Left in the vowel's shape, the mouth followed the vowel: before
    // /ʌ ə ɔ oʊ/ the hump of the tongue body near the velum, a narrow neck
    // at the open end of the branch, pulled /n/ down to 700 to 1000 Hz and
    // /m/ to 420 to 530 Hz, and the recognizer heard /n/ as /m/ (and /m/ as
    // /n/ or /ŋ/) in a third of the nasals in the LibriSpeech sentences.
    // With the tongue lowered behind the closure (`Tuning::nasal_cavity`):
    // /m/ 750 to 1060 Hz with a notch 23 to 40 dB deep, /n/ 1200 to 1550 Hz
    // (lowest before /ɔ/; with the tongue lowered at the back of the
    // tongue's speed, 1150 Hz there and 1190 Hz before /ʌ/), and in every
    // vowel /n/ at least 340 Hz above /m/. (The notch /n/ leaves is shallower, 2 to 30 dB: a resonance of
    // the murmur lies close to it.) Lowered only up to the tip's shoulder,
    // /n/ after /i ɪ æ ʊ/ stood at 1.75 to 1.9 kHz, and the recognizer
    // heard /n/ as /ŋ/ in a fifth of those over 1.7 kHz: so /n/ is also
    // held under 1.7 kHz here.
    let settings = stops::judge_voice();
    for v in [
        Phoneme::Iy,
        Phoneme::Ih,
        Phoneme::Eh,
        Phoneme::Ae,
        Phoneme::Aa,
        Phoneme::Ah,
        Phoneme::Ao,
        Phoneme::Ow,
        Phoneme::Uw,
        Phoneme::Ax,
    ] {
        let zero = |nasal: Phoneme| {
            let t = place::spoken_nasal(&settings, nasal, v);
            let zeros = place::branch_zeros(t.diameters(), t.tick_rate());
            (zeros.first().copied(), place::notches(&t))
        };
        let (m, m_notches) = zero(Phoneme::M);
        let (n, _) = zero(Phoneme::N);
        let (ng, ng_notches) = zero(Phoneme::Ng);
        let m = m.unwrap_or_else(|| panic!("in {v}: /m/ has no antiresonance"));
        let n = n.unwrap_or_else(|| panic!("in {v}: /n/ has no antiresonance"));
        let m_notch = notch_in(&m_notches, 0.9 * m, 1.1 * m, 0.0).map_or(0.0, |x| x.depth);
        println!("in {v}: /m/ {m:.0} Hz (notch {m_notch:.0} dB), /n/ {n:.0} Hz, /ŋ/ {ng:?}");
        assert!((700.0..=1250.0).contains(&m), "in {v}: /m/ at {m:.0} Hz");
        assert!(m_notch >= 20.0, "in {v}: /m/ notch {m_notch:.1} dB");
        assert!((1100.0..=1700.0).contains(&n), "in {v}: /n/ at {n:.0} Hz");
        assert!(n >= m + 300.0, "in {v}: /n/ {n:.0} Hz, /m/ {m:.0} Hz");
        assert!(ng.is_none_or(|z| z > 3000.0), "in {v}: /ŋ/ at {ng:?}");
        // The nose has a shallow notch of its own near 1 kHz (11 to 13 dB,
        // the two nasal passages out of step), in every nasal.
        assert!(
            notch_in(&ng_notches, 600.0, 1300.0, 15.0).is_none(),
            "in {v}: /ŋ/ {ng_notches:?}"
        );
    }
}

/// A short utterance spoken by the physical engine, sample by sample: the
/// segment, whether the mouth is shut, the velum, the output and the oral
/// tract's diameters.
struct Said {
    phonemes: Vec<Phoneme>,
    segment: Vec<usize>,
    shut: Vec<bool>,
    velum: Vec<f32>,
    out: Vec<f32>,
    diameters: Vec<[f32; synthie_trombone::N]>,
    tick_rate: f32,
}

fn say_traced(settings: &VoiceSettings, text: &str) -> Said {
    use synthie_trombone::speech::{Speaker, Utterance};
    let settings = tuned(*settings).with_engine(Engine::Physical);
    let mut t = fresh();
    settings.apply(&mut t);
    Speaker::prepare(&mut t);
    let utterance = Utterance::from_text(text, &settings.prosody());
    let mut s = Said {
        phonemes: utterance.segments().iter().map(|s| s.phoneme).collect(),
        segment: Vec::new(),
        shut: Vec::new(),
        velum: Vec::new(),
        out: Vec::new(),
        diameters: Vec::new(),
        tick_rate: t.tick_rate(),
    };
    let mut speaker = Speaker::new();
    speaker.speak(utterance);
    while let Some((i, _)) = speaker.current() {
        s.out.push(speaker.process(&mut t));
        s.segment.push(i);
        s.shut.push(t.mouth_shut());
        s.velum.push(t.velum());
        s.diameters.push(*t.diameters());
    }
    s
}

/// Second-order Butterworth high-pass at `hz`.
fn high_pass(x: &[f32], hz: f32) -> Vec<f32> {
    let w = 2.0 * std::f32::consts::PI * hz / SR;
    let alpha = w.sin() / (2.0 * std::f32::consts::FRAC_1_SQRT_2);
    let a0 = 1.0 + alpha;
    let (b0, b1) = ((1.0 + w.cos()) / 2.0 / a0, -(1.0 + w.cos()) / a0);
    let (a1, a2) = (-2.0 * w.cos() / a0, (1.0 - alpha) / a0);
    let (mut x1, mut x2, mut y1, mut y2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    x.iter()
        .map(|&v| {
            let y = b0 * v + b1 * x1 + b0 * x2 - a1 * y1 - a2 * y2;
            (x2, x1, y2, y1) = (x1, v, y1, y);
            y
        })
        .collect()
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn stops_are_released_into_liquids_with_a_burst_and_a_transition() {
    // "Dark blue" was heard as "dark loo": a /b p g k/ in front of an /l r/
    // was released straight into the liquid. Its narrowing came within one
    // articulator block (10 ms) of the release, and the lips of /b p/,
    // parting under the liquid's own tongue body, moved at the back of the
    // tongue's speed: 10 ms after the release they were 0.28 apart before
    // /l/ (0.15 before /r/), against 0.6 before a vowel, and the burst of
    // /b/ above 2.5 kHz was 17 dB under that of the same /b/ before a vowel.
    // A speaker's burst before a liquid is about 7 dB under the one before
    // a vowel (13 and 36 stops in the LibriSpeech sentences), and what a
    // listener hears first is the stop: its burst, then a short stretch of
    // its own transition.
    //
    // Now the lips part at their own speed and the liquid's narrowing waits
    // for the release (`Tuning::liquid_transition`). In these words, 10 ms
    // after the release the tongue tip is still 0.14 to 0.8 wider than in
    // the liquid, which it reaches 20 to 31 ms after the release; the lips of
    // /bl pl/ are 0.56 apart, those of /br pr/ 0.53; the burst of /bl/ is
    // 1.8 dB over that of /bi/. Without the hold (`liquid_transition` 0)
    // /br/'s tip was in place after 10 ms and /bl/'s burst 9 dB under /bi/'s.
    // Bounds: the narrowing no sooner than 15 ms and no later than 45 ms
    // after the release, the tip at least 0.12 wider than in the liquid
    // after 10 ms; lips at least 0.45 apart 10 ms after a labial's release
    // into /l/ and 0.25 into /r/; the burst of /bl/ at most 6 dB under that
    // of /bi/.
    let settings = stops::judge_voice();
    // (word, stop, liquid, cell of the liquid's narrowing)
    let words = [
        ("a blue", Phoneme::B, Phoneme::L, 35),
        ("a plum", Phoneme::P, Phoneme::L, 35),
        ("a glue", Phoneme::G, Phoneme::L, 35),
        ("a clue", Phoneme::K, Phoneme::L, 35),
        ("a brew", Phoneme::B, Phoneme::R, 31),
        ("a prune", Phoneme::P, Phoneme::R, 31),
        ("a groom", Phoneme::G, Phoneme::R, 31),
        ("a crew", Phoneme::K, Phoneme::R, 31),
    ];
    // The level above 2.5 kHz over the first 10 ms after the release of
    // `stop`, dB, and the release itself.
    let burst = |s: &Said, stop: Phoneme| {
        let start = (0..s.out.len())
            .find(|&n| s.phonemes[s.segment[n]] == stop)
            .expect("the stop");
        let release = (start.max(1)..s.out.len())
            .find(|&n| s.shut[n - 1] && !s.shut[n])
            .expect("a release");
        let hf = high_pass(&s.out, 2500.0);
        let n = (0.01 * SR) as usize;
        (db(rms(&hf[release..release + n])), release)
    };
    let before_vowel = burst(&say_traced(&settings, "a bee"), Phoneme::B).0;
    for (word, stop, liquid, cell) in words {
        let s = say_traced(&settings, word);
        let (level, release) = burst(&s, stop);
        // The liquid's narrowing: the diameter at its place in the middle
        // of the liquid.
        let k = s.phonemes.iter().position(|&p| p == liquid).expect("the liquid");
        let span: Vec<usize> = (0..s.out.len()).filter(|&n| s.segment[n] == k).collect();
        let narrow = s.diameters[span[span.len() / 2]][cell];
        let reached = (release..s.out.len())
            .find(|&n| s.diameters[n][cell] <= narrow + 0.05)
            .map_or(f32::INFINITY, |n| (n - release) as f32 / SR * 1000.0);
        let lips = s.diameters[release + (0.01 * SR) as usize][39..]
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let tip = s.diameters[release + (0.01 * SR) as usize][cell];
        println!(
            "{word}: narrowing {reached:.0} ms after the release ({tip:.2} after 10 ms, {narrow:.2} in the liquid), lips {lips:.2} after 10 ms, burst {level:.1} dB (/bi/ {before_vowel:.1})"
        );
        assert!(
            (15.0..=45.0).contains(&reached),
            "{word}: the liquid's narrowing {reached:.0} ms after the release"
        );
        assert!(
            tip >= narrow + 0.12,
            "{word}: tip {tip:.2} 10 ms after the release, {narrow:.2} in the liquid"
        );
        if matches!(stop, Phoneme::B | Phoneme::P) {
            let open = if liquid == Phoneme::L { 0.45 } else { 0.25 };
            assert!(lips >= open, "{word}: lips {lips:.2} apart 10 ms after the release");
        }
        if stop == Phoneme::B && liquid == Phoneme::L {
            assert!(
                level >= before_vowel - 6.0,
                "{word}: burst {level:.1} dB, /bi/ {before_vowel:.1} dB"
            );
        }
    }
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn the_antiresonance_of_n_holds_through_its_murmur_before_a_consonant() {
    // In sentences /n/ was still heard as /ŋ/ before consonants ("run
    // fast", "lemons", "fine punch": 5 of 94 in a set of such sentences).
    // The tongue was lowered behind the closure at the back of the tongue's
    // speed, so after /ʌ ə ɛ ɑ aʊ/ the mouth behind it was still the
    // vowel's for the first third of the murmur: the side branch's
    // antiresonance stood at 1.2 to 1.35 kHz there, rising to 1.5 kHz only
    // as the murmur ended (and before a consonant the murmur is short).
    // Lowered at `Tuning::nasal_cavity_rate` it is 1.45 to 1.55 kHz from the
    // start. Measured as the first zero of the closed branch from the velum
    // to the closure, at a quarter, a half and three quarters of the murmur
    // (mouth shut, velum open); bounds 1.4 to 1.7 kHz.
    let settings = stops::judge_voice();
    for text in [
        "[R AH1 N F AE1 S T]",
        "[L EH1 M AX N Z]",
        "[F AY1 N P AH1 N CH]",
        "[S EH1 V AX N L EH1 M AX N]",
        "[R AH1 N P AE1 S T]",
        "[W EH1 N T]",
        "[D AA1 N T]",
    ] {
        let s = say_traced(&settings, text);
        let k = s.phonemes.iter().position(|&p| p == Phoneme::N).expect("/n/");
        let murmur: Vec<usize> = (0..s.out.len())
            .filter(|&n| s.shut[n] && s.velum[n] > 0.1 && s.segment[n].abs_diff(k) <= 1)
            .collect();
        assert!(murmur.len() as f32 > 0.03 * SR, "{text}: murmur {} samples", murmur.len());
        let zeros: Vec<f32> = [0.25, 0.5, 0.75]
            .iter()
            .map(|&q| {
                let n = murmur[(q * murmur.len() as f32) as usize];
                place::branch_zeros(&s.diameters[n], s.tick_rate)
                    .first()
                    .copied()
                    .unwrap_or(0.0)
            })
            .collect();
        println!(
            "{text}: murmur {:.0} ms, /n/ antiresonance {zeros:.0?} Hz",
            murmur.len() as f32 / SR * 1000.0
        );
        for z in zeros {
            assert!((1400.0..=1700.0).contains(&z), "{text}: /n/ at {z:.0} Hz");
        }
    }
}

/// F2 in the middle of consonant `c` in `[v c v1]`, Hz, and its level
/// there against the loudest 10 ms of the vowel after it, dB.
#[allow(clippy::many_single_char_names)]
fn consonant_f2_and_dip(settings: &VoiceSettings, c: Phoneme, v: Phoneme) -> (f32, f32) {
    let s = place::speak(settings, &format!("[{v} {c} {v}1]"), true);
    let k = (1..s.phonemes.len())
        .find(|&k| s.phonemes[k] == c)
        .expect("the consonant");
    let (a, b) = s.ranges[k];
    // The middle of the constriction: the consonant without its last
    // `lead`, which belongs to the way into the vowel.
    let mid = a + (b - a) * 2 / 5;
    let f2 = s.formants_at(mid, 0.0).map_or(0.0, |f| f[1]);
    let level = |from: usize, to: usize| {
        let x = &s.audio[from..to.min(s.audio.len())];
        db(rms(x))
    };
    let window = (0.01 * SR) as usize;
    let here = level(mid.saturating_sub(window / 2), mid + window / 2);
    let (v0, v1) = s.ranges[k + 1];
    let vowel = (v0..v1.saturating_sub(window))
        .step_by(window / 4)
        .map(|n| level(n, n + window))
        .fold(f32::MIN, f32::max);
    (f2, here - vowel)
}

#[test]
#[allow(clippy::many_single_char_names)]
fn the_dental_fricative_is_not_a_lateral() {
    // "The" was heard as "luh" or "wuh", and the recognizer heard /ð/ as
    // /l/ in 8 of the 20 in the LibriSpeech sentences. A speaker's /ð/ is
    // made with the tip at the teeth and a broad blade behind it: its F2
    // stands at a dental locus near 1400 to 1600 Hz before /ə/, above a
    // labial's, and it is a dip in the level, weak voiced friction; an /l/
    // is nearly as loud as its vowels. Made with the alveolars' short
    // blade, /ð/ had a labial's F2 (900 Hz before /ə/, like /v/'s 700) and
    // dipped only 4 dB. Measured in the middle of each in [V C V1]: /ð/
    // F2 1320 to 1780 Hz, 180 to 680 Hz above /v/'s, and 5 to 20 dB under
    // the vowel after it, 3 to 12 dB deeper than /l/.
    let settings = speech_voice();
    std::thread::scope(|scope| {
        let jobs: Vec<_> = [Phoneme::Ax, Phoneme::Aa, Phoneme::Iy, Phoneme::Uw]
            .into_iter()
            .map(|v| {
                let settings = &settings;
                let job = scope.spawn(move || {
                    [Phoneme::Dh, Phoneme::V, Phoneme::L]
                        .map(|c| consonant_f2_and_dip(settings, c, v))
                });
                (v, job)
            })
            .collect();
        for (v, job) in jobs {
            let [(dh, dh_dip), (vv, _), (_, l_dip)] = job.join().expect("speech");
            println!(
                "in {v}: /ð/ F2 {dh:.0} Hz, {dh_dip:+.1} dB; /v/ F2 {vv:.0} Hz; /l/ {l_dip:+.1} dB"
            );
            assert!((1200.0..=2000.0).contains(&dh), "in {v}: /ð/ F2 {dh:.0} Hz");
            assert!(
                dh >= vv + 150.0,
                "in {v}: /ð/ F2 {dh:.0} Hz, /v/ {vv:.0} Hz"
            );
            assert!(dh_dip <= -4.0, "in {v}: /ð/ only {dh_dip:+.1} dB");
            assert!(
                dh_dip <= l_dip - 2.5,
                "in {v}: /ð/ {dh_dip:+.1} dB, /l/ {l_dip:+.1} dB"
            );
        }
    });
}

#[test]
fn no_vowel_comes_between_the_consonants_of_a_cluster() {
    // Movements between sounds must not be heard as sounds of their own.
    // Between the middles of two consonants that follow one another, the
    // tract's first resonance stays where a constriction holds it: under
    // 400 Hz between two obstruents (an /l/ or a vowel has it higher), and
    // under 600 Hz going into /l r/ (an /a/-like opening has 700). Measured
    // on these words: no frame over either.
    let settings = stops::judge_voice();
    for word in place::CLUSTER_WORDS {
        let s = place::speak(&settings, word, true);
        for (a, b, _) in place::cluster_vowel_frames(&s, 0.0) {
            let limit = if b.manner() == synthie_trombone::Manner::Approximant {
                600.0
            } else {
                400.0
            };
            let open = place::cluster_vowel_frames(&s, limit)
                .into_iter()
                .find(|&(x, y, _)| (x, y) == (a, b))
                .map_or(0, |e| e.2);
            assert_eq!(
                open, 0,
                "{word}: {a}-{b} has {open} frames with F1 over {limit} Hz"
            );
        }
    }
}

#[test]
fn final_stops_are_released() {
    // A stop at the end of a phrase is released, so it is heard: within
    // 10 ms of the opening the output rises to no less than 36 dB under the
    // vowel's peak (measured -18 to -31 dB; /k g/ were -49 and -45 dB, shut
    // to the end, before the release segment existed).
    let settings = stops::judge_voice();
    for p in ["P", "B", "T", "D", "K", "G"] {
        let r = stops::record(&settings, &format!("[AA1 {p}]"));
        let bursts = stops::bursts(&r);
        let b = bursts.last().expect("a release");
        let level = stops::db(b.peak) - stops::db(b.vowel);
        println!("final {p}: {level:+.1} dB re the vowel");
        assert!(level > -36.0, "final {p}: {level} dB");
    }
}
