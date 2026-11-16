//! How the physical engine gets from one shape to the next, measured from
//! the tract's own transfer function every 5 ms while it speaks (never LPC
//! on the output).
//!
//! * Out of an /n t d/ into /u/ the second formant does not fall from a
//!   palatal height onto the vowel: that fall is /ju/, and "canoe" was
//!   heard as "can you" while the lips of /u/ only began to round as the
//!   vowel did.
//! * Out of an /n/ into a vowel the nose is shut within 15 ms: left
//!   coupled, it gave the start of the /u/ of "canoe" a peak near 700 Hz,
//!   heard as /ʌ oʊ l/.
//! * From /l/ into /ɪ/ the first formant does not rise above both the
//!   /l/'s and the vowel's (an /aɪ/: "slid" heard as "slide"), and the /l/
//!   keeps the low second formant of an /l/.
//! * After /t d/ the /u/ is fronted and glides part of the way back.
//! * A /t d/ between a vowel and an unstressed one is a tap: shut for a
//!   few milliseconds, the voice going on through it, little pressure
//!   behind it.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]

#[path = "../examples/support/place.rs"]
mod place;
#[path = "../examples/support/spectrum.rs"]
mod spectrum;

use place::Spoken;
use synthie_trombone::{Phoneme, Preset, VoiceSettings};

fn voice() -> VoiceSettings {
    // The tuning under test: the default one, or, when the environment
    // variable `SYNTHIE_TROMBONE_PROFILE` names a profile file, that
    // profile's (as in `physical_gates`).
    static PROFILE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let settings = Preset::Deep.settings();
    let profile = PROFILE.get_or_init(|| {
        let path = std::env::var("SYNTHIE_TROMBONE_PROFILE").ok()?;
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        println!("gates run with the profile {path}");
        Some(text)
    });
    let Some(profile) = profile else {
        return settings;
    };
    let laid = settings.with_profile(profile).expect("a voice profile");
    VoiceSettings {
        tuning: laid.tuning,
        ..settings
    }
}

/// The frames from sample `from` to sample `to` whose tract is open and
/// whose nose is shut (with the velum open the peaks are the nose's too).
fn oral(s: &Spoken, from: usize, to: usize) -> Vec<(usize, [f32; 3])> {
    (Spoken::frame_of(from)..Spoken::frame_of(to).min(s.frames.len()))
        .filter_map(|k| {
            let f = &s.frames[k];
            (f.velum < 0.1).then_some(())?;
            f.formants.map(|x| (k, x))
        })
        .collect()
}

/// The last segment of phoneme `p` (its first one is `k`) and the next.
fn after(s: &Spoken, p: Phoneme) -> (usize, usize) {
    let k = s.find(p, 1).unwrap_or_else(|| panic!("no {p}"));
    let (_, end) = s.consonant_span(k);
    (k, end + 1)
}

#[test]
fn the_second_formant_does_not_fall_onto_u_after_a_coronal() {
    for (text, c) in [
        ("a canoe", Phoneme::N),
        ("the new", Phoneme::N),
        ("to do", Phoneme::D),
        ("two", Phoneme::T),
    ] {
        let s = place::speak(&voice(), text, true);
        let (_, v) = after(&s, c);
        assert_eq!(s.phonemes[v], Phoneme::Uw, "{text}: {:?}", s.phonemes);
        // After /t d/ the /u/ is a fronted one (`speech::fronts_u`),
        // gliding back part of the way to the plain one over its second
        // half (`Tuning::fronted_nucleus`); after /n/ it is the plain back
        // /u/ (fronted there, it was heard as /ju/: "canoe" as "can use").
        let fronted = matches!(c, Phoneme::T | Phoneme::D);
        let glides_back = s.phonemes.get(v + 1) == Some(&Phoneme::Uw);
        assert_eq!(glides_back, fronted, "{text}: {:?}", s.phonemes);
        let (v0, v1) = s.ranges[v];
        let end = if glides_back { s.ranges[v + 1].1 } else { v1 };
        // The stop's own transition, its first 20 ms, falls from the
        // alveolar locus as a speaker's does; what follows is the vowel.
        let settled = v0 + (0.02 * place::SR) as usize;
        if fronted {
            // The LibriSpeech reader's /u/ after /t d/ starts at 1700 to
            // 2200 Hz and ends some 400 Hz lower ("two" 1690 to 1300, "too"
            // 1760 to 1340, "do" 1700 to 1810); the owner's "canoe" falls
            // from 1900 to 1330. The engine's back /u/ stood at 1000 to
            // 1100; gliding back to it, "do" fell from 1860 to 1140.
            let frames = oral(&s, settled, end);
            assert!(frames.len() >= 4, "{text}: {} oral frames", frames.len());
            let onset = frames[0].1[1];
            let last = frames.last().expect("frames").1[1];
            println!("{text}: F2 {onset:.0} to {last:.0} Hz");
            assert!(
                onset >= 1550.0,
                "{text}: the fronted /u/ starts at {onset:.0} Hz"
            );
            assert!(
                (100.0..=600.0).contains(&(onset - last)),
                "{text}: F2 {onset:.0} to {last:.0} Hz over the /u/"
            );
            continue;
        }
        let mid = usize::midpoint(v0, end);
        let frames = oral(&s, settled, mid + 2 * place::FRAME);
        assert!(frames.len() >= 4, "{text}: {} oral frames", frames.len());
        let target = frames.last().expect("frames").1[1];
        let highest = frames.iter().map(|f| f.1[1]).fold(0.0f32, f32::max);
        // Before round 8 the first oral frame of the /u/ of "canoe" stood
        // at 1816 Hz and the vowel's middle near 1150: a fall of 650 Hz
        // over 100 ms, while the lips rounded: /ju/.
        assert!(
            highest - target <= 200.0,
            "{text}: F2 falls from {highest:.0} to {target:.0} Hz into the /u/"
        );
        assert!(target <= 1300.0, "{text}: the /u/ holds {target:.0} Hz");
    }
}

/// Level of `x` (at `place::SR`) between `lo` and `hi` Hz, dB.
fn band(x: &[f32], lo: f32, hi: f32) -> f32 {
    let step = 10.0;
    let w: Vec<f32> = x
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let h = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / x.len() as f32).cos();
            v * h
        })
        .collect();
    let mag = spectrum::spectrum(&w, place::SR, hi + step, step);
    let lo = (lo / step) as usize;
    let power: f32 = mag[lo..].iter().map(|m| m * m).sum();
    10.0 * power.max(1e-30).log10()
}

#[test]
fn the_nose_is_shut_early_in_a_vowel_after_n() {
    // Shutting at its usual speed (some 45 ms), the velum left the first
    // 35 ms of the /u/ of "canoe" coupled to the nose: even a quarter open
    // it moves the coupled tract's loudest peak to some 700 Hz, and the
    // recognizer heard /ʌ oʊ l/ there. In the owner's "canoe" and the
    // LibriSpeech reader's "the new" the 550 to 950 Hz band stands 16 to
    // 26 dB under the 200 to 450 Hz band from 5 ms after the release; the
    // engine's stood 7 to 9 dB under until 35 ms (`Tuning::velum_release`).
    for text in ["a canoe", "the new", "the birch canoe slid"] {
        let s = place::speak(&voice(), text, false);
        let (_, v) = after(&s, Phoneme::N);
        assert_eq!(s.phonemes[v], Phoneme::Uw, "{text}: {:?}", s.phonemes);
        let v0 = s.ranges[v].0;
        let ms = |t: f32| v0 + (t * 0.001 * place::SR) as usize;
        let velum = s.frames[Spoken::frame_of(ms(15.0))].velum;
        assert!(velum < 0.1, "{text}: velum {velum:.2} 15 ms into the /u/");
        let x = &s.audio[ms(15.0)..ms(35.0)];
        let pole = band(x, 550.0, 950.0) - band(x, 200.0, 450.0);
        assert!(
            pole < -12.0,
            "{text}: 550-950 Hz {pole:.1} dB against 200-450 Hz, 15 to 35 ms into the /u/"
        );
    }
}

#[test]
fn the_first_formant_runs_flat_from_l_into_ih() {
    for text in ["slid", "lid", "a lid"] {
        let s = place::speak(&voice(), text, true);
        let (l, v) = after(&s, Phoneme::L);
        assert_eq!(s.phonemes[v], Phoneme::Ih, "{text}: {:?}", s.phonemes);
        let (l0, l1) = (s.ranges[l].0, s.ranges[v - 1].1);
        let (v0, v1) = s.ranges[v];
        let l_mid = usize::midpoint(l0, l1);
        let v_mid = usize::midpoint(v0, v1);
        let lateral = s.formants_at(l_mid, 0.0).expect("an open /l/");
        let vowel = s.formants_at(v_mid, 0.0).expect("an open /ɪ/");
        let path = oral(&s, l_mid, v_mid);
        let peak = path.iter().map(|f| f.1[0]).fold(0.0f32, f32::max);
        // Under the old /l/ body: 472 Hz in the /l/, 585 between, 456 in
        // the /ɪ/. The LibriSpeech reader's runs flat ("sleep": 311 to 328
        // Hz) or rises some 60 Hz above both ("exactly": 420, 490, 431).
        assert!(
            peak <= lateral[0].max(vowel[0]) + 60.0,
            "{text}: F1 {:.0} in the /l/, {peak:.0} between, {:.0} in the /ɪ/",
            lateral[0],
            vowel[0]
        );
        // An /l/, not a vowel: the dark second formant, and a rise of
        // its own into the front vowel.
        assert!(lateral[1] < 1450.0, "{text}: /l/ F2 {:.0}", lateral[1]);
        assert!(
            vowel[1] - lateral[1] > 500.0,
            "{text}: F2 {:.0} to {:.0}",
            lateral[1],
            vowel[1]
        );
    }
}

#[test]
fn an_l_before_a_vowel_is_a_lateral() {
    // Measured with LPC in the middle of the /l/ (the LibriSpeech reader's
    // /l/ after a consonant or between vowels, 17 of them; the owner's
    // "slid", "planks", "glue", "blue"): F1 310 to 450 Hz (the owner's 310
    // to 350), F2 950 to 1200, F3 2450 to 2900, never the low F3 of /r/;
    // in a cluster the reader holds it some 20 to 35 ms before the second
    // formant sets off for the vowel, the first rising into it. The
    // engine's /l/, a gap meant to hiss with the incisors shut in front of
    // it until round 10, stood at 490 to 500 Hz in the tract (470 by LPC),
    // above the /ɪ/ after it, with F3 near 3250, and was heard as /r/ after
    // /s/ ("slim", "sleep", "slate"). The tract reads some 30 Hz above LPC
    // on the output in the first formant.
    for text in ["slid", "a lid", "sled", "plan", "glad", "a slim chance"] {
        let s = place::speak(&voice(), text, true);
        let (l, v) = after(&s, Phoneme::L);
        let (l0, l1) = (s.ranges[l].0, s.ranges[v - 1].1);
        let (v0, v1) = s.ranges[v];
        let l_mid = usize::midpoint(l0, l1);
        let lateral = s.formants_at(l_mid, 0.0).expect("an open /l/");
        let vowel = s
            .formants_at(usize::midpoint(v0, v1), 0.0)
            .expect("a vowel");
        println!(
            "{text}: /l/ {:.0} {:.0} {:.0}, vowel {:.0} {:.0}",
            lateral[0], lateral[1], lateral[2], vowel[0], vowel[1]
        );
        assert!(lateral[0] <= 450.0, "{text}: /l/ F1 {:.0}", lateral[0]);
        // Never above the vowel's (the owner's rises from 330 into 440).
        assert!(
            lateral[0] <= vowel[0],
            "{text}: F1 {:.0} in the /l/, {:.0} in the vowel",
            lateral[0],
            vowel[0]
        );
        assert!(
            (950.0..=1350.0).contains(&lateral[1]),
            "{text}: /l/ F2 {:.0}",
            lateral[1]
        );
        assert!(
            (2450.0..=3400.0).contains(&lateral[2]),
            "{text}: /l/ F3 {:.0}",
            lateral[2]
        );
        // The lateral shape is held before the release: frames from the
        // /l/'s start on whose second formant is within 100 Hz of its
        // lowest there.
        let path = oral(&s, l0, v0);
        let low = path.iter().map(|f| f.1[1]).fold(f32::INFINITY, f32::min);
        let held = path.iter().filter(|f| f.1[1] <= low + 100.0).count();
        let held_ms = 1000.0 * (held * place::FRAME) as f32 / place::SR;
        assert!(held_ms >= 15.0, "{text}: the /l/ is held {held_ms:.0} ms");
    }
}

#[test]
fn an_s_after_a_k_hisses_at_the_end_of_an_utterance() {
    // Until round 10 the last /s/ of "planks" took the /k/'s tongue at the
    // end of an utterance, and with it the velar place: a faint hiss at
    // the velum, 50 to 56 dB under the vowel (mid-phrase it hissed).
    let rms = |x: &[f32]| (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt();
    for (text, vowel) in [
        ("planks", Phoneme::Ae),
        ("tanks", Phoneme::Ae),
        ("cakes", Phoneme::Ey),
        ("the smooth planks.", Phoneme::Ae),
    ] {
        let s = place::speak(&voice(), text, false);
        let v = s.find(vowel, 1).expect("the vowel");
        let k = s.find(Phoneme::S, v).expect("the /s/");
        let (s0, s1) = s.ranges[k];
        let (v0, v1) = s.ranges[v];
        let hiss = rms(&s.audio[s0..s1]);
        let level = 20.0 * (hiss / rms(&s.audio[v0..v1])).log10();
        println!("{text}: /s/ {level:.1} dB against the vowel");
        assert!(
            level > -35.0,
            "{text}: the final /s/ {level:.1} dB under the vowel"
        );
    }
}

/// What the larynx and the tract do through one /t d/ between vowels:
/// how long the mouth is shut, the least the folds open and close over any
/// 10 ms of it against their swing in the vowel before, and how far the
/// pressure above the folds (its mean over 10 ms, a voice period) rises
/// over the vowel's, against the lungs'.
struct Touch {
    shut_ms: f32,
    swing: f32,
    pressure: f32,
}

/// Speak `text` and measure its first `p` from 30 ms before it to 30 ms
/// after it; also how many segments it has.
fn touch(text: &str, p: Phoneme) -> (Touch, usize) {
    use synthie_trombone::speech::{Speaker, Utterance};
    use synthie_trombone::{Engine, Trombone};
    let settings = voice().with_engine(Engine::Physical);
    let mut t = Trombone::new(place::SR);
    settings.apply(&mut t);
    Speaker::prepare(&mut t);
    let u = Utterance::from_text(text, &settings.prosody());
    let segments = u.segments().to_vec();
    let k = segments
        .iter()
        .position(|s| s.phoneme == p)
        .unwrap_or_else(|| panic!("{text}: no {p}"));
    let span = segments
        .iter()
        .skip(k)
        .take_while(|s| s.phoneme == p)
        .count();
    let mut sp = Speaker::new();
    sp.speak(u);
    let margin = (0.03 * place::SR) as usize;
    let window = (0.01 * place::SR) as usize;
    let (mut log, mut at) = (Vec::new(), None::<(usize, usize)>);
    while let Some((i, _)) = sp.current() {
        let n = log.len();
        if i >= k && i < k + span {
            at = Some(at.map_or((n, n), |(a, _)| (a, n)));
        }
        let g = t.gauges();
        log.push((g.mouth_shut, g.glottal_area, g.supraglottal_pressure));
        sp.process(&mut t);
    }
    let (a, b) = at.expect("the consonant was spoken");
    let (from, to) = (a.saturating_sub(margin), (b + margin).min(log.len()));
    let swing = |r: &[(bool, f32, f32)]| {
        let (lo, hi) = r.iter().fold((f32::INFINITY, 0.0f32), |(lo, hi), x| {
            (lo.min(x.1), hi.max(x.1))
        });
        hi - lo
    };
    let vowel = swing(&log[from.saturating_sub(4 * window)..from]);
    let least = (from..to.saturating_sub(window))
        .step_by(window / 4)
        .map(|s| swing(&log[s..s + window]))
        .fold(f32::INFINITY, f32::min);
    let shut = log[from..to].iter().filter(|x| x.0).count();
    let mean = |r: &[(bool, f32, f32)]| r.iter().map(|x| x.2).sum::<f32>() / r.len() as f32;
    let before = mean(&log[from.saturating_sub(4 * window)..from]);
    let pressure = (from..to.saturating_sub(window))
        .step_by(window / 8)
        .map(|s| mean(&log[s..s + window]) - before)
        .fold(0.0f32, f32::max);
    (
        Touch {
            shut_ms: 1000.0 * shut as f32 / place::SR,
            swing: least / vowel,
            pressure: pressure / t.lung_pressure(),
        },
        span,
    )
}

#[test]
fn a_t_or_d_between_vowels_is_a_tap() {
    // A tap: a touch of the tip with the voice going on through it, no
    // pressure built behind it and nothing after it (one segment: no
    // burst, no breath). The LibriSpeech reader's last 37 to 66 ms as
    // aligned, dip 5 to 13 dB under the vowel beside them and are voiced
    // throughout ("that it", "fled in", "British", "middle").
    let taps = [
        ("slid on", Phoneme::D),
        ("get it", Phoneme::T),
        ("water", Phoneme::T),
        ("a lot of", Phoneme::T),
    ];
    let stops = [
        ("a toe", Phoneme::T),
        ("winter", Phoneme::T),
        ("get", Phoneme::T),
        ("a dog", Phoneme::D),
    ];
    for &(text, p) in taps.iter().chain(&stops) {
        let (m, span) = touch(text, p);
        println!(
            "{text}: {span} segments, shut {:.1} ms, swing {:.2}, pressure {:.2}",
            m.shut_ms, m.swing, m.pressure
        );
    }
    for (text, p) in taps {
        let (tap, span) = touch(text, p);
        assert_eq!(span, 1, "{text}: the tap is one segment");
        assert!(
            (2.0..=20.0).contains(&tap.shut_ms),
            "{text}: shut for {:.1} ms",
            tap.shut_ms
        );
        assert!(
            tap.swing > 0.3,
            "{text}: the folds' swing falls to {:.2} of the vowel's",
            tap.swing
        );
        assert!(
            tap.pressure < 0.5,
            "{text}: mouth pressure {:.2} of the lungs'",
            tap.pressure
        );
    }
    // Not before a stressed vowel, nor after a consonant, nor at the end
    // of a phrase: a stop, held shut.
    for (text, p) in stops {
        let (stop, _) = touch(text, p);
        assert!(
            stop.shut_ms > 25.0,
            "{text}: shut for {:.1} ms",
            stop.shut_ms
        );
    }
}

#[test]
fn a_phrase_opening_on_a_nasal_or_l_is_voiced_before_its_vowel() {
    // The LibriSpeech reader's opening /n m/ ("Nor", "Near", "Not",
    // "Mister") stand 8 to 20 dB under the vowel after them from their
    // first 10 ms on, for 40 to 55 ms. Until round 11 the lungs took the
    // first 40 to 45 ms of an opening /n m l/ to bring the folds going, and
    // that much of it was 33 dB under the vowel: the recognizer heard no
    // /n/ in "Nine of them" and "Not at all", and an /l/ before a front
    // vowel as /t f v/, of which only the release was left.
    let rms = |x: &[f32]| (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt();
    let window = (0.01 * place::SR) as usize;
    for (text, c) in [
        ("Nine of them.", Phoneme::N),
        ("Not at all.", Phoneme::N),
        ("More of it.", Phoneme::M),
        ("Lift it up.", Phoneme::L),
        ("Let it go.", Phoneme::L),
    ] {
        let s = place::speak(&voice(), text, false);
        let k = s.find(c, 0).expect("the consonant");
        let (_, e) = s.consonant_span(k);
        let (c0, _) = s.ranges[k];
        let (v0, v1) = s.ranges[e + 1];
        let loudest = (v0..v1.saturating_sub(window))
            .step_by(window / 2)
            .map(|i| rms(&s.audio[i..i + window]))
            .fold(0.0f32, f32::max);
        // How long before the vowel the consonant is within 20 dB of it.
        let audible = (c0..v0.saturating_sub(window))
            .step_by(window / 4)
            .find(|&i| 20.0 * (rms(&s.audio[i..i + window]) / loudest).log10() > -20.0)
            .map_or(0.0, |i| 1000.0 * (v0 - i) as f32 / place::SR);
        println!("{text}: the opening {c} is within 20 dB of the vowel for {audible:.0} ms");
        assert!(
            audible >= 35.0,
            "{text}: the opening {c} is within 20 dB of its vowel for {audible:.0} ms"
        );
    }
}

#[test]
fn a_vowel_before_a_dark_l_is_dark_already() {
    // The LibriSpeech reader's vowels before an /l/ that closes their
    // syllable have their F2 near 1000 Hz from a quarter of the way in
    // ("quilter" 1090, "welcome" 1070, "help" 1190) and fall some 70 Hz
    // into the /l/; until round 11 the engine's held their own shape up to
    // the /l/ (1460 to 1660 Hz by LPC) and fell some 500 Hz into it.
    let mut light = voice();
    light.tuning.articulation.dark_l_share = 0.0;
    let end_f2 = |settings: &VoiceSettings, text: &str, v: Phoneme| {
        let s = place::speak(settings, text, true);
        let k = s.find(v, 0).expect("the vowel");
        let l = s.find(Phoneme::L, k).expect("the /l/");
        let v1 = s.ranges[l - 1].1;
        s.formants_at(v1, -5.0).expect("an open vowel")[1]
    };
    for (text, v) in [
        ("help me", Phoneme::Eh),
        ("quilter", Phoneme::Ih),
        ("a pill", Phoneme::Ih),
        ("a sell", Phoneme::Eh),
    ] {
        let dark = end_f2(&voice(), text, v);
        let own = end_f2(&light, text, v);
        println!("{text}: F2 {dark:.0} Hz at the /l/ ({own:.0} on the vowel's own shape)");
        assert!(
            dark <= own - 150.0,
            "{text}: F2 {dark:.0} Hz at the /l/, {own:.0} on the vowel's own shape"
        );
    }
}
