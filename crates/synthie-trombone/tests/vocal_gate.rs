//! Wave 3, stage 3B: the non-verbal vocalizations and the hesitation
//! fillers, measured as what they are: a laugh's calls (their rate, their
//! falling pitch and level), a cough's pressure behind the shut folds, its
//! burst and what follows it, a sigh's breath in and out (its noise and
//! its falling pitch), the hum of a chuckle and of "hmm", the narrowed
//! tube of a throat clearing; tags in text, bracketed phonemes left alone,
//! a sentence interrupted by a cough that carries on, and everything finite
//! at the highest pressure there is.

#![allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

#[path = "../examples/support/vocal.rs"]
mod vocal;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use synthie_trombone::speech::{Filler, Speaker, Token, Utterance, phonemize};
use synthie_trombone::{Engine, Phoneme, Talker, Trombone, Vocal, Vocalization};
use vocal::{SR, calls, db, deep, envelope, median, noise_share, record, rms, voice, voiced_runs};

struct Counting;

thread_local! {
    static WATCHED: Cell<bool> = const { Cell::new(false) };
}

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call is passed on to the system allocator unchanged; the
// counting reads a thread-local flag and bumps an atomic.
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

/// A vocalization alone on the deep voice at 100 Hz, recorded with 0.3 s
/// after it.
fn alone(v: Vocal) -> (vocal::Rec, f32) {
    let mut t = voice(&deep());
    t.set_frequency(100.0);
    let seconds = t.vocalize(v).expect("the score has room");
    (record(&mut t, seconds + 0.3), seconds)
}

/// The loudest 10 ms of the deep voice saying a sentence, dB.
fn speech_peak() -> f32 {
    let settings = deep().with_engine(Engine::Physical);
    let mut t = voice(&settings);
    let mut speaker = Speaker::new();
    speaker.speak(Utterance::from_text(
        "The birch canoe slid on the smooth planks.",
        &settings.prosody(),
    ));
    let mut out = Vec::new();
    while speaker.is_speaking() {
        out.push(speaker.process(&mut t));
    }
    db(envelope(&out, 10.0, 5.0).into_iter().fold(0.0, f32::max))
}

#[test]
fn a_laugh_is_calls_at_four_to_five_and_a_half_a_second_falling_in_pitch_and_level() {
    let (r, _) = alone(Vocal::new(Vocalization::Laugh));
    assert!(r.finite);
    let c = calls(&r);
    assert!(c.len() >= 4, "calls {c:?}");
    let rate = (c.len() - 1) as f32 / (c[c.len() - 1].0 - c[0].0);
    assert!(
        (4.0..=5.5).contains(&rate),
        "{rate:.2} calls per second: {c:?}"
    );
    // Each call is a burst of voice some 60 to 200 ms long, the breath
    // between them unvoiced.
    for (_, length, _, _) in &c {
        assert!((0.06..=0.2).contains(length), "{c:?}");
    }
    let (first, last) = (c[0], c[c.len() - 1]);
    assert!(
        last.2 < 0.95 * first.2,
        "pitch {:.0} -> {:.0} Hz",
        first.2,
        last.2
    );
    assert!(
        last.3 < first.3 - 3.0,
        "level {:.1} -> {:.1} dB",
        first.3,
        last.3
    );
    // Well above speech: the voice was at 100 Hz.
    assert!(first.2 > 150.0, "{:.0} Hz", first.2);
    // Made by pulses of the lungs' pressure: each call's pressure below
    // the folds well over that between the calls.
    for w in c.windows(2) {
        let (gap_from, gap_to) = (((w[0].0 + w[0].1) * SR) as usize, (w[1].0 * SR) as usize);
        let between = r.lung[gap_from..gap_to]
            .iter()
            .copied()
            .fold(f32::MAX, f32::min);
        let during = r.lung[(w[1].0 * SR) as usize..((w[1].0 + w[1].1) * SR) as usize]
            .iter()
            .copied()
            .fold(f32::MIN, f32::max);
        assert!(
            during > between + 3.0,
            "lungs {during:.1} in a call, {between:.1} between: {c:?}"
        );
    }
}

#[test]
fn a_chuckle_hums_through_the_nose() {
    let (r, _) = alone(Vocal::new(Vocalization::Chuckle));
    assert!(r.finite);
    let c = calls(&r);
    assert!(c.len() >= 3, "{c:?}");
    // The lips are shut, the voice goes out of the nose.
    let (a, b) = voiced_runs(&r, 30.0)[0];
    let nose = rms(&r.nose[a..b]);
    let lips = rms(&r.lips[a..b]);
    assert!(nose > 5.0 * lips, "nose {nose:.1} lips {lips:.1} cm3/s");
}

#[test]
fn a_cough_builds_its_pressure_behind_shut_folds_and_bursts() {
    let tuning = deep().tuning;
    let (r, _) = alone(Vocal::new(Vocalization::Cough));
    assert!(r.finite);
    // The compression: the glottis shut while the pressure below it
    // builds to the cough's.
    let shut: Vec<usize> = (0..r.len())
        .filter(|&k| r.area[k] < 0.002 && r.sub[k] > 5.0)
        .collect();
    let held = shut.len() as f32 / SR;
    assert!(held > 0.12, "shut for {held:.3} s");
    let behind = shut.iter().map(|&k| r.sub[k]).fold(0.0, f32::max);
    let want = tuning.anatomy.cough_pressure;
    assert!(
        behind > 0.9 * want && behind < 1.1 * want,
        "{behind:.1} cmH2O behind the folds"
    );
    // Nothing comes out while they are shut (but the end of the breath in).
    let last_shut = *shut.last().unwrap();
    let quiet = rms(&r.out[last_shut - (0.08 * SR) as usize..last_shut]);
    // The burst: from the opening to its loudest 10 ms within 50 ms (the
    // expulsive phase's supramaximal flow lasts 30 to 50 ms in people), well
    // over the speech of the same voice.
    let env = envelope(&r.out[last_shut..], 10.0, 1.0);
    let (at, peak) = env
        .iter()
        .enumerate()
        .take(150)
        .fold((0, 0.0f32), |m, (k, &x)| if x > m.1 { (k, x) } else { m });
    assert!(at <= 50, "the burst peaks {at} ms after the opening");
    assert!(
        db(peak) - db(quiet) > 30.0,
        "burst {:.1} dB, shut {:.1} dB",
        db(peak),
        db(quiet)
    );
    let over = db(peak) - speech_peak();
    assert!(
        (3.0..=20.0).contains(&over),
        "burst {over:+.1} dB against speech"
    );
    // Its noise: the jets' own. The flow is high.
    let flow = r.flow[last_shut..].iter().copied().fold(0.0, f32::max);
    assert!(flow > 1500.0, "{flow:.0} cm3/s");
    // Then a voice (a breathy "uh") as the push dies away.
    let after: Vec<_> = voiced_runs(&r, 50.0)
        .into_iter()
        .filter(|&(a, _)| a > last_shut + (0.03 * SR) as usize)
        .collect();
    assert!(!after.is_empty(), "no voice after the burst");
}

#[test]
fn a_throat_clearing_is_two_weaker_voiced_coughs_through_a_narrowed_tube() {
    let mut t = voice(&deep());
    t.set_frequency(100.0);
    let rest = t.diameters()[1];
    let seconds = t.vocalize(Vocalization::ClearThroat).unwrap();
    let mut narrowest = f32::MAX;
    let mut r = vocal::Rec {
        finite: true,
        ..vocal::Rec::default()
    };
    for _ in 0..((seconds + 0.2) * SR) as usize {
        let out = t.process();
        r.push(out, &t);
        narrowest = narrowest.min(t.diameters()[1]);
    }
    assert!(r.finite);
    assert!(
        narrowest < 0.7 * rest,
        "epilarynx {narrowest:.3} against {rest:.3}"
    );
    let c = calls(&r);
    assert_eq!(c.len(), 2, "{c:?}");
    // Weaker than a cough.
    let (cough, _) = alone(Vocal::new(Vocalization::Cough));
    let peak = |x: &[f32]| envelope(x, 10.0, 5.0).into_iter().fold(0.0, f32::max);
    assert!(db(peak(&r.out)) < db(peak(&cough.out)) - 3.0);
    // Each opens from shut folds with pressure behind them.
    let shut = (0..r.len())
        .filter(|&k| r.area[k] < 0.002 && r.sub[k] > 8.0)
        .count();
    assert!(shut as f32 / SR > 0.1);
}

#[test]
fn a_sigh_breathes_in_deep_then_out_long_breathy_and_falling() {
    let (r, _) = alone(Vocal::new(Vocalization::Sigh));
    assert!(r.finite);
    // In, then out.
    let first_in = r
        .flow
        .iter()
        .position(|&f| f < -300.0)
        .expect("air goes in");
    let first_out = r
        .flow
        .iter()
        .position(|&f| f > 300.0)
        .expect("air goes out");
    assert!(first_in < first_out);
    let low = r.lung.iter().copied().fold(0.0f32, f32::min);
    assert!(low < -5.0, "lungs {low:.1} cmH2O");
    // A long breath out, voiced most of it, with much breath in the voice.
    let runs = voiced_runs(&r, 300.0);
    assert_eq!(runs.len(), 1, "{runs:?}");
    let (a, b) = runs[0];
    assert!((b - a) as f32 / SR > 0.7);
    let hz = median(&r.pitch[a..b]);
    let breathy = noise_share(&r.out[a..b], hz);
    let mut modal = voice(&deep());
    modal.articulate_phoneme(Phoneme::Aa);
    modal.set_frequency(100.0);
    modal.note_on();
    let m = record(&mut modal, 1.0);
    let clean = noise_share(&m.out[(0.3 * SR) as usize..], 100.0);
    assert!(
        breathy > clean + 0.1 && breathy > 0.2,
        "noise {breathy:.2} against {clean:.2}"
    );
    // Falling.
    let q = (b - a) / 4;
    let (start, end) = (median(&r.pitch[a..a + q]), median(&r.pitch[b - q..b]));
    assert!(end < 0.85 * start, "{start:.0} -> {end:.0} Hz");
    // High airflow on the way out.
    let out_flow = r.flow[a..b].iter().sum::<f32>() / (b - a) as f32;
    assert!(out_flow > 200.0, "{out_flow:.0} cm3/s");
}

#[test]
fn every_vocalization_stays_finite_at_the_highest_pressure() {
    let mut settings = deep();
    settings.tuning.anatomy.cough_pressure = 100.0;
    settings.tuning.anatomy.laugh_pressure = 30.0;
    for kind in Vocalization::ALL {
        let mut t = voice(&settings);
        let seconds = t.vocalize(Vocal::new(kind).with_strength(2.0)).unwrap();
        let r = record(&mut t, seconds + 0.5);
        assert!(r.finite, "{kind:?}");
        let peak = r.out.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        assert!(peak < 20.0, "{kind:?}: {peak}");
        // It dies away: nothing grows after it.
        let tail = rms(&r.out[r.len() - (0.1 * SR) as usize..]);
        assert!(tail < 1e-3, "{kind:?}: {tail}");
        assert!(r.lung.iter().all(|&p| p <= 100.5), "{kind:?}");
    }
}

#[test]
fn no_vocalization_clicks_more_than_speech_does() {
    // The largest step from one sample to the next against the sound
    // around it: the voice's own closures make steps of some 5 times the
    // RMS of their 10 ms; a click (a gap closing in a stream, a handover)
    // makes many times that.
    let settings = deep().with_engine(Engine::Physical);
    let mut t = voice(&settings);
    let mut speaker = Speaker::new();
    speaker.speak(Utterance::from_text(
        "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background.",
        &settings.prosody(),
    ));
    let mut out = Vec::new();
    while speaker.is_speaking() {
        out.push(speaker.process(&mut t));
    }
    let speech = vocal::click_ratio(&out, 50.0);
    for kind in Vocalization::ALL {
        let (r, _) = alone(Vocal::new(kind));
        let ratio = vocal::click_ratio(&r.out, 50.0);
        assert!(
            ratio < 1.6 * speech,
            "{kind:?}: {ratio:.1} against speech's {speech:.1}"
        );
    }
}

#[test]
fn tags_parse_and_bracketed_phonemes_stay_phonemes() {
    let tokens = phonemize("[laugh] [HH AX L OW1] [clear-throat] world [cough]");
    assert_eq!(tokens[0], Token::Vocal(Vocalization::Laugh));
    assert!(matches!(
        tokens[1],
        Token::Phone {
            phoneme: Phoneme::Hh,
            ..
        }
    ));
    assert!(tokens.contains(&Token::Vocal(Vocalization::ClearThroat)));
    assert_eq!(tokens.last(), Some(&Token::Vocal(Vocalization::Cough)));
    for name in ["sigh", "laugh", "chuckle", "clear-throat", "cough"] {
        assert!(
            phonemize(&format!("[{name}]"))
                .iter()
                .any(|t| matches!(t, Token::Vocal(_)))
        );
    }
    // Only the tags as written: the rest is ARPAbet as before.
    for text in ["[Laugh]", "[L AE F]", "[K ER1 L]", "[HH AX L OW1] world"] {
        assert!(
            !phonemize(text).iter().any(|t| matches!(t, Token::Vocal(_))),
            "{text}"
        );
    }
    // A tag in text is a pause segment of the event's length.
    let settings = deep().with_engine(Engine::Physical);
    let u = Utterance::from_text("Well, [sigh] I suppose so.", &settings.prosody());
    let k = (0..u.segments().len())
        .find(|&k| u.vocal_at(k).is_some())
        .unwrap();
    let v = u.vocal_at(k).unwrap();
    assert_eq!(v.kind, Vocalization::Sigh);
    assert_eq!(u.segments()[k].phoneme, Phoneme::Sil);
    let want = v.seconds(&settings.tuning);
    assert!((u.segments()[k].duration - want).abs() < 1e-6);
    let plain = Utterance::from_text("Well, I suppose so.", &settings.prosody());
    assert!((0..plain.segments().len()).all(|k| plain.vocal_at(k).is_none()));
}

#[test]
fn fillers_are_words_to_the_front_end_and_hold_low_and_level() {
    let tokens = phonemize("well um yes");
    assert!(tokens.contains(&Token::Filler(Filler::Um)));
    assert_eq!(Filler::from_word("hmm"), Some(Filler::Hmm));
    assert_eq!(Filler::from_word("umbrella"), None);
    let mut settings = deep().with_engine(Engine::Physical);
    assert_eq!(settings.tuning.prosody.fillers, 1.0, "on by default");
    settings.tuning.prosody.fillers = 0.0;
    let off = Utterance::from_text("I think um we should.", &settings.prosody());
    settings.tuning.prosody.fillers = 1.0;
    let on = Utterance::from_text("I think um we should.", &settings.prosody());
    assert!(on.duration() > off.duration() + 0.4);
    // The filler: its phones held, low and nearly level.
    let s = on.segments();
    let m = (1..s.len())
        .find(|&k| s[k].phoneme == Phoneme::M && s[k - 1].phoneme == Phoneme::Ah)
        .expect("um");
    let (ah, hum) = (on.phone_of(m - 1).unwrap(), on.phone_of(m).unwrap());
    let phones: Vec<usize> = (0..s.len())
        .filter(|&k| on.phone_of(k).is_some_and(|p| p == ah || p == hum))
        .collect();
    assert_eq!(s[phones[0]].phoneme, Phoneme::Ah);
    let held: f32 = phones.iter().map(|&k| s[k].duration).sum();
    let p = &settings.tuning.prosody;
    assert!((held - 1.25 * p.filler_length).abs() < 0.02, "{held}");
    let base = settings.prosody().pitch;
    let (start, end) = (s[phones[0]].pitch.1, s[*phones.last().unwrap()].pitch.1);
    assert!(
        start < 0.92 * base && end < start && end > 0.85 * start,
        "{start} {end} of {base}"
    );
    // The hum: the nose open, the lips shut; then a pause.
    let m = *phones.last().unwrap();
    assert_eq!(s[m].phoneme, Phoneme::M);
    assert!(s[m].articulation.velum > 0.5);
    // (less the head start a phrase opening on a sonorant takes from it,
    // `sonorant_onset`).
    let onset = settings.tuning.prosody.sonorant_onset;
    assert!(s[m + 1].phoneme == Phoneme::Sil && s[m + 1].duration >= p.filler_pause - onset - 1e-4);
}

#[test]
fn tags_and_fillers_sit_with_the_breaths_and_the_creak() {
    // The living voice's defaults: a breath between sentences, a statement
    // creaking at its end.
    let mut settings = deep().with_engine(Engine::Physical);
    settings.tuning.prosody.fillers = 1.0;
    assert!(settings.tuning.prosody.breath_intake > 0.0 && settings.tuning.prosody.final_fry > 0.0);
    let u = Utterance::from_text(
        "I suppose so. [sigh] That was fine, um, I think. And then we left.",
        &settings.prosody(),
    );
    let s = u.segments();
    let sigh = (0..s.len()).find(|&k| u.vocal_at(k).is_some()).unwrap();
    // The sigh brings its own breath in: no breath of the speech's in the
    // pauses on either side of it, up to the phones around it.
    let before = (0..sigh)
        .rev()
        .find(|&k| s[k].phoneme != Phoneme::Sil)
        .unwrap();
    let after = (sigh..s.len())
        .find(|&k| s[k].phoneme != Phoneme::Sil)
        .unwrap();
    assert!((before..after).all(|k| s[k].gesture.intake == 0.0));
    // The sentences around it breathe still: a breath before the last.
    assert!(s[after..].iter().any(|x| x.gesture.intake > 0.0));
    // The statement before the tag creaks at its end; the filler does not
    // (it ends no statement).
    assert!(s[..=before].iter().any(|x| x.gesture.creak > 0.0));
    let um = (1..s.len())
        .find(|&k| s[k].phoneme == Phoneme::M && s[k - 1].phoneme == Phoneme::Ah)
        .unwrap();
    assert_eq!(s[um].gesture.creak, 0.0);
    assert_eq!(s[um - 1].gesture.creak, 0.0);
    // A filler that ends a statement creaks with it.
    let hmm = Utterance::from_text("Hmm.", &settings.prosody());
    assert!(
        hmm.segments()
            .iter()
            .any(|x| x.phoneme == Phoneme::M && x.gesture.creak > 0.0)
    );
}

#[test]
fn hmm_is_hummed_through_the_nose_with_the_mouth_shut() {
    let mut settings = deep().with_engine(Engine::Physical);
    settings.tuning.prosody.fillers = 1.0;
    let mut t = voice(&settings);
    let mut speaker = Speaker::new();
    let u = Utterance::from_text("Hmm.", &settings.prosody());
    let m = (0..u.segments().len())
        .find(|&k| u.segments()[k].phoneme == Phoneme::M)
        .unwrap();
    let from: f32 = u.segments()[..m].iter().map(|s| s.duration).sum();
    let length = u.segments()[m].duration;
    speaker.speak(u);
    let mut r = vocal::Rec {
        finite: true,
        ..vocal::Rec::default()
    };
    while speaker.is_speaking() {
        let out = speaker.process(&mut t);
        r.push(out, &t);
    }
    let (a, b) = (
        ((from + 0.1) * SR) as usize,
        ((from + length - 0.05) * SR) as usize,
    );
    let nose = rms(&r.nose[a..b]);
    let lips = rms(&r.lips[a..b]);
    assert!(
        nose > 20.0 && nose > 10.0 * lips,
        "nose {nose:.1} lips {lips:.1} cm3/s"
    );
    // Voiced and low.
    let hz = median(&r.pitch[a..b]);
    assert!(hz > 0.0 && hz < settings.prosody().pitch, "{hz:.0} Hz");
}

#[test]
fn a_cough_pushed_mid_sentence_leaves_the_voice_speaking_afterwards() {
    let settings = deep().with_engine(Engine::Physical);
    let text = "The birch canoe slid on the smooth planks.";
    let mut plain = Talker::new(SR, settings);
    let alone = plain.bake(text);
    let mut talker = Talker::new(SR, settings);
    talker.say(text);
    let mut out = Vec::new();
    let mut cough = 0.0;
    while talker.is_speaking() {
        if out.len() == (0.8 * SR) as usize {
            cough = talker.vocalize(Vocalization::Cough).unwrap();
        }
        out.push(talker.process());
        assert!(out.len() < (8.0 * SR) as usize, "never ends");
    }
    // The sentence waited for the cough and was all said after it.
    let grown = (out.len() as f32 - (alone.len() as f32 - 0.25 * SR)) / SR;
    assert!(
        (grown - cough).abs() < 0.02,
        "{grown:.3} s longer for a {cough:.3} s cough"
    );
    // The voice speaks after the cough: the stretch after it is as loud
    // as the same stretch of the sentence alone.
    let resume = ((0.8 + cough + 0.1) * SR) as usize;
    let n = (0.8 * SR) as usize;
    let after = rms(&out[resume..resume + n]);
    let same = rms(&alone[(0.9 * SR) as usize..(0.9 * SR) as usize + n]);
    assert!(
        (db(after) - db(same)).abs() < 3.0,
        "{:.1} against {:.1} dB",
        db(after),
        db(same)
    );
    assert!(out.iter().all(|x| x.is_finite()));
}

#[test]
fn tags_in_text_play_without_allocating_and_the_speech_goes_on() {
    let settings = deep().with_engine(Engine::Physical);
    let u = Utterance::from_text("Well, [laugh] that was close.", &settings.prosody());
    let laugh = (0..u.segments().len())
        .find(|&k| u.vocal_at(k).is_some())
        .unwrap();
    let start: f32 = u.segments()[..laugh].iter().map(|s| s.duration).sum();
    let length = u.segments()[laugh].duration;
    let total = u.duration();
    let mut t = voice(&settings);
    let mut speaker = Speaker::new();
    speaker.speak(u);
    let mut r = vocal::Rec {
        finite: true,
        ..vocal::Rec::default()
    };
    ALLOCATIONS.store(0, Ordering::Relaxed);
    while speaker.is_speaking() {
        WATCHED.with(|w| w.set(true));
        let out = speaker.process(&mut t);
        WATCHED.with(|w| w.set(false));
        r.push(out, &t);
    }
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
    assert!(r.finite);
    let laugh_calls = calls(&r)
        .into_iter()
        .filter(|c| c.0 > start && c.0 < start + length)
        .count();
    assert!(laugh_calls >= 4, "{laugh_calls} calls");
    // Speech after it: voiced in the last phrase.
    let after = voiced_runs(&r, 60.0)
        .into_iter()
        .filter(|&(a, _)| a as f32 / SR > start + length)
        .count();
    assert!(after >= 2, "{after} voiced runs after the laugh");
    assert!((r.len() as f32 / SR - total).abs() < 0.1);
}

#[test]
fn the_stream_carries_a_cough_into_the_speech_it_plays() {
    use synthie_trombone::stream::{When, voice_stream};
    let settings = deep().with_engine(Engine::Physical);
    let run = |cough: bool| {
        let (mut stream, mut player) = voice_stream(4096);
        stream.set_prosody(settings.prosody());
        let mut t = voice(&settings);
        stream.push_text("The birch canoe slid on the smooth planks.");
        stream.flush_text();
        let mut out = Vec::new();
        let at = (0.8 * SR) as u64;
        if cough {
            stream
                .vocalize(
                    Vocal::new(Vocalization::Cough).with_strength(0.8),
                    When::At(at),
                )
                .unwrap();
        }
        let mut quiet = 0;
        while quiet < (0.3 * SR) as usize && out.len() < (8.0 * SR) as usize {
            out.push(player.process(&mut t));
            stream.collect();
            quiet = if player.is_idle() { quiet + 1 } else { 0 };
        }
        out
    };
    let plain = run(false);
    let with = run(true);
    let cough = Vocal::new(Vocalization::Cough)
        .with_strength(0.8)
        .seconds(&settings.tuning);
    let grown = (with.len() as f32 - plain.len() as f32) / SR;
    assert!(
        (grown - cough).abs() < 0.02,
        "{grown:.3} s against {cough:.3}"
    );
    // The burst is in it.
    let peak = |x: &[f32]| x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(peak(&with) > 1.5 * peak(&plain));
}

#[test]
fn a_vocalization_can_be_started_again_and_again() {
    let mut t = voice(&deep());
    for k in 0..200 {
        let kind = Vocalization::ALL[k % Vocalization::ALL.len()];
        assert!(t.vocalize(kind).is_some(), "refused at {k}");
        for _ in 0..(k % 7) * 480 {
            t.process();
        }
    }
    assert!(t.is_vocalizing());
    t.stop_vocalizing();
    assert!(!t.is_vocalizing());
    let r = record(&mut t, 0.5);
    assert!(r.finite);
}

#[test]
fn a_vocalization_is_refused_on_the_classic_engine() {
    let mut t = Trombone::new(SR);
    assert_eq!(t.vocalize(Vocalization::Laugh), None);
    assert!(!t.is_vocalizing());
}
