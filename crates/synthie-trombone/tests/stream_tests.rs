//! The voice stream: events at their samples, a full stream that refuses
//! and never blocks, gestures, and text arriving in chunks spoken as the
//! same text arriving whole.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::time::Instant;
use synthie_trombone::gesture::{Target, var};
use synthie_trombone::speech::{Prosody, Speaker, Utterance};
use synthie_trombone::stream::{
    Event, Full, Granularity, StreamBuffer, UTTERANCES, When, voice_stream,
};
use synthie_trombone::{Constriction, Engine, Phoneme, Preset, Trombone};
use synthie_trombone::{Gesture, Tier};

const SR: f32 = 48_000.0;

/// A gesture opening the velar port to `port`.
fn velum(port: f32) -> Gesture {
    Gesture::new(Tier::Velum, Target::NONE.with(var::velum::PORT, port)).dominance(2.0)
}

#[test]
fn timed_events_land_on_their_sample() {
    let (mut stream, mut player) = voice_stream(256);
    let mut voice = Trombone::new(SR);
    stream.push_at(1_000, Event::NoteOn(57.0)).unwrap();
    stream.push_at(1_500, Event::NoteOff).unwrap();
    // As soon as possible, but after the timed one before it.
    stream.push(Event::Velum(0.7)).unwrap();
    for n in 0..2_000u64 {
        assert_eq!(player.now(), n);
        player.process(&mut voice);
        // After sample n has been rendered, every event due before it
        // was carried out before it.
        assert_eq!(voice.is_voiced(), (1_000..1_500).contains(&n), "sample {n}");
        if n < 1_500 {
            assert!(voice.velum_target() < 0.01, "sample {n}");
        } else {
            assert!((voice.velum_target() - 0.7).abs() < 1e-6, "sample {n}");
        }
    }
    assert_eq!(stream.now(), 2_000);
    assert!(player.is_idle());
}

#[test]
fn an_event_in_the_past_is_carried_out_at_once() {
    let (mut stream, mut player) = voice_stream(64);
    let mut voice = Trombone::new(SR);
    let mut buf = [0.0; 480];
    player.render(&mut voice, &mut buf);
    stream.push_at(10, Event::Pitch(220.0)).unwrap();
    player.process(&mut voice);
    assert!((voice.target_frequency() - 220.0).abs() < 1e-3);
}

#[test]
fn every_event_survives_the_wire() {
    let events = [
        Event::NoteOn(61.5),
        Event::NoteOff,
        Event::Note(40.0),
        Event::Pitch(97.0),
        Event::Voiced(true),
        Event::AlwaysVoice(true),
        Event::Tenseness(0.3),
        Event::Lung(11.0),
        Event::Vibrato {
            amount: 0.02,
            hz: 6.0,
        },
        Event::Glide(0.5),
        Event::Tongue {
            index: 20.0,
            diameter: 2.5,
        },
        Event::Constriction {
            slot: 2,
            constriction: Some(Constriction::new(36.0, 0.4)),
        },
        Event::Constriction {
            slot: 3,
            constriction: None,
        },
        Event::Velum(0.4),
        Event::Articulate(Phoneme::Sh),
        Event::Shape(synthie_trombone::inverse::Shape {
            tongue_index: 20.0,
            tongue_diameter: 2.4,
            lips: Some(0.8),
            narrowing: None,
            pharynx: 0.1,
        }),
        Event::TractLength(0.8),
        Event::ArticulationSpeed(2.0),
        Event::Gesture(velum(0.5)),
        Event::StopSpeaking,
    ];
    // A voice driven through the stream ends where one driven directly
    // does.
    let (mut stream, mut player) = voice_stream(1024);
    let mut a = Trombone::new(SR);
    let mut b = Trombone::new(SR);
    a.set_engine(Engine::Physical);
    b.set_engine(Engine::Physical);
    for e in events {
        stream.push(e).unwrap();
    }
    player.process(&mut a);
    b.set_note(61.5);
    b.note_on();
    b.note_off();
    b.set_note(40.0);
    b.set_frequency(97.0);
    b.set_voiced(true);
    b.set_always_voice(true);
    b.set_tenseness(0.3);
    b.set_lung_pressure(11.0);
    b.set_vibrato(0.02, 6.0);
    b.set_glide(0.5);
    b.set_tongue(20.0, 2.5);
    b.set_constriction(2, Some(Constriction::new(36.0, 0.4)));
    b.set_constriction(3, None);
    b.set_velum(0.4);
    b.articulate_phoneme(Phoneme::Sh);
    b.set_shape(&synthie_trombone::inverse::Shape {
        tongue_index: 20.0,
        tongue_diameter: 2.4,
        lips: Some(0.8),
        narrowing: None,
        pharynx: 0.1,
    });
    b.set_tract_length(0.8);
    b.set_articulation_speed(2.0);
    b.push_gesture(velum(0.5));
    b.process();
    let mut x = vec![0.0; 9_600];
    let mut y = vec![0.0; 9_600];
    player.render(&mut a, &mut x);
    b.render(&mut y);
    assert_eq!(x, y, "the streamed voice sounds different");
    assert_eq!(a.target_diameters(), b.target_diameters());
}

#[test]
fn a_full_stream_refuses_hands_back_and_recovers() {
    let mut buffer = StreamBuffer::<32>::new();
    let (mut stream, mut player) = buffer.split();
    let mut voice = Trombone::new(SR);
    let mut accepted = 0u16;
    let refused = loop {
        let started = Instant::now();
        let r = stream.push(Event::Pitch(100.0 + f32::from(accepted)));
        assert!(started.elapsed().as_millis() < 5, "a push waited");
        match r {
            Ok(()) => accepted += 1,
            Err(full) => break full,
        }
    };
    // A pitch event is two words: 32 hold 16.
    assert_eq!(accepted, 16);
    assert_eq!(refused, Full(Event::Pitch(116.0)));
    assert_eq!(stream.free(), 0);
    let Err(Full(g)) = stream.push_gesture(velum(1.0), When::Now) else {
        panic!("a full stream took a gesture");
    };
    assert_eq!(g, velum(1.0));
    // The player takes all that was queued, in order, and there is room
    // again.
    player.process(&mut voice);
    assert!((voice.target_frequency() - 115.0).abs() < 1e-3);
    assert_eq!(stream.free(), 32);
    stream.push(Event::Pitch(300.0)).unwrap();
    player.process(&mut voice);
    assert!((voice.target_frequency() - 300.0).abs() < 1e-3);
}

#[test]
fn a_no_std_buffer_carries_gestures_between_threads() {
    let mut buffer = StreamBuffer::<1024>::new();
    let (mut stream, mut player) = buffer.split();
    std::thread::scope(|s| {
        s.spawn(move || {
            let mut sent = 0;
            while sent < 200 {
                let target = f32::from(u8::try_from(sent % 10).unwrap()) / 9.0;
                let g = velum(target);
                if stream.push_gesture(g, When::Now).is_ok() {
                    sent += 1;
                } else {
                    std::thread::yield_now();
                }
            }
            while stream.push(Event::NoteOn(48.0)).is_err() {
                std::thread::yield_now();
            }
        });
        let mut voice = Trombone::new(SR);
        let started = Instant::now();
        while !voice.is_voiced() {
            assert!(
                started.elapsed().as_secs() < 10,
                "the gestures never arrived"
            );
            player.process(&mut voice);
        }
        // The gestures reached the score, and the score the velum.
        let velum = voice.tier_activity()[Tier::Velum.index()];
        assert!(velum.activation > 0.5 && velum.gestures > 0, "{velum:?}");
        assert!(voice.velum_target() > 0.2, "{}", voice.velum_target());
    });
}

/// Seconds of sound a speaker playing `utterances` one after another
/// takes until it falls silent, and the sound.
fn speak_all(utterances: Vec<Utterance>, engine: Engine) -> Vec<f32> {
    let mut voice = Trombone::new(SR);
    voice.set_engine(engine);
    let mut speaker = Speaker::new();
    Speaker::prepare(&mut voice);
    let mut out = Vec::new();
    for u in utterances {
        speaker.speak(u);
        while speaker.is_speaking() {
            out.push(speaker.process(&mut voice));
        }
    }
    out
}

const TEXT: &str = "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue \
                    background. She sells sea shells by the sea shore.";

/// The text as a language model might send it: a few characters at a time,
/// cut anywhere, words included.
fn chunks(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = text;
    let mut k = 0;
    while !rest.is_empty() {
        let mut n = (3 + (k * 7) % 9).min(rest.len());
        while !rest.is_char_boundary(n) {
            n += 1;
        }
        out.push(&rest[..n]);
        rest = &rest[n..];
        k += 1;
    }
    out
}

/// Render a stream fed with `chunks` (one chunk every 5 ms) until it falls
/// silent: the sound from the first sample of speech on, and each
/// utterance the speaker took up.
fn stream_text(
    chunks: &[&str],
    prosody: &Prosody,
    granularity: Granularity,
) -> (Vec<f32>, Vec<Utterance>) {
    let (mut stream, mut player) = voice_stream(1024);
    stream.set_prosody(*prosody);
    stream.set_granularity(granularity);
    let mut voice = Trombone::new(SR);
    voice.set_engine(prosody.engine);
    let mut out = Vec::new();
    let mut said: Vec<Utterance> = Vec::new();
    let mut fed = 0;
    loop {
        if out.len() % 240 == 0 && fed < chunks.len() {
            stream.push_text(chunks[fed]);
            fed += 1;
            if fed == chunks.len() {
                stream.flush_text();
            }
        }
        out.push(player.process(&mut voice));
        if let Some(sp) = player.speaker()
            && sp.is_speaking()
            && said.last() != Some(sp.utterance())
        {
            said.push(sp.utterance().clone());
        }
        if fed == chunks.len() && player.is_idle() {
            break;
        }
        assert!(out.len() < 60 * 48_000, "the stream never fell silent");
    }
    (trimmed(out), said)
}

/// Seconds of sound from the first sample of speech to its end.
fn speech_seconds(x: &[f32]) -> f32 {
    let last = x.iter().rposition(|s| s.abs() > 1e-4).unwrap_or(0);
    last as f32 / SR
}

fn trimmed(mut x: Vec<f32>) -> Vec<f32> {
    let first = x.iter().position(|s| s.abs() > 1e-4).unwrap_or(0);
    x.drain(..first);
    x
}

#[test]
fn chunked_text_is_spoken_as_the_text_whole() {
    let sentences = [
        "The birch canoe slid on the smooth planks.",
        " Glue the sheet to the dark blue background.",
        " She sells sea shells by the sea shore.",
    ];
    // The physical engine also with speech through the gesture score.
    for (engine, score) in [
        (Engine::Classic, 0.0),
        (Engine::Physical, 0.0),
        (Engine::Physical, 1.0),
    ] {
        let mut settings = Preset::Deep.settings().with_engine(engine);
        settings.tuning.articulation.gesture_score = score;
        let prosody = settings.prosody();
        let (streamed, said) = stream_text(&chunks(TEXT), &prosody, Granularity::Sentence);
        // Each sentence is spoken as it is compiled on its own, however
        // the chunks cut it: nothing resets where they meet.
        let alone: Vec<Utterance> = sentences
            .iter()
            .map(|s| Utterance::from_text(s, &prosody))
            .collect();
        assert_eq!(said, alone, "{engine:?}");
        // And the pauses between them are the text's own: the whole is as
        // long as the text spoken as one utterance.
        let whole = trimmed(speak_all(
            vec![Utterance::from_text(TEXT, &prosody)],
            engine,
        ));
        let apart = trimmed(speak_all(alone, engine));
        println!(
            "{engine:?}, score {score}: streamed {:.3} s, the text as one utterance {:.3} s, \
             its sentences back to back {:.3} s",
            speech_seconds(&streamed),
            speech_seconds(&whole),
            speech_seconds(&apart)
        );
        assert!(
            (speech_seconds(&streamed) - speech_seconds(&whole)).abs() < 0.02,
            "{engine:?}: {} against {}",
            speech_seconds(&streamed),
            speech_seconds(&whole)
        );
    }
}

#[test]
fn phrase_by_phrase_keeps_the_pauses_too() {
    let text = "When the sunlight strikes raindrops in the air, they act as a prism, and \
                form a rainbow.";
    for engine in [Engine::Classic, Engine::Physical] {
        let prosody = Preset::Deep.settings().with_engine(engine).prosody();
        let (streamed, said) = stream_text(&chunks(text), &prosody, Granularity::Phrase);
        assert_eq!(said.len(), 3, "{engine:?}");
        let whole = trimmed(speak_all(
            vec![Utterance::from_text(text, &prosody)],
            engine,
        ));
        println!(
            "{engine:?}, phrase by phrase: streamed {:.3} s, as one utterance {:.3} s",
            speech_seconds(&streamed),
            speech_seconds(&whole)
        );
        assert!(
            (speech_seconds(&streamed) - speech_seconds(&whole)).abs() < 0.05,
            "{engine:?}: {} against {}",
            speech_seconds(&streamed),
            speech_seconds(&whole)
        );
    }
}

#[test]
fn phrases_go_out_as_soon_as_they_end() {
    let prosody = Prosody::default();
    let (mut stream, mut player) = voice_stream(1024);
    stream.set_prosody(prosody);
    stream.set_granularity(Granularity::Phrase);
    let sent = stream.push_text("Well, I thi");
    assert_eq!((sent.utterances, sent.held), (1, " I thi".len()));
    let sent = stream.push_text("nk so. Yes");
    assert_eq!(sent.utterances, 1);
    assert_eq!(stream.held_text(), " Yes");
    let sent = stream.flush_text();
    assert_eq!((sent.utterances, sent.held), (1, 0));
    let mut voice = Trombone::new(SR);
    let mut said = 0;
    let mut was = false;
    for _ in 0..(10.0 * SR) as usize {
        player.process(&mut voice);
        let now = player.speaker().is_some_and(Speaker::is_speaking);
        if now && !was {
            said += 1;
        }
        was = now;
        if player.is_idle() {
            break;
        }
    }
    assert!(player.is_idle());
    assert!(said >= 1);
}

#[test]
fn too_much_speech_is_held_back_not_lost() {
    let (mut stream, mut player) = voice_stream(1024);
    let many: String = (0..UTTERANCES + 5).fold(String::new(), |mut s, k| {
        use std::fmt::Write as _;
        let _ = write!(s, "Sentence {k}. ");
        s
    });
    let sent = stream.push_text(&many);
    assert_eq!(sent.utterances, UTTERANCES);
    assert!(sent.full && sent.held > 0);
    let u = Utterance::from_text("one more", &Prosody::default());
    assert!(matches!(stream.queue(u, When::Now), Err(Full(_))));
    // Once the player has begun the queue there is room again.
    let mut voice = Trombone::new(SR);
    let mut spoken = 0;
    let mut buf = [0.0; 480];
    for _ in 0..2_000 {
        player.render(&mut voice, &mut buf);
        spoken += stream.push_text("").utterances;
        if player.is_idle() && stream.held_text().is_empty() {
            break;
        }
    }
    assert_eq!(spoken, 5);
    assert!(player.is_idle());
}

#[test]
fn stop_speaking_drops_the_queue() {
    let (mut stream, mut player) = voice_stream(1024);
    stream.push_text("One sentence here. And another one. And a third. ");
    let mut voice = Trombone::new(SR);
    let mut buf = vec![0.0; 4_800];
    player.render(&mut voice, &mut buf);
    assert!(!player.is_idle());
    stream.push(Event::StopSpeaking).unwrap();
    player.render(&mut voice, &mut buf);
    assert!(player.is_idle());
    // An interrupting utterance replaces what is said.
    stream.push_text("Something long to say, on and on and on. ");
    player.render(&mut voice, &mut buf);
    let u = Utterance::from_text("Stop.", &Prosody::default());
    let segments = u.segments().len();
    stream.speak(u, When::Now).unwrap();
    player.process(&mut voice);
    assert_eq!(
        player.speaker().unwrap().utterance().segments().len(),
        segments
    );
}

#[test]
fn compiling_a_sentence_is_quick() {
    let prosody = Preset::Deep
        .settings()
        .with_engine(Engine::Physical)
        .prosody();
    let sentence = "The birch canoe slid on the smooth planks.";
    let _ = Utterance::from_text(sentence, &prosody);
    let n = 50;
    let t0 = Instant::now();
    for _ in 0..n {
        let _ = Utterance::from_text(sentence, &prosody);
    }
    let each = t0.elapsed().as_secs_f64() / f64::from(n) * 1e3;
    println!("compiling {sentence:?}: {each:.2} ms");
    assert!(each < 20.0, "{each} ms");
}
