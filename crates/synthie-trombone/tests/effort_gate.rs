//! The vocal effort states (hybrid roadmap, stage 3), measured as what
//! they are: normal changes nothing; each state has its own level, pitch,
//! spectral balance and lungs; a whisper's folds never meet; a nervous
//! voice trembles at 7 Hz; tags switch the state where they stand and a
//! stream carries it; whatever is thrown at the voice in any order (states,
//! morphs, coughs, block lengths) it stays finite, bounded and speaking;
//! nothing allocates.

#![allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp
)]

#[path = "../examples/support/effort.rs"]
mod effort;

use effort::{Rec, SR, level, median, rms, settings, signature, speak, tremor, voiced_share};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use synthie_trombone::stream::{Event, voice_stream};
use synthie_trombone::{
    Effort, Engine, MorphTarget, Phoneme, Preset, Talker, Trombone, Tuning, Vocalization,
};

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

/// A sentence of vowels, glides and nasals: the voice runs through it.
const VOICED: &str = "We were away a year ago. Mary may win more money.";

fn spoken(e: Effort) -> Rec {
    speak(&settings(e), VOICED, |_, _| {})
}

#[test]
fn normal_changes_nothing() {
    let plain = Preset::Deep.settings().with_engine(Engine::Physical);
    assert_eq!(plain.with_effort(Effort::Normal), plain);
    assert_eq!(Effort::of(&Tuning::DEFAULT), Effort::Normal);
    let text = "She sells sea shells by the sea shore. Glue the sheet.";
    let a = speak(&plain, text, |_, _| {});
    let b = speak(&plain.with_effort(Effort::Normal), text, |t, _| {
        t.set_effort(Effort::Normal);
    });
    assert!(a.out == b.out, "normal moved the sound");
    // A tag for the state the voice is in is no sound and no pause.
    let c = speak(
        &plain,
        "[normal] She sells sea shells by the sea shore. [normal] Glue the sheet.",
        |_, _| {},
    );
    assert!(a.out == c.out, "a [normal] tag moved the sound");
    // The classic engine has no regime for a tag to switch.
    let classic = Preset::Deep.settings();
    let d = speak(&classic, "Hello there.", |_, _| {});
    let e = speak(&classic, "[shout] Hello there.", |_, _| {});
    assert!(d.out == e.out, "a tag moved the classic engine's sound");
}

#[test]
fn each_state_has_its_own_signature() {
    // Measured (examples/effort.rs), the deep voice, level / pitch / 2.5-5
    // kHz against 0-1 kHz / lungs / seconds:
    //   normal    -25.6 dB  116 Hz  -24.7 dB   6.2 cmH2O  4.10 s
    //   whisper   -39.4 dB  (none)    0.0 dB   7.7        4.10
    //   shout     -16.0 dB  145 Hz  -20.0 dB  12.9        4.34
    //   exhausted -30.6 dB   88 Hz  -25.2 dB   4.6        4.97
    //   nervous   -24.9 dB  127 Hz  -24.5 dB   6.2        4.13
    let n = signature(&spoken(Effort::Normal));
    let w = signature(&spoken(Effort::Whisper));
    let s = signature(&spoken(Effort::Shout));
    let x = signature(&spoken(Effort::Exhausted));
    let v = signature(&spoken(Effort::Nervous));
    let semitones = |a: f32, b: f32| 12.0 * (a / b).log2();

    // A shout: 6 dB and more louder, the lungs at 14 to 18 cmH2O at their
    // fullest, four semitones up, a flatter spectrum, a wider jaw.
    assert!(s.level > n.level + 6.0, "shout {s:?} against {n:?}");
    assert!(
        (3.0..5.0).contains(&semitones(s.pitch, n.pitch)),
        "shout {s:?} against {n:?}"
    );
    assert!(s.tilt > n.tilt + 2.5, "shout {s:?} against {n:?}");
    assert!(s.h1_h2 < n.h1_h2 - 0.5, "shout {s:?} against {n:?}");
    assert!(s.jaw > 1.2 * n.jaw, "shout {s:?} against {n:?}");
    let lungs = spoken(Effort::Shout)
        .lung
        .iter()
        .fold(0.0f32, |m, &p| m.max(p));
    assert!(
        (14.0..=18.0).contains(&lungs),
        "a shout's lungs at {lungs:.1} cmH2O"
    );

    // Exhausted: 4 dB and more quieter, the lungs at 5 to 6 cmH2O at their
    // fullest, the pitch sunk, the sentence slower.
    assert!(x.level < n.level - 4.0, "exhausted {x:?} against {n:?}");
    assert!(
        semitones(x.pitch, n.pitch) < -1.0,
        "exhausted {x:?} against {n:?}"
    );
    assert!(
        x.seconds > 1.12 * n.seconds,
        "exhausted {x:?} against {n:?}"
    );
    let lungs = spoken(Effort::Exhausted)
        .lung
        .iter()
        .fold(0.0f32, |m, &p| m.max(p));
    assert!(
        (5.0..=6.0).contains(&lungs),
        "an exhausted speaker's lungs at {lungs:.1} cmH2O"
    );

    // A whisper: no voice, a spectrum without the voice's slope, quieter
    // than speech and louder than nothing.
    assert!(w.voiced < 0.1, "whisper {w:?}");
    assert!(w.tilt > n.tilt + 15.0, "whisper {w:?} against {n:?}");
    assert!(
        w.level < n.level - 8.0 && w.level > n.level - 25.0,
        "whisper {w:?} against {n:?}"
    );

    // Nervous: the level and the spectrum of the voice as it is, a little
    // higher, and trembling (its own test below).
    assert!(
        (v.level - n.level).abs() < 3.0,
        "nervous {v:?} against {n:?}"
    );
    assert!(
        (0.5..3.0).contains(&semitones(v.pitch, n.pitch)),
        "nervous {v:?} against {n:?}"
    );

    // No two states sound alike: each pair differs by 4 dB of level, a
    // semitone and a half of pitch, 3 dB of spectral balance, or by
    // whether there is a voice or a tremor at all.
    let all = [
        ("normal", n),
        ("whisper", w),
        ("shout", s),
        ("exhausted", x),
        ("nervous", v),
    ];
    for (i, (a_name, a)) in all.iter().enumerate() {
        for (b_name, b) in &all[i + 1..] {
            let apart = (a.level - b.level).abs() > 4.0
                || (a.tilt - b.tilt).abs() > 3.0
                || (a.voiced > 0.5) != (b.voiced > 0.5)
                || (a.voiced > 0.5 && semitones(a.pitch, b.pitch).abs() > 1.5)
                || (a.tremor.1 - b.tremor.1).abs() > 0.012;
            assert!(apart, "{a_name} and {b_name} sound alike: {a:?} {b:?}");
        }
    }
}

#[test]
fn a_whispers_folds_never_meet() {
    let rec = spoken(Effort::Whisper);
    // From the first sound to the last the glottis stands open: a slit,
    // not a breath's width, and never shut.
    let loud: Vec<usize> = (0..rec.out.len() / 480)
        .filter(|&k| rms(&rec.out[k * 480..(k + 1) * 480]) > 0.1 * rms(&rec.out))
        .collect();
    let (first, last) = (loud[0] * 480, (loud[loud.len() - 1] + 1) * 480);
    let area = &rec.area[first..last];
    let narrowest = area.iter().fold(f32::MAX, |m, &a| m.min(a));
    let mut sorted = area.to_vec();
    sorted.sort_by(f32::total_cmp);
    let typical = sorted[sorted.len() / 2];
    // (Measured: 0.25 cm², a slit of 0.05 between the edges and the chink
    // behind them; a voice's glottis shuts in every cycle.)
    assert!(narrowest > 0.03, "the folds met: {narrowest} cm²");
    assert!((0.1..0.4).contains(&typical), "the chink is {typical} cm²");
    let voice = spoken(Effort::Normal);
    let shut = voice.area.iter().filter(|&&a| a < 0.01).count();
    assert!(shut > voice.area.len() / 10, "a voice's glottis shuts");
    // Air rushes through it faster than through a voice's glottis on
    // average: the jet that makes the sound.
    let flow = rec.flow[first..last].iter().sum::<f32>() / (last - first) as f32;
    let voice_flow = voice.flow.iter().sum::<f32>() / voice.flow.len() as f32;
    assert!(
        flow > 1.2 * voice_flow,
        "whisper {flow:.0} cm³/s, voice {voice_flow:.0}"
    );
    assert!(voiced_share(&rec.pitch) < 0.1);
}

#[test]
fn a_nervous_voice_trembles_at_seven_hertz() {
    // Measured: the pitch's 6 to 8.5 Hz tremor is 0.04 of it at 7.0 Hz
    // (normal: 0.017 at 6.75 Hz, the syllables' own rhythm), the lungs'
    // deeper.
    let n = spoken(Effort::Normal);
    let v = spoken(Effort::Nervous);
    let (rate, depth) = tremor(&v.pitch);
    assert!((6.5..=7.5).contains(&rate), "tremor at {rate} Hz");
    assert!(depth > 0.03 && depth < 0.07, "tremor depth {depth}");
    assert!(
        depth > 1.8 * tremor(&n.pitch).1,
        "normal trembles {:?}",
        tremor(&n.pitch)
    );
    let (lung_rate, lung_depth) = tremor(&v.lung);
    assert!(
        (6.5..=7.5).contains(&lung_rate),
        "the lungs tremble at {lung_rate} Hz"
    );
    // (The lungs follow what they are asked within 30 ms, which takes
    // some of a 7 Hz tremor: 0.08 of their pressure against 0.04 of the
    // pitch.)
    assert!(
        lung_depth > 1.5 * depth,
        "lungs {lung_depth} against pitch {depth}"
    );
    assert!(
        lung_depth > 1.5 * tremor(&n.lung).1,
        "normal lungs {:?}",
        tremor(&n.lung)
    );
    // And the voice is still a voice.
    assert!(voiced_share(&v.pitch) > 0.6);
}

#[test]
fn tags_switch_the_state_where_they_stand() {
    let text =
        "Mary may win. [whisper] We were away. [shout] Away we are! [normal] Mary may win more.";
    let rec = speak(&settings(Effort::Normal), text, |_, _| {});
    // The states in the order of the tags, each for a stretch.
    let mut order = vec![rec.effort[0]];
    for &e in &rec.effort {
        if e != *order.last().unwrap() {
            order.push(e);
        }
    }
    assert_eq!(
        order,
        [
            Effort::Normal,
            Effort::Whisper,
            Effort::Shout,
            Effort::Normal
        ]
    );
    let span = |e: Effort, nth: usize| {
        let mut runs = Vec::new();
        let mut start = None;
        for (k, &x) in rec.effort.iter().enumerate() {
            match (x == e, start) {
                (true, None) => start = Some(k),
                (false, Some(s)) => {
                    runs.push((s, k));
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(s) = start {
            runs.push((s, rec.effort.len()));
        }
        runs[nth]
    };
    let (a, b) = span(Effort::Whisper, 0);
    let (c, d) = span(Effort::Shout, 0);
    let (e, f) = span(Effort::Normal, 0);
    let (g, h) = span(Effort::Normal, 1);
    // The whispered sentence has no voice once the glide is over (the
    // glottis never shuts in it, as it does in every cycle of a voice), the
    // shouted one is the loudest and the highest, and the voice comes back.
    let glide = (0.2 * SR) as usize;
    let shuts = |from: usize, to: usize| rec.area[from..to].iter().filter(|&&x| x < 0.01).count();
    assert_eq!(shuts(a + glide, b), 0, "the folds met in the whisper");
    assert!(shuts(e, f) > (f - e) / 20 && shuts(g + glide, h) > (h - g - glide) / 20);
    assert!(voiced_share(&rec.pitch[e..f]) > 0.5 && voiced_share(&rec.pitch[g + glide..h]) > 0.4);
    assert!(level(&rec.out[c..d]) > level(&rec.out[e..f]) + 5.0);
    // (Against the same words with no tags: a tag changes no timing.)
    let plain = speak(
        &settings(Effort::Normal),
        "Mary may win. We were away. Away we are! Mary may win more.",
        |_, _| {},
    );
    assert_eq!(plain.out.len(), rec.out.len());
    let lift = median(&rec.pitch[c + glide..d]) / median(&plain.pitch[c + glide..d]);
    assert!(
        (1.15..1.4).contains(&lift),
        "the shout lifts the pitch {lift}"
    );
    assert!((level(&rec.out[g + glide..h]) - level(&rec.out[e..f])).abs() < 4.0);
    // The next utterance starts in the state it was compiled for, whatever
    // tag the last one ended on.
    let mut talker = Talker::new(SR, settings(Effort::Normal));
    talker.say("We were. [whisper] Away.");
    talker.say("Mary may win.");
    let mut seen = Vec::new();
    while talker.is_speaking() {
        talker.process();
        let e = talker.voice_mut().effort();
        if seen.last() != Some(&e) {
            seen.push(e);
        }
    }
    assert_eq!(seen, [Effort::Normal, Effort::Whisper, Effort::Normal]);
    // An utterance compiled for a state speaks in it from its first sound.
    let whole = speak(&settings(Effort::Shout), "Mary may win.", |_, _| {});
    assert!(whole.effort.iter().all(|&e| e == Effort::Shout));
}

/// A small deterministic generator for the fuzz.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }

    fn unit(&mut self) -> f32 {
        self.next() as f32 / 2_147_483_648.0
    }
}

#[test]
fn the_voice_stays_finite_bounded_and_speaking_whatever_comes() {
    let texts = [
        "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background.",
        "Wait. [whisper] I heard something. [shout] Run! [exhausted] I can not. [nervous] Is it gone? [normal] Yes.",
        "Papa, um, [cough] she sells sea shells. [laugh] [shout] By the sea shore! [sigh] [tired] The juice of lemons.",
        "[sob] Why worry, Mary? [clear-throat] [projected] We were away a year ago. [whisper] Uh, zip it. [chuckle]",
    ];
    // The loudest a shouting voice gets, for the bound: nothing may run
    // away past a few times that.
    let bound = 12.0
        * spoken(Effort::Shout)
            .out
            .iter()
            .fold(0.0f32, |m, &x| m.max(x.abs()));
    for seed in 0..6u64 {
        let mut rng = Lcg(0x5EED_0000 + seed);
        let preset =
            [Preset::Deep, Preset::Bright, Preset::Child, Preset::Elder][rng.below(4) as usize];
        let state = Effort::ALL[rng.below(5) as usize];
        let voice = preset
            .settings()
            .with_engine(Engine::Physical)
            .with_effort(state);
        let mut talker = Talker::new(SR, voice);
        if rng.below(2) == 0 {
            talker
                .voice_mut()
                .set_block([64, 128, 512][rng.below(3) as usize]);
        }
        talker.say(texts[rng.below(4) as usize]);
        talker.say(texts[rng.below(4) as usize]);
        let mut next = 0usize;
        let mut k = 0usize;
        let mut peak = 0.0f32;
        while talker.is_speaking() {
            if k == next {
                // Something else every 20 to 320 ms.
                match rng.below(8) {
                    0..=3 => talker.set_effort(Effort::ALL[rng.below(5) as usize]),
                    4 => talker.set_morph(MorphTarget {
                        tract_scale: 0.75 + 0.5 * rng.unit(),
                        larynx_depth: -0.3 + 1.1 * rng.unit(),
                        fold_thickness: 0.6 + rng.unit(),
                        epilarynx_flare: 0.6 + rng.unit(),
                    }),
                    5 => {
                        let v = Vocalization::ALL[rng.below(5) as usize];
                        talker.vocalize(v);
                    }
                    6 => talker
                        .voice_mut()
                        .set_lung_pressure(1.0 + 39.0 * rng.unit()),
                    _ => talker.voice_mut().set_tenseness(rng.unit()),
                }
                next = k + (0.02 * SR) as usize + rng.below((0.3 * SR) as u32) as usize;
            }
            let x = talker.process();
            assert!(x.is_finite(), "seed {seed}: sample {k} is {x}");
            peak = peak.max(x.abs());
            k += 1;
            assert!(
                k < (90.0 * SR) as usize,
                "seed {seed}: the voice never finished"
            );
        }
        assert!(
            peak < bound,
            "seed {seed}: the sound ran away to {peak} (bound {bound})"
        );
        assert!(peak > 1e-4, "seed {seed}: nothing was heard");
        let g = talker.voice_mut().gauges();
        assert!(
            g.lung_pressure.is_finite() && g.glottal_flow.is_finite(),
            "seed {seed}"
        );
        // Afterwards the voice is itself again: its next sentence (which
        // starts in the state the voice was made in, whatever came last)
        // comes out as loud as from a voice that went through none of it
        // (within 6 dB: the body may still be gliding back).
        talker.voice_mut().stop_vocalizing();
        talker.set_morph(MorphTarget::NEUTRAL);
        talker
            .voice_mut()
            .set_lung_pressure(voice.tuning.anatomy.lung_pressure);
        talker.voice_mut().set_tenseness(voice.tenseness);
        let mut out = Vec::new();
        talker.say("Mary may win more money.");
        while talker.is_speaking() {
            out.push(talker.process());
        }
        let fresh = Talker::new(SR, voice).bake("Mary may win more money.");
        assert!(out.iter().all(|x| x.is_finite()), "seed {seed}");
        assert!(
            (level(&out) - level(&fresh)).abs() < 6.0,
            "seed {seed}: the voice came back at {:.1} dB against {:.1}",
            level(&out),
            level(&fresh)
        );
    }
}

#[test]
fn a_state_glides_in_without_allocating_at_any_block_length_and_through_a_stream() {
    let hold = |block: u32| {
        let mut v = Trombone::new(SR);
        settings(Effort::Normal).apply(&mut v);
        synthie_trombone::speech::Speaker::prepare(&mut v);
        v.set_block(block);
        v.articulate_phoneme(Phoneme::Aa);
        v.settle();
        v.set_frequency(105.0);
        v.note_on();
        let mut buf = vec![0.0f32; 24_000];
        v.render(&mut buf);
        v
    };
    // The same course at any block length: the lungs of a shout a tenth of
    // a second in, and where they end up.
    let mut ends = Vec::new();
    for block in [128, 512] {
        let mut v = hold(block);
        v.set_effort(Effort::Shout);
        let mut buf = vec![0.0f32; 2400];
        v.render(&mut buf);
        let mid = v.gauges().lung_pressure;
        let mut buf = vec![0.0f32; 24_000];
        v.render(&mut buf);
        ends.push((mid, v.gauges().lung_pressure));
        assert_eq!(v.effort(), Effort::Shout);
        assert_eq!(v.target_frequency(), 105.0);
        assert!(v.frequency() > 1.2 * 105.0, "{}", v.frequency());
    }
    assert!((ends[0].0 - ends[1].0).abs() < 1.0, "{ends:?}");
    assert!(
        (ends[0].1 - ends[1].1).abs() < 0.2 && ends[0].1 > 15.0,
        "{ends:?}"
    );

    // No allocation, on the voice and through a stream's player.
    let (mut stream, mut player) = voice_stream(1024);
    let mut v = hold(512);
    let mut block = [0.0f32; 512];
    ALLOCATIONS.store(0, Ordering::Relaxed);
    for k in 0..50usize {
        WATCHED.with(|w| w.set(true));
        v.set_effort(Effort::ALL[k % 5]);
        v.render(&mut block);
        WATCHED.with(|w| w.set(false));
        assert!(block.iter().all(|x| x.is_finite()));
    }
    for e in [Effort::Nervous, Effort::Whisper, Effort::Normal] {
        stream.push(Event::Effort(e)).unwrap();
        WATCHED.with(|w| w.set(true));
        player.render(&mut v, &mut block);
        WATCHED.with(|w| w.set(false));
        assert_eq!(v.effort(), e);
    }
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
    // Back in the normal state the voice is asked for what it was asked.
    let mut buf = vec![0.0f32; 24_000];
    v.render(&mut buf);
    assert!(
        (v.frequency() / 105.0 - 1.0).abs() < 0.05,
        "{}",
        v.frequency()
    );
    assert!((v.gauges().lung_pressure - 8.0).abs() < 0.5);
}
