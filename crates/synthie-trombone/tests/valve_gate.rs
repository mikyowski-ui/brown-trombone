//! The ventricular valve above the folds (close-out of waves 3 and 4,
//! c3-valve), measured as what it is: at rest it changes nothing; shut, it
//! holds the lungs' pressure with no flow through it; let go with the
//! pressure behind it, it is open within 5 ms; the cough's burst through
//! the larynx thrown open against the cough before it (3.5 L/s at its
//! peak, 39 ms from 10 to 90 %); the throat clearing's flutter at its rate,
//! in the valve and in the sound, and its voice against the cough's; and
//! everything finite at the highest pressure there is, with no allocation
//! and the same course at any block length.

#![allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp
)]

#[path = "../examples/support/vocal.rs"]
mod vocal;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use synthie_trombone::gesture::{Target, Window, var};
use synthie_trombone::speech::{Speaker, Utterance};
use synthie_trombone::{Engine, Gesture, Tier, Trombone, Vocal, Vocalization};
use vocal::{SR, deep, record, rms, vibrating, voice};

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

/// The cough on `main` before the valve (902644d, measured with
/// `examples/vocal.rs`): its peak glottal flow, cm³/s, and the time its
/// flow took from 10 to 90 % of that peak, seconds.
const OLD_COUGH_PEAK: f32 = 3488.0;
const OLD_COUGH_RISE: f32 = 0.039;

/// The deep voice, silent, its folds apart for breath and the voice off.
fn breathing() -> Trombone {
    let mut t = voice(&deep());
    t.set_voiced(false);
    t.set_glottal_spread(0.3);
    record(&mut t, 0.1);
    t
}

/// The rest area of the valve's cell, cm²: the tract's own there.
fn rest_area(t: &Trombone) -> f32 {
    t.gauges().valve_area
}

/// A vocalization alone on the deep voice at 100 Hz, recorded with 0.3 s
/// after it.
fn alone(v: Vocal) -> vocal::Rec {
    let mut t = voice(&deep());
    t.set_frequency(100.0);
    let seconds = t.vocalize(v).expect("the score has room");
    record(&mut t, seconds + 0.3)
}

/// The cough's burst: the sample its folds open at (the last with the
/// glottis shut and pressure behind it), its peak flow, the seconds from
/// the opening to that peak and from 10 to 90 % of it.
fn burst(r: &vocal::Rec) -> (usize, f32, f32, f32) {
    let open = (0..r.len())
        .rfind(|&k| r.area[k] < 0.002 && r.sub[k] > 5.0)
        .expect("the folds shut with pressure behind them");
    let flow = &r.flow[open..(open + (0.2 * SR) as usize).min(r.len())];
    let peak = flow.iter().copied().fold(0.0f32, f32::max);
    let at = flow.iter().position(|&f| f == peak).unwrap();
    let from = flow.iter().position(|&f| f >= 0.1 * peak).unwrap();
    let to = flow.iter().position(|&f| f >= 0.9 * peak).unwrap();
    (open, peak, at as f32 / SR, (to - from) as f32 / SR)
}

#[test]
fn at_rest_the_valve_changes_nothing() {
    // A sentence, its valve set to rest every block, is the sentence bit
    // for bit.
    let settings = deep().with_engine(Engine::Physical);
    let text = "The birch canoe slid on the smooth planks.";
    let say = |touch: bool| {
        let mut t = voice(&settings);
        let mut speaker = Speaker::new();
        speaker.speak(Utterance::from_text(text, &settings.prosody()));
        let mut out = Vec::new();
        while speaker.is_speaking() {
            if touch && out.len() % 512 == 0 {
                t.set_ventricular_valve(0.0);
                t.set_aryepiglottic_flutter(0.0);
            }
            out.push(speaker.process(&mut t));
        }
        out
    };
    let (plain, touched) = (say(false), say(true));
    assert_eq!(plain.len(), touched.len());
    assert!(
        plain
            .iter()
            .zip(&touched)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    // Shut, fluttering and let go again, the valve's cell is the tract's
    // own, to the bit, as on a voice that never used it.
    let mut used = breathing();
    let mut never = breathing();
    used.set_ventricular_valve(1.0);
    used.set_aryepiglottic_flutter(1.0);
    record(&mut used, 0.1);
    used.set_ventricular_valve(-1.0);
    record(&mut used, 0.1);
    used.set_ventricular_valve(0.0);
    used.set_aryepiglottic_flutter(0.0);
    record(&mut used, 0.5);
    record(&mut never, 0.7);
    assert_eq!(rest_area(&used).to_bits(), rest_area(&never).to_bits());
    assert!(rest_area(&never) > 0.4, "{}", rest_area(&never));
}

#[test]
fn shut_it_holds_the_lungs_pressure_with_no_flow_through_it() {
    for push in [16.0, 40.0, 100.0] {
        let mut t = breathing();
        t.set_ventricular_valve(1.0);
        record(&mut t, 0.1);
        assert_eq!(t.gauges().valve_area, 0.0, "shut");
        t.set_lung_push(push);
        let r = record(&mut t, 0.5);
        assert!(r.finite);
        // Over the last 0.2 s: the pressure under the valve is the lungs',
        // and nothing leaves the mouth and nose (the leak bound: under 0.1
        // cm³/s, where a breath goes at some 300); what still goes through
        // the open folds is the walls between them and the valve yielding.
        let tail = (0.3 * SR) as usize..r.len();
        let most = |x: &[f32]| x.iter().fold(0.0f32, |m, f| m.max(f.abs()));
        let leak = most(&r.lips[tail.clone()]) + most(&r.nose[tail.clone()]);
        assert!(
            leak < 0.1,
            "{push} cmH2O: {leak:.3} cm3/s through the valve"
        );
        let walls = most(&r.flow[tail.clone()]);
        assert!(
            walls < 0.05 * push,
            "{push} cmH2O: {walls:.2} cm3/s into the walls"
        );
        println!("{push} cmH2O: leak {leak:.2e}, walls {walls:.2} cm3/s");
        let below = t.pressure(0);
        assert!(
            below > 0.97 * push && below < 1.01 * push,
            "{push} cmH2O: {below:.1} under the valve"
        );
        assert!(r.valve[tail].iter().all(|&a| a == 0.0));
        assert!(t.pressure(5).abs() < 0.5, "{:.2} above it", t.pressure(5));
    }
}

#[test]
fn let_go_with_the_pressure_behind_it_it_is_open_within_five_ms() {
    let open = |push: f32| {
        let mut t = breathing();
        let rest = rest_area(&t);
        t.set_ventricular_valve(1.0);
        record(&mut t, 0.1);
        t.set_lung_push(push);
        record(&mut t, 0.4);
        t.set_ventricular_valve(0.0);
        let r = record(&mut t, 0.1);
        assert!(r.finite);
        let at = r
            .valve
            .iter()
            .position(|&a| a >= 0.9 * rest)
            .unwrap_or(r.len());
        (
            at as f32 / SR,
            r.flow.iter().copied().fold(0.0f32, f32::max),
        )
    };
    let (thrown, flow) = open(40.0);
    assert!(thrown < 0.005, "90 % open after {:.2} ms", 1e3 * thrown);
    assert!(flow > 2000.0, "{flow:.0} cm3/s");
    // With nothing behind it the muscle lets it go at its own pace: the
    // pressure is what throws it open.
    let (relaxed, _) = open(0.0);
    assert!(
        relaxed > 2.0 * thrown && relaxed > 0.005,
        "{:.2} ms with no pressure, {:.2} with 40 cmH2O",
        1e3 * relaxed,
        1e3 * thrown
    );
}

#[test]
fn the_cough_bursts_through_the_larynx_thrown_open() {
    let r = alone(Vocal::new(Vocalization::Cough));
    assert!(r.finite);
    let (open, peak, at, rise) = burst(&r);
    // People: 6 to 12 L/s, rising in 30 to 50 ms, at 50 to 100 cmH2O. At
    // the model's 40 cmH2O the lungs and trachea take 21 cmH2O of it at the
    // peak: 6.2 L/s, 1.8 times the cough before the valve.
    assert!(
        peak > 1.6 * OLD_COUGH_PEAK,
        "peak {peak:.0} cm3/s (before: {OLD_COUGH_PEAK:.0})"
    );
    assert!(
        (0.03..=0.06).contains(&at),
        "peak {:.1} ms after the opening",
        1e3 * at
    );
    assert!(rise < OLD_COUGH_RISE, "10-90 % in {:.1} ms", 1e3 * rise);
    // Thrown open: the valve's cell at the open area through the burst.
    let wide = r.valve[open..open + (0.05 * SR) as usize]
        .iter()
        .copied()
        .fold(0.0f32, f32::max);
    let want = deep().tuning.anatomy.valve_open;
    assert!(wide > 0.95 * want, "{wide:.2} cm2 at the valve");
    // Shut through the compression, with no pressed buzz behind it: from
    // the lungs' push to the burst the folds do not vibrate.
    let push = r.sub.iter().position(|&p| p > 5.0).unwrap();
    assert!(r.valve[push..open].iter().all(|&a| a == 0.0));
    let n = (0.01 * SR) as usize;
    let frames = vibrating(&r);
    assert!(
        frames[push / n..open / n].iter().all(|&v| !v),
        "the folds buzz behind the valve"
    );
}

/// The share of the 10 ms frames of a recording in which the folds
/// vibrate, from the first loud one to the last.
fn voiced_share(r: &vocal::Rec) -> f32 {
    let n = (0.01 * SR) as usize;
    let levels: Vec<f32> = r.out.chunks(n).map(rms).collect();
    let loudest = levels.iter().copied().fold(0.0f32, f32::max);
    let loud: Vec<usize> = (0..levels.len())
        .filter(|&k| levels[k] > 0.03 * loudest)
        .collect();
    let (a, b) = (loud[0], *loud.last().unwrap());
    let frames = vibrating(r);
    let voiced = frames[a..=b].iter().filter(|&&v| v).count();
    voiced as f32 / (b - a + 1) as f32
}

#[test]
fn the_throat_clearing_scrapes_through_the_fluttering_valve() {
    let tuning = deep().tuning;
    let rate = tuning.anatomy.flutter_rate;
    let r = alone(Vocal::new(Vocalization::ClearThroat));
    assert!(r.finite);
    // The scrapes: where the valve is open but narrowed (shut, it is 0;
    // at rest 0.45 cm²).
    let rest = r.valve[0];
    let scraping: Vec<usize> = (0..r.len())
        .filter(|&k| r.valve[k] > 0.0 && r.valve[k] < 0.7 * rest)
        .collect();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for &k in &scraping {
        match runs.last_mut() {
            Some(run) if k - run.1 <= (0.03 * SR) as usize => run.1 = k,
            _ => runs.push((k, k)),
        }
    }
    runs.retain(|&(a, b)| b - a > (0.08 * SR) as usize);
    assert_eq!(runs.len(), 2, "{runs:?}");
    for &(a, b) in &runs {
        // The valve's area swings at the folds' rate: its troughs.
        let v = &r.valve[a..b];
        let troughs = (1..v.len() - 1)
            .filter(|&k| v[k] < v[k - 1] && v[k] <= v[k + 1])
            .count();
        let hz = troughs as f32 / ((b - a) as f32 / SR);
        assert!((0.9 * rate..=1.1 * rate).contains(&hz), "{hz:.1} Hz");
        // And the sound with it: the envelope of the scrape repeats at the
        // flutter's period.
        let out = &r.out[a..b];
        let m = (0.002 * SR) as usize;
        let env: Vec<f32> = out.windows(m).step_by(8).map(rms).collect();
        let mean = env.iter().sum::<f32>() / env.len() as f32;
        let e: Vec<f32> = env.iter().map(|x| x - mean).collect();
        let lag_of = |hz: f32| (SR / hz / 8.0) as usize;
        let corr = |lag: usize| -> f32 {
            let (mut ab, mut aa) = (0.0, 0.0);
            for k in 0..e.len() - lag {
                ab += e[k] * e[k + lag];
                aa += e[k] * e[k];
            }
            ab / aa
        };
        let (best, at) = (lag_of(120.0)..=lag_of(40.0))
            .map(|l| (corr(l), l))
            .fold((f32::MIN, 0), |m, x| if x.0 > m.0 { x } else { m });
        let heard = SR / (8.0 * at as f32);
        assert!(best > 0.3, "envelope correlation {best:.2}");
        assert!(
            (0.9 * rate..=1.1 * rate).contains(&heard),
            "heard at {heard:.1} Hz"
        );
    }
    // A scrape, not a voice: far less of it voiced than the cough is.
    let cough = alone(Vocal::new(Vocalization::Cough));
    let (scrape, coughed) = (voiced_share(&r), voiced_share(&cough));
    println!("voiced {scrape:.2} of the throat clearing, {coughed:.2} of the cough");
    assert!(
        scrape < 0.1 && scrape < 0.25 * coughed,
        "voiced {scrape:.2} of the throat clearing, {coughed:.2} of the cough"
    );
}

#[test]
fn everything_stays_finite_at_the_highest_pressure() {
    // The cough's pressure at the top of its range and at the strongest:
    // the lungs at the most they push (100 cmH2O).
    let mut settings = deep().with_engine(Engine::Physical);
    settings.tuning.anatomy.cough_pressure = 100.0;
    for kind in [Vocalization::Cough, Vocalization::ClearThroat] {
        let mut t = voice(&settings);
        let seconds = t.vocalize(Vocal::new(kind).with_strength(2.0)).unwrap();
        let r = record(&mut t, seconds + 0.5);
        assert!(r.finite, "{kind:?}");
        assert!(r.lung.iter().all(|&p| p <= 100.5), "{kind:?}");
        assert!(r.out.iter().all(|x| x.abs() < 50.0), "{kind:?}");
        // No energy that grows: it dies away after the event.
        let n = (0.1 * SR) as usize;
        let loud = r.out.chunks(n).map(rms).fold(0.0f32, f32::max);
        let last = rms(&r.out[r.len() - n..]);
        assert!(
            last < 1e-3 * loud,
            "{kind:?}: {last:.2e} against {loud:.2e}"
        );
    }
    // The valve shut against 100 cmH2O and thrown open.
    let mut t = breathing();
    t.set_ventricular_valve(1.0);
    record(&mut t, 0.1);
    t.set_lung_push(100.0);
    record(&mut t, 0.3);
    t.set_ventricular_valve(-1.0);
    let r = record(&mut t, 0.3);
    assert!(r.finite);
    t.set_lung_push(0.0);
    t.set_ventricular_valve(0.0);
    let r = record(&mut t, 1.0);
    assert!(r.finite);
    // Then it breathes as a voice that never shut it does.
    let mut never = breathing();
    let plain = record(&mut never, 0.5);
    let (after, breath) = (
        rms(&r.out[r.len() - 4800..]),
        rms(&plain.out[plain.len() - 4800..]),
    );
    assert!(
        after < 2.0 * breath,
        "{after:.2e} against a breath's {breath:.2e}"
    );
}

#[test]
fn the_larynx_tier_moves_it_and_lets_it_go() {
    let mut t = breathing();
    let rest = rest_area(&t);
    let shut = Gesture::new(
        Tier::LarynxAndPitch,
        Target::NONE.with(var::larynx::VALVE, 1.0),
    )
    .window(Window::between(0.0, 0.2).ramps(0.02, 0.02));
    assert!(t.push_gesture(shut));
    let r = record(&mut t, 0.15);
    assert!(r.valve[(0.1 * SR) as usize..].iter().all(|&a| a == 0.0));
    record(&mut t, 0.4);
    assert_eq!(rest_area(&t).to_bits(), rest.to_bits());
    let flutter = Gesture::new(
        Tier::LarynxAndPitch,
        Target::NONE.with(var::larynx::FLUTTER, 0.8),
    )
    .window(Window::between(0.0, 0.3).ramps(0.02, 0.02));
    assert!(t.push_gesture(flutter));
    let r = record(&mut t, 0.25);
    let v = &r.valve[(0.1 * SR) as usize..];
    let lo = v.iter().copied().fold(f32::MAX, f32::min);
    assert!(lo < 0.3 * rest, "{lo:.3} of {rest:.3} cm2");
}

#[test]
fn coughs_and_throat_clearings_in_text_play_without_allocating() {
    let settings = deep().with_engine(Engine::Physical);
    let u = Utterance::from_text(
        "Clear throat. [clear-throat] Cough. [cough] Done.",
        &settings.prosody(),
    );
    let mut t = voice(&settings);
    let mut speaker = Speaker::new();
    speaker.speak(u);
    let mut r = vocal::Rec {
        finite: true,
        ..vocal::Rec::default()
    };
    r.out.reserve(10 * SR as usize);
    ALLOCATIONS.store(0, Ordering::Relaxed);
    let mut shut = 0;
    while speaker.is_speaking() {
        WATCHED.with(|w| w.set(true));
        let out = speaker.process(&mut t);
        let valve = t.gauges().valve_area;
        WATCHED.with(|w| w.set(false));
        shut += usize::from(valve == 0.0);
        r.push(out, &t);
    }
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
    assert!(r.finite);
    assert!(
        shut as f32 / SR > 0.3,
        "the valve shut {:.2} s",
        shut as f32 / SR
    );
}

#[test]
fn the_valve_moves_alike_at_any_block_length() {
    // Set by its setter at the same samples, the valve follows the same
    // course whatever the control block (its own speed is the tick's).
    let course = |block: u32| {
        let mut t = Trombone::new(SR);
        deep().with_engine(Engine::Physical).apply(&mut t);
        t.set_block(block);
        t.set_voiced(false);
        t.set_glottal_spread(0.3);
        t.settle();
        let mut out = Vec::new();
        for k in 0..(0.6 * SR) as usize {
            match k {
                0 => t.set_ventricular_valve(1.0),
                4800 => t.set_lung_push(30.0),
                14400 => t.set_ventricular_valve(-1.0),
                19200 => t.set_ventricular_valve(0.5),
                21600 => t.set_aryepiglottic_flutter(0.8),
                24000 => {
                    t.set_ventricular_valve(0.0);
                    t.set_aryepiglottic_flutter(0.0);
                }
                _ => {}
            }
            t.process();
            if k % 64 == 0 {
                out.push(t.gauges().valve_area);
            }
        }
        out
    };
    let long = course(512);
    let travel = long.iter().copied().fold(0.0f32, f32::max);
    for block in [256u32, 128, 64] {
        let short = course(block);
        let worst = long
            .iter()
            .zip(&short)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst <= 0.01 * travel,
            "block {block}: {:.2} % of the travel",
            100.0 * worst / travel
        );
    }
}
