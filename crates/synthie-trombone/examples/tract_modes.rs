//! Generate `src/tract_modes.rs`: the tongue body of the physical engine as
//! a neutral shape and three principal modes, computed from measured MRI
//! area functions (Story 2008, Table I; `data/story2008_area_functions.csv`).
//!
//! ```sh
//! cargo run -p synthie-trombone --example tract_modes --release
//! ```
//!
//! It
//!
//! 1. reads the eleven vowels' area functions (44 sections each, glottis
//!    first, each vowel with its own section length);
//! 2. checks the transcription: a lossless chain-matrix tube made of each
//!    row, closed at the glottis and radiating at the lips through a
//!    baffle's inertance, must give the paper's own calculated F2 and F3
//!    within 3 % (they land within 0.4 % on F3 and 2.4 % on F2, the back
//!    rounded vowels' F2 low; the paper's F1, 5 to 22 % higher, includes
//!    yielding walls, which a lossless tube has not);
//! 3. computes the principal components of the equivalent diameters
//!    `sqrt(4A/π)` over the eleven vowels (Story's own method, 2005), over
//!    all 44 sections (to compare with what was found before) and over the
//!    sections the modal body shapes: from the top of the epilarynx (the
//!    first four sections are the tube above the folds) to where the lips
//!    begin (the last five sections, which the lips' own tier shapes);
//! 4. writes the neutral shape, the first three modes of the body's
//!    sections, each vowel's coefficients and their ranges, in the engine's
//!    diameter units, together with the least-squares projection of a shape
//!    of the cosine tongue onto the modes.
//!
//! Section `i` (from 1) is cell `i − 1`: both have 44. The engine's
//! diameter `d` is a cross-section of `1.6·π/4·(d·length)²` cm²
//! (`phys::AREA_SCALE`); the measured speaker's unrounded vowels are 16.1 to
//! 17.3 cm long, the native tract 16.04 cm. The measured shapes are made
//! the native size's by scaling them, as a whole, by the native tract's
//! length over the speaker's mean unrounded one ([`SPEAKER_LENGTH`]): a
//! similar tube, one size smaller (see [`SCALE`] for what was measured).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    clippy::many_single_char_names
)]

use std::fmt::Write as _;

/// Sections (and cells) of a tract.
const N: usize = 44;
/// First body cell: the top of the tube above the folds as the engine has
/// it (cells 0 to 6 at diameter 0.6, 2.55 cm at the native size). The
/// speaker's own tube is his first four sections (1.5 cm), and with it as
/// the epilarynx (the body from cell 4) the tract's fourth formant stood at
/// 3.8 to 4.3 kHz instead of 3.3 to 3.6 and its 4 to 8 kHz band 5 to 12 dB
/// louder against 0 to 1 kHz in every vowel but /ʌ ɑ ɔ/: the tube's length
/// is the epilarynx's to set (`AnatomyConfig`'s, wave 2B), the body starts
/// where it ends.
const FROM: usize = 7;
/// First lip cell (`tract::LIP_START`): the body ends before it.
const TO: usize = 39;
/// First cell of the cosine tongue (`tract::BLADE_START`): a cosine shape
/// is projected onto the modes over its own cells only.
const COSINE_FROM: usize = 10;
/// First cell of the mouth whose opening the jaw follows on the modal
/// body: from the soft palate on.
const ORAL_FROM: usize = 25;
/// Modes kept.
const MODES: usize = 3;
/// Speed of sound, cm/s (as the engine's).
const C: f64 = 35_000.0;
/// Length of the native tract, cm.
const NATIVE_LENGTH: f64 = 16.04;
/// Cross-section scale of the engine (`phys::AREA_SCALE`).
const AREA_SCALE: f64 = 1.6;

/// The paper's formants calculated from the area functions (no piriform
/// sinuses), F1/F2/F3 Hz, in the order of the CSV.
const CALCULATED: [(&str, [f64; 3]); 11] = [
    ("i", [325.0, 2139.0, 2976.0]),
    ("ɪ", [413.0, 2103.0, 2626.0]),
    ("e", [484.0, 1887.0, 2404.0]),
    ("ɛ", [565.0, 1595.0, 2200.0]),
    ("æ", [810.0, 1655.0, 2310.0]),
    ("ʌ", [593.0, 836.0, 3099.0]),
    ("ɑ", [694.0, 942.0, 2948.0]),
    ("ɔ", [557.0, 709.0, 3176.0]),
    ("o", [499.0, 768.0, 2421.0]),
    ("ʊ", [477.0, 789.0, 2499.0]),
    ("u", [314.0, 702.0, 2298.0]),
];

/// How the measured diameters become the engine's: the native tract's
/// length over the speaker's mean unrounded length ([`SPEAKER_LENGTH`]),
/// over `sqrt(AREA_SCALE)`. Measured on the engine with the eleven vowels at
/// their measured coefficients: scaled so (a similar tube of the native
/// size) their F1/F2 stood on average 1.5 percentage points nearer the
/// vowel targets than with the speaker's own areas (which make the body
/// some 8 % wider against the engine's lips, epilarynx and port).
fn scale(speaker: f64) -> f64 {
    NATIVE_LENGTH / speaker / AREA_SCALE.sqrt()
}

/// Mean tract length of the speaker's unrounded vowels, cm (/i ɪ e ɛ æ ʌ ɑ
/// ɔ/), computed from the data.
fn speaker_length(rows: &[Row]) -> f64 {
    let unrounded: Vec<&Row> = rows
        .iter()
        .filter(|r| !matches!(r.vowel.as_str(), "o" | "ʊ" | "u"))
        .collect();
    unrounded.iter().map(|r| r.section * N as f64).sum::<f64>() / unrounded.len() as f64
}

struct Row {
    vowel: String,
    /// Section length, cm.
    section: f64,
    /// Areas, cm².
    area: [f64; N],
}

fn read() -> Vec<Row> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/story2008_area_functions.csv"
    );
    let text = std::fs::read_to_string(path).expect("read the area functions");
    let mut rows = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut cells = line.split(',');
        let vowel = cells.next().expect("a vowel").to_string();
        let section: f64 = cells
            .next()
            .expect("a section length")
            .parse()
            .expect("a number");
        let values: Vec<f64> = cells.map(|c| c.parse().expect("an area")).collect();
        assert_eq!(values.len(), N, "{vowel}: 44 sections");
        let mut area = [0.0; N];
        area.copy_from_slice(&values);
        rows.push(Row {
            vowel,
            section,
            area,
        });
    }
    assert_eq!(rows.len(), 11, "eleven vowels");
    rows
}

/// First three resonances of a lossless tube of `areas` (glottis first),
/// each section `length` cm, closed at the glottis and radiating at the lips through the
/// inertance of a piston in a baffle: the zeros of `D − C·X` of the chain
/// matrix (`X` the reactance at the lips), the volume velocity transfer
/// being its inverse.
fn lossless_formants(areas: &[f64], length: f64) -> [f64; 3] {
    // [[a, j b], [j c, d]]: every section's matrix and their product have
    // real diagonal and imaginary off-diagonal elements.
    let d_of = |f: f64| {
        let k = 2.0 * std::f64::consts::PI * f / C * length;
        let (s, co) = k.sin_cos();
        let (mut a, mut b, mut c, mut d) = (1.0, 0.0, 0.0, 1.0);
        for &area in areas {
            let z = 1.0 / area; // ρc/A, up to a constant that cancels
            let (a2, b2, c2, d2) = (co, z * s, s / z, co);
            (a, b, c, d) = (
                a * a2 - b * c2,
                a * b2 + b * d2,
                c * a2 + d * c2,
                -c * b2 + d * d2,
            );
        }
        // The lips radiate into a baffle: a reactance of `ρc/A · 8ka/3π`
        // (its inertance), lossless.
        let lips = areas[areas.len() - 1];
        let radius = (lips / std::f64::consts::PI).sqrt();
        let x = 8.0 * (2.0 * std::f64::consts::PI * f / C) * radius
            / (3.0 * std::f64::consts::PI)
            / lips;
        d - c * x
    };
    let mut out = [0.0; 3];
    let mut found = 0;
    let mut f = 50.0;
    let mut last = d_of(f);
    while found < 3 && f < 6000.0 {
        let next = f + 1.0;
        let v = d_of(next);
        if last.signum() != v.signum() {
            let (mut lo, mut hi, mut vlo) = (f, next, last);
            for _ in 0..50 {
                let mid = 0.5 * (lo + hi);
                let vm = d_of(mid);
                if vm.signum() == vlo.signum() {
                    lo = mid;
                    vlo = vm;
                } else {
                    hi = mid;
                }
            }
            out[found] = 0.5 * (lo + hi);
            found += 1;
        }
        f = next;
        last = v;
    }
    out
}

/// Eigenvalues and eigenvectors (columns) of a symmetric matrix, by Jacobi
/// rotations; sorted by decreasing eigenvalue.
fn eigen(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v = vec![vec![0.0; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }
    for _ in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        if off < 1e-24 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k][p], v[k][q]);
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[j][j].total_cmp(&a[i][i]));
    let values = order.iter().map(|&i| a[i][i]).collect();
    let vectors = order
        .iter()
        .map(|&i| (0..n).map(|k| v[k][i]).collect())
        .collect();
    (values, vectors)
}

/// Principal components of `x` (one row per vowel): the mean, the share of
/// the variance of each component, the modes (each scaled so that its
/// coefficients have unit RMS over the vowels) and the coefficients.
struct Pca {
    mean: Vec<f64>,
    share: Vec<f64>,
    modes: Vec<Vec<f64>>,
    coef: Vec<Vec<f64>>,
}

fn pca(x: &[Vec<f64>]) -> Pca {
    let n = x.len();
    let m = x[0].len();
    let mean: Vec<f64> = (0..m)
        .map(|j| x.iter().map(|r| r[j]).sum::<f64>() / n as f64)
        .collect();
    let xc: Vec<Vec<f64>> = x
        .iter()
        .map(|r| r.iter().zip(&mean).map(|(v, mu)| v - mu).collect())
        .collect();
    let gram: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|k| xc[i].iter().zip(&xc[k]).map(|(a, b)| a * b).sum())
                .collect()
        })
        .collect();
    let (values, vectors) = eigen(gram);
    let total: f64 = values.iter().map(|v| v.max(0.0)).sum();
    let share = values.iter().map(|v| v.max(0.0) / total).collect();
    let rn = (n as f64).sqrt();
    let mut modes = Vec::new();
    let mut coef = vec![Vec::new(); n];
    for u in &vectors {
        // m_k = Xcᵀ u_k / sqrt(n); c_ik = u_k[i] · sqrt(n).
        let mode: Vec<f64> = (0..m)
            .map(|j| (0..n).map(|i| xc[i][j] * u[i]).sum::<f64>() / rn)
            .collect();
        modes.push(mode);
        for i in 0..n {
            coef[i].push(u[i] * rn);
        }
    }
    Pca {
        mean,
        share,
        modes,
        coef,
    }
}

/// Make the signs of the first three components those used to describe
/// them: /ɑ/ positive on the first, /u/ negative on the second, /i/
/// positive on the third.
fn orient(p: &mut Pca, vowels: &[String]) {
    let at = |v: &str| vowels.iter().position(|w| w == v).expect("a vowel");
    let wants = [(at("ɑ"), 1.0), (at("u"), -1.0), (at("i"), 1.0)];
    for (k, &(i, sign)) in wants.iter().enumerate() {
        if p.coef[i][k] * sign < 0.0 {
            for c in &mut p.coef {
                c[k] = -c[k];
            }
            for v in &mut p.modes[k] {
                *v = -*v;
            }
        }
    }
}

/// The cosine tongue's rest diameter of cell `i` for tongue `(index,
/// diameter)`, as `Tract::rest_diameter` makes it (cells of the tongue only).
fn cosine(i: usize, index: f64, diameter: f64) -> f64 {
    let t = 1.1 * std::f64::consts::PI * (index - i as f64) / (32.0 - 10.0);
    let fixed = 2.0 + (diameter - 2.0) / 1.5;
    let mut curve = (1.5 - fixed + 1.7) * t.cos();
    if i == 8 || i == 38 {
        curve *= 0.8;
    }
    if i == 10 || i == 37 {
        curve *= 0.94;
    }
    1.5 - curve
}

/// Solve the 3×3 system `a · x = b`.
fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> [f64; 3] {
    let det = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(a);
    let mut x = [0.0; 3];
    for k in 0..3 {
        let mut m = a;
        for r in 0..3 {
            m[r][k] = b[r];
        }
        x[k] = det(m) / d;
    }
    x
}

fn main() {
    let rows = read();
    let vowels: Vec<String> = rows.iter().map(|r| r.vowel.clone()).collect();

    // 2. The transcription against the paper's own calculation.
    println!("Lossless tube against the paper's calculated formants (Hz, error %):");
    let mut worst: f64 = 0.0;
    for (r, (v, paper)) in rows.iter().zip(&CALCULATED) {
        assert_eq!(&r.vowel, v, "the rows' order");
        let f = lossless_formants(&r.area, r.section);
        let e: Vec<f64> = (0..3).map(|k| 100.0 * (f[k] / paper[k] - 1.0)).collect();
        worst = worst.max(e[1].abs()).max(e[2].abs());
        println!(
            "  {v:2} {:5.1} cm  F1 {:5.0} ({:+5.1})  F2 {:5.0} ({:+5.1})  F3 {:5.0} ({:+5.1})",
            r.section * N as f64,
            f[0],
            e[0],
            f[1],
            e[1],
            f[2],
            e[2]
        );
    }
    assert!(
        worst < 3.0,
        "the transcription: F2/F3 off by {worst:.1} % somewhere"
    );
    println!("  worst F2/F3 error {worst:.2} %");

    let diam = |a: f64| (4.0 * a / std::f64::consts::PI).sqrt();
    // 3. Over all 44 sections, to compare with the earlier computation.
    let all: Vec<Vec<f64>> = rows
        .iter()
        .map(|r| r.area.iter().map(|&a| diam(a)).collect())
        .collect();
    let p_all = pca(&all);
    let cum = |s: &[f64], k: usize| 100.0 * s[..k].iter().sum::<f64>();
    println!(
        "All 44 sections: variance {:.1} {:.1} {:.1} {:.1} % (two {:.1}, three {:.1}, four {:.1} %)",
        100.0 * p_all.share[0],
        100.0 * p_all.share[1],
        100.0 * p_all.share[2],
        100.0 * p_all.share[3],
        cum(&p_all.share, 2),
        cum(&p_all.share, 3),
        cum(&p_all.share, 4)
    );
    // The body's sections.
    let body: Vec<Vec<f64>> = all.iter().map(|r| r[FROM..TO].to_vec()).collect();
    let mut p = pca(&body);
    orient(&mut p, &vowels);
    println!(
        "Body (cells {FROM}..{}): variance {:.1} {:.1} {:.1} {:.1} % (two {:.1}, three {:.1}, four {:.1} %)",
        TO - 1,
        100.0 * p.share[0],
        100.0 * p.share[1],
        100.0 * p.share[2],
        100.0 * p.share[3],
        cum(&p.share, 2),
        cum(&p.share, 3),
        cum(&p.share, 4)
    );
    // How well three modes give back each vowel's body, RMS diameter, cm.
    for (i, v) in vowels.iter().enumerate() {
        let mut rms = 0.0;
        for j in 0..TO - FROM {
            let mut d = p.mean[j];
            for k in 0..MODES {
                d += p.coef[i][k] * p.modes[k][j];
            }
            rms += (d - body[i][j]).powi(2);
        }
        let rms = (rms / (TO - FROM) as f64).sqrt();
        println!(
            "  {v:2} coefficients {:+.2} {:+.2} {:+.2} {:+.2}; three modes miss its body by {:.3} cm RMS",
            p.coef[i][0], p.coef[i][1], p.coef[i][2], p.coef[i][3], rms
        );
    }

    // 4. In engine units.
    let speaker = speaker_length(&rows);
    // `MODES_SCALE` overrides the scale, to measure what it does.
    let k = std::env::var("MODES_SCALE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| scale(speaker));
    println!(
        "Speaker's mean unrounded tract {speaker:.2} cm; engine diameter = {k:.4} × equivalent diameter"
    );
    let neutral: Vec<f64> = p.mean.iter().map(|v| v * k).collect();
    let modes: Vec<Vec<f64>> = p.modes[..MODES]
        .iter()
        .map(|m| m.iter().map(|v| v * k).collect())
        .collect();
    // Ranges of the coefficients over the vowels.
    let mut range = [(f64::INFINITY, f64::NEG_INFINITY); MODES];
    for c in &p.coef {
        for k in 0..MODES {
            range[k].0 = range[k].0.min(c[k]);
            range[k].1 = range[k].1.max(c[k]);
        }
    }
    // The jaw: it lowers as the tongue does, so it follows how open the
    // mouth is along the tongue, the narrowest diameter from the soft
    // palate to the lips (three modes), between the closest and the most
    // open of the measured vowels. Measured against the lips' own opening
    // (mean over the last five sections): /æ ɑ/ open, /i ɪ/ closed.
    let oral = |c: &[f64]| {
        (ORAL_FROM..TO)
            .map(|i| {
                let mut d = neutral[i - FROM];
                for k in 0..MODES {
                    d += c[k] * modes[k][i - FROM];
                }
                d
            })
            .fold(f64::INFINITY, f64::min)
    };
    let orals: Vec<f64> = p.coef.iter().map(|c| oral(c)).collect();
    let closed = orals.iter().copied().fold(f64::INFINITY, f64::min);
    let open = orals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!("Narrowest of the mouth (cells {ORAL_FROM}..{}) and the lips' opening, engine diameters:", TO - 1);
    for (i, v) in vowels.iter().enumerate() {
        println!(
            "  {v:2} {:.2} {:.2}",
            orals[i],
            k * all[i][TO..].iter().sum::<f64>() / (N - TO) as f64
        );
    }
    // The least-squares projection over the cosine tongue's cells: q = P (d − neutral).
    let cells = COSINE_FROM..TO;
    let mut g = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            g[a][b] = cells
                .clone()
                .map(|i| modes[a][i - FROM] * modes[b][i - FROM])
                .sum();
        }
    }
    let mut projection = vec![vec![0.0; TO - COSINE_FROM]; 3];
    for (j, i) in cells.clone().enumerate() {
        let col = [
            modes[0][i - FROM],
            modes[1][i - FROM],
            modes[2][i - FROM],
        ];
        let x = solve3(g, col);
        for a in 0..3 {
            projection[a][j] = x[a];
        }
    }
    // What the projection does with the classic vowels' tongues.
    // What three modes keep of each vowel's acoustics: the lossless tube of
    // the measured row, of the row with its body made of three modes, and
    // of that body in the native tract's 44 cells of 0.3645 cm with the
    // speaker's epilarynx and lips scaled as the body is.
    println!("Lossless F1/F2/F3, Hz: measured row | its body from three modes | that in the native tract:");
    for (i, r) in rows.iter().enumerate() {
        let mut three = r.area;
        for j in FROM..TO {
            let mut d = p.mean[j - FROM];
            for kk in 0..MODES {
                d += p.coef[i][kk] * p.modes[kk][j - FROM];
            }
            three[j] = std::f64::consts::PI / 4.0 * d.max(0.1).powi(2);
        }
        let a = lossless_formants(&r.area, r.section);
        let b = lossless_formants(&three, r.section);
        let native: Vec<f64> = three.iter().map(|v| v * (k * k * AREA_SCALE)).collect();
        let c = lossless_formants(&native, NATIVE_LENGTH / N as f64);
        println!(
            "  {:2} {:4.0} {:4.0} {:4.0} | {:4.0} {:4.0} {:4.0} | {:4.0} {:4.0} {:4.0}",
            r.vowel, a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]
        );
    }
    println!("Classic tongues projected (index, diameter → q1 q2 q3, RMS miss cm):");
    for (name, ti, td) in [
        ("i", 28.0, 2.15),
        ("ɑ", 16.5, 2.25),
        ("u", 23.0, 2.05),
        ("ə", 20.0, 3.2),
        ("æ", 22.0, 2.95),
    ] {
        let mut q = [0.0; 3];
        for (j, i) in cells.clone().enumerate() {
            let d = cosine(i, ti, td) - neutral[i - FROM];
            for a in 0..3 {
                q[a] += projection[a][j] * d;
            }
        }
        let mut miss = 0.0;
        for i in cells.clone() {
            let mut d = neutral[i - FROM];
            for a in 0..3 {
                d += q[a] * modes[a][i - FROM];
            }
            miss += (d - cosine(i, ti, td)).powi(2);
        }
        println!(
            "  {name} ({ti}, {td}) → {:+.2} {:+.2} {:+.2}, {:.3}",
            q[0],
            q[1],
            q[2],
            (miss / cells.len() as f64).sqrt() / k
        );
    }

    // Write the table.
    let list = |v: &[f64]| {
        v.iter()
            .map(|x| format!("{x:.5}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut src = String::new();
    let _ = writeln!(
        src,
        "//! The tongue body of the physical engine as measured modes (see\n\
         //! `Tract::modal_diameter`), computed by `examples/tract_modes.rs` from\n\
         //! `data/story2008_area_functions.csv` (Story 2008, Table I: eleven\n\
         //! vowels of one adult male from MRI); do not edit. Regenerate with\n\
         //! `cargo run -p synthie-trombone --example tract_modes --release`.\n\
         //!\n\
         //! Principal components of the equivalent diameters over the eleven\n\
         //! vowels, over cells {FROM} to {last} (the body: above the epilarynx,\n\
         //! behind the lips). Variance carried by the first four: {s0:.1}, {s1:.1},\n\
         //! {s2:.1} and {s3:.1} % ({c3:.1} % with three). In engine diameters: the\n\
         //! speaker's shapes scaled by {k:.4} (a similar tube of the native size;\n\
         //! his mean unrounded tract is {speaker:.2} cm).\n\n\
         /// First cell the modes shape.\n\
         pub(crate) const FROM: usize = {FROM};\n\
         /// First cell after them (the lips').\n\
         pub(crate) const TO: usize = {TO};\n\
         /// First cell a cosine tongue is projected over.\n\
         pub(crate) const COSINE_FROM: usize = {COSINE_FROM};\n\
         /// The mean of the eleven vowels, engine diameters, cells `FROM..TO`.\n\
         #[rustfmt::skip]\n\
         pub(crate) const NEUTRAL: [f32; {nb}] = [{neutral}];",
        last = TO - 1,
        s0 = 100.0 * p.share[0],
        s1 = 100.0 * p.share[1],
        s2 = 100.0 * p.share[2],
        s3 = 100.0 * p.share[3],
        c3 = cum(&p.share, 3),
        nb = TO - FROM,
        neutral = list(&neutral),
    );
    let _ = writeln!(
        src,
        "/// The three modes, engine diameters per unit of coefficient (each\n\
         /// coefficient has unit RMS over the vowels), cells `FROM..TO`: the\n\
         /// pharynx wide and the front of the mouth narrow against the reverse\n\
         /// (/i/ against /ɑ/); a narrowing at the soft palate against an open\n\
         /// middle (/u ʊ o/ against the rest); the whole lower tract wider, the\n\
         /// lower pharynx most (/i ɑ u/ against /æ ɛ ɪ/).\n\
         #[rustfmt::skip]\n\
         pub(crate) const MODES: [[f32; {nb}]; 3] = [",
        nb = TO - FROM
    );
    for m in &modes {
        let _ = writeln!(src, "    [{}],", list(m));
    }
    let _ = writeln!(src, "];");
    let _ = writeln!(
        src,
        "/// Each measured vowel's coefficients (IPA, as the data writes it).\n\
         #[rustfmt::skip]\n\
         pub(crate) const MEASURED: [(&str, [f32; 3]); 11] = ["
    );
    for (v, c) in vowels.iter().zip(&p.coef) {
        let _ = writeln!(src, "    (\"{v}\", [{:.4}, {:.4}, {:.4}]),", c[0], c[1], c[2]);
    }
    let _ = writeln!(src, "];");
    let _ = writeln!(
        src,
        "/// Lowest and highest coefficient of each mode over the vowels.\n\
         pub(crate) const RANGE: [(f32, f32); 3] = [{}];",
        range
            .iter()
            .map(|(a, b)| format!("({a:.4}, {b:.4})"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let _ = writeln!(
        src,
        "/// First cell of the mouth along the tongue whose narrowest diameter the\n\
         /// jaw of the modal body follows (from the soft palate to the lips)...\n\
         pub(crate) const ORAL_FROM: usize = {ORAL_FROM};\n\
         /// ...and that diameter, three modes, of the closest and the most open\n\
         /// of the measured vowels: the jaw shut and fully open.\n\
         pub(crate) const ORAL_RANGE: (f32, f32) = ({closed:.4}, {open:.4});"
    );
    let _ = writeln!(
        src,
        "/// Least-squares projection of a body shape onto the modes over cells\n\
         /// `COSINE_FROM..TO`: coefficient `k` is `Σ PROJECTION[k][j]·(d − neutral)`\n\
         /// over those cells (`j` from `COSINE_FROM`).\n\
         #[rustfmt::skip]\n\
         pub(crate) const PROJECTION: [[f32; {}]; 3] = [",
        TO - COSINE_FROM
    );
    for row in &projection {
        let _ = writeln!(src, "    [{}],", list(row));
    }
    let _ = writeln!(src, "];");
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/tract_modes.rs");
    std::fs::write(path, &src).expect("write src/tract_modes.rs");
    println!("wrote {path}");
}
