//! The tongue body of the physical engine as measured modes (see
//! `Tract::modal_diameter`), computed by `examples/tract_modes.rs` from
//! `data/story2008_area_functions.csv` (Story 2008, Table I: eleven
//! vowels of one adult male from MRI); do not edit. Regenerate with
//! `cargo run -p synthie-trombone --example tract_modes --release`.
//!
//! Principal components of the equivalent diameters over the eleven
//! vowels, over cells 4 to 38 (the body: above the epilarynx,
//! behind the lips). Variance carried by the first four: 78.1, 13.7,
//! 6.0 and 1.2 % (97.8 % with three). In engine diameters: the
//! speaker's shapes scaled by 0.7491 (a similar tube of the native size;
//! his mean unrounded tract is 16.93 cm).

/// First cell the modes shape.
pub(crate) const FROM: usize = 4;
/// First cell after them (the lips').
pub(crate) const TO: usize = 39;
/// First cell a cosine tongue is projected over.
pub(crate) const COSINE_FROM: usize = 10;
/// The mean of the eleven vowels, engine diameters, cells `FROM..TO`.
#[rustfmt::skip]
pub(crate) const NEUTRAL: [f32; 35] = [0.73431, 0.88827, 0.99221, 0.98457, 0.89615, 0.81475, 0.80306, 0.84546, 0.88504, 0.91817, 0.95715, 0.98544, 1.00452, 1.02413, 1.03655, 1.01643, 0.98776, 0.99988, 1.03672, 1.04581, 1.03255, 1.01861, 1.02327, 1.06691, 1.15313, 1.25356, 1.34962, 1.44293, 1.52984, 1.59774, 1.64376, 1.66550, 1.65655, 1.59410, 1.46923];
/// The three modes, engine diameters per unit of coefficient (each
/// coefficient has unit RMS over the vowels), cells `FROM..TO`: the
/// pharynx wide and the front of the mouth narrow against the reverse
/// (/i/ against /ɑ/); a narrowing at the soft palate against an open
/// middle (/u ʊ o/ against the rest); the whole lower tract wider, the
/// lower pharynx most (/i ɑ u/ against /æ ɛ ɪ/).
#[rustfmt::skip]
pub(crate) const MODES: [[f32; 35]; 3] = [
    [0.03553, -0.01088, -0.09131, -0.17432, -0.20443, -0.19669, -0.21369, -0.27111, -0.35836, -0.44587, -0.49889, -0.51618, -0.51811, -0.51566, -0.49780, -0.47372, -0.44392, -0.35619, -0.21076, -0.07300, 0.03651, 0.13398, 0.23078, 0.32814, 0.42108, 0.50596, 0.58207, 0.64362, 0.67700, 0.68944, 0.68484, 0.66202, 0.61939, 0.56249, 0.49294],
    [-0.08189, -0.13103, -0.14279, -0.10890, -0.08114, -0.08684, -0.14020, -0.20565, -0.22316, -0.18504, -0.11626, -0.04436, 0.01199, 0.05300, 0.08010, 0.10293, 0.12307, 0.14972, 0.18942, 0.23727, 0.27615, 0.30096, 0.31471, 0.30046, 0.25317, 0.20009, 0.15687, 0.10278, 0.02520, -0.05339, -0.12036, -0.19628, -0.26872, -0.29669, -0.25842],
    [0.12479, 0.17113, 0.18839, 0.16427, 0.14052, 0.15270, 0.18030, 0.20230, 0.20784, 0.17727, 0.12724, 0.08806, 0.06182, 0.04644, 0.03823, 0.04083, 0.04524, 0.04974, 0.06370, 0.08656, 0.09769, 0.09619, 0.10471, 0.12974, 0.15606, 0.16795, 0.15227, 0.12143, 0.09903, 0.07862, 0.05405, 0.04146, 0.03659, 0.01524, -0.01596],
];
/// Each measured vowel's coefficients (IPA, as the data writes it).
#[rustfmt::skip]
pub(crate) const MEASURED: [(&str, [f32; 3]); 11] = [
    ("i", [-1.6424, 0.4668, 2.0128]),
    ("ɪ", [-1.1987, 0.7763, -0.7679]),
    ("e", [-0.9556, 0.1954, -0.5781]),
    ("ɛ", [-0.4661, 0.7604, -1.1735]),
    ("æ", [-0.0665, 0.3448, -1.2064]),
    ("ʌ", [0.9361, 0.5091, 0.1292]),
    ("ɑ", [1.3567, 1.0961, 1.2000]),
    ("ɔ", [1.4088, 0.5622, 0.4367]),
    ("o", [0.7517, -1.2823, -0.5489]),
    ("ʊ", [0.4159, -1.4457, -0.5004]),
    ("u", [-0.5399, -1.9832, 0.9964]),
];
/// Lowest and highest coefficient of each mode over the vowels.
pub(crate) const RANGE: [(f32, f32); 3] = [(-1.6424, 1.4088), (-1.9832, 1.0961), (-1.2064, 2.0128)];
/// The cosine tongue a modal body carries, place and height, as
/// `TONGUE[k][0] + Σ TONGUE[k][m + 1]·q_m` (a least-squares fit over
/// the projections of cosine tongues): what the jaw follows.
pub(crate) const TONGUE: [[f32; 4]; 2] = [
    [19.46237, -4.63160, -1.81878, -0.11354],
    [0.62265, 0.51978, -0.23889, 0.98035],
];
/// Least-squares projection of a body shape onto the modes over cells
/// `COSINE_FROM..TO`: coefficient `k` is `Σ PROJECTION[k][j]·(d − neutral)`
/// over those cells (`j` from `COSINE_FROM`).
#[rustfmt::skip]
pub(crate) const PROJECTION: [[f32; 29]; 3] = [
    [-0.04371, -0.05456, -0.06858, -0.08014, -0.08501, -0.08482, -0.08308, -0.08140, -0.07787, -0.07395, -0.06927, -0.05554, -0.03314, -0.01225, 0.00471, 0.02022, 0.03500, 0.04868, 0.06116, 0.07301, 0.08497, 0.09525, 0.10045, 0.10229, 0.10184, 0.09788, 0.09052, 0.08236, 0.07361],
    [-0.18463, -0.25322, -0.27244, -0.22929, -0.15139, -0.07303, -0.01269, 0.03027, 0.05833, 0.07950, 0.09770, 0.12281, 0.15844, 0.19922, 0.23435, 0.25951, 0.27147, 0.25249, 0.20192, 0.14972, 0.11427, 0.07250, 0.00588, -0.06253, -0.11906, -0.18757, -0.25519, -0.27651, -0.23279],
    [0.57078, 0.65518, 0.68023, 0.58549, 0.42434, 0.29203, 0.20069, 0.14480, 0.11275, 0.11286, 0.11869, 0.12017, 0.14270, 0.18913, 0.20545, 0.18960, 0.20582, 0.27751, 0.36221, 0.40710, 0.36986, 0.29242, 0.24720, 0.20904, 0.15646, 0.14190, 0.14973, 0.09830, 0.00082],
];
