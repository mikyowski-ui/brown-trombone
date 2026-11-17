//! The tongue body of the physical engine as measured modes (see
//! `Tract::modal_diameter`), computed by `examples/tract_modes.rs` from
//! `data/story2008_area_functions.csv` (Story 2008, Table I: eleven
//! vowels of one adult male from MRI); do not edit. Regenerate with
//! `cargo run -p synthie-trombone --example tract_modes --release`.
//!
//! Principal components of the equivalent diameters over the eleven
//! vowels, over cells 7 to 38 (the body: above the epilarynx,
//! behind the lips). Variance carried by the first four: 79.5, 13.5,
//! 5.1 and 1.2 % (98.1 % with three). In engine diameters: the
//! speaker's shapes scaled by 0.7491 (a similar tube of the native size;
//! his mean unrounded tract is 16.93 cm).

/// First cell the modes shape.
pub(crate) const FROM: usize = 7;
/// First cell after them (the lips').
pub(crate) const TO: usize = 39;
/// First cell a cosine tongue is projected over.
pub(crate) const COSINE_FROM: usize = 10;
/// The mean of the eleven vowels, engine diameters, cells `FROM..TO`.
#[rustfmt::skip]
pub(crate) const NEUTRAL: [f32; 32] = [0.98457, 0.89615, 0.81475, 0.80306, 0.84546, 0.88504, 0.91817, 0.95715, 0.98544, 1.00452, 1.02413, 1.03655, 1.01643, 0.98776, 0.99988, 1.03672, 1.04581, 1.03255, 1.01861, 1.02327, 1.06691, 1.15313, 1.25356, 1.34962, 1.44293, 1.52984, 1.59774, 1.64376, 1.66550, 1.65655, 1.59410, 1.46923];
/// The three modes, engine diameters per unit of coefficient (each
/// coefficient has unit RMS over the vowels), cells `FROM..TO`: the
/// pharynx wide and the front of the mouth narrow against the reverse
/// (/i/ against /ɑ/); a narrowing at the soft palate against an open
/// middle (/u ʊ o/ against the rest); the whole lower tract wider, the
/// lower pharynx most (/i ɑ u/ against /æ ɛ ɪ/).
#[rustfmt::skip]
pub(crate) const MODES: [[f32; 32]; 3] = [
    [-0.17366, -0.20389, -0.19612, -0.21295, -0.27021, -0.35743, -0.44510, -0.49840, -0.51593, -0.51804, -0.51570, -0.49790, -0.47385, -0.44409, -0.35641, -0.21103, -0.07330, 0.03617, 0.13359, 0.23037, 0.32782, 0.42091, 0.50592, 0.58208, 0.64369, 0.67717, 0.68972, 0.68519, 0.66249, 0.61999, 0.56310, 0.49341],
    [-0.09438, -0.06923, -0.07407, -0.12514, -0.18874, -0.20585, -0.17055, -0.10647, -0.03838, 0.01551, 0.05506, 0.08153, 0.10483, 0.12507, 0.15172, 0.19302, 0.24340, 0.28371, 0.30869, 0.32319, 0.31095, 0.26586, 0.21388, 0.16978, 0.11369, 0.03438, -0.04577, -0.11445, -0.19138, -0.26423, -0.29378, -0.25822],
    [0.16979, 0.14599, 0.15942, 0.19206, 0.21999, 0.22698, 0.19349, 0.13894, 0.09519, 0.06473, 0.04629, 0.03540, 0.03487, 0.03857, 0.04243, 0.05127, 0.06738, 0.07300, 0.06829, 0.07571, 0.10291, 0.13426, 0.15138, 0.13859, 0.11102, 0.09599, 0.08292, 0.06356, 0.05831, 0.06051, 0.04083, 0.00606],
];
/// Each measured vowel's coefficients (IPA, as the data writes it).
#[rustfmt::skip]
pub(crate) const MEASURED: [(&str, [f32; 3]); 11] = [
    ("i", [-1.6376, 0.6541, 1.8646]),
    ("ɪ", [-1.2016, 0.7197, -0.8824]),
    ("e", [-0.9589, 0.1283, -0.5200]),
    ("ɛ", [-0.4710, 0.6419, -1.1566]),
    ("æ", [-0.0694, 0.2356, -1.1919]),
    ("ʌ", [0.9366, 0.5348, 0.0138]),
    ("ɑ", [1.3566, 1.1868, 1.1408]),
    ("ɔ", [1.4077, 0.5838, 0.4496]),
    ("o", [0.7540, -1.2764, -0.6394]),
    ("ʊ", [0.4179, -1.4849, -0.3697]),
    ("u", [-0.5343, -1.9237, 1.2911]),
];
/// Lowest and highest coefficient of each mode over the vowels.
pub(crate) const RANGE: [(f32, f32); 3] = [(-1.6376, 1.4077), (-1.9237, 1.1868), (-1.1919, 1.8646)];
/// The cosine tongue a modal body carries, place and height, as
/// `TONGUE[k][0] + Σ TONGUE[k][m + 1]·q_m` (a least-squares fit over
/// the projections of cosine tongues): what the jaw follows.
pub(crate) const TONGUE: [[f32; 4]; 2] = [[18.90350, -4.39381, -1.96426, 0.29630], [-3.18250, 2.12049, -1.04905, 2.64815]];
/// Least-squares projection of a body shape onto the modes over cells
/// `COSINE_FROM..TO`: coefficient `k` is `Σ PROJECTION[k][j]·(d − neutral)`
/// over those cells (`j` from `COSINE_FROM`).
#[rustfmt::skip]
pub(crate) const PROJECTION: [[f32; 29]; 3] = [
    [-0.04175, -0.05223, -0.06614, -0.07805, -0.08357, -0.08397, -0.08263, -0.08121, -0.07781, -0.07389, -0.06928, -0.05568, -0.03324, -0.01221, 0.00480, 0.02025, 0.03502, 0.04888, 0.06165, 0.07369, 0.08565, 0.09587, 0.10105, 0.10291, 0.10247, 0.09858, 0.09136, 0.08313, 0.07405],
    [-0.13499, -0.19618, -0.21314, -0.17829, -0.11484, -0.04854, 0.00360, 0.04149, 0.06684, 0.08830, 0.10659, 0.13115, 0.16900, 0.21432, 0.25132, 0.27529, 0.28851, 0.27536, 0.23186, 0.18344, 0.14515, 0.09732, 0.02686, -0.04473, -0.10536, -0.17507, -0.24198, -0.26734, -0.23199],
    [0.57493, 0.66417, 0.68961, 0.59311, 0.43159, 0.29933, 0.20619, 0.14896, 0.11416, 0.10930, 0.11672, 0.12157, 0.13695, 0.17268, 0.18033, 0.16025, 0.17622, 0.25187, 0.34296, 0.39407, 0.35854, 0.28223, 0.24552, 0.21543, 0.16674, 0.16028, 0.17570, 0.12429, 0.02333],
];
