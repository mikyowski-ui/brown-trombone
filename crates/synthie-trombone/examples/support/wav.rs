#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Minimal 16-bit WAV writer shared by the examples: mono, or two channels
//! interleaved.

use std::io::{self, Write};

/// Write `samples` as a 16-bit PCM mono WAV file.
pub fn write(path: &str, sample_rate: u32, samples: &[f32]) -> io::Result<()> {
    write_channels(path, sample_rate, 1, samples)
}

/// Write interleaved left and right `samples` as a 16-bit PCM stereo WAV
/// file.
#[allow(dead_code)] // not every example writes stereo
pub fn write_stereo(path: &str, sample_rate: u32, samples: &[f32]) -> io::Result<()> {
    write_channels(path, sample_rate, 2, samples)
}

fn write_channels(path: &str, sample_rate: u32, channels: u16, samples: &[f32]) -> io::Result<()> {
    let data_len = u32::try_from(samples.len() * 2).expect("file too long");
    let block = 2 * channels;
    let mut f = io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_len).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?; // PCM
    f.write_all(&channels.to_le_bytes())?;
    f.write_all(&sample_rate.to_le_bytes())?;
    f.write_all(&(sample_rate * u32::from(block)).to_le_bytes())?;
    f.write_all(&block.to_le_bytes())?;
    f.write_all(&16u16.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&data_len.to_le_bytes())?;
    for s in samples {
        f.write_all(&((s.clamp(-1.0, 1.0) * 32_767.0) as i16).to_le_bytes())?;
    }
    f.flush()
}
