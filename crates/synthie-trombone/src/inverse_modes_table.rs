//! Placeholder: regenerate with
//! `cargo run -p synthie-trombone --example babble --release -- --modal`.

#![allow(clippy::unreadable_literal, clippy::excessive_precision)]

pub(super) const DIMS: [usize; 4] = [2, 2, 2, 2];
pub(super) const F_LO: [f32; 3] = [5.0, 6.0, 7.0];
pub(super) const F_HI: [f32; 3] = [7.0, 8.0, 8.5];
pub(super) static NODES: &[u8] = b"_PAAA_PAAA_PAAA_PAAA_PAAA_PAAA_PAAA_PAAA_PAAA_PAAA_PAAA_PAAA_PAAA_PAAA_PAAA_PAAA";
pub(super) static ROUGH: &[u8] = b"B";
pub(super) const REACH_F1: (f32, f32) = (250.0, 800.0);
pub(super) const REACH_F2: (f32, f32) = (600.0, 2700.0);
pub(super) const REACH_SIZE: [usize; 2] = [1, 1];
pub(super) static REACH: &[u8] = b"B";
