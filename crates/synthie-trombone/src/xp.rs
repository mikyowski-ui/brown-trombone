//! TEMPORARY experiment knobs (r11-canoe); removed before commit.
#[cfg(feature = "std")]
pub(crate) fn x(name: &str, d: f32) -> f32 {
    extern crate std;
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}
#[cfg(not(feature = "std"))]
pub(crate) fn x(_name: &str, d: f32) -> f32 {
    d
}
