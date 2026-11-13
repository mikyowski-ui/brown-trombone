//! Voice profiles: a voice as a JSON file.
//!
//! A profile is laid over a base voice (a preset, usually). Every key is
//! optional:
//!
//! ```json
//! {
//!   "persona": { "size": 0.9, "effort": 0.6, "breathiness": 0.3, "clarity": 0.7 },
//!   "pitch": 120.0,
//!   "engine": "Physical",
//!   "tuning": { "lead": 0.035, "jet_efficiency": 0.05 }
//! }
//! ```
//!
//! The persona is applied first, then the voice's own fields
//! ([`VoiceSettings`]), then the [`Tuning`] fields. Unknown keys are an
//! error, so a misspelt name does not pass silently.

use crate::tuning::{Pass, Tuning};
use crate::{Persona, VoiceSettings};
use serde_json::{Map, Value};
use std::fmt::{self, Write as _};
use std::string::{String, ToString};

/// Why a profile could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProfileError {
    /// The text is not JSON, or a value has the wrong type.
    Json(String),
    /// A key that no voice, persona or tuning field has.
    UnknownKey(String),
}

impl fmt::Display for ProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProfileError::Json(e) => write!(f, "not a voice profile: {e}"),
            ProfileError::UnknownKey(k) => write!(f, "unknown key in voice profile: {k}"),
        }
    }
}

impl std::error::Error for ProfileError {}

#[allow(clippy::needless_pass_by_value)] // used as a `map_err` adapter
fn json(e: impl ToString) -> ProfileError {
    ProfileError::Json(e.to_string())
}

/// Lay `over`'s keys onto a serialisable value, refusing keys it lacks.
fn overlay<T>(base: &T, over: Map<String, Value>, what: &str) -> Result<T, ProfileError>
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let Value::Object(mut merged) = serde_json::to_value(base).map_err(json)? else {
        return Err(json("not an object"));
    };
    for (key, value) in over {
        if !merged.contains_key(&key) {
            return Err(ProfileError::UnknownKey(format!("{what}{key}")));
        }
        merged.insert(key, value);
    }
    serde_json::from_value(Value::Object(merged)).map_err(json)
}

fn object(value: Value, what: &str) -> Result<Map<String, Value>, ProfileError> {
    match value {
        Value::Object(map) => Ok(map),
        _ => Err(json(format!("{what} must be an object"))),
    }
}

impl VoiceSettings {
    /// This voice with a JSON profile laid over it: the `persona` first,
    /// then the voice's own fields, then the [`Tuning`] fields. Every key
    /// is optional; an unknown one is an error.
    ///
    /// # Errors
    ///
    /// When the text is not a JSON object, a value has the wrong type, or
    /// a key is not one a voice has.
    pub fn with_profile(&self, text: &str) -> Result<Self, ProfileError> {
        let mut over = object(serde_json::from_str(text).map_err(json)?, "a profile")?;
        let mut voice = *self;
        if let Some(persona) = over.remove("persona") {
            let persona = overlay(&Persona::NEUTRAL, object(persona, "persona")?, "persona.")?;
            voice = voice.with_persona(&persona);
        }
        let tuning = over.remove("tuning");
        let mut voice: VoiceSettings = overlay(&voice, over, "")?;
        if let Some(tuning) = tuning {
            for (key, value) in object(tuning, "tuning")? {
                let number = value
                    .as_f64()
                    .ok_or_else(|| json(format!("tuning.{key} must be a number")))?;
                #[allow(clippy::cast_possible_truncation)]
                if !voice.tuning.set(&key, number as f32) {
                    return Err(ProfileError::UnknownKey(format!("tuning.{key}")));
                }
            }
        }
        voice.tuning = voice.tuning.clamped();
        Ok(voice)
    }

    /// This voice as a JSON profile.
    #[must_use]
    pub fn to_profile(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

/// Every [`Tuning`] field with its default, allowed range and tuning pass,
/// as JSON: `{"name": {"lo": .., "hi": .., "default": .., "pass": "a"}}`.
/// This is what a program that tunes a voice needs to know.
#[must_use]
pub fn tuning_parameters() -> String {
    let mut out = String::from("{\n");
    for (k, f) in Tuning::FIELDS.iter().enumerate() {
        let pass = match f.pass {
            Pass::A => "a",
            Pass::B => "b",
            Pass::C => "c",
        };
        let comma = if k + 1 < Tuning::FIELDS.len() {
            ","
        } else {
            ""
        };
        let _ = writeln!(
            out,
            "  \"{}\": {{\"lo\": {}, \"hi\": {}, \"default\": {}, \"pass\": \"{pass}\"}}{comma}",
            f.name, f.lo, f.hi, f.default
        );
    }
    out.push('}');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Engine, Preset};

    #[test]
    fn an_empty_profile_changes_nothing() {
        let base = Preset::Bright.settings();
        assert_eq!(base.with_profile("{}").unwrap(), base);
    }

    #[test]
    fn a_voice_survives_a_round_trip() {
        for preset in Preset::ALL {
            let voice = preset.settings().with_engine(Engine::Physical);
            let back = Preset::Deep.settings().with_profile(&voice.to_profile());
            assert_eq!(back.unwrap(), voice, "{preset}");
        }
    }

    #[test]
    fn keys_overlay_in_order() {
        let base = Preset::Deep.settings();
        let voice = base
            .with_profile(
                r#"{"persona": {"size": 0.8}, "pitch": 200.0, "engine": "Physical",
                    "tuning": {"lead": 0.04, "jet_efficiency": 0.05}}"#,
            )
            .unwrap();
        // The explicit pitch wins over the persona's.
        assert!((voice.pitch - 200.0).abs() < 1e-4);
        assert!((voice.tract_length - 0.8).abs() < 1e-6);
        assert_eq!(voice.engine, Engine::Physical);
        assert!((voice.tuning.lead - 0.04).abs() < 1e-7);
        assert!((voice.tuning.jet_efficiency - 0.05).abs() < 1e-7);
        // Untouched fields keep the base's values.
        assert!((voice.tuning.stop_length - base.tuning.stop_length).abs() < 1e-9);
    }

    #[test]
    fn misspelt_keys_are_refused() {
        let base = Preset::Deep.settings();
        for bad in [
            r#"{"pich": 120}"#,
            r#"{"tuning": {"leed": 0.03}}"#,
            r#"{"persona": {"sise": 1.0}}"#,
        ] {
            assert!(
                matches!(base.with_profile(bad), Err(ProfileError::UnknownKey(_))),
                "{bad}"
            );
        }
        assert!(matches!(
            base.with_profile("[1]"),
            Err(ProfileError::Json(_))
        ));
        assert!(matches!(
            base.with_profile(r#"{"tuning": {"lead": "soon"}}"#),
            Err(ProfileError::Json(_))
        ));
    }

    #[test]
    fn out_of_range_tuning_is_pulled_in() {
        let voice = Preset::Deep
            .settings()
            .with_profile(r#"{"tuning": {"lead": 5.0}}"#)
            .unwrap();
        assert!((voice.tuning.lead - 0.06).abs() < 1e-7);
    }

    #[test]
    fn tuning_parameters_list_every_field() {
        let text = tuning_parameters();
        let parsed: Value = serde_json::from_str(&text).unwrap();
        let map = parsed.as_object().unwrap();
        assert_eq!(map.len(), Tuning::FIELDS.len());
        for f in Tuning::FIELDS {
            let entry = &map[f.name];
            assert!((entry["default"].as_f64().unwrap() - f64::from(f.default)).abs() < 1e-6);
            assert!(entry["lo"].as_f64().unwrap() < entry["hi"].as_f64().unwrap());
        }
    }
}
