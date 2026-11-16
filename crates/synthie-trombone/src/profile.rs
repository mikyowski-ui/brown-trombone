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
//!
//! The tuning may be given flat, as above, or by its three parts, as
//! [`VoiceSettings::to_profile`] writes it:
//!
//! ```json
//! { "tuning": { "anatomy": { "jet_efficiency": 0.05 }, "articulation": { "lead": 0.035 } } }
//! ```
//!
//! or both mixed. Each field is named once: a field given both flat and in
//! its part is refused ([`ProfileError::Repeated`]), so which one counts
//! never depends on the order of the keys. Inside a part only that part's
//! fields are known (`{"anatomy": {"lead": ..}}` is an unknown key).

use crate::tuning::{Part, Pass, Tuning};
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
    /// A tuning field given twice: flat and within its part.
    Repeated(String),
}

impl fmt::Display for ProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProfileError::Json(e) => write!(f, "not a voice profile: {e}"),
            ProfileError::UnknownKey(k) => write!(f, "unknown key in voice profile: {k}"),
            ProfileError::Repeated(k) => {
                write!(
                    f,
                    "tuning field given twice in voice profile (flat and in its part): {k}"
                )
            }
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

/// Lay a profile's `"tuning"` object onto `tuning`: flat keys, parts'
/// objects, or both, each field at most once.
fn lay_tuning(tuning: &mut Tuning, over: Map<String, Value>) -> Result<(), ProfileError> {
    let mut seen = [false; Tuning::FIELDS.len()];
    let mut lay = |part: Option<Part>, key: &str, value: &Value| {
        let path = match part {
            Some(p) => format!("tuning.{}.{key}", p.name()),
            None => format!("tuning.{key}"),
        };
        let Some(i) = Tuning::FIELDS
            .iter()
            .position(|f| f.name == key && part.is_none_or(|p| f.part == p))
        else {
            return Err(ProfileError::UnknownKey(path));
        };
        let number = value
            .as_f64()
            .ok_or_else(|| json(format!("{path} must be a number")))?;
        if std::mem::replace(&mut seen[i], true) {
            return Err(ProfileError::Repeated(path));
        }
        #[allow(clippy::cast_possible_truncation)]
        tuning.set(key, number as f32);
        Ok(())
    };
    for (key, value) in over {
        match Part::from_name(&key) {
            Some(part) => {
                for (k, v) in object(value, &format!("tuning.{key}"))? {
                    lay(Some(part), &k, &v)?;
                }
            }
            None => lay(None, &key, &value)?,
        }
    }
    Ok(())
}

impl VoiceSettings {
    /// This voice with a JSON profile laid over it: the `persona` first,
    /// then the voice's own fields, then the [`Tuning`] fields, flat or by
    /// their parts, each field once ([`ProfileError::Repeated`]). Every
    /// key is optional; an unknown one is an error.
    ///
    /// # Errors
    ///
    /// When the text is not a JSON object, a value has the wrong type, a
    /// key is not one a voice has, or a tuning field is given twice.
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
            lay_tuning(&mut voice.tuning, object(tuning, "tuning")?)?;
        }
        voice.tuning = voice.tuning.clamped();
        Ok(voice)
    }

    /// This voice as a JSON profile, its tuning by its three parts.
    #[must_use]
    pub fn to_profile(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

/// Every [`Tuning`] field with its default, allowed range and tuning pass,
/// as JSON, by part: `{"anatomy": {"name": {"lo": .., "hi": .., "default":
/// .., "pass": "a"}, ..}, "articulation": {..}, "prosody": {..}}`. This is
/// what a program that tunes a voice needs to know.
#[must_use]
pub fn tuning_parameters() -> String {
    let mut out = String::from("{\n");
    for (p, part) in Part::ALL.into_iter().enumerate() {
        let _ = writeln!(out, "  \"{}\": {{", part.name());
        let fields: Vec<_> = Tuning::FIELDS.iter().filter(|f| f.part == part).collect();
        for (k, f) in fields.iter().enumerate() {
            let pass = match f.pass {
                Pass::A => "a",
                Pass::B => "b",
                Pass::C => "c",
            };
            let comma = if k + 1 < fields.len() { "," } else { "" };
            let _ = writeln!(
                out,
                "    \"{}\": {{\"lo\": {}, \"hi\": {}, \"default\": {}, \"pass\": \"{pass}\"}}{comma}",
                f.name, f.lo, f.hi, f.default
            );
        }
        let comma = if p + 1 < Part::ALL.len() { "," } else { "" };
        let _ = writeln!(out, "  }}{comma}");
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
        assert!((voice.tuning.articulation.lead - 0.04).abs() < 1e-7);
        assert!((voice.tuning.anatomy.jet_efficiency - 0.05).abs() < 1e-7);
        // Untouched fields keep the base's values.
        assert!((voice.tuning.prosody.stop_length - base.tuning.prosody.stop_length).abs() < 1e-9);
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
        assert!((voice.tuning.articulation.lead - 0.06).abs() < 1e-7);
    }

    #[test]
    fn tuning_parameters_list_every_field() {
        let text = tuning_parameters();
        let parsed: Value = serde_json::from_str(&text).unwrap();
        let parts = parsed.as_object().unwrap();
        assert_eq!(parts.len(), 3);
        let count: usize = parts.values().map(|p| p.as_object().unwrap().len()).sum();
        assert_eq!(count, Tuning::FIELDS.len());
        for f in Tuning::FIELDS {
            let entry = &parts[f.part.name()][f.name];
            assert!((entry["default"].as_f64().unwrap() - f64::from(f.default)).abs() < 1e-6);
            assert!(entry["lo"].as_f64().unwrap() < entry["hi"].as_f64().unwrap());
        }
    }

    #[test]
    fn tuning_loads_flat_nested_or_mixed() {
        let base = Preset::Deep.settings();
        let flat = base
            .with_profile(r#"{"tuning": {"voice_tilt": 14, "lead": 0.04, "body_rate": 12}}"#)
            .unwrap();
        let nested = base
            .with_profile(
                r#"{"tuning": {"anatomy": {"voice_tilt": 14}, "articulation": {"lead": 0.04,
                    "body_rate": 12}}}"#,
            )
            .unwrap();
        let mixed = base
            .with_profile(
                r#"{"tuning": {"anatomy": {"voice_tilt": 14}, "lead": 0.04, "body_rate": 12}}"#,
            )
            .unwrap();
        assert_eq!(flat, nested);
        assert_eq!(flat, mixed);
        assert!((flat.tuning.anatomy.voice_tilt - 14.0).abs() < 1e-6);
        assert!((flat.tuning.articulation.tongue_body.stiffness - 12.0).abs() < 1e-6);
        assert_ne!(flat, base);
    }

    #[test]
    fn a_tuning_field_is_given_once() {
        let base = Preset::Deep.settings();
        for twice in [
            r#"{"tuning": {"voice_tilt": 14, "anatomy": {"voice_tilt": 12}}}"#,
            r#"{"tuning": {"anatomy": {"voice_tilt": 12}, "voice_tilt": 14}}"#,
        ] {
            assert!(
                matches!(base.with_profile(twice), Err(ProfileError::Repeated(_))),
                "{twice}"
            );
        }
        // A field in another part than its own, or a misspelt one in its
        // part, is unknown.
        for bad in [
            r#"{"tuning": {"anatomy": {"lead": 0.03}}}"#,
            r#"{"tuning": {"prosody": {"accent_sise": 8}}}"#,
            r#"{"tuning": {"anatomy": {"anatomy": {}}}}"#,
        ] {
            assert!(
                matches!(base.with_profile(bad), Err(ProfileError::UnknownKey(_))),
                "{bad}"
            );
        }
        assert!(matches!(
            base.with_profile(r#"{"tuning": {"anatomy": 3}}"#),
            Err(ProfileError::Json(_))
        ));
    }

    #[test]
    fn profiles_are_written_by_part() {
        let voice = Preset::Deep.settings();
        let written: Value = serde_json::from_str(&voice.to_profile()).unwrap();
        let tuning = written["tuning"].as_object().unwrap();
        assert_eq!(
            tuning.keys().map(String::as_str).collect::<Vec<_>>(),
            ["anatomy", "articulation", "prosody"]
        );
        for f in Tuning::FIELDS {
            let v = tuning[f.part.name()][f.name].as_f64().unwrap();
            assert!((v - f64::from(f.default)).abs() < 1e-6, "{}", f.name);
        }
        // And the serde form reads either.
        let flat: Tuning = serde_json::from_str(r#"{"voice_tilt": 14}"#).unwrap();
        let nested: Tuning = serde_json::from_str(r#"{"anatomy": {"voice_tilt": 14}}"#).unwrap();
        assert_eq!(flat, nested);
        assert!(serde_json::from_str::<Tuning>(r#"{"voise_tilt": 14}"#).is_err());
        assert!(
            serde_json::from_str::<Tuning>(r#"{"voice_tilt": 1, "anatomy": {"voice_tilt": 2}}"#)
                .is_err()
        );
    }
}
