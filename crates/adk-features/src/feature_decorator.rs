//! C0647: `features._feature_decorator`'s `experimental`/
//! `working_in_progress`/`stable` decorators, ported as a plain guard
//! function rather than a decorator — Rust has no runtime decorator
//! mechanism to gate an arbitrary class/function the way a Python
//! `@decorator` can wrap one.
//!
//! **Adaptation, disclosed at length**: the source's three decorators
//! (`working_in_progress`/`experimental`/`stable`) each call
//! `_make_feature_decorator` with a *caller-asserted* `feature_stage`,
//! which is checked against the feature's actual registered stage at
//! decoration time (module load) — using `@experimental` on a feature
//! registered `Stable` raises a `ValueError`, catching a
//! decorator/registration mismatch. In this port, [`feature_config`]
//! is a fixed, exhaustive `match` (see that module's own doc) — every
//! [`FeatureName`] already carries exactly one hardcoded
//! [`FeatureStage`], baked in at one single place, with no way for a
//! caller to independently assert a *different* stage the way a
//! Python call site choosing the wrong decorator could. That makes the
//! source's three-way split, and the mismatch check itself,
//! structurally moot here — not narrowed, collapsed, since nothing in
//! this design can disagree with the registry about a feature's stage
//! in the first place. This port therefore exposes one
//! [`check_feature_enabled`] guard function (the actual runtime
//! behavior every one of the three decorators shares: raise unless
//! `is_feature_enabled` at call time), called manually at the top of
//! whatever function/constructor body would have carried the
//! decorator, rather than three stage-asserting wrapper functions with
//! nothing left to assert.
//!
//! Wiring this guard into the specific ~dozen call sites the source
//! decorates across the whole codebase is its own separate,
//! much larger undertaking (each one is a not-yet-ported piece of
//! production code in its own right) and isn't done in this batch —
//! this lands the mechanism itself.

use crate::feature_registry::{is_feature_enabled, FeatureName};

/// The source's `RuntimeError(f"Feature {feature_name} is not enabled.")`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureNotEnabledError {
    pub feature_name: FeatureName,
}

impl std::fmt::Display for FeatureNotEnabledError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Feature {} is not enabled.", self.feature_name.as_str())
    }
}

impl std::error::Error for FeatureNotEnabledError {}

/// C0647: the shared runtime check every one of the source's
/// `working_in_progress`/`experimental`/`stable` decorators performs —
/// see the module doc for why the three are collapsed into this one
/// function in this port.
pub fn check_feature_enabled(feature_name: FeatureName) -> Result<(), FeatureNotEnabledError> {
    if is_feature_enabled(feature_name) {
        Ok(())
    } else {
        Err(FeatureNotEnabledError { feature_name })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_registry::TemporaryFeatureOverride;

    #[test]
    fn passes_for_an_enabled_feature() {
        let _guard = TemporaryFeatureOverride::new(FeatureName::ComputerUse, true);
        assert!(check_feature_enabled(FeatureName::ComputerUse).is_ok());
    }

    #[test]
    fn errors_for_a_disabled_feature() {
        let _guard = TemporaryFeatureOverride::new(FeatureName::ComputerUse, false);
        let error = check_feature_enabled(FeatureName::ComputerUse).unwrap_err();
        assert_eq!(error.to_string(), "Feature COMPUTER_USE is not enabled.");
    }
}
