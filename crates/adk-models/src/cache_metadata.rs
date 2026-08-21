//! Capability C0122: `CacheMetadata`, ported from
//! `google.adk.models.cache_metadata`.

use adk_platform::time::get_time;
use rusty_serde::{Deserialize, Serialize};

#[derive(Debug, rusty_err::Error)]
pub enum CacheMetadataError {
    #[error(
        "cache_name, expire_time, and invocations_used must all be set (active cache) or all be None (fingerprint-only state)"
    )]
    InconsistentActiveState,
}

/// Two-state cache metadata: an "active cache" (`cache_name`/`expire_time`/
/// `invocations_used` all set) or "fingerprint-only" (all three `None`,
/// only `fingerprint`/`contents_count` set for prefix matching).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(deny_unknown_fields)]
pub struct CacheMetadata {
    #[rusty_serde(default)]
    pub cache_name: Option<String>,
    #[rusty_serde(default)]
    pub expire_time: Option<f64>,
    pub fingerprint: String,
    #[rusty_serde(default)]
    pub invocations_used: Option<u32>,
    pub contents_count: u32,
    #[rusty_serde(default)]
    pub created_at: Option<f64>,
}

impl CacheMetadata {
    /// Validates the all-or-nothing active-cache invariant, matching the
    /// source's `model_validator`.
    pub fn new(
        cache_name: Option<String>,
        expire_time: Option<f64>,
        fingerprint: String,
        invocations_used: Option<u32>,
        contents_count: u32,
        created_at: Option<f64>,
    ) -> Result<Self, CacheMetadataError> {
        let set_count = [
            cache_name.is_some(),
            expire_time.is_some(),
            invocations_used.is_some(),
        ]
        .into_iter()
        .filter(|set| *set)
        .count();
        if set_count != 0 && set_count != 3 {
            return Err(CacheMetadataError::InconsistentActiveState);
        }
        Ok(Self {
            cache_name,
            expire_time,
            fingerprint,
            invocations_used,
            contents_count,
            created_at,
        })
    }

    /// True if the cache will expire within the next 2 minutes (or has
    /// already expired). Always `false` in the fingerprint-only state.
    pub fn expire_soon(&self) -> bool {
        match self.expire_time {
            Some(expire_time) => get_time() > expire_time - 120.0,
            None => false,
        }
    }
}

impl std::fmt::Display for CacheMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.cache_name {
            None => write!(
                f,
                "Fingerprint-only: {} contents, fingerprint={}...",
                self.contents_count,
                &self.fingerprint[..self.fingerprint.len().min(8)]
            ),
            Some(cache_name) => {
                let cache_id = cache_name.rsplit('/').next().unwrap_or(cache_name);
                let expire_time = self.expire_time.expect("active cache has expire_time");
                let invocations_used = self
                    .invocations_used
                    .expect("active cache has invocations_used");
                let minutes_until_expiry = (expire_time - get_time()) / 60.0;
                write!(
                    f,
                    "Cache {cache_id}: used {invocations_used} invocations, cached {} contents, expires in {minutes_until_expiry:.1}min",
                    self.contents_count
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_only_state_is_valid() {
        assert!(CacheMetadata::new(None, None, "fp".to_string(), None, 3, None).is_ok());
    }

    #[test]
    fn active_state_requires_all_three_fields() {
        assert!(CacheMetadata::new(
            Some("projects/p/cachedContents/1".to_string()),
            Some(get_time() + 1000.0),
            "fp".to_string(),
            Some(1),
            3,
            Some(get_time()),
        )
        .is_ok());
    }

    #[test]
    fn a_partial_active_state_is_rejected() {
        let result = CacheMetadata::new(
            Some("projects/p/cachedContents/1".to_string()),
            None,
            "fp".to_string(),
            None,
            3,
            None,
        );
        assert!(matches!(
            result,
            Err(CacheMetadataError::InconsistentActiveState)
        ));
    }

    #[test]
    fn expire_soon_is_false_in_fingerprint_only_state() {
        let metadata = CacheMetadata::new(None, None, "fp".to_string(), None, 1, None).unwrap();
        assert!(!metadata.expire_soon());
    }

    #[test]
    fn expire_soon_is_true_within_the_2_minute_buffer() {
        let metadata = CacheMetadata::new(
            Some("projects/p/cachedContents/1".to_string()),
            Some(get_time() + 60.0),
            "fp".to_string(),
            Some(1),
            1,
            Some(get_time()),
        )
        .unwrap();
        assert!(metadata.expire_soon());
    }

    #[test]
    fn expire_soon_is_false_well_before_expiry() {
        let metadata = CacheMetadata::new(
            Some("projects/p/cachedContents/1".to_string()),
            Some(get_time() + 10_000.0),
            "fp".to_string(),
            Some(1),
            1,
            Some(get_time()),
        )
        .unwrap();
        assert!(!metadata.expire_soon());
    }
}
