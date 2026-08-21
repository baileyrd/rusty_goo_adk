//! Capability C0025: `NodeInfo`, ported from `google.adk.events.event`.

use crate::node_path_builder::NodePathBuilder;
use rusty_serde::{Deserialize, Serialize};

/// Identifies which workflow node produced an event, and (via
/// [`NodeInfo::run_id`]/[`NodeInfo::parent_run_id`]/[`NodeInfo::name`])
/// where in the node tree it sits — those three are computed properties in
/// the source, derived by parsing [`NodeInfo::path`] through
/// [`NodePathBuilder`] rather than stored directly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct NodeInfo {
    pub path: String,
    #[rusty_serde(default)]
    pub output_for: Option<Vec<String>>,
    #[rusty_serde(default)]
    pub message_as_output: Option<bool>,
}

impl NodeInfo {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            output_for: None,
            message_as_output: None,
        }
    }

    fn builder(&self) -> NodePathBuilder {
        NodePathBuilder::from_string(&self.path)
    }

    /// The `run_id` of the node execution that produced this event.
    pub fn run_id(&self) -> Option<String> {
        self.builder().run_id().map(str::to_string)
    }

    /// The `run_id` of the *parent* node execution, if any.
    pub fn parent_run_id(&self) -> Option<String> {
        self.builder().parent_run_id().map(str::to_string)
    }

    /// The leaf node's plain name (no `run_id` suffix).
    pub fn name(&self) -> String {
        self.builder().node_name().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computed_properties_are_derived_from_path() {
        let info = NodeInfo::new("workflow@r1/step_one@r2");
        assert_eq!(info.name(), "step_one");
        assert_eq!(info.run_id(), Some("r2".to_string()));
        assert_eq!(info.parent_run_id(), Some("r1".to_string()));
    }

    #[test]
    fn serializes_with_camel_case_field_names() {
        let mut info = NodeInfo::new("a/b");
        info.message_as_output = Some(true);
        let json = rusty_serde::json::to_string(&info).unwrap();
        assert!(json.contains("\"messageAsOutput\":true"));
        assert!(!json.contains("message_as_output"));
    }
}
