//! Capability C0032: `_NodePathBuilder` — a slash-separated hierarchical
//! node path in `node@run_id` format, ported from
//! `google.adk.events._node_path_builder`. Backs [`crate::NodeInfo`]'s
//! computed `run_id`/`parent_run_id`/`name` properties.

/// One segment of a node path: a plain node name, optionally tagged with
/// the workflow `run_id` that executed it (`node@run_id`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeSegment {
    pub node: String,
    pub run_id: Option<String>,
}

/// A slash-separated hierarchical node path (`a/b@run1/c`), identifying a
/// specific workflow-node execution within an agent/workflow tree.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NodePathBuilder {
    segments: Vec<NodeSegment>,
}

impl NodePathBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Parses a slash-separated string like `"a/b@run1/c"`.
    pub fn from_string(s: &str) -> Self {
        if s.is_empty() {
            return Self::new();
        }
        let segments = s
            .split('/')
            .map(|part| match part.split_once('@') {
                Some((node, run_id)) => NodeSegment {
                    node: node.to_string(),
                    run_id: Some(run_id.to_string()),
                },
                None => NodeSegment {
                    node: part.to_string(),
                    run_id: None,
                },
            })
            .collect();
        Self { segments }
    }

    pub fn segments(&self) -> &[NodeSegment] {
        &self.segments
    }

    /// The leaf segment's plain node name (no `run_id` suffix), or `""`
    /// for an empty path.
    pub fn node_name(&self) -> &str {
        self.segments.last().map(|s| s.node.as_str()).unwrap_or("")
    }

    /// The leaf segment's full string form (`"node"` or `"node@run_id"`),
    /// or `""` for an empty path — `_NodePathBuilder.leaf_segment` in the
    /// source, a distinct property from [`Self::node_name`]/`NodeInfo
    /// .name` (which strips `@run_id`; confirmed by reading `event.py`'s
    /// own `Event.name`/`NodeInfo.name`, a separate computed property
    /// from `_NodePathBuilder.leaf_segment`, which does not strip it).
    /// An earlier revision of this method aliased `node_name` directly,
    /// conflating the two — fixed here (no prior caller depended on that
    /// behavior, verified before changing it) since
    /// `workflow_rehydration_utils::reconstruct_node_states` (C0320)
    /// needs the real, run_id-inclusive segment as a map key.
    pub fn leaf_segment(&self) -> String {
        match self.segments.last() {
            Some(segment) => match &segment.run_id {
                Some(run_id) => format!("{}@{}", segment.node, run_id),
                None => segment.node.clone(),
            },
            None => String::new(),
        }
    }

    /// The `run_id` tagged on the leaf segment, if any — this is the
    /// `run_id` a [`crate::NodeInfo`] reports for the node that produced
    /// it.
    pub fn run_id(&self) -> Option<&str> {
        self.segments.last().and_then(|s| s.run_id.as_deref())
    }

    /// The `run_id` tagged on the *parent* segment, if any and if a parent
    /// exists — this is the `parent_run_id` a [`crate::NodeInfo`] reports.
    pub fn parent_run_id(&self) -> Option<&str> {
        if self.segments.len() < 2 {
            return None;
        }
        self.segments[self.segments.len() - 2].run_id.as_deref()
    }

    /// The parent path (all but the last segment). `None` for the root.
    pub fn parent(&self) -> Option<Self> {
        if self.segments.is_empty() {
            return None;
        }
        Some(Self {
            segments: self.segments[..self.segments.len() - 1].to_vec(),
        })
    }

    /// Appends one segment and returns the extended path.
    pub fn append(&self, node: impl Into<String>, run_id: Option<String>) -> Self {
        let mut segments = self.segments.clone();
        segments.push(NodeSegment {
            node: node.into(),
            run_id,
        });
        Self { segments }
    }

    /// True if `self` is `other`, or nested under it (prefix match).
    pub fn is_descendant_of(&self, other: &Self) -> bool {
        if other.segments.len() > self.segments.len() {
            return false;
        }
        self.segments[..other.segments.len()] == other.segments[..]
    }

    /// True if `self` is exactly one segment below `other`.
    pub fn is_direct_child_of(&self, other: &Self) -> bool {
        self.segments.len() == other.segments.len() + 1 && self.is_descendant_of(other)
    }

    /// Builds the direct child path for the given node name/run_id, one
    /// segment below `self`.
    pub fn get_direct_child(&self, node: impl Into<String>, run_id: Option<String>) -> Self {
        self.append(node, run_id)
    }

    /// Renders back to the source's slash-separated `node@run_id` string
    /// form.
    pub fn to_slash_string(&self) -> String {
        self.segments
            .iter()
            .map(|s| match &s.run_id {
                Some(run_id) => format!("{}@{}", s.node, run_id),
                None => s.node.clone(),
            })
            .collect::<Vec<_>>()
            .join("/")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_string_round_trips_through_to_slash_string() {
        let p = NodePathBuilder::from_string("a/b@run1/c");
        assert_eq!(p.to_slash_string(), "a/b@run1/c");
    }

    #[test]
    fn node_name_is_the_leaf_segment_without_run_id() {
        let p = NodePathBuilder::from_string("a/b@run1");
        assert_eq!(p.node_name(), "b");
    }

    #[test]
    fn run_id_and_parent_run_id_come_from_leaf_and_parent_respectively() {
        let p = NodePathBuilder::from_string("a@r1/b@r2");
        assert_eq!(p.run_id(), Some("r2"));
        assert_eq!(p.parent_run_id(), Some("r1"));
    }

    #[test]
    fn parent_run_id_is_none_without_a_parent_or_untagged_parent() {
        let root_level = NodePathBuilder::from_string("a@r1");
        assert_eq!(root_level.parent_run_id(), None);

        let untagged_parent = NodePathBuilder::from_string("a/b@r2");
        assert_eq!(untagged_parent.parent_run_id(), None);
    }

    #[test]
    fn is_direct_child_of_requires_exactly_one_more_segment() {
        let a = NodePathBuilder::from_string("a");
        let ab = NodePathBuilder::from_string("a/b");
        let abc = NodePathBuilder::from_string("a/b/c");
        assert!(ab.is_direct_child_of(&a));
        assert!(
            !abc.is_direct_child_of(&a),
            "grandchild is not a direct child"
        );
        assert!(abc.is_direct_child_of(&ab));
    }

    #[test]
    fn get_direct_child_builds_the_expected_path() {
        let a = NodePathBuilder::from_string("a");
        let child = a.get_direct_child("b", Some("run-9".to_string()));
        assert_eq!(child.to_slash_string(), "a/b@run-9");
        assert!(child.is_direct_child_of(&a));
    }
}
