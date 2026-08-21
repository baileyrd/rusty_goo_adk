//! Capability C0031: `_BranchPath` — a dot-separated hierarchical branch
//! path in `segment@run_id` format, ported from
//! `google.adk.events._branch_path`.

/// One segment of a branch path: a plain agent-tree segment name, optionally
/// tagged with the workflow `run_id` that produced it (`segment@run_id`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchSegment {
    pub name: String,
    pub run_id: Option<String>,
}

/// A dot-separated hierarchical branch path (`a.b@run1.c`), used to scope
/// which sub-agent/workflow-node produced a given event.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BranchPath {
    segments: Vec<BranchSegment>,
}

impl BranchPath {
    /// An empty branch path (the root).
    pub fn new() -> Self {
        Self::default()
    }

    /// Parses a dot-separated string like `"a.b@run1.c"` into a
    /// [`BranchPath`]. An empty string parses to the empty (root) path.
    pub fn from_string(s: &str) -> Self {
        if s.is_empty() {
            return Self::new();
        }
        let segments = s
            .split('.')
            .map(|part| match part.split_once('@') {
                Some((name, run_id)) => BranchSegment {
                    name: name.to_string(),
                    run_id: Some(run_id.to_string()),
                },
                None => BranchSegment {
                    name: part.to_string(),
                    run_id: None,
                },
            })
            .collect();
        Self { segments }
    }

    /// The path's segments, in order from root to leaf.
    pub fn segments(&self) -> &[BranchSegment] {
        &self.segments
    }

    /// Every `run_id` tagged anywhere along the path, in order.
    pub fn run_ids(&self) -> Vec<&str> {
        self.segments
            .iter()
            .filter_map(|s| s.run_id.as_deref())
            .collect()
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

    /// True if `self` is `other`, or a descendant of it (a strict prefix
    /// match on segments — a path is not its own descendant unless equal
    /// is also accepted by the caller's own semantics; this matches the
    /// source's `is_descendant_of`, which treats an exact match as a
    /// (trivial) descendant).
    pub fn is_descendant_of(&self, other: &Self) -> bool {
        if other.segments.len() > self.segments.len() {
            return false;
        }
        self.segments[..other.segments.len()] == other.segments[..]
    }

    /// The longest common segment-prefix shared by `a` and `b`.
    pub fn common_prefix(a: &Self, b: &Self) -> Self {
        let n = a
            .segments
            .iter()
            .zip(b.segments.iter())
            .take_while(|(x, y)| x == y)
            .count();
        Self {
            segments: a.segments[..n].to_vec(),
        }
    }

    /// Appends one segment (by name, optionally with a `run_id`) and
    /// returns the extended path.
    pub fn append(&self, name: impl Into<String>, run_id: Option<String>) -> Self {
        let mut segments = self.segments.clone();
        segments.push(BranchSegment {
            name: name.into(),
            run_id,
        });
        Self { segments }
    }

    /// Convenience constructor: a new sub-branch one level under `self`.
    pub fn create_sub_branch(&self, name: impl Into<String>, run_id: Option<String>) -> Self {
        self.append(name, run_id)
    }

    /// Renders back to the source's dot-separated `segment@run_id` string
    /// form.
    pub fn to_dotted_string(&self) -> String {
        self.segments
            .iter()
            .map(|s| match &s.run_id {
                Some(run_id) => format!("{}@{}", s.name, run_id),
                None => s.name.clone(),
            })
            .collect::<Vec<_>>()
            .join(".")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_string_round_trips_through_to_dotted_string() {
        let p = BranchPath::from_string("a.b@run1.c");
        assert_eq!(p.to_dotted_string(), "a.b@run1.c");
    }

    #[test]
    fn empty_string_is_the_root() {
        let p = BranchPath::from_string("");
        assert!(p.segments().is_empty());
        assert_eq!(p.to_dotted_string(), "");
    }

    #[test]
    fn run_ids_extracts_every_tagged_segment() {
        let p = BranchPath::from_string("a@r1.b.c@r2");
        assert_eq!(p.run_ids(), vec!["r1", "r2"]);
    }

    #[test]
    fn parent_drops_the_last_segment() {
        let p = BranchPath::from_string("a.b.c");
        let parent = p.parent().unwrap();
        assert_eq!(parent.to_dotted_string(), "a.b");
        assert!(parent
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .is_none());
    }

    #[test]
    fn is_descendant_of_matches_prefix_not_substring() {
        let ab = BranchPath::from_string("a.b");
        let abc = BranchPath::from_string("a.b.c");
        let axc = BranchPath::from_string("a.x.c");
        assert!(abc.is_descendant_of(&ab));
        assert!(
            ab.is_descendant_of(&ab),
            "a path is its own trivial descendant"
        );
        assert!(
            !axc.is_descendant_of(&ab),
            "partial-prefix match must not count"
        );
    }

    #[test]
    fn common_prefix_stops_at_first_divergence() {
        let a = BranchPath::from_string("x.y.z");
        let b = BranchPath::from_string("x.y.w");
        assert_eq!(BranchPath::common_prefix(&a, &b).to_dotted_string(), "x.y");
    }

    #[test]
    fn append_and_create_sub_branch_extend_by_one_segment() {
        let root = BranchPath::new();
        let child = root.append("agent1", Some("run-42".to_string()));
        assert_eq!(child.to_dotted_string(), "agent1@run-42");
        let grandchild = child.create_sub_branch("agent2", None);
        assert_eq!(grandchild.to_dotted_string(), "agent1@run-42.agent2");
    }
}
