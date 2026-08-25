//! Capability C0297 (partial): `Edge`/`Graph`/`RouteValue`/
//! `DEFAULT_ROUTE`, ported from `google.adk.workflow._graph`. Part of
//! the P7 workflow/graph engine — see `workflow_node_state.rs`'s module
//! doc for the standing crate-placement decision.
//!
//! **`Graph::from_edge_items`/`parse_edge_items` (C0327), narrowed —
//! see `workflow_graph_parser.rs`'s own doc**: the chain-building sugar
//! now ports, but only over `NodeLike::Node`/`NodeLike::Start` — no
//! `BaseAgent`/`BaseTool`/raw-callable chain elements, since those need
//! `build_node`/`is_node_like` (C0326), which itself needs
//! `FunctionNode`/`_ToolNode`/`_ParallelWorker` (C0313/C0316/C0317) and
//! `BaseTool` — `BaseTool` lives in `adk-tools`, which already depends
//! on this crate, so `adk-agents` depending back would be the same
//! crate-cycle shape already disclosed for C0355/C0356. A future batch
//! that lands the node-wrapper types can widen `NodeLike` once they
//! exist.

use std::collections::HashSet;

use crate::workflow_base_node::BaseNode;

/// `workflow._graph.DEFAULT_ROUTE`.
pub const DEFAULT_ROUTE: &str = "__DEFAULT__";

/// `workflow._graph.RouteValue` — a single routing value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteValue {
    Bool(bool),
    Int(i64),
    Str(String),
}

/// `Edge.route`'s type — a single [`RouteValue`] or a list of them (the
/// edge is followed when the emitted route matches any value in the
/// list).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteSpec {
    Single(RouteValue),
    Many(Vec<RouteValue>),
}

/// `workflow._graph.Edge` — an edge in the workflow graph.
#[derive(Debug, Clone)]
pub struct Edge {
    pub from_node: BaseNode,
    pub to_node: BaseNode,
    pub route: Option<RouteSpec>,
}

impl Edge {
    pub fn new(from_node: BaseNode, to_node: BaseNode, route: Option<RouteSpec>) -> Self {
        Self {
            from_node,
            to_node,
            route,
        }
    }
}

/// Converts a single dynamically-typed emitted-route [`rusty_serde::value::Value`]
/// into the [`RouteValue`] [`Graph::get_next_pending_nodes`] matches
/// against — `Value::Bool`/`Value::Int`/`Value::String` map directly;
/// `Value::UInt` widens into `RouteValue::Int` (lossy only for a route
/// value outside `i64::MAX`, never realistic for a route tag); anything
/// else (`Value::Null`/`Value::Float`/`Value::Seq`/`Value::Map`) has no
/// `RouteValue` equivalent and matches nothing.
fn value_to_route_value(value: &rusty_serde::value::Value) -> Option<RouteValue> {
    use rusty_serde::value::Value;
    match value {
        Value::Bool(b) => Some(RouteValue::Bool(*b)),
        Value::Int(i) => Some(RouteValue::Int(*i)),
        Value::UInt(u) => Some(RouteValue::Int(*u as i64)),
        Value::String(s) => Some(RouteValue::Str(s.clone())),
        _ => None,
    }
}

/// Converts a node's emitted `ctx.route` — a raw, dynamically-typed
/// `Value` (this port never refines it further, matching the source's
/// own dynamically-typed `child.route`, see `workflow_rehydration_utils
/// .rs`'s own disclosure of the same choice for `ChildScanState::route`)
/// — into the [`RouteSpec`] [`Graph::get_next_pending_nodes`] needs.
/// `Value::Seq` becomes [`RouteSpec::Many`] (filtering out any
/// unconvertible element; `None` if every element is unconvertible —
/// same as no route emitted at all); any other convertible scalar
/// becomes [`RouteSpec::Single`]; everything else is `None`.
pub(crate) fn value_to_route_spec(value: &rusty_serde::value::Value) -> Option<RouteSpec> {
    match value {
        rusty_serde::value::Value::Seq(items) => {
            let values: Vec<RouteValue> = items.iter().filter_map(value_to_route_value).collect();
            if values.is_empty() {
                None
            } else {
                Some(RouteSpec::Many(values))
            }
        }
        other => value_to_route_value(other).map(RouteSpec::Single),
    }
}

/// `workflow._graph.Graph` — a workflow graph.
#[derive(Debug, Clone, Default)]
pub struct Graph {
    pub nodes: Vec<BaseNode>,
    pub edges: Vec<Edge>,
    terminal_node_names: HashSet<String>,
}

impl Graph {
    /// `Graph.__init__` + `model_post_init`: `nodes` is always inferred
    /// from `edges` (deduplicated by node identity, preserving
    /// first-seen order) — matching the source's own "nodes are
    /// inferred from edges, do not set nodes explicitly" invariant,
    /// enforced here structurally since this constructor is the only
    /// way to build a [`Graph`] rather than by a runtime check against
    /// a directly-settable field.
    pub fn new(edges: Vec<Edge>) -> Self {
        let mut nodes: Vec<BaseNode> = Vec::new();
        for edge in &edges {
            for node in [&edge.from_node, &edge.to_node] {
                if !nodes.iter().any(|existing| existing.ptr_eq(node)) {
                    nodes.push(node.clone());
                }
            }
        }
        Self {
            nodes,
            edges,
            terminal_node_names: HashSet::new(),
        }
    }

    /// Nodes with no outgoing edges — computed by [`Self::validate`].
    /// Empty until validation has run, matching the source's own
    /// `_terminal_node_names: set[str] = PrivateAttr(default_factory=set)`.
    pub fn terminal_node_names(&self) -> &HashSet<String> {
        &self.terminal_node_names
    }

    /// `Graph.get_next_pending_nodes`: the next nodes to transition to
    /// PENDING state, given the route(s) `node_name`'s run just emitted.
    pub fn get_next_pending_nodes(
        &self,
        node_name: &str,
        routes_to_match: Option<&RouteSpec>,
    ) -> Vec<String> {
        let mut next_pending_nodes = Vec::new();
        let mut matched_specific_route = false;
        let mut default_route_node: Option<String> = None;
        let mut has_routing_edges = false;

        for edge in &self.edges {
            if edge.from_node.name() != node_name {
                continue;
            }
            let Some(route) = &edge.route else {
                // Edges with no route tag are always triggered.
                next_pending_nodes.push(edge.to_node.name().to_string());
                continue;
            };

            has_routing_edges = true;
            if matches!(route, RouteSpec::Single(RouteValue::Str(s)) if s == DEFAULT_ROUTE) {
                default_route_node = Some(edge.to_node.name().to_string());
                continue;
            }

            let edge_routes: Vec<&RouteValue> = match route {
                RouteSpec::Single(value) => vec![value],
                RouteSpec::Many(values) => values.iter().collect(),
            };
            let edge_matched = match routes_to_match {
                Some(RouteSpec::Many(candidates)) => candidates
                    .iter()
                    .any(|candidate| edge_routes.contains(&candidate)),
                Some(RouteSpec::Single(value)) => edge_routes.contains(&value),
                None => false,
            };
            if edge_matched {
                next_pending_nodes.push(edge.to_node.name().to_string());
                matched_specific_route = true;
            }
        }

        if !matched_specific_route {
            if let Some(node) = default_route_node {
                next_pending_nodes.push(node);
            }
        }

        if has_routing_edges && next_pending_nodes.is_empty() {
            eprintln!(
                "Node '{node_name}' has conditional/DEFAULT edges but none were matched by \
                 the emitted route(s): {routes_to_match:?}. The branch will end."
            );
        }

        next_pending_nodes
    }

    /// `Graph.validate_graph`: validates the graph and records
    /// [`Self::terminal_node_names`].
    pub fn validate(&mut self) -> Result<(), String> {
        self.terminal_node_names =
            crate::workflow_graph_validation::validate_graph(&self.nodes, &self.edges)?;
        Ok(())
    }

    /// `Graph.from_edge_items` (C0327, narrowed — see
    /// `workflow_graph_parser.rs`'s own doc): builds a `Graph` from a
    /// list of edge items (explicit edges or chains).
    pub fn from_edge_items(
        edge_items: Vec<crate::workflow_graph_parser::EdgeItem>,
    ) -> Result<Self, String> {
        let edges = crate::workflow_graph_parser::parse_edge_items(edge_items)?;
        Ok(Self::new(edges))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow_base_node::{start, BaseNode, NoopNodeBehavior};

    fn node(name: &str) -> BaseNode {
        BaseNode::new(name, NoopNodeBehavior).unwrap()
    }

    #[test]
    fn new_infers_and_deduplicates_nodes_from_edges() {
        let a = node("a");
        let b = node("b");
        let graph = Graph::new(vec![
            Edge::new(start(), a.clone(), None),
            Edge::new(a.clone(), b.clone(), None),
        ]);
        // `a` appears in two edges but should only be inferred once.
        assert_eq!(graph.nodes.len(), 3);
        assert!(graph.nodes.iter().any(|n| n.ptr_eq(&start())));
        assert!(graph.nodes.iter().any(|n| n.ptr_eq(&a)));
        assert!(graph.nodes.iter().any(|n| n.ptr_eq(&b)));
    }

    #[test]
    fn get_next_pending_nodes_follows_unconditional_edges() {
        let a = node("a");
        let b = node("b");
        let graph = Graph::new(vec![Edge::new(a.clone(), b.clone(), None)]);
        assert_eq!(graph.get_next_pending_nodes("a", None), vec!["b"]);
    }

    #[test]
    fn get_next_pending_nodes_matches_a_single_route() {
        let a = node("a");
        let b = node("b");
        let c = node("c");
        let graph = Graph::new(vec![
            Edge::new(
                a.clone(),
                b.clone(),
                Some(RouteSpec::Single(RouteValue::Str("yes".to_string()))),
            ),
            Edge::new(
                a.clone(),
                c.clone(),
                Some(RouteSpec::Single(RouteValue::Str("no".to_string()))),
            ),
        ]);
        let matched = graph.get_next_pending_nodes(
            "a",
            Some(&RouteSpec::Single(RouteValue::Str("yes".to_string()))),
        );
        assert_eq!(matched, vec!["b"]);
    }

    #[test]
    fn get_next_pending_nodes_falls_back_to_the_default_route() {
        let a = node("a");
        let b = node("b");
        let default_target = node("default_target");
        let graph = Graph::new(vec![
            Edge::new(
                a.clone(),
                b.clone(),
                Some(RouteSpec::Single(RouteValue::Str("yes".to_string()))),
            ),
            Edge::new(
                a.clone(),
                default_target.clone(),
                Some(RouteSpec::Single(RouteValue::Str(
                    DEFAULT_ROUTE.to_string(),
                ))),
            ),
        ]);
        let matched = graph.get_next_pending_nodes(
            "a",
            Some(&RouteSpec::Single(RouteValue::Str("unmatched".to_string()))),
        );
        assert_eq!(matched, vec!["default_target"]);
    }

    #[test]
    fn get_next_pending_nodes_prefers_a_specific_match_over_default() {
        let a = node("a");
        let b = node("b");
        let default_target = node("default_target");
        let graph = Graph::new(vec![
            Edge::new(
                a.clone(),
                b.clone(),
                Some(RouteSpec::Single(RouteValue::Str("yes".to_string()))),
            ),
            Edge::new(
                a.clone(),
                default_target.clone(),
                Some(RouteSpec::Single(RouteValue::Str(
                    DEFAULT_ROUTE.to_string(),
                ))),
            ),
        ]);
        let matched = graph.get_next_pending_nodes(
            "a",
            Some(&RouteSpec::Single(RouteValue::Str("yes".to_string()))),
        );
        assert_eq!(matched, vec!["b"]);
    }

    #[test]
    fn get_next_pending_nodes_returns_empty_when_nothing_matches() {
        let a = node("a");
        let b = node("b");
        let graph = Graph::new(vec![Edge::new(
            a.clone(),
            b.clone(),
            Some(RouteSpec::Single(RouteValue::Str("yes".to_string()))),
        )]);
        let matched = graph.get_next_pending_nodes(
            "a",
            Some(&RouteSpec::Single(RouteValue::Str("unmatched".to_string()))),
        );
        assert!(matched.is_empty());
    }

    #[test]
    fn value_to_route_spec_converts_each_scalar_variant() {
        use rusty_serde::value::Value;
        assert_eq!(
            value_to_route_spec(&Value::Bool(true)),
            Some(RouteSpec::Single(RouteValue::Bool(true)))
        );
        assert_eq!(
            value_to_route_spec(&Value::Int(-3)),
            Some(RouteSpec::Single(RouteValue::Int(-3)))
        );
        assert_eq!(
            value_to_route_spec(&Value::UInt(7)),
            Some(RouteSpec::Single(RouteValue::Int(7)))
        );
        assert_eq!(
            value_to_route_spec(&Value::String("yes".to_string())),
            Some(RouteSpec::Single(RouteValue::Str("yes".to_string())))
        );
    }

    #[test]
    fn value_to_route_spec_converts_a_seq_to_many_filtering_unconvertible_entries() {
        use rusty_serde::value::Value;
        let value = Value::Seq(vec![
            Value::String("a".to_string()),
            Value::Null,
            Value::Int(1),
        ]);
        assert_eq!(
            value_to_route_spec(&value),
            Some(RouteSpec::Many(vec![
                RouteValue::Str("a".to_string()),
                RouteValue::Int(1),
            ]))
        );
    }

    #[test]
    fn value_to_route_spec_is_none_for_unconvertible_values() {
        use rusty_serde::value::Value;
        assert_eq!(value_to_route_spec(&Value::Null), None);
        assert_eq!(value_to_route_spec(&Value::Float(1.5)), None);
        assert_eq!(
            value_to_route_spec(&Value::Seq(vec![Value::Null, Value::Float(1.0)])),
            None
        );
    }
}
