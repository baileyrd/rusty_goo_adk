//! Capability C0328: `validate_graph` and its helpers, ported from
//! `google.adk.workflow.utils._graph_validation`. Part of the P7
//! workflow/graph engine — see `workflow_node_state.rs`'s module doc
//! for the standing crate-placement decision.
//!
//! **`_validate_chat_agent_wiring`, disclosed narrowing**: the source
//! checks `isinstance(to_node, LlmAgent)` — reachable in the source
//! because `LlmAgent` (via `BaseAgent`) *is* a `BaseNode` subclass
//! there. This port's [`crate::workflow_base_node::BaseNode`] and
//! [`crate::llm_agent::LlmAgent`] are separate, unfused types (the same
//! C0092 tree-fusion gap disclosed throughout this port) — until a
//! future batch builds the LlmAgent-as-node wrapper (`build_node`'s
//! `isinstance(node_like, LlmAgent)` branch, C0326, itself blocked on
//! that same gap), no `BaseNode` value constructed in this port can
//! ever actually be an `LlmAgent`. So this function is ported as a real,
//! called step in [`validate_graph`]'s sequence (preserving the call
//! order for when a future batch gives it something to check), but its
//! body is necessarily a no-op today — there is nothing yet for it to
//! find.

use std::collections::{HashMap, HashSet};

use crate::workflow_base_node::{start, BaseNode};
use crate::workflow_graph::{Edge, RouteSpec, RouteValue, DEFAULT_ROUTE};

fn detect_unconditional_cycles(edges: &[Edge], node_names: &HashSet<String>) -> Result<(), String> {
    let mut unconditional_adj: HashMap<&str, Vec<&str>> = node_names
        .iter()
        .map(|name| (name.as_str(), Vec::new()))
        .collect();
    for edge in edges {
        if edge.route.is_none() {
            unconditional_adj
                .entry(edge.from_node.name())
                .or_default()
                .push(edge.to_node.name());
        }
    }

    let mut in_stack: HashSet<String> = HashSet::new();
    let mut done: HashSet<String> = HashSet::new();

    fn dfs(
        node: &str,
        path: &mut Vec<String>,
        in_stack: &mut HashSet<String>,
        done: &mut HashSet<String>,
        adj: &HashMap<&str, Vec<&str>>,
    ) -> Result<(), String> {
        in_stack.insert(node.to_string());
        path.push(node.to_string());
        if let Some(neighbors) = adj.get(node) {
            for &neighbor in neighbors {
                if in_stack.contains(neighbor) {
                    let cycle_start = path.iter().position(|n| n == neighbor).unwrap();
                    let mut cycle = path[cycle_start..].to_vec();
                    cycle.push(neighbor.to_string());
                    return Err(format!(
                        "Graph validation failed. Unconditional cycle detected: {}. Cycles \
                         must include at least one conditional (routed) edge to avoid infinite \
                         loops.",
                        cycle.join(" -> ")
                    ));
                }
                if !done.contains(neighbor) {
                    dfs(neighbor, path, in_stack, done, adj)?;
                }
            }
        }
        path.pop();
        in_stack.remove(node);
        done.insert(node.to_string());
        Ok(())
    }

    let mut names: Vec<&String> = node_names.iter().collect();
    names.sort();
    for name in names {
        if !done.contains(name.as_str()) {
            dfs(
                name,
                &mut Vec::new(),
                &mut in_stack,
                &mut done,
                &unconditional_adj,
            )?;
        }
    }
    Ok(())
}

fn validate_duplicate_node_names(nodes: &[BaseNode]) -> Result<HashSet<String>, String> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for node in nodes {
        *counts.entry(node.name()).or_insert(0) += 1;
    }
    let mut duplicates: Vec<&str> = counts
        .iter()
        .filter(|(_, count)| **count > 1)
        .map(|(name, _)| *name)
        .collect();
    duplicates.sort();

    if !duplicates.is_empty() {
        return Err(format!(
            "Graph validation failed. Duplicate node names found: {duplicates:?}. This means \
             multiple distinct node objects have the same name. If you intended to reuse the \
             same node, ensure you pass the exact same object instance. If you intended to have \
             distinct nodes, ensure they have unique names."
        ));
    }
    Ok(nodes.iter().map(|node| node.name().to_string()).collect())
}

fn validate_start_node(node_names: &HashSet<String>) -> Result<(), String> {
    if !node_names.contains(start().name()) {
        return Err(format!(
            "Graph validation failed. START node (name: '{}') not found in graph nodes.",
            start().name()
        ));
    }
    Ok(())
}

fn validate_connectivity(edges: &[Edge], node_names: &HashSet<String>) -> Result<(), String> {
    let mut to_nodes: HashSet<String> = HashSet::new();
    let mut adj: HashMap<&str, HashSet<&str>> = node_names
        .iter()
        .map(|name| (name.as_str(), HashSet::new()))
        .collect();
    for edge in edges {
        adj.entry(edge.from_node.name())
            .or_default()
            .insert(edge.to_node.name());
        to_nodes.insert(edge.to_node.name().to_string());
    }

    let mut reachable: HashSet<String> = HashSet::new();
    let mut stack = vec![start().name().to_string()];
    while let Some(node) = stack.pop() {
        if reachable.contains(&node) {
            continue;
        }
        reachable.insert(node.clone());
        if let Some(neighbors) = adj.get(node.as_str()) {
            for &neighbor in neighbors {
                if !reachable.contains(neighbor) {
                    stack.push(neighbor.to_string());
                }
            }
        }
    }

    let mut unreachable: Vec<&String> = node_names.difference(&reachable).collect();
    unreachable.sort();
    if !unreachable.is_empty() {
        return Err(format!(
            "Graph validation failed. The following nodes are unreachable from START: {unreachable:?}"
        ));
    }
    if to_nodes.contains(start().name()) {
        return Err(
            "Graph validation failed. START node must not have incoming edges.".to_string(),
        );
    }
    Ok(())
}

fn validate_duplicate_edges(edges: &[Edge]) -> Result<(), String> {
    let mut seen: HashSet<(String, String)> = HashSet::new();
    for edge in edges {
        let key = (
            edge.from_node.name().to_string(),
            edge.to_node.name().to_string(),
        );
        if !seen.insert(key) {
            return Err(format!(
                "Graph validation failed. Duplicate edge found: from={}, to={}",
                edge.from_node.name(),
                edge.to_node.name()
            ));
        }
    }
    Ok(())
}

fn validate_start_edges(edges: &[Edge]) -> Result<(), String> {
    for edge in edges {
        if edge.from_node.name() == start().name() && edge.route.is_some() {
            return Err(format!(
                "Graph validation failed. Edges from START must not have routes (edge to {} has \
                 a route).",
                edge.to_node.name()
            ));
        }
    }
    Ok(())
}

fn is_default_route(route: &RouteSpec) -> bool {
    matches!(route, RouteSpec::Single(RouteValue::Str(s)) if s == DEFAULT_ROUTE)
}

fn validate_default_routes(edges: &[Edge]) -> Result<(), String> {
    let mut default_route_edges: HashMap<String, String> = HashMap::new();
    for edge in edges {
        if let Some(RouteSpec::Many(values)) = &edge.route {
            if values
                .iter()
                .any(|v| matches!(v, RouteValue::Str(s) if s == DEFAULT_ROUTE))
            {
                return Err(format!(
                    "Graph validation failed. DEFAULT_ROUTE cannot be combined with other \
                     routes in a list (edge from={}, to={}). Use a separate edge for \
                     DEFAULT_ROUTE.",
                    edge.from_node.name(),
                    edge.to_node.name()
                ));
            }
        }
        if edge.route.as_ref().is_some_and(is_default_route) {
            let from_name = edge.from_node.name().to_string();
            if let Some(existing_to) = default_route_edges.get(&from_name) {
                return Err(format!(
                    "Graph validation failed. Multiple DEFAULT_ROUTE edges found from node {} to \
                     {} and {}",
                    from_name,
                    existing_to,
                    edge.to_node.name()
                ));
            }
            default_route_edges.insert(from_name, edge.to_node.name().to_string());
        }
    }
    Ok(())
}

fn validate_static_schemas(edges: &[Edge]) -> Result<(), String> {
    for edge in edges {
        let (from_node, to_node) = (&edge.from_node, &edge.to_node);
        if let (Some(output_schema), Some(input_schema)) =
            (from_node.output_schema(), to_node.input_schema())
        {
            if output_schema != input_schema {
                return Err(format!(
                    "Graph validation failed. Schema mismatch on edge {} -> {}. Output schema \
                     {output_schema:?} does not match input schema {input_schema:?}.",
                    from_node.name(),
                    to_node.name()
                ));
            }
        }
    }
    Ok(())
}

/// `_validate_chat_agent_wiring` — see this module's own doc for why
/// this is currently, correctly, always a no-op in this port.
fn validate_chat_agent_wiring(_edges: &[Edge]) -> Result<(), String> {
    Ok(())
}

fn compute_terminal_nodes(nodes: &[BaseNode], edges: &[Edge]) -> HashSet<String> {
    let from_names: HashSet<&str> = edges.iter().map(|edge| edge.from_node.name()).collect();
    nodes
        .iter()
        .filter(|node| node.name() != start().name() && !from_names.contains(node.name()))
        .map(|node| node.name().to_string())
        .collect()
}

/// `validate_graph`: validates the workflow graph and returns the set
/// of terminal node names.
pub fn validate_graph(nodes: &[BaseNode], edges: &[Edge]) -> Result<HashSet<String>, String> {
    let node_names = validate_duplicate_node_names(nodes)?;
    validate_start_node(&node_names)?;
    validate_start_edges(edges)?;
    validate_connectivity(edges, &node_names)?;
    validate_duplicate_edges(edges)?;
    validate_default_routes(edges)?;
    detect_unconditional_cycles(edges, &node_names)?;
    validate_static_schemas(edges)?;
    validate_chat_agent_wiring(edges)?;
    Ok(compute_terminal_nodes(nodes, edges))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow_base_node::NoopNodeBehavior;
    use crate::workflow_graph::Graph;

    fn node(name: &str) -> BaseNode {
        BaseNode::new(name, NoopNodeBehavior).unwrap()
    }

    #[test]
    fn a_minimal_valid_graph_passes() {
        let a = node("a");
        let mut graph = Graph::new(vec![Edge::new(start(), a.clone(), None)]);
        assert!(graph.validate().is_ok());
        assert_eq!(
            graph.terminal_node_names(),
            &["a".to_string()].into_iter().collect()
        );
    }

    #[test]
    fn rejects_duplicate_node_names() {
        let a1 = node("a");
        let a2 = node("a");
        let edges = vec![
            Edge::new(start(), a1.clone(), None),
            Edge::new(a1.clone(), a2.clone(), None),
        ];
        let nodes = vec![start(), a1, a2];
        let err = validate_graph(&nodes, &edges).unwrap_err();
        assert!(err.contains("Duplicate node names"), "{err}");
    }

    #[test]
    fn rejects_a_graph_missing_start() {
        let a = node("a");
        let b = node("b");
        let nodes = vec![a.clone(), b.clone()];
        let edges = vec![Edge::new(a, b, None)];
        let err = validate_graph(&nodes, &edges).unwrap_err();
        assert!(err.contains("START node"), "{err}");
    }

    #[test]
    fn rejects_unreachable_nodes() {
        let a = node("a");
        let b = node("b");
        let nodes = vec![start(), a.clone(), b];
        let edges = vec![Edge::new(start(), a, None)];
        let err = validate_graph(&nodes, &edges).unwrap_err();
        assert!(err.contains("unreachable"), "{err}");
    }

    #[test]
    fn rejects_incoming_edges_to_start() {
        let a = node("a");
        let nodes = vec![start(), a.clone()];
        let edges = vec![
            Edge::new(start(), a.clone(), None),
            Edge::new(a, start(), None),
        ];
        let err = validate_graph(&nodes, &edges).unwrap_err();
        assert!(err.contains("must not have incoming edges"), "{err}");
    }

    #[test]
    fn rejects_duplicate_edges() {
        let a = node("a");
        let nodes = vec![start(), a.clone()];
        let edges = vec![
            Edge::new(start(), a.clone(), None),
            Edge::new(start(), a, None),
        ];
        let err = validate_graph(&nodes, &edges).unwrap_err();
        assert!(err.contains("Duplicate edge"), "{err}");
    }

    #[test]
    fn rejects_a_start_edge_with_a_route() {
        let a = node("a");
        let nodes = vec![start(), a.clone()];
        let edges = vec![Edge::new(
            start(),
            a,
            Some(RouteSpec::Single(RouteValue::Str("x".to_string()))),
        )];
        let err = validate_graph(&nodes, &edges).unwrap_err();
        assert!(err.contains("must not have routes"), "{err}");
    }

    #[test]
    fn rejects_default_route_combined_with_other_routes_in_a_list() {
        let a = node("a");
        let b = node("b");
        let nodes = vec![start(), a.clone(), b.clone()];
        let edges = vec![
            Edge::new(start(), a.clone(), None),
            Edge::new(
                a,
                b,
                Some(RouteSpec::Many(vec![
                    RouteValue::Str("x".to_string()),
                    RouteValue::Str(DEFAULT_ROUTE.to_string()),
                ])),
            ),
        ];
        let err = validate_graph(&nodes, &edges).unwrap_err();
        assert!(err.contains("DEFAULT_ROUTE cannot be combined"), "{err}");
    }

    #[test]
    fn rejects_multiple_default_route_edges_from_the_same_node() {
        let a = node("a");
        let b = node("b");
        let c = node("c");
        let nodes = vec![start(), a.clone(), b.clone(), c.clone()];
        let edges = vec![
            Edge::new(start(), a.clone(), None),
            Edge::new(
                a.clone(),
                b,
                Some(RouteSpec::Single(RouteValue::Str(
                    DEFAULT_ROUTE.to_string(),
                ))),
            ),
            Edge::new(
                a,
                c,
                Some(RouteSpec::Single(RouteValue::Str(
                    DEFAULT_ROUTE.to_string(),
                ))),
            ),
        ];
        let err = validate_graph(&nodes, &edges).unwrap_err();
        assert!(err.contains("Multiple DEFAULT_ROUTE"), "{err}");
    }

    #[test]
    fn rejects_an_unconditional_cycle() {
        let a = node("a");
        let b = node("b");
        let nodes = vec![start(), a.clone(), b.clone()];
        let edges = vec![
            Edge::new(start(), a.clone(), None),
            Edge::new(a.clone(), b.clone(), None),
            Edge::new(b, a, None),
        ];
        let err = validate_graph(&nodes, &edges).unwrap_err();
        assert!(err.contains("Unconditional cycle"), "{err}");
    }

    #[test]
    fn allows_a_cycle_with_a_conditional_edge() {
        let a = node("a");
        let b = node("b");
        let nodes = vec![start(), a.clone(), b.clone()];
        let edges = vec![
            Edge::new(start(), a.clone(), None),
            Edge::new(a.clone(), b.clone(), None),
            Edge::new(
                b,
                a,
                Some(RouteSpec::Single(RouteValue::Str("retry".to_string()))),
            ),
        ];
        assert!(validate_graph(&nodes, &edges).is_ok());
    }

    #[test]
    fn computes_terminal_nodes() {
        let a = node("a");
        let b = node("b");
        let nodes = vec![start(), a.clone(), b.clone()];
        let edges = vec![Edge::new(start(), a.clone(), None), Edge::new(a, b, None)];
        let terminal = validate_graph(&nodes, &edges).unwrap();
        assert_eq!(terminal, ["b".to_string()].into_iter().collect());
    }
}
