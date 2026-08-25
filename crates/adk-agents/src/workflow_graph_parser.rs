//! Capability C0327 (narrowed): `parse_edge_items` and the
//! `NodeLike`/`RoutingMap`/`ChainElement`/`EdgeItem` chain-building
//! sugar `Graph::from_edge_items` builds on, ported from
//! `google.adk.workflow.utils._graph_parser`. Part of the P7
//! workflow/graph engine — see `workflow_node_state.rs`'s module doc
//! for the standing crate-placement decision.
//!
//! **`NodeLike`, narrowed**: the source's `NodeLike` is `BaseNode |
//! BaseTool | Callable[..., Any] | Literal["START"]`. This port's
//! [`NodeLike`] keeps only the `BaseNode`/`"START"` cases —
//! `BaseAgent`/`BaseTool`/raw-callable chain elements need
//! `build_node`/`is_node_like` (C0326), themselves needing
//! `FunctionNode`/`_ToolNode`/`_ParallelWorker` and `BaseTool` — see
//! `workflow_graph.rs`'s own module doc for why that stays out of
//! scope (the `adk-tools`/`adk-agents` crate-cycle, C0355/C0356).
//!
//! **`_get_or_build_node`, degenerates to identity dedup**: for the
//! source's *own* `BaseNode`-already case, `build_node(node_like)`
//! already returns `node_like` unchanged (`wrapped is not node_like`
//! is `False`) — the function only ever *builds* something new for the
//! `BaseAgent`/`BaseTool`/callable cases this port's narrowed
//! [`NodeLike`] excludes entirely. So [`get_or_build_node`] reduces to:
//! `"START"` → the singleton [`crate::workflow_base_node::start`];
//! otherwise, dedup by [`BaseNode::ptr_eq`] against nodes already seen
//! in this parse. The source dedups by `id(node_like)` (identity of the
//! *input* reference) rather than the built node's identity — for a
//! `BaseNode` input, input and output are the same object, so deduping
//! on the (here, only possible) output identity via `ptr_eq` is the
//! exact same behavior, not a narrowing.
//!
//! **Runtime `isinstance(route_key, ...)`/`is_node_like` checks,
//! dropped**: the source validates routing-map keys are `RouteValue`
//! and chain elements are `NodeLike` at runtime, since Python has
//! nothing else enforcing it. This port's [`RouteValue`]/[`NodeLike`]
//! enums make invalid values unrepresentable at the type level, so
//! there is nothing left to check at parse time.

use crate::workflow_base_node::{start, BaseNode};
use crate::workflow_graph::{Edge, RouteSpec, RouteValue};

/// `workflow._graph.NodeLike`, narrowed — see this module's own doc.
#[derive(Debug, Clone)]
pub enum NodeLike {
    Node(BaseNode),
    Start,
}

/// A `NodeLike`, or a fan-out tuple of them — the value side of a
/// [`RoutingMap`] entry, or a routing-map target more generally.
#[derive(Debug, Clone)]
pub enum RoutingTarget {
    Single(NodeLike),
    FanOut(Vec<NodeLike>),
}

/// `workflow._graph.RoutingMap`: a mapping from route values to
/// destination nodes — syntactic sugar for declaring multiple routed
/// edges from a single source. A `Vec` of pairs (not a `HashMap`)
/// preserves insertion order, matching Python `dict`'s own iteration
/// order (`_expand_routing_map` iterates `routing_map.items()`).
pub type RoutingMap = Vec<(RouteValue, RoutingTarget)>;

/// `workflow._graph.ChainElement`: a single [`NodeLike`], a fan-out
/// tuple of them, or a [`RoutingMap`].
#[derive(Debug, Clone)]
pub enum ChainElement {
    Single(NodeLike),
    FanOut(Vec<NodeLike>),
    RoutingMap(RoutingMap),
}

/// `workflow._graph.EdgeItem`: an explicit [`Edge`], or a chain
/// (sequence of [`ChainElement`]s to be expanded pairwise).
pub enum EdgeItem {
    Edge(Edge),
    Chain(Vec<ChainElement>),
}

fn flatten_routing_target(target: &RoutingTarget) -> Vec<NodeLike> {
    match target {
        RoutingTarget::Single(node) => vec![node.clone()],
        RoutingTarget::FanOut(nodes) => nodes.clone(),
    }
}

/// `_flatten_element`: flattens a chain element into a list of
/// individual nodes.
fn flatten_element(element: &ChainElement) -> Vec<NodeLike> {
    match element {
        ChainElement::Single(node) => vec![node.clone()],
        ChainElement::FanOut(nodes) => nodes.clone(),
        ChainElement::RoutingMap(map) => map
            .iter()
            .flat_map(|(_, target)| flatten_routing_target(target))
            .collect(),
    }
}

/// `_get_or_build_node` — see this module's own doc for why this
/// degenerates to identity dedup in this port.
fn get_or_build_node(node_like: &NodeLike, node_map: &mut Vec<BaseNode>) -> BaseNode {
    let node = match node_like {
        NodeLike::Start => start(),
        NodeLike::Node(node) => node.clone(),
    };
    if let Some(existing) = node_map.iter().find(|seen| seen.ptr_eq(&node)) {
        return existing.clone();
    }
    node_map.push(node.clone());
    node
}

/// `_expand_routing_map`: expands a routing map into individual
/// `(from, to, route)` triples.
fn expand_routing_map<'a>(
    from_element: &'a ChainElement,
    routing_map: &'a RoutingMap,
) -> Result<Vec<(&'a ChainElement, &'a RoutingTarget, RouteValue)>, String> {
    if routing_map.is_empty() {
        return Err(
            "Routing map must not be empty. Provide at least one route -> node mapping."
                .to_string(),
        );
    }
    Ok(routing_map
        .iter()
        .map(|(route, target)| (from_element, target, route.clone()))
        .collect())
}

/// `_process_explicit_edge`: processes an explicit `Edge` object,
/// running its `from_node`/`to_node` through [`get_or_build_node`] so
/// they collapse onto the same instance if seen elsewhere in this
/// parse (matching the source's own dedup-by-identity behavior).
fn process_explicit_edge(edge: Edge, node_map: &mut Vec<BaseNode>, graph_edges: &mut Vec<Edge>) {
    let from_node = get_or_build_node(&NodeLike::Node(edge.from_node), node_map);
    let to_node = get_or_build_node(&NodeLike::Node(edge.to_node), node_map);
    graph_edges.push(Edge::new(from_node, to_node, edge.route));
}

/// `_process_routing_map_edge`: processes edges where the destination
/// is a routing map.
fn process_routing_map_edge(
    from_el: &ChainElement,
    to_el: &RoutingMap,
    node_map: &mut Vec<BaseNode>,
    graph_edges: &mut Vec<Edge>,
) -> Result<(), String> {
    if matches!(from_el, ChainElement::RoutingMap(_)) {
        return Err(
            "Consecutive routing maps are not allowed in a chain. Split them into separate edge items."
                .to_string(),
        );
    }

    for (exp_from, exp_to, route) in expand_routing_map(from_el, to_el)? {
        for from_node in flatten_element(exp_from) {
            for to_node in flatten_routing_target(exp_to) {
                let from_node = get_or_build_node(&from_node, node_map);
                let to_node = get_or_build_node(&to_node, node_map);
                graph_edges.push(Edge::new(
                    from_node,
                    to_node,
                    Some(RouteSpec::Single(route.clone())),
                ));
            }
        }
    }
    Ok(())
}

/// `_process_unconditional_edge`: processes unconditional edges
/// between elements.
fn process_unconditional_edge(
    from_el: &ChainElement,
    to_el: &ChainElement,
    node_map: &mut Vec<BaseNode>,
    graph_edges: &mut Vec<Edge>,
) {
    for from_node in flatten_element(from_el) {
        for to_node in flatten_element(to_el) {
            let from_node = get_or_build_node(&from_node, node_map);
            let to_node = get_or_build_node(&to_node, node_map);
            graph_edges.push(Edge::new(from_node, to_node, None));
        }
    }
}

/// `_process_chain`: processes a chain of elements (pairwise).
fn process_chain(
    chain: &[ChainElement],
    node_map: &mut Vec<BaseNode>,
    graph_edges: &mut Vec<Edge>,
) -> Result<(), String> {
    for pair in chain.windows(2) {
        let from_el = &pair[0];
        let to_el = &pair[1];
        if let ChainElement::RoutingMap(map) = to_el {
            process_routing_map_edge(from_el, map, node_map, graph_edges)?;
        } else {
            process_unconditional_edge(from_el, to_el, node_map, graph_edges);
        }
    }
    Ok(())
}

/// `parse_edge_items`: parses a list of edge items into a flat list of
/// `Edge` objects.
pub fn parse_edge_items(edge_items: Vec<EdgeItem>) -> Result<Vec<Edge>, String> {
    let mut node_map: Vec<BaseNode> = Vec::new();
    let mut graph_edges: Vec<Edge> = Vec::new();

    for item in edge_items {
        match item {
            EdgeItem::Edge(edge) => process_explicit_edge(edge, &mut node_map, &mut graph_edges),
            EdgeItem::Chain(chain) => process_chain(&chain, &mut node_map, &mut graph_edges)?,
        }
    }

    Ok(graph_edges)
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
    fn an_explicit_edge_item_passes_through() {
        let a = node("a");
        let b = node("b");
        let edges =
            parse_edge_items(vec![EdgeItem::Edge(Edge::new(a.clone(), b.clone(), None))]).unwrap();
        assert_eq!(edges.len(), 1);
        assert!(edges[0].from_node.ptr_eq(&a));
        assert!(edges[0].to_node.ptr_eq(&b));
    }

    #[test]
    fn a_simple_chain_builds_unconditional_edges() {
        let a = node("a");
        let b = node("b");
        let c = node("c");
        let edges = parse_edge_items(vec![EdgeItem::Chain(vec![
            ChainElement::Single(NodeLike::Node(a.clone())),
            ChainElement::Single(NodeLike::Node(b.clone())),
            ChainElement::Single(NodeLike::Node(c.clone())),
        ])])
        .unwrap();
        assert_eq!(edges.len(), 2);
        assert!(edges[0].from_node.ptr_eq(&a) && edges[0].to_node.ptr_eq(&b));
        assert!(edges[1].from_node.ptr_eq(&b) && edges[1].to_node.ptr_eq(&c));
    }

    #[test]
    fn a_fan_out_tuple_produces_one_edge_per_target() {
        let a = node("a");
        let b = node("b");
        let c = node("c");
        let edges = parse_edge_items(vec![EdgeItem::Chain(vec![
            ChainElement::Single(NodeLike::Node(a.clone())),
            ChainElement::FanOut(vec![NodeLike::Node(b.clone()), NodeLike::Node(c.clone())]),
        ])])
        .unwrap();
        assert_eq!(edges.len(), 2);
        assert!(edges.iter().all(|e| e.from_node.ptr_eq(&a)));
    }

    #[test]
    fn a_routing_map_tags_each_edge_with_its_route() {
        let a = node("a");
        let b = node("b");
        let c = node("c");
        let routing_map: RoutingMap = vec![
            (
                RouteValue::Str("yes".to_string()),
                RoutingTarget::Single(NodeLike::Node(b.clone())),
            ),
            (
                RouteValue::Str("no".to_string()),
                RoutingTarget::Single(NodeLike::Node(c.clone())),
            ),
        ];
        let edges = parse_edge_items(vec![EdgeItem::Chain(vec![
            ChainElement::Single(NodeLike::Node(a.clone())),
            ChainElement::RoutingMap(routing_map),
        ])])
        .unwrap();
        assert_eq!(edges.len(), 2);
        assert_eq!(
            edges[0].route,
            Some(RouteSpec::Single(RouteValue::Str("yes".to_string())))
        );
        assert_eq!(
            edges[1].route,
            Some(RouteSpec::Single(RouteValue::Str("no".to_string())))
        );
    }

    #[test]
    fn an_empty_routing_map_is_rejected() {
        let a = node("a");
        let err = parse_edge_items(vec![EdgeItem::Chain(vec![
            ChainElement::Single(NodeLike::Node(a)),
            ChainElement::RoutingMap(Vec::new()),
        ])])
        .unwrap_err();
        assert!(err.contains("must not be empty"));
    }

    #[test]
    fn consecutive_routing_maps_are_rejected() {
        let b = node("b");
        let map1: RoutingMap = vec![(
            RouteValue::Str("x".to_string()),
            RoutingTarget::Single(NodeLike::Node(b.clone())),
        )];
        let map2: RoutingMap = vec![(
            RouteValue::Str("y".to_string()),
            RoutingTarget::Single(NodeLike::Node(b)),
        )];
        let err = parse_edge_items(vec![EdgeItem::Chain(vec![
            ChainElement::RoutingMap(map1),
            ChainElement::RoutingMap(map2),
        ])])
        .unwrap_err();
        assert!(err.contains("Consecutive routing maps"));
    }

    #[test]
    fn the_same_node_reference_collapses_to_one_instance() {
        let a = node("a");
        let b = node("b");
        // `a` appears in two separate edge items -- both must resolve to
        // the exact same node, not two distinct copies.
        let edges = parse_edge_items(vec![
            EdgeItem::Edge(Edge::new(a.clone(), b.clone(), None)),
            EdgeItem::Chain(vec![
                ChainElement::Single(NodeLike::Node(a.clone())),
                ChainElement::Single(NodeLike::Node(node("c"))),
            ]),
        ])
        .unwrap();
        assert!(edges[0].from_node.ptr_eq(&edges[1].from_node));
    }

    #[test]
    fn start_resolves_to_the_singleton_start_node() {
        let a = node("a");
        let edges = parse_edge_items(vec![EdgeItem::Chain(vec![
            ChainElement::Single(NodeLike::Start),
            ChainElement::Single(NodeLike::Node(a)),
        ])])
        .unwrap();
        assert!(edges[0].from_node.ptr_eq(&start()));
    }

    #[test]
    fn from_edge_items_builds_a_graph_with_inferred_nodes() {
        let a = node("a");
        let b = node("b");
        let graph = Graph::from_edge_items(vec![EdgeItem::Chain(vec![
            ChainElement::Single(NodeLike::Start),
            ChainElement::Single(NodeLike::Node(a.clone())),
            ChainElement::Single(NodeLike::Node(b.clone())),
        ])])
        .unwrap();
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.edges.len(), 2);
    }
}
