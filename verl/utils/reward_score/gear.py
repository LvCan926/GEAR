from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Optional


ALLOWED_NODE_TYPES = {"foundation", "bonus", "penalty", "trigger"}
ALLOWED_EDGE_TYPES = {"weak_prerequisite", "strong_prerequisite", "trigger"}


@dataclass(frozen=True)
class GearNode:
    rubric_idx: int
    rubric_id: str
    weight: float
    node_type: str
    criterion: str


@dataclass(frozen=True)
class GearEdge:
    parent: str
    child: str
    edge_type: str


@dataclass
class GearGraph:
    nodes: List[GearNode]
    edges: List[GearEdge]
    topo_order: List[str]
    id_to_index: Dict[str, int]
    parent_edges_by_child: Dict[str, List[GearEdge]]


@dataclass
class GearAggregationResult:
    aggregation_mode: str
    reward: float
    flat_reward: float
    hard_reward: float
    dag_reward: float
    p_list: List[float]
    q_list: List[float]
    flat_q_list: List[float]
    hard_q_list: List[float]
    dag_q_list: List[float]
    criteria_met_list: List[bool]
    node_types: List[str]
    graph_edges: List[Dict[str, str]]
    rubric_ids: List[str]


def _clamp_probability(value: Any, default: float = 0.0) -> float:
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, prob))


def _normalize_node_type(node_type: Optional[str], weight: float) -> str:
    if node_type in ALLOWED_NODE_TYPES:
        return node_type
    return "penalty" if weight < 0 else "foundation"


def _normalize_edge_type(edge_type: Optional[str]) -> Optional[str]:
    if edge_type in ALLOWED_EDGE_TYPES:
        return edge_type
    return None


def _rubric_tags(rubric: Dict[str, Any]) -> Dict[str, Any]:
    tags = rubric.get("tags", {})
    return tags if isinstance(tags, dict) else {}


def _rubric_id(rubric: Dict[str, Any], rubric_idx: int) -> str:
    rubric_id = rubric.get("id")
    if isinstance(rubric_id, str) and rubric_id:
        return rubric_id
    return f"r{rubric_idx + 1}"


def _build_nodes(rubrics: Iterable[Dict[str, Any]], graph: Optional[Dict[str, Any]]) -> List[GearNode]:
    graph_nodes = {}
    if isinstance(graph, dict):
        for node in graph.get("nodes", []) or []:
            if not isinstance(node, dict):
                continue
            node_id = node.get("id")
            if isinstance(node_id, str) and node_id:
                graph_nodes[node_id] = node

    nodes: List[GearNode] = []
    for rubric_idx, rubric in enumerate(rubrics):
        weight = float(rubric.get("points", 0.0))
        rubric_id = _rubric_id(rubric, rubric_idx)
        tags = _rubric_tags(rubric)
        graph_node = graph_nodes.get(rubric_id, {})
        node_type = _normalize_node_type(
            graph_node.get("node_type") or tags.get("node_type"),
            weight,
        )
        nodes.append(
            GearNode(
                rubric_idx=rubric_idx,
                rubric_id=rubric_id,
                weight=weight,
                node_type=node_type,
                criterion=str(rubric.get("criterion", "")),
            )
        )
    return nodes


def _would_create_cycle(adjacency: Dict[str, List[str]], parent: str, child: str) -> bool:
    stack = [child]
    seen = set()
    while stack:
        current = stack.pop()
        if current == parent:
            return True
        if current in seen:
            continue
        seen.add(current)
        stack.extend(adjacency.get(current, []))
    return False


def _build_edges(nodes: List[GearNode], graph: Optional[Dict[str, Any]]) -> List[GearEdge]:
    if not isinstance(graph, dict):
        return []

    node_ids = {node.rubric_id for node in nodes}
    seen = set()
    adjacency: Dict[str, List[str]] = defaultdict(list)
    normalized_edges: List[GearEdge] = []

    for raw_edge in graph.get("edges", []) or []:
        if not isinstance(raw_edge, dict):
            continue
        parent = raw_edge.get("parent")
        child = raw_edge.get("child")
        edge_type = _normalize_edge_type(raw_edge.get("type"))
        if not parent or not child or not edge_type:
            continue
        if parent == child or parent not in node_ids or child not in node_ids:
            continue
        edge_key = (parent, child, edge_type)
        if edge_key in seen:
            continue
        if _would_create_cycle(adjacency, parent, child):
            continue
        seen.add(edge_key)
        adjacency[parent].append(child)
        normalized_edges.append(GearEdge(parent=parent, child=child, edge_type=edge_type))

    return normalized_edges


def _topological_sort(nodes: List[GearNode], edges: List[GearEdge]) -> List[str]:
    indegree = {node.rubric_id: 0 for node in nodes}
    adjacency: Dict[str, List[str]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.parent].append(edge.child)
        indegree[edge.child] += 1

    queue = deque([node.rubric_id for node in nodes if indegree[node.rubric_id] == 0])
    topo_order: List[str] = []

    while queue:
        current = queue.popleft()
        topo_order.append(current)
        for child in adjacency.get(current, []):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    if len(topo_order) != len(nodes):
        return [node.rubric_id for node in nodes]
    return topo_order


def parse_gear_graph(reward_model: Optional[Dict[str, Any]]) -> GearGraph:
    reward_model = reward_model or {}
    rubrics = reward_model.get("rubrics", []) or []
    graph = reward_model.get("graph")

    nodes = _build_nodes(rubrics, graph)
    edges = _build_edges(nodes, graph)
    topo_order = _topological_sort(nodes, edges)
    id_to_index = {node.rubric_id: node.rubric_idx for node in nodes}

    parent_edges_by_child: Dict[str, List[GearEdge]] = defaultdict(list)
    for edge in edges:
        parent_edges_by_child[edge.child].append(edge)

    return GearGraph(
        nodes=nodes,
        edges=edges,
        topo_order=topo_order,
        id_to_index=id_to_index,
        parent_edges_by_child=dict(parent_edges_by_child),
    )


def _normalization_denominator(nodes: List[GearNode], normalization_mode: str) -> float:
    if normalization_mode == "positive_sum":
        denom = sum(max(node.weight, 0.0) for node in nodes)
    else:
        denom = sum(abs(node.weight) for node in nodes)
    return denom if denom > 0 else 1.0


def _score_from_q(graph: GearGraph, q_by_id: Dict[str, float], normalization_mode: str) -> float:
    denom = _normalization_denominator(graph.nodes, normalization_mode)
    numerator = 0.0
    for node in graph.nodes:
        numerator += node.weight * q_by_id.get(node.rubric_id, 0.0)
    return numerator / denom


def _flat_q(graph: GearGraph, p_by_id: Dict[str, float]) -> Dict[str, float]:
    return {node.rubric_id: p_by_id.get(node.rubric_id, 0.0) for node in graph.nodes}


def _hard_q(graph: GearGraph, p_by_id: Dict[str, float], met_by_id: Dict[str, bool]) -> Dict[str, float]:
    q_by_id: Dict[str, float] = {}
    for node_id in graph.topo_order:
        active = True
        for edge in graph.parent_edges_by_child.get(node_id, []):
            if not met_by_id.get(edge.parent, False):
                active = False
                break
        q_by_id[node_id] = p_by_id.get(node_id, 0.0) if active else 0.0
    return q_by_id


def _dag_q_approx(
    graph: GearGraph,
    p_by_id: Dict[str, float],
    lambda_by_edge_type: Dict[str, float],
) -> Dict[str, float]:
    q_by_id: Dict[str, float] = {}
    for node_id in graph.topo_order:
        gate = 1.0
        for edge in graph.parent_edges_by_child.get(node_id, []):
            lambda_value = _clamp_probability(lambda_by_edge_type.get(edge.edge_type, 1.0), default=1.0)
            parent_q = q_by_id.get(edge.parent, p_by_id.get(edge.parent, 0.0))
            gate *= parent_q + (1.0 - parent_q) * lambda_value
        q_by_id[node_id] = p_by_id.get(node_id, 0.0) * gate
    return q_by_id


def _conditional_prob(
    node_id: str,
    state_by_id: Dict[str, int],
    p_by_id: Dict[str, float],
    parent_edges: Dict[str, List[GearEdge]],
    lambda_by_edge_type: Dict[str, float],
) -> float:
    gate = 1.0
    for edge in parent_edges.get(node_id, []):
        lambda_value = _clamp_probability(lambda_by_edge_type.get(edge.edge_type, 1.0), default=1.0)
        parent_state = state_by_id.get(edge.parent, 0)
        gate *= 1.0 if parent_state == 1 else lambda_value
    return _clamp_probability(p_by_id.get(node_id, 0.0), default=0.0) * gate


def _dag_q_exact(
    graph: GearGraph,
    p_by_id: Dict[str, float],
    lambda_by_edge_type: Dict[str, float],
) -> Dict[str, float]:
    node_ids = graph.topo_order
    marginals = {node_id: 0.0 for node_id in node_ids}

    for assignment in product((0, 1), repeat=len(node_ids)):
        state_by_id = {node_id: assignment[idx] for idx, node_id in enumerate(node_ids)}
        joint_prob = 1.0
        for node_id in node_ids:
            prob_true = _conditional_prob(
                node_id=node_id,
                state_by_id=state_by_id,
                p_by_id=p_by_id,
                parent_edges=graph.parent_edges_by_child,
                lambda_by_edge_type=lambda_by_edge_type,
            )
            joint_prob *= prob_true if state_by_id[node_id] == 1 else (1.0 - prob_true)
            if joint_prob == 0.0:
                break
        if joint_prob == 0.0:
            continue
        for node_id, state in state_by_id.items():
            if state == 1:
                marginals[node_id] += joint_prob

    return marginals


def aggregate_gear_reward(
    reward_model: Optional[Dict[str, Any]],
    p_list: List[float],
    criteria_met_list: List[bool],
    aggregation_mode: str = "dag",
    normalization_mode: str = "positive_sum",
    inference_mode: str = "approx",
    exact_if_num_nodes_le: int = 10,
    lambda_by_edge_type: Optional[Dict[str, float]] = None,
) -> GearAggregationResult:
    graph = parse_gear_graph(reward_model)
    lambda_by_edge_type = lambda_by_edge_type or {}

    p_by_id = {
        node.rubric_id: _clamp_probability(p_list[node.rubric_idx], default=0.0)
        for node in graph.nodes
    }
    met_by_id = {
        node.rubric_id: bool(criteria_met_list[node.rubric_idx])
        for node in graph.nodes
    }

    flat_q = _flat_q(graph, p_by_id)
    hard_q = _hard_q(graph, p_by_id, met_by_id)

    use_exact = inference_mode == "exact_auto" and len(graph.nodes) <= exact_if_num_nodes_le
    if use_exact:
        dag_q = _dag_q_exact(graph, p_by_id, lambda_by_edge_type)
    else:
        dag_q = _dag_q_approx(graph, p_by_id, lambda_by_edge_type)

    flat_reward = _score_from_q(graph, flat_q, normalization_mode)
    hard_reward = _score_from_q(graph, hard_q, normalization_mode)
    dag_reward = _score_from_q(graph, dag_q, normalization_mode)

    q_by_mode = {
        "flat": flat_q,
        "hard": hard_q,
        "dag": dag_q,
    }
    selected_mode = aggregation_mode if aggregation_mode in q_by_mode else "dag"
    selected_q = q_by_mode[selected_mode]
    selected_reward = {
        "flat": flat_reward,
        "hard": hard_reward,
        "dag": dag_reward,
    }[selected_mode]

    def _list_from_q(q_by_id: Dict[str, float]) -> List[float]:
        return [q_by_id.get(node.rubric_id, 0.0) for node in graph.nodes]

    return GearAggregationResult(
        aggregation_mode=selected_mode,
        reward=selected_reward,
        flat_reward=flat_reward,
        hard_reward=hard_reward,
        dag_reward=dag_reward,
        p_list=[p_by_id.get(node.rubric_id, 0.0) for node in graph.nodes],
        q_list=_list_from_q(selected_q),
        flat_q_list=_list_from_q(flat_q),
        hard_q_list=_list_from_q(hard_q),
        dag_q_list=_list_from_q(dag_q),
        criteria_met_list=[met_by_id.get(node.rubric_id, False) for node in graph.nodes],
        node_types=[node.node_type for node in graph.nodes],
        graph_edges=[
            {"parent": edge.parent, "child": edge.child, "type": edge.edge_type}
            for edge in graph.edges
        ],
        rubric_ids=[node.rubric_id for node in graph.nodes],
    )
