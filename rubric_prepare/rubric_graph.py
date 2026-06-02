import argparse
import json
import os
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()

PROGRESS_PRINT_EVERY = int(os.getenv("GEAR_ANNOTATE_PROGRESS_PRINT_EVERY", "20"))
PROGRESS_HEARTBEAT_SEC = float(os.getenv("GEAR_ANNOTATE_PROGRESS_HEARTBEAT_SEC", "30"))


ALLOWED_NODE_TYPES = {"foundation", "bonus", "penalty", "trigger"}
ALLOWED_EDGE_TYPES = {"weak_prerequisite", "strong_prerequisite", "trigger"}
ALLOWED_EDGE_LABELS = ALLOWED_EDGE_TYPES | {"no_edge"}
EDGE_TYPE_PRIORITY = {
    "trigger": 0,
    "strong_prerequisite": 1,
    "weak_prerequisite": 2,
}
ALLOWED_TYPE_PAIR_TO_EDGE_TYPES = {
    ("foundation", "foundation"): {"weak_prerequisite", "strong_prerequisite"},
    ("foundation", "bonus"): {"weak_prerequisite", "strong_prerequisite"},
    ("foundation", "penalty"): {"weak_prerequisite", "strong_prerequisite"},
    ("trigger", "bonus"): {"trigger"},
    ("trigger", "penalty"): {"trigger"},
}


NODE_SYSTEM_PROMPT = """
You are annotating HealthBench rubrics for structured reward aggregation.

Task:
- Classify each rubric into exactly one node_type.

Allowed node_type values:
- foundation
- bonus
- penalty
- trigger

Definitions:
- foundation: a core criterion for whether the answer is basically correct, safe, responsive, or complete.
- bonus: a helpful extra detail that improves the answer, but is not the main requirement.
- penalty: an undesirable mistake, omission, or unsafe statement that should subtract credit when present.
- trigger: a gating condition whose main role is to activate another rubric, usually a context-specific bonus or penalty.

Few-shot examples:
- "States that potassium 3.9 mmol/L is normal." -> foundation
- "Adds that warm compresses may help a likely muscle strain." -> bonus
- "Incorrectly says the symptoms are definitely from low potassium." -> penalty
- "Notes that the patient is pregnant, which would activate pregnancy-specific safety rubrics." -> trigger

Return JSON only with this schema:
{
  "nodes": [
    {"id": "r1", "node_type": "foundation"}
  ]
}

Rules:
- Use only the provided rubric ids.
- Do not modify rubric text.
- Every rubric id must appear exactly once.
- Prefer foundation over bonus only when the criterion is truly core.
- Use trigger only when the rubric mainly serves as a gate for some other rubric.
""".strip()


EDGE_SYSTEM_PROMPT = """
You are annotating directed rubric dependencies for GEAR.

Task:
- For each candidate directed pair, decide exactly one label.

Allowed labels:
- no_edge
- weak_prerequisite
- strong_prerequisite
- trigger

Definitions:
- weak_prerequisite: the child is meaningfully related to the parent and usually makes more sense if the parent is satisfied, but the dependency is not absolute.
- strong_prerequisite: the child strongly depends on the parent; if the parent is absent, the child should usually not receive credit or penalty.
- trigger: the parent activates a context-specific child bonus or child penalty.
- no_edge: the two rubrics are largely independent.

Few-shot examples:
- Parent: "Explains that potassium 3.9 mmol/L is normal."
  Child: "Concludes that low potassium is unlikely to be the main cause."
  Label: strong_prerequisite
- Parent: "Suggests a likely muscle strain from frequent soccer."
  Child: "Adds that warm compresses may help."
  Label: weak_prerequisite
- Parent: "Mentions the patient is pregnant."
  Child: "Incorrectly recommends a pregnancy-contraindicated medication."
  Label: trigger
- Parent: "Recommends colon cancer screening starting now."
  Child: "Mentions using warm compresses for leg cramps."
  Label: no_edge

Return JSON only with this schema:
{
  "edges": [
    {"parent": "r1", "child": "r2", "type": "weak_prerequisite"}
  ]
}

Rules:
- Use only the provided candidate pairs.
- Return exactly one decision for each candidate pair.
- Use "no_edge" only when there is no meaningful dependency.
- Do not make the graph artificially sparse; add an edge when the child clearly depends on, specializes, or is activated by the parent.
""".strip()


WRITINGBENCH_NODE_SYSTEM_PROMPT = """
You are annotating WritingBench rubrics for structured reward aggregation.

Task:
- Classify each writing-evaluation rubric into exactly one node_type.

Allowed node_type values:
- foundation
- bonus
- penalty
- trigger

Definitions for WritingBench:
- foundation: a core writing criterion needed for a high-quality answer, such as task fulfillment, content coverage, factual grounding, structure, format compliance, audience fit, style compliance, or length compliance.
- bonus: an extra refinement that improves polish, creativity, nuance, or expressiveness but is not central to satisfying the user's writing request.
- penalty: an explicitly undesirable failure mode that should subtract credit when present. Do not use penalty merely because a criterion has low score bands.
- trigger: a gating condition whose main role is to activate another criterion. Use this rarely for conditional requirements.

Return JSON only with this schema:
{
  "nodes": [
    {"id": "r1", "node_type": "foundation"}
  ]
}

Rules:
- Use only the provided rubric ids.
- Every rubric id must appear exactly once.
- Most WritingBench checklist items are foundation criteria.
- Prefer foundation for style, format, length, and content requirements when they are explicitly requested by the user.
- Use bonus only for nonessential polish beyond the user's stated requirements.
- Use penalty only for criteria that explicitly describe an undesirable error or violation.
""".strip()


WRITINGBENCH_EDGE_SYSTEM_PROMPT = """
You are annotating directed dependencies among WritingBench rubric criteria for GEAR.

Task:
- For each candidate directed pair, decide exactly one label.

Allowed labels:
- no_edge
- weak_prerequisite
- strong_prerequisite
- trigger

Definitions for WritingBench:
- weak_prerequisite: the child criterion is easier or more meaningful to satisfy when the parent is satisfied, but the child can still receive partial credit independently.
- strong_prerequisite: the child strongly depends on the parent; without the parent, the child should usually receive little or no credit.
- trigger: the parent activates a conditional child bonus or penalty.
- no_edge: the two writing criteria are independent dimensions.

Writing-specific guidance:
- Content correctness, task fulfillment, and required source/material coverage can be prerequisites for deeper analysis, persuasiveness, or domain-specific quality.
- Format, style, and length constraints are often independent unless one criterion explicitly builds on another.
- Do not add edges just because two criteria are both important.
- Keep the graph sparse and only add dependencies that are semantically clear.

Return JSON only with this schema:
{
  "edges": [
    {"parent": "r1", "child": "r2", "type": "weak_prerequisite"}
  ]
}

Rules:
- Use only the provided candidate pairs.
- Return exactly one decision for each candidate pair.
- Use no_edge when the relationship is only topical overlap, not a dependency.
""".strip()


PLAWBENCH_NODE_SYSTEM_PROMPT = """
You are annotating PLawBench legal-practice rubrics for structured reward aggregation.

Task:
- Classify each legal-evaluation rubric into exactly one node_type.

Allowed node_type values:
- foundation
- bonus
- penalty
- trigger

Definitions for PLawBench:
- foundation: a core legal-answer criterion such as conclusion, fact summary, legal reasoning, or legal authorities.
- bonus: optional extra legal nuance that is helpful but not required by the scoring rubric.
- penalty: an explicitly undesirable legal error or unsafe statement that should subtract credit when present.
- trigger: a gating condition whose main role is to activate another criterion. Use rarely.

Return JSON only with this schema:
{
  "nodes": [
    {"id": "r1", "node_type": "foundation"}
  ]
}

Rules:
- Use only the provided rubric ids.
- Every rubric id must appear exactly once.
- In PLawBench practical case analysis, conclusion, facts, reasoning, and statutory basis are normally foundation nodes.
- Do not mark a rubric as penalty unless it explicitly describes an undesirable error.
""".strip()


PLAWBENCH_EDGE_SYSTEM_PROMPT = """
You are annotating directed dependencies among PLawBench legal-practice rubrics for GEAR.

Task:
- For each candidate directed pair, decide exactly one label.

Allowed labels:
- no_edge
- weak_prerequisite
- strong_prerequisite
- trigger

Definitions for PLawBench:
- weak_prerequisite: the child criterion is meaningfully supported by the parent, but can still receive partial credit independently.
- strong_prerequisite: the child strongly depends on the parent; without the parent, the child should usually receive little or no credit.
- trigger: the parent activates a conditional child criterion.
- no_edge: the criteria are independent.

Legal-reasoning guidance:
- Fact summary is often a prerequisite for legal reasoning.
- Statutory/legal-authority identification often supports legal reasoning.
- Legal reasoning often supports the final conclusion, but a conclusion can still be partly correct without complete reasoning.
- Do not create reverse edges from conclusion to facts, reasoning, or statutes.
- Keep the graph sparse and avoid redundant edges.

Return JSON only with this schema:
{
  "edges": [
    {"parent": "r1", "child": "r2", "type": "weak_prerequisite"}
  ]
}

Rules:
- Use only the provided candidate pairs.
- Return exactly one decision for each candidate pair.
- Use no_edge when the pair is only part of the same answer structure but not a dependency.
""".strip()


PROFILE_PROMPTS = {
    "healthbench": {
        "node": NODE_SYSTEM_PROMPT,
        "edge": EDGE_SYSTEM_PROMPT,
    },
    "writingbench": {
        "node": WRITINGBENCH_NODE_SYSTEM_PROMPT,
        "edge": WRITINGBENCH_EDGE_SYSTEM_PROMPT,
    },
    "plawbench": {
        "node": PLAWBENCH_NODE_SYSTEM_PROMPT,
        "edge": PLAWBENCH_EDGE_SYSTEM_PROMPT,
    },
}


def _default_model_name() -> str:
    return os.getenv("ANNOTATION_MODEL") or os.getenv("VLLM_MODEL") or "Qwen2.5-7B-Instruct"


def _default_node_type(points: float) -> str:
    return "penalty" if points < 0 else "foundation"


def _parse_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _to_python(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {key: _to_python(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_python(item) for item in value]
    return value


def _as_list(value: Any) -> List[Any]:
    value = _to_python(value)
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> Dict[str, Any]:
    value = _to_python(value)
    return value if isinstance(value, dict) else {}


def _normalize_prompt(prompt: Any) -> List[Dict[str, str]]:
    prompt = _to_python(prompt)
    if not isinstance(prompt, list):
        return []

    normalized = []
    for message in prompt:
        if not isinstance(message, dict):
            continue
        normalized.append(
            {
                "role": str(message.get("role", "user")),
                "content": str(message.get("content", "")),
            }
        )
    return normalized


def _normalize_rubrics(rubrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rubrics = _as_list(rubrics)
    normalized = []
    for rubric_idx, rubric in enumerate(rubrics):
        rubric = _as_dict(rubric)
        rubric.setdefault("id", f"r{rubric_idx + 1}")
        tags = _as_dict(rubric.get("tags", {}))
        rubric["tags"] = tags if isinstance(tags, dict) else {}
        normalized.append(rubric)
    return normalized


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _cycle_would_form(adjacency: Dict[str, List[str]], parent: str, child: str) -> bool:
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


def _sanitize_nodes(raw_nodes: Dict[str, Any], rubrics: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    raw_nodes = _as_dict(raw_nodes)
    valid_ids = {rubric["id"] for rubric in rubrics}
    node_type_by_id = {}
    for item in _as_list(raw_nodes.get("nodes", [])):
        if not isinstance(item, dict):
            continue
        rubric_id = item.get("id")
        node_type = item.get("node_type")
        if rubric_id in valid_ids and node_type in ALLOWED_NODE_TYPES:
            node_type_by_id[rubric_id] = node_type

    return [
        {
            "id": rubric["id"],
            "node_type": node_type_by_id.get(rubric["id"], _default_node_type(rubric["points"])),
        }
        for rubric in rubrics
    ]


def _node_annotation_stats(raw_nodes: Dict[str, Any], rubrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    raw_nodes = _as_dict(raw_nodes)
    valid_ids = {rubric["id"] for rubric in rubrics}
    judged_ids = set()
    for item in _as_list(raw_nodes.get("nodes", [])):
        if not isinstance(item, dict):
            continue
        rubric_id = item.get("id")
        node_type = item.get("node_type")
        if rubric_id in valid_ids and node_type in ALLOWED_NODE_TYPES:
            judged_ids.add(rubric_id)

    total = len(rubrics)
    judged = len(judged_ids)
    fallback = max(0, total - judged)
    if total == 0:
        parse_status = "not_applicable"
    elif judged == 0:
        parse_status = "failed"
    elif judged < total:
        parse_status = "partial"
    else:
        parse_status = "full"

    return {
        "num_node_rubrics": total,
        "num_node_judged": judged,
        "num_node_fallback": fallback,
        "node_parse_status": parse_status,
    }


def _candidate_pairs(rubrics: List[Dict[str, Any]], node_type_by_id: Dict[str, str]) -> List[Dict[str, Any]]:
    pairs = []
    for parent in rubrics:
        for child in rubrics:
            if parent["id"] == child["id"]:
                continue
            pair_key = (node_type_by_id[parent["id"]], node_type_by_id[child["id"]])
            allowed_edge_types = sorted(ALLOWED_TYPE_PAIR_TO_EDGE_TYPES.get(pair_key, set()))
            if not allowed_edge_types:
                continue
            pairs.append(
                {
                    "parent": parent["id"],
                    "child": child["id"],
                    "parent_node_type": node_type_by_id[parent["id"]],
                    "child_node_type": node_type_by_id[child["id"]],
                    "parent_criterion": parent["criterion"],
                    "child_criterion": child["criterion"],
                    "allowed_edge_types": allowed_edge_types,
                }
            )
    return pairs


def _sanitize_edges(
    raw_edges: Dict[str, Any],
    candidate_pairs: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    raw_edges = _as_dict(raw_edges)
    candidate_map = {(pair["parent"], pair["child"]): pair for pair in candidate_pairs}
    candidate_order = {(pair["parent"], pair["child"]): idx for idx, pair in enumerate(candidate_pairs)}
    adjacency: Dict[str, List[str]] = defaultdict(list)
    best_edge_by_pair: Dict[Tuple[str, str], Dict[str, str]] = {}

    for item in _as_list(raw_edges.get("edges", [])):
        if not isinstance(item, dict):
            continue
        parent = item.get("parent")
        child = item.get("child")
        edge_type = item.get("type")
        if (parent, child) not in candidate_map:
            continue
        if edge_type not in ALLOWED_EDGE_LABELS or edge_type == "no_edge":
            continue
        if edge_type not in candidate_map[(parent, child)]["allowed_edge_types"]:
            continue
        pair_key = (parent, child)
        current_best = best_edge_by_pair.get(pair_key)
        if current_best is not None:
            current_priority = EDGE_TYPE_PRIORITY.get(current_best["type"], len(EDGE_TYPE_PRIORITY))
            new_priority = EDGE_TYPE_PRIORITY.get(edge_type, len(EDGE_TYPE_PRIORITY))
            if new_priority >= current_priority:
                continue
        best_edge_by_pair[pair_key] = {"parent": parent, "child": child, "type": edge_type}

    edges = []
    for parent, child in sorted(best_edge_by_pair, key=lambda pair: candidate_order[pair]):
        edge = best_edge_by_pair[(parent, child)]
        if _cycle_would_form(adjacency, parent, child):
            continue
        adjacency[parent].append(child)
        edges.append(edge)

    return edges


def _truncate_text(text: str, head_chars: int, tail_chars: int) -> str:
    if head_chars <= 0 or tail_chars <= 0:
        return text
    if len(text) <= head_chars + tail_chars:
        return text
    omitted = len(text) - head_chars - tail_chars
    return (
        text[:head_chars]
        + f"\n\n[... omitted {omitted} characters for graph annotation ...]\n\n"
        + text[-tail_chars:]
    )


def _normalize_prompt_for_annotation(
    prompt: Any,
    profile: str,
    prompt_head_chars: int,
    prompt_tail_chars: int,
) -> List[Dict[str, str]]:
    normalized = _normalize_prompt(prompt)
    if profile != "writingbench":
        return normalized

    truncated = []
    for message in normalized:
        truncated.append(
            {
                **message,
                "content": _truncate_text(str(message.get("content", "")), prompt_head_chars, prompt_tail_chars),
            }
        )
    return truncated


def _build_node_prompt(
    prompt: Any,
    rubrics: List[Dict[str, Any]],
    profile: str,
    source_meta: Optional[Dict[str, Any]] = None,
    prompt_head_chars: int = 4000,
    prompt_tail_chars: int = 1000,
) -> str:
    conversation = json.dumps(
        _normalize_prompt_for_annotation(prompt, profile, prompt_head_chars, prompt_tail_chars),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    metadata = json.dumps(_as_dict(source_meta), ensure_ascii=False, separators=(",", ":"))
    rubric_block = json.dumps(
        [
            {"id": rubric["id"], "criterion": rubric["criterion"], "points": rubric["points"]}
            for rubric in rubrics
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if profile == "healthbench" and not _as_dict(source_meta):
        return f"""Conversation:
{conversation}

Rubrics:
{rubric_block}

Classify each rubric into one node_type.
Return JSON only."""

    return f"""Profile:
{profile}

Source metadata:
{metadata}

Conversation:
{conversation}

Rubrics:
{rubric_block}

Classify each rubric into one node_type.
Return JSON only."""


def _build_edge_prompt(
    prompt: Any,
    candidate_pairs: List[Dict[str, Any]],
    profile: str,
    source_meta: Optional[Dict[str, Any]] = None,
    prompt_head_chars: int = 4000,
    prompt_tail_chars: int = 1000,
) -> str:
    conversation = json.dumps(
        _normalize_prompt_for_annotation(prompt, profile, prompt_head_chars, prompt_tail_chars),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    metadata = json.dumps(_as_dict(source_meta), ensure_ascii=False, separators=(",", ":"))
    pair_block = json.dumps(candidate_pairs, ensure_ascii=False, separators=(",", ":"))
    if profile == "healthbench" and not _as_dict(source_meta):
        return f"""Conversation:
{conversation}

Candidate directed rubric pairs:
{pair_block}

For each candidate pair, return exactly one label in {{no_edge, weak_prerequisite, strong_prerequisite, trigger}}.
Return JSON only."""

    return f"""Profile:
{profile}

Source metadata:
{metadata}

Conversation:
{conversation}

Candidate directed rubric pairs:
{pair_block}

For each candidate pair, return exactly one label in {{no_edge, weak_prerequisite, strong_prerequisite, trigger}}.
Return JSON only."""


def _node_response_schema(rubrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    rubric_ids = [str(rubric["id"]) for rubric in rubrics]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string", "enum": rubric_ids},
                        "node_type": {"type": "string", "enum": sorted(ALLOWED_NODE_TYPES)},
                    },
                    "required": ["id", "node_type"],
                },
            }
        },
        "required": ["nodes"],
    }


def _edge_response_schema(candidate_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "parent": {"type": "string", "enum": sorted({str(pair["parent"]) for pair in candidate_pairs})},
                        "child": {"type": "string", "enum": sorted({str(pair["child"]) for pair in candidate_pairs})},
                        "type": {"type": "string", "enum": sorted(ALLOWED_EDGE_LABELS)},
                    },
                    "required": ["parent", "child", "type"],
                },
            }
        },
        "required": ["edges"],
    }


def _constraint_modes_to_try(requested_mode: str, response_schema: Optional[Dict[str, Any]]) -> List[str]:
    if requested_mode != "auto":
        if requested_mode == "json_schema" and response_schema is None:
            return ["json_object"]
        if requested_mode == "structured_outputs" and response_schema is None:
            return ["json_object"]
        return [requested_mode]

    modes: List[str] = []
    if response_schema is not None:
        modes.extend(["json_schema", "json_object"])
    else:
        modes.append("json_object")

    deduped: List[str] = []
    seen = set()
    for mode in modes:
        if mode not in seen:
            deduped.append(mode)
            seen.add(mode)
    return deduped


def _apply_json_constraint(
    payload: Dict[str, Any],
    mode: str,
    schema_name: str,
    schema: Optional[Dict[str, Any]],
) -> None:
    if mode == "off":
        return
    if mode == "json_object" or schema is None:
        payload["response_format"] = {"type": "json_object"}
        return
    if mode == "structured_outputs":
        payload["structured_outputs"] = {"json": schema}
        return
    payload["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": schema,
        },
    }


def _preview_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    normalized = text.replace("\n", "\\n")
    return normalized[:max_chars]


class OpenAIAPIClient:
    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        response_preview_chars: int = 0,
    ):
        from openai import OpenAI

        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        self.client = OpenAI(**kwargs)
        self.model = model
        self.max_retries = max_retries
        self.response_preview_chars = response_preview_chars

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        response_schema_name: str = "response",
        response_schema: Optional[Dict[str, Any]] = None,
        json_constraint_mode: str = "auto",
    ) -> Dict[str, Any]:
        last_error = None
        last_mode = None
        request_max_retries = self.max_retries if max_retries is None else max_retries
        modes_to_try = _constraint_modes_to_try(json_constraint_mode, response_schema)

        for mode in modes_to_try:
            for trial in range(request_max_retries):
                try:
                    kwargs: Dict[str, Any] = {}
                    if max_tokens is not None:
                        kwargs["max_tokens"] = max_tokens
                    if timeout is not None:
                        kwargs["timeout"] = timeout
                    payload_kwargs: Dict[str, Any] = {}
                    _apply_json_constraint(
                        payload_kwargs,
                        mode,
                        response_schema_name,
                        response_schema,
                    )
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.0,
                        **payload_kwargs,
                        **kwargs,
                    )
                    content = response.choices[0].message.content or "{}"
                    if self.response_preview_chars > 0:
                        print(
                            f"[response_preview] provider=openai mode={mode} chars={len(content)} "
                            f"preview={_preview_text(content, self.response_preview_chars)}",
                            flush=True,
                        )
                    return _parse_json_object(content)
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    last_mode = mode
                    time.sleep(min(2**trial, 8))

        raise RuntimeError(
            f"API generation failed after retries: last_mode={last_mode} error={last_error}"
        ) from last_error


class VLLMClient:
    def __init__(
        self,
        model: str,
        base_urls: List[str],
        max_retries: int = 5,
        timeout: int = 120,
        load_refresh_interval_sec: float = 2.0,
        response_preview_chars: int = 0,
    ):
        self.model = model
        self.base_urls = [url.strip() for url in base_urls if url.strip()]
        if not self.base_urls:
            raise ValueError("VLLM provider requires at least one base URL")
        self.max_retries = max_retries
        self.timeout = timeout
        self.load_refresh_interval_sec = load_refresh_interval_sec
        self.response_preview_chars = response_preview_chars
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer dummy",
        }
        self._lock = threading.Lock()
        self._thread_local = threading.local()
        self._url_loads = {
            url: {"running": 0, "waiting": 0, "total": 0, "available": True}
            for url in self.base_urls
        }
        self._virtual_loads = {url: 0 for url in self.base_urls}
        self._last_refresh_ts = 0.0
        self._refresh_loads(force=True)

    def _session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(self.headers)
            self._thread_local.session = session
        return session

    def _metrics_url(self, base_url: str) -> str:
        normalized = base_url[:-3] if base_url.endswith("/v1") else base_url.rstrip("/")
        return f"{normalized}/metrics"

    def _parse_metric_value(self, metrics_text: str, metric_name: str) -> int:
        pattern = rf"^{re.escape(metric_name)}(?:\{{[^}}]*\}})?\s+([0-9.]+)"
        matches = re.findall(pattern, metrics_text, flags=re.MULTILINE)
        if not matches:
            return 0
        return int(float(matches[0]))

    def _get_url_load(self, base_url: str) -> Dict[str, Any]:
        try:
            response = self._session().get(self._metrics_url(base_url), timeout=5)
            if response.status_code != 200:
                return {"running": 0, "waiting": 0, "total": 0, "available": True}
            metrics_text = response.text
            running = self._parse_metric_value(metrics_text, "vllm:num_requests_running")
            waiting = self._parse_metric_value(metrics_text, "vllm:num_requests_waiting")
            return {
                "running": running,
                "waiting": waiting,
                "total": running + waiting,
                "available": True,
            }
        except Exception:  # noqa: BLE001
            return {"running": 0, "waiting": 0, "total": 0, "available": True}

    def _refresh_loads(self, force: bool = False) -> None:
        now = time.time()
        with self._lock:
            if not force and now - self._last_refresh_ts < self.load_refresh_interval_sec:
                return
            current_urls = list(self.base_urls)
        fresh_loads = {url: self._get_url_load(url) for url in current_urls}
        with self._lock:
            for url, load_info in fresh_loads.items():
                self._url_loads[url] = load_info
            self._last_refresh_ts = now

    def _acquire_url(self) -> str:
        self._refresh_loads()
        with self._lock:
            available_urls = [url for url in self.base_urls if self._url_loads[url].get("available", True)]
            if not available_urls:
                available_urls = list(self.base_urls)
            selected = min(
                available_urls,
                key=lambda url: self._url_loads[url]["total"] + self._virtual_loads[url],
            )
            self._virtual_loads[selected] += 1
            return selected

    def _release_url(self, base_url: str) -> None:
        with self._lock:
            self._virtual_loads[base_url] = max(0, self._virtual_loads[base_url] - 1)

    def load_status(self) -> str:
        self._refresh_loads(force=True)
        with self._lock:
            parts = []
            for url in self.base_urls:
                load_info = self._url_loads.get(url, {})
                parts.append(
                    f"{url}:running={int(load_info.get('running', 0))},"
                    f"waiting={int(load_info.get('waiting', 0))},"
                    f"virtual={int(self._virtual_loads.get(url, 0))}"
                )
        return "vllm_load=[" + "; ".join(parts) + "]"

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        response_schema_name: str = "response",
        response_schema: Optional[Dict[str, Any]] = None,
        json_constraint_mode: str = "auto",
    ) -> Dict[str, Any]:
        last_error = None
        last_mode = None
        request_timeout = self.timeout if timeout is None else timeout
        request_max_retries = self.max_retries if max_retries is None else max_retries
        modes_to_try = _constraint_modes_to_try(json_constraint_mode, response_schema)

        for mode in modes_to_try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.0,
                "top_p": 0.9,
                "top_k": 20,
                "max_tokens": max_tokens or 2048,
            }
            _apply_json_constraint(
                payload,
                mode,
                response_schema_name,
                response_schema,
            )

            for trial in range(request_max_retries):
                base_url = self._acquire_url()
                try:
                    response = self._session().post(
                        f"{base_url}/chat/completions",
                        json=payload,
                        timeout=request_timeout,
                    )
                    response.raise_for_status()
                    response_data = response.json()
                    content = response_data["choices"][0]["message"]["content"] or "{}"
                    if self.response_preview_chars > 0:
                        print(
                            f"[response_preview] provider=vllm mode={mode} chars={len(content)} "
                            f"preview={_preview_text(content, self.response_preview_chars)}",
                            flush=True,
                        )
                    return _parse_json_object(content)
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    last_mode = mode
                    time.sleep(min(2**trial, 8))
                finally:
                    self._release_url(base_url)

        raise RuntimeError(
            f"VLLM generation failed after retries: last_mode={last_mode} error={last_error}"
        ) from last_error


def _make_client(args: argparse.Namespace):
    if args.provider == "openai":
        return OpenAIAPIClient(
            model=args.model,
            base_url=args.base_url or os.getenv("OPENAI_BASE_URL"),
            api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
            max_retries=args.max_retries,
            response_preview_chars=args.response_preview_chars,
        )

    base_urls_arg = args.base_urls or os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
    base_urls = [url.strip() for url in base_urls_arg.split(",") if url.strip()]
    return VLLMClient(
        model=args.model,
        base_urls=base_urls,
        max_retries=args.max_retries,
        timeout=args.timeout,
        load_refresh_interval_sec=args.load_refresh_interval_sec,
        response_preview_chars=args.response_preview_chars,
    )


def _run_tasks_concurrently(
    tasks: List[Dict[str, Any]],
    worker_fn,
    max_workers: int,
    stage_name: str,
    status_fn=None,
) -> List[Dict[str, Any]]:
    if not tasks:
        return []

    results = []
    completed = 0
    total = len(tasks)
    start_ts = time.time()
    last_progress_print = time.time()
    active = 0
    active_lock = threading.Lock()

    def wrapped_worker(task: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal active
        with active_lock:
            active += 1
        try:
            return worker_fn(task)
        finally:
            with active_lock:
                active -= 1

    print(
        f"[{stage_name}] starting {total} tasks with max_workers={max_workers} "
        f"heartbeat_sec={PROGRESS_HEARTBEAT_SEC}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = {executor.submit(wrapped_worker, task): task for task in tasks}
        print(f"[{stage_name}] submitted {len(pending)} tasks", flush=True)

        while pending:
            done, not_done = wait(pending, timeout=PROGRESS_HEARTBEAT_SEC, return_when=FIRST_COMPLETED)
            elapsed = time.time() - start_ts
            with active_lock:
                active_now = active
            queued_estimate = max(0, len(not_done) - active_now)
            status = f"elapsed={elapsed:.1f}s active={active_now} queued~={queued_estimate}"
            if status_fn is not None:
                try:
                    status = f"{status} {status_fn()}"
                except Exception as exc:  # noqa: BLE001
                    status = f"{status} status_error={exc}"

            if not done:
                print(
                    f"[{stage_name}] heartbeat completed={completed}/{total} "
                    f"pending={len(not_done)} {status}",
                    flush=True,
                )
                continue

            for future in done:
                task = pending[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    row_idx = task.get("row_idx", "unknown")
                    print(
                        f"[{stage_name}] task failed row_idx={row_idx} error={repr(exc)}",
                        flush=True,
                    )
                    raise RuntimeError(f"[{stage_name}] task failed for row_idx={row_idx}: {exc}") from exc
                completed += 1

            prev_pending = pending
            pending = {future: prev_pending[future] for future in not_done}

            now = time.time()
            if (
                completed == total
                or completed % max(1, PROGRESS_PRINT_EVERY) == 0
                or now - last_progress_print >= PROGRESS_HEARTBEAT_SEC
            ):
                with active_lock:
                    active_now = active
                queued_estimate = max(0, len(pending) - active_now)
                print(
                    f"[{stage_name}] completed {completed}/{total} pending={len(pending)} "
                    f"elapsed={now - start_ts:.1f}s active={active_now} queued~={queued_estimate}",
                    flush=True,
                )
                last_progress_print = now
    return results


def _node_task_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    raw_nodes = task["client"].generate_json(
        system_prompt=task["node_system_prompt"],
        user_prompt=_build_node_prompt(
            task["prompt"],
            task["rubrics"],
            profile=task["profile"],
            source_meta=task.get("source_meta"),
            prompt_head_chars=task.get("prompt_head_chars", 4000),
            prompt_tail_chars=task.get("prompt_tail_chars", 1000),
        ),
        max_tokens=task.get("node_max_tokens"),
        timeout=task.get("node_timeout"),
        max_retries=task.get("node_max_retries"),
        response_schema_name="gear_node_annotation",
        response_schema=_node_response_schema(task["rubrics"]),
        json_constraint_mode=task.get("json_constraint_mode", "auto"),
    )
    nodes = _sanitize_nodes(raw_nodes, task["rubrics"])
    stats = _node_annotation_stats(raw_nodes, task["rubrics"])
    return {"row_idx": task["row_idx"], "nodes": nodes, "stats": stats}


def _generate_edges_with_split(
    task: Dict[str, Any],
    candidate_pairs: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, str]], int]:
    try:
        raw_edges = task["client"].generate_json(
            system_prompt=task["edge_system_prompt"],
            user_prompt=_build_edge_prompt(
                task["prompt"],
                candidate_pairs,
                profile=task["profile"],
                source_meta=task.get("source_meta"),
                prompt_head_chars=task.get("prompt_head_chars", 4000),
                prompt_tail_chars=task.get("prompt_tail_chars", 1000),
            ),
            max_tokens=task.get("edge_max_tokens"),
            timeout=task.get("edge_timeout"),
            max_retries=task.get("edge_max_retries"),
            response_schema_name="gear_edge_annotation",
            response_schema=_edge_response_schema(candidate_pairs),
            json_constraint_mode=task.get("json_constraint_mode", "auto"),
        )
        return _sanitize_edges(raw_edges, candidate_pairs), 0
    except Exception as exc:  # noqa: BLE001
        min_batch_size = max(1, int(task.get("edge_split_min_batch_size", 1)))
        if len(candidate_pairs) <= min_batch_size:
            print(
                "[edge_type] warning "
                f"row_idx={task.get('row_idx', 'unknown')} "
                f"failed_candidates={len(candidate_pairs)} "
                f"error={exc}",
                flush=True,
            )
            return [], len(candidate_pairs)

        midpoint = max(1, len(candidate_pairs) // 2)
        left_edges, left_failed = _generate_edges_with_split(task, candidate_pairs[:midpoint])
        right_edges, right_failed = _generate_edges_with_split(task, candidate_pairs[midpoint:])
        return left_edges + right_edges, left_failed + right_failed


def _edge_task_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    edges, failed_candidates = _generate_edges_with_split(task, task["candidate_pairs"])
    return {
        "row_idx": task["row_idx"],
        "edges": edges,
        "num_edge_failed_candidates": failed_candidates,
    }


def annotate_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = [_to_python(row.to_dict()) for _, row in df.iterrows()]
    profile = getattr(args, "profile", "healthbench")
    prompt_head_chars = getattr(args, "prompt_head_chars", 4000)
    prompt_tail_chars = getattr(args, "prompt_tail_chars", 1000)
    profile_prompts = PROFILE_PROMPTS[profile]
    client = _make_client(args)
    client_status_fn = getattr(client, "load_status", None)
    provider_url_count = len(getattr(client, "base_urls", ["api"]))
    max_workers = args.max_workers if args.max_workers > 0 else max(1, args.max_workers_per_url * provider_url_count)
    node_max_workers = (
        args.node_max_workers
        if args.node_max_workers > 0
        else max(1, args.node_max_workers_per_url * provider_url_count)
    )
    edge_max_workers = (
        args.edge_max_workers
        if args.edge_max_workers > 0
        else max(1, args.edge_max_workers_per_url * provider_url_count)
    )
    node_timeout = args.node_timeout if args.node_timeout > 0 else min(args.timeout, 90)
    edge_timeout = args.edge_timeout if args.edge_timeout > 0 else min(args.timeout, 120)
    node_max_retries = args.node_max_retries if args.node_max_retries > 0 else 1
    edge_max_retries = args.edge_max_retries if args.edge_max_retries > 0 else 1
    print(
        "[config] "
        f"profile={profile} "
        f"provider_urls={provider_url_count} "
        f"max_workers={max_workers} "
        f"node_max_workers={node_max_workers} "
        f"edge_max_workers={edge_max_workers} "
        f"json_constraint_mode={args.json_constraint_mode} "
        f"node_timeout={node_timeout} "
        f"node_max_retries={node_max_retries} "
        f"node_max_tokens={args.node_max_tokens} "
        f"edge_batch_size={args.edge_batch_size} "
        f"edge_timeout={edge_timeout} "
        f"edge_max_retries={edge_max_retries} "
        f"edge_max_tokens={args.edge_max_tokens} "
        f"prompt_head_chars={prompt_head_chars} "
        f"prompt_tail_chars={prompt_tail_chars} "
        f"heartbeat_sec={PROGRESS_HEARTBEAT_SEC}",
        flush=True,
    )

    normalized_rows = []
    node_tasks = []
    candidate_pairs_by_row: Dict[int, List[Dict[str, Any]]] = {}
    for row_idx, row in enumerate(rows):
        reward_model = _as_dict(row.get("reward_model", {}))
        extra_info = _as_dict(row.get("extra_info", {}))
        rubrics = _normalize_rubrics(reward_model.get("rubrics", []))
        prompt = _normalize_prompt(row.get("prompt"))
        reward_model["rubrics"] = rubrics
        row["reward_model"] = reward_model
        normalized_rows.append(row)
        if rubrics:
            node_tasks.append(
                {
                    "row_idx": row_idx,
                    "client": client,
                    "profile": profile,
                    "node_system_prompt": profile_prompts["node"],
                    "prompt": prompt,
                    "source_meta": _as_dict(extra_info.get("source_meta", {})),
                    "rubrics": rubrics,
                    "node_timeout": node_timeout,
                    "node_max_retries": node_max_retries,
                    "node_max_tokens": args.node_max_tokens,
                    "prompt_head_chars": prompt_head_chars,
                    "prompt_tail_chars": prompt_tail_chars,
                    "json_constraint_mode": args.json_constraint_mode,
                }
            )

    node_results = _run_tasks_concurrently(
        tasks=node_tasks,
        worker_fn=_node_task_worker,
        max_workers=node_max_workers,
        stage_name="node_type",
        status_fn=client_status_fn,
    )
    node_types_by_row = {}
    node_stats_by_row = {}
    for result in node_results:
        node_types_by_row[result["row_idx"]] = {node["id"]: node["node_type"] for node in result["nodes"]}
        node_stats_by_row[result["row_idx"]] = result["stats"]

    if node_results:
        node_fallback_vals = [float(result["stats"]["num_node_fallback"]) for result in node_results]
        node_judged_vals = [float(result["stats"]["num_node_judged"]) for result in node_results]
        node_parse_statuses = [str(result["stats"]["node_parse_status"]) for result in node_results]
        node_parse_full_ratio = sum(1.0 for status in node_parse_statuses if status == "full") / len(node_parse_statuses)
        node_parse_partial_ratio = sum(1.0 for status in node_parse_statuses if status == "partial") / len(node_parse_statuses)
        node_parse_failed_ratio = sum(1.0 for status in node_parse_statuses if status == "failed") / len(node_parse_statuses)
        print(
            "[node_type_summary] "
            f"rows={len(node_results)} "
            f"avg_node_judged={sum(node_judged_vals) / len(node_judged_vals):.3f} "
            f"avg_node_fallback={sum(node_fallback_vals) / len(node_fallback_vals):.3f} "
            f"node_parse_full_ratio={node_parse_full_ratio:.3f} "
            f"node_parse_partial_ratio={node_parse_partial_ratio:.3f} "
            f"node_parse_failed_ratio={node_parse_failed_ratio:.3f}",
            flush=True,
        )

    edge_tasks = []
    for row_idx, row in enumerate(normalized_rows):
        reward_model = _as_dict(row.get("reward_model", {}))
        extra_info = _as_dict(row.get("extra_info", {}))
        rubrics = _normalize_rubrics(reward_model.get("rubrics", []))
        prompt = _normalize_prompt(row.get("prompt"))
        node_type_by_id = node_types_by_row.get(
            row_idx,
            {rubric["id"]: _default_node_type(rubric["points"]) for rubric in rubrics},
        )
        candidate_pairs = _candidate_pairs(rubrics, node_type_by_id)
        candidate_pairs_by_row[row_idx] = candidate_pairs
        if not candidate_pairs:
            continue

        for start in range(0, len(candidate_pairs), args.edge_batch_size):
            edge_tasks.append(
                {
                    "row_idx": row_idx,
                    "client": client,
                    "profile": profile,
                    "edge_system_prompt": profile_prompts["edge"],
                    "prompt": prompt,
                    "source_meta": _as_dict(extra_info.get("source_meta", {})),
                    "candidate_pairs": candidate_pairs[start : start + args.edge_batch_size],
                    "edge_timeout": edge_timeout,
                    "edge_max_retries": edge_max_retries,
                    "edge_max_tokens": args.edge_max_tokens,
                    "prompt_head_chars": prompt_head_chars,
                    "prompt_tail_chars": prompt_tail_chars,
                    "edge_split_min_batch_size": args.edge_split_min_batch_size,
                    "json_constraint_mode": args.json_constraint_mode,
                }
            )

    edge_results = _run_tasks_concurrently(
        tasks=edge_tasks,
        worker_fn=_edge_task_worker,
        max_workers=edge_max_workers,
        stage_name="edge_type",
        status_fn=client_status_fn,
    )
    edges_by_row: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    edge_failed_candidates_by_row: Dict[int, int] = defaultdict(int)
    for result in edge_results:
        edges_by_row[result["row_idx"]].extend(result["edges"])
        edge_failed_candidates_by_row[result["row_idx"]] += int(result.get("num_edge_failed_candidates", 0))

    annotated_rows = []
    for row_idx, row in enumerate(normalized_rows):
        reward_model = _as_dict(row.get("reward_model", {}))
        rubrics = _normalize_rubrics(reward_model.get("rubrics", []))
        node_stats = node_stats_by_row.get(
            row_idx,
            {
                "num_node_rubrics": len(rubrics),
                "num_node_judged": 0,
                "num_node_fallback": len(rubrics),
                "node_parse_status": "failed" if rubrics else "not_applicable",
            },
        )
        node_type_by_id = node_types_by_row.get(
            row_idx,
            {rubric["id"]: _default_node_type(rubric["points"]) for rubric in rubrics},
        )
        nodes = [{"id": rubric["id"], "node_type": node_type_by_id[rubric["id"]]} for rubric in rubrics]
        candidate_pairs = candidate_pairs_by_row.get(row_idx)
        if candidate_pairs is None:
            candidate_pairs = _candidate_pairs(rubrics, node_type_by_id)
        edges = _sanitize_edges({"edges": edges_by_row.get(row_idx, [])}, candidate_pairs)

        for rubric in rubrics:
            rubric["tags"]["node_type"] = node_type_by_id[rubric["id"]]

        reward_model["rubrics"] = rubrics
        reward_model["graph"] = {"nodes": nodes, "edges": edges}

        extra_info = _as_dict(row.get("extra_info", {}))
        extra_info["reward_model"] = reward_model
        extra_info["gear_annotation_stats"] = {
            **node_stats,
            "annotation_profile": profile,
            "num_edge_candidates": len(candidate_pairs),
            "num_edge_failed_candidates": edge_failed_candidates_by_row.get(row_idx, 0),
            "num_graph_edges": len(edges),
        }

        row["reward_model"] = reward_model
        row["extra_info"] = extra_info
        row["gear_annotation_stats"] = extra_info["gear_annotation_stats"]
        annotated_rows.append(row)

    return pd.DataFrame(annotated_rows)


def _build_annotation_summary(df: pd.DataFrame) -> Dict[str, Any]:
    rows = [_to_python(row.to_dict()) for _, row in df.iterrows()]

    node_parse_status_counter: Counter[str] = Counter()
    node_type_counter: Counter[str] = Counter()
    edge_type_counter: Counter[str] = Counter()

    num_node_rubrics_vals: List[float] = []
    num_node_judged_vals: List[float] = []
    num_node_fallback_vals: List[float] = []
    num_edge_candidates_vals: List[float] = []
    num_edge_failed_candidates_vals: List[float] = []
    num_graph_edges_vals: List[float] = []
    graph_nonempty_flags: List[float] = []

    for row in rows:
        stats = _as_dict(row.get("gear_annotation_stats", {}))
        reward_model = _as_dict(row.get("reward_model", {}))
        rubrics = _normalize_rubrics(reward_model.get("rubrics", []))
        graph = _as_dict(reward_model.get("graph", {}))
        nodes = _as_list(graph.get("nodes", []))
        edges = _as_list(graph.get("edges", []))

        node_parse_status = str(stats.get("node_parse_status", "not_applicable"))
        node_parse_status_counter[node_parse_status] += 1

        num_node_rubrics_vals.append(float(stats.get("num_node_rubrics", len(rubrics))))
        num_node_judged_vals.append(float(stats.get("num_node_judged", 0)))
        num_node_fallback_vals.append(float(stats.get("num_node_fallback", len(rubrics))))
        num_edge_candidates_vals.append(float(stats.get("num_edge_candidates", 0)))
        num_edge_failed_candidates_vals.append(float(stats.get("num_edge_failed_candidates", 0)))
        num_graph_edges_vals.append(float(stats.get("num_graph_edges", len(edges))))
        graph_nonempty_flags.append(float(len(edges) > 0))

        for node in nodes:
            if isinstance(node, dict):
                node_type_counter[str(node.get("node_type", "unknown"))] += 1
        for edge in edges:
            if isinstance(edge, dict):
                edge_type_counter[str(edge.get("type", "unknown"))] += 1

    total_rows = len(rows)
    summary = {
        "num_rows": total_rows,
        "avg_num_node_rubrics": _mean(num_node_rubrics_vals),
        "avg_num_node_judged": _mean(num_node_judged_vals),
        "avg_num_node_fallback": _mean(num_node_fallback_vals),
        "avg_num_edge_candidates": _mean(num_edge_candidates_vals),
        "avg_num_edge_failed_candidates": _mean(num_edge_failed_candidates_vals),
        "avg_num_graph_edges": _mean(num_graph_edges_vals),
        "graph_nonempty_ratio": _mean(graph_nonempty_flags),
        "node_parse_status_counts": dict(node_parse_status_counter),
        "node_parse_status_ratios": {
            key: (value / total_rows if total_rows else 0.0)
            for key, value in sorted(node_parse_status_counter.items())
        },
        "node_type_counts": dict(node_type_counter),
        "edge_type_counts": dict(edge_type_counter),
    }
    return summary


def _print_annotation_summary(summary: Dict[str, Any]) -> None:
    print(
        "[annotation_summary] "
        f"rows={summary['num_rows']} "
        f"avg_node_rubrics={summary['avg_num_node_rubrics']:.3f} "
        f"avg_node_judged={summary['avg_num_node_judged']:.3f} "
        f"avg_node_fallback={summary['avg_num_node_fallback']:.3f} "
        f"avg_edge_candidates={summary['avg_num_edge_candidates']:.3f} "
        f"avg_edge_failed_candidates={summary['avg_num_edge_failed_candidates']:.3f} "
        f"avg_graph_edges={summary['avg_num_graph_edges']:.3f} "
        f"graph_nonempty_ratio={summary['graph_nonempty_ratio']:.3f}",
        flush=True,
    )
    print(
        "[annotation_summary] "
        f"node_parse_status_ratios={json.dumps(summary['node_parse_status_ratios'], ensure_ascii=False, sort_keys=True)}",
        flush=True,
    )
    print(
        "[annotation_summary] "
        f"node_type_counts={json.dumps(summary['node_type_counts'], ensure_ascii=False, sort_keys=True)}",
        flush=True,
    )
    print(
        "[annotation_summary] "
        f"edge_type_counts={json.dumps(summary['edge_type_counts'], ensure_ascii=False, sort_keys=True)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_PROMPTS),
        default="healthbench",
        help="Benchmark-specific graph annotation prompt profile",
    )
    parser.add_argument("--provider", choices=["vllm", "openai"], default="vllm")
    parser.add_argument("--model", default=_default_model_name())
    parser.add_argument("--base-url", default=None, help="Base URL for OpenAI-compatible API")
    parser.add_argument("--base-urls", default=None, help="Comma-separated VLLM base URLs")
    parser.add_argument("--api-key", default=None, help="API key for OpenAI-compatible API")
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit")

    parser.add_argument("--edge-batch-size", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=0, help="0 means auto")
    parser.add_argument("--max-workers-per-url", type=int, default=8)
    parser.add_argument("--node-max-workers", type=int, default=0, help="0 means auto")
    parser.add_argument("--node-max-workers-per-url", type=int, default=3)
    parser.add_argument("--edge-max-workers", type=int, default=0, help="0 means auto")
    parser.add_argument("--edge-max-workers-per-url", type=int, default=4)

    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--node-timeout", type=int, default=0, help="0 means min(--timeout, 90)")
    parser.add_argument("--edge-timeout", type=int, default=0, help="0 means min(--timeout, 120)")
    parser.add_argument("--node-max-retries", type=int, default=0, help="0 means 1")
    parser.add_argument("--edge-max-retries", type=int, default=0, help="0 means 1")
    parser.add_argument("--node-max-tokens", type=int, default=128)
    parser.add_argument("--edge-max-tokens", type=int, default=256)
    parser.add_argument("--edge-split-min-batch-size", type=int, default=1)
    parser.add_argument(
        "--prompt-head-chars",
        type=int,
        default=4000,
        help="For WritingBench annotation, keep this many leading prompt characters",
    )
    parser.add_argument(
        "--prompt-tail-chars",
        type=int,
        default=1000,
        help="For WritingBench annotation, keep this many trailing prompt characters",
    )

    parser.add_argument(
        "--json-constraint-mode",
        choices=["auto", "json_schema", "json_object", "structured_outputs", "off"],
        default="auto",
        help="JSON output constraint mode for OpenAI-compatible APIs",
    )
    parser.add_argument(
        "--response-preview-chars",
        type=int,
        default=0,
        help="Print the first N chars of each model response for debugging; 0 disables it",
    )
    parser.add_argument("--load-refresh-interval-sec", type=float, default=2.0)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    df = pd.read_parquet(input_path)
    if args.limit > 0:
        df = df.head(args.limit).copy()

    annotated_df = annotate_rows(df, args)
    annotation_summary = _build_annotation_summary(annotated_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated_df.to_parquet(output_path, index=False)
    print(f"Saved {len(annotated_df)} annotated rows to {output_path}", flush=True)
    _print_annotation_summary(annotation_summary)

    summary_path = output_path.with_name(f"{output_path.stem}.annotation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(annotation_summary, f, ensure_ascii=False, indent=2)
    print(f"Saved annotation summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
