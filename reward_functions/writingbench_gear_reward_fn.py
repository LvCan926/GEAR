from __future__ import annotations

import sys
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Dict, List, Optional


_SCORED_SPEC = importlib_util.spec_from_file_location(
    "scored_gear_reward_shared",
    Path(__file__).resolve().with_name("scored_gear_reward_fn.py"),
)
_SCORED_MODULE = importlib_util.module_from_spec(_SCORED_SPEC)
assert _SCORED_SPEC.loader is not None
sys.modules[_SCORED_SPEC.name] = _SCORED_MODULE
_SCORED_SPEC.loader.exec_module(_SCORED_MODULE)

MAX_CONCURRENT_WORKERS = _SCORED_MODULE.MAX_CONCURRENT_WORKERS


def _normalize_pi_mode(pi_mode: str) -> str:
    normalized = str(pi_mode or "judge_prob").strip().lower()
    if normalized == "binary":
        raise ValueError("WritingBench does not support binary scoring; use pi_mode='judge_prob'.")
    if normalized in {"judge_prob", "prob", "normalized_score", "score", "soft"}:
        return "judge_prob"
    raise ValueError(f"Unsupported WritingBench pi_mode={pi_mode!r}; use 'judge_prob'.")


def compute_score_batched(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: List[Dict[str, Any]],
    max_workers_per_url: int = MAX_CONCURRENT_WORKERS,
    aggregation_mode: str = "dag",
    pi_mode: str = "judge_prob",
    normalization_mode: str = "positive_sum",
    inference_mode: str = "approx",
    exact_if_num_nodes_le: int = 10,
    lambda_by_edge_type: Optional[Dict[str, float]] = None,
    graph_source: str = "dataset",
    acc_mode: str = "standard",
    score_met_threshold: float = 0.7,
    final_reward_mode: str = "aggregate",
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    normalized_pi_mode = _normalize_pi_mode(pi_mode)
    return _SCORED_MODULE.compute_score_batched(
        data_sources=data_sources,
        solution_strs=solution_strs,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
        max_workers_per_url=max_workers_per_url,
        aggregation_mode=aggregation_mode,
        normalization_mode=normalization_mode,
        inference_mode=inference_mode,
        exact_if_num_nodes_le=exact_if_num_nodes_le,
        lambda_by_edge_type=lambda_by_edge_type,
        graph_source=graph_source,
        acc_mode=acc_mode,
        score_met_threshold=score_met_threshold,
        pi_mode=normalized_pi_mode,
        final_reward_mode=final_reward_mode,
        **kwargs,
    )