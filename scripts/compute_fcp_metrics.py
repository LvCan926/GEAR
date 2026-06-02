#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


METHOD_TO_SCORE_FIELD = {
    "flat": "flat_q_list",
    "hard": "hard_q_list",
    "gear": "dag_q_list",
}

DEFAULT_EDGE_TYPES = ("weak_prerequisite", "strong_prerequisite")
ALL_EDGE_TYPES = ("weak_prerequisite", "strong_prerequisite", "trigger")


@dataclass
class RowContribution:
    leakage_sum: Dict[str, float] = field(default_factory=dict)
    preservation_sum: Dict[str, float] = field(default_factory=dict)
    leakage_count: int = 0
    preservation_count: int = 0
    considered_edges: int = 0
    invalid_edges: int = 0


@dataclass
class Totals:
    leakage_sum: Dict[str, float] = field(default_factory=dict)
    preservation_sum: Dict[str, float] = field(default_factory=dict)
    leakage_count: int = 0
    preservation_count: int = 0
    considered_edges: int = 0
    invalid_edges: int = 0
    rows_total: int = 0
    rows_used: int = 0
    rows_without_edges: int = 0
    rows_missing_fields: int = 0


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value if isinstance(value, list) else []


def _float_list(value: Any) -> List[float]:
    out = []
    for item in _as_list(value):
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result):
        return default
    return result


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_no}")
            records.append(record)
    return records


def load_reward_models(path: Path) -> List[Dict[str, Any]]:
    df = pd.read_parquet(path)
    if "reward_model" not in df.columns:
        raise ValueError(f"{path} does not contain a reward_model column")
    return [_as_dict(value) for value in df["reward_model"].tolist()]


def infer_repeat_n(num_records: int, num_examples: int, repeat_n: Optional[int]) -> int:
    if repeat_n is not None:
        if repeat_n <= 0:
            raise ValueError("--repeat-n must be positive")
        return repeat_n
    if num_examples > 0 and num_records > num_examples and num_records % num_examples == 0:
        return num_records // num_examples
    return 1


def reward_model_for_record(
    reward_models: Sequence[Dict[str, Any]],
    record_idx: int,
    repeat_n: int,
) -> Dict[str, Any]:
    parquet_idx = record_idx // repeat_n
    if parquet_idx >= len(reward_models):
        raise IndexError(
            f"JSONL row {record_idx} maps to parquet row {parquet_idx}, "
            f"but parquet has only {len(reward_models)} rows. "
            "Check --repeat-n and make sure validation_shuffle was False."
        )
    return reward_models[parquet_idx]


def rubric_ids_from_model(reward_model: Dict[str, Any]) -> List[str]:
    rubric_ids = []
    for idx, rubric in enumerate(_as_list(reward_model.get("rubrics"))):
        rubric = _as_dict(rubric)
        rubric_ids.append(str(rubric.get("id") or f"r{idx + 1}"))
    return rubric_ids


def weights_for_rubrics(record: Dict[str, Any], reward_model: Dict[str, Any], rubric_ids: List[str]) -> List[float]:
    record_weights = _float_list(record.get("rubric_weights"))
    if record_weights and len(record_weights) == len(rubric_ids):
        return record_weights

    rubrics = [_as_dict(rubric) for rubric in _as_list(reward_model.get("rubrics"))]
    id_to_weight: Dict[str, float] = {}
    ordered_weights: List[float] = []
    for idx, rubric in enumerate(rubrics):
        rubric_id = str(rubric.get("id") or f"r{idx + 1}")
        weight = _safe_float(rubric.get("points"), default=0.0)
        id_to_weight[rubric_id] = weight
        ordered_weights.append(weight)

    if rubric_ids and all(rubric_id in id_to_weight for rubric_id in rubric_ids):
        return [id_to_weight[rubric_id] for rubric_id in rubric_ids]
    if len(ordered_weights) == len(rubric_ids):
        return ordered_weights

    raise ValueError(
        f"Could not align rubric weights: log has {len(rubric_ids)} rubric ids, "
        f"parquet reward_model has {len(rubrics)} rubrics."
    )


def edges_for_record(record: Dict[str, Any], reward_model: Dict[str, Any]) -> List[Dict[str, Any]]:
    record_edges = [_as_dict(edge) for edge in _as_list(record.get("graph_edges"))]
    if record_edges:
        return record_edges
    graph = _as_dict(reward_model.get("graph"))
    return [_as_dict(edge) for edge in _as_list(graph.get("edges"))]


def score_lists_for_record(record: Dict[str, Any], num_rubrics: int) -> Dict[str, List[float]]:
    score_lists: Dict[str, List[float]] = {}
    p_list = _float_list(record.get("p_list"))
    for method, field_name in METHOD_TO_SCORE_FIELD.items():
        values = _float_list(record.get(field_name))
        if method == "flat" and not values:
            values = p_list
        if len(values) != num_rubrics:
            raise ValueError(
                f"Field {field_name} for method {method} has length {len(values)}, "
                f"expected {num_rubrics}."
            )
        score_lists[method] = values
    return score_lists


def compute_row_contribution(
    record: Dict[str, Any],
    reward_model: Dict[str, Any],
    edge_types: Optional[set[str]],
    threshold: float,
) -> RowContribution:
    record_rubric_ids = [str(item) for item in _as_list(record.get("rubric_ids"))]
    rubric_ids = record_rubric_ids or rubric_ids_from_model(reward_model)
    if not rubric_ids:
        raise ValueError("Missing rubric_ids in JSONL and reward_model.rubrics in parquet")

    p_list = _float_list(record.get("p_list"))
    if len(p_list) != len(rubric_ids):
        raise ValueError(f"p_list has length {len(p_list)}, expected {len(rubric_ids)}")

    weights = weights_for_rubrics(record, reward_model, rubric_ids)
    score_lists = score_lists_for_record(record, len(rubric_ids))
    id_to_idx = {rubric_id: idx for idx, rubric_id in enumerate(rubric_ids)}
    positive_weight_sum = sum(weight for weight in weights if weight > 0.0)

    contribution = RowContribution(
        leakage_sum={method: 0.0 for method in score_lists},
        preservation_sum={method: 0.0 for method in score_lists},
    )

    for edge in edges_for_record(record, reward_model):
        edge_type = str(edge.get("type") or edge.get("edge_type") or "")
        if edge_types is not None and edge_type not in edge_types:
            continue
        parent = str(edge.get("parent") or "")
        child = str(edge.get("child") or "")
        if parent not in id_to_idx or child not in id_to_idx:
            contribution.invalid_edges += 1
            continue

        contribution.considered_edges += 1
        parent_idx = id_to_idx[parent]
        child_idx = id_to_idx[child]
        parent_p = p_list[parent_idx]
        child_p = p_list[child_idx]
        child_weight = abs(weights[child_idx])

        if child_p >= threshold and parent_p < threshold:
            contribution.leakage_count += 1
            if positive_weight_sum > 0.0:
                weight_factor = child_weight / positive_weight_sum
                for method, values in score_lists.items():
                    contribution.leakage_sum[method] += weight_factor * values[child_idx]

        if child_p >= threshold and parent_p >= threshold:
            contribution.preservation_count += 1
            for method, values in score_lists.items():
                contribution.preservation_sum[method] += values[child_idx] / child_p

    return contribution


def add_contribution(totals: Totals, contribution: RowContribution) -> None:
    totals.rows_used += 1
    totals.leakage_count += contribution.leakage_count
    totals.preservation_count += contribution.preservation_count
    totals.considered_edges += contribution.considered_edges
    totals.invalid_edges += contribution.invalid_edges
    if contribution.considered_edges == 0:
        totals.rows_without_edges += 1
    for method, value in contribution.leakage_sum.items():
        totals.leakage_sum[method] = totals.leakage_sum.get(method, 0.0) + value
    for method, value in contribution.preservation_sum.items():
        totals.preservation_sum[method] = totals.preservation_sum.get(method, 0.0) + value


def compute_totals(contributions: Iterable[RowContribution], rows_total: int, rows_missing_fields: int = 0) -> Totals:
    totals = Totals(rows_total=rows_total, rows_missing_fields=rows_missing_fields)
    for contribution in contributions:
        add_contribution(totals, contribution)
    return totals


def _ratio(numerator: float, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / denominator


def _percentile_interval(values: Sequence[float], ci: float) -> Optional[Tuple[float, float]]:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return None
    alpha = (1.0 - ci) / 2.0
    return (
        float(np.percentile(finite_values, 100.0 * alpha)),
        float(np.percentile(finite_values, 100.0 * (1.0 - alpha))),
    )


def bootstrap_intervals(
    contributions: Sequence[RowContribution],
    methods: Sequence[str],
    n_bootstrap: int,
    ci: float,
    seed: int,
) -> Dict[str, Dict[str, Optional[Tuple[float, float]]]]:
    intervals = {
        method: {"leakage": None, "preservation": None}
        for method in methods
    }
    if not contributions or n_bootstrap <= 0:
        return intervals

    rng = np.random.default_rng(seed)
    leakage_samples = {method: [] for method in methods}
    preservation_samples = {method: [] for method in methods}

    for _ in range(n_bootstrap):
        sampled_idxs = rng.integers(0, len(contributions), size=len(contributions))
        totals = compute_totals((contributions[idx] for idx in sampled_idxs), rows_total=len(contributions))
        for method in methods:
            leakage = _ratio(totals.leakage_sum.get(method, 0.0), totals.leakage_count)
            preservation = _ratio(totals.preservation_sum.get(method, 0.0), totals.preservation_count)
            if leakage is not None:
                leakage_samples[method].append(leakage)
            if preservation is not None:
                preservation_samples[method].append(preservation)

    for method in methods:
        intervals[method]["leakage"] = _percentile_interval(leakage_samples[method], ci)
        intervals[method]["preservation"] = _percentile_interval(preservation_samples[method], ci)
    return intervals


def summarize(
    totals: Totals,
    intervals: Dict[str, Dict[str, Optional[Tuple[float, float]]]],
    methods: Sequence[str],
) -> Dict[str, Any]:
    method_summaries = {}
    for method in methods:
        leakage = _ratio(totals.leakage_sum.get(method, 0.0), totals.leakage_count)
        preservation = _ratio(totals.preservation_sum.get(method, 0.0), totals.preservation_count)
        method_summaries[method] = {
            "fcp_leakage": leakage,
            "fcp_leakage_ci": intervals.get(method, {}).get("leakage"),
            "fcp_preservation": preservation,
            "fcp_preservation_ci": intervals.get(method, {}).get("preservation"),
        }

    return {
        "methods": method_summaries,
        "counts": {
            "rows_total": totals.rows_total,
            "rows_used": totals.rows_used,
            "rows_missing_fields": totals.rows_missing_fields,
            "rows_without_considered_edges": totals.rows_without_edges,
            "considered_edges": totals.considered_edges,
            "invalid_edges": totals.invalid_edges,
            "leakage_cases": totals.leakage_count,
            "satisfied_cases": totals.preservation_count,
        },
    }


def _format_value(value: Optional[float]) -> str:
    return "NA" if value is None else f"{value:.6f}"


def _format_ci(interval: Optional[Tuple[float, float]]) -> str:
    if interval is None:
        return "NA"
    return f"[{interval[0]:.6f}, {interval[1]:.6f}]"


def print_table(summary: Dict[str, Any], methods: Sequence[str]) -> None:
    counts = summary["counts"]
    print(
        f"Rows used: {counts['rows_used']}/{counts['rows_total']} | "
        f"considered edges: {counts['considered_edges']} | "
        f"leakage cases: {counts['leakage_cases']} | "
        f"satisfied cases: {counts['satisfied_cases']}"
    )
    if counts["rows_missing_fields"]:
        print(f"Rows skipped because of missing/misaligned fields: {counts['rows_missing_fields']}")
    if counts["rows_without_considered_edges"]:
        print(f"Rows without considered edges: {counts['rows_without_considered_edges']}")
    if counts["invalid_edges"]:
        print(f"Invalid edges skipped: {counts['invalid_edges']}")

    print()
    print("| method | L_FCP lower is better | 95% CI | P higher is better | 95% CI |")
    print("|---|---:|---:|---:|---:|")
    for method in methods:
        metrics = summary["methods"][method]
        print(
            f"| {method} | "
            f"{_format_value(metrics['fcp_leakage'])} | "
            f"{_format_ci(metrics['fcp_leakage_ci'])} | "
            f"{_format_value(metrics['fcp_preservation'])} | "
            f"{_format_ci(metrics['fcp_preservation_ci'])} |"
        )


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    def default(value: Any) -> Any:
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=default) + "\n", encoding="utf-8")


def write_csv(path: Path, summary: Dict[str, Any], methods: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "fcp_leakage",
                "fcp_leakage_ci_low",
                "fcp_leakage_ci_high",
                "fcp_preservation",
                "fcp_preservation_ci_low",
                "fcp_preservation_ci_high",
                "leakage_cases",
                "satisfied_cases",
            ],
        )
        writer.writeheader()
        counts = summary["counts"]
        for method in methods:
            metrics = summary["methods"][method]
            leakage_ci = metrics["fcp_leakage_ci"]
            preservation_ci = metrics["fcp_preservation_ci"]
            writer.writerow(
                {
                    "method": method,
                    "fcp_leakage": metrics["fcp_leakage"],
                    "fcp_leakage_ci_low": None if leakage_ci is None else leakage_ci[0],
                    "fcp_leakage_ci_high": None if leakage_ci is None else leakage_ci[1],
                    "fcp_preservation": metrics["fcp_preservation"],
                    "fcp_preservation_ci_low": None if preservation_ci is None else preservation_ci[0],
                    "fcp_preservation_ci_high": None if preservation_ci is None else preservation_ci[1],
                    "leakage_cases": counts["leakage_cases"],
                    "satisfied_cases": counts["satisfied_cases"],
                }
            )


def parse_edge_types(raw_edge_types: Sequence[str]) -> Optional[set[str]]:
    if len(raw_edge_types) == 1 and raw_edge_types[0].lower() == "all":
        return set(ALL_EDGE_TYPES)
    edge_types = {edge_type.strip() for edge_type in raw_edge_types if edge_type.strip()}
    unknown = edge_types - set(ALL_EDGE_TYPES)
    if unknown:
        raise ValueError(f"Unknown edge types: {sorted(unknown)}")
    return edge_types


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute False Credit Propagation diagnostics from a HealthBench "
            "validation JSONL dump and the corresponding validation parquet."
        )
    )
    parser.add_argument("jsonl", type=Path, help="Validation generation JSONL, e.g. validation_log/20.jsonl")
    parser.add_argument(
        "--val-parquet",
        type=Path,
        required=True,
        help="The validation parquet used for the same run, e.g. healthbench_val_gear.parquet",
    )
    parser.add_argument(
        "--edge-types",
        nargs="+",
        default=list(DEFAULT_EDGE_TYPES),
        help="Dependency edge types to include. Use 'all' for weak_prerequisite strong_prerequisite trigger.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Support threshold for p_i and p_j.")
    parser.add_argument(
        "--repeat-n",
        type=int,
        default=None,
        help="Validation responses per original prompt. By default inferred when JSONL length is a multiple of parquet length.",
    )
    parser.add_argument("--bootstrap", type=int, default=1000, help="Number of row-level bootstrap samples.")
    parser.add_argument("--ci", type=float, default=0.95, help="Confidence interval mass, e.g. 0.95.")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap random seed.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON summary path.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional CSV summary path.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on the first malformed row instead of skipping it.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.jsonl.exists():
        raise SystemExit(f"JSONL not found: {args.jsonl}")
    if not args.val_parquet.exists():
        raise SystemExit(f"Validation parquet not found: {args.val_parquet}")
    if not (0.0 < args.ci < 1.0):
        raise SystemExit("--ci must be between 0 and 1")

    edge_types = parse_edge_types(args.edge_types)
    records = load_jsonl(args.jsonl)
    reward_models = load_reward_models(args.val_parquet)
    repeat_n = infer_repeat_n(len(records), len(reward_models), args.repeat_n)

    contributions: List[RowContribution] = []
    rows_missing_fields = 0
    for record_idx, record in enumerate(records):
        try:
            reward_model = reward_model_for_record(reward_models, record_idx, repeat_n)
            contributions.append(
                compute_row_contribution(
                    record=record,
                    reward_model=reward_model,
                    edge_types=edge_types,
                    threshold=args.threshold,
                )
            )
        except Exception as exc:  # noqa: BLE001
            if args.strict:
                raise
            rows_missing_fields += 1
            print(f"[warn] skipping JSONL row {record_idx}: {exc}")

    methods = list(METHOD_TO_SCORE_FIELD.keys())
    totals = compute_totals(contributions, rows_total=len(records), rows_missing_fields=rows_missing_fields)
    intervals = bootstrap_intervals(
        contributions=contributions,
        methods=methods,
        n_bootstrap=args.bootstrap,
        ci=args.ci,
        seed=args.seed,
    )
    summary = summarize(totals=totals, intervals=intervals, methods=methods)
    summary["config"] = {
        "jsonl": str(args.jsonl),
        "val_parquet": str(args.val_parquet),
        "edge_types": sorted(edge_types or []),
        "threshold": args.threshold,
        "repeat_n": repeat_n,
        "bootstrap": args.bootstrap,
        "ci": args.ci,
        "seed": args.seed,
    }

    print_table(summary, methods)
    if args.output_json:
        write_json(args.output_json, summary)
    if args.output_csv:
        write_csv(args.output_csv, summary, methods)


if __name__ == "__main__":
    main()
