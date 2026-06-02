import argparse
import copy
import json
import os
import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if line:
                item = json.loads(line)
                item["_source_index"] = line_idx
                data.append(item)
    return data


def stratified_split(
    rows: List[Dict[str, Any]],
    key_fn: Callable[[Dict[str, Any]], Tuple[str, ...]],
    val_size: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if val_size <= 0 or val_size >= len(rows):
        raise ValueError(f"val_size must be in [1, {len(rows) - 1}], got {val_size}")

    rng = random.Random(seed)
    grouped: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)

    keys = sorted(grouped)
    quotas: Dict[Tuple[str, ...], int] = {}
    remainders = []
    total = len(rows)
    for key in keys:
        group_len = len(grouped[key])
        exact = group_len * val_size / total
        quota = int(exact)
        quotas[key] = min(quota, group_len)
        remainders.append((exact - quota, group_len, key))

    remaining = val_size - sum(quotas.values())
    for _, _, key in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        if quotas[key] < len(grouped[key]):
            quotas[key] += 1
            remaining -= 1

    train_rows = []
    val_rows = []
    for key in keys:
        group = list(grouped[key])
        rng.shuffle(group)
        val_count = quotas[key]
        val_rows.extend(group[:val_count])
        train_rows.extend(group[val_count:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def build_prompt(example: Dict[str, Any]) -> List[Dict[str, str]]:
    context = str(example.get("context", "")).strip()
    question = str(example.get("question", "")).strip()
    content = (
        "请基于以下案例材料，按照“结论 + 案情简述 + 分析过程 + 依据法条”的结构回答问题。\n\n"
        f"【案例材料】\n{context}\n\n"
        f"【问题】\n{question}"
    )
    return [{"role": "user", "content": content}]


def _parse_points(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid PLawBench points value: {value!r}") from exc


def convert_example(example: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_prompt(example)

    rubrics = []
    for rubric_idx, rubric in enumerate(example.get("rubrics", []) or []):
        criterion = str(rubric.get("criterion", "")).strip()
        if not criterion:
            continue
        points = _parse_points(rubric.get("points"))
        category = str(rubric.get("tags", "")).strip()
        rubrics.append(
            {
                "id": f"r{rubric_idx + 1}",
                "criterion": criterion,
                "points": points,
                "tags": {
                    "verifier": "llm",
                    "score_type": "partial_points",
                    "category": category,
                },
            }
        )

    if not rubrics:
        raise ValueError(f"No rubrics found for PLawBench label={example.get('label')}")

    reward_model = {
        "style": "rubric",
        "rubrics": rubrics,
        "ground_truth": "",
    }

    source_meta = {
        "source_index": example.get("_source_index"),
        "label": example.get("label"),
    }
    return {
        "data_source": "plawbench",
        "prompt": prompt,
        "ability": "legal_reasoning",
        "reward_model": reward_model,
        "extra_info": {
            "prompt": prompt,
            "reward_model": copy.deepcopy(reward_model),
            "source_meta": source_meta,
        },
    }


def process_dataset(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [convert_example(row) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        default="raw_data/Plawbench/practical_case_analysis_250.jsonl",
    )
    parser.add_argument("--output_dir", default="data/plaw_bench")
    parser.add_argument("--val_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    rows = load_jsonl(args.input_file)
    train_rows, val_rows = stratified_split(
        rows,
        key_fn=lambda row: (str(row.get("label", "")),),
        val_size=args.val_size,
        seed=args.seed,
    )

    train_dataset = process_dataset(train_rows)
    val_dataset = process_dataset(val_rows)

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "plawbench_train.parquet")
    val_path = os.path.join(args.output_dir, "plawbench_val.parquet")
    pd.DataFrame(train_dataset).to_parquet(train_path, index=False)
    pd.DataFrame(val_dataset).to_parquet(val_path, index=False)

    print("\nDataset Information:")
    print(f"Input size: {len(rows)}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Training parquet: {train_path}")
    print(f"Validation parquet: {val_path}")
    print("\nTraining set sample example:")
    print(json.dumps(train_dataset[0], indent=2, ensure_ascii=False))
    print("\nValidation set sample example:")
    print(json.dumps(val_dataset[0], indent=2, ensure_ascii=False))

    if args.hdfs_dir:
        from verl.utils.hdfs_io import copy, makedirs

        makedirs(args.hdfs_dir)
        copy(src=args.output_dir, dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
