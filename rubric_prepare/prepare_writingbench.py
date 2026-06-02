import argparse
import copy
import json
import os
import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd


CHECKLIST_SCORE_KEYS = ("1-2", "3-4", "5-6", "7-8", "9-10")


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
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


def format_checklist_item(item: Dict[str, Any]) -> str:
    lines = [
        str(item.get("name", "")).strip(),
        "",
        str(item.get("criteria_description", "")).strip(),
        "",
        "Scoring rubric:",
    ]
    for key in CHECKLIST_SCORE_KEYS:
        value = str(item.get(key, "")).strip()
        if value:
            lines.append(f"{key}: {value}")
    return "\n".join(line for line in lines if line != "").strip()


def convert_example(example: Dict[str, Any]) -> Dict[str, Any]:
    prompt = [{"role": "user", "content": str(example.get("query", ""))}]

    rubrics = []
    for rubric_idx, checklist_item in enumerate(example.get("checklist", []) or []):
        criterion = format_checklist_item(checklist_item)
        if not criterion:
            continue
        rubrics.append(
            {
                "id": f"r{rubric_idx + 1}",
                "criterion": criterion,
                "points": 10.0,
                "tags": {
                    "verifier": "llm",
                    "score_type": "scale_1_10",
                    "criterion_name": str(checklist_item.get("name", "")).strip(),
                },
            }
        )

    if not rubrics:
        raise ValueError(f"No checklist rubrics found for WritingBench index={example.get('index')}")

    reward_model = {
        "style": "rubric",
        "rubrics": rubrics,
        "ground_truth": "",
    }

    source_meta = {
        "index": example.get("index"),
        "domain1": example.get("domain1"),
        "domain2": example.get("domain2"),
        "lang": example.get("lang"),
    }
    return {
        "data_source": "writingbench",
        "prompt": prompt,
        "ability": "writing",
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
        default="raw_data/Writingbench/benchmark_all.jsonl",
    )
    parser.add_argument("--output_dir", default="data/writing_bench")
    parser.add_argument("--val_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    rows = load_jsonl(args.input_file)
    train_rows, val_rows = stratified_split(
        rows,
        key_fn=lambda row: (str(row.get("lang", "")), str(row.get("domain1", ""))),
        val_size=args.val_size,
        seed=args.seed,
    )

    train_dataset = process_dataset(train_rows)
    val_dataset = process_dataset(val_rows)

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "writingbench_train.parquet")
    val_path = os.path.join(args.output_dir, "writingbench_val.parquet")
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
