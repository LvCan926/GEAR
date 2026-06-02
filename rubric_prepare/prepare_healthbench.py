import copy
import os
import json
import datasets
import pandas as pd
from typing import List, Dict, Any
from dataclasses import dataclass

DEFAULT_NODE_TYPE_BY_SIGN = {
    "positive": "foundation",
    "negative": "penalty",
}

@dataclass
class RubricItem:
    id: str
    criterion: str
    points: float
    tags: Dict[str, Any]

    def __str__(self) -> str:
        return self.criterion

    @classmethod
    def from_dict(cls, d: dict) -> "RubricItem":
        tags_data = d.get("tags", [])
        # If tags is in list format, try to parse as dictionary
        if isinstance(tags_data, list):
            tags_dict = {}
            for tag in tags_data:
                if isinstance(tag, str) and ":" in tag:
                    key, value = tag.split(":", 1)
                    tags_dict[key] = value
                elif isinstance(tag, str):
                    # For tags without colon, use tag itself as key with value True
                    tags_dict[tag] = True
            tags_data = tags_dict
        elif not isinstance(tags_data, dict):
            # If neither list nor dict, set as empty dict
            tags_data = {}
            
        return cls(
            id=d.get("id", ""),
            criterion=d["criterion"],
            points=d["points"],
            tags=tags_data
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "criterion": self.criterion,
            "points": self.points,
            "tags": self.tags
        }


def infer_node_type(points: float) -> str:
    return DEFAULT_NODE_TYPE_BY_SIGN["negative"] if points < 0 else DEFAULT_NODE_TYPE_BY_SIGN["positive"]


def normalize_rubric(example_rubric: Dict[str, Any], rubric_idx: int) -> RubricItem:
    rubric = RubricItem.from_dict(example_rubric)
    rubric_id = rubric.id or f"r{rubric_idx + 1}"
    tags = dict(rubric.tags or {})
    tags.setdefault("node_type", infer_node_type(rubric.points))
    return RubricItem(
        id=rubric_id,
        criterion=rubric.criterion,
        points=rubric.points,
        tags=tags,
    )

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def make_map_fn(split: str):
    """Construct data mapping function"""
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        # Extract prompt
        prompt = example['prompt']
        
        # Extract rubrics
        rubrics = [normalize_rubric(r, rubric_idx) for rubric_idx, r in enumerate(example['rubrics'])]
        
        # Construct reward_model field
        reward_model = {
            "style": "rubric",
            "rubrics": [r.to_dict() for r in rubrics],
            "graph": {
                "nodes": [
                    {"id": r.id, "node_type": r.tags.get("node_type", infer_node_type(r.points))}
                    for r in rubrics
                ],
                "edges": [],
            },
            "ground_truth": ""  # Use empty string
        }
        
        # Construct data format required by verl
        data = {
            "data_source": "healthbench",
            "prompt": prompt,  # Keep outer prompt
            "ability": "medical_chat",
            "reward_model": reward_model,  # Keep outer reward_model
            "extra_info": {
                "prompt": prompt,  # Also put prompt in extra_info
                "reward_model": copy.deepcopy(reward_model)  # Keep in sync while avoiding aliasing
            }
        }
        return data
    
    return process_fn

def process_dataset(data_list: List[Dict[str, Any]], split: str) -> datasets.Dataset:
    """Process dataset"""
    dataset = datasets.Dataset.from_list(data_list)
    processed_dataset = dataset.map(
        function=make_map_fn(split),
        with_indices=True,
        remove_columns=dataset.column_names  # Remove all original columns
    )
    # Shuffle the data
    
    shuffled_dataset = processed_dataset.shuffle(seed=42)
    return shuffled_dataset

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='raw_data/Healthbench')
    parser.add_argument('--output_dir', default='data/health_bench')
    parser.add_argument('--hdfs_dir', default=None)
    
    args = parser.parse_args()
    
    # # Load training data
    # train_data = []
    # for i in range(1, 4):  # group1-3
    #     file_path = os.path.join(args.local_dir, f'healthbench_group{i}.jsonl')
    #     train_data.extend(load_jsonl(file_path))

    # Load training data
    train_file = os.path.join(args.local_dir, 'healthbench_train.jsonl')
    train_data = load_jsonl(train_file)
    
    # Load validation data
    val_file = os.path.join(args.local_dir, 'healthbench_eval.jsonl')
    val_data = load_jsonl(val_file)
    # Keep only first 100 validation data as "validation set"
    val_data = val_data
    
    # Process training and validation sets
    train_dataset = process_dataset(train_data, 'train')
    val_dataset = process_dataset(val_data, 'val')
    
    # Save as parquet format
    os.makedirs(args.output_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(args.output_dir, 'healthbench_train.parquet'))
    val_dataset.to_parquet(os.path.join(args.output_dir, 'healthbench_val.parquet'))
    
    # Print dataset information
    print("\nDataset Information:")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Print data sample examples
    print("\nTraining set sample example:")
    print(json.dumps(train_dataset[0], indent=2, ensure_ascii=False))
    print("\nValidation set sample example:")
    print(json.dumps(val_dataset[0], indent=2, ensure_ascii=False))
    
    if args.hdfs_dir:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(args.hdfs_dir)
        copy(src=args.output_dir, dst=args.hdfs_dir)

if __name__ == '__main__':
    main()
