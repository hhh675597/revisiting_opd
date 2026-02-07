"""
Prepare multi-task training data for AlfWorld + Math on-policy distillation.

This script creates training datasets that combine:
1. AlfWorld placeholder samples (env loads its own game files)
2. Math problem samples (env uses env_kwargs from dataset)

Usage:
    python prepare_multitask_train_data.py \
        --alfworld_num 16 \
        --math_data_path /path/to/math/train.parquet \
        --output_dir /path/to/multitask_data \
        --batching_mode sequential
"""

import argparse
import os

import datasets
import pandas as pd


def create_alfworld_placeholders(num_samples: int) -> datasets.Dataset:
    """
    Create placeholder samples for AlfWorld.
    
    AlfWorld environment ignores dataset content and loads its own game files
    internally. These placeholders just trigger environment creation.
    
    Args:
        num_samples: Number of placeholder samples (typically = batch_size)
    
    Returns:
        Dataset with minimal placeholder data
    """
    samples = []
    for i in range(num_samples):
        samples.append({
            "task_type": "alfworld",
            "data_source": "alfworld_placeholder",
            "prompt": [{"role": "user", "content": "placeholder"}],
            "env_kwargs": {"task_type": "alfworld"},
            "extra_info": {
                "index": i,
                "split": "train",
            }
        })
    
    return datasets.Dataset.from_list(samples)


def load_and_process_math_data(data_path: str) -> datasets.Dataset:
    """
    Load math dataset and add task_type field.
    
    Math environment uses env_kwargs from the dataset to get the actual
    problem and ground truth.
    
    Args:
        data_path: Path to math parquet file
    
    Returns:
        Dataset with task_type and properly formatted env_kwargs
    """
    print(f"Loading math dataset from {data_path}")
    dataset = datasets.load_dataset("parquet", data_files=data_path, split="train")
    print(f"  Loaded {len(dataset)} math samples")
    
    def add_task_type(example, idx):
        # Add task_type
        example["task_type"] = "math"
        
        # Ensure env_kwargs has task_type
        if "env_kwargs" in example and example["env_kwargs"] is not None:
            env_kwargs = dict(example["env_kwargs"])
            env_kwargs["task_type"] = "math"
            example["env_kwargs"] = env_kwargs
        else:
            # Create env_kwargs if missing
            example["env_kwargs"] = {
                "task_type": "math",
                "question": example.get("prompt", [{}])[0].get("content", ""),
                "ground_truth": example.get("reward_model", {}).get("ground_truth", ""),
                "data_source": example.get("data_source", "math"),
            }
        
        return example
    
    dataset = dataset.map(add_task_type, with_indices=True)
    return dataset


def combine_datasets(
    alfworld_dataset: datasets.Dataset,
    math_dataset: datasets.Dataset,
    batching_mode: str = "sequential",
    shuffle: bool = True,
    seed: int = 42,
) -> datasets.Dataset:
    """
    Combine AlfWorld and Math datasets.
    
    Args:
        alfworld_dataset: AlfWorld placeholder dataset
        math_dataset: Math problem dataset
        batching_mode: "sequential" (recommended) or "mixed"
        shuffle: Whether to shuffle the combined dataset
        seed: Random seed
    
    Returns:
        Combined dataset
    """
    print(f"Combining datasets (mode: {batching_mode})")
    print(f"  AlfWorld: {len(alfworld_dataset)} samples")
    print(f"  Math: {len(math_dataset)} samples")
    
    if batching_mode == "sequential":
        # For sequential batching, we can just concatenate
        # The sampler will group by task_type anyway
        combined = datasets.concatenate_datasets([alfworld_dataset, math_dataset])
        
        if shuffle:
            print(f"  Shuffling combined dataset (seed={seed})")
            combined = combined.shuffle(seed=seed)
    else:
        # Mixed mode
        combined = datasets.concatenate_datasets([alfworld_dataset, math_dataset])
        if shuffle:
            combined = combined.shuffle(seed=seed)
    
    print(f"  Total: {len(combined)} samples")
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Prepare multi-task training data for AlfWorld + Math"
    )
    parser.add_argument(
        "--alfworld_num",
        type=int,
        default=16,
        help="Number of AlfWorld placeholder samples (typically = batch_size)",
    )
    parser.add_argument(
        "--math_data_path",
        type=str,
        required=True,
        help="Path to math training parquet file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for processed datasets",
    )
    parser.add_argument(
        "--batching_mode",
        type=str,
        default="sequential",
        choices=["sequential", "mixed"],
        help="Batching mode for multi-task training",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=True,
        help="Shuffle the combined dataset",
    )
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Do not shuffle the combined dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )
    parser.add_argument(
        "--separate_files",
        action="store_true",
        help="Also save separate task files in addition to combined file",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create AlfWorld placeholders
    print(f"\n[1/4] Creating {args.alfworld_num} AlfWorld placeholder samples...")
    alfworld_dataset = create_alfworld_placeholders(args.alfworld_num)
    
    # Load and process Math data
    print(f"\n[2/4] Loading Math dataset...")
    math_dataset = load_and_process_math_data(args.math_data_path)
    
    # Save separate files if requested
    if args.separate_files:
        print(f"\n[3/4] Saving separate task files...")
        alfworld_path = os.path.join(args.output_dir, "alfworld_train.parquet")
        math_path = os.path.join(args.output_dir, "math_train.parquet")
        
        alfworld_dataset.to_parquet(alfworld_path)
        print(f"  Saved AlfWorld: {alfworld_path}")
        
        math_dataset.to_parquet(math_path)
        print(f"  Saved Math: {math_path}")
    else:
        print(f"\n[3/4] Skipping separate files (use --separate_files to enable)")
    
    # Combine datasets
    print(f"\n[4/4] Combining datasets...")
    shuffle = args.shuffle and not args.no_shuffle
    combined_dataset = combine_datasets(
        alfworld_dataset,
        math_dataset,
        batching_mode=args.batching_mode,
        shuffle=shuffle,
        seed=args.seed,
    )
    
    # Save combined dataset
    output_path = os.path.join(args.output_dir, "test.parquet") # be careful
    combined_dataset.to_parquet(output_path)
    print(f"\nSaved combined dataset: {output_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics:")
    print("="*60)
    
    # Count by task_type
    task_counts = {}
    for sample in combined_dataset:
        task_type = sample["task_type"]
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    
    for task_type, count in sorted(task_counts.items()):
        percentage = 100 * count / len(combined_dataset)
        print(f"  {task_type:12s}: {count:6d} samples ({percentage:5.1f}%)")
    
    print(f"  {'Total':12s}: {len(combined_dataset):6d} samples")
    print("="*60)
    
    print(f"\n✓ Multi-task training data prepared successfully!")
    print(f"  Output: {output_path}")
    print(f"  Batching mode: {args.batching_mode}")


if __name__ == "__main__":
    main()
