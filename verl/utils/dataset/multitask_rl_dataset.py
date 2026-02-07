"""
Multi-Task RLHF Dataset for on-policy distillation training.

This module provides dataset classes for multi-task training that:
1. Support loading multiple task datasets with task_type field
2. Provide sequential batching mode
3. Route samples to correct environment managers
"""

import copy
import logging
import os
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset, Sampler
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

logger = logging.getLogger(__name__)


class MultiTaskRLHFDataset(RLHFDataset):
    """
    Multi-Task RLHF Dataset that extends RLHFDataset with task_type awareness.
    
    This dataset expects each sample to have a `task_type` field 
    that identifies which task the sample belongs to (e.g., "alfworld", "math"). 
    The task_type is used for:
    1. Routing to the correct environment manager during rollout
    2. Routing to the correct reference model for OPD advantage computation
    
    The dataset supports two batching modes:
    - "sequential": Each minibatch contains samples from only one task
    - TODO "mixed": Samples from all tasks are mixed in each minibatch
    
    Args:
        data_files: Path(s) to parquet file(s) containing task_type field
        tokenizer: Tokenizer for processing prompts
        config: Configuration with options like batching_mode
        processor: Optional multimodal processor
    """
    
    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        # Get multi-task specific config
        self.batching_mode = config.get("batching_mode", "sequential")
        self.task_type_key = config.get("task_type_key", "task_type")
        
        # Initialize parent class
        super().__init__(data_files, tokenizer, config, processor)
        
        # Build task indices for sequential batching
        self._build_task_indices()
        
        logger.info(
            f"MultiTaskRLHFDataset initialized with batching_mode={self.batching_mode}, "
            f"task_types={list(self.task_indices.keys())}"
        )
    
    def _build_task_indices(self):
        """Build indices grouped by task_type for sequential batching."""
        self.task_indices: Dict[str, List[int]] = defaultdict(list)
        
        for idx in range(len(self.dataframe)):
            row = self.dataframe[idx]
            task_type = row.get(self.task_type_key, "unknown")
            self.task_indices[task_type].append(idx)
        
        # Log task distribution
        for task_type, indices in self.task_indices.items():
            logger.info(f"  Task '{task_type}': {len(indices)} samples")
    
    def __getitem__(self, item):
        """
        Get item with task_type field included.
        
        Returns:
            dict containing all fields from parent class plus task_type
        """
        row_dict = super().__getitem__(item)
        
        # Ensure task_type is in the output
        if self.task_type_key not in row_dict:
            # Try to get from original dataframe
            original_row = self.dataframe[item]
            row_dict[self.task_type_key] = original_row.get(
                self.task_type_key, "unknown"
            )
        
        return row_dict
    
    def get_task_types(self) -> List[str]:
        """Return list of unique task types in the dataset."""
        return list(self.task_indices.keys())
    
    def get_task_indices(self, task_type: str) -> List[int]:
        """Return indices for a specific task type."""
        return self.task_indices.get(task_type, [])
    
    def get_task_dataset_size(self, task_type: str) -> int:
        """Return number of samples for a specific task type."""
        return len(self.task_indices.get(task_type, []))


class SequentialTaskSampler(Sampler):
    """
    Sampler that yields indices grouped by task_type for sequential batching.
    
    Each batch will contain samples from only one task type. 
    Tasks are cycled through in a round-robin fashion.
    
    Args:
        dataset: MultiTaskRLHFDataset instance
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle within each task
        drop_last: Whether to drop incomplete batches
        seed: Random seed for shuffling
    """
    
    def __init__(
        self,
        dataset: MultiTaskRLHFDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        
        self.task_types = dataset.get_task_types()
        self.task_indices = {
            task_type: dataset.get_task_indices(task_type)
            for task_type in self.task_types
        }
    
    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling."""
        self.epoch = epoch
    
    def __iter__(self) -> Iterator[int]:
        """
        Yield indices in sequential task batches.
        
        Samples are organized such that each batch_size consecutive indices belong to the same task type.
        """
        # warning: alfworld(training data logic isn't the same) not checked yet

        # Create generator for deterministic shuffling
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # Shuffle indices within each task if requested
        task_indices_shuffled = {}
        for task_type, indices in self.task_indices.items():
            indices = np.array(indices)
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=g).numpy()
                indices = indices[perm]
            task_indices_shuffled[task_type] = list(indices)
        
        # Create batches for each task
        task_batches = {}
        for task_type, indices in task_indices_shuffled.items():
            batches = []
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
            task_batches[task_type] = batches
        
        # Interleave batches from different tasks (round-robin)
        task_batch_iters = {
            task_type: iter(batches)
            for task_type, batches in task_batches.items()
        }
        
        # Cycle through tasks
        task_cycle = list(self.task_types)
        if self.shuffle:
            perm = torch.randperm(len(task_cycle), generator=g).numpy()
            task_cycle = [task_cycle[i] for i in perm]
        
        # Yield indices task by task
        active_tasks = set(task_cycle)
        while active_tasks:
            for task_type in task_cycle:
                if task_type not in active_tasks:
                    continue
                try:
                    batch = next(task_batch_iters[task_type])
                    for idx in batch:
                        yield idx
                except StopIteration:
                    active_tasks.discard(task_type)
    
    def __len__(self) -> int:
        """Return total number of samples."""
        total = 0
        for indices in self.task_indices.values():
            if self.drop_last:
                total += (len(indices) // self.batch_size) * self.batch_size
            else:
                total += len(indices)
        return total


class TaskGroupedBatchSampler(Sampler):
    """
    Batch sampler that groups samples by task_type.
    
    Unlike SequentialTaskSampler which yields individual indices,
    this sampler yields batches of indices directly.
    
    Args:
        dataset: MultiTaskRLHFDataset instance
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle
        drop_last: Whether to drop incomplete batches
        seed: Random seed
    """
    
    def __init__(
        self,
        dataset: MultiTaskRLHFDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        
        self.task_types = dataset.get_task_types()
        self.task_indices = {
            task_type: dataset.get_task_indices(task_type)
            for task_type in self.task_types
        }
    
    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling."""
        self.epoch = epoch
    
    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches of indices, each batch from a single task."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # Create batches for each task
        all_batches = []
        for task_type, indices in self.task_indices.items():
            indices = np.array(indices)
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=g).numpy()
                indices = indices[perm]
            
            for i in range(0, len(indices), self.batch_size):
                batch = list(indices[i:i + self.batch_size])
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)
        
        # Shuffle batch order if requested
        if self.shuffle:
            batch_perm = torch.randperm(len(all_batches), generator=g).numpy()
            all_batches = [all_batches[i] for i in batch_perm]
        
        for batch in all_batches:
            yield batch
    
    def __len__(self) -> int:
        """Return number of batches."""
        total_batches = 0
        for indices in self.task_indices.values():
            n_batches = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size > 0:
                n_batches += 1
            total_batches += n_batches
        return total_batches


def create_multitask_dataloader(
    dataset: MultiTaskRLHFDataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for multi-task training.
    
    The batching behavior depends on dataset.batching_mode:
    - "mixed": Standard shuffled batching (samples from different tasks mixed)
    - "sequential": Each batch contains samples from only one task
    
    Args:
        dataset: MultiTaskRLHFDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        drop_last: Whether to drop incomplete batches
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory
        seed: Random seed
        
    Returns:
        DataLoader configured for the batching mode
    """
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
    
    if dataset.batching_mode == "sequential":
        # Use batch sampler for task-grouped batches
        batch_sampler = TaskGroupedBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
    else:
        # Mixed mode: standard DataLoader behavior
        if shuffle:
            sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(seed))
        else:
            sampler = SequentialSampler(dataset)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )


def multitask_collate_fn(data_list: list[dict]) -> dict:
    """
    Collate function for multi-task batches.
    
    This is an alias for the standard collate_fn but can be extended
    for multi-task specific processing if needed.
    """
    return collate_fn(data_list)
