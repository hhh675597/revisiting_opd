"""
Multi-Task Environment Manager for on-policy distillation training.

This manager routes environment interactions to task-specific environment managers
based on the task_type field in the batch data.
"""

from typing import List, Tuple, Dict, Any
from collections import defaultdict
import numpy as np

from agent_system.environments.base import EnvironmentManagerBase


class MultiEnvironmentManager(EnvironmentManagerBase):
    """
    Environment manager that routes to task-specific environment managers.
    
    This manager handles batches that may contain different task types and routes
    the reset() and step() calls to the appropriate environment manager based on
    the task_type field.
    
    For sequential batching mode (recommended), each batch contains only one task type,
    making routing straightforward.
    
    Args:
        env_managers: Dict mapping task_type to EnvironmentManagerBase instances
                     e.g., {"alfworld": AlfWorldEnvManager, "math": MathEnvManager}
        config: Configuration object
    """
    
    def __init__(
        self,
        env_managers: Dict[str, EnvironmentManagerBase],
        config,
    ):
        # Don't call super().__init__ as we don't have envs/projection_f
        self.env_managers = env_managers
        self.config = config
        
        # Track current task type for each index in the batch
        self.current_task_types = None
        self.batch_size = None
        
        # Track which indices belong to which task
        self.task_indices = {}
        
        print(f"[MultiEnvironmentManager] Initialized with tasks: {list(env_managers.keys())}")
    
    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        """
        Reset environments, routing each sample to its task-specific environment manager.
        
        Args:
            kwargs: List of environment kwargs, each with task_type field
        
        Returns:
            observations: Dict with 'text', 'image', 'anchor' keys
            infos: List of info dicts from environments
        """
        self.batch_size = len(kwargs)
        
        # Group kwargs by task_type
        task_groups = defaultdict(list)
        self.task_indices = defaultdict(list)
        
        for i, kw in enumerate(kwargs):
            task_type = kw.get("task_type", "unknown")
            if task_type == "unknown":
                raise ValueError(f"Sample {i} missing task_type in env_kwargs: {kw}")
            
            task_groups[task_type].append(kw)
            self.task_indices[task_type].append(i)
        
        # Store task types for this batch
        self.current_task_types = [kw.get("task_type") for kw in kwargs]
        
        # Initialize result containers
        all_obs = {
            'text': [None] * self.batch_size,
            'image': [None] * self.batch_size,
            'anchor': [None] * self.batch_size,
        }
        all_infos = [None] * self.batch_size
        
        # Reset each task's environment manager
        for task_type, task_kwargs in task_groups.items():
            if task_type not in self.env_managers:
                raise ValueError(
                    f"No environment manager found for task_type '{task_type}'. "
                    f"Available: {list(self.env_managers.keys())}"
                )
            
            env_mgr = self.env_managers[task_type]
            obs, infos = env_mgr.reset(task_kwargs)
            
            # Map results back to original indices
            for j, orig_idx in enumerate(self.task_indices[task_type]):
                # Handle different observation formats
                if obs.get('text') is not None:
                    all_obs['text'][orig_idx] = obs['text'][j]
                if obs.get('image') is not None:
                    all_obs['image'][orig_idx] = obs['image'][j]
                if obs.get('anchor') is not None:
                    all_obs['anchor'][orig_idx] = obs['anchor'][j]
                
                all_infos[orig_idx] = infos[j]
        
        # Clean up None values for unused modalities
        if all(x is None for x in all_obs['text']):
            all_obs['text'] = None
        if all(x is None for x in all_obs['image']):
            all_obs['image'] = None
        if all(x is None for x in all_obs['anchor']):
            all_obs['anchor'] = None
        
        return all_obs, all_infos
    
    def step(self, text_actions: List[str]):
        """
        Execute actions in environments, routing to appropriate task managers.
        
        Args:
            text_actions: List of text actions for each environment
        
        Returns:
            next_observations: Dict with observation data
            rewards: Array of rewards
            dones: Array of done flags
            infos: List of info dicts
        """
        if self.current_task_types is None:
            raise RuntimeError("Must call reset() before step()")
        
        # Group actions by task type
        task_actions = defaultdict(list)
        for i, action in enumerate(text_actions):
            task_type = self.current_task_types[i]
            task_actions[task_type].append(action)
        
        # Initialize result containers
        all_next_obs = {
            'text': [None] * self.batch_size,
            'image': [None] * self.batch_size,
            'anchor': [None] * self.batch_size,
        }
        all_rewards = [None] * self.batch_size
        all_dones = [None] * self.batch_size
        all_infos = [None] * self.batch_size

        print(f"[DBG] step() called with {len(text_actions)} actions")
        print(f"[DBG] Task distribution: {[(k, len(v)) for k, v in task_actions.items()]}")
        print(f"[DBG] Task env expects: {[((k, v.envs.num_processes) if hasattr(v.envs, 'num_processes') else (k, 'N/A')) for k, v in self.env_managers.items()]}")
        
        # Step each task's environment manager
        for task_type, actions in task_actions.items():
            env_mgr = self.env_managers[task_type]
            next_obs, rewards, dones, infos = env_mgr.step(actions)
            
            # Map results back to original indices
            for j, orig_idx in enumerate(self.task_indices[task_type]):
                # Handle observations
                if next_obs.get('text') is not None:
                    all_next_obs['text'][orig_idx] = next_obs['text'][j]
                if next_obs.get('image') is not None:
                    all_next_obs['image'][orig_idx] = next_obs['image'][j]
                if next_obs.get('anchor') is not None:
                    all_next_obs['anchor'][orig_idx] = next_obs['anchor'][j]
                
                all_rewards[orig_idx] = rewards[j]
                all_dones[orig_idx] = dones[j]
                all_infos[orig_idx] = infos[j]
        
        # Clean up None values
        if all(x is None for x in all_next_obs['text']):
            all_next_obs['text'] = None
        if all(x is None for x in all_next_obs['image']):
            all_next_obs['image'] = None
        if all(x is None for x in all_next_obs['anchor']):
            all_next_obs['anchor'] = None
        
        # Convert to numpy arrays
        all_rewards = np.array(all_rewards)
        all_dones = np.array(all_dones)
        
        return all_next_obs, all_rewards, all_dones, all_infos
    
    def success_evaluator(
        self,
        total_infos: List[List[Dict]],
        total_batch_list: List[List[Dict]],
        episode_rewards: np.ndarray,
        episode_lengths: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate success metrics by delegating to task-specific evaluators.
        
        Args:
            total_infos: List of info lists for each trajectory
            total_batch_list: List of batch lists for each trajectory
            episode_rewards: Episode rewards array
            episode_lengths: Episode lengths array
        
        Returns:
            Dict of success metrics aggregated across all tasks
        """
        # Group trajectories by task type
        task_trajectories = defaultdict(lambda: {
            'infos': [],
            'batches': [],
            'indices': [],
        })
        
        for batch_idx in range(len(total_batch_list)):
            task_type = self.current_task_types[batch_idx]
            task_trajectories[task_type]['infos'].append(total_infos[batch_idx])
            task_trajectories[task_type]['batches'].append(total_batch_list[batch_idx])
            task_trajectories[task_type]['indices'].append(batch_idx)
        
        # Collect success metrics from each task
        all_success = defaultdict(list)
        
        for task_type, data in task_trajectories.items():
            env_mgr = self.env_managers[task_type]
            indices = data['indices']
            
            # Extract rewards and lengths for this task
            task_rewards = episode_rewards[indices]
            task_lengths = episode_lengths[indices]
            
            # Get success metrics from task-specific evaluator
            task_success = env_mgr.success_evaluator(
                total_infos=data['infos'],
                total_batch_list=data['batches'],
                episode_rewards=task_rewards,
                episode_lengths=task_lengths,
            )
            
            # Aggregate metrics
            for key, values in task_success.items():
                # Expand values back to full batch indices
                for val in values:
                    all_success[key].append(val)
        
        # Convert to numpy arrays
        return {key: np.array(values) for key, values in all_success.items()}
    
    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        """
        Process a single batch - delegated to task-specific managers.
        This is called by success_evaluator if needed.
        """
        # Determine task type for this batch
        task_type = self.current_task_types[batch_idx]
        env_mgr = self.env_managers[task_type]
        
        # Delegate to task-specific manager
        if hasattr(env_mgr, '_process_batch'):
            env_mgr._process_batch(batch_idx, total_batch_list, total_infos, success)
