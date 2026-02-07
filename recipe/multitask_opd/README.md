# Multi-Task On-Policy Distillation Training

This recipe implements multi-task on-policy distillation (OPD) training that supports training a single model on multiple tasks simultaneously (e.g., AlfWorld + Math).

## Overview

### Architecture

The multi-task training system consists of:

1. **MultiTaskDataset**: Combines datasets from different tasks with `task_type` field
2. **MultiEnvironmentManager**: Routes environment interactions to task-specific managers
3. **Task-specific Reference Models**: Each task can have its own teacher model for OPD _not implemented yet_

### Key Design Principles

1. **Sequential Batching** (& coming soon _mixed batching_): Each minibatch contains samples from only one task type
   - Simpler implementation
   - Better for tasks with different characteristics
   - Easier debugging

2. **Minimal Dataset for AlfWorld**: AlfWorld environment loads its own game files internally
   - Dataset only needs placeholder samples to trigger environment creation
   - Actual game data comes from AlfWorld's internal game file directory

3. **Full Dataset for Math**: Math environment uses `env_kwargs` from dataset
   - Dataset must contain actual problems and ground truth answers

## Quick Start

### 1. Prepare Multi-Task Training Data

```bash
# Prepare combined AlfWorld + Math dataset
python examples/data_preprocess/prepare_multitask_train_data.py \
    --alfworld_num 16 \
    --math_data_path /path/to/math/train.parquet \
    --output_dir /data/multitask_data \
    --batching_mode sequential \
    --shuffle
```

This creates:
- 16 AlfWorld placeholder samples (will spawn 16 × group_n parallel environments)
- All math samples from the input file
- Combined dataset with `task_type` field

### 2. Configure Multi-Task Training

Create a config file (e.g., `config/multitask_trainer.yaml`) or override via command line:

```yaml
# Enable multi-task training
multitask:
  enable: true
  batching_mode: sequential  # or "mixed"
  
  tasks:
    - name: alfworld
      env_name: alfworld/AlfredTWEnv
      ref_model_path: /path/to/alfworld/teacher/model
      eval_dataset: eval_in_distribution
    
    - name: math
      env_name: math
      ref_model_path: /path/to/math/teacher/model

# Update data paths
data:
  train_files: /data/multitask_data/train.parquet
  val_files: /data/multitask_data/val.parquet
  train_batch_size: 16
  batching_mode: sequential  # Must match multitask.batching_mode

# Environment config (applies to all tasks unless overridden)
env:
  env_name: multitask  # Special flag for multi-task mode
  rollout:
    n: 8  # Group size for environments
```

### 3. Run Training

```bash
python -m verl.trainer.main_ppo \
    multitask.enable=true \
    multitask.batching_mode=sequential \
    +multitask.tasks[0].name=alfworld \
    +multitask.tasks[0].env_name=alfworld/AlfredTWEnv \
    +multitask.tasks[0].ref_model_path=/path/to/alfworld/teacher \
    +multitask.tasks[1].name=math \
    +multitask.tasks[1].env_name=math \
    +multitask.tasks[1].ref_model_path=/path/to/math/teacher \
    data.train_files=/data/multitask_data/train.parquet \
    env.env_name=multitask
```

## Data Format Requirements

### AlfWorld Samples (Minimal)

```python
{
    "task_type": "alfworld",
    "data_source": "alfworld_placeholder",
    "prompt": [{"role": "user", "content": "placeholder"}],
    "env_kwargs": {"task_type": "alfworld"},
    "extra_info": {"index": 0, "split": "train"}
}
```

**Note**: AlfWorld environment ignores most fields and loads games from its internal directory (~8800 game files).

### Math Samples (Full Data)

```python
{
    "task_type": "math",
    "data_source": "dapo-math",
    "ability": "math",
    "prompt": [{"role": "user", "content": "What is 2+2?"}],
    "reward_model": {
        "style": "rule",
        "ground_truth": "4"
    },
    "env_kwargs": {
        "task_type": "math",
        "question": "What is 2+2?",
        "ground_truth": "4",
        "data_source": "dapo-math"
    },
    "extra_info": {
        "split": "train",
        "index": 0,
        "question": "What is 2+2?",
        "answer": "4"
    }
}
```

## How It Works

### Training Loop Flow

1. **Data Loading**:
   ```python
   # MultiTaskDataset loads combined data with task_type field
   batch = dataloader.next()
   # batch contains samples of one task type (sequential mode)
   ```

2. **Environment Reset**:
   ```python
   # MultiEnvironmentManager routes based on task_type
   obs, infos = multi_env_manager.reset(env_kwargs)
   # Internally:
   #   - AlfWorld samples from ~8800 internal game files
   #   - Math uses question/ground_truth from env_kwargs
   ```

3. **Rollout**:
   ```python
   # Model generates actions
   actions = model.generate(obs)
   
   # MultiEnvironmentManager routes to correct task env
   next_obs, rewards, dones, infos = multi_env_manager.step(actions)
   ```

4. **OPD Advantage Computation**:
   ```python
   # MultiRefPolicyManager routes to task-specific teacher
   ref_log_probs = multi_ref_policy.compute_ref_log_prob(batch)
   # Uses alfworld teacher for alfworld samples
   # Uses math teacher for math samples
   ```

5. **Policy Update**:
   ```python
   # Standard PPO update with OPD advantages
   loss = compute_ppo_loss(advantages, log_probs, ref_log_probs)
   ```

### Batching Modes

#### Sequential (Recommended)

- Each minibatch contains samples from only one task
- Tasks cycle through batches: AlfWorld → Math → AlfWorld → ...
- Simpler routing logic
- Better for tasks with different characteristics

#### Mixed

- Samples from different tasks mixed in one minibatch
- Requires careful index tracking
- More complex but potentially better data efficiency

## Environment-Specific Notes

### AlfWorld

- **Game Files**: Loads from `~/.cache/alfworld/`
- **Training Games**: ~3500 games in `json_2.1.1/train/`
- **Validation Games**: ~140 games in `json_2.1.1/valid_seen/`
- **Dataset Role**: Only triggers environment creation (16 samples → 16 × group_n environments)
- **Data Generation**: Fresh trajectories every epoch by sampling from game files

### Math

- **Data Source**: Parquet file contains all problems
- **Dataset Role**: Provides actual problem data via `env_kwargs`
- **Environment**: Uses question/ground_truth from dataset
- **Data Generation**: Single-turn interactions (question → solution → reward)

## Configuration Reference

```yaml
multitask:
  enable: true  # Must be true for multi-task training
  batching_mode: sequential  # "sequential" or "mixed"
  
  tasks:
    - name: <task_identifier>        # Used for routing (e.g., "alfworld", "math")
      env_name: <env_type>           # Environment type (e.g., "alfworld/AlfredTWEnv", "math")
      ref_model_path: <path>         # Path to task-specific teacher model
      eval_dataset: <dataset_name>   # (AlfWorld only) "eval_in_distribution" or "eval_out_of_distribution"

env:
  env_name: multitask  # Special flag to use make_multitask_envs()
  rollout:
    n: 8  # Number of parallel environments per batch sample
```

## Troubleshooting

### Error: "No environment manager found for task_type"

**Cause**: Dataset contains `task_type` not defined in config

**Solution**: Ensure all task_types in data match `multitask.tasks[*].name` in config

### Error: "Sample missing task_type in env_kwargs"

**Cause**: Dataset samples missing `task_type` field in `env_kwargs`

**Solution**: Re-run data preprocessing script to add `task_type` field

### AlfWorld not loading games

**Cause**: AlfWorld data not found in cache directory

**Solution**: 
```bash
# Download AlfWorld data
python -m alfworld.agents.environment.alfred_tw_env download
# Or set ALFWORLD_DATA environment variable
export ALFWORLD_DATA=/path/to/alfworld/data
```

### Batch size mismatch

**Cause**: Sequential batching requires consistent batch sizes

**Solution**: Ensure enough samples of each task type to form complete batches

## Performance Tips

1. **Balance Task Samples**: For sequential batching, balance the number of samples per task
2. **Adjust Group Size**: Set `env.rollout.n` based on available resources
3. **Batch Size**: Keep `train_batch_size` reasonable (16-32) for multi-task training


## Examples

See:
- `examples/data_preprocess/prepare_multitask_train_data.py` - Data preparation
- `recipe/multitask_opd/run_multitask_opd.sh` - Training script example
