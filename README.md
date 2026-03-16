# verl-agent-multi

Multi-task on-policy distillation (OPD) training for LLM agents, built on top of [verl-agent](https://github.com/langfengQ/verl-agent). This codebase supports single-task and multi-task OPD with both the **original OPD** objective and our **Teacher-TopK** objective.

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
  - [Math Dataset](#math-dataset)
  - [Multi-Task Dataset (ALFWorld + Math)](#multi-task-dataset-alfworld--math)
- [Single-Task Training](#single-task-training)
  - [Math](#math)
  - [ALFWorld](#alfworld)
- [Multi-Task Training](#multi-task-training)
- [Configuring the Training Objective](#configuring-the-training-objective)
  - [Original OPD](#original-opd)
  - [Teacher-TopK (Ours)](#teacher-topk-ours)
  - [GRPO Baseline (Multi-Task)](#grpo-baseline-multi-task)
- [Evaluation](#evaluation)
- [Model Merging](#model-merging)
- [Key Configuration Reference](#key-configuration-reference)

---

## Installation

```bash
conda create -n verl-agent python==3.10 -y
conda activate verl-agent

pip3 install vllm==0.8.5
pip3 install flash-attn==2.7.0.post2 --no-build-isolation --no-cache-dir
pip install -e .
```

For ALFWorld environments, also install:

```bash
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
pip install alfworld
alfworld-download -f
```

---

## Data Preparation

### Math Dataset

Prepare the math evaluation set (AIME 2024):

```bash
python examples/data_preprocess/verl_agent_math.py \
    --local_dir ./data/math_opd
```

This saves `test.parquet` to the specified directory. Training data should be a parquet file in the same format (with `prompt`, `reward_model`, `env_kwargs` columns).

### Multi-Task Dataset (ALFWorld + Math)

For multi-task training, combine ALFWorld placeholder samples and math problem samples into a single dataset:

```bash
python examples/data_preprocess/prepare_multitask_data.py \
    --alfworld_num 128 \
    --math_data_path /path/to/math/train.parquet \
    --output_dir ./data/multitask_data \
    --batching_mode sequential
```

- `--alfworld_num`: number of ALFWorld placeholder samples (ALFWorld loads its own game files internally, so these are just triggers).
- `--math_data_path`: path to the math training parquet.
- `--batching_mode`: `sequential` (recommended) groups samples by task type so each training step handles one task at a time.

---

## Single-Task Training

### Math

Train a student model on math using OPD with a math teacher:

```bash
bash run_opd_math_qwen2.5-7b_it.sh
```

Key variables to configure in the script:

```bash
STUDENT_MODEL="/path/to/Qwen2.5-7B-Instruct"
MATH_TEACHER="/path/to/OpenThinker3-7B"           # teacher / reference model
TRAIN_DATA="/path/to/math_opd/train.parquet"
VAL_DATA="/path/to/math_opd/test.parquet"
```

The entry point for single-task is `verl.trainer.main_ppo_multitask` (also used for multi-task). Single-task math uses `env.env_name=math`.

### ALFWorld

Train a student on ALFWorld with an ALFWorld-specific teacher:

```bash
bash run_opd_alfworld_qwen2.5_7b_it.sh
```

Key variables:

```bash
STUDENT_MODEL="/path/to/Qwen2.5-7B-Instruct"
ALFWORLD_TEACHER="/path/to/alfworld-teacher-gigpo-qwen2.5-7b"  # reference model
```

The single-task ALFWorld entry point uses `verl.trainer.main_ppo` with `env.env_name=alfworld/AlfredTWEnv`.

---

## Multi-Task Training

Multi-task training alternates between tasks in each batch (sequential batching). The trainer dynamically swaps the reference model per task.

```bash
bash run_opd_multitask_qwen2.5-7b_it.sh
```

The multi-task config block in the `.sh` file defines per-task settings:

```bash
+multitask.enable=True
+multitask.batching_mode=sequential

# Task 0: ALFWorld
+multitask.tasks.task0.name=alfworld
+multitask.tasks.task0.env_name=alfworld/AlfredTWEnv
+multitask.tasks.task0.ref_model_path=${ALFWORLD_TEACHER}
+multitask.tasks.task0.eval_dataset=eval_in_distribution
+multitask.tasks.task0.max_response_length=512

# Task 1: Math
+multitask.tasks.task1.name=math
+multitask.tasks.task1.env_name=math
+multitask.tasks.task1.ref_model_path=${MATH_TEACHER}
+multitask.tasks.task1.max_response_length=16384
```

- Each task specifies its own `ref_model_path` (teacher), `env_name`, and `max_response_length`.
- `env.env_name=multitask` tells the trainer to dispatch to the appropriate environment per sample.
- `data.batching_mode=sequential` ensures each step trains on one task at a time (e.g., step 1: math, step 2: alfworld, step 3: math, ...).

---

## Configuring the Training Objective

### Original OPD

The original on-policy distillation uses token-level KL divergence between the student and teacher as the advantage signal. It is controlled by `algorithm.adv_estimator=opd`.

Example script: `run_original_opd_math_qwen2.5_7b_it.sh`

Key flags for original OPD:

```bash
python3 -m verl.trainer.main_ppo_multitask \
    algorithm.adv_estimator=opd \
    actor_rollout_ref.actor.kl_loss_type=k1 \
    +actor_rollout_ref.actor.opd_mask_special_tokens=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    algorithm.use_kl_in_reward=True \
    actor_rollout_ref.ref.model.path=${TEACHER_MODEL} \
    ...
```

| Flag | Value | Meaning |
|------|-------|---------|
| `algorithm.adv_estimator` | `opd` | Use OPD advantage estimator (KL-based advantages) |
| `actor_rollout_ref.actor.kl_loss_type` | `k1` | Token-level KL approximation: `k1` = KL₁ = `exp(ref - student) - (ref - student) - 1` |
| `actor_rollout_ref.actor.use_kl_loss` | `False` | No separate KL regularization loss term (KL is already in the advantage) |
| `algorithm.use_kl_in_reward` | `True` | Fold KL penalty into the reward signal |
| `+actor_rollout_ref.actor.opd_mask_special_tokens` | `True/False` | Whether to mask special tokens (e.g., `<think>`, `</think>`) from the KL computation |

### Teacher-TopK (Ours)

Our Teacher-TopK objective replaces the OPD advantage with a memory-efficient full reverse-KL loss computed over only the teacher's top-K token positions. Since the loss is applied directly (not via advantages), `algorithm.adv_estimator` is set to `placeholder`.

Example script: `run_opd_math_qwen2.5-7b_it.sh`

Key flags for Teacher-TopK:

```bash
python3 -m verl.trainer.main_ppo_multitask \
    algorithm.adv_estimator=placeholder \
    actor_rollout_ref.actor.kl_loss_type=full_reverse \
    +actor_rollout_ref.actor.kl_topk_tokens=32 \
    +actor_rollout_ref.actor.norm_to_one_for_kl=True \
    +actor_rollout_ref.actor.clip_log_ratio=False \
    +actor_rollout_ref.actor.opd_mask_special_tokens=False \
    +actor_rollout_ref.actor.entropy_top_ratio=0.2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=1 \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.ref.model.path=${TEACHER_MODEL} \
    ...
```

| Flag | Value | Meaning |
|------|-------|---------|
| `algorithm.adv_estimator` | `placeholder` | No RL advantage computation; advantages are zeroed out |
| `actor_rollout_ref.actor.kl_loss_type` | `full_reverse` | Full reverse KL divergence (KL(teacher \|\| student)) computed over top-K logits |
| `+actor_rollout_ref.actor.kl_topk_tokens` | `32` | Number of top-K token positions (selected by the teacher) to compute KL over |
| `+actor_rollout_ref.actor.norm_to_one_for_kl` | `True` | Normalize the top-K probability slice to sum to 1 before KL computation |
| `+actor_rollout_ref.actor.clip_log_ratio` | `False` | Whether to clip log-probability ratios in KL |
| `+actor_rollout_ref.actor.entropy_top_ratio` | `0.2` | Only train on the top 20% highest-entropy tokens per sample (entropy-based masking) |
| `actor_rollout_ref.actor.use_kl_loss` | `True` | Enable the KL loss term in the actor update |
| `actor_rollout_ref.actor.kl_loss_coef` | `1` | Weight of the KL loss |
| `algorithm.use_kl_in_reward` | `False` | KL is applied as a direct loss, not folded into rewards |

### GRPO Baseline (Multi-Task)

For comparison, you can also run multi-task training with GRPO (reward-based advantage, no distillation):

```bash
bash run_grpo_multitask_qwen2.5-7b_it.sh
```

Key differences from OPD:

```bash
algorithm.adv_estimator=grpo
actor_rollout_ref.ref.model.path=${STUDENT_MODEL}   # ref = student itself (no teacher)
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=1e-3            # small KL regularization
```

---

## Evaluation

Use the evaluation script to run validation-only mode on a trained checkpoint:

```bash
bash evaluate_multi.sh
```

Configure the evaluation by setting:

```bash
STUDENT_MODEL="/path/to/checkpoint/actor_merged"  # merged checkpoint path
trainer.val_before_train=True
trainer.val_only=True
```

The evaluation script supports the same multi-task config block, so you can evaluate on both ALFWorld and math simultaneously.

---

## Model Merging

After training with FSDP, merge sharded checkpoints into a single model directory:

```bash
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir /path/to/ckpts/global_step_XXX/actor \
    --target_dir /path/to/ckpts/global_step_XXX/actor_merged
```

The merged model can then be loaded as `STUDENT_MODEL` for evaluation or further training.

---

## Key Configuration Reference

### Common Settings

| Config | Description | Typical Value |
|--------|-------------|---------------|
| `data.train_batch_size` | Number of prompts per training step | `16` |
| `data.val_batch_size` | Number of prompts per validation step | `128` |
| `env.rollout.n` | Group size (rollouts per prompt) | `8` |
| `data.max_prompt_length` | Maximum prompt token length | `2048` |
| `data.max_response_length` | Maximum response token length | `16384` (math) / `512` (alfworld) |
| `data.batching_mode` | How to batch multi-task data | `sequential` |
| `actor_rollout_ref.actor.optim.lr` | Learning rate | `1e-6` to `2e-6` |
| `actor_rollout_ref.rollout.top_p` | Nucleus sampling top-p during rollout | `0.9` |
| `trainer.n_gpus_per_node` | GPUs per node | `8` |
| `trainer.save_freq` | Checkpoint save frequency (steps) | `80` |
| `trainer.test_freq` | Validation frequency (steps) | `80` |

### OPD-Specific

| Config | Description |
|--------|-------------|
| `algorithm.adv_estimator` | `opd` for original OPD, `placeholder` for Teacher-TopK |
| `algorithm.use_kl_in_reward` | `True` for original OPD, `False` for Teacher-TopK |
| `actor_rollout_ref.actor.kl_loss_type` | `k1` (original OPD) or `full_reverse` (Teacher-TopK) |
| `+actor_rollout_ref.actor.kl_topk_tokens` | Top-K token count for Teacher-TopK (e.g., `32`) |
| `+actor_rollout_ref.actor.entropy_top_ratio` | Fraction of highest-entropy tokens to train on (e.g., `0.2`) |
| `+actor_rollout_ref.actor.norm_to_one_for_kl` | Normalize top-K probabilities before KL |
| `+actor_rollout_ref.actor.opd_mask_special_tokens` | Mask special tokens from KL computation |

### Multi-Task Specific

| Config | Description |
|--------|-------------|
| `+multitask.enable` | Enable multi-task mode |
| `+multitask.batching_mode` | `sequential` (one task per step) |
| `+multitask.tasks.taskN.name` | Task name (e.g., `alfworld`, `math`) |
| `+multitask.tasks.taskN.env_name` | Environment name for this task |
| `+multitask.tasks.taskN.ref_model_path` | Path to task-specific teacher model |
| `+multitask.tasks.taskN.max_response_length` | Max response length for this task |
| `+multitask.tasks.taskN.eval_dataset` | Evaluation dataset name (optional) |
| `env.env_name` | Set to `multitask` for multi-task training |
