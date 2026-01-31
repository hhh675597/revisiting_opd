#!/bin/bash
# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Multi-Task On-Policy Distillation Training
# Trains a single model on both AlfWorld and Math tasks

ray stop --force
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=offline
export RAY_worker_register_timeout_seconds=600

TIME_STAMP=$(date +"%m%d_%H%M%S")
project_name='multitask_opd'
exp_name='alfworld-math-opd-qwen2.5-7b-it'

set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

CKPTS_DIR=${CKPTS_DIR:-"${PWD}/ckpts/${exp_name}_${TIME_STAMP}"}

num_cpus_per_env_worker=0.1

# Batch configuration
train_data_size=16  # Number of samples per task type
val_data_size=128
group_size=8        # Parallel environments per sample

# Data paths
MULTITASK_DATA_DIR="/data/home/zdhs0086/hhh/verl-agent/data/multitask_opd"
TRAIN_DATA="${MULTITASK_DATA_DIR}/train.parquet"
VAL_DATA="${MULTITASK_DATA_DIR}/val.parquet"

# Model paths
STUDENT_MODEL="/data/home/zdhs0086/hhh/verl-agent/models/Qwen2.5-7B-Instruct"
ALFWORLD_TEACHER="/data/home/zdhs0086/hhh/verl-agent/models/alfworld-teacher-gigpo-qwen2.5-7b"
MATH_TEACHER="/data/home/zdhs0086/hhh/verl-agent/models/math-teacher-opd-qwen2.5-7b"

# Prepare multi-task data if not exists
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "Preparing multi-task training data..."
    python3 -m examples.data_preprocess.prepare_multitask_train_data \
        --alfworld_num ${train_data_size} \
        --math_data_path /data/home/zdhs0086/hhh/verl-agent/data/math_opd/train.parquet \
        --output_dir ${MULTITASK_DATA_DIR} \
        --batching_mode sequential \
        --shuffle
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=opd \
    +algorithm.opd.gamma=0.0 \
    +algorithm.opd.reward_weight=0.0 \
    multitask.enable=true \
    multitask.batching_mode=sequential \
    +multitask.tasks[0].name=alfworld \
    +multitask.tasks[0].env_name=alfworld/AlfredTWEnv \
    +multitask.tasks[0].ref_model_path=${ALFWORLD_TEACHER} \
    +multitask.tasks[0].eval_dataset=eval_in_distribution \
    +multitask.tasks[1].name=math \
    +multitask.tasks[1].env_name=math \
    +multitask.tasks[1].ref_model_path=${MATH_TEACHER} \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.train_batch_size=${train_data_size} \
    data.val_batch_size=${val_data_size} \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.batching_mode=sequential \
    actor_rollout_ref.model.path=${STUDENT_MODEL} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.0 \
    algorithm.use_kl_in_reward=True \
    env.env_name=multitask \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=${group_size} \
    env.resources_per_worker.num_cpus=${num_cpus_per_env_worker} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','mlflow'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    ray_init.num_cpus=96 \
    2>&1 | tee /data/home/zdhs0086/hhh/verl-agent/data/logs/multitask/${exp_name}_${TIME_STAMP}.log
