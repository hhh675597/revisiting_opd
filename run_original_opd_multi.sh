# in this experiment, we rollout one task at a time
# e.g., step1:math->step2:alfworld->step3:math->...

# only for testing multi-task dataload & rollout

ray stop --force
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=offline
export RAY_worker_register_timeout_seconds=600

TIME_STAMP=$(date +"%m%d_%H%M%S")
project_name='multitask_opd'
exp_name='multitask_ori_opd_mask'

set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

CKPTS_DIR=${CKPTS_DIR:-"${PWD}/ckpts/${exp_name}_${TIME_STAMP}"}

num_cpus_per_env_worker=0.1


train_data_size=16  
val_data_size=128
group_size=8

MULTITASK_DATA_DIR="/data/home/zdhs0010/agentic/verl-agent-multi/data/multitask_data_test"
TRAIN_DATA="${MULTITASK_DATA_DIR}/train.parquet"
VAL_DATA="${MULTITASK_DATA_DIR}/test_allat1.parquet"

STUDENT_MODEL="/data/home/zdhs0010/agentic/model/qwen-2.5-7b-it"
ALFWORLD_TEACHER="/data/home/zdhs0010/agentic/model/alfworld-teacher-gigpo-qwen2.5-7b"
MATH_TEACHER="/data/home/zdhs0010/agentic/model/OpenThinker3-7B"

python3 -m verl.trainer.main_ppo_multitask \
    algorithm.adv_estimator=opd \
    actor_rollout_ref.actor.kl_loss_type=k1 \
    +actor_rollout_ref.actor.opd_mask_special_tokens=True \
    +multitask.enable=True \
    +multitask.batching_mode=sequential \
    +multitask.tasks.task0.name=alfworld \
    +multitask.tasks.task0.env_name=alfworld/AlfredTWEnv \
    +multitask.tasks.task0.ref_model_path=${ALFWORLD_TEACHER} \
    +multitask.tasks.task0.eval_dataset=eval_in_distribution \
    +multitask.tasks.task0.max_response_length=512 \
    +multitask.tasks.task1.name=math \
    +multitask.tasks.task1.env_name=math \
    +multitask.tasks.task1.ref_model_path=${MATH_TEACHER} \
    +multitask.tasks.task1.max_response_length=16384 \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.train_batch_size=${train_data_size} \
    data.val_batch_size=${val_data_size} \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='middle' \
    data.return_raw_chat=True \
    +data.batching_mode=sequential \
    actor_rollout_ref.model.path=${STUDENT_MODEL} \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((2048 + 16384)) \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.0 \
    algorithm.use_kl_in_reward=True \
    env.env_name=multitask \
    env.seed=0 \
    env.max_steps=30 \
    env.rollout.n=${group_size} \
    env.resources_per_worker.num_cpus=${num_cpus_per_env_worker} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','mlflow'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=400 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    +trainer.visualize_distribution=false \
    +trainer.visualize_distribution_freq=3 \
    +trainer.visualize_distribution_samples=1 \
    +trainer.visualize_distribution_ref_tokens=3 \
    +trainer.visualize_distribution_dir="${CKPTS_DIR}/visualizations" \
    ray_init.num_cpus=96 \
    2>&1 | tee /data/home/zdhs0010/agentic/verl-agent-multi/data/logs/multitask/${exp_name}_${TIME_STAMP}.log
