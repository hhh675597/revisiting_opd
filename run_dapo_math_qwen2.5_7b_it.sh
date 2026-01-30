# use dapo to train a math teacher
# also test math environment compatibility

ray stop --force
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=offline
export RAY_worker_register_timeout_seconds=600

TIME_STAMP=$(date +"%m%d_%H%M%S")
project_name='opd'
exp_name='dapo-math-qwen2.5-7b-it'

set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

CKPTS_DIR=${CKPTS_DIR:-"${PWD}/ckpts/${exp_name}_${TIME_STAMP}"}

train_data_size=16
val_data_size=128
group_size=8

clip_ratio_low=0.2
clip_ratio_high=0.28
enable_filter_groups=True
max_num_gen_batches=10


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/data/home/zdhs0086/hhh/verl-agent/data/math_ori/train.parquet \
    data.val_files=/data/home/zdhs0086/hhh/verl-agent/data/math_ori/test.parquet \
    data.train_batch_size=${train_data_size} \
    data.val_batch_size=${val_data_size} \
    data.max_prompt_length=2048 \
    data.max_response_length=30720 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/data/home/zdhs0086/hhh/verl-agent/models/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((2048 + 30720)) \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.0 \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    env.env_name=math \
    env.seed=0 \
    env.rollout.n=${group_size} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','mlflow'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=1000 \
    trainer.val_before_train=True \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.resume_mode=auto \
    ray_init.num_cpus=96 \
    2>&1 | tee /data/home/zdhs0086/hhh/verl-agent/data/logs/math/${exp_name}_${TIME_STAMP}.log
