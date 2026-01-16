ray stop --force
# ray start --head

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=offline
export RAY_worker_register_timeout_seconds=600

TIME_STAMP=$(date +"%m%d_%H%M%S")
project_name='opd'
exp_name='alfworld-opd-qwen2.5-7b-it'

set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

CKPTS_DIR=${CKPTS_DIR:-"${PWD}/ckpts/${exp_name}_${TIME_STAMP}"}

num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

train_data_size=16
val_data_size=128
group_size=8

# We only use data preparation to indicate the modality and the data size.
# python3 -m examples.data_preprocess.prepare \
#     --mode 'text' \
#     --train_data_size $train_data_size \
#     --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=opd \
    +algorithm.opd.gamma=0.0 \
    +algorithm.opd.reward_weight=0.0 \
    actor_rollout_ref.ref.model.path=/data/home/zdhs0086/hhh/verl-agent/models/alfworld-teacher-gigpo-qwen2.5-7b \
    data.train_files=/data/home/zdhs0086/hhh/verl-agent/data/verl-agent/text/train.parquet \
    data.val_files=/data/home/zdhs0086/hhh/verl-agent/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/data/home/zdhs0086/hhh/verl-agent/models/Qwen2.5-7B-Instruct \
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
    actor_rollout_ref.rollout.name=$ENGINE \
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
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
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
    2>&1 | tee /data/home/zdhs0086/hhh/verl-agent/data/logs/alfworld/${exp_name}_${TIME_STAMP}.log

# consider changing tp_size