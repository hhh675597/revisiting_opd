python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir /data/home/zdhs0086/hhh/verl-agent/ckpts/multitask_k32_topp_no_mask_lr2e-6_0224_230632/global_step_400/actor \
    --target_dir /data/home/zdhs0086/hhh/verl-agent/ckpts/multitask_k32_topp_no_mask_lr2e-6_0224_230632/global_step_400/actor_merged