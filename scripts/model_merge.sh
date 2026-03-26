# python scripts/model_merger.py merge \
#     --backend fsdp \
#     --local_dir /data/home/zdhs0010/agentic/verl-agent-multi/ckpts/multitask_ori_opd_mask_0302_183558/global_step_80/actor \
#     --target_dir /data/home/zdhs0010/agentic/verl-agent-multi/ckpts/multitask_ori_opd_mask_0302_183558/global_step_80/actor_merged

# python scripts/model_merger.py merge \
#     --backend fsdp \
#     --local_dir /data/home/zdhs0010/agentic/verl-agent-multi/ckpts/multitask_ori_opd_mask_0302_183558/global_step_160/actor \
#     --target_dir /data/home/zdhs0010/agentic/verl-agent-multi/ckpts/multitask_ori_opd_mask_0302_183558/global_step_160/actor_merged

# python scripts/model_merger.py merge \
#     --backend fsdp \
#     --local_dir /data/home/zdhs0010/agentic/verl-agent-multi/ckpts/multitask_ori_opd_mask_0302_183558/global_step_240/actor \
#     --target_dir /data/home/zdhs0010/agentic/verl-agent-multi/ckpts/multitask_ori_opd_mask_0302_183558/global_step_240/actor_merged

python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir /data/home/zdhs0010/agentic/verl-agent-multi/ckpts/multitask_student_topk_0324_142026/global_step_400/actor \
    --target_dir /data/home/zdhs0010/agentic/verl-agent-multi/ckpts/multitask_student_topk_0324_142026/global_step_400/actor_merged