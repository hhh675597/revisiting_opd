total_size=128

python examples/data_preprocess/prepare_multitask_data.py \
    --alfworld_num ${total_size} \
    --math_data_path /data/home/zdhs0086/hhh/verl-agent/data/math_ori/test.parquet \
    --output_dir /data/home/zdhs0086/hhh/verl-agent/data/multitask_data_test \
    --batching_mode sequential
