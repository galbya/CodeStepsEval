#!/bin/bash

# Define the paths to the models
base_model_paths=(
    "/home/ubuntu/hf_local_models/CodeLlama-7b-hf"
    "/home/ubuntu/hf_local_models/deepseek-coder-6.7b-base"
    "/home/ubuntu/hf_local_models/Llama-3-8B"
    "/home/ubuntu/hf_local_models/CodeQwen1.5-7B"
)

# Define the path to the data file
data_path="./data/dataset/CodeStepsEval277_CPP_for_code_generation.jsonl"

# Define the sample size and max tokens
sample_size=100
max_tokens=800
temperature=0.8
gpu_memory_utilization=0.9
enable_prefix_caching=True
max_model_len=4096
batch_size=20
max_num_seqs=256

output_dir="./generated_data/generated_codes/CodeStepsEval277"

# Loop through each model path
for model_path in "${base_model_paths[@]}"; do
    # 设置输出路径
    model_name=$(basename $model_path)
    output_path="${output_dir}/CodeStepsEval277_CPP_${model_name}_temp${temperature}_topp0.95_num${sample_size}_max${max_tokens}_code_solution.jsonl"
    
    # 打印当前正在处理的模型
    echo "Processing model: $model_name"
    
    # 调用 Python 脚本
    python3 code_generation.py \
        --data_path "$data_path" \
        --model_path "$model_path" \
        --sample_size $sample_size \
        --max_tokens $max_tokens \
        --temperature $temperature \
        --gpu_memory_utilization $gpu_memory_utilization \
        --enable_prefix_caching $enable_prefix_caching \
        --max_model_len $max_model_len \
        --batch_size $batch_size \
        --output_path "$output_path" \
        --max_num_seqs $max_num_seqs
done

echo "All models processed successfully."
