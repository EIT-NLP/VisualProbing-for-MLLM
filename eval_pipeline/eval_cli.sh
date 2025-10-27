#!/bin/bash
# 获取传入的参数
model_path=$1 # 传入完整的模型路径，绝对路径
datasets=$2 # 传入单个需要测试的数据集
logs_dir=$3 # 传入log需要保存的路径
output_results=$4 # 传入保存的结果的路径

# pip install openai
tasks1=("cvbench" "mmvet" "mnist")
tasks2=("mme" "ocrbench" "textvqa" "docvqa" "realworldqa" "pope" "mmbench" "gqa")
log_samples_suffix=$(echo "$model_path" | sed -E 's|.*/([^/]+/[^/]+)$|\1|')
MODEL_NAME=$log_samples_suffix
echo $MODEL_NAME
log_file="${logs_dir}/${log_samples_suffix}.log"  # 指定日志文件路径
# 检查并创建目录
log_dir=$(dirname "$log_file")
if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

start_time=$(date +%s)   # 记录开始时间

# 检查datasets字段是否在tasks1或tasks2中

found_in_tasks1=false
for task in "${tasks1[@]}"; do
    echo "$task"
    echo "$datasets"
    if [[ "$task" == "$datasets" ]]; then
       
        found_in_tasks1=true
        break
    fi
done



if [ "$found_in_tasks1" = true ]; then
    # 在tasks1中找到${datasets}
    echo "Dataset ${datasets} detected in tasks1."
    log_file="${log_file}_${tasks1}"
    echo "Start eval: ${log_samples_suffix}" | tee -a $log_file
    echo "Dataset ${datasets} detected in tasks1. Executing tool1..." | tee -a $log_file
    if [ "${datasets}" = "cvbench" ]; then
      
        python -m mobilevlm.eval.model_vqa_cvbench \
            --model-path "${model_path}" \
            --question-file  /code/chr/download/CV-Bench/test.jsonl \
            --image-folder  /code/chr/download/CV-Bench \
            --answers-file  "${output_results}/CVBench/answers/${MODEL_NAME}.jsonl" \
            --temperature 0 \
            --conv-mode v1 2>&1 | tee -a $log_file && 
            
        # # use gpt
        python /code/chr/eval_pipeline/tasks/cvbench/gpt4o_grader_batch_parallel.py \
            --answer_file "${output_results}/CVBench/answers/${MODEL_NAME}.jsonl" \
            --output_dir "${output_results}/CVBench/answers/${MODEL_NAME%%/*}" 2>&1 | tee -a $log_file ;
        
        # calculate
        python /code/chr/eval_pipeline/tasks/cvbench/cvbench_gpt_and_match.py \
            --json_path "${output_results}/CVBench/answers/${MODEL_NAME}_gpt.jsonl" \
            --result_path "${output_results}/CVBench/answers/${MODEL_NAME}.txt" 2>&1 | tee -a $log_file 
    fi

    if [ "${datasets}" = "mnist" ]; then
      
        python -m mobilevlm.eval.model_vqa_mnist \
            --model-path "${model_path}" \
            --question-file  /code/chr/eval_pipeline/tasks/MNIST/mnist_dataset.jsonl \
            --image-folder  /code/chr/eval_pipeline/tasks/MNIST/DataImages-Test \
            --answers-file  "${output_results}/MNIST/answers/${MODEL_NAME}.jsonl" \
            --temperature 0 \
            --conv-mode v1 2>&1 | tee -a $log_file &&

        python /code/chr/eval_pipeline/tasks/MNIST/mnist_match.py \
            --json_path /code/chr/eval_pipeline/tasks/MNIST/mnist_dataset.jsonl \
            --result_path "${output_results}/MNIST/answers/${MODEL_NAME}.jsonl"
    fi
    
    
    if [ "${datasets}" = "mmvet" ]; then
        python -m mobilevlm.eval.model_vqa_mmvet \
        --model-path "${model_path}" \
        --question-file /data/JoeyLin/llava/llava1-5/data/eval/mm-vet/llava-mm-vet.jsonl \
        --image-folder /code/chr/download/mm-vet/images \
        --answers-file "${output_results}/mm-vet/answers/${MODEL_NAME}.jsonl" \
        --temperature 0 \
        --conv-mode v1 | tee -a $log_file &&

        mkdir -p "${output_results}/mm-vet/results/${MODEL_NAME}" | tee -a $log_file

        python /code/chr/eval_pipeline/tasks/mmvet/convert_mmvet_for_eval.py \
            --src "${output_results}/mm-vet/answers/${MODEL_NAME}.jsonl" \
            --dst "${output_results}/mm-vet/results/${MODEL_NAME}.json" | tee -a $log_file

    fi

else
    log_file="${log_file}_tool2"
    echo "Start eval: ${log_samples_suffix}" | tee -a $log_file
    echo "Dataset ${datasets} detected in tasks2. Executing tool2..." | tee -a "$log_file"
    python3 -m accelerate.commands.launch \
            --num_processes=4 \
            -m lmms_eval \
            --model mobilevlm \
            --model_args pretrained="${model_path}" \
            --tasks ${datasets} \
            --batch_size 1 \
            --log_samples \
            --log_samples_suffix "${log_samples_suffix}" \
            --output_path "${output_results}" 2>&1| tee -a $log_file


fi


end_time=$(date +%s)  # 记录结束时间
elapsed_time=$((end_time - start_time))  # 计算运行时间
echo "End eval: ${log_samples_suffix}" | tee -a $log_file
echo "Model: $model_name took $elapsed_time seconds to run." | tee -a $log_file
