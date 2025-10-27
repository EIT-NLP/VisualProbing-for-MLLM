#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
# pip install openai ezcolorlog open_clip_torch
# 定义模型名称数组
model_names=(

    # layer from 1-24
    "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240722_120221/mobilevlm-linear-1.finetune"
    "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240723_223338/mobilevlm-linear-2.finetune"
    "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240721_185730/mobilevlm-linear-3.finetune"
    "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240725_133137/mobilevlm-linear-4.finetune"
    "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240725_220334/mobilevlm-linear-5.finetune"
    "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240721_172927/mobilevlm-linear-6.finetune"
    "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240726_111604/mobilevlm-linear-7.finetune"
    "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240726_193949/mobilevlm-linear-8.finetune"
    "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240721_102556/mobilevlm-linear-9.finetune"
    "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240727_125901/mobilevlm-linear-10.finetune"
    "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240727_220950/mobilevlm-linear-11.finetune"
    "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240721_081558/mobilevlm-linear-12.finetune"
    "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240728_064017/mobilevlm-linear-13.finetune"
    "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240728_150420/mobilevlm-linear-14.finetune"
    "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240722_044436/mobilevlm-linear-15.finetune"
    "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240729_192100/mobilevlm-linear-16.finetune"
    "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240724_161749/mobilevlm-linear-17.finetune"
    "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240721_010307/mobilevlm-linear-18.finetune"
    "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240723_130201/mobilevlm-linear-19.finetune"
    "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240725_005434/mobilevlm-linear-20.finetune"
    "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240722_145944/mobilevlm-linear-21.finetune"
    "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240724_070214/mobilevlm-linear-22.finetune"
    "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240722_233836/mobilevlm-linear-23.finetune"
    "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240722_015504/mobilevlm-linear-24.finetune"


    # # # CLIP+linear+mob+737k
    # "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240906_184553/linear-3-bts256-bts128_737k.finetune"
    # "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240906_095126/linear-18-bts256-bts128_737k.finetune"
    # "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240731_120805/linear-23-bts256-bts128_737k.finetune"
    # "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240910_051215/linear-24-bts256-bts128_737k.finetune"  
    
    # # CLIP+linear+mob+1M
    # "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240911_201712/linear-3-clip-mob14-1M.finetune"
    # "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240911_084941/linear-18-clip-mob14-1M.finetune"
    # "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240910_141136/linear-23-clip-mob14-1M.finetune"
    # "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240912_110401/linear-24-clip-mob14-1M.finetune"

    # mob2.7b linear + clip
    # "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240912_094017/linear-23-clip-mob27.finetune"
    # "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240916_195046/linear-18-clip-mob27.finetune"
    # "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20240917_105317/linear-24-clip-mob27.finetune"
    # "/code/chr/MobileVLM/checkpoint/mobilevlm_v2_1.7b_20241029_094612/linear-3-clip-mob27.finetune"
)
# 设置最大并行任务数量
# MAX_JOBS=1

# 获取当前后台运行任务数的函数
function running_jobs {
    jobs -rp | wc -l
}

# mme,ocrbench,textvqa,docvqa,realworldqa,pope,mmbench,gqa
# 第一个for循环并行执行，控制最大并行任务数量
for model_name in "${model_names[@]}"
do
    bash ./eval_cli.sh \
        $model_name \
        mme,ocrbench,textvqa,realworldqa,pope,mmbench,gqa,seedbench \
        /code/chr/eval_pipeline/eval_logs/emnlp \
        /code/chr/eval_pipeline/eval_results/emnlp &

    # 如果达到最大并行任务数，等待其中一个任务完成
    while [ $(running_jobs) -ge $MAX_JOBS ]
    do
        sleep 60  # 等待1秒后检查
    done
done &  # 把第一个for循环放入后台

MAX_JOBS=1
# 第二个for循环顺序执行并行化
for model_name in "${model_names[@]}"
do
    bash ./eval_cli.sh \
        $model_name \
        cvbench \
        /code/chr/eval_pipeline/eval_logs/emnlp \
        /code/chr/eval_pipeline/eval_results/emnlp &

    # 如果达到最大并行任务数，等待其中一个任务完成
    while [ $(running_jobs) -ge $MAX_JOBS ]
    do
        sleep 60  # 等待1秒后检查
    done
done &  # 把第二个for循环放入后台

# MAX_JOBS=1
# # 第三个for循环顺序执行并行化
# for model_name in "${model_names[@]}"
# do
#     bash ./eval_cli.sh \
#         $model_name \
#         mmvet \
#         /code/chr/eval_pipeline/eval_logs/convnext \
#         /code/chr/eval_pipeline/eval_results/convnext &
#     # 如果达到最大并行任务数，等待其中一个任务完成
#     while [ $(running_jobs) -ge $MAX_JOBS ]
#     do
#         sleep 60  # 等待1秒后检查
#     done
# done &  # 把第三个for循环放入后台

# MAX_JOBS=1
# # 第三个for循环顺序执行并行化
# for model_name in "${model_names[@]}"
# do
#     bash ./eval_cli.sh \
#         $model_name \
#         mnist \
#         /code/chr/eval_pipeline/eval_logs/convnext \
#         /code/chr/eval_pipeline/eval_results/convnext &
#     # 如果达到最大并行任务数，等待其中一个任务完成
#     while [ $(running_jobs) -ge $MAX_JOBS ]
#     do
#         sleep 60  # 等待1秒后检查
#     done
# done &  # 把第三个for循环放入后台

# 等待所有后台任务完成
wait