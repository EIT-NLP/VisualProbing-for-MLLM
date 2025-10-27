#!/bin/bash
# Function to get GPU memory usage and utilization
### LLM
LANGUAGE_MODEL=/code/chr/download/MobileLLaMA-1.4B-Base
# LANGUAGE_MODEL=/code/chr/download/MobileLLaMA-2.7B-Base
# LANGUAGE_MODEL=/code/chr/download/vicuna-7b-v1.5
#################################################


### Visual Encoder
VISION_MODEL=/model/BJiao/openaiclip-vit/openaiclip-vit-large-patch14-336
# VISION_MODEL=/code/chr/download/dinov2-large
# VISION_MODEL="/code/chr/download/siglip-so400m-patch14-384"
# VISION_MODEL=/code/chr/download/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup
################################################

get_gpu_status() {
  # Extract memory usage and utilization from nvidia-smi output for the first GPU
  memory_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | awk '{print $1}')
  utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1 | awk '{print $1}')
  
  echo $memory_free $utilization
}

# Main loop to monitor GPU status
while true; do
  read memory_free utilization <<< $(get_gpu_status)
  echo $memory_free $utilization 
  # Check if memory is less than 100MB or utilization is less than 1%
  if (( memory_free > 70000 )) && (( utilization < 10 )); then
    # You may need to modifiy here to train different layers
    bash run_layer18.sh mobilevlm_v2_1.7b pretrain-finetune ${LANGUAGE_MODEL} ${VISION_MODEL}
    echo "Finished!"
    break
    
  fi
  echo "Monitoring GPU usage..."
  # Wait for 5 minutes before the next check
  sleep 600
done