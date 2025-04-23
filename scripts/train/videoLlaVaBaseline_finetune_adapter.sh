export OMP_NUM_THREADS=8
# export NCCL_IB_DISABLE=1
export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
#export NCCL_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=enp226s0f0
export NCCL_DEBUG=INFO

# Base model and vision model names (adjust as needed)
LLM_VERSION="Qwen/Qwen2-7B-Instruct"
# LLM_VERSION="Qwen/Qwen2.5-1.5B-Instruct"
# VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

PROMPT_VERSION="qwen_1_5"

# Use a descriptive run name
BASE_RUN_NAME="videoLLaVaBaselinefinetune_only-adapters-${VISION_MODEL_VERSION//\//_}-${LLM_VERSION//\//_}"

# export WANDB_NAME=$BASE_RUN_NAME
# export WANDB_PROJECT=VideoEncoders

# wandb online


# KAREN PATHS
CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29501 \
    /datastor1/jiahuikchen/video_llava_encoder/llava/train/train_mem.py \
    --deepspeed /datastor1/jiahuikchen/video_llava_encoder/scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path /datastor1/jiahuikchen/video_llava_encoder/finetune.yaml \
    --video_folder /data/samyakp/llava_video_data \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts "mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --freeze_backbone True \
    --unfreeze_mm_vision_tower False \
    --tune_mm_mlp_adapter True \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(384, 768), (768, 384), (768, 768), (1152, 384), (384, 1152)]" \
    --bf16 True \
    --run_name ${BASE_RUN_NAME} \
    --output_dir "/data/jiahuic/vid_llava_checkpoints/${BASE_RUN_NAME}" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to wandb
exit 0;

    # --mm_patch_merge_type spatial_unpad \