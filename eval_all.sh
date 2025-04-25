CHECKPOINT="/data/jiahuic/vid_llava_checkpoints/CLIP_MLCD_multiEncoder_finetune_only-adapters-multi_image_encoder-Qwen_Qwen2-7B-Instruct/checkpoint-6000/mm_projector.bin"
VISION_TOWER="multi_image_encoder"

DATASETS=(
  "mvbench_action_localization"
  "mvbench_egocentric_navigation"
  "mvbench_moving_direction"
  "mvbench_moving_count"
  "vinoground_textscore_subset"
  "vinoground_videoscore_subset"
  "temporalbench_subset"
)

for DATASET in "${DATASETS[@]}"; do
  echo "=========================================="
  echo " Evaluating on dataset: ${DATASET}"
  echo " Using checkpoint: ${CHECKPOINT}"
  echo "=========================================="
  
  CUDA_VISIBLE_DEVICES=1 python evals.py \
    --vision-tower="${VISION_TOWER}" \
    --checkpoint-dir="${CHECKPOINT}" \
    --dataset-name="${DATASET}"
done

CHECKPOINT="/data/jiahuic/vid_llava_checkpoints/videoLlaVaCLIPBaselinefinetune_only-adapters-openai_clip-vit-base-patch16-Qwen_Qwen2-7B-Instruct/checkpoint-6000/mm_projector.bin"
VISION_TOWER="openai/clip-vit-base-patch16"

for DATASET in "${DATASETS[@]}"; do
  echo "=========================================="
  echo " Evaluating on dataset: ${DATASET}"
  echo " Using adapter checkpoint: ${CHECKPOINT}"
  echo "=========================================="
  
  CUDA_VISIBLE_DEVICES=1 python evals.py \
    --vision-tower="${VISION_TOWER}" \
    --checkpoint-dir="${CHECKPOINT}" \
    --dataset-name="${DATASET}"
done

echo "All evaluations complete."