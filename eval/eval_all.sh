# # List of (CHECKPOINT|VISION_TOWER) pairs
# CHECKPOINTS=(
#   # SigLIP + MLCD and CLIP + Dino
#   "/data/jiahuic/vid_llava_checkpoints/SigLIP_MLCD_multiEncoder_finetune_only-adapters-multi_image_siglip_mlcd-Qwen_Qwen2-7B-Instruct/mm_projector.bin|multi_image_siglip_mlcd" 
#   "/data/jiahuic/vid_llava_checkpoints/CLIP_DinoV2_multiEncoder_finetune_only-adapters-multi_image_clip_dino-Qwen_Qwen2-7B-Instruct/mm_projector.bin|multi_image_clip_dino" 
#   # # Apr 27 eval (all image encoders except for CLIP + Dino and SigLIP + MLCD)
#   # "/data/jiahuic/vid_llava_checkpoints/max32768_videoLlaVaCLIPBaselinefinetune_only-adapters-openai_clip-vit-base-patch16-Qwen_Qwen2-7B-Instruct/mm_projector.bin|openai/clip-vit-base-patch16"
#   # "/data/jiahuic/vid_llava_checkpoints/videoLlaVa14PatchCLIPBaselinefinetune_only-adapters-openai_clip-vit-large-patch14-336-Qwen_Qwen2-7B-Instruct/mm_projector.bin|openai/clip-vit-large-patch14-336"
#   # "/data/jiahuic/vid_llava_checkpoints/14PatchCLIP_MLCD_multiEncoder_finetune_only-adapters-multi_image-Qwen_Qwen2-7B-Instruct/mm_projector.bin|multi_image_clip_mlcd"
#   # "/data/jiahuic/vid_llava_checkpoints/videoLLaVaSigLIPBaselinefinetune_only-adapters-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct/mm_projector.bin|google/siglip-so400m-patch14-384"
#   # "/data/jiahuic/vid_llava_checkpoints/SigLIP_CLIP_multiEncoder_finetune_only-adapters-multi_image_siglip_clip-Qwen_Qwen2-7B-Instruct/mm_projector.bin|multi_image_siglip_clip"
#   # "/data/jiahuic/vid_llava_checkpoints/SigLIP_CLIP_MLCD_multiEncoder_finetune_only-adapters-multi_image_siglip_clip_mlcd-Qwen_Qwen2-7B-Instruct/mm_projector.bin|multi_image_siglip_clip_mlcd"
#   # "/data/jiahuic/vid_llava_checkpoints/SigLIP_DinoV2_multiEncoder_finetune_only-adapters-multi_image_siglip_dino-Qwen_Qwen2-7B-Instruct/mm_projector.bin|multi_image_siglip_dino"
# )

# DATASETS=(
#   "mvbench_action_localization"
#   "mvbench_egocentric_navigation"
#   "mvbench_moving_direction"
#   "mvbench_moving_count"
#   "vinoground_textscore_subset"
#   "vinoground_videoscore_subset"
#   "temporalbench_subset"
# )

# for CKPT_PAIR in "${CHECKPOINTS[@]}"; do
#   # Split the tuple into CHECKPOINT and VISION_TOWER
#   IFS='|' read -r CHECKPOINT VISION_TOWER <<< "$CKPT_PAIR"

#   for DATASET in "${DATASETS[@]}"; do
#     echo "=========================================="
#     echo " Evaluating on dataset: ${DATASET}"
#     echo " Using checkpoint: ${CHECKPOINT}"
#     echo " Using vision tower: ${VISION_TOWER}"
#     echo "=========================================="

#     CUDA_VISIBLE_DEVICES=4,5,6,7 python evals.py \
#       --vision-tower="${VISION_TOWER}" \
#       --checkpoint-dir="${CHECKPOINT}" \
#       --dataset-name="${DATASET}"
#   done
# done


#CHECKPOINT="/data/jiahuic/vid_llava_checkpoints/CLIP_MLCD_multiEncoder_finetune_only-adapters-multi_image_encoder-Qwen_Qwen2-7B-Instruct/checkpoint-6000/mm_projector.bin"
CHECKPOINT="/data/jiahuic/vid_llava_checkpoints/internVideo2Global_videoLLaVA_finetune_only-adapters-video_embedding_internVideo2_global-Qwen_Qwen2-7B-Instruct/checkpoint-9000/mm_projector.bin"
# VISION_TOWER="multi_image_encoder"
VISION_TOWER="internVideo2_global_embedding"

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
  JSON_PATH="/data/samyakp/llava_video_data/eval_videos/${DATASET}_with_internVideo2_global_embeddings.json"

  CUDA_VISIBLE_DEVICES=1 python evals.py \
    --vision-tower="${VISION_TOWER}" \
    --checkpoint-dir="${CHECKPOINT}" \
    --dataset-name="${DATASET}" \
    --json-path="${JSON_PATH}"
done

# CHECKPOINT="/data/jiahuic/vid_llava_checkpoints/videoLlaVaCLIPBaselinefinetune_only-adapters-openai_clip-vit-base-patch16-Qwen_Qwen2-7B-Instruct/checkpoint-6000/mm_projector.bin"
# VISION_TOWER="openai/clip-vit-base-patch16"

# for DATASET in "${DATASETS[@]}"; do
#   echo "=========================================="
#   echo " Evaluating on dataset: ${DATASET}"
#   echo " Using adapter checkpoint: ${CHECKPOINT}"
#   echo "=========================================="
  
#   CUDA_VISIBLE_DEVICES=7 python evals.py \
#     --vision-tower="${VISION_TOWER}" \
#     --checkpoint-dir="${CHECKPOINT}" \
#     --dataset-name="${DATASET}"
# done

echo "All evaluations complete."