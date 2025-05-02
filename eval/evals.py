import copy
import numpy as np
import torch
import pandas as pd
from datasets import load_dataset
from decord import VideoReader, cpu
from PIL import Image
from transformers import AutoTokenizer

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import tokenizer_image_token
from llava.conversation import conv_templates
from llava.model.language_model.llava_qwen import (
    LlavaQwenForCausalLM,
    LlavaQwenConfig,
)
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
import argparse
import os
import json

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None, metadata={"help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"})

    mm_tunable_parts: Optional[str] = field(
        default=None, metadata={"help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_mlp_adapter,mm_language_model"'}
    )
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)
    unfreeze_mm_vision_tower: bool = field(default=False)
    unfreeze_language_model: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.0)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_perceiver_depth: Optional[int] = field(default=3)
    mm_perceiver_latents: Optional[int] = field(default=32)
    mm_perceiver_ff_mult: Optional[float] = field(default=4)
    mm_perceiver_pretrained: Optional[str] = field(default=None)
    mm_qformer_depth: Optional[int] = field(default=3)
    mm_qformer_latents: Optional[int] = field(default=32)
    mm_qformer_pretrained: Optional[str] = field(default=None)

    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)

    s2: Optional[bool] = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")

    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)


    mm_newline_position: Optional[str] = field(default="grid")
    delay_load: Optional[bool] = field(default=True)
    add_faster_video: Optional[bool] = field(default=False)
    faster_token_stride: Optional[int] = field(default=10)


# def sample_frames(video_path: str, num_frames: int) -> np.ndarray:
#     vr   = VideoReader(video_path, ctx=cpu(0))
#     idxs = np.linspace(0, len(vr)-1, num_frames, dtype=int)
#     return vr.get_batch(idxs).asnumpy()   

# Uniform frame sampling
def sample_frames(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)


def parse_args():
    p = argparse.ArgumentParser(description="Run LLaVA-Qwen video VQA on a HuggingFace dataset split")
    p.add_argument("--vision-tower", required=True, choices=[
        "openai/clip-vit-base-patch16", 
        "google/siglip-so400m-patch14-384",
        # "multi_image_encoder", # Can only run from commit 1ce46bb34081f187e1f2107b79656daa54964d5d or earlier
        "openai/clip-vit-large-patch14-336",
        "multi_image_clip_mlcd",
        "multi_image_clip_dino",
        "multi_image_siglip_clip",
        "multi_image_siglip_mlcd",
        "multi_image_siglip_clip_mlcd",
        "multi_image_siglip_dino",
        "multi_image_clip_dino",
        "multi_image_siglip_mlcd",
        "video_embedding",
        "internVideo2_global_embedding",
        "internVideo2_patch_embedding",
        # "video_embedding_videoMAE_global",
        "video_embedding_videoMAE",
        "video_embedding_videoMAE_patch",
        "video_embedding_concat_global",
        "video_embedding_concat_patch",
        "videoMAE_patch_with_siglip",
        "internVideo2_patch_with_siglip"
    ])
    p.add_argument("--checkpoint-dir", required=True, default="/data/jiahuic/vid_llava_checkpoints/CLIP_MLCD_multiEncoder_finetune_only-adapters-multi_image_encoder-Qwen_Qwen2-7B-Instruct/checkpoint-6000/mm_projector.bin")
    p.add_argument("--dataset-name", required=True, choices=["mvbench_action_localization", "mvbench_egocentric_navigation", 
                                                            "mvbench_moving_direction", "mvbench_moving_count",
                                                            "vinoground_textscore_subset", "vinoground_videoscore_subset", "temporalbench_subset"])
    p.add_argument("--json-path")
    p.add_argument("--num-frames", type=int, default=16)
    return p.parse_args()

PRETRAINED      = "Qwen/Qwen2-7B-Instruct"
SPLIT           = "train"
PROMPT_VERSION  = "qwen_1_5"
DEVICE          = "cuda"
VIDEO_PATHS = {"mvbench_action_localization": "/data/samyakp/llava_video_data/sta/sta_video/",
                "mvbench_egocentric_navigation": "/data/samyakp/llava_video_data/vlnqa/",
                "mvbench_moving_direction": "/data/samyakp/llava_video_data/clevrer/video_validation/",
                "mvbench_moving_count": "/data/samyakp/llava_video_data/clevrer/video_validation/",
                "vinoground_textscore_subset":"/data/samyakp/llava_video_data/",
                "vinoground_videoscore_subset":"/data/samyakp/llava_video_data/",
                "temporalbench_subset":"/data/samyakp/llava_video_data/"}

def simulate_patches(emb: torch.Tensor, num_patches: int = 100) -> torch.Tensor:
    """
    Returns a tensor of shape [num_patches, C], where each row is the same vector.
    """
    if emb.ndim == 1:
        vec = emb
    else:
        vec = emb.reshape(-1, emb.shape[-1])[0] 

    patches = vec.unsqueeze(0).repeat(num_patches, 1) 
    return patches

def upsample_patch_embeds(internvideo2_emb, videomae_emb):
    """
    videoMAE patch feature size: torch.Size([1568, 768])
    InternVideo2 patch feature size: torch.Size([1025, 1408])
    
    Same upsample as in training (see LazySupervisedDataset in train.py)
    on eval video embeddings
    """
    # Upsample InternVideo2 patches to match videoMAE
    internvideo2_emb_batched = internvideo2_emb.permute(1, 0).unsqueeze(0)  
    internvideo2_upsample = torch.nn.functional.interpolate(internvideo2_emb_batched, size=videomae_emb.shape[0], mode='linear', align_corners=False)
    internvideo2_emb = internvideo2_upsample.squeeze(0).permute(1, 0)

    return internvideo2_emb 


def main():
    args = parse_args()

    output_dir = f"output_csvs/{args.vision_tower.replace('/', '_')}"
    os.makedirs(output_dir, exist_ok=True)
    if 'checkpoint' in args.checkpoint_dir:
        ckpt_base      = args.checkpoint_dir.split("/")[-2].replace("checkpoint-", "")
        output_csv     = f"{output_dir}/{args.dataset_name}_{ckpt_base}.csv"
    else:
        output_csv     = f"{output_dir}/{args.dataset_name}.csv"

    cfg = LlavaQwenConfig.from_pretrained(PRETRAINED, trust_remote_code=True)
    
    vision_tower = args.vision_tower
    cfg.model_type               = "llava_qwen"
    cfg.vision_tower             = vision_tower
    cfg.mm_use_im_patch_token    = False
    cfg.mm_use_im_start_end      = False
    cfg.mm_patch_merge_type      = "flat"
    cfg.mm_spatial_pool_mode     = "average"
    cfg.mm_spatial_pool_stride   = 1
    cfg.tokenizer_model_max_length = 4096
    cfg.tokenizer_padding_side     = "right"

    model = LlavaQwenForCausalLM.from_pretrained(PRETRAINED, config=cfg,trust_remote_code=True, low_cpu_mem_usage=True).eval()

    model_args = ModelArguments(vision_tower=vision_tower)
    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=None)

    model = model.half().to(DEVICE)
    adapter_sd = torch.load(args.checkpoint_dir, map_location="cpu")
    proj_sd    = {k.replace("model.mm_projector.", ""): v for k,v in adapter_sd.items()}
    model.get_model().mm_projector.load_state_dict(proj_sd, strict=True)


    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED, trust_remote_code=True)
    tokenizer.padding_side = "right"
    processor = model.get_model().get_vision_tower().image_processor

    if "video" in args.vision_tower.lower():
        with open(args.json_path) as f:
            ds = json.load(f)

    else:
        ds = load_dataset("Nguyencent/CS381V-hardest-vqa", args.dataset_name)[SPLIT]

    records = []
    for ex in tqdm(ds):
        if "mvbench" in args.dataset_name:
            question    = ex["question"]
            candidates  = ex["candidates"]
            groundtruth = ex.get("answer", "")


            opts_text = "\n".join(f"{cand}" for cand in candidates)
            full_q = (
                f"{DEFAULT_IMAGE_TOKEN}\n"
                f"{question}\n\n"
                f"Options:\n{opts_text}\n\n"
            )
            local_video_path = ex["video"]
        elif "vinoground" in args.dataset_name or "temporal" in args.dataset_name:
            question = ex['question']
            groundtruth = ex.get("GT", "")
            full_q = (
                f"{DEFAULT_IMAGE_TOKEN}\n"
                f"{question}\n\n"
            )
            local_video_path = ex["video_name"]


        conv = copy.deepcopy(conv_templates[PROMPT_VERSION])
        conv.append_message(conv.roles[0], full_q)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()


        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(DEVICE)


        base_path = VIDEO_PATHS[args.dataset_name]
        video_path = f'{base_path}{local_video_path}'
        raw_frames = sample_frames(video_path, args.num_frames)
        pil_frames = [Image.fromarray(f) for f in raw_frames]
        pixel_values = (processor.preprocess(raw_frames, return_tensors="pt")["pixel_values"].half().to(DEVICE))
        img_sizes = [(im.width, im.height) for im in pil_frames]

      
        # print("has <Image> token", (input_ids == IMAGE_TOKEN_INDEX).any().item())
        # Generate
 
        if "video" in args.vision_tower.lower():
            if args.vision_tower == "video_embedding_concat_global":
                # ASSUMES JSON_PATH="/data/samyakp/llava_video_data/eval_videos/${DATASET}_with_internVideo2_global_embeddings.json" is passed in
                internvid2_global_path = ex["internVideo2_global_embedding"]
                # Get path of videoMAE embedding of same eval video
                videomae_global_path = '/' + os.path.join(*internvid2_global_path.split('/')[:-2], *[s.replace('internvideo2', 'videoMAE') for s in internvid2_global_path.split('/')[-2:]])
                internvid2_embed = torch.load(internvid2_global_path, map_location=DEVICE).unsqueeze(0)
                videomae_embed = torch.load(videomae_global_path, map_location=DEVICE)
                emb = torch.cat([internvid2_embed, videomae_embed], dim=-1)
            elif args.vision_tower == "video_embedding_concat_patch":
                # ASSUMES JSON_PATH="/data/samyakp/llava_video_data/eval_videos/${DATASET}_with_internVideo2_patch_embeddings.json" is passed in
                internvid2_patch_path = ex["internVideo2_patch_embedding"] 
                # Get path of videoMAE embedding of same eval video
                videomae_patch_path = '/' + os.path.join(*internvid2_patch_path.split('/')[:-2], *[s.replace('internvideo2', 'videoMAE') for s in internvid2_patch_path.split('/')[-2:]])
                internvid2_embed = torch.load(internvid2_patch_path, map_location=DEVICE)
                videomae_embed = torch.load(videomae_patch_path, map_location=DEVICE)
                # Upsample InternVideo2 patch num to videoMAE's patch num
                upsampled_internvideo2_embed = upsample_patch_embeds(internvid2_embed, videomae_embed)
                # Concatenate along feature dim
                emb = torch.cat([upsampled_internvideo2_embed, videomae_embed], dim=-1)
            elif args.vision_tower == "video_embedding_videoMAE":
                emb = torch.load(ex["videoMAE_global_embedding"], map_location=DEVICE) 
            elif "videoMAE_patch" in args.vision_tower:
                emb = torch.load(ex["videoMAE_patch_embedding"], map_location=DEVICE) 
            elif "internVideo2_patch" in args.vision_tower:
                emb = torch.load(ex["internVideo2_patch_embedding"], map_location=DEVICE) 
            else:
                emb = torch.load(ex[args.vision_tower], map_location=DEVICE)
            emb = emb.to(DEVICE)                                      
            emb = emb.half()
            if emb.ndim == 1:
                emb = emb.unsqueeze(0)
            num_patches = 1
            # emb = simulate_patches(emb, num_patches)
            if "_with_siglip" not in args.vision_tower:
                out_ids = model.generate(
                    inputs            = input_ids,
                    images            = None,
                    video_embeddings  = [emb],
                    modalities        = ["video"],
                    do_sample         = False,
                    temperature       = 0.0,
                    max_new_tokens    = 16,
                )
            else:
               out_ids = model.generate(
                    inputs            = input_ids,
                    images            = [pixel_values],
                    video_embeddings  = [emb],
                    modalities        = ["video"],
                    do_sample         = False,
                    temperature       = 0.0,
                    max_new_tokens    = 16,
                ) 
        else:
            out_ids = model.generate(
                inputs      = input_ids,
                images      = [pixel_values],
                image_sizes = img_sizes,
                modalities  = ["video"],
                do_sample   = False,
                temperature = 0.0,
                max_new_tokens=16,
                )

        model_answer = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()

        records.append({
            "question": question,
            "model_answer": model_answer,
            "groundtruth": groundtruth,
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} entries to {output_csv}")


if __name__ == "__main__":
    main()
