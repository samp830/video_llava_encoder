# USE DIFFUSION CONDA ENV
import os
import json
import torch
from transformers import VideoMAEImageProcessor, VideoMAEModel 
import cv2
import numpy as np
from tqdm import tqdm


MV_BENCH_ACTION = "/data/samyakp/llava_video_data/eval_videos/mvbench_action_localization.json"
MV_BENCH_EGOCENTRIC = "/data/samyakp/llava_video_data/eval_videos/mvbench_egocentric_navigation.json"
MV_BENCH_MOVING_COUNT = "/data/samyakp/llava_video_data/eval_videos/mvbench_moving_count.json"
MV_BENCH_MOVING_DIRECTION = "/data/samyakp/llava_video_data/eval_videos/mvbench_moving_direction.json"
TEMPORAL_BENCH = "/data/samyakp/llava_video_data/eval_videos/temporalbench_subset.json"
VINOGROUND_TEXT = "/data/samyakp/llava_video_data/eval_videos/vinoground_textscore_subset.json"
VINOGROUND_VIDEO = "/data/samyakp/llava_video_data/eval_videos/vinoground_videoscore_subset.json"

data_files = [MV_BENCH_ACTION, MV_BENCH_EGOCENTRIC, MV_BENCH_MOVING_COUNT, MV_BENCH_MOVING_DIRECTION, TEMPORAL_BENCH, VINOGROUND_TEXT, VINOGROUND_VIDEO]

VIDEO_PATHS = {
    "mvbench_action_localization": "/data/samyakp/llava_video_data/sta/sta_video/",
    "mvbench_egocentric_navigation": "/data/samyakp/llava_video_data/vlnqa/",
    "mvbench_moving_direction": "/data/samyakp/llava_video_data/clevrer/video_validation/",
    "mvbench_moving_count": "/data/samyakp/llava_video_data/clevrer/video_validation/",
    "vinoground_textscore_subset":"/data/samyakp/llava_video_data/",
    "vinoground_videoscore_subset":"/data/samyakp/llava_video_data/",
    "temporalbench_subset":"/data/samyakp/llava_video_data/"}

DEVICE = "cuda:7"

# Load VideoMAE
videoMAE_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videoMAE-base")
videoMAE_model = VideoMAEModel.from_pretrained("MCG-NJU/videoMAE-base").to(DEVICE)


def load_video_frames(
    video_path, 
    num_frames=16, 
    size=224
):
    """Loads and preprocesses exactly num_frames from a video file, handling unreadable frames.
    Returns an empty list if the video is invalid or unreadable.
    """
    if not os.path.isfile(video_path):
        print(f"File does not exist: {video_path}")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"Video has no frames: {video_path}")
        cap.release()
        return []

    frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = []
    last_valid_frame = None
    black_frame = np.zeros((size, size, 3), dtype=np.uint8)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {idx} from {video_path}, using last valid frame.")
            if last_valid_frame is not None:
                frames.append(last_valid_frame.copy())
            else:
                frames.append(black_frame.copy())
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size))
        frames.append(frame)
        last_valid_frame = frame  # Update last good frame

    cap.release()

    return frames


for data_file in data_files:
    # Output JSON with emebdding paths
    ds_name = data_file.split("/")[-1].replace(".json", "")
    OUTPUT_DIR = f"/data/samyakp/llava_video_data/eval_videos/{ds_name}/videoMAE_patch_embeddings"
    # OUTPUT_DIR = f"/data/samyakp/llava_video_data/eval_videos/{ds_name}/videoMAE_global_embeddings"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    EXISTING_EMBEDDINGS = set(os.path.join(OUTPUT_DIR, fname) for fname in os.listdir(OUTPUT_DIR))
    new_json = data_file.split(".json")[0] + "_with_videoMAE_patch_embeddings.json"
    # new_json = data_file.split(".json")[0] + "_with_videoMAE_global_embeddings.json"

    with open(data_file, "r") as f:
        records = json.load(f)

    base_path = VIDEO_PATHS[ds_name]
    for rec in tqdm(records):
        if "vinoground" in data_file or "temporal" in data_file:
            video_path = rec["video_name"]
        else:
            video_path = rec["video"]
        
        if "nextqa" in data_file:
            vid_id     = rec["id"]
        else:
            vid_id = video_path.split("/")[-1].replace(".mp4", "")
        # Keep track of output path to skip already encoded videos
        
        videoMAE_embed_path = os.path.join(OUTPUT_DIR, f"{vid_id}_videoMAE.pt")

        # Only embed video if it doesn't already exist
        if videoMAE_embed_path not in EXISTING_EMBEDDINGS:
            video_path = base_path + video_path 
            print(f"Processing {video_path}...")
            frames = load_video_frames(video_path)
            if len(frames) == 0:
                print(f"Failed to load frames from {video_path}")
                continue

            # VideoMAE forward 
            videomae_inputs = videoMAE_processor([frames], return_tensors="pt")
            videomae_inputs = {k: v.to(DEVICE) for k, v in videomae_inputs.items()}
            with torch.no_grad():
                videomae_outputs = videoMAE_model(**videomae_inputs)
                # # Global video embedding
                # videomae_embedding = videomae_outputs.last_hidden_state.mean(dim=1)
                # Token/patch embeddings
                videomae_embedding = videomae_outputs.last_hidden_state.squeeze(0)

            torch.save(videomae_embedding, videoMAE_embed_path)
        else:
            print(f"Embedding already exists for {video_path}:\t{videoMAE_embed_path}\tskipping...")

        rec["videoMAE_patch_embedding"] = videoMAE_embed_path
        # rec["videoMAE_global_embedding"] = videoMAE_embed_path

        with open(new_json, "w") as f:
            json.dump(records, f, indent=2)

print("Done")