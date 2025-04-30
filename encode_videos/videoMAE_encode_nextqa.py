# USE DIFFUSION CONDA ENV
import os
import json
import torch
from transformers import VideoMAEImageProcessor, VideoMAEModel 
import cv2
import numpy as np
from tqdm import tqdm


DATA_PATH = "/data/samyakp/llava_video_data/30_60_s_nextqa/"
MC_JSON = "30_60_s_nextqa_mc_qa_processed_all_video_embeddings.json"
OE_JSON = "30_60_s_nextqa_oe_qa_processed_all_video_embeddings.json"

EMBEDDINGS_OUT_DIR = os.path.join(DATA_PATH, "videoMAE_patch_embeddings/")
DEVICE = "cuda:7"


# Load VideoMAE
videomae_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
videomae_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(DEVICE)# , device_map="auto") #buggy with VideoMAE


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


# DATA_PATH + MC_JSON, 
data_files = [DATA_PATH + OE_JSON, DATA_PATH + MC_JSON]
# SET OF EXISTING EMBEDDINGS (most entries will NOT have unique videos)
EXISTING_EMBEDDINGS = set(os.path.join(EMBEDDINGS_OUT_DIR, fname) for fname in os.listdir(EMBEDDINGS_OUT_DIR))

for data_file in data_files:
    # Load your JSON data (replace with your file path or variable)
    with open(data_file, 'r') as f:
        data = json.load(f)

    # New JSON with emebdding paths
    new_json = data_file.split(".json")[0] + "_videoMAE_patch.json"

    for entry in tqdm(data):
        video_path = f"/data/samyakp/llava_video_data/{entry['video']}"
        video_id = entry['id']
        videomae_embed_path = os.path.join(EMBEDDINGS_OUT_DIR, f"{video_id}_videoMAE.pt")

        # Only embed video if it doesn't already exist
        if videomae_embed_path not in EXISTING_EMBEDDINGS:
            print(f"Processing {video_path}...")
            frames = load_video_frames(video_path)
            if len(frames) == 0:
                print(f"Failed to load frames from {video_path}")
                continue
            
            # VideoMAE forward 
            videomae_inputs = videomae_processor([frames], return_tensors="pt")
            videomae_inputs = {k: v.to(DEVICE) for k, v in videomae_inputs.items()}
            with torch.no_grad():
                videomae_outputs = videomae_model(**videomae_inputs)
                # # Global video embedding
                # videomae_embedding = videomae_outputs.last_hidden_state.mean(dim=1)
                # Token/patch embeddings
                videomae_embedding = videomae_outputs.last_hidden_state.squeeze(0)

            torch.save(videomae_embedding, videomae_embed_path)
        else:
            print(f"Embedding already exists for {video_path}:\n{videomae_embed_path}\nskipping...")

        entry["videoMAE_patch_embedding"] = videomae_embed_path 
        with open(new_json, 'w') as f:
            json.dump(data, f, indent=4)
