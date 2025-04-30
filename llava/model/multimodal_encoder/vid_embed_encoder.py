import torch
import torch.nn as nn
import torch.nn.functional as F
from llava.utils import rank0_print

from transformers import CLIPImageProcessor
# from .clip_encoder import CLIPVisionTower 
# from .siglip_encoder import SigLipVisionTower, SigLipImageProcessor
# from .mlcd_encoder import MLCDVisionTower
# from .hf_vision import HFVisionTower


class VideoEmbeddingVisionTower(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, **kwargs):
        """
        VisionTower that goes with pre-computed video embeddings.
        Video embedding tensors are loaded in _get_item() in LazySupervisedDataset (llava/train/train.py)

        vision_tower: must be a string formatted like below
            "video_embedding_<video model>" or 
            "video_embedding_<video model>_with_<image_encoder>" or
            "video_embedding_concat_<global or patch>
            <video model> options: [videoMAE, internVideo2]

            TO ADDITIONALLY USE IMAGE ENCODER(S) AND CONCAT VIDEO + IMAGE FEATURES
            add "_with_<image_encoder>" to the vision_tower string "video_embedding_<video model>_with_<image_encoder>"
            <image_encoder> options are any combo of: [clip, mlcd, siglip, dino]
        """
        super().__init__()

        self.vision_tower_name = vision_tower
        # # Validate the vision tower name/string
        # options = [
        #     "concat_global",
        #     "concat_patch",
        #     "internVideo2_global",
        #     "videoMAE_global",
        #     "videoMAE_patch",
        #     "internVideo2_patch"
        # ]

        # count = sum(opt in self.vision_tower_name for opt in options)
        # if count != 1:
        #     raise ValueError(
        #         f"Exactly one of {options} must be present in vision_tower_name, "
        #         f"but found {count} in '{self.vision_tower_name}'."
        #     )

        self.is_loaded = False
        self.config = vision_tower_cfg

        # # All possible video embedding models
        # video_models = ["videoMAE", "internVideo2"]
        # self.video_embedding = None
        # # TODO: add option for both (pre-loaded features will be concatenated in dataset class)
        # # self.video_embedding = "internvideo2"
        # for model in video_models:
        #     if model in vision_tower:
        #         self.video_embedding = model
        #         break
        # if self.video_embedding is None:
        #     raise ValueError(f"Unknown video embedding for: {vision_tower}")

        rank0_print(f"Loading {self.vision_tower_name} Embedding Vision Tower")

        self.use_vision_encoder = False
        if "_with_" in vision_tower:
            self.use_vision_encoder = True
            # TODO: init vision tower once we know best image encoder
            # implement forward() and set self.image_processor accordingly
            # add something like self.image_encoder and set it to the model
        else:
            # Just use dummy, CLIP Image Processor 
            # (not used if only using video embeddings, needed for image loading in Dataset class)
            self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

        self.is_loaded = True

        # Internvideo2 and videoMAE have the same global feature dim
        if "concat_global" in self.vision_tower_name:
            self._hidden_size = 768 * 2
        elif "concat_patch" in self.vision_tower_name:
            self._hidden_size = 768 + 1408 
        elif "internVideo2_global" in self.vision_tower_name or "videoMAE" in self.vision_tower_name:
            self._hidden_size = 768
        elif "internVideo2_patch" in self.vision_tower_name:
            # internVideo2 patch features are size (1025, 1408)
            self._hidden_size = 1408

    def forward(self, images):
        """
        If only using video embeddings, this method is never used 
        (video embeddings are loaded in _get_item() in LazySupervisedDataset)

        Concatenation of video emebeddings and image embeddings is handled in prepare_inputs_labels_for_multimodal()
        """
        # if self.use_vision_encoder:
        #    return self.image_encoder(images)
        return images

    @property
    def hidden_size(self):
        # TODO: for video embedding + image encoder
        # if self.use_vision_encoder:
        #    return self.image_encoder.hidden_size + self._hidden_size
        return self._hidden_size

    @property
    def num_patches_per_side(self):
        # For video embeddings only, doesn't matter
        # if self.use_vision_encoder:
        #    return self.image_encoder.num_patches_per_side
        return 14

    @property
    def num_patches(self):
        # For video embeddings only, doesn't matter
        # if self.use_vision_encoder:
        #    return self.image_encoder.num_patches
        return 14

    @property
    def image_size(self):
        # For video embeddings only, doesn't matter
        # if self.use_vision_encoder:
        #    return self.image_encoder.image_size
        return 14