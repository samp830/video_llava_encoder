import torch
import torch.nn as nn
import torch.nn.functional as F
from llava.utils import rank0_print

from .clip_encoder import CLIPVisionTower
from .siglip_encoder import SigLipVisionTower
from .mlcd_encoder import MLCDVisionTower


class MultiImageEncoderVisionTower(nn.Module):
    def __init__(self, vision_tower_cfg, **kwargs):
        super().__init__()

        self.vision_tower_name = "multi_image_encoder"
        self.is_loaded = False
        self.config = vision_tower_cfg

        rank0_print(f"Loading TriEncoder (CLIP, SigLIP, MLCD) Vision Tower")
        # CLIP
        # I should've used: "openai/clip-vit-large-patch14-336" to match 14-patch MLCD
        self.clip = CLIPVisionTower("openai/clip-vit-base-patch16", args=vision_tower_cfg, **kwargs)
        # MLCD
        self.mlcd = MLCDVisionTower("DeepGlint-AI/mlcd-vit-bigG-patch14-448", args=vision_tower_cfg, **kwargs)

        # SigLIP 
        # self.siglip = SigLipVisionTower("google/siglip-so400m-patch14-384", vision_tower_cfg=vision_tower_cfg, **kwargs)

        # Both CLIP and MLCD use the CLIPImageProcessor
        self.image_processor = self.clip.image_processor
        self.is_loaded = True

        # Calculate total features size
        self._hidden_size = (
            self.clip.hidden_size +
            # self.siglip.hidden_size +
            self.mlcd.hidden_size 
        )

    def forward(self, images):
        # Output shapes: (batch, num_patches, features)
        #  torch.Size([33, 196, 768])
        clip_features = self.clip(images)
        # torch.Size([33, 256, 1664])
        # TODO: SigLIP features should be directly concatenated (14 patches)
        # siglip_features = self.siglip(images)
        mlcd_features = self.mlcd(images)

        # DEBUG
        # rank0_print(f"\n\n\nclip_features shape: {clip_features.shape}")
        # rank0_print(f"mlcd_features shape: {mlcd_features.shape}")
        # rank0_print(f"\n\n\ninput images to vision tower shape: {images.shape}")

        # Upsample number of CLIP patches to match MLCD
        # Permute to (bathc, features, num_patches)for interpolation
        mlcd_num_patches = mlcd_features.shape[1]
        clip_features_up = clip_features.permute(0, 2, 1)
        clip_features_up = F.interpolate(clip_features_up, size=mlcd_num_patches, mode='nearest')
        clip_features = clip_features_up.permute(0, 2, 1)  # Back to (batch, num_patches, features)

        # TODO: COULD TRY INTERLEAVING PATCHES HERE INSTEAD OF UPSIZING MLCD
        # # Reshape to (batch, features, patch height, patch width)
        # batch_size = mlcd_features.shape[0]
        # mlcd_feat_dim = mlcd_features.shape[-1]
        # mlcd_features_patches = mlcd_features.permute(0, 2, 1).reshape(batch_size, mlcd_feat_dim, 14, 14)

        # Concatenate features from all models
        # torch.Size([33, 256, 2432])       
        features = torch.cat((clip_features, mlcd_features), dim=-1)

        return features


    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def num_patches_per_side(self):
        # Using MLCD's number of patches, since in forward pass CLIP features are upsampled to match MLCD
        return 16 # MLCD patch size: 14 and images are 224x224, so 224/14 = 16
        # return self.mlcd.num_patches_per_side

    @property
    def num_patches(self):
        return self.clip.num_patches

    @property
    def image_size(self):
        return self.clip.image_size