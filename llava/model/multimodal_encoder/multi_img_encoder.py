import torch
import torch.nn as nn
import torch.nn.functional as F
from llava.utils import rank0_print

from .clip_encoder import CLIPVisionTower
from .siglip_encoder import SigLipVisionTower, SigLipImageProcessor
from .mlcd_encoder import MLCDVisionTower


class MultiImageEncoderVisionTower(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, **kwargs):
        super().__init__()

        self.vision_tower_name = vision_tower
        self.is_loaded = False
        self.config = vision_tower_cfg

        # All possible vision towers
        self.clip = None
        self.mlcd = None
        self.siglip = None

        rank0_print(f"Loading Multi-Image Encoder: {vision_tower} Vision Tower")

        # Keep track of initialized vision towers
        self.vision_towers = []
        if "clip" in vision_tower:
            # CLIP
            self.clip = CLIPVisionTower("openai/clip-vit-large-patch14-336", args=vision_tower_cfg, **kwargs)
            self.vision_towers.append(self.clip)

        if "mlcd" in vision_tower:
            # MLCD
            self.mlcd = MLCDVisionTower("DeepGlint-AI/mlcd-vit-bigG-patch14-448", args=vision_tower_cfg, **kwargs)
            self.vision_towers.append(self.mlcd)

        if "siglip" in vision_tower:
            # SigLIP 
            self.siglip = SigLipVisionTower("google/siglip-so400m-patch14-384", vision_tower_cfg=vision_tower_cfg, **kwargs)
            self.vision_towers.append(self.siglip)
 

        # Both CLIP and MLCD use the CLIPImageProcessor, SigLIP can't take the same image dims as input
        # If SigLIP is used, we resize images from 384x384 to 336x336 in forward pass
        if self.siglip is not None:
            self.image_processor = self.siglip.image_processor
        elif self.clip is not None:
            self.image_processor = self.clip.image_processor
        elif self.mlcd is not None:
            self.image_processor = self.mlcd.image_processor
        else:
            raise RuntimeError("No vision tower was initialized!")

        self.is_loaded = True

        # Calculate total features size from only initialized towers
        self._hidden_size = sum(getattr(tower, 'hidden_size', 0) for tower in self.vision_towers)

    def forward(self, images):
        """
        Pass images through each initialized vision tower and concatenate the features along dim=-1.
        """
        # CLIPImageProcessor: images shape is torch.Size([33, 3, 336, 336])  
        # SigLIPImageProcessor: images shape is torch.Size([33, 3, 384, 384])
        features = []
        # Output shapes for ALL towers: (batch, num_patches, features) 
        if self.clip is not None:
            # SigLIPImageProcessor images are 384x384, but CLIP needs 336x336
            if images.shape[-1] == 384:
                # Resize images to 336x336 for CLIP
                resized = F.interpolate(images, size=(336, 336), mode='bicubic', align_corners=False, antialias=True)
                clip_features = self.clip(resized)
            else:
                clip_features = self.clip(images)
            features.append(clip_features)
        
        if self.mlcd is not None:
            # SigLIPImageProcessor images are 384x384, but MLCD needs 336x336
            if images.shape[-1] == 384:
                # Resize images to 336x336 for MLCD 
                resized = F.interpolate(images, size=(336, 336), mode='bicubic', align_corners=False, antialias=True)
                mlcd_features = self.mlcd(resized) 
            else:
                mlcd_features = self.mlcd(images)
            features.append(mlcd_features)
        
        if self.siglip is not None:
            siglip_features = self.siglip(images)
            features.append(siglip_features)
        
        if not features:
            raise RuntimeError("No vision tower is initialized!")
        
        # Must resize features from all towers to same number of patches 
        # (only happens if SigLIP is used)
        max_patches = max(f.size(1) for f in features)
        resized_features = []
        for f in features:
            if f.size(1) < max_patches:
                # Permute to (batch, features, num_patches) for interpolation
                f_up = f.permute(0, 2, 1)
                f_up = F.interpolate(f_up, size=max_patches, mode='linear', align_corners=False)
                f_up = f_up.permute(0, 2, 1)
                resized_features.append(f_up)
            else:
                resized_features.append(f)
        # Concatenate features along the last dimension
        return torch.cat(resized_features, dim=-1)

        # DEBUG
        # rank0_print(f"\n\n\nclip_features shape: {clip_features.shape}")
        # rank0_print(f"mlcd_features shape: {mlcd_features.shape}")
        # rank0_print(f"\n\n\ninput images to vision tower shape: {images.shape}")

        # ONLY NECESSARY FOR 16-PATCH MODELS ("openai/clip-vit-base-patch16")
        # # Upsample number of CLIP patches to match MLCD
        # # Permute to (bathc, features, num_patches)for interpolation
        # mlcd_num_patches = mlcd_features.shape[1]
        # clip_features_up = clip_features.permute(0, 2, 1)
        # clip_features_up = F.interpolate(clip_features_up, size=mlcd_num_patches, mode='nearest')
        # clip_features = clip_features_up.permute(0, 2, 1)  # Back to (batch, num_patches, features)

        # TODO: COULD TRY INTERLEAVING PATCHES HERE INSTEAD OF UPSIZING MLCD
        # # Reshape to (batch, features, patch height, patch width)
        # batch_size = mlcd_features.shape[0]
        # mlcd_feat_dim = mlcd_features.shape[-1]
        # mlcd_features_patches = mlcd_features.permute(0, 2, 1).reshape(batch_size, mlcd_feat_dim, 14, 14)

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def num_patches_per_side(self):
        if self.siglip is not None:
            return self.siglip.num_patches_per_side
        else:
            return self.clip.num_patches_per_side

    @property
    def num_patches(self):
        if self.siglip is not None:
            return self.siglip.num_patches
        else:
            return self.clip.num_patches

    @property
    def image_size(self):
        if self.siglip is not None:
            return self.siglip.image_size
        else:
            return self.clip.image_size