"""Frozen DINOv2 reference encoder and checkpointed MR feature projection."""
from pathlib import Path
import numpy as np
import PIL
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from miku.vendor.dinov2.models.vision_transformer import vit_large


class FrozenDinoEmbedder_v2(nn.Module):
    def __init__(self, weights_path, *, device="cuda", size=448, linear_ckpt_path=None):
        super().__init__()
        self.device = device
        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(weights_path)
        if linear_ckpt_path is None or not Path(linear_ckpt_path).is_file():
            raise FileNotFoundError("PBR requires a local dino_linear.pt projection checkpoint")
        model = vit_large(img_size=518, patch_size=14, init_values=1.0,
                          ffn_layer="mlp", block_chunks=0, num_register_tokens=4,
                          interpolate_antialias=True, interpolate_offset=0.0)
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)
        self.dino_model = model.to(device).eval().requires_grad_(False)
        self.transform = transforms.Compose([
            transforms.Resize(size, transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.linear = nn.Linear(128 + 1024, 128)
        state = torch.load(linear_ckpt_path, map_location="cpu", weights_only=True)
        self.linear.load_state_dict(state.get("state_dict", state), strict=True)
        self.linear.to(device)
        self.eval().requires_grad_(False)

    def forward(self, image):

        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            z = self.dino_model.forward_features(image)['x_prenorm']

        z = F.layer_norm(z, z.shape[-1:])
        return z

    def encode(self, image):
        return self(image)

    def process_dino_and_vae(self, vae_latents, images):

        dino_feats=[]
        if isinstance(images, torch.Tensor):
            images=images.cpu().permute(0, 2, 3, 1).numpy() * 255
            for i in range(images.shape[0]):
                image = Image.fromarray(images[i].astype(np.uint8)).convert("RGB")
                dino_feat = self.encode(image)   # [1, N, 1024]
                dino_feats.append(dino_feat)
        elif isinstance(images, PIL.Image.Image):
            dino_feat = self.encode(images)   # [1, N, 1024]
            dino_feats.append(dino_feat)
        dino_feat = torch.cat(dino_feats, dim=0)  # [B, N, 1024]
        # 去 CLS
        if dino_feat.shape[1] == vae_latents.shape[1] + 5:
            dino_feat = dino_feat[:, 5:]


        if dino_feat.shape[1] != vae_latents.shape[1]:
            dino_feat = F.interpolate(
                dino_feat.permute(0, 2, 1),
                size=vae_latents.shape[1],
                mode='linear'
            ).permute(0, 2, 1)

        dino_vae_feat = torch.cat([dino_feat, vae_latents], dim=-1)

        dino_vae_feat = self.linear(dino_vae_feat)

        return dino_vae_feat
    
