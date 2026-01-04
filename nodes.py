"""
Omni-Ramp Extractor & Gradient Reconstruction Tool
Author: Yurii Borodin (ninpo_3d)
License: MIT
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def kmeans_centroids(X, n_clusters, iter=10):
    """
    Finds dominant colors in a point cloud using simplified K-Means.
    This allows us to extract the 'True Palette' of the VFX asset.
    """
    if X.shape[0] < n_clusters: return X
    
    indices = torch.randperm(X.shape[0])[:n_clusters]
    centroids = X[indices]
    
    for _ in range(iter):
        dists = torch.cdist(X, centroids)
        labels = torch.argmin(dists, dim=1)
        new_centroids = []
        for k in range(n_clusters):
            mask = labels == k
            if mask.any(): 
                new_centroids.append(X[mask].mean(dim=0))
            else: 
                new_centroids.append(centroids[k]) # Stability fallback
        new_centroids = torch.stack(new_centroids)
        if torch.norm(new_centroids - centroids) < 1e-4: break
        centroids = new_centroids
        
    return centroids

# ==============================================================================
# NODE CLASSES
# ==============================================================================

class RampExtractor:
    """
    The main node that analyzes an image and deconstructs it into a Gradient Ramp and a Grayscale Mask.
    """
    
    DESCRIPTION = """
    Omni-Ramp Extractor.
    Deconstructs visual effects (Fire, Smoke, Magic) into a Gradient Ramp and a Grayscale Mask.

    MODES:
    1. Universal (Spectrum):
       - Uses AI clustering (K-Means) to find the exact color palette of your image.
       - Perfect for Fire, Magic, Plasma, and complex textures (Bricks, Moss).
       - Preserves the "Energy" and saturation of colors better than standard methods.
       
    2. Linear (Fast Luma):
       - Standard Rec.709 brightness formula (Grayscale conversion).
       - Faster, but tends to make Fire look dull/washed out.
       - Use this for simple white Smoke or Fog.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                
                "mode": (["Universal (Spectrum)", "Linear (Fast Luma)"], {
                    "default": "Universal (Spectrum)",
                    "tooltip": "Universal: Reconstructs palette (High Quality). Linear: Standard brightness math (Fast)."
                }),
                
                "samples": ("INT", {
                    "default": 256, "min": 2, "max": 1024, 
                    "tooltip": "Width of the output Ramp texture. 256 is standard for VFX."
                }),
                
                "fill_gaps": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "Fill holes in gradient with interpolation. Keeps the ramp smooth."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("grayscale_map", "ramp")
    FUNCTION = "execute"
    CATEGORY = "omni_extractor/Gradients"

    @torch.inference_mode()
    def execute(self, image: torch.Tensor, mode: str, samples: int, fill_gaps: bool):
        device = image.device
        img = image[0].to(device).to(torch.float32)
        h, w, c = img.shape
        flat_img = img.reshape(-1, 3)
        
        scalar_map = None
        ramp_output_tensor = None

        # ============================================================
        # MODE A: UNIVERSAL (SPECTRUM ANALYSIS)
        # ============================================================
        if mode == "Universal (Spectrum)":
            # 1. Palette Reconstruction
            # We treat the entire image as a single dataset.
            # No manual splitting (Blue vs Fire) is needed anymore because K-Means
            # naturally finds the distinct clusters if they exist.
            
            # Sampling optimization for large images to keep K-Means fast
            if flat_img.shape[0] > 10000:
                sample_pixels = flat_img[torch.randperm(flat_img.shape[0])[:10000]]
            else:
                sample_pixels = flat_img

            # Extract dominant colors
            palette = kmeans_centroids(sample_pixels, samples)
            
            # Sort Palette by Luma (Dark -> Bright)
            # This ensures the gradient goes from Black (0.0) to White (1.0) logically.
            luma = palette[:, 0] * 0.2126 + palette[:, 1] * 0.7152 + palette[:, 2] * 0.0722
            sorted_indices = torch.argsort(luma)
            master_ramp = palette[sorted_indices]
            
            # 2. Encoding (The Reconstruction)
            # Find the closest color in the Master Ramp for every original pixel.
            # This maps the complex colors back to a 0..1 scalar value based on the sorted ramp.
            chunk_size = 100000 
            indices_list = []
            
            # Process in chunks to save VRAM
            for i in range(0, flat_img.shape[0], chunk_size):
                chunk = flat_img[i : i + chunk_size]
                dists = torch.cdist(chunk, master_ramp)
                closest = torch.argmin(dists, dim=1)
                indices_list.append(closest)
            
            full_indices = torch.cat(indices_list)
            scalar_map = full_indices.float() / (samples - 1)
            scalar_map = scalar_map.reshape(h, w)
            ramp_output_tensor = master_ramp

        # ============================================================
        # MODE B: LINEAR (REC.709) - Legacy Fallback
        # ============================================================
        else:
            # Standard perceptual brightness
            scalar_map = img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722
            
            # Histogram-based ramp building
            keys = scalar_map.flatten()
            indices = (torch.clamp(keys, 0.0, 0.9999) * samples).long()
            
            ramp_sum = torch.zeros((samples, 3), device=device)
            ramp_count = torch.zeros((samples, 1), device=device)
            
            ramp_sum.scatter_add_(0, indices.unsqueeze(1).expand(-1, 3), flat_img)
            ramp_count.scatter_add_(0, indices.unsqueeze(1), torch.ones_like(indices.unsqueeze(1), dtype=torch.float32))
            
            ramp_output_tensor = ramp_sum / torch.clamp(ramp_count, min=1e-6)
            
            # Gap filling
            if fill_gaps:
                valid_mask = ramp_count > 0
                last_v = ramp_output_tensor[0]
                for i in range(samples):
                    if valid_mask[i]: last_v = ramp_output_tensor[i]
                    else: ramp_output_tensor[i] = last_v
                last_v = ramp_output_tensor[-1]
                for i in range(samples-1, -1, -1):
                    if valid_mask[i]: last_v = ramp_output_tensor[i]
                    else: ramp_output_tensor[i] = last_v

        # Output formatting
        grayscale_output = scalar_map.unsqueeze(-1).repeat(1, 1, 3).unsqueeze(0).clamp(0, 1).to(torch.float32)
        final_ramp = ramp_output_tensor.unsqueeze(0).unsqueeze(0).to(torch.float32)
        
        return (grayscale_output, final_ramp)

class GradientPreview:
    """
    Visualizer Node.
    Maps the Grayscale mask back to Color using the Ramp.
    Now supports both Smooth (Bilinear) and Sharp (Nearest) modes.
    """
    DESCRIPTION = """
    Gradient Preview.
    Maps the Grayscale mask back to Color using the Ramp.
    
    MODES:
    - Linear (Smooth): Good for Smoke, Fog, and soft gradients. Blends colors.
    - Nearest (Sharp): Good for Fire, Magic, and exact palette matching. Preserves peak colors.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "grayscale": ("IMAGE",), 
                "ramp": ("IMAGE",),
                "interpolation": (["Linear (Smooth)", "Nearest (Sharp)"], {"default": "Linear (Smooth)"})
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "apply"
    CATEGORY = "omni_extractor/Gradients"

    @torch.inference_mode()
    def apply(self, grayscale: torch.Tensor, ramp: torch.Tensor, interpolation: str):
        # 1. Setup Data & Devices
        device = grayscale.device
        grayscale = grayscale.to(torch.float32)
        ramp = ramp.to(torch.float32).to(device)
        
        batch, h, w, _ = grayscale.shape

        # 2. Prepare Coordinate Grid
        grid = torch.zeros((batch, h, w, 2), dtype=torch.float32, device=device)
        grid[..., 0] = (grayscale[..., 0] * 2.0) - 1.0 
        grid[..., 1] = 0.0

        # 3. Prepare Ramp Texture
        ramp_texture = ramp.permute(0, 3, 1, 2)
        if batch > 1:
            ramp_texture = ramp_texture.repeat(batch, 1, 1, 1)

        # 4. Hardware Sampling with Mode Selection
        # Select interpolation mode based on user input
        sample_mode = 'bilinear' if interpolation == "Linear (Smooth)" else 'nearest'
        
        # NOTE: align_corners=True is critical for precise color matching at the edges
        sampled_tensor = F.grid_sample(
            ramp_texture, 
            grid, 
            align_corners=True, 
            mode=sample_mode, 
            padding_mode='border'
        )

        # 5. Output Formatting
        result = sampled_tensor.permute(0, 2, 3, 1)
        
        return (result,)

class AssetPacker:
    """
    Utility to save the decomposed assets.
    """
    DESCRIPTION = """
    Asset Packer.
    Saves your extracted Grayscale mask, Ramp, and original Alpha into separate files or channels.
    Use this to export the final assets for Unreal/Unity.
    """
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_name": ("STRING", {"default": "VFX_Packed", "tooltip": "Subfolder name in output directory ComfyUI/output/"}),
                "file_name": ("STRING", {"default": "VFX_Asset", "tooltip": "Base filename prefix"}),
                "bit_depth": (["8-bit", "16-bit"], {"default": "8-bit", "tooltip": "16-bit is recommended for Height Maps/Masks"}),
                "save_metadata": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image_R": ("IMAGE",), "image_G": ("IMAGE",), "image_B": ("IMAGE",), "image_A": ("IMAGE",),
                "ramp_R": ("IMAGE",), "ramp_G": ("IMAGE",), "ramp_B": ("IMAGE",), "ramp_A": ("IMAGE",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "omni_extractor/IO"

    def save(self, folder_name, file_name, bit_depth, save_metadata, 
             image_R=None, image_G=None, image_B=None, image_A=None, 
             ramp_R=None, ramp_G=None, ramp_B=None, ramp_A=None, 
             prompt=None, extra_pnginfo=None):
        results = []
        img_map = {"R": image_R, "G": image_G, "B": image_B, "A": image_A}
        active_imgs = {k: v for k, v in img_map.items() if v is not None}
        if not active_imgs: return {"ui": {"images": []}}
        is_16 = bit_depth == "16-bit"
        path = os.path.join(self.output_dir, folder_name)
        if not os.path.exists(path): os.makedirs(path)
        metadata = PngInfo()
        if save_metadata:
            if prompt: metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo:
                for x in extra_pnginfo: metadata.add_text(x, json.dumps(extra_pnginfo[x]))
        base = next(iter(active_imgs.values()))
        batch, h, w, _ = base.shape
        for i in range(batch):
            if len(active_imgs) == 1:
                key = next(iter(active_imgs.keys()))
                img_tensor = active_imgs[key][i, ..., 0]
                if is_16:
                    data = (img_tensor.cpu().numpy() * 65535).clip(0, 65535).astype(np.uint16)
                    mode = 'I;16'
                else:
                    data = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    mode = 'L'
            else:
                r = image_R[i,...,0] if image_R is not None else torch.zeros((h,w))
                g = image_G[i,...,0] if image_G is not None else torch.zeros((h,w))
                b = image_B[i,...,0] if image_B is not None else torch.zeros((h,w))
                if image_A is not None:
                    packed = torch.stack((r, g, b, image_A[i,...,0]), dim=-1)
                    mode = 'RGBA'
                else:
                    packed = torch.stack((r, g, b), dim=-1)
                    mode = 'RGB'
                if is_16:
                    data = (packed.cpu().numpy() * 65535).clip(0, 65535).astype(np.uint16)
                else:
                    data = (packed.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            img_pil = Image.fromarray(data, mode=mode)
            fname = f"{file_name}_{i:03}.png"
            img_pil.save(os.path.join(path, fname), pnginfo=metadata)
            results.append({"filename": fname, "subfolder": folder_name, "type": "output"})
        return {"ui": {"images": results}}