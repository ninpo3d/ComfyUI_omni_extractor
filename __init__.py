import os
import folder_paths

# --- Path registration for portability ---
# Ensures the 'examples' folder is discoverable by ComfyUI's internal path system
base_path = os.path.dirname(os.path.realpath(__file__))
examples_path = os.path.join(base_path, "examples")

# Register the local examples directory as a global image path
if os.path.exists(examples_path):
    folder_paths.add_model_folder_path("image", examples_path)

# --- Node Mapping ---
from .nodes import RampExtractor, GradientPreview, AssetPacker

NODE_CLASS_MAPPINGS = {
    "RampExtractor": RampExtractor,
    "GradientPreview": GradientPreview,
    "AssetPacker": AssetPacker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RampExtractor": "Omni-Ramp (Extractor)",
    "GradientPreview": "Gradient Preview",
    "AssetPacker": "Asset Packer"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']