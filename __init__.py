import os

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