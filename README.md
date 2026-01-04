Omni-Extractor for ComfyUI (2026)
A set of specialized nodes designed for remastering and optimizing legacy assets into a modern, decoupled format.

Installation
Clone or copy the ComfyUI_omni_extractor folder into your ComfyUI/custom_nodes/ directory.

Restart ComfyUI.

Find the nodes under the Omni-Extractor category.

Core Nodes
1. Omni-Ramp (Extractor)
Deconstructs a source color texture into two separate components: a high-precision grayscale mask and a 1D color LUT (Ramp).

Why: To decouple the asset's structure from its color state for ultimate shader flexibility and VRAM efficiency.

2. Gradient Preview
A validation tool that re-applies the extracted color ramp onto the grayscale mask, allowing you to preview the final reconstruction directly within ComfyUI.

3. Asset Packer
Handles the final export to disk with a focus on data integrity.

Adaptive Packing: Automatically switches between lightweight Grayscale (L/I;16) for single inputs and multi-channel containers.

16-bit Support: Crucial: Always use 16-bit Grayscale for masks to ensure zero-banding during spectral reconstruction.

Clean Naming: Saves files to ComfyUI/output/ using exact user-defined paths without unwanted auto-counters.

Quick Start (Examples)
To see the technology in action, simply drag and drop the Example_Workflow.json from the examples folder into ComfyUI.

Portable Logic: Thanks to our integrated path resolution, the example images will load automatically from the repository folder.

Bonus: Extracted ramps are independent and can be applied to any other grayscale masks to generate infinite visual variations for free.