# Omni-Extractor for ComfyUI (2026)
A set of specialized nodes designed for remastering and optimizing legacy assets into a modern, decoupled format: **Spectral DNA**.

## The Vision: Beyond Pixels
This project isn't just about textures; it's about **Data Disentanglement**. By separating an asset's structural topology from its spectral state, we achieve up to **80% VRAM/Disk reduction** while solving legacy "energy loss" issues in VFX masks.

## Installation
1. Clone or copy the `ComfyUI_omni_extractor` folder into your `ComfyUI/custom_nodes/` directory.
2. Restart ComfyUI.
3. Find the nodes under the **Omni-Extractor** category.

## Core Nodes
* **Omni-Ramp (Extractor)**: Deconstructs source color textures into two separate components: a high-precision grayscale mask (Topology) and a 1D color LUT (Spectral DNA).
* **Gradient Preview**: A validation tool that re-applies the extracted DNA onto the topology mask for real-time reconstruction preview.
* **Asset Packer**: Handles final export with 16-bit (L/I;16) precision to ensure zero-banding.

## Future Horizon: Semantic Video Compression
The principles of **Spectral DNA** are natively scalable to temporal data. We envision a transition toward **Semantic Video Compression**:
* **Structural Stream**: Transmitting high-fidelity 16-bit Topology for motion and form.
* **Spectral Stream**: Updating the DNA (state/color) at a lower frequency, drastically reducing bandwidth.
* **AI-Native**: A foundational format for future neural decoders to restore visual fidelity with extreme efficiency.

## Quick Start (Examples)
To see the technology in action, simply drag and drop the `Example_Workflow.json` from the examples folder into ComfyUI.
* **Portable Logic**: Integrated path resolution ensures example images load automatically.
* **Bonus**: Extracted ramps (DNA) are independent and can be applied to any other masks to generate infinite visual variations for free.
