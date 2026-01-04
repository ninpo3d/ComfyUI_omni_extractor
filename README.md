# Omni-Extractor for ComfyUI (2026)
A set of specialized nodes designed for remastering and optimizing texture data into a modern, decoupled format: **State Atlas & 16-bit Topology**.

## The Vision: Signal Reconstruction
This project isn't just about textures; it's about **Data Disentanglement**. By separating an asset's **Structural Topology** (Spatial Index) from its **State Atlas** (Signal Data), we achieve up to **80% VRAM reduction** while solving legacy "energy loss" issues in VFX masks.

## Why it matters
* **16-bit Precision**: We use **I;16** format for topology, providing **65,536 steps** of precision to ensure zero-banding in gradients and normal maps.
* **Rec.709 Fix**: Bypasses legacy luminosity formulas that create "dirty" artifacts in high-energy VFX like blue plasma or fire.
* **Hardware Native**: Designed to leverage **GPU Texture Management Unit (TMU)** for native bilinear interpolation during reconstruction.



## Quick Reconstruction (HLSL/Engine)
The beauty of the protocol is its simplicity. To restore the data in any engine, you only need two lines of code:

```hlsl
// 16-bit Topology serves as a high-precision UV coordinate
float latentCoord = TopologyMap.Sample(MapSampler, UV).r;
float3 finalSignal = StateAtlas.Sample(AtlasSampler, float2(latentCoord, 0.5)).rgb;
return pow(finalSignal, 2.2); // Zero-artifact result
```

## Quick Start & "Input" Lifehack
1. **Load Workflow**: Drag & drop `examples/Example_Workflow.json` into ComfyUI.
2. **Handle Images**: If the example images don't appear automatically, **simply copy the contents of the `examples/` folder to your `ComfyUI/input` directory**. This is a standard workaround for ComfyUI's path handling in legacy environments.
3. **Infinite Variations**: Extracted **State Atlases** are independent—apply them to any other topology masks to generate infinite visual variations for free.

## Core Nodes
* **Omni-Ramp (Extractor)**: Deconstructs source textures into a 16-bit **Structural Topology** mask and a **State Atlas** (1D Ramp) using K-Means clustering.
* **Gradient Preview**: A validation tool for real-time reconstruction preview within ComfyUI.
* **Asset Packer**: Handles final export, enforcing **16-bit (L/I;16)** precision for masks and 8-bit for atlases to optimize bandwidth.

## Installation
1. Clone or copy the `ComfyUI_omni_extractor` folder into your `ComfyUI/custom_nodes/` directory.
2. Restart ComfyUI.
3. Find the nodes under the **Omni-Extractor** category.

## Future Horizon: Semantic Video Compression
The principles of **Omni-Extraction** are natively scalable to temporal data. We envision a transition toward **Semantic Video Compression**:
* **Structural Stream**: Transmitting high-fidelity 16-bit Topology for motion and form.
* **State Stream**: Updating the Atlas at a lower frequency, drastically reducing bandwidth.
* **AI-Native**: A foundational format for future neural decoders to restore visual fidelity with extreme efficiency.

---
*Developed by Yurii Borodin (@ninpo3d). Technical Artist @ Ninpo 3D.*
