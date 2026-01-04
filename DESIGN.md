# Design Notes and Conceptual Background

This document describes the ideas behind the VFX Tools package.
It is not required to use the tools.

---

## Motivation

Modern VFX assets often contain tightly coupled representations:
spatial structure, state, metadata, and presentation are mixed together.

This coupling:
- increases memory usage
- complicates automation
- produces noisy data for downstream AI training

The tools in this package were designed to break this coupling early.

---

## Disentangled Representation

The core idea is to separate:

1. Structural information  
   (spatial layout, topology, ordering)

2. State representation  
   (values, gradients, parameters, spectral data)

By handling these layers independently, the pipeline gains:
- lower VRAM pressure
- deterministic behavior
- cleaner intermediate data

This separation is intentional and enforced by design.

---

## The Algorithmic Core: Spectrum vs. Luma

Traditional VFX pipelines rely on linear conversion (Rec.709) to generate masks:
`Luma = 0.21*R + 0.72*G + 0.07*B`

This approach mathematically "crushes" high-energy phenomena (Fire, Plasma, Magic) because the dominant Red/Blue channels are often weighted down, resulting in loss of perceived energy and detail.

This package introduces **Spectrum-Based Reconstruction**:

1.  **Palette Extraction**: Instead of calculating brightness, the algorithm extracts the dominant chromatic palette of the asset using clustering (K-Means).
2.  **Nearest Neighbor Encoding**: Each pixel is mapped to its closest spectral match in the generated palette.
3.  **Topology Preservation**: This creates a gradient map that preserves the *identity* of the hue (e.g., dark blue gas) rather than just its luminance value.

This shift allows for the preservation of "visual energy" that is typically lost in standard grayscale conversion.

---

## Micro-Pipeline Model

The tools form a minimal but complete pipeline:

1. Operator  
   Transforms structured input into a reduced or projected representation.

2. Controller  
   Observes and constrains the transformation.
   Acts as a validation and decision layer.

3. Pack  
   Encodes the result into a compact, transferable form.

Each stage has a single responsibility.
Stages can be recombined or replaced without breaking the overall model.

---

## Determinism and Reproducibility

Operations are designed to be:
- order-aware
- repeatable
- free of implicit state

This is critical both for production workflows and for dataset generation,
where reproducibility directly affects training stability.

---

## Implications for Dataset Construction

When applied consistently, this approach produces data that is:
- lower dimensional
- structurally explicit
- easier to cluster and analyze

Such datasets are better suited for:
- supervised training
- representation learning
- long-term reuse across models

The tools do not perform training themselves.
They prepare information in a form that makes training cleaner and more predictable.

---

## Closing Note

This package is not intended to define a standard.
It documents one possible way to structure and preserve information.

The design favors clarity over flexibility and longevity over convenience.
