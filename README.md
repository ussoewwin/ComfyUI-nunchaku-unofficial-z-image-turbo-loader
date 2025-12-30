<div align="center" id="nunchaku_logo">
  <img src="https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/logo/v2/nunchaku-compact-transparent.png" alt="logo" width="220"></img>
</div>

<div align="center">

<h2>⚠️ <span style="font-size: 1.5em; color: #ff6b6b;">WARNING</span></h2>

<p style="font-size: 1.2em; font-weight: bold; color: #d63031;">
This is an <strong>UNOFFICIAL</strong> test version of the node.<br>
It may not work correctly depending on your environment.<br>
This repository may be closed or archived after the official node is released.
</p>

</div>

This is an **unofficial** model loader for **Nunchaku Z Image Turbo**, based on [ComfyUI-nunchaku](https://github.com/nunchaku-tech/ComfyUI-nunchaku) with custom additions.

## Changelog

### Version 2.0

- Added SDXL DIT Loader support
- Added SDXL LoRA support
- Added ControlNet support for SDXL models
- See [Release Notes v2.0](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-z-image-turbo-loader/releases/tag/2.0) for details

### Version 1.1

- Added Diffsynth ControlNet support for Z-Image-Turbo models
  - Note: Does not work with standard model patch loader. Requires a custom node developed by the author.
- See [Release Notes v1.1](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-z-image-turbo-loader/releases/tag/1.1) for details

### 2025-12-25

- Fixed import error for `NunchakuZImageDiTLoader` node by improving alternative import method with better path resolution (see [Issue #1](issues/1))

## Requirements

**Nunchaku library**: You **MUST** have the Nunchaku library version from **December 24, 2025** (2025-12-24) installed. This is a hard requirement - other versions will not work.

**Pre-built package**: For Windows with Python 3.13 and PyTorch 2.9.1+cu130, a pre-built package is available at [ussoewwin/nunchaku-build-on-cu130-windows](https://huggingface.co/ussoewwin/nunchaku-build-on-cu130-windows). This package includes version 1.1.0dev20251224.

**Building from source**: If you use a different environment, you need to build the Nunchaku library from source. The build instructions are not provided in this repository; please refer to the official Nunchaku repository for build documentation.

## Nodes

### Nunchaku-ussoewwin Z-Image-Turbo DiT Loader

A ComfyUI node for loading Nunchaku-quantized Z-Image-Turbo models. This node provides support for loading 4-bit quantized Z-Image-Turbo models that have been processed using SVDQuant quantization.

<img src="png/node.png" alt="Nunchaku-ussoewwin Z-Image-Turbo DiT Loader Node" width="400">

#### Features

- **Model Loading**: Loads Nunchaku-quantized Z-Image-Turbo diffusion transformer models
- **CPU Offloading**: Automatic or manual CPU offloading support to reduce VRAM usage
- **Memory Management**: Configurable GPU memory usage with transformer block offloading options
- **Hardware Compatibility**: Automatic hardware compatibility checks for quantization support
- **Precision Support**: Supports both INT4 and FP4 quantization precisions


## License

Licensed under the Apache License, Version 2.0. See [LICENCE.txt](LICENCE.txt) for details.
