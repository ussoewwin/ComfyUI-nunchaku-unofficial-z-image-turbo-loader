<div align="center" id="nunchaku_logo">
  <img src="https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/logo/v2/nunchaku-compact-transparent.png" alt="logo" width="220"></img>
</div>

> **⚠️ Warning**: This is an **unofficial** test version of the node. It may not work correctly depending on your environment.

This is an **unofficial** model loader for **Nunchaku Z Image Turbo**, based on [ComfyUI-nunchaku](https://github.com/nunchaku-tech/ComfyUI-nunchaku) with custom additions.

## Nodes

### Nunchaku-ussoewwin Z-Image-Turbo DiT Loader

A ComfyUI node for loading Nunchaku-quantized Z-Image-Turbo models. This node provides support for loading 4-bit quantized Z-Image-Turbo models that have been processed using SVDQuant quantization.

#### Features

- **Model Loading**: Loads Nunchaku-quantized Z-Image-Turbo diffusion transformer models
- **CPU Offloading**: Automatic or manual CPU offloading support to reduce VRAM usage
- **Memory Management**: Configurable GPU memory usage with transformer block offloading options
- **Hardware Compatibility**: Automatic hardware compatibility checks for quantization support
- **Precision Support**: Supports both INT4 and FP4 quantization precisions

#### Input Parameters

**Required:**
- **model_name** (string): The filename of the Nunchaku Z-Image-Turbo model to load. Models should be placed in ComfyUI's `models/diffusion_models/` directory.

**Optional:**
- **cpu_offload** (string, default: "auto"): Controls CPU offloading behavior
  - `"auto"`: Automatically enables CPU offload if GPU memory is less than 15GB
  - `"enable"`: Force enables CPU offload regardless of GPU memory
  - `"disable"`: Disables CPU offload, keeping the entire model on GPU
  
- **num_blocks_on_gpu** (integer, default: 1, range: 1-60): When CPU offload is enabled, specifies how many transformer blocks remain on GPU memory. Increasing this value decreases CPU RAM usage but increases GPU memory usage.

- **use_pin_memory** (string, default: "disable"): Controls pinned memory usage for CPU-GPU transfers
  - `"enable"`: Uses pinned memory for faster CPU-GPU data transfer (increases system memory usage)
  - `"disable"`: Uses regular memory

#### Output

- **MODEL**: Returns a ComfyUI model object that can be connected to other ComfyUI nodes for image generation workflows.

#### Technical Details

The node implements the following loading process:

1. **State Dictionary Loading**: Loads the model weights and metadata from the specified file
2. **Config Parsing**: Extracts quantization configuration (precision, rank, skip_refiners) from metadata
3. **Model Building**: Constructs the `NunchakuZImageTransformer2DModel` from the configuration
4. **Quantization Patching**: Applies quantization patches to the model based on the quantization config
5. **Scale Key Patching**: Patches scale keys for proper weight scaling
6. **Model Wrapping**: Wraps the model in ComfyUI's `NunchakuModelPatcher` for proper memory management

#### Model Architecture

- **Base Architecture**: Z-Image-Turbo is based on Lumina2 architecture
- **Transformer**: Uses `NunchakuZImageTransformer2DModel` instead of standard NextDiT
- **Dimension**: Z-Image-Turbo uses a dimension of 3840
- **Model Type**: FLOW-type model (uses Flow Matching sampling)

#### Usage Notes

- Ensure you have the required Nunchaku package installed (version >= 1.0.0)
- Models must be quantized using SVDQuant and saved with proper metadata
- The node automatically detects and handles different quantization precisions (INT4, FP4)
- CPU offload can significantly reduce VRAM usage but may increase inference time due to CPU-GPU transfers
- For best performance on high-end GPUs (>=15GB), disable CPU offload

#### Example Workflow

1. Place your Nunchaku-quantized Z-Image-Turbo model in `models/diffusion_models/`
2. Add the "Nunchaku-ussoewwin Z-Image-Turbo DiT Loader" node to your workflow
3. Select your model from the dropdown
4. Configure CPU offload based on your GPU memory
5. Connect the MODEL output to your sampling nodes

## License

Licensed under the Apache License, Version 2.0. See [LICENCE.txt](LICENCE.txt) for details.
