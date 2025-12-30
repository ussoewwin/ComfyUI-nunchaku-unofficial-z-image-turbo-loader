"""
This module provides the :class:`NunchakuSDXLDiTLoader` class for loading Nunchaku SDXL models.
"""

import json
import logging
import os

import comfy.model_management
import comfy.utils
import comfy.sd
import torch
from comfy import model_detection, model_management
from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel, convert_sdxl_state_dict
from nunchaku.models.linear import SVDQW4A4Linear
from nunchaku.utils import check_hardware_compatibility, get_precision, get_precision_from_quantization_config

from ...model_configs.sdxl import NunchakuSDXL
from ...model_patcher import NunchakuModelPatcher
from ..utils import get_filename_list, get_full_path_or_raise

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_CLIP_DEBUG_CONTEXT = False
_WARNED_TEXT_PROJ_IDENTITY: set[str] = set()

def _clip_debug_enabled() -> bool:
    # Prefer dedicated flag; fall back to global SDXL debug.
    if _CLIP_DEBUG_CONTEXT:
        return True
    v = os.environ.get("NUNCHAKU_SDXL_CLIP_DEBUG", "")
    if v in ("1", "true", "True", "yes", "YES", "on", "ON"):
        return True
    v2 = os.environ.get("NUNCHAKU_SDXL_DEBUG", "")
    return v2 in ("1", "true", "True", "yes", "YES", "on", "ON")

def _clip_debug(msg: str) -> None:
    if _clip_debug_enabled():
        # print() for ComfyUI console visibility
        print(f"[NUNCHAKU_SDXL_CLIP_DEBUG] {msg}")

def _get_sdxl_clip_type() -> tuple[comfy.sd.CLIPType, str]:
    """
    ComfyUI selects SDXL dual-CLIP model primarily by the number of TE state_dicts (len==2),
    not by CLIPType. Some ComfyUI versions do not define CLIPType.SDXL at all.
    """
    if hasattr(comfy.sd.CLIPType, "SDXL"):
        return getattr(comfy.sd.CLIPType, "SDXL"), "CLIPType.SDXL exists"
    return comfy.sd.CLIPType.STABLE_DIFFUSION, "CLIPType.SDXL not present; SDXLClipModel is auto-selected for 2 TEs"

def _summarize_clip_sd(sd: dict[str, torch.Tensor], label: str) -> None:
    if not _clip_debug_enabled():
        return
    keys = list(sd.keys())
    keys_set = set(keys)
    bigg_marker = "text_model.encoder.layers.30.mlp.fc1.weight"  # used by comfy/sdxl_clip.py to identify CLIP-G
    l_marker = "text_model.embeddings.token_embedding.weight"
    _clip_debug(
        f"{label}: num_keys={len(keys)} "
        f"has_text_model={any(k.startswith('text_model.') for k in keys)} "
        f"has_model={any(k.startswith('model.') for k in keys)} "
        f"has_resblocks={any('transformer.resblocks.' in k for k in keys)} "
        f"has_bigg_marker={bigg_marker in keys_set} "
        f"has_l_marker={l_marker in keys_set}"
    )
    head = sorted(keys)[:25]
    _clip_debug(f"{label}: head_keys={head}")

def _normalize_comfy_clip_sd(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Normalize common SDXL CLIP extraction variants into ComfyUI expected keys.

    Known issues:
    - Some extractors save CLIP-L with a double prefix: "text_model.text_model.*"
    - Some save the projection as "text_model.text_projection.weight" instead of "text_projection.weight"
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        nk = k

        # Fix double-prefix variants (most common for CLIP-L extractions)
        if nk.startswith("text_model.text_model."):
            nk = "text_model." + nk[len("text_model.text_model.") :]

        # Fix projection naming variants
        if nk.startswith("text_model.text_projection."):
            nk = "text_projection." + nk[len("text_model.text_projection.") :]

        out[nk] = v

    return out

def _ensure_text_projection_weight(
    sd: dict[str, torch.Tensor],
    *,
    label: str = "clip",
) -> tuple[dict[str, torch.Tensor], bool]:
    """
    ComfyUI's CLIP text models expect 'text_projection.weight' (Linear bias=False).
    Some extracted CLIP-L files omit this key, which causes large 'clip missing' logs and
    leaves projection at default initialization (often zeros).

    IMPORTANT:
    Filling identity is a *compatibility fallback* only. It makes the model "run" but it is
    NOT guaranteed to be equivalent to standard SDXL because text_projection is typically a
    learned mapping.

    Behavior is controlled by env:
      NUNCHAKU_SDXL_CLIP_TEXT_PROJECTION_MISSING = "identity" | "skip"
        - identity (default): fill identity and warn once
        - skip: do nothing (will likely behave worse than identity due to default init)
    """
    if "text_projection.weight" in sd:
        return sd, False

    mode = str(os.getenv("NUNCHAKU_SDXL_CLIP_TEXT_PROJECTION_MISSING", "identity")).strip().lower()
    # Safety policy: NEVER hard-fail here. Missing projection is common in extracted files;
    # running (with a loud warning) is preferred over stopping the workflow.
    # If user sets a "strict" value by mistake, treat it as identity with an extra warning.
    if mode in ("error", "raise", "strict"):
        try:
            key = f"{label}:strict_requested"
            if key not in _WARNED_TEXT_PROJ_IDENTITY:
                _WARNED_TEXT_PROJ_IDENTITY.add(key)
                print(
                    "[NUNCHAKU_SDXL_CLIP_WARNING] "
                    f"{label}: NUNCHAKU_SDXL_CLIP_TEXT_PROJECTION_MISSING='{mode}' was requested, "
                    "but hard-fail is disabled. Falling back to identity to keep running."
                )
        except Exception:
            pass
        mode = "identity"
    if mode in ("skip", "none", "off"):
        # Still warn once: this is almost always a broken extraction.
        try:
            key = f"{label}:skip"
            if key not in _WARNED_TEXT_PROJ_IDENTITY:
                _WARNED_TEXT_PROJ_IDENTITY.add(key)
                print(
                    "[NUNCHAKU_SDXL_CLIP_WARNING] "
                    f"{label}: 'text_projection.weight' is missing; leaving it uninitialized (mode=skip). "
                    "This is NOT equivalent to standard SDXL and may degrade prompt behavior."
                )
        except Exception:
            pass
        return sd, False

    tok_key = "text_model.embeddings.token_embedding.weight"
    if tok_key not in sd:
        return sd, False

    tok = sd[tok_key]
    if not torch.is_tensor(tok) or tok.ndim != 2:
        return sd, False

    hidden = int(tok.shape[1])
    # Keep on CPU; safetensors load returns CPU tensors by default via comfy.utils.load_torch_file
    eye = torch.eye(hidden, dtype=tok.dtype, device=tok.device)
    out = dict(sd)
    out["text_projection.weight"] = eye
    # Warn once per label. This is a "runs but not equivalent" landmine.
    try:
        key = f"{label}:identity:{hidden}:{str(tok.dtype)}"
        if key not in _WARNED_TEXT_PROJ_IDENTITY:
            _WARNED_TEXT_PROJ_IDENTITY.add(key)
            print(
                "[NUNCHAKU_SDXL_CLIP_WARNING] "
                f"{label}: 'text_projection.weight' was missing; filled identity ({hidden}x{hidden}). "
                "This makes it run, but is NOT guaranteed equivalent to standard SDXL. "
                "If you have a proper SDXL CLIP file, use that instead."
            )
    except Exception:
        pass
    return out, True

def _find_openai_clip_prefix(sd: dict[str, torch.Tensor]) -> str | None:
    """
    Detect OpenAI/OpenCLIP-style text tower keys with an arbitrary prefix.
    Returns the prefix (including trailing ".") up to "transformer.resblocks.".

    Examples:
      - "model.transformer.resblocks.0..."           -> "model."
      - "text_model.transformer.resblocks.0..."      -> "text_model."
      - "cond_stage_model.model.transformer..."      -> "cond_stage_model.model."
    """
    needle = "transformer.resblocks."
    for k in sd.keys():
        i = k.find(needle)
        if i >= 0:
            return k[:i]
    return None

def _is_openai_clip_style(sd: dict[str, torch.Tensor]) -> bool:
    # Common OpenCLIP/OpenAI-CLIP style keys (text tower) with arbitrary prefix.
    prefix = _find_openai_clip_prefix(sd)
    if prefix is None:
        return False
    return (prefix + "token_embedding.weight") in sd or (prefix + "positional_embedding") in sd

def _convert_openai_clip_sd_to_comfy(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert OpenAI/OpenCLIP-style text tower state_dict (keys like "model.transformer.resblocks.*")
    into ComfyUI clip_model.py format (keys like "text_model.encoder.layers.*").

    This is needed for some SDXL extractions where CLIP-G is saved in OpenCLIP naming.
    """
    prefix = _find_openai_clip_prefix(sd)
    if prefix is None:
        return sd

    _clip_debug(f"OpenAI/OpenCLIP style detected. prefix='{prefix}'")

    # Infer embed_dim from in_proj_weight
    sample_k = None
    for k in sd.keys():
        if k.startswith(prefix + "transformer.resblocks.") and k.endswith(".attn.in_proj_weight"):
            sample_k = k
            break
    if sample_k is None:
        # Can't convert; return as-is
        return sd
    in_proj_w = sd[sample_k]
    embed_dim = in_proj_w.shape[1]

    out: dict[str, torch.Tensor] = {}

    # Embeddings + final LN + projection
    tok_k = prefix + "token_embedding.weight"
    pos_k = prefix + "positional_embedding"
    ln_w_k = prefix + "ln_final.weight"
    ln_b_k = prefix + "ln_final.bias"

    if tok_k in sd:
        out["text_model.embeddings.token_embedding.weight"] = sd[tok_k]
    if pos_k in sd:
        # positional_embedding is (seq, dim) already, matches Embedding.weight shape
        out["text_model.embeddings.position_embedding.weight"] = sd[pos_k]
    if ln_w_k in sd:
        out["text_model.final_layer_norm.weight"] = sd[ln_w_k]
    if ln_b_k in sd:
        out["text_model.final_layer_norm.bias"] = sd[ln_b_k]

    # Projection can appear with multiple key spellings depending on extractor.
    if (prefix + "text_projection") in sd:
        out["text_projection.weight"] = sd[prefix + "text_projection"]
    elif (prefix + "text_projection.weight") in sd:
        out["text_projection.weight"] = sd[prefix + "text_projection.weight"]
    elif "text_projection.weight" in sd:
        out["text_projection.weight"] = sd["text_projection.weight"]
    elif (prefix + "text_projection") in sd:
        # ComfyUI uses Linear(..., bias=False) => key is ".weight"
        out["text_projection.weight"] = sd[prefix + "text_projection"]

    # Transformer blocks
    # OpenAI naming:
    #   model.transformer.resblocks.{i}.ln_1.(weight|bias)
    #   model.transformer.resblocks.{i}.attn.in_proj_weight / in_proj_bias
    #   model.transformer.resblocks.{i}.attn.out_proj.(weight|bias)
    #   model.transformer.resblocks.{i}.ln_2.(weight|bias)
    #   model.transformer.resblocks.{i}.mlp.c_fc.(weight|bias)
    #   model.transformer.resblocks.{i}.mlp.c_proj.(weight|bias)
    #
    # ComfyUI naming:
    #   text_model.encoder.layers.{i}.layer_norm1.(weight|bias)
    #   text_model.encoder.layers.{i}.self_attn.(q_proj|k_proj|v_proj).(weight|bias)
    #   text_model.encoder.layers.{i}.self_attn.out_proj.(weight|bias)
    #   text_model.encoder.layers.{i}.layer_norm2.(weight|bias)
    #   text_model.encoder.layers.{i}.mlp.fc1.(weight|bias)
    #   text_model.encoder.layers.{i}.mlp.fc2.(weight|bias)
    #
    # We split in_proj into q/k/v.
    block_prefix = prefix + "transformer.resblocks."
    for k, v in sd.items():
        if not k.startswith(block_prefix):
            continue
        rest = k[len(block_prefix) :]
        parts = rest.split(".")
        if len(parts) < 3:
            continue
        layer_idx = parts[0]
        tail = ".".join(parts[1:])

        prefix = f"text_model.encoder.layers.{layer_idx}."

        if tail.startswith("ln_1."):
            out[prefix + "layer_norm1." + tail[len("ln_1.") :]] = v
        elif tail.startswith("ln_2."):
            out[prefix + "layer_norm2." + tail[len("ln_2.") :]] = v
        elif tail.startswith("mlp.c_fc."):
            out[prefix + "mlp.fc1." + tail[len("mlp.c_fc.") :]] = v
        elif tail.startswith("mlp.c_proj."):
            out[prefix + "mlp.fc2." + tail[len("mlp.c_proj.") :]] = v
        elif tail.startswith("attn.out_proj."):
            out[prefix + "self_attn.out_proj." + tail[len("attn.out_proj.") :]] = v
        elif tail == "attn.in_proj_weight":
            # (3*D, D) => q/k/v: (D, D)
            out[prefix + "self_attn.q_proj.weight"] = v[:embed_dim, :]
            out[prefix + "self_attn.k_proj.weight"] = v[embed_dim : 2 * embed_dim, :]
            out[prefix + "self_attn.v_proj.weight"] = v[2 * embed_dim :, :]
        elif tail == "attn.in_proj_bias":
            out[prefix + "self_attn.q_proj.bias"] = v[:embed_dim]
            out[prefix + "self_attn.k_proj.bias"] = v[embed_dim : 2 * embed_dim]
            out[prefix + "self_attn.v_proj.bias"] = v[2 * embed_dim :]

    # Keep any tokenizer/config metadata keys if present (safe to ignore if unused)
    for k, v in sd.items():
        if k in {"vocab", "merges", "tokenizer", "tokenizer.json", "merges.txt", "vocab.json"}:
            out[k] = v

    return out

def _get_attr_no_warn(obj, name: str, default=None):
    """
    Some ComfyUI config objects implement __getattr__ that emits warnings when probing missing attrs.
    Avoid that by using object.__getattribute__ directly.
    """
    try:
        return object.__getattribute__(obj, name)
    except Exception:
        return default


_SVDQ_PARAM_LIKE_NAMES = {
    # SVDQW4A4Linear nn.Parameter names (saved directly as "<name>", not "<name>.weight")
    "qweight",
    "bias",
    "wscales",
    "smooth_factor",
    "smooth_factor_orig",
    "proj_down",
    "proj_up",
    "wcscales",
}


def _infer_rank_from_state_dict(sd: dict[str, torch.Tensor]) -> int | None:
    # Prefer new naming
    for k, v in sd.items():
        if k.endswith(".proj_down") and hasattr(v, "shape") and len(v.shape) == 2:
            return int(v.shape[1])
    # Legacy naming
    for k, v in sd.items():
        if (k.endswith(".lora_down") or k.endswith(".lora_down.weight")) and hasattr(v, "shape") and len(v.shape) == 2:
            return int(v.shape[1])
    return None


def _normalize_nunchaku_sdxl_state_dict_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Normalize common checkpoint naming variants to match current Nunchaku SDXL modules.
    - legacy LoRA naming: *.lora_down / *.lora_up -> *.proj_down / *.proj_up
    - legacy module saves: *.proj_down.weight -> *.proj_down (and similar for other SVDQ params)
    """
    # Only run convert if legacy keys exist in transformer blocks
    needs_convert = any(
        ".transformer_blocks." in k and (".lora_down" in k or ".lora_up" in k or ".smooth_orig" in k)
        for k in sd.keys()
    )
    already_normalized = any(
        ".transformer_blocks." in k and (".proj_down" in k or ".proj_up" in k or ".smooth_factor" in k)
        for k in sd.keys()
    )

    out = convert_sdxl_state_dict(sd) if (needs_convert and not already_normalized) else dict(sd)

    normalized: dict[str, torch.Tensor] = {}
    for k, v in out.items():
        new_k = k
        # Fix older checkpoints that saved SVDQ parameters as submodules with ".weight"
        if ".transformer_blocks." in new_k and new_k.endswith(".weight"):
            base = new_k.rsplit(".", 1)[0]  # strip ".weight"
            last = base.split(".")[-1]
            if last in _SVDQ_PARAM_LIKE_NAMES:
                new_k = base
        normalized[new_k] = v

    return normalized


def _pop_and_apply_svdq_wtscale(unet: torch.nn.Module, sd: dict[str, torch.Tensor]) -> None:
    """
    Some checkpoints include "<module>.wtscale" as a tensor key, but SVDQW4A4Linear keeps wtscale
    as a Python float (not part of state_dict). Pop and assign it to avoid "Unexpected key(s)".
    Also ensure wcscales exists when nvfp4 is used.
    """
    for n, m in unet.named_modules():
        if not isinstance(m, SVDQW4A4Linear):
            continue
        if getattr(m, "wtscale", None) is None:
            continue

        key = f"{n}.wtscale"
        if key in sd:
            val = sd.pop(key)
            try:
                m.wtscale = float(val.item()) if torch.is_tensor(val) else float(val)
            except Exception:
                pass
        else:
            m.wtscale = 1.0

        wc_key = f"{n}.wcscales"
        if getattr(m, "wcscales", None) is not None and wc_key not in sd:
            sd[wc_key] = torch.ones_like(m.wcscales)


def _fill_missing_svdq_proj(unet: torch.nn.Module, sd: dict[str, torch.Tensor]) -> None:
    """
    Some checkpoints omit proj_down/proj_up for SVDQ layers (while still providing qweight/wscales/smooth).
    In that case, we fill missing proj_* with zeros so the model is well-defined (low-rank contribution = 0).
    """
    for n, m in unet.named_modules():
        if not isinstance(m, SVDQW4A4Linear):
            continue
        down_k = f"{n}.proj_down"
        up_k = f"{n}.proj_up"
        if down_k not in sd:
            sd[down_k] = torch.zeros_like(m.proj_down)
        if up_k not in sd:
            sd[up_k] = torch.zeros_like(m.proj_up)


def load_diffusion_model_state_dict(
    sd: dict[str, torch.Tensor], metadata: dict[str, str] = {}, model_options: dict = {}
):
    """
    Load a Nunchaku-quantized SDXL diffusion model.

    This function follows the same pattern as NunchakuSDXLUNet2DConditionModel.from_pretrained():
    1. Build UNet from config
    2. Patch model with quantization
    3. Convert state dict
    4. Load state dict

    Parameters
    ----------
    sd : dict[str, torch.Tensor]
        The state dictionary of the model.
    metadata : dict[str, str], optional
        Metadata containing quantization configuration (default is empty dict).
    model_options : dict, optional
        Additional model options such as dtype or custom operations.

    Returns
    -------
    comfy.model_patcher.ModelPatcher
        The patched and loaded SDXL model ready for inference.
    """
    quantization_config = json.loads(metadata.get("quantization_config", "{}"))

    # Determine precision from metadata if possible.
    # Nunchaku SDXL unet safetensors commonly use:
    #   {"rank": 128, "precision": "nvfp4"}
    # Some other model types may use a nested structure requiring get_precision_from_quantization_config().
    precision_from_metadata = None
    if isinstance(quantization_config, dict):
        if "precision" in quantization_config:
            precision_from_metadata = quantization_config.get("precision")
            if precision_from_metadata == "fp4":
                precision_from_metadata = "nvfp4"
        elif "weight" in quantization_config:
            precision_from_metadata = get_precision_from_quantization_config(quantization_config)

    precision_auto = get_precision()
    if precision_auto == "fp4":
        precision_auto = "nvfp4"
    precision = precision_from_metadata if precision_from_metadata else precision_auto

    rank = quantization_config.get("rank", None)
    if rank is None:
        inferred_rank = _infer_rank_from_state_dict(sd)
        rank = inferred_rank if inferred_rank is not None else 32

    dtype = model_options.get("dtype", None)

    # Allow loading unets from checkpoint files
    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    temp_sd = comfy.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        sd = temp_sd

    parameters = comfy.utils.calculate_parameters(sd)
    weight_dtype = comfy.utils.weight_dtype(sd)

    load_device = model_management.get_torch_device()

    # Hardware compatibility check only supports some quantization_config formats (with nested "weight" info).
    # Many Nunchaku SDXL safetensors only provide {"rank": ..., "precision": ...}.
    if isinstance(quantization_config, dict) and "weight" in quantization_config:
        check_hardware_compatibility(quantization_config, load_device)

    offload_device = model_management.unet_offload_device()

    # Use _build_model pattern to get UNet, state_dict, and metadata (same as from_pretrained)
    # Since we already have sd and metadata from ComfyUI, we build from config directly
    # Following the exact pattern in NunchakuSDXLUNet2DConditionModel.from_pretrained()
    # _build_model does: from_config(config).to(torch_dtype) in meta device context
    config = json.loads(metadata.get("config", "{}"))

    torch_dtype = dtype if dtype is not None else torch.bfloat16

    # Build UNet from config (exactly like _build_model in NunchakuModelLoaderMixin)
    # _build_model does: with torch.device("meta"): transformer = cls.from_config(config).to(torch_dtype)
    with torch.device("meta"):
        unet = NunchakuSDXLUNet2DConditionModel.from_config(config).to(torch_dtype)

    # Precision must be consistent with the checkpoint.
    # If the checkpoint provides quantization_config, prefer it over environment auto-detection.
    # Use checkpoint precision if available; otherwise fall back to environment auto-detection.
    precision_use = precision

    unet._patch_model(precision=precision_use, rank=rank)
    unet = unet.to_empty(device=load_device)

    # Convert/normalize state dict (handles legacy LoRA naming + some older serialization variants)
    converted_sd = _normalize_nunchaku_sdxl_state_dict_keys(sd)
    _pop_and_apply_svdq_wtscale(unet, converted_sd)
    _fill_missing_svdq_proj(unet, converted_sd)

    # Load state dict
    missing, unexpected = unet.load_state_dict(converted_sd, strict=False)
    if missing:
        logger.warning(f"SDXL UNet load_state_dict missing keys: {len(missing)} (showing up to 20): {missing[:20]}")
    if unexpected:
        logger.warning(
            f"SDXL UNet load_state_dict unexpected keys: {len(unexpected)} (showing up to 20): {unexpected[:20]}"
        )

    # Create model config and wrap in ComfyUI model structure
    model_config = NunchakuSDXL(
        {
            "model_channels": 320,
            "use_linear_in_transformer": True,
            "transformer_depth": [0, 0, 2, 2, 10, 10],
            "context_dim": 2048,
            "adm_in_channels": 2816,
            "use_temporal_attention": False,
            "rank": rank,
            "precision": precision_use,
            "transformer_offload_device": None,
        }
    )
    model_config.optimizations["fp8"] = False

    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    scaled_fp8 = _get_attr_no_warn(model_config, "scaled_fp8", None)
    if scaled_fp8 is not None:
        weight_dtype = None

    if dtype is None:
        unet_dtype = model_management.unet_dtype(
            model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype
        )
    else:
        unet_dtype = dtype

    manual_cast_dtype = model_management.unet_manual_cast(
        unet_dtype, load_device, model_config.supported_inference_dtypes
    )
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
    model_config.custom_operations = model_options.get("custom_operations", model_config.custom_operations)
    if model_options.get("fp8_optimizations", False):
        model_config.optimizations["fp8"] = True

    # Create model wrapper. Our model_base/sdxl.py disables UNet creation, so we must not
    # pass fake tensors/shapes here (dummy_sd causes shape inference corruption).
    model = model_config.get_model({}, "", load_device)
    # Set the pre-built and patched UNet
    model.diffusion_model = unet
    model.diffusion_model.eval()
    if comfy.model_management.force_channels_last():
        model.diffusion_model.to(memory_format=torch.channels_last)
    model = model.to(offload_device)
    return NunchakuModelPatcher(model, load_device=load_device, offload_device=offload_device)


class NunchakuSDXLDiTLoader:
    """
    Loader for Nunchaku SDXL models.

    Attributes
    ----------
    RETURN_TYPES : tuple
        Output types for the node ("MODEL", "CLIP").
    FUNCTION : str
        Name of the function to call ("load_model").
    CATEGORY : str
        Node category ("Nunchaku-ussoewwin").
    TITLE : str
        Node title ("Nunchaku-ussoewwin SDXL DiT Loader").
    """

    @classmethod
    def INPUT_TYPES(s):
        """
        Define the input types and tooltips for the node.

        Returns
        -------
        dict
            A dictionary specifying the required inputs and their descriptions for the node interface.
        """
        return {
            "required": {
                "model_name": (
                    get_filename_list("diffusion_models"),
                    {"tooltip": "The Nunchaku SDXL UNet model file (often UNet-only)."},
                ),
            },
            "optional": {
                "debug_model": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Print SDXL apply_model / ControlNet debug info during sampling."},
                ),
            },
        }

    # NOTE: This node loads UNet only. CLIP must be loaded via standard loaders.
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Nunchaku-ussoewwin"
    TITLE = "Nunchaku-ussoewwin SDXL DiT Loader"

    def load_model(self, model_name: str, debug_model: bool = False, **kwargs):
        """
        Load the SDXL model from file and return a patched model.
        UNet is loaded from model_name (Nunchaku quantized UNet-only safetensors).

        Parameters
        ----------
        model_name : str
            The filename of the SDXL UNet model to load.

        Returns
        -------
        tuple
            A tuple containing the loaded and patched model.
        """
        model_path = get_full_path_or_raise("diffusion_models", model_name)
        sd, metadata = comfy.utils.load_torch_file(model_path, return_metadata=True)

        # Enable/disable model-side debug prints (ControlNet mapping etc.)
        try:
            from ...model_base import sdxl as model_base_sdxl
            model_base_sdxl.set_nunchaku_sdxl_debug(bool(debug_model))
        except Exception as e:
            logger.debug(f"Failed to set NunchakuSDXL debug flag: {e}")

        model = load_diffusion_model_state_dict(
            sd, metadata=metadata, model_options={}
        )

        return (model,)


class NunchakuSDXLDiTLoaderDualCLIP:
    """
    Loader for Nunchaku SDXL UNet + SDXL Dual CLIP (CLIP-L + CLIP-G).

    Rationale:
    - Nunchaku SVDQ models are UNet-only (no CLIP).
    - Some workflows prefer not to rely on standard checkpoint loaders for CLIP.
    - This node loads UNet from `diffusion_models` and CLIP-L/CLIP-G from `text_encoders`.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (
                    get_filename_list("diffusion_models"),
                    {"tooltip": "Nunchaku SDXL UNet model (UNet-only safetensors)."},
                ),
                "clip_l_name": (
                    get_filename_list("text_encoders"),
                    {"tooltip": "SDXL CLIP-L text encoder (e.g. clip_l.safetensors)."},
                ),
                "clip_g_name": (
                    get_filename_list("text_encoders"),
                    {"tooltip": "SDXL CLIP-G text encoder (ViT-bigG)."},
                ),
            },
            "optional": {
                "device": (["default", "cpu"], {"advanced": True}),
                "debug": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Print detailed CLIP debug info (key detection/convert/probe)."},
                ),
                "debug_model": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Print SDXL apply_model / ControlNet debug info during sampling."},
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_model"
    CATEGORY = "Nunchaku-ussoewwin"
    TITLE = "Nunchaku-ussoewwin SDXL DiT Loader (DualCLIP)"

    def load_model(
        self,
        model_name: str,
        clip_l_name: str,
        clip_g_name: str,
        device: str = "default",
        debug: bool = False,
        debug_model: bool = False,
        **kwargs,
    ):
        # Enable/disable model-side debug prints (ControlNet mapping etc.)
        try:
            from ...model_base import sdxl as model_base_sdxl
            model_base_sdxl.set_nunchaku_sdxl_debug(bool(debug_model))
        except Exception as e:
            logger.debug(f"Failed to set NunchakuSDXL debug flag: {e}")

        global _CLIP_DEBUG_CONTEXT
        _prev_debug = _CLIP_DEBUG_CONTEXT
        _CLIP_DEBUG_CONTEXT = bool(debug)
        try:
            # 1) Load UNet (Nunchaku SDXL).
            model_path = get_full_path_or_raise("diffusion_models", model_name)
            sd, metadata = comfy.utils.load_torch_file(model_path, return_metadata=True)
            model = load_diffusion_model_state_dict(sd, metadata=metadata, model_options={})

            # 2) Load SDXL Dual CLIP (L + G).
            # We load state_dicts ourselves so we can normalize OpenCLIP/OpenAI-style keys
            # (e.g. "model.transformer.resblocks.*") into ComfyUI's expected "text_model.*" format.
            from ..utils import folder_paths

            clip_path_l = get_full_path_or_raise("text_encoders", clip_l_name)
            clip_path_g = get_full_path_or_raise("text_encoders", clip_g_name)

            sd_l = comfy.utils.load_torch_file(clip_path_l, safe_load=True)
            sd_g = comfy.utils.load_torch_file(clip_path_g, safe_load=True)

            _clip_debug(f"clip_l_file='{clip_l_name}' path='{clip_path_l}'")
            _summarize_clip_sd(sd_l, "clip_l:before")
            _clip_debug(f"clip_g_file='{clip_g_name}' path='{clip_path_g}'")
            _summarize_clip_sd(sd_g, "clip_g:before")

            # Normalize common Comfy-format variants (e.g. "text_model.text_model.*")
            sd_l = _normalize_comfy_clip_sd(sd_l)
            sd_g = _normalize_comfy_clip_sd(sd_g)
            _clip_debug("clip_l/clip_g: applied Comfy-format normalization (if needed)")

            if _is_openai_clip_style(sd_l):
                sd_l = _convert_openai_clip_sd_to_comfy(sd_l)
                _clip_debug("clip_l: converted OpenAI/OpenCLIP -> Comfy format")
            else:
                _clip_debug("clip_l: no OpenAI/OpenCLIP conversion applied")

            if _is_openai_clip_style(sd_g):
                sd_g = _convert_openai_clip_sd_to_comfy(sd_g)
                _clip_debug("clip_g: converted OpenAI/OpenCLIP -> Comfy format")
            else:
                _clip_debug("clip_g: no OpenAI/OpenCLIP conversion applied")

            # Ensure projection exists (some CLIP-L extracts omit it)
            sd_l, filled_l = _ensure_text_projection_weight(sd_l, label="clip_l")
            if filled_l:
                _clip_debug("clip_l: filled missing text_projection.weight with identity")
            sd_g, filled_g = _ensure_text_projection_weight(sd_g, label="clip_g")
            if filled_g:
                _clip_debug("clip_g: filled missing text_projection.weight with identity")

            _summarize_clip_sd(sd_l, "clip_l:after")
            _summarize_clip_sd(sd_g, "clip_g:after")

            model_options = {}
            if device == "cpu":
                model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

            # For 2 text encoders, ComfyUI will pick SDXLClipModel automatically in load_text_encoder_state_dicts.
            clip_type, clip_type_reason = _get_sdxl_clip_type()
            _clip_debug(
                f"load_text_encoder_state_dicts: clip_type={clip_type} ({clip_type_reason}), device={device}, "
                f"model_options_keys={list(model_options.keys())}"
            )
            clip = comfy.sd.load_text_encoder_state_dicts(
                [sd_l, sd_g],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=clip_type,
                model_options=model_options,
            )

            # Post-load sanity probes (do not dump full state_dict).
            try:
                _clip_debug(f"post_load: cond_stage_model_class={clip.cond_stage_model.__class__.__name__}")
                clip_sd = clip.cond_stage_model.state_dict()
                clip_keys = list(clip_sd.keys())
                ck = set(clip_keys)
                # Show how ComfyUI named the loaded modules (prefix discovery)
                _clip_debug(f"post_load: num_state_dict_keys={len(clip_keys)} head_keys={sorted(clip_keys)[:25]}")
                # Check for core markers anywhere (with or without clip_l/clip_g prefix)
                markers = [
                    "text_model.embeddings.token_embedding.weight",
                    "text_projection.weight",
                    "text_model.encoder.layers.30.mlp.fc1.weight",
                ]
                for m in markers:
                    found = any(k.endswith(m) for k in clip_keys)
                    _clip_debug(f"post_load: has_suffix('{m}')={found}")
                # Norm probe for first matching key per marker (if any)
                for m in markers:
                    mk = next((k for k in clip_keys if k.endswith(m)), None)
                    if mk is not None and mk in ck and torch.is_tensor(clip_sd[mk]):
                        t = clip_sd[mk]
                        _clip_debug(f"post_load: probe {mk} shape={tuple(t.shape)} dtype={t.dtype} norm={float(t.float().norm().cpu())}")
            except Exception as e:
                _clip_debug(f"post_load: probe failed: {e}")

            return (model, clip)
        finally:
            _CLIP_DEBUG_CONTEXT = _prev_debug

