import logging
import os
from pathlib import Path

import torch
import yaml
from packaging.version import InvalidVersion, Version

# vanilla and LTS compatibility snippet
try:
    from comfy_compatibility.vanilla import prepare_vanilla_environment

    prepare_vanilla_environment()

    from comfy.model_downloader import add_known_models
    from comfy.model_downloader_types import HuggingFile

    capability = torch.cuda.get_device_capability(0 if torch.cuda.is_available() else None)
    sm = f"{capability[0]}{capability[1]}"
    precision = "fp4" if sm == "120" else "int4"

    # add known models

    models_yaml_path = Path(__file__).parent / "test_data" / "models.yaml"
    with open(models_yaml_path, "r") as f:
        nunchaku_models_yaml = yaml.safe_load(f)

    NUNCHAKU_SVDQ_MODELS = []
    for model in nunchaku_models_yaml["models"]:
        filename = model["filename"]
        if not filename.startswith("svdq-"):
            continue
        if "{precision}" in filename:
            filename = filename.format(precision=precision)
        NUNCHAKU_SVDQ_MODELS.append(HuggingFile(repo_id=model["repo_id"], filename=filename))

    NUNCHAKU_SVDQ_TEXT_ENCODER_MODELS = [
        HuggingFile(repo_id="nunchaku-tech/nunchaku-t5", filename="awq-int4-flux.1-t5xxl.safetensors"),
    ]

    add_known_models("diffusion_models", *NUNCHAKU_SVDQ_MODELS)
    add_known_models("text_encoders", *NUNCHAKU_SVDQ_TEXT_ENCODER_MODELS)
except (ImportError, ModuleNotFoundError):
    pass

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info("=" * 40 + " ComfyUI-nunchaku Initialization " + "=" * 40)

from .utils import get_package_version, get_plugin_version

# Check if _patch_model method exists in NunchakuZImageTransformer2DModel
try:
    from nunchaku.models.transformers.transformer_zimage import NunchakuZImageTransformer2DModel
    if hasattr(NunchakuZImageTransformer2DModel, '_patch_model'):
        logger.info("NunchakuZImageTransformer2DModel._patch_model method found - patch may not be required")
    else:
        logger.warning("NunchakuZImageTransformer2DModel._patch_model method NOT found - patch is required")

    # ---------------------------------------------------------------------
    # ComfyUI ModelPatcher integration:
    # - ComfyUI passes ControlNet patches via transformer_options["patches"]["double_block"]
    # - Nunchaku Z-Image forward does not natively call those patches.
    # This monkey-patch wraps each transformer block and calls double_block patches
    # after each block, matching ComfyUI's NextDiT/QwenImage patching behavior.
    # ---------------------------------------------------------------------
    if not getattr(NunchakuZImageTransformer2DModel, "_comfyui_mp_patched", False):
        import inspect
        from comfy.ldm.flux.layers import EmbedND

        _orig_forward = NunchakuZImageTransformer2DModel.forward

        def _apply_double_block_patches(
            parent: "NunchakuZImageTransformer2DModel",
            block_index: int,
            unified_in,
            unified_out,
            adaln_input,
        ):
            """
            Apply ComfyUI ModelPatcher's double_block patches after a transformer block.
            IMPORTANT: do NOT change module hierarchy (LoRA matching depends on `layers.N.*` paths).
            """
            transformer_options = getattr(parent, "_comfyui_transformer_options", None)
            if not isinstance(transformer_options, dict):
                return unified_out

            patches = transformer_options.get("patches", {})
            if not isinstance(patches, dict):
                return unified_out

            double_block_patches = patches.get("double_block", [])
            if not double_block_patches:
                return unified_out

            # ComfyUI conventions
            transformer_options["block_index"] = block_index
            transformer_options.setdefault("block_type", "double")

            original_x_list = getattr(parent, "_comfyui_original_x_list", None)
            cap_len = getattr(parent, "_comfyui_cap_len", None)
            if not isinstance(cap_len, int) or cap_len < 0:
                cap_feats_list = getattr(parent, "_comfyui_cap_feats", None)
                if isinstance(cap_feats_list, list) and len(cap_feats_list) > 0 and hasattr(cap_feats_list[0], "shape"):
                    cap_len = int(cap_feats_list[0].shape[0])
                else:
                    cap_len = 0

            img_len = getattr(parent, "_comfyui_img_len", None)
            if not isinstance(img_len, int) or img_len < 0:
                img_len = 0

            unified = unified_out
            for p in double_block_patches:
                # Z-Image-Turbo uses List[torch.Tensor] for x, but ZImageControlPatch expects (B,C,H,W) tensor
                patch_x = original_x_list
                if isinstance(original_x_list, list) and len(original_x_list) > 0:
                    patch_x = torch.stack(original_x_list, dim=0)  # (B, C, F, H, W)
                    if patch_x.shape[2] == 1:
                        patch_x = patch_x.squeeze(2)  # (B, C, H, W)

                patch_in = {"x": patch_x, "block_index": block_index, "transformer_options": transformer_options}

                if unified is not None and hasattr(unified, "shape"):
                    # diffusers Z-Image order: [img_tokens, txt_tokens]
                    patch_in["img"] = unified[:, :img_len]
                    patch_in["txt"] = unified[:, img_len:img_len + cap_len]
                else:
                    patch_in["img"] = unified
                    patch_in["txt"] = None

                if unified_in is not None and hasattr(unified_in, "shape"):
                    patch_in["img_input"] = unified_in[:, :img_len]
                else:
                    patch_in["img_input"] = None

                # Build ComfyUI-compatible RoPE freqs for image tokens (pe) using EmbedND (Lumina style)
                pe_cached = getattr(parent, "_comfyui_pe_img", None)
                pe_key = getattr(parent, "_comfyui_pe_key", None)
                rope_options = transformer_options.get("rope_options", None) if isinstance(transformer_options, dict) else None
                h_scale = 1.0
                w_scale = 1.0
                h_start = 0.0
                w_start = 0.0
                if isinstance(rope_options, dict):
                    try:
                        h_scale = float(rope_options.get("scale_y", 1.0))
                        w_scale = float(rope_options.get("scale_x", 1.0))
                        h_start = float(rope_options.get("shift_y", 0.0))
                        w_start = float(rope_options.get("shift_x", 0.0))
                    except Exception:
                        h_scale = 1.0
                        w_scale = 1.0
                        h_start = 0.0
                        w_start = 0.0

                want_key = (img_len, cap_len, getattr(patch_x, "shape", None), h_scale, w_scale, h_start, w_start, unified.device, unified.dtype)
                if pe_cached is None or pe_key != want_key:
                    # Default Z-Image settings from Nunchaku docs: rope_theta=256, axes_dims=[32,48,48] (sum=128)
                    rope_theta = 256
                    axes_dims = (32, 48, 48)
                    head_dim = sum(axes_dims)
                    rope_embedder = getattr(parent, "_comfyui_rope_embedder", None)
                    if rope_embedder is None or getattr(rope_embedder, "theta", None) != rope_theta:
                        rope_embedder = EmbedND(dim=head_dim, theta=rope_theta, axes_dim=list(axes_dims))
                        parent._comfyui_rope_embedder = rope_embedder

                    b = int(patch_x.shape[0]) if hasattr(patch_x, "shape") else 1
                    ids = torch.zeros((b, img_len, 3), dtype=torch.float32, device=unified.device)

                    x_list = getattr(parent, "_comfyui_original_x_list", None)
                    h_tokens = 0
                    w_tokens = 0
                    try:
                        if isinstance(x_list, list) and len(x_list) > 0 and hasattr(x_list[0], "shape"):
                            _, f, h, w = x_list[0].shape
                            patch_size = int(getattr(parent, "_comfyui_patch_size", 2))
                            h_tokens = h // patch_size
                            w_tokens = w // patch_size
                    except Exception:
                        h_tokens = 0
                        w_tokens = 0

                    if h_tokens > 0 and w_tokens > 0:
                        n = min(img_len, h_tokens * w_tokens)
                        cap_offset = float(cap_len + 1)
                        ids[:, :n, 0] = cap_offset
                        ys = (torch.arange(h_tokens, dtype=torch.float32, device=unified.device) * h_scale + h_start).view(-1, 1).repeat(1, w_tokens).flatten()
                        xs = (torch.arange(w_tokens, dtype=torch.float32, device=unified.device) * w_scale + w_start).view(1, -1).repeat(h_tokens, 1).flatten()
                        ids[:, :n, 1] = ys[:n]
                        ids[:, :n, 2] = xs[:n]

                    pe_img = rope_embedder(ids).movedim(1, 2)  # (B, seq, 1, head_dim/2, 2, 2)
                    pe_img = pe_img.to(dtype=unified.dtype).contiguous()
                    parent._comfyui_pe_img = pe_img
                    parent._comfyui_pe_key = want_key
                    pe_cached = pe_img

                patch_in["pe"] = pe_cached
                patch_in["vec"] = adaln_input
                patch_in["block_type"] = transformer_options.get("block_type", "double")

                patch_out = p(patch_in)
                if isinstance(patch_out, dict) and unified is not None and hasattr(unified, "shape"):
                    if "img" in patch_out:
                        unified[:, :img_len] = patch_out["img"]
                    if "txt" in patch_out:
                        unified[:, img_len:img_len + cap_len] = patch_out["txt"]

            return unified

        def _ensure_layers_patched_in_place(parent: "NunchakuZImageTransformer2DModel"):
            if getattr(parent, "_comfyui_layers_patched_in_place", False):
                return
            try:
                layers = getattr(parent, "layers", None)
            except Exception:
                layers = None
            if layers is None:
                return
            try:
                for idx, layer in enumerate(layers):
                    if getattr(layer, "_comfyui_double_block_patched", False):
                        continue
                    orig_layer_forward = layer.forward

                    def _layer_forward_patched(*args, __orig=orig_layer_forward, __idx=idx, __parent=parent, **kwargs):
                        unified_in = args[0] if len(args) > 0 else None
                        adaln_input = args[3] if len(args) > 3 else None
                        out = __orig(*args, **kwargs)
                        return _apply_double_block_patches(__parent, __idx, unified_in, out, adaln_input)

                    layer.forward = _layer_forward_patched
                    layer._comfyui_double_block_patched = True
                parent._comfyui_layers_patched_in_place = True
                logger.info("[ZImageTurbo] Patched transformer block forwards (in-place) for double_block ControlNet patches")
            except Exception as e:
                logger.exception(f"[ZImageTurbo] Failed to patch layer forwards in-place: {e}")

        def _patched_forward(self, x, t, cap_feats=None, *args, control=None, transformer_options=None, **kwargs):
            # Store for block wrappers
            if isinstance(transformer_options, dict):
                self._comfyui_transformer_options = transformer_options
            else:
                self._comfyui_transformer_options = {}
            self._comfyui_original_x_list = x
            self._comfyui_cap_feats = cap_feats
            try:
                if isinstance(cap_feats, list) and len(cap_feats) > 0 and hasattr(cap_feats[0], "shape"):
                    self._comfyui_cap_len = int(cap_feats[0].shape[0])
            except Exception:
                self._comfyui_cap_len = 0

            # Z-Image (diffusers) unified token order is: img tokens first, then txt tokens.
            # Compute img token length from the input x list shape and patch sizes (default: 2 / 1).
            try:
                patch_size = int(kwargs.get("patch_size", 2))
            except Exception:
                patch_size = 2
            try:
                f_patch_size = int(kwargs.get("f_patch_size", 1))
            except Exception:
                f_patch_size = 1

            try:
                if isinstance(x, list) and len(x) > 0 and hasattr(x[0], "shape"):
                    # x element shape: (C, F, H, W)
                    _, f, h, w = x[0].shape
                    img_tokens = (h // patch_size) * (w // patch_size) * (f // f_patch_size)
                    # SEQ_MULTI_OF in diffusers ZImage is 32
                    img_tokens = int(((img_tokens + 31) // 32) * 32)
                    self._comfyui_img_len = img_tokens
                else:
                    self._comfyui_img_len = 0
            except Exception:
                self._comfyui_img_len = 0

            _ensure_layers_patched_in_place(self)

            # Log once per forward if patches exist (avoid spam)
            try:
                patches = self._comfyui_transformer_options.get("patches", {})
                dbl = patches.get("double_block", []) if isinstance(patches, dict) else []
                if dbl and not getattr(self, "_comfyui_logged_double_block", False):
                    logger.info(f"[ZImageTurbo] double_block patches detected: {len(dbl)} (will apply after each block)")
                    self._comfyui_logged_double_block = True
            except Exception:
                pass

            # Call original forward with only supported kwargs
            try:
                sig = inspect.signature(_orig_forward)
                allowed = set(sig.parameters.keys())
                call_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
            except Exception:
                call_kwargs = kwargs
            return _orig_forward(self, x, t, cap_feats=cap_feats, **call_kwargs)

        NunchakuZImageTransformer2DModel.forward = _patched_forward
        NunchakuZImageTransformer2DModel._comfyui_mp_patched = True
        logger.info("[ZImageTurbo] Installed ModelPatcher-compatible forward() patch")
except ImportError as e:
    logger.warning(f"Could not import NunchakuZImageTransformer2DModel to check _patch_model: {e}")

nunchaku_full_version = get_package_version("nunchaku").split("+")[0].strip()

logger.info(f"Nunchaku version: {nunchaku_full_version}")
logger.info(f"ComfyUI-nunchaku version: {get_plugin_version()}")


min_nunchaku_version = "1.0.0"
nunchaku_version = nunchaku_full_version.split("+")[0].strip()
nunchaku_major_minor_patch_version = ".".join(nunchaku_version.split(".")[:3])

try:
    if Version(nunchaku_major_minor_patch_version) < Version(min_nunchaku_version):
        logger.warning(
            f"ComfyUI-nunchaku {get_plugin_version()} requires nunchaku >= v{min_nunchaku_version}, "
            f"but found nunchaku {nunchaku_full_version}. Please update nunchaku."
        )
except InvalidVersion:
    logger.warning(
        f"Could not parse nunchaku version: {nunchaku_full_version}. "
        f"Please ensure you have at least v{min_nunchaku_version}."
    )

NODE_CLASS_MAPPINGS = {}

try:
    from .nodes.models.zimage import NunchakuZImageDiTLoader

    NODE_CLASS_MAPPINGS["NunchakuUssoewwinZImageDiTLoader"] = NunchakuZImageDiTLoader
except (ImportError, ModuleNotFoundError) as e:
    logger.exception(f"Node `NunchakuZImageDiTLoader` import failed: {e}")
    # Try alternative import method using absolute path
    try:
        import importlib.util
        from pathlib import Path

        # Get the directory where __init__.py is located
        current_dir = Path(__file__).parent.resolve()
        zimage_path = current_dir / "nodes" / "models" / "zimage.py"

        if not zimage_path.exists():
            raise FileNotFoundError(f"zimage.py not found at {zimage_path}")

        spec = importlib.util.spec_from_file_location(
            "nodes.models.zimage",
            str(zimage_path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create spec for {zimage_path}")

        zimage_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(zimage_module)
        NunchakuZImageDiTLoader = zimage_module.NunchakuZImageDiTLoader
        NODE_CLASS_MAPPINGS["NunchakuUssoewwinZImageDiTLoader"] = NunchakuZImageDiTLoader
        logger.info(f"Successfully loaded NunchakuZImageDiTLoader using alternative method from {zimage_path}")
    except Exception as e2:
        logger.exception(f"Alternative import method also failed: {e2}")

try:
    from .nodes.models.sdxl import NunchakuSDXLDiTLoader, NunchakuSDXLDiTLoaderDualCLIP

    NODE_CLASS_MAPPINGS["NunchakuUssoewwinSDXLDiTLoader"] = NunchakuSDXLDiTLoader
    NODE_CLASS_MAPPINGS["NunchakuUssoewwinSDXLDiTLoaderDualCLIP"] = NunchakuSDXLDiTLoaderDualCLIP
except (ImportError, ModuleNotFoundError) as e:
    logger.exception(f"Node `NunchakuSDXLDiTLoader` import failed: {e}")
    # Try alternative import method using absolute path
    try:
        import importlib.util
        from pathlib import Path

        # Get the directory where __init__.py is located
        current_dir = Path(__file__).parent.resolve()
        sdxl_path = current_dir / "nodes" / "models" / "sdxl.py"

        if not sdxl_path.exists():
            raise FileNotFoundError(f"sdxl.py not found at {sdxl_path}")

        spec = importlib.util.spec_from_file_location(
            "nodes.models.sdxl",
            str(sdxl_path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create spec for {sdxl_path}")

        sdxl_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sdxl_module)
        NunchakuSDXLDiTLoader = sdxl_module.NunchakuSDXLDiTLoader
        NODE_CLASS_MAPPINGS["NunchakuUssoewwinSDXLDiTLoader"] = NunchakuSDXLDiTLoader
        if hasattr(sdxl_module, "NunchakuSDXLDiTLoaderDualCLIP"):
            NODE_CLASS_MAPPINGS["NunchakuUssoewwinSDXLDiTLoaderDualCLIP"] = sdxl_module.NunchakuSDXLDiTLoaderDualCLIP
        logger.info(f"Successfully loaded NunchakuSDXLDiTLoader using alternative method from {sdxl_path}")
    except Exception as e2:
        logger.exception(f"Alternative import method also failed: {e2}")

try:
    from .nodes.lora.sdxl_v3 import GENERATED_NODES as SDXL_LORA_V3_NODES
    NODE_CLASS_MAPPINGS.update(SDXL_LORA_V3_NODES)
except (ImportError, ModuleNotFoundError) as e:
    logger.exception(f"Node `SDXL LoRA V3` import failed: {e}")

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
WEB_DIRECTORY = "js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
logger.info("=" * (80 + len(" ComfyUI-nunchaku Initialization ")))
