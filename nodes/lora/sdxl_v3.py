"""
This module provides the :class:`NunchakuUssoewwinSDXLLoraStackV3` node
for applying multiple LoRAs to SDXL models within ComfyUI.
"""

import logging
import os
import re
import weakref

import folder_paths
import comfy.lora
import comfy.lora_convert
import comfy.sd
import comfy.utils
import torch

logger = logging.getLogger(__name__)

# Keep weak refs to root model objects without creating nn.Module cycles.
# key: id(root_model) -> root_model (weak)
_NUNCHAKU_ROOT_MODEL_WEAK: "weakref.WeakValueDictionary[int, object]" = weakref.WeakValueDictionary()

# -----------------------------------------------------------------------------
# Safety / "exception" handling for runtime LoRA on quantized layers
#
# Just like QwenImage's AWQ modulation layers (img_mod.1/txt_mod.1) are extremely
# sensitive and are skipped by default in QwenImageLoraLoader v2.2.4:
#   https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/2.2.4
#
# Nunchaku SDXL SVDQ runtime forward-add can also have "should be mappable but
# destabilizes output" exception targets depending on model/LoRA.
#
# We DO NOT skip anything by default (to avoid hiding behavior), but we provide an
# opt-in skip mechanism controlled by env var:
#   NUNCHAKU_SDXL_SVDQ_RUNTIME_SKIP
#
# Format: comma-separated substrings. If any substring matches the runtime module
# dot-path (e.g. "down_blocks.2.attentions.0.transformer_blocks.3.attn2.to_q"),
# runtime application for that base is skipped, and a [SKIP] log is printed.
# -----------------------------------------------------------------------------
_SVDQ_RUNTIME_SKIP = str(os.getenv("NUNCHAKU_SDXL_SVDQ_RUNTIME_SKIP", "")).strip()


def _svdq_runtime_skip_tokens() -> list[str]:
    if not _SVDQ_RUNTIME_SKIP:
        return []
    toks = []
    for t in _SVDQ_RUNTIME_SKIP.split(","):
        s = str(t).strip()
        if s:
            toks.append(s)
    return toks


def _svdq_runtime_should_skip(mod_path: str) -> tuple[bool, str]:
    """
    Returns (skip, reason). Reason is a short string for logs.
    """
    tokens = _svdq_runtime_skip_tokens()
    if not tokens:
        return False, ""
    for t in tokens:
        if t in mod_path:
            return True, f"env:NUNCHAKU_SDXL_SVDQ_RUNTIME_SKIP token='{t}'"
    return False, ""


def _dbg_print(enabled: bool, msg: str) -> None:
    # Prefer print for "must show" debug in ComfyUI console.
    if enabled:
        print(msg)


class _NunchakuSVDQRuntime:
    """
    Per-ModelPatcher runtime storage for quantized (SVDQ) LoRA matrices.

    IMPORTANT:
    ComfyUI ModelPatcher.clone() shares the same underlying model object, but patches are supposed to be
    per-patcher. Storing runtime LoRAs in this attachment (which supports on_model_patcher_clone)
    prevents accidental sharing/leakage across patchers.
    """

    def __init__(self, debug: bool = False):
        self.debug = bool(debug)
        # module_id -> List[(down_cpu, up_cpu, scale)]
        self.loras: dict[int, list[tuple[torch.Tensor, torch.Tensor, float]]] = {}
        # Prepared per-module fused matrices cached per-device/dtype to avoid per-forward .to() and Python loops.
        # module_id -> {"device": torch.device, "dtype": torch.dtype, "down_t": Tensor, "up_t": Tensor}
        self.prepared: dict[int, dict] = {}
        # module_id set that indicates prepared cache must be rebuilt (LoRA changed)
        self.dirty: set[int] = set()
        # module_id set for one-time "forward ACTIVE" confirmation logs
        self.forward_seen: set[int] = set()
        # one-time global confirmation for this patcher instance
        self.forward_active_logged: bool = False

    def on_model_patcher_clone(self):
        n = _NunchakuSVDQRuntime(self.debug)
        n.loras = {k: v[:] for k, v in self.loras.items()}
        # Do NOT copy prepared GPU tensors across clones (device/offload may differ). Rebuild lazily.
        n.prepared = {}
        n.dirty = set(n.loras.keys())
        n.forward_seen = set(self.forward_seen)
        n.forward_active_logged = bool(self.forward_active_logged)
        return n


def _runtime_lora_compute_dtype(x: torch.Tensor, base_out: torch.Tensor) -> torch.dtype:
    """
    Decide compute dtype for runtime SVDQ LoRA matmuls.
    Default is to match `x.dtype` (fast), instead of forcing float32 (slow).
    Override via env:
      NUNCHAKU_SDXL_SVDQ_RUNTIME_DTYPE = "x" | "out" | "fp16" | "bf16" | "fp32"
    """
    mode = str(os.getenv("NUNCHAKU_SDXL_SVDQ_RUNTIME_DTYPE", "x")).strip().lower()
    if mode in ("x", "input"):
        return x.dtype
    if mode in ("out", "output"):
        return base_out.dtype
    if mode in ("fp16", "float16"):
        return torch.float16
    if mode in ("bf16", "bfloat16"):
        return torch.bfloat16
    if mode in ("fp32", "float32"):
        return torch.float32
    # Fallback: prefer x dtype when it's a supported float type; otherwise follow base_out.
    if x.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return x.dtype
    return base_out.dtype


def _runtime_lora_should_check_finite(runtime: _NunchakuSVDQRuntime) -> bool:
    """
    NaN/Inf check is extremely expensive; keep it OFF by default for performance.
    Enable via debug or env:
      NUNCHAKU_SDXL_SVDQ_RUNTIME_CHECK_FINITE=1
    """
    if getattr(runtime, "debug", False):
        return True
    v = str(os.getenv("NUNCHAKU_SDXL_SVDQ_RUNTIME_CHECK_FINITE", "0")).strip().lower()
    return v in ("1", "true", "yes", "on")


def _try_import_svdq_linear():
    try:
        from nunchaku.models.linear import SVDQW4A4Linear  # type: ignore

        return SVDQW4A4Linear
    except Exception:
        return None


def _strip_runtime_svdq_keys(
    lora_converted: dict, model, debug: bool
) -> tuple[dict, dict]:
    """
    Prevent noisy 'lora key not loaded' spam from comfy.lora.load_lora() by removing
    UNet LoRA entries that we will apply via SVDQ runtime (forward-add) instead.

    This does NOT change tensors; it only filters which keys are fed into load_lora().
    """
    if not isinstance(lora_converted, dict):
        return lora_converted, {"stripped": False, "reason": "lora_converted is not a dict"}

    svdq_cls = _try_import_svdq_linear()
    if svdq_cls is None:
        return lora_converted, {"stripped": False, "reason": "no_svdq_import"}

    try:
        # Expect a ModelPatcher-like object (same as _apply_runtime_lora_to_svdq_modules expects).
        if model is None or not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
            return lora_converted, {"stripped": False, "reason": "no_diffusion_model"}
        diffusion_model = model.model.diffusion_model
    except Exception:
        return lora_converted, {"stripped": False, "reason": "no_diffusion_model"}

    # Identify bases that resolve to actual SVDQ modules.
    runtime_bases: set[str] = set()
    for k in list(lora_converted.keys()):
        if not isinstance(k, str):
            continue
        base = _lora_base_key_from_any(k)
        if not base.startswith("lora_unet_"):
            continue
        mod_path = _svdq_lora_base_to_module_dot_path(base)
        if mod_path is None:
            continue
        try:
            mod = _resolve_dot_path(diffusion_model, mod_path)
        except Exception:
            continue
        if isinstance(mod, svdq_cls):
            runtime_bases.add(base)

    if not runtime_bases:
        return lora_converted, {"stripped": False, "reason": "no_runtime_bases"}

    # Filter out all entries whose base belongs to runtime_bases.
    out = {}
    removed = 0
    for k, v in lora_converted.items():
        if isinstance(k, str):
            b = _lora_base_key_from_any(k)
            if b in runtime_bases:
                removed += 1
                continue
        out[k] = v

    stats = {"stripped": True, "runtime_bases": len(runtime_bases), "removed_keys": removed, "runtime_bases_set": runtime_bases}
    _dbg_print(True, f"[NUNCHAKU_SDXL_LORA_DEBUG] strip runtime SVDQ keys for load_lora: {stats}")
    return out, stats


def _coverage_report_for_unet(
    lora_converted: dict,
    key_map: dict,
    runtime_bases: set[str],
    runtime_skipped_bases: set[str],
    debug: bool,
) -> dict:
    """
    "完全にマッピング" のための可観測性:
    UNet系ベースキーが
      - standard (key_mapに存在) or
      - runtime (SVDQ runtimeで処理)
    のどちらかに全て割り当てられていることを確認する。
    """
    # Collect UNet-related keys and bases.
    bases: set[str] = set()
    unet_keys: list[str] = []
    for k in lora_converted.keys():
        if not isinstance(k, str):
            continue
        b = _lora_base_key_from_any(k)
        if b.startswith("lora_unet_"):
            bases.add(b)
            unet_keys.append(k)

    standard_bases = {b for b in bases if b in key_map}
    runtime_hit = {b for b in bases if b in runtime_bases}
    unmapped = sorted(list(bases - standard_bases - runtime_hit))

    # Key-level accounting:
    # comfy.lora.load_lora() logs "lora key not loaded: <key>" for any key not consumed.
    # For Nunchaku SDXL, most quantized SVDQ targets cannot be standard-patched => those keys
    # are expected to remain "not loaded" and are applied at runtime instead.
    standard_keys = 0
    runtime_keys = 0
    runtime_keys_unsupported = 0
    runtime_keys_skipped = 0
    unmapped_keys = 0
    unmapped_key_samples: list[str] = []
    runtime_unsupported_key_samples: list[str] = []
    runtime_unsupported_bases: set[str] = set()
    runtime_skipped_key_samples: list[str] = []

    # Runtime SVDQ forward-add currently supports only vanilla LoRA matrices.
    # Derived/variant keys (DoRA, norms, diff, etc.) must be explicitly reported so
    # "unmapped_bases=0" doesn't get misread as "fully supported".
    runtime_supported_suffixes = (
        ".lora_up.weight",
        ".lora_down.weight",
        ".alpha",
    )
    for k in unet_keys:
        b = _lora_base_key_from_any(k)
        if b in key_map:
            standard_keys += 1
        elif b in runtime_bases:
            runtime_keys += 1
            if b in runtime_skipped_bases:
                runtime_keys_skipped += 1
                if len(runtime_skipped_key_samples) < 40:
                    runtime_skipped_key_samples.append(k)
                continue
            try:
                suffix = k[len(b) :] if isinstance(k, str) and k.startswith(b) else ""
            except Exception:
                suffix = ""
            if suffix and suffix not in runtime_supported_suffixes:
                runtime_keys_unsupported += 1
                runtime_unsupported_bases.add(b)
                if len(runtime_unsupported_key_samples) < 40:
                    runtime_unsupported_key_samples.append(k)
        else:
            unmapped_keys += 1
            if len(unmapped_key_samples) < 40:
                unmapped_key_samples.append(k)

    stats = {
        # base-level coverage (what "完全にマッピング" refers to)
        "unet_bases_total": len(bases),
        "standard_bases": len(standard_bases),
        "runtime_bases": len(runtime_hit),
        "unmapped_bases": len(unmapped),
        "unmapped_base_samples": unmapped[:40],
        # key-level breakdown (explains "lora key not loaded" spam without suppressing it)
        "unet_keys_total": len(unet_keys),
        "standard_keys": standard_keys,
        "runtime_keys_expected_not_loaded": runtime_keys,
        "runtime_keys_skipped": runtime_keys_skipped,
        "runtime_skipped_bases": len(runtime_skipped_bases),
        "runtime_skipped_key_samples": sorted(runtime_skipped_key_samples)[:40],
        "runtime_keys_unsupported": runtime_keys_unsupported,
        "runtime_unsupported_bases": len(runtime_unsupported_bases),
        "runtime_unsupported_key_samples": sorted(runtime_unsupported_key_samples)[:40],
        "unmapped_keys": unmapped_keys,
        "unmapped_key_samples": sorted(unmapped_key_samples)[:40],
    }
    # Always print coverage summary (logs are meaningful for auditing).
    print(f"[NUNCHAKU_SDXL_LORA_DEBUG] UNet mapping coverage: {stats}")
    # Also print a grep-friendly one-line summary to allow strict matching like "unmapped_bases=0".
    try:
        print(
            "[NUNCHAKU_SDXL_LORA_DEBUG] UNet mapping coverage summary: "
            f"unet_bases_total={stats.get('unet_bases_total')} "
            f"standard_bases={stats.get('standard_bases')} "
            f"runtime_bases={stats.get('runtime_bases')} "
            f"unmapped_bases={stats.get('unmapped_bases')}"
        )
    except Exception:
        pass
    # In debug mode, treat unmapped as a hard error (this is what "complete mapping" means).
    if debug and stats["unmapped_bases"] > 0:
        raise ValueError(f"UNet LoRA unmapped bases detected: {stats['unmapped_base_samples']}")
    return stats


def _resolve_attr_or_index(obj, seg: str):
    # seg can be "foo" (attr) or "0" (index for list/ModuleList)
    if seg.isdigit():
        return obj[int(seg)]
    return getattr(obj, seg)


def _resolve_dot_path(root, dot_path: str):
    cur = root
    for seg in dot_path.split("."):
        if seg == "":
            continue
        cur = _resolve_attr_or_index(cur, seg)
    return cur


def _lora_base_key_from_any(k: str) -> str:
    """
    Normalize "base key" extraction across all ComfyUI weight adapters.

    comfy.lora.load_lora() decides whether a key is loadable by comparing the *base* (prefix)
    against model key_map. For "完全にマッピング" auditing, we must collapse adapter-specific
    suffixes into the same base consistently.
    """
    if not isinstance(k, str) or not k:
        return ""

    # IMPORTANT: keep this list aligned with ComfyUI's built-in adapters in comfy/weight_adapter/*
    suffixes = (
        # Common scale/variants (LoRAAdapter + weight_decompose helpers)
        ".alpha",
        ".lora_alpha",
        ".dora_scale",
        ".w_norm",
        ".b_norm",
        ".diff",
        ".diff_b",
        ".set_weight",
        ".reshape_weight",

        # Standard LoRA tensor names
        ".lora_up.weight",
        ".lora_down.weight",
        ".lora_mid.weight",
        "_lora.up.weight",
        "_lora.down.weight",

        # PEFT / diffusers naming variants
        ".lora_A.weight",
        ".lora_B.weight",
        ".lora_A.default.weight",
        ".lora_B.default.weight",

        # LoHaAdapter (loha.py)
        ".hada_w1_a",
        ".hada_w1_b",
        ".hada_w2_a",
        ".hada_w2_b",
        ".hada_t1",
        ".hada_t2",

        # LoKrAdapter (lokr.py)
        ".lokr_w1",
        ".lokr_w2",
        ".lokr_w1_a",
        ".lokr_w1_b",
        ".lokr_w2_a",
        ".lokr_w2_b",
        ".lokr_t2",

        # GLoRAAdapter (glora.py)
        ".a1.weight",
        ".a2.weight",
        ".b1.weight",
        ".b2.weight",

        # OFTAdapter / BOFTAdapter (oft.py / boft.py)
        ".oft_blocks",
        ".rescale",
    )

    for s in suffixes:
        if k.endswith(s):
            return k[: -len(s)]
    return k


def _normalize_peft_lora_ab_to_comfy(lora_sd: dict, debug: bool) -> tuple[dict, dict]:
    """
    Normalize PEFT-style LoRA keys into ComfyUI-friendly keys so comfy.lora_convert.convert_lora()
    and comfy.lora.load_lora() can work.

    Supported transforms (no tensor math; only key rename):
    - *.lora_A(.default).weight -> *.lora_down.weight
    - *.lora_B(.default).weight -> *.lora_up.weight
    - *.lora_alpha -> *.alpha
    """
    if not isinstance(lora_sd, dict):
        return lora_sd, {"normalized": False, "reason": "lora_sd is not a dict"}

    out = dict(lora_sd)
    changed = 0
    samples: list[tuple[str, str]] = []

    def _rename(old_k: str, new_k: str):
        nonlocal changed
        if old_k == new_k:
            return
        if old_k not in out:
            return
        if new_k in out:
            # Keep existing destination key if present; do not overwrite.
            return
        out[new_k] = out.pop(old_k)
        changed += 1
        if len(samples) < 20:
            samples.append((old_k, new_k))

    for k in list(out.keys()):
        if not isinstance(k, str):
            continue
        if k.endswith(".lora_A.weight"):
            _rename(k, k[: -len(".lora_A.weight")] + ".lora_down.weight")
        elif k.endswith(".lora_B.weight"):
            _rename(k, k[: -len(".lora_B.weight")] + ".lora_up.weight")
        elif k.endswith(".lora_A.default.weight"):
            _rename(k, k[: -len(".lora_A.default.weight")] + ".lora_down.weight")
        elif k.endswith(".lora_B.default.weight"):
            _rename(k, k[: -len(".lora_B.default.weight")] + ".lora_up.weight")
        elif k.endswith(".lora_alpha"):
            _rename(k, k[: -len(".lora_alpha")] + ".alpha")

    stats = {"normalized": changed > 0, "changed_keys": changed, "sample_renames": samples}
    if debug:
        _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] PEFT A/B normalize: {stats}")
    return out, stats


def _normalize_unet_to_out_indexing(lora_converted: dict, debug: bool) -> tuple[dict, dict]:
    """
    Normalize common UNet LoRA base-key variants for attention "to_out".

    Some trainers emit:
      - ..._attn1_to_out  (no "_0")
      - ..._attn2_to_out  (no "_0")
    while ComfyUI/diffusers state_dict keys and our key_map convention typically use:
      - ..._attn1_to_out_0
      - ..._attn2_to_out_0

    This is a *key rename only* on the already-converted LoRA dict (output of comfy.lora_convert.convert_lora()).
    It does not modify tensor values.
    """
    if not isinstance(lora_converted, dict):
        return lora_converted, {"normalized": False, "reason": "lora_converted is not a dict"}

    out: dict = {}
    changed = 0
    samples: list[tuple[str, str]] = []

    for k, v in lora_converted.items():
        if not isinstance(k, str):
            out[k] = v
            continue

        base = _lora_base_key_from_any(k)
        new_base = base
        if base.endswith("_attn1_to_out"):
            new_base = base + "_0"
        elif base.endswith("_attn2_to_out"):
            new_base = base + "_0"

        if new_base != base:
            new_k = new_base + k[len(base) :]
            # If destination exists, keep the original (avoid overwriting).
            if new_k not in out and new_k not in lora_converted:
                out[new_k] = v
                changed += 1
                if len(samples) < 30:
                    samples.append((k, new_k))
                continue

        out[k] = v

    stats = {"normalized": changed > 0, "changed_keys": changed, "sample_renames": samples}
    if debug and stats["normalized"]:
        _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] normalize to_out indexing: {stats}")
    return out, stats


def _svdq_lora_base_to_module_dot_path(base: str) -> str | None:
    """
    Map SDXL UNet LoRA base keys (after A1111->diffusers conversion / qkv fusion)
    to an actual module dot-path under diffusion_model.

    Examples:
    - lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_qkv
      -> down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_qkv
    - lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_q
      -> down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q
    - lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_ff_net_0_proj
      -> down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj
    """
    if not isinstance(base, str):
        return None
    if not base.startswith("lora_unet_"):
        return None

    # strip prefix
    rest = base[len("lora_unet_") :]

    # Fast path: only handle transformer-block internals we know are quantized in Nunchaku SDXL
    tail_map = {
        "attn1_to_qkv": "attn1.to_qkv",
        "attn1_to_out_0": "attn1.to_out.0",
        "attn1_to_out": "attn1.to_out.0",
        "attn2_to_q": "attn2.to_q",
        "attn2_to_out_0": "attn2.to_out.0",
        "attn2_to_out": "attn2.to_out.0",
        "ff_net_0_proj": "ff.net.0.proj",
        "ff_net_2": "ff.net.2",
    }

    for tail_key, tail_path in tail_map.items():
        suffix = f"_{tail_key}"
        if not rest.endswith(suffix):
            continue

        prefix = rest[: -len(suffix)]

        # The runtime target is ALWAYS under diffusion_model and uses diffusers structure:
        # - down_blocks.{i}.attentions.{j}.transformer_blocks.{k}. ...
        # - up_blocks.{i}.attentions.{j}.transformer_blocks.{k}. ...
        # - mid_block.attentions.{j}.transformer_blocks.{k}. ...
        #
        # IMPORTANT:
        # Do NOT do naive "_" -> "." conversion (it breaks "down_blocks" into "down.blocks").
        m = re.match(
            r"^(down_blocks|up_blocks)_(\d+)_attentions_(\d+)_transformer_blocks_(\d+)$",
            prefix,
        )
        if m is not None:
            block_type, bi, ai, ti = m.group(1), m.group(2), m.group(3), m.group(4)
            return f"{block_type}.{bi}.attentions.{ai}.transformer_blocks.{ti}.{tail_path}"

        m = re.match(r"^mid_block_attentions_(\d+)_transformer_blocks_(\d+)$", prefix)
        if m is not None:
            ai, ti = m.group(1), m.group(2)
            return f"mid_block.attentions.{ai}.transformer_blocks.{ti}.{tail_path}"

    return None


def _ensure_svdq_forward_patched(mod, debug: bool) -> bool:
    """
    Patch SVDQW4A4Linear.forward to add runtime LoRA deltas, without touching the quantized weights.
    The LoRA deltas are stored on the module as _nunchaku_runtime_loras = [(down, up, scale), ...]
    where down: (rank, in_features), up: (out_features, rank).
    """
    if hasattr(mod, "_nunchaku_runtime_lora_patched") and getattr(mod, "_nunchaku_runtime_lora_patched"):
        return False

    orig_forward = getattr(mod, "forward", None)
    if orig_forward is None:
        return False

    # Safety: if an older buggy version stored a nn.Module on this attribute,
    # torch would register it in _modules and create a cycle (-> RecursionError on .to()).
    try:
        if hasattr(mod, "_modules") and isinstance(getattr(mod, "_modules"), dict):
            if "_nunchaku_runtime_root_model" in mod._modules:
                del mod._modules["_nunchaku_runtime_root_model"]
        if hasattr(mod, "_nunchaku_runtime_root_model"):
            try:
                delattr(mod, "_nunchaku_runtime_root_model")
            except Exception:
                pass
    except Exception:
        pass

    # Store original forward and mark patched (we keep LoRAs per-current_patcher, not on the module itself)
    setattr(mod, "_nunchaku_runtime_lora_orig_forward", orig_forward)
    setattr(mod, "_nunchaku_runtime_lora_patched", True)

    def _forward_with_runtime_lora(x, output=None):
        base_out = orig_forward(x, output)

        # Per-patcher runtime LoRAs:
        # - ModelPatcher.clone() shares the same underlying model object, so storing loras on the module
        #   would "leak" between patchers. Instead, read from the currently executing patcher.
        root_id = getattr(mod, "_nunchaku_runtime_root_id", None)
        root_model = None
        try:
            if isinstance(root_id, int):
                root_model = _NUNCHAKU_ROOT_MODEL_WEAK.get(root_id, None)
        except Exception:
            root_model = None
        patcher = getattr(root_model, "current_patcher", None) if root_model is not None else None
        if patcher is None and root_model is not None:
            patcher = getattr(root_model, "_nunchaku_runtime_last_patcher", None)
        attachments = getattr(patcher, "attachments", None) if patcher is not None else None
        if not isinstance(attachments, dict):
            # If we can't find a patcher, runtime LoRA cannot be applied.
            # Print once per module to avoid spamming.
            try:
                warned = getattr(mod, "_nunchaku_runtime_warned_no_patcher", False)
                if not warned:
                    setattr(mod, "_nunchaku_runtime_warned_no_patcher", True)
                    print(
                        "[NUNCHAKU_SDXL_LORA_DEBUG] SVDQ runtime forward has no patcher/attachments; "
                        "runtime LoRA NOT applied (model.current_patcher missing?)"
                    )
            except Exception:
                pass
            return base_out
        runtime = attachments.get("_nunchaku_svdq_runtime", None)
        if not isinstance(runtime, _NunchakuSVDQRuntime):
            return base_out

        loras = runtime.loras.get(id(mod), None)
        if not loras:
            return base_out

        # One-time "runtime is actually ACTIVE" confirmation (always; no spam)
        try:
            if not runtime.forward_active_logged:
                runtime.forward_active_logged = True
                total_loras = 0
                try:
                    total_loras = sum(len(v) for v in runtime.loras.values())
                except Exception:
                    total_loras = 0
                print(
                    "[NUNCHAKU_SDXL_LORA_DEBUG] SVDQ runtime forward ACTIVE: "
                    f"patched_modules={len(runtime.loras)} total_loras={total_loras}"
                )

            # Per-module one-time detail (debug only)
            if runtime.debug and id(mod) not in runtime.forward_seen:
                runtime.forward_seen.add(id(mod))
                _dbg_print(
                    True,
                    f"[NUNCHAKU_SDXL_LORA_DEBUG] SVDQ runtime forward ACTIVE: "
                    f"module={type(mod).__name__} id={id(mod)} loras={len(loras)} "
                    f"x_shape={tuple(getattr(x, 'shape', ())) } out_shape={tuple(getattr(base_out, 'shape', ())) }",
                )
        except Exception:
            pass

        # x can be (B,S,in) or (N,in).
        # Performance critical: avoid per-forward tensor transfers and Python loops.
        x_shape = x.shape
        x2 = x.reshape(-1, x_shape[-1])
        device = base_out.device
        if x2.device != device:
            x2 = x2.to(device=device)

        compute_dtype = _runtime_lora_compute_dtype(x2, base_out)
        if x2.dtype != compute_dtype:
            x2 = x2.to(dtype=compute_dtype)

        # Lazily prepare fused matrices for this module and device/dtype.
        prep = runtime.prepared.get(id(mod), None)
        needs_rebuild = False
        if not isinstance(prep, dict):
            needs_rebuild = True
        else:
            try:
                if prep.get("device", None) != device or prep.get("dtype", None) != compute_dtype:
                    needs_rebuild = True
            except Exception:
                needs_rebuild = True
        if id(mod) in getattr(runtime, "dirty", set()):
            needs_rebuild = True

        if needs_rebuild:
            # Fuse multiple LoRAs into a single pair of matrices to reduce kernel launches:
            #   add = (x @ D^T) @ U^T
            # where:
            #   D: (R_total, in_features)  (concat rows)
            #   U: (out_features, R_total) (concat columns), with per-LoRA scale folded into U.
            try:
                downs = []
                ups = []
                for (down_cpu, up_cpu, scale) in loras:
                    d = down_cpu.to(device=device, dtype=compute_dtype)
                    u = up_cpu.to(device=device, dtype=compute_dtype)
                    if scale != 1.0:
                        u = u * float(scale)
                    downs.append(d)
                    ups.append(u)
                d_cat = torch.cat(downs, dim=0).contiguous()  # (R_total, in)
                u_cat = torch.cat(ups, dim=1).contiguous()  # (out, R_total)
                down_t = d_cat.transpose(0, 1).contiguous()  # (in, R_total)
                up_t = u_cat.transpose(0, 1).contiguous()  # (R_total, out)
                runtime.prepared[id(mod)] = {"device": device, "dtype": compute_dtype, "down_t": down_t, "up_t": up_t}
                try:
                    runtime.dirty.discard(id(mod))
                except Exception:
                    pass
                prep = runtime.prepared.get(id(mod), None)
                if runtime.debug:
                    _dbg_print(
                        True,
                        "[NUNCHAKU_SDXL_LORA_DEBUG] SVDQ runtime prepared fused LoRA: "
                        f"module_id={id(mod)} device={device} dtype={compute_dtype} "
                        f"rank_total={int(d_cat.shape[0])} in={int(d_cat.shape[1])} out={int(u_cat.shape[0])}",
                    )
            except Exception as e:
                # If preparation fails, keep sampling safe by skipping runtime LoRA for this module.
                if runtime.debug:
                    _dbg_print(True, f"[NUNCHAKU_SDXL_LORA_DEBUG] SVDQ runtime prepare failed -> SKIP add: module_id={id(mod)} err={e}")
                return base_out

        if not isinstance(prep, dict):
            return base_out
        down_t = prep.get("down_t", None)
        up_t = prep.get("up_t", None)
        if down_t is None or up_t is None:
            return base_out

        # Core math: 2 GEMMs.
        add = (x2 @ down_t) @ up_t

        # Optional NaN/Inf safety check (OFF by default for performance).
        if _runtime_lora_should_check_finite(runtime):
            try:
                if not torch.isfinite(add).all():
                    if runtime.debug:
                        _dbg_print(
                            True,
                            f"[NUNCHAKU_SDXL_LORA_DEBUG] SVDQ runtime produced NaN/Inf -> SKIP add: "
                            f"module_id={id(mod)} add_min={float(add.nan_to_num().min().detach().cpu())} "
                            f"add_max={float(add.nan_to_num().max().detach().cpu())}",
                        )
                    return base_out
            except Exception:
                return base_out

        add = add.reshape(*x_shape[:-1], base_out.shape[-1]).to(dtype=base_out.dtype)
        return base_out + add

    # bind as method
    mod.forward = _forward_with_runtime_lora  # type: ignore
    _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] patched SVDQ forward: {type(mod).__name__}")
    return True


def _apply_runtime_lora_to_svdq_modules(model, lora_converted: dict, strength: float, debug: bool) -> dict:
    """
    Apply LoRA to Nunchaku SDXL quantized linears (SVDQW4A4Linear) by patching forward and
    attaching LoRA matrices. This handles the keys that cannot be applied via *.weight patching.
    """
    svdq_cls = _try_import_svdq_linear()
    if svdq_cls is None:
        _dbg_print(debug, "[NUNCHAKU_SDXL_LORA_DEBUG] SVDQW4A4Linear not importable; skip runtime SVDQ LoRA apply.")
        return {"enabled": False, "reason": "no_svdq_import"}

    if model is None or not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
        return {"enabled": False, "reason": "no_diffusion_model"}

    diffusion_model = model.model.diffusion_model
    root_model = model.model

    # Ensure ComfyUI can set current_patcher during sampling.
    # ModelPatcher.pre_run() only sets it if the attribute already exists on the model.
    try:
        if not hasattr(root_model, "current_patcher"):
            setattr(root_model, "current_patcher", None)
    except Exception:
        pass

    # Fallback pointer for environments where current_patcher is not used/set.
    # This is best-effort and does not suppress logs.
    try:
        setattr(root_model, "_nunchaku_runtime_last_patcher", model)
    except Exception:
        pass

    # Per-patcher storage (prevents LoRA leakage across clones)
    try:
        attachments = getattr(model, "attachments", None)
        if not isinstance(attachments, dict):
            attachments = {}
            setattr(model, "attachments", attachments)
        runtime = attachments.get("_nunchaku_svdq_runtime", None)
        if not isinstance(runtime, _NunchakuSVDQRuntime):
            runtime = _NunchakuSVDQRuntime(debug=debug)
            attachments["_nunchaku_svdq_runtime"] = runtime
        runtime.debug = bool(debug)
    except Exception:
        runtime = None

    # Register root model weakly so patched modules can reach current_patcher without creating cycles.
    try:
        _NUNCHAKU_ROOT_MODEL_WEAK[id(root_model)] = root_model
    except Exception:
        pass

    # build base->tensors map (we only need lora_up/down + alpha)
    bases = {}
    for k in list(lora_converted.keys()):
        if not isinstance(k, str):
            continue
        b = _lora_base_key_from_any(k)
        if not b.startswith("lora_unet_"):
            continue
        if b not in bases:
            bases[b] = []
        bases[b].append(k)

    applied = 0
    skipped = 0
    skipped_policy = 0
    errors = []
    unsupported = []
    skipped_policy_bases: set[str] = set()

    for base in sorted(bases.keys()):
        mod_path = _svdq_lora_base_to_module_dot_path(base)
        if mod_path is None:
            continue
        # Optional policy skip (explicit; logged). This is for "exception" layers that destabilize output.
        try:
            skip, reason = _svdq_runtime_should_skip(mod_path)
        except Exception:
            skip, reason = False, ""
        if skip:
            skipped_policy += 1
            skipped_policy_bases.add(base)
            print(
                "[NUNCHAKU_SDXL_LORA_DEBUG] [SKIP] SVDQ runtime apply skipped by policy: "
                f"base='{base}' path='{mod_path}' reason={reason}"
            )
            continue

        try:
            mod = _resolve_dot_path(diffusion_model, mod_path)
        except Exception as e:
            skipped += 1
            if debug:
                _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] SVDQ resolve failed: base='{base}' path='{mod_path}' err={e}")
            continue

        if not isinstance(mod, svdq_cls):
            # not quantized module (or different type) -> handled by standard loader
            continue

        try:
            up_key = f"{base}.lora_up.weight"
            down_key = f"{base}.lora_down.weight"
            alpha_key = f"{base}.alpha"
            if up_key not in lora_converted or down_key not in lora_converted:
                skipped += 1
                if debug:
                    _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] SVDQ runtime skip (missing up/down): base='{base}'")
                continue

            # Explicitly report variant keys that runtime forward-add cannot support (DoRA, norms, diff, etc.).
            # This keeps "unmapped_bases=0" honest: mapped != fully supported.
            try:
                variant_suffixes = (
                    ".dora_scale",
                    ".w_norm",
                    ".b_norm",
                    ".diff",
                    ".diff_b",
                    ".set_weight",
                    ".reshape_weight",
                    ".lora_mid.weight",
                )
                present_variants = [f"{base}{s}" for s in variant_suffixes if f"{base}{s}" in lora_converted]
                if present_variants:
                    unsupported.append((base, mod_path, present_variants[:20]))
            except Exception:
                pass

            up = lora_converted[up_key]
            down = lora_converted[down_key]
            rank = int(down.shape[0]) if hasattr(down, "shape") else 1
            alpha = float(lora_converted[alpha_key].item()) if alpha_key in lora_converted else None
            # mimic comfy.weight_adapter.LoRAAdapter alpha handling
            alpha_scale = (alpha / rank) if (alpha is not None and rank > 0) else 1.0
            scale = float(strength) * float(alpha_scale)

            _ensure_svdq_forward_patched(mod, debug)
            # Store ONLY the root id on the module (ints are safe; nn.Module would create a cycle)
            setattr(mod, "_nunchaku_runtime_root_id", int(id(root_model)))

            if isinstance(runtime, _NunchakuSVDQRuntime):
                lst = runtime.loras.get(id(mod), None)
                if not isinstance(lst, list):
                    lst = []
                    runtime.loras[id(mod)] = lst
                lst.append((down.detach().cpu(), up.detach().cpu(), scale))
                # Invalidate prepared cache for this module (rebuild lazily on first forward)
                try:
                    runtime.dirty.add(id(mod))
                    if id(mod) in runtime.prepared:
                        del runtime.prepared[id(mod)]
                except Exception:
                    pass
            applied += 1

            if debug and applied <= 20:
                _dbg_print(
                    debug,
                    f"[NUNCHAKU_SDXL_LORA_DEBUG] SVDQ runtime applied: base='{base}' -> '{mod_path}' "
                    f"rank={rank} alpha={alpha} scale={scale}",
                )
        except Exception as e:
            errors.append((base, mod_path, str(e)))

    _dbg_print(
        debug,
        f"[NUNCHAKU_SDXL_LORA_DEBUG] SVDQ runtime apply summary: applied_modules={applied} skipped={skipped} errors={len(errors)}",
    )
    # Always print a concise summary so users can confirm runtime attachment without enabling debug.
    try:
        print(
            "[NUNCHAKU_SDXL_LORA_DEBUG] SVDQ runtime attached: "
            f"applied_modules={applied} skipped={skipped} skipped_policy={skipped_policy} errors={len(errors)} strength={float(strength)}"
        )
    except Exception:
        pass
    # Always print unsupported runtime variants summary (auditable; does not suppress any logs).
    try:
        if unsupported:
            print(
                "[NUNCHAKU_SDXL_LORA_DEBUG] SVDQ runtime NOTE: unsupported variant keys detected "
                f"(runtime currently supports only up/down/alpha). count={len(unsupported)}"
            )
            for base, path, keys in unsupported[:40]:
                print(
                    "[NUNCHAKU_SDXL_LORA_DEBUG] SVDQ runtime unsupported variant: "
                    f"base='{base}' path='{path}' keys={keys}"
                )
    except Exception:
        pass
    if debug and errors:
        for b, p, e in errors[:20]:
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] SVDQ runtime error: base='{b}' path='{p}' err={e}")

    return {
        "enabled": True,
        "applied_modules": applied,
        "skipped": skipped,
        "skipped_policy": skipped_policy,
        "skipped_policy_bases_set": skipped_policy_bases,
        "errors": errors,
    }


def _is_nunchaku_sdxl_model_patcher(model) -> bool:
    """
    Detect our Nunchaku SDXL ModelPatcher instance (or compatible wrapper).
    We use this to avoid ComfyUI's SDXL diffusers->input_blocks remap that overwrites
    correct state_dict-based mappings for Nunchaku's diffusers-structured UNet.
    """
    try:
        if model is None:
            return False
        if type(model).__name__ == "NunchakuModelPatcher":
            return True
        m = getattr(model, "model", None)
        if m is None:
            return False
        # model_base.sdxl.NunchakuSDXL
        if type(m).__name__ == "NunchakuSDXL":
            return True
        mod = getattr(type(m), "__module__", "")
        # Keep this tolerant: module names don't include the folder name (and hyphens are invalid).
        return ("model_base.sdxl" in mod) or ("comfyui_nunchaku.model_base.sdxl" in mod)
    except Exception:
        return False


def _build_unet_key_map_state_dict_only(base_model) -> dict:
    """
    Build UNet key_map ONLY from the actual state_dict keys (diffusion_model.*),
    without calling comfy.utils.unet_to_diffusers().

    Why:
    - comfy.lora.model_lora_keys_unet() first adds correct mappings from state_dict,
      then *overwrites* many of them with a SDXL legacy mapping to input_blocks/output_blocks.
    - Nunchaku SDXL UNet is diffusers-structured (down_blocks/up_blocks/mid_block),
      so the overwrite causes patches to target non-existent keys => "NOT LOADED diffusion_model.input_blocks...".
    """
    key_map: dict[str, str] = {}
    sd = base_model.state_dict()
    for k in sd.keys():
        if not isinstance(k, str):
            continue
        if not k.startswith("diffusion_model."):
            continue

        if k.endswith(".weight"):
            key_lora = k[len("diffusion_model.") : -len(".weight")].replace(".", "_")
            key_map[f"lora_unet_{key_lora}"] = k
            # generic lora format without any weird key names
            key_map[k[:-len(".weight")]] = k
        else:
            # generic lora format for non-.weight keys
            key_map[k] = k

    return key_map


def _load_lora_for_models_state_dict_only(model, clip, lora_sd, strength_model, strength_clip, debug: bool):
    """
    Like comfy.sd.load_lora_for_models(), but uses state_dict-only UNet key_map to prevent
    wrong SDXL remaps to input_blocks/output_blocks.
    """
    key_map = {}
    if model is not None:
        try:
            key_map.update(_build_unet_key_map_state_dict_only(model.model))
        except Exception as e:
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] state_dict-only key_map_unet build failed: {e}")

    if clip is not None:
        try:
            key_map.update(comfy.lora.model_lora_keys_clip(clip.cond_stage_model, {}))
        except Exception as e:
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] key_map_clip build failed: {e}")

    lora_converted = comfy.lora_convert.convert_lora(lora_sd)
    # Normalize common base-key variants (does not suppress logs; only improves mapping determinism).
    try:
        lora_converted, _ = _normalize_unet_to_out_indexing(lora_converted, debug)
    except Exception as e:
        _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] normalize to_out indexing failed: {e}")

    # QI/ZIT-style "inspect every key" dump (debug only).
    # We print ONE line per converted key showing where it maps:
    # - standard: key_map has the base (patchable state_dict target exists)
    # - runtime: SVDQ quantized layer (forward-add) target exists
    # - runtime_skip: runtime target exists but skipped by policy
    # - runtime_unsupported: runtime target exists but this key is an unsupported variant (e.g. dora_scale)
    # - unmapped: neither standard nor runtime mapping found (should be 0 if coverage report is clean)
    if debug:
        def _group_of_base(b: str) -> str:
            if not isinstance(b, str):
                return "regular"
            if "_attn" in b:
                if any(x in b for x in ("_to_qkv", "_to_q", "_to_k", "_to_v")):
                    return "qkv"
                if "_to_out" in b:
                    return "regular"
                return "attn"
            if "_ff_net_" in b or "_ff_" in b:
                return "ff"
            return "regular"

        # Keys are already in ComfyUI "converted" form; keep deterministic ordering for audit logs.
        try:
            for k in sorted([x for x in lora_converted.keys() if isinstance(x, str)]):
                base = _lora_base_key_from_any(k)
                suffix = k[len(base):] if base and k.startswith(base) else ""
                group = _group_of_base(base)

                mapped_to = None
                route = "unmapped"
                reason = ""

                # Standard patch route
                if base in key_map:
                    mapped_to = key_map.get(base)
                    route = "standard"
                else:
                    # Runtime route (quantized SVDQ)
                    mod_path = _svdq_lora_base_to_module_dot_path(base)
                    if mod_path is not None:
                        # Determine if this key is supported by runtime (we only apply up/down/alpha).
                        runtime_supported = suffix in (".lora_down.weight", ".lora_up.weight", ".alpha")
                        if not runtime_supported:
                            route = "runtime_unsupported"
                            mapped_to = mod_path
                        else:
                            skip, skip_reason = _svdq_runtime_should_skip(mod_path)
                            if skip:
                                route = "runtime_skip"
                                mapped_to = mod_path
                                reason = skip_reason
                            else:
                                route = "runtime"
                                mapped_to = mod_path

                # Match QI/ZIT log style (same shape):
                #   Key: <key> -> Mapped to: <target> (Group: <group>)
                # Route/Reason are appended into the "Mapped to" text as tags, not as extra fields.
                tag = ""
                if route == "standard":
                    tag = " [standard]"
                elif route == "runtime":
                    tag = " [runtime]"
                elif route == "runtime_skip":
                    tag = f" [SKIP {reason}]" if reason else " [SKIP]"
                elif route == "runtime_unsupported":
                    tag = " [unsupported]"
                elif route == "unmapped":
                    tag = " [unmapped]"

                mapped_txt = f"{mapped_to}{tag}"
                print(f"Key: {k} -> Mapped to: {mapped_txt} (Group: {group})")
        except Exception as e:
            _dbg_print(True, f"[NUNCHAKU_SDXL_LORA_DEBUG] per-key mapping dump failed: {e}")
    # Strip runtime SVDQ keys BEFORE feeding to comfy.lora.load_lora() to avoid misleading
    # "lora key not loaded" logs for keys that are correctly handled by runtime forward-add.
    # (ZIT-style: runtime-handled keys don't produce "NOT LOADED" spam.)
    # IMPORTANT: Only keys that resolve to actual SVDQ modules are stripped.
    # Unmapped keys (neither standard nor runtime) are NOT stripped, so they will produce
    # "NOT LOADED" logs - this is intentional and desired, as it reveals true coverage gaps.
    lora_converted_stripped = lora_converted
    strip_stats_for_coverage = None
    if model is not None:
        try:
            lora_converted_stripped, strip_stats_for_coverage = _strip_runtime_svdq_keys(lora_converted, model, debug)
        except Exception as e:
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] strip runtime keys failed (using full): {e}")
            lora_converted_stripped = lora_converted
    loaded = comfy.lora.load_lora(lora_converted_stripped, key_map)

    if model is not None:
        new_modelpatcher = model.clone()
        k = set(new_modelpatcher.add_patches(loaded, strength_model))
        # Nunchaku SDXL: apply runtime LoRA for quantized SVDQ modules (forward-add), for keys that can't map to *.weight.
        # NOTE: Use the original lora_converted (not stripped) because runtime needs the full dict.
        runtime_apply_stats = {}
        try:
            runtime_apply_stats = _apply_runtime_lora_to_svdq_modules(new_modelpatcher, lora_converted, strength_model, debug)
        except Exception as e:
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] SVDQ runtime apply failed: {e}")
            runtime_apply_stats = {}
    else:
        k = set()
        new_modelpatcher = None
        runtime_apply_stats = {}

    if clip is not None:
        new_clip = clip.clone()
        k1 = set(new_clip.add_patches(loaded, strength_clip))
    else:
        k1 = set()
        new_clip = None

    # Match ComfyUI behavior: warn for patch targets that didn't get applied to either model or clip.
    for x in loaded:
        if (x not in k) and (x not in k1):
            logging.warning("NOT LOADED {}".format(x))

    # "Complete mapping" audit: every UNet base key is either standard-mappable or runtime-handled.
    try:
        # Detect runtime-bases: reuse strip_stats from earlier to avoid duplicate work.
        runtime_bases = set()
        runtime_skipped_bases = set()
        try:
            if strip_stats_for_coverage is not None:
                runtime_bases = strip_stats_for_coverage.get("runtime_bases_set", set()) if isinstance(strip_stats_for_coverage, dict) else set()
            else:
                # Fallback: recalculate if strip_stats wasn't computed earlier.
                _, strip_stats = _strip_runtime_svdq_keys(lora_converted, model, debug)
                runtime_bases = strip_stats.get("runtime_bases_set", set()) if isinstance(strip_stats, dict) else set()
            if not isinstance(runtime_bases, set):
                runtime_bases = set()
        except Exception:
            runtime_bases = set()
        try:
            runtime_skipped_bases = runtime_apply_stats.get("skipped_policy_bases_set", set()) if isinstance(runtime_apply_stats, dict) else set()
            if not isinstance(runtime_skipped_bases, set):
                runtime_skipped_bases = set()
        except Exception:
            runtime_skipped_bases = set()
        _coverage_report_for_unet(lora_converted, key_map, runtime_bases, runtime_skipped_bases, debug)
    except Exception as e:
        # Only raise in debug (coverage_report does); in non-debug, keep sampling safe.
        if debug:
            raise
        _dbg_print(True, f"[NUNCHAKU_SDXL_LORA_DEBUG] UNet mapping coverage check failed: {e}")

    return (new_modelpatcher, new_clip)


def _safe_type_info(x) -> str:
    try:
        return f"{type(x).__module__}.{type(x).__name__}"
    except Exception:
        return "<unknown>"


def _safe_len(x) -> int | None:
    try:
        return len(x)
    except Exception:
        return None


def _top_counts(items: list[str], topn: int = 30) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for it in items:
        counts[it] = counts.get(it, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:topn]


def _prefix_bucket(k: str, max_parts: int = 4) -> str:
    """
    Group keys by the first few path parts to make huge logs readable.
    Supports both '_' and '.' separators (LoRA formats vary).
    """
    if not isinstance(k, str) or not k:
        return ""
    # Prefer '_' splitting (most ComfyUI lora keys); fall back to '.'.
    parts = k.split("_")
    if len(parts) <= 1:
        parts = k.split(".")
    return "_".join(parts[:max_parts])


def _best_suffix_candidates(query: str, candidates: list[str], topn: int = 5) -> list[str]:
    """
    Very cheap "候補提示":
    - Compare by common suffix length (string) to suggest nearest key_map keys.
    """
    if not query:
        return []

    scored: list[tuple[int, str]] = []
    q = query
    for c in candidates:
        # common suffix length
        i = 1
        max_i = min(len(q), len(c))
        while i <= max_i and q[-i] == c[-i]:
            i += 1
        common = i - 1
        if common <= 0:
            continue
        scored.append((common, c))

    scored.sort(key=lambda t: (-t[0], t[1]))
    return [c for _, c in scored[:topn]]


def _lora_unet_base_to_possible_state_dict_weights(lora_unet_base: str) -> list[str]:
    """
    Heuristic mapping from a ComfyUI-style UNet LoRA base key:
      lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_q
    into possible UNet state_dict weight keys:
      diffusion_model.down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q.weight
      diffusion_model.down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_q.weight

    Debug-only: used to decide if a "not loaded" key is due to missing weight keys
    (common with quantized/packed layers) vs. purely key mapping mismatch.
    """
    if not isinstance(lora_unet_base, str) or not lora_unet_base.startswith("lora_unet_"):
        return []

    body = lora_unet_base[len("lora_unet_") :]
    dotted = body.replace("_", ".")

    # Fix common "block names" that contain underscores (our '_'->'.' conversion breaks them)
    dotted = dotted.replace("down.blocks.", "down_blocks.")
    dotted = dotted.replace("up.blocks.", "up_blocks.")
    dotted = dotted.replace("mid.block.", "mid_block.")
    dotted = dotted.replace("transformer.blocks.", "transformer_blocks.")

    # Restore tokens that are not pure hierarchy separators
    dotted = dotted.replace("proj.in", "proj_in")
    dotted = dotted.replace("proj.out", "proj_out")
    dotted = dotted.replace("time.emb.proj", "time_emb_proj")

    # Convert attnX.to.(q/k/v/qkv/out.*) -> attnX.to_(...)
    for a in ("attn1", "attn2"):
        dotted = dotted.replace(f"{a}.to.qkv", f"{a}.to_qkv")
        dotted = dotted.replace(f"{a}.to.q", f"{a}.to_q")
        dotted = dotted.replace(f"{a}.to.k", f"{a}.to_k")
        dotted = dotted.replace(f"{a}.to.v", f"{a}.to_v")
        dotted = dotted.replace(f"{a}.to.out.0", f"{a}.to_out.0")
        dotted = dotted.replace(f"{a}.to.out", f"{a}.to_out")

    base = f"diffusion_model.{dotted}"
    candidates = [
        base + ".weight",
        base.replace(".to_", ".processor.to_") + ".weight",
        base.replace(".to_out.0", ".to_out") + ".weight",
        base.replace(".to_out.0", ".to_out").replace(".to_", ".processor.to_") + ".weight",
    ]

    out: list[str] = []
    seen = set()
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _detect_lora_type(sd: dict) -> dict:
    """
    QI/ZIT並みに「LoRAタイプ判別」を出すための簡易ヒューリスティック。
    ここで言う "type" は厳密な規格名ではなく、キーの傾向からの推定。
    """
    keys = [k for k in sd.keys() if isinstance(k, str)]
    key_set = set(keys)

    # Common flags
    def any_prefix(p: str) -> bool:
        return any(k.startswith(p) for k in keys)

    def any_contains(s: str) -> bool:
        return any(s in k for k in keys)

    flags = {
        # UNet
        "unet_a1111_input_blocks": any_prefix("lora_unet_input_blocks_"),
        "unet_a1111_output_blocks": any_prefix("lora_unet_output_blocks_"),
        "unet_a1111_middle_block": any_prefix("lora_unet_middle_block_"),
        "unet_comfy_diffusers_blocks": any_prefix("lora_unet_down_blocks_") or any_prefix("lora_unet_up_blocks_") or any_prefix("lora_unet_mid_block_"),
        "unet_diffusers_prefix": any_prefix("unet.") or any_prefix("unet_"),
        "unet_generic_diffusion_model_prefix": any_prefix("diffusion_model."),

        # Text encoder
        "clip_a1111_te": any_prefix("lora_te_") or any_prefix("lora_te1_") or any_prefix("lora_te2_"),
        "clip_diffusers_te": any_prefix("text_encoder.") or any_prefix("text_encoder_2.") or any_prefix("text_encoders."),

        # Adapter variants
        "has_dora_scale": any_contains(".dora_scale"),
        "has_diff": any_contains(".diff") or any_contains(".diff_b") or any_contains(".set_weight"),
        "has_w_norm": any_contains(".w_norm") or any_contains(".b_norm"),

        # LyCORIS-ish / other training tool signatures (very rough)
        "has_lycoris_prefix": any_prefix("lycoris_"),
        "has_loha": any_contains("hada_w1") or any_contains("hada_w2"),
        "has_lokr": any_contains("lokr_") or any_contains("lokr.") or any_contains("lokr_w") or any_contains("lokr_diff"),
        "has_ia3": any_contains("ia3_") or any_contains("ia3."),
    }

    # Determine primary "format" tags for logging
    tags: list[str] = []
    if flags["unet_a1111_input_blocks"] or flags["unet_a1111_output_blocks"] or flags["unet_a1111_middle_block"]:
        tags.append("UNet=A1111(input_blocks/output_blocks/middle_block)")
    if flags["unet_comfy_diffusers_blocks"]:
        tags.append("UNet=Comfy(diffusers down/up/mid blocks)")
    if flags["unet_diffusers_prefix"]:
        tags.append("UNet=Diffusers(unet.*)")

    if flags["clip_a1111_te"]:
        tags.append("CLIP=A1111(lora_te*)")
    if flags["clip_diffusers_te"]:
        tags.append("CLIP=Diffusers(text_encoder*)")

    if flags["has_dora_scale"]:
        tags.append("DoRA=present")
    if flags["has_lycoris_prefix"]:
        tags.append("LyCORIS=present")
    if flags["has_loha"]:
        tags.append("LoHa=maybe")
    if flags["has_lokr"]:
        tags.append("LoKr=maybe")
    if flags["has_ia3"]:
        tags.append("IA3=maybe")
    if flags["has_diff"] or flags["has_w_norm"]:
        tags.append("DIFF/SET/NORM=present")

    # Some direct key examples (helps user verify quickly)
    samples = sorted(keys)[:40]

    return {
        "num_keys": len(keys),
        "flags": flags,
        "tags": tags,
        "samples_head": samples,
        "has_metadata_like_keys": any(k in key_set for k in ("__metadata__", "metadata", "vocab", "merges")),
    }


def _a1111_base_to_input_dot(base_wo_prefix: str) -> str | None:
    """
    Convert "input_blocks_4_1_transformer_blocks_0_attn1_to_q" style into
    "input_blocks.4.1.transformer_blocks.0.attn1.to_q" style (approx).

    This is intentionally narrow: it only targets the common SDXL A1111 UNet LoRA naming.
    Returns None if it cannot parse.
    """
    if not isinstance(base_wo_prefix, str) or not base_wo_prefix:
        return None

    # Detect which root we are dealing with and parse the numeric indices
    root = None
    rest = None
    a = b = None

    for r in ("input_blocks_", "output_blocks_"):
        if base_wo_prefix.startswith(r):
            root = r[:-1]  # drop trailing "_"
            rest = base_wo_prefix[len(r):]
            break
    if root is None:
        if base_wo_prefix.startswith("middle_block_"):
            # middle_block has only one attention/resnet index in most keys: middle_block_1_...
            # We'll keep it as "middle_block.<idx>.<tail>"
            root = "middle_block"
            rest = base_wo_prefix[len("middle_block_"):]
            parts = rest.split("_", 1)
            if len(parts) != 2:
                return None
            a = parts[0]
            tail = parts[1]
            prefix = f"{root}.{a}."
            return prefix + _a1111_tail_to_dot(tail)
        return None

    # input/output blocks: expect "{a}_{b}_{tail...}"
    parts = rest.split("_", 2)
    if len(parts) < 3:
        return None
    a, b, tail = parts[0], parts[1], parts[2]
    prefix = f"{root}.{a}.{b}."
    return prefix + _a1111_tail_to_dot(tail)


def _a1111_tail_to_dot(tail: str) -> str:
    """
    Transform the tail part after block indices into dot-form expected by comfy.utils.unet_to_diffusers mappings.
    Example:
      transformer_blocks_0_attn1_to_q -> transformer_blocks.0.attn1.to_q
      ff_net_0_proj -> ff.net.0.proj
      attn1_to_out_0 -> attn1.to_out.0
    """
    if not isinstance(tail, str) or not tail:
        return ""

    # NOTE:
    # A1111 UNet LoRA tail uses "_" as a *hierarchy separator*, but some tokens themselves include "_" (e.g. to_q, proj_in).
    # We must NOT blindly do "_" -> ".", otherwise:
    #   - to_q becomes to.q (mapping miss)
    #   - proj_in becomes proj.in (mapping miss)
    # which is exactly what caused many "unknown base" and ultimately hit_bases=0.

    out = tail

    # 1) Protect tokens that must keep "_" inside (placeholders without underscores)
    out = out.replace("proj_in", "PROJIN")
    out = out.replace("proj_out", "PROJOUT")
    out = out.replace("time_emb_proj", "TIMEEMBPROJ")
    out = out.replace("to_qkv", "TOQKV")
    out = out.replace("to_q", "TOQ")
    out = out.replace("to_k", "TOK")
    out = out.replace("to_v", "TOV")
    out = out.replace("to_out", "TOOUT")

    # 2) Structural rewrites (keep token semantics)
    # transformer_blocks_{n}_... -> transformer_blocks.{n}. ...
    out = out.replace("transformer_blocks_", "transformer_blocks.")
    # ff_net_{n}_... -> ff.net.{n}. ...
    out = out.replace("ff_net_", "ff.net.")
    # attn1_to_* / attn2_to_* become attnX.to_*  (we protected to_q/k/v/out above)
    out = out.replace("attn1_to_", "attn1.")
    out = out.replace("attn2_to_", "attn2.")

    # 3) Now convert remaining hierarchy "_" separators into "."
    out = out.replace("_", ".")

    # 4) Restore protected tokens back (with underscores where needed)
    out = out.replace("PROJIN", "proj_in")
    out = out.replace("PROJOUT", "proj_out")
    out = out.replace("TIMEEMBPROJ", "time_emb_proj")
    out = out.replace("TOQKV", "to_qkv")
    out = out.replace("TOQ", "to_q")
    out = out.replace("TOK", "to_k")
    out = out.replace("TOV", "to_v")
    out = out.replace("TOOUT", "to_out")

    # 5) Special-case A1111 naming: to_out_0 -> to_out.0 (diffusers expects ModuleList index)
    # This must happen after restoration.
    out = out.replace("to_out.0", "to_out.0")  # idempotent
    out = out.replace("to_out.0.", "to_out.0.")  # idempotent
    out = out.replace("to_out.0", "to_out.0")
    out = out.replace("to_out.0", "to_out.0")
    out = out.replace("to_out.0", "to_out.0")
    out = out.replace("to_out.0", "to_out.0")
    out = out.replace("to_out.0", "to_out.0")
    out = out.replace("to_out.0", "to_out.0")
    # If A1111 tail had explicit "to_out_0", after step (3) it becomes "to_out.0" already, so nothing else required.

    # Some keys become like "transformer.blocks.0..." if we naïvely replace; fix "transformer_blocks."
    out = out.replace("transformer.blocks.", "transformer_blocks.")

    return out


def _fuse_sdxl_attn1_qkv_lora(lora_sd: dict, debug: bool) -> tuple[dict, dict]:
    """
    Nunchaku SDXL self-attn (attn1) uses fused QKV (to_qkv). Many SDXL LoRAs are trained on unfused Q/K/V.
    This function fuses (attn1_to_q, attn1_to_k, attn1_to_v) LoRA weights into a single (attn1_to_qkv) LoRA.

    Why this works with ComfyUI's LoRA adapter:
    - ComfyUI applies deltaW = (alpha / rank) * (up @ down)
    - We build a block-diagonal LoRA with rank_total = rq+rk+rv and set alpha_total = rank_total (so global scale=1)
    - We pre-scale each sub-up matrix by (alpha_i / rank_i) to preserve original per-projection scaling.

    Returns: (new_state_dict, stats)
    """
    if not isinstance(lora_sd, dict):
        return lora_sd, {"fused": False, "reason": "lora_sd is not a dict"}

    def _get_tensor(k: str):
        v = lora_sd.get(k, None)
        return v if hasattr(v, "shape") else None

    # Group by prefix before "_attn1_to_" (works for lora_unet_* keys)
    # We accept partial presence of q/k/v, because some LoRAs may omit one of them.
    groups: dict[str, dict[str, str]] = {}
    for k in list(lora_sd.keys()):
        if not isinstance(k, str):
            continue
        base = _lora_base_key(k)
        if "_attn1_to_" not in base:
            continue
        if base.endswith("_attn1_to_q") or base.endswith("_attn1_to_k") or base.endswith("_attn1_to_v"):
            prefix = base.rsplit("_attn1_to_", 1)[0]
            suffix = base.rsplit("_attn1_to_", 1)[1]
            groups.setdefault(prefix, {})[suffix] = base

    if not groups:
        return lora_sd, {"fused": False, "reason": "no attn1_to_q/k/v bases found"}

    out = dict(lora_sd)
    fused_count = 0
    fused_prefixes = []
    errors: list[str] = []

    for prefix, parts in groups.items():
        # Skip if fused already exists (avoid overwriting user-provided fused weights).
        fused_base = f"{prefix}_attn1_to_qkv"
        if (
            f"{fused_base}.lora_down.weight" in out
            or f"{fused_base}.lora_up.weight" in out
            or f"{fused_base}.alpha" in out
        ):
            continue

        # Collect present projections
        present = {}
        for p in ("q", "k", "v"):
            b = parts.get(p, None)
            if not b:
                continue
            up = _get_tensor(f"{b}.lora_up.weight")
            dn = _get_tensor(f"{b}.lora_down.weight")
            if up is None or dn is None:
                continue
            present[p] = (b, up, dn)

        if not present:
            continue

        try:
            # Determine dimensions from any present projection
            any_b, any_up, any_dn = next(iter(present.values()))
            in_dim = int(any_dn.shape[1])
            dim = int(any_up.shape[0])

            # Validate dimensions across present projections
            for p, (b, up, dn) in present.items():
                if int(dn.shape[1]) != in_dim:
                    raise ValueError(f"in_features mismatch: {p}={tuple(dn.shape)} expected_in={in_dim}")
                if int(up.shape[0]) != dim:
                    raise ValueError(f"out_features mismatch: {p}={tuple(up.shape)} expected_out={dim}")

            # Per-projection alpha scaling (ComfyUI uses alpha / rank)
            def _alpha_of(base: str, r: int) -> float:
                a = out.get(f"{base}.alpha", None)
                try:
                    return float(a.item()) if a is not None else float(r)
                except Exception:
                    return float(r)

            # Build fused:
            # - to_qkv output dim is 3*dim (q,k,v stacked)
            # - rank is sum of present ranks; missing projections contribute zero rows
            ranks = {p: int(dn.shape[0]) for p, (_, _, dn) in present.items()}
            r_total = sum(int(r) for r in ranks.values())
            if r_total <= 0:
                continue

            dn_fused = any_dn.new_zeros((r_total, in_dim))
            up_fused = any_up.new_zeros((3 * dim, r_total))

            # Fill blocks in rank dimension; place outputs in q/k/v segments.
            # q segment rows: [0:dim], k: [dim:2*dim], v: [2*dim:3*dim]
            rank_cursor = 0
            for p, row_off in (("q", 0), ("k", dim), ("v", 2 * dim)):
                if p not in present:
                    continue
                b, up, dn = present[p]
                r = int(dn.shape[0])
                if r <= 0:
                    continue
                scale = _alpha_of(b, r) / max(r, 1)
                dn_fused[rank_cursor : rank_cursor + r, :] = dn
                up_fused[row_off : row_off + dim, rank_cursor : rank_cursor + r] = up * float(scale)
                rank_cursor += r

            out[f"{fused_base}.lora_down.weight"] = dn_fused
            out[f"{fused_base}.lora_up.weight"] = up_fused
            # Set alpha so global scale becomes 1.0 (alpha / rank_total = 1)
            out[f"{fused_base}.alpha"] = dn_fused.new_tensor(float(r_total))

            # Remove original q/k/v keys to avoid accidental application on unfused targets
            for p in ("q", "k", "v"):
                if p not in present:
                    continue
                b = present[p][0]
                for sfx in (".lora_down.weight", ".lora_up.weight", ".alpha", ".dora_scale", ".w_norm", ".b_norm"):
                    out.pop(f"{b}{sfx}", None)

            fused_count += 1
            if len(fused_prefixes) < 20:
                fused_prefixes.append(prefix)
        except Exception as e:
            errors.append(f"{prefix}: {e}")

    stats = {
        "fused": fused_count > 0,
        "fused_groups": fused_count,
        "fused_prefix_samples": fused_prefixes,
        "errors": errors[:20],
    }
    if debug:
        _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] attn1 qkv fusion: {stats}")
    return out, stats


def _convert_a1111_unet_lora_keys_to_comfy_diffusers(lora_sd: dict, model, debug: bool) -> tuple[dict, dict]:
    """
    Convert A1111 SDXL UNet LoRA keys (lora_unet_input_blocks_*/output_blocks_*/middle_block_*)
    to ComfyUI's diffusers-mapped keys (lora_unet_down_blocks_*/up_blocks_*/mid_block_*).

    Returns: (converted_state_dict, stats)
    """
    # Build reverse base mapping: input_blocks.* -> down_blocks./up_blocks./mid_block.
    try:
        unet_config = model.model.model_config.unet_config
    except Exception:
        unet_config = None

    synth_used = False
    synth_reason = None

    if not isinstance(unet_config, dict):
        # Some custom model configs don't expose the full StableDiffusion UNet config.
        # We can still synthesize a minimal SDXL mapping config from known SDXL base structure.
        unet_config = {}
        synth_used = True
        synth_reason = "model.model.model_config.unet_config missing"

    # If required fields are missing, synthesize SDXL defaults.
    # This Nunchaku loader is SDXL base UNet (not refiner) and matches the common SDXL block layout.
    if "num_res_blocks" not in unet_config or "channel_mult" not in unet_config:
        synth_used = True
        synth_reason = synth_reason or "unet_config missing num_res_blocks/channel_mult"
        # SDXL base defaults in ComfyUI terms
        unet_config = dict(unet_config)
        unet_config.setdefault("num_res_blocks", [2, 2, 2])
        unet_config.setdefault("channel_mult", [1, 2, 4])
        # Down transformer depths: provided by loader (len=6) or fallback to SDXL base
        unet_config.setdefault("transformer_depth", unet_config.get("transformer_depth", [0, 0, 2, 2, 10, 10]))
        # Up transformer depths: l=sum(num_res_blocks)+num_blocks -> 9 for [2,2,2]
        unet_config.setdefault("transformer_depth_output", [0, 0, 0, 2, 2, 2, 10, 10, 10])
        # Mid transformer depth (SDXL base uses 10)
        unet_config.setdefault("transformer_depth_middle", 10)

    # Ensure transformer_depth_output exists; if missing, synthesize from transformer_depth.
    if "transformer_depth_output" not in unet_config:
        synth_used = True
        synth_reason = synth_reason or "unet_config missing transformer_depth_output"
        try:
            nrb = list(unet_config.get("num_res_blocks", [2, 2, 2]))
            nb = len(unet_config.get("channel_mult", [1, 2, 4]))
            l = sum(int(x) for x in nrb) + int(nb)
            td = list(unet_config.get("transformer_depth", [0, 0, 2, 2, 10, 10]))
            # Repeat each down-depth per block to approximate up-depth (common SDXL symmetry)
            # Down depths are per-resnet; up needs per-(resnet+1) so we extend by repeating last per-block value once.
            per_block = []
            idx = 0
            for br in nrb:
                vals = td[idx:idx + br]
                idx += br
                per_block.append(int(vals[-1]) if vals else 0)
            td_out = []
            for v in per_block:
                # each up-block has (num_res_blocks+1) entries
                td_out.extend([v, v, v])
            # Trim/pad to length l
            if len(td_out) < l:
                td_out.extend([0] * (l - len(td_out)))
            unet_config["transformer_depth_output"] = td_out[:l]
        except Exception:
            unet_config["transformer_depth_output"] = [0, 0, 0, 2, 2, 2, 10, 10, 10]

    # Ensure transformer_depth_middle exists
    if "transformer_depth_middle" not in unet_config:
        synth_used = True
        synth_reason = synth_reason or "unet_config missing transformer_depth_middle"
        unet_config["transformer_depth_middle"] = 10

    diff_map = comfy.utils.unet_to_diffusers(unet_config)  # diffusers_key(with weight/bias) -> original_key(input/output/middle)
    if not diff_map:
        stats = {"converted": False, "reason": "unet_to_diffusers returned empty mapping", "synth_used": synth_used, "synth_reason": synth_reason}
        _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] LoRA UNet key conversion skipped: {stats}")
        return lora_sd, stats

    # Reverse base mapping (strip .weight/.bias)
    reverse_base: dict[str, str] = {}
    for dkey, okey in diff_map.items():
        if not (isinstance(dkey, str) and isinstance(okey, str)):
            continue
        for suffix in (".weight", ".bias"):
            if dkey.endswith(suffix) and okey.endswith(suffix):
                reverse_base[okey[:-len(suffix)]] = dkey[:-len(suffix)]

    # Perform conversion by rewriting base keys
    converted = {}
    total = 0
    changed = 0
    changed_bases: list[tuple[str, str]] = []
    unknown_bases: list[str] = []

    for k, v in lora_sd.items():
        total += 1
        if not isinstance(k, str):
            converted[k] = v
            continue

        base = _lora_base_key(k)
        suffix = k[len(base):] if base else ""

        if base.startswith("lora_unet_input_blocks_") or base.startswith("lora_unet_output_blocks_") or base.startswith("lora_unet_middle_block_"):
            base_wo = base[len("lora_unet_"):]
            input_dot = _a1111_base_to_input_dot(base_wo)
            if input_dot is None:
                unknown_bases.append(base)
                converted[k] = v
                continue

            # Try mapping in both "input_blocks.*" and "output_blocks.*" and "middle_block.*"
            diffusers_dot = reverse_base.get(input_dot, None)
            if diffusers_dot is None:
                # No mapping found; keep original
                unknown_bases.append(base)
                converted[k] = v
                continue

            new_base = "lora_unet_" + diffusers_dot.replace(".", "_")
            new_k = new_base + suffix
            converted[new_k] = v
            changed += 1
            if len(changed_bases) < 30:
                changed_bases.append((base, new_base))
            continue

        # Leave other keys untouched (CLIP, already-diffusers, etc)
        converted[k] = v

    stats = {
        "converted": True,
        "total_keys": total,
        "changed_keys": changed,
        "changed_bases_samples": changed_bases,
        "unknown_base_samples": unknown_bases[:40],
        "reverse_base_size": len(reverse_base),
        "synth_used": synth_used,
        "synth_reason": synth_reason,
    }

    _dbg_print(
        debug,
        "[NUNCHAKU_SDXL_LORA_DEBUG] LoRA UNet key conversion (A1111->diffusers): "
        f"reverse_base_size={stats['reverse_base_size']} total_keys={total} changed_keys={changed} "
        f"synth_used={stats['synth_used']} synth_reason={stats['synth_reason']}",
    )
    if debug:
        _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] conversion samples (old_base -> new_base): {stats['changed_bases_samples']}")
        if stats["unknown_base_samples"]:
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] conversion unknown base samples: {stats['unknown_base_samples']}")

    # Post-pass: Nunchaku SDXL self-attn (attn1) uses fused to_qkv, so try to fuse q/k/v LoRAs.
    try:
        converted2, fuse_stats = _fuse_sdxl_attn1_qkv_lora(converted, debug)
        stats["attn1_qkv_fusion"] = fuse_stats
        return converted2, stats
    except Exception as e:
        stats["attn1_qkv_fusion"] = {"fused": False, "reason": str(e)}
        return converted, stats


def _lora_base_key(k: str) -> str:
    """
    Return the "base" LoRA key used by comfy.lora.load_lora(), i.e. the prefix before:
      - .lora_down.weight / .lora_up.weight
      - .alpha / .dora_scale / .w_norm / .b_norm / .diff / .diff_b / .set_weight

    IMPORTANT:
    Do NOT split on the first dot, because diffusers-style LoRA keys include many dots
    (e.g. "unet.down_blocks.0....to_q.lora_down.weight").
    """
    if not isinstance(k, str) or not k:
        return ""

    suffixes = (
        ".lora_down.weight",
        ".lora_up.weight",
        # PEFT / diffusers variants
        ".lora_A.weight",
        ".lora_B.weight",
        ".lora_A.default.weight",
        ".lora_B.default.weight",
        ".alpha",
        ".lora_alpha",
        ".dora_scale",
        ".w_norm",
        ".b_norm",
        ".diff",
        ".diff_b",
        ".set_weight",
    )
    for s in suffixes:
        if k.endswith(s):
            return k[: -len(s)]

    # Some adapters may store weights without the trailing ".weight" (rare). Keep sane fallback.
    for s in (".lora_down", ".lora_up"):
        if k.endswith(s):
            return k[: -len(s)]

    # Fallback: return the whole key (safe; might be unmapped but won't incorrectly collapse to "unet")
    return k


def _apply_lora_filtered(model, clip, lora_sd, strength_model: float, strength_clip: float, debug: bool):
    """
    Apply LoRA using the same mechanism as comfy.sd.load_lora_for_models, but with filtering to
    reduce 'lora key not loaded' spam and to provide clearer diagnostics.
    """
    # Build key maps exactly like ComfyUI does.
    unet_key_map = {}
    clip_key_map = {}
    if model is not None:
        unet_key_map = comfy.lora.model_lora_keys_unet(model.model, unet_key_map)
    if clip is not None:
        clip_key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, clip_key_map)

    key_map = {}
    key_map.update(unet_key_map)
    key_map.update(clip_key_map)

    _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] model={_safe_type_info(model)} clip={_safe_type_info(clip)}")
    _dbg_print(
        debug,
        "[NUNCHAKU_SDXL_LORA_DEBUG] key_map sizes: "
        f"unet={len(unet_key_map)} clip={len(clip_key_map)} total={len(key_map)}",
    )
    if debug and model is not None:
        try:
            sd_keys = list(model.model.state_dict().keys())
            has_diffusion_model = any(k.startswith("diffusion_model.") for k in sd_keys)
            _dbg_print(
                debug,
                "[NUNCHAKU_SDXL_LORA_DEBUG] model.model.state_dict: "
                f"keys={len(sd_keys)} diffusion_model.*={has_diffusion_model} sample0={sd_keys[0] if sd_keys else 'None'}",
            )
        except Exception as e:
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] state_dict probe failed: {e}")

    # Convert then filter LoRA to keys that are actually mappable.
    lora_converted = comfy.lora_convert.convert_lora(lora_sd)
    allowed_bases = set(key_map.keys())

    total_keys = 0
    kept_keys = 0
    kept_bases = set()
    filtered = {}
    sample_keys = []
    sample_bases = []
    for k, v in lora_converted.items():
        total_keys += 1
        base = _lora_base_key(k)
        if debug and len(sample_keys) < 12:
            sample_keys.append(k)
            sample_bases.append(base)
        if base in allowed_bases:
            filtered[k] = v
            kept_keys += 1
            kept_bases.add(base)

    _dbg_print(
        debug,
        "[NUNCHAKU_SDXL_LORA_DEBUG] lora keys: "
        f"converted_total={total_keys} kept={kept_keys} kept_bases={len(kept_bases)}",
    )
    if debug:
        _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] lora sample keys={sample_keys}")
        _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] lora sample bases={sample_bases}")

    if model is not None and len(unet_key_map) == 0:
        _dbg_print(
            debug,
            "[NUNCHAKU_SDXL_LORA_DEBUG] WARNING: unet_key_map is empty. "
            "This usually means the model doesn't expose ComfyUI UNet weights (diffusion_model.*), "
            "so standard LoRA patching cannot affect the UNet.",
        )

    loaded = comfy.lora.load_lora(filtered, key_map)

    if model is not None:
        new_modelpatcher = model.clone()
        new_modelpatcher.add_patches(loaded, strength_model)
    else:
        new_modelpatcher = None

    if clip is not None:
        new_clip = clip.clone()
        new_clip.add_patches(loaded, strength_clip)
    else:
        new_clip = None

    return new_modelpatcher, new_clip


def _apply_lora_verbose_like_qi(model, clip, lora_sd, strength_model: float, strength_clip: float, debug: bool):
    """
    QI/ZIT側と同じ「大量ログ（未ロードキーを全部吐く）」で検証できるように、
    comfy.sd.load_lora_for_models() を“そのまま”使う。
    """
    if debug:
        # ------------------------------------------------------------------
        # QI/ZIT並みの「巨大ログ」：型情報＋当たり判定＋候補提示まで一気に出す
        # ------------------------------------------------------------------
        _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] 🔍 model type: '{type(model).__name__}'")
        _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] 🔍 model module: {type(model).__module__}")
        _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] 🔍 Type repr: {repr(type(model))}")
        _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] 🔍 Has 'model' attr? {hasattr(model, 'model')}")
        _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] 🔍 Has 'clone' attr? {hasattr(model, 'clone')}")
        _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] 🔍 clip type: '{type(clip).__name__}' module={type(clip).__module__}")
        _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] 🔍 clip has 'cond_stage_model'? {hasattr(clip, 'cond_stage_model')}")

        # ---- target model structure probe (SDXL UNet側が patchable かの確定) ----
        try:
            base = getattr(model, "model", None)
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] model.model={_safe_type_info(base)}")
            dm = getattr(base, "diffusion_model", None)
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] model.model.diffusion_model={_safe_type_info(dm)}")
            try:
                n_params = sum(1 for _ in dm.parameters()) if dm is not None and hasattr(dm, "parameters") else None
            except Exception:
                n_params = None
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] diffusion_model.parameters count={n_params}")

            sdk = list(base.state_dict().keys()) if base is not None else []
            has_dm = any(k.startswith("diffusion_model.") for k in sdk)
            head_sd = sdk[:30]
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] model.model.state_dict: keys={len(sdk)} diffusion_model.*={has_dm} head={head_sd}")
        except Exception as e:
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] model structure probe failed: {e}")

        # ---- key_map (ComfyUIと同じ) ----
        key_map_unet = {}
        key_map_clip = {}
        key_map_all = {}
        try:
            key_map_unet = comfy.lora.model_lora_keys_unet(model.model, {})
        except Exception as e:
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] key_map_unet build failed: {e}")
        try:
            key_map_clip = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, {})
        except Exception as e:
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] key_map_clip build failed: {e}")
        key_map_all.update(key_map_unet)
        key_map_all.update(key_map_clip)
        _dbg_print(
            debug,
            f"[NUNCHAKU_SDXL_LORA_DEBUG] key_map sizes: unet={len(key_map_unet)} clip={len(key_map_clip)} total={len(key_map_all)}",
        )
        if debug:
            # Show what *kind* of keys exist in key_map (this directly explains hit_bases=0).
            try:
                km_keys = list(key_map_all.keys())
                # Prefix histograms (first few parts)
                km_prefixes = [_prefix_bucket(k, 4) for k in km_keys[:5000]]
                _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] key_map prefix top={_top_counts(km_prefixes, 40)}")
                # A few representative keys for expected UNet LoRA formats
                samples = []
                for pref in ("lora_unet_down_blocks", "lora_unet_up_blocks", "lora_unet_mid_block", "unet.down_blocks", "diffusion_model.down_blocks"):
                    for k in km_keys:
                        if isinstance(k, str) and k.startswith(pref):
                            samples.append(k)
                            if len(samples) >= 20:
                                break
                    if len(samples) >= 20:
                        break
                _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] key_map sample keys={samples}")
            except Exception as e:
                _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] key_map sample dump failed: {e}")

        # ---- LoRA type detection (QI/ZIT並みに表示) ----
        try:
            lora_type = _detect_lora_type(lora_sd)
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] LoRA type detection: tags={lora_type['tags']}")
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] LoRA type detection: flags={lora_type['flags']}")
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] LoRA type detection: samples_head={lora_type['samples_head']}")
        except Exception as e:
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] LoRA type detection failed: {e}")

        # ---- A1111 SDXL UNet LoRA key conversion (input_blocks->down_blocks) ----
        try:
            lora_sd_conv, conv_stats = _convert_a1111_unet_lora_keys_to_comfy_diffusers(lora_sd, model, debug)
        except Exception as e:
            lora_sd_conv = lora_sd
            conv_stats = {"converted": False, "reason": str(e)}
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] conversion failed: {e}")

        # ---- LoRA keys summary + hit test (AFTER conversion) ----
        try:
            lora_conv = comfy.lora_convert.convert_lora(lora_sd_conv)
            lora_keys = list(lora_conv.keys())
            lora_keys_sorted = sorted(lora_keys)

            head = lora_keys_sorted[:80]
            tail = lora_keys_sorted[-20:] if len(lora_keys_sorted) > 100 else []
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] lora converted keys: num={len(lora_keys)} head={head}")
            if tail:
                _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] lora converted keys: tail={tail}")

            # quick format flags
            flags = {
                "has_lora_unet_": any(k.startswith("lora_unet_") for k in lora_keys),
                "has_lora_te_": any(k.startswith("lora_te") for k in lora_keys),
                "has_unet.": any(k.startswith("unet.") for k in lora_keys),
                "has_diffusion_model.": any(k.startswith("diffusion_model.") for k in lora_keys),
                "has_text_encoder.": any(k.startswith("text_encoder.") for k in lora_keys),
                "has_text_encoders.": any(k.startswith("text_encoders.") for k in lora_keys),
            }
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] lora key format flags: {flags}")

            # base key hit test (what comfy.lora.load_lora() uses)
            bases = [_lora_base_key(k) for k in lora_keys]
            bases_set = set(bases)
            allowed_all = set(key_map_all.keys())
            allowed_unet = set(key_map_unet.keys())
            allowed_clip = set(key_map_clip.keys())

            hit_bases = sorted(list(bases_set.intersection(allowed_all)))
            miss_bases = sorted(list(bases_set.difference(allowed_all)))

            hit_unet = sorted(list(bases_set.intersection(allowed_unet)))
            hit_clip = sorted(list(bases_set.intersection(allowed_clip)))

            _dbg_print(
                debug,
                "[NUNCHAKU_SDXL_LORA_DEBUG] base-key hit test: "
                f"bases_total={len(bases_set)} hit_bases={len(hit_bases)} miss_bases={len(miss_bases)}",
            )
            _dbg_print(
                debug,
                "[NUNCHAKU_SDXL_LORA_DEBUG] hit split: "
                f"hit_unet={len(hit_unet)} hit_clip={len(hit_clip)} (note: overlap possible)",
            )
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] hit_bases sample={hit_bases[:40]}")
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] hit_unet sample={hit_unet[:40]}")
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] hit_clip sample={hit_clip[:40]}")
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] miss_bases sample={miss_bases[:40]}")

            # prefix histograms to pinpoint mismatched naming convention quickly
            miss_prefixes = [_prefix_bucket(b, 4) for b in miss_bases[:2000]]
            hit_prefixes = [_prefix_bucket(b, 4) for b in hit_bases[:2000]]
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] miss prefix top={_top_counts(miss_prefixes, 40)}")
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] hit  prefix top={_top_counts(hit_prefixes, 40)}")

            # Debug diagnostic: check if representative miss_bases correspond to any existing *.weight
            # in the UNet state_dict. This helps confirm "quantized layers have no patchable weight".
            try:
                sd_keys = set()
                if model is not None and hasattr(model, "model") and hasattr(model.model, "state_dict"):
                    sd_keys = set(model.model.state_dict().keys())

                for b in miss_bases[:12]:
                    cand = _lora_unet_base_to_possible_state_dict_weights(b)
                    present = [c for c in cand if c in sd_keys]
                    _dbg_print(
                        debug,
                        f"[NUNCHAKU_SDXL_LORA_DEBUG] miss->state_dict weight probe: base='{b}' "
                        f"candidates={cand} present={present}",
                    )
            except Exception as e:
                _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] miss->state_dict weight probe failed: {e}")

            # very small "候補提示" for a few misses
            if miss_bases and key_map_all:
                km_keys = list(key_map_all.keys())
                show_n = 8
                for b in miss_bases[:show_n]:
                    cand = _best_suffix_candidates(b, km_keys, topn=5)
                    _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] suggest for miss_base='{b}' -> {cand}")

            # sanity: if unet key_map is empty, UNetには絶対当たらない
            if len(key_map_unet) == 0:
                _dbg_print(
                    debug,
                    "[NUNCHAKU_SDXL_LORA_DEBUG] !!! WARNING: unet key_map is EMPTY. "
                    "This means ComfyUI cannot patch UNet weights via standard LoRA mapping. "
                    "LoRAs may still apply to CLIP if clip key_map is non-empty.",
                )
        except Exception as e:
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] lora convert/hit-test failed: {e}")

    # これが重要：未ロードキーは comfy/lora.py が大量に warning を出す（QI/ZITと同じ検証圧）
    # ただし SDXL A1111形式はキー変換が必要なので、適用側も変換後sdを渡す。
    lora_sd_to_apply = lora_sd
    try:
        lora_sd_to_apply, _ = _convert_a1111_unet_lora_keys_to_comfy_diffusers(lora_sd, model, debug)
    except Exception:
        lora_sd_to_apply = lora_sd

    # Nunchaku SDXL: avoid ComfyUI SDXL remap overwrite (down_blocks -> input_blocks) that breaks patch targets.
    if _is_nunchaku_sdxl_model_patcher(model):
        _dbg_print(debug, "[NUNCHAKU_SDXL_LORA_DEBUG] using state_dict-only key_map (avoid SDXL input_blocks overwrite)")
        return _load_lora_for_models_state_dict_only(model, clip, lora_sd_to_apply, strength_model, strength_clip, debug)

    return comfy.sd.load_lora_for_models(model, clip, lora_sd_to_apply, strength_model, strength_clip)


def _apply_lora_sdxl_nunchaku_safe(model, clip, lora_sd, strength_model: float, strength_clip: float, debug: bool):
    """
    Apply LoRA for SDXL with Nunchaku-specific safety/compat:
    - Always runs A1111 UNet key conversion + attn1 qkv fusion when applicable
    - For Nunchaku SDXL ModelPatcher, ALWAYS uses state_dict-only UNet key_map and applies SVDQ runtime LoRA
    - Avoids noisy per-key warnings unless debug=True
    """
    # Normalize common PEFT key styles early (doesn't change tensors, only key names).
    lora_sd_to_apply, _ = _normalize_peft_lora_ab_to_comfy(lora_sd, debug)
    try:
        lora_sd_to_apply, _ = _convert_a1111_unet_lora_keys_to_comfy_diffusers(lora_sd_to_apply, model, debug)
    except Exception:
        # Keep normalized dict if conversion fails.
        lora_sd_to_apply = lora_sd_to_apply

    if _is_nunchaku_sdxl_model_patcher(model):
        _dbg_print(debug, "[NUNCHAKU_SDXL_LORA_DEBUG] applying via state_dict-only key_map (Nunchaku SDXL)")
        return _load_lora_for_models_state_dict_only(model, clip, lora_sd_to_apply, strength_model, strength_clip, debug)

    # Fallback to standard ComfyUI behavior for non-Nunchaku models.
    if debug:
        _dbg_print(debug, "[NUNCHAKU_SDXL_LORA_DEBUG] applying via comfy.sd.load_lora_for_models (non-Nunchaku model)")
    return comfy.sd.load_lora_for_models(model, clip, lora_sd_to_apply, strength_model, strength_clip)


class NunchakuUssoewwinSDXLLoraStackV3:
    """
    Node for loading and applying multiple LoRAs to an SDXL model + SDXL CLIP with dynamic UI.

    Notes:
    - This node is intentionally implemented WITHOUT any absolute paths.
    - LoRA application uses ComfyUI's built-in LoRA logic (comfy.sd.load_lora_for_models),
      so it works with normal SDXL LoRAs that target UNet and/or CLIP.
    """
    @classmethod
    def IS_CHANGED(cls, model, clip, lora_count, toggle_all=True, **kwargs):
        """
        Detect changes to trigger node re-execution.
        Returns a hash of relevant parameters to detect changes.
        """
        import hashlib
        m = hashlib.sha256()
        m.update(str(model).encode())
        m.update(str(clip).encode())
        m.update(str(lora_count).encode())
        m.update(str(toggle_all).encode())
        # Hash all LoRA parameters
        for i in range(1, 11):
            m.update(kwargs.get(f"lora_name_{i}", "").encode())
            m.update(str(kwargs.get(f"lora_strength_{i}", 0)).encode())
            m.update(str(kwargs.get(f"enabled_{i}", True)).encode())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def INPUT_TYPES(s):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        
        inputs = {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "The diffusion model to apply LoRAs to."},
                ),
                "clip": (
                    "CLIP",
                    {"tooltip": "The SDXL CLIP (Dual CLIP). LoRAs targeting CLIP will be applied here."},
                ),
                "lora_count": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Number of LoRA slots to process.",
                    },
                ),
                "toggle_all": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Enable/disable all LoRAs at once.",
                    },
                ),
                "debug": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Print debug logs (similar information volume to ComfyUI-QwenImageLoraLoader).",
                    },
                ),
            },
            "optional": {},
        }

        # Add all LoRA inputs (up to 10 slots) as optional
        for i in range(1, 11):
            inputs["optional"][f"enabled_{i}"] = (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": f"Enable/disable LoRA {i}.",
                },
            )
            inputs["optional"][f"lora_name_{i}"] = (
                loras,
                {"tooltip": f"The file name of LoRA {i}. Select 'None' to skip this slot."},
            )
            inputs["optional"][f"lora_strength_{i}"] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": f"Strength for LoRA {i}.",
                },
            )

        return inputs

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model with all LoRAs applied.", "The modified CLIP with LoRAs applied (if applicable).")
    FUNCTION = "load_lora_stack"
    TITLE = "Nunchaku-ussoewwin SDXL LoRA Stack V3"
    CATEGORY = "Nunchaku-ussoewwin"
    DESCRIPTION = "Apply multiple LoRAs to an SDXL model + CLIP in a single node with dynamic UI control."

    def load_lora_stack(self, model, clip, lora_count, toggle_all=True, debug=False, **kwargs):
        loras_to_apply = []
        
        # Log toggle_all state
        logger.info(f"[LoRA Stack Status] toggle_all: {toggle_all}")
        logger.info(f"[LoRA Stack Status] Processing {lora_count} LoRA slot(s):")
        
        # Process only the number of LoRAs specified by lora_count
        for i in range(1, lora_count + 1):
            lora_name = kwargs.get(f"lora_name_{i}")
            lora_strength = kwargs.get(f"lora_strength_{i}", 1.0)
            enabled_individual = kwargs.get(f"enabled_{i}", True)
            # Check if this LoRA is enabled (considering both toggle_all and individual enabled_<i>)
            enabled = toggle_all and enabled_individual
            
            # Log each LoRA slot status
            status_parts = []
            status_parts.append(f"Slot {i}:")
            if lora_name and lora_name != "None":
                status_parts.append(f"'{lora_name}'")
                status_parts.append(f"strength={lora_strength}")
            else:
                status_parts.append("(no LoRA selected)")
            
            status_parts.append(f"toggle_all={toggle_all}")
            status_parts.append(f"enabled_{i}={enabled_individual}")
            status_parts.append(f"final_enabled={enabled}")
            
            if enabled and lora_name and lora_name != "None" and abs(lora_strength) > 1e-5:
                status_parts.append("→ APPLIED ✓")
                loras_to_apply.append((lora_name, lora_strength))
            else:
                status_parts.append("→ SKIPPED ✗")
            
            logger.info(f"[LoRA Stack Status] {' | '.join(status_parts)}")
        
        # Log summary
        logger.info(f"[LoRA Stack Status] Summary: {len(loras_to_apply)} LoRA(s) will be applied out of {lora_count} slot(s)")

        if not loras_to_apply:
            return (model, clip)

        out_model = model
        out_clip = clip

        # QI/ZIT style preface (debug only): announce composition count up-front.
        _dbg_print(bool(debug), f"Composing {len(loras_to_apply)} LoRAs...")

        # Apply sequentially (each LoRA is independently loaded + inspected).
        # NOTE: We intentionally do NOT suppress any ComfyUI "lora key not loaded" logs.
        for lora_name, strength in loras_to_apply:
            lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
            lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
            _dbg_print(debug, f"[NUNCHAKU_SDXL_LORA_DEBUG] applying: name='{lora_name}' strength={strength} path='{lora_path}'")

            # QI/ZIT requested behavior: LoRA "format/type" detection (pre-convert, pre-normalize)
            # This is a heuristic summary of key styles and adapter variants present in the file.
            try:
                det = _detect_lora_type(lora_sd if isinstance(lora_sd, dict) else {})
                _dbg_print(bool(debug), f"--- DEBUG: Inspecting keys for LoRA '{lora_name}' (Strength: {float(strength)}) ---")
                _dbg_print(
                    bool(debug),
                    "[NUNCHAKU_SDXL_LORA_DEBUG] lora type detection: "
                    f"num_keys={det.get('num_keys')} tags={det.get('tags')} flags={det.get('flags')}",
                )
                _dbg_print(bool(debug), f"[NUNCHAKU_SDXL_LORA_DEBUG] lora key samples (head): {det.get('samples_head')}")
            except Exception as e:
                _dbg_print(bool(debug), f"[NUNCHAKU_SDXL_LORA_DEBUG] lora type detection failed: {e}")

            # Always use the Nunchaku-safe SDXL apply path so A1111 key conversion + SVDQ runtime work
            # even when debug is OFF (normal usage).
            out_model, out_clip = _apply_lora_sdxl_nunchaku_safe(out_model, out_clip, lora_sd, strength, strength, debug)

        return (out_model, out_clip)

GENERATED_NODES = {
    "NunchakuUssoewwinSDXLLoraStackV3": NunchakuUssoewwinSDXLLoraStackV3
}

GENERATED_DISPLAY_NAMES = {
    "NunchakuUssoewwinSDXLLoraStackV3": "Nunchaku-ussoewwin SDXL LoRA Stack V3"
}
