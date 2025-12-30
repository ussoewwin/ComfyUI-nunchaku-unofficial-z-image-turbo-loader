"""
Nunchaku SDXL model base.

This module provides a wrapper for ComfyUI's SDXL model base.
"""

import torch
import logging
import os
import comfy.model_management
import comfy.utils
import comfy.conds
from comfy.model_base import ModelType, SDXL

from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel

logger = logging.getLogger(__name__)

_SDXL_DEBUG_CONTEXT = False

def set_nunchaku_sdxl_debug(enabled: bool) -> None:
    """
    Enable/disable debug prints for NunchakuSDXL wrapper at runtime.
    This is intended to be toggled from the loader node UI (BOOLEAN input),
    so users don't need environment variables.
    """
    global _SDXL_DEBUG_CONTEXT
    _SDXL_DEBUG_CONTEXT = bool(enabled)

def _tinfo(x):
    if x is None:
        return "None"
    if torch.is_tensor(x):
        return f"shape={tuple(x.shape)} dtype={x.dtype} device={x.device}"
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}(len={len(x)})"
    if isinstance(x, dict):
        return f"dict(keys={list(x.keys())})"
    return f"{type(x).__name__}"

def _list_tensor_shapes(xs, limit: int = 6):
    if not isinstance(xs, list):
        return "None"
    shapes = []
    for v in xs:
        if v is None:
            shapes.append("None")
        elif torch.is_tensor(v):
            shapes.append(str(tuple(v.shape)))
        else:
            shapes.append(type(v).__name__)
        if len(shapes) >= limit:
            break
    more = "" if len(xs) <= limit else f", ...(+{len(xs)-limit})"
    return "[" + ", ".join(shapes) + more + "]"

def _dbg_enabled() -> bool:
    if _SDXL_DEBUG_CONTEXT:
        return True
    return os.getenv("NUNCHAKU_SDXL_DEBUG", "0") == "1"

def _dbg_print(msg: str) -> None:
    # ComfyUI environments often don't show logging.INFO from custom modules reliably.
    # Use print for debug gated by env var.
    if _dbg_enabled():
        print(msg)


class NunchakuSDXL(SDXL):
    """
    Wrapper for the Nunchaku SDXL model.

    SDXL uses UNet2DConditionModel, so we inherit from SDXL and replace
    the UNet model with NunchakuSDXLUNet2DConditionModel.
    
    This class overrides _apply_model to convert ComfyUI's calling convention
    (context parameter) to diffusers' convention (encoder_hidden_states and added_cond_kwargs).

    Parameters
    ----------
    model_config : object
        Model configuration object.
    model_type : ModelType, optional
        Type of the model (default is ModelType.EPS).
    device : torch.device or str, optional
        Device to load the model onto.
    """

    def __init__(self, model_config, model_type=ModelType.EPS, device=None, **kwargs):
        """
        Initialize the NunchakuSDXL model.

        Parameters
        ----------
        model_config : object
            Model configuration object.
        model_type : ModelType, optional
            Type of the model (default is ModelType.EPS).
        device : torch.device or str, optional
            Device to load the model onto.
        **kwargs
            Additional keyword arguments.
        """
        # SDXL uses NunchakuSDXLUNet2DConditionModel
        unet_model = kwargs.get("unet_model", NunchakuSDXLUNet2DConditionModel)

        # Remove ComfyUI-specific keys from unet_config that UNet2DConditionModel doesn't accept
        # The UNet is already built in load_diffusion_model_state_dict, so we disable creation
        unet_config = model_config.unet_config.copy()
        unet_config["disable_unet_model_creation"] = True

        # Temporarily set unet_config to avoid passing invalid keys
        original_unet_config = model_config.unet_config
        model_config.unet_config = unet_config

        try:
            # IMPORTANT:
            # We must run SDXL.__init__ so attributes like noise_augmentor are created.
            # Using super(SDXL, self).__init__ skips SDXL and calls BaseModel directly, leaving SDXL fields unset.
            super().__init__(model_config, model_type, device=device)
        finally:
            # Restore original unet_config
            model_config.unet_config = original_unet_config

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        """
        Apply the diffusion model with ComfyUI's calling convention.
        
        This method converts ComfyUI's parameters to diffusers' UNet2DConditionModel format:
        - context (c_crossattn) -> encoder_hidden_states
        - adm (additional model inputs) -> added_cond_kwargs with time_ids and text_embeds
        
        Parameters
        ----------
        x : torch.Tensor
            Input latent tensor.
        t : torch.Tensor
            Timestep tensor.
        c_concat : torch.Tensor, optional
            Concatenated conditioning (for inpainting).
        c_crossattn : torch.Tensor, optional
            Cross-attention context (text embeddings).
        control : dict, optional
            ControlNet outputs.
        transformer_options : dict, optional
            Additional transformer options.
        **kwargs
            Additional keyword arguments.
            
        Returns
        -------
        torch.Tensor
            Model output tensor.
        """
        # Get base model output (handles sigma calculation, device casting, etc.)
        sigma = t
        xc = self.model_sampling.calculate_input(sigma, x)

        if c_concat is not None:
            xc = torch.cat([xc] + [comfy.model_management.cast_to_device(c_concat, xc.device, xc.dtype)], dim=1)

        context = c_crossattn
        dtype = self.get_dtype()

        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype

        xc = xc.to(dtype)
        device = xc.device
        t = self.model_sampling.timestep(t).float()
        if context is not None:
            context = comfy.model_management.cast_to_device(context, device, dtype)

        # Handle additional conditions.
        # NOTE:
        # In ComfyUI, SDXL's pooled conditioning is computed in extra_conds() (before sampling)
        # and passed into apply_model as keyword tensors (e.g. "y").
        # Our previous implementation incorrectly called encode_adm() again here, but at this
        # stage "pooled_output" is no longer present, causing KeyError.
        extra_conds = {}
        for o in kwargs:
            extra = kwargs[o]

            if hasattr(extra, "dtype"):
                extra = comfy.model_management.cast_to_device(extra, device, dtype)
            elif isinstance(extra, list):
                ex = []
                for ext in extra:
                    ex.append(comfy.model_management.cast_to_device(ext, device, dtype))
                extra = ex
            extra_conds[o] = extra

        t = self.process_timestep(t, x=x, **extra_conds)
        if "latent_shapes" in extra_conds:
            xc = comfy.utils.unpack_latents(xc, extra_conds.pop("latent_shapes"))

        # Convert to diffusers format:
        # NunchakuSDXLUNet2DConditionModel expects added_cond_kwargs with raw SDXL time_ids (B, 6)
        # and pooled text embeds (B, 1280).
        text_embeds = extra_conds.get("text_embeds", None)
        time_ids = extra_conds.get("time_ids", None)
        if text_embeds is not None and time_ids is not None:
            added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        else:
            added_cond_kwargs = None

        if _dbg_enabled():
            cnt = getattr(self, "_nunchaku_sdxl_dbg_apply_count", 0) + 1
            self._nunchaku_sdxl_dbg_apply_count = cnt
            if cnt <= 3 or cnt % 50 == 0:
                _dbg_print(
                    "[NunchakuSDXL/_apply_model] "
                    f"xc={_tinfo(xc)} t={_tinfo(t)} context={_tinfo(context)} "
                    f"added_cond_kwargs={'present' if added_cond_kwargs else 'None'} "
                    f"text_embeds={_tinfo(text_embeds)} time_ids={_tinfo(time_ids)} "
                    f"keys(extra_conds)={sorted(list(extra_conds.keys()))} control={_tinfo(control)}"
                )

        # ControlNet support:
        # ComfyUI ControlNet returns a dict with keys: {"input": [...], "middle": [...], "output": [...]}
        # (see comfy/controlnet.py ControlBase.control_merge).
        #
        # Diffusers UNet2DConditionModel expects ControlNet residuals as:
        # - down_block_additional_residuals: tuple[Tensor]
        # - mid_block_additional_residual: Tensor
        #
        # We map (ComfyUI ControlNet output for SDXL CLDM):
        # - control["output"] (filtered non-None) -> down_block_additional_residuals
        # - first non-None from control["middle"] -> mid_block_additional_residual
        down_block_additional_residuals = None
        mid_block_additional_residual = None
        if isinstance(control, dict):
            # For comfy/cldm/cldm.py SDXL ControlNet, outputs are under "output" and "middle"
            # (see comfy/cldm/cldm.py forward: returns {"middle": out_middle, "output": out_output}).
            inp = control.get("output", None)
            mid = control.get("middle", None)
            if isinstance(inp, list):
                down_list = [v for v in inp if v is not None]
                if len(down_list) > 0:
                    down_block_additional_residuals = tuple(
                        comfy.model_management.cast_to_device(v, device, dtype) for v in down_list
                    )
            if isinstance(mid, list):
                for v in mid:
                    if v is not None:
                        mid_block_additional_residual = comfy.model_management.cast_to_device(v, device, dtype)
                        break

        if _dbg_enabled() and isinstance(control, dict):
            cnt = getattr(self, "_nunchaku_sdxl_dbg_ctrl_count", 0) + 1
            self._nunchaku_sdxl_dbg_ctrl_count = cnt
            if cnt <= 3 or cnt % 50 == 0:
                _dbg_print(
                    "[NunchakuSDXL/_apply_model] "
                    f"control keys={list(control.keys())} "
                    f"input={_list_tensor_shapes(control.get('input', None))} "
                    f"middle={_list_tensor_shapes(control.get('middle', None))} "
                    f"output={_list_tensor_shapes(control.get('output', None))} "
                    f"-> down_res={'len='+str(len(down_block_additional_residuals)) if down_block_additional_residuals else 'None'} "
                    f"mid_res={_tinfo(mid_block_additional_residual)}"
                )

        # Call diffusers UNet format
        # NunchakuSDXLUNet2DConditionModel is a diffusers UNet2DConditionModel
        if isinstance(self.diffusion_model, NunchakuSDXLUNet2DConditionModel):
            # Convert ComfyUI format to diffusers format
            model_output = self.diffusion_model(
                sample=xc,
                timestep=t,
                encoder_hidden_states=context,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_block_additional_residuals,
                mid_block_additional_residual=mid_block_additional_residual,
                return_dict=False,
            )
            # diffusers returns a tuple (sample,), ComfyUI expects tensor
            if isinstance(model_output, tuple):
                model_output = model_output[0]
        else:
            # Fallback to base class implementation for non-Nunchaku models
            model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)
        
        if len(model_output) > 1 and not torch.is_tensor(model_output):
            model_output, _ = comfy.utils.pack_latents(model_output)

        return self.model_sampling.calculate_denoised(sigma, model_output.float(), x)

    def extra_conds(self, **kwargs):
        """
        Build conditioning tensors for the sampler.

        ComfyUI's default SDXL path creates a single 'y' ADM tensor that includes embedded time information.
        Diffusers SDXL expects *raw* time_ids (B, 6) and text_embeds (B, 1280) in added_cond_kwargs.

        So for the Nunchaku SDXL (diffusers UNet) backend we generate:
        - c_crossattn: standard cross-attn tensor from clip
        - text_embeds: pooled_output from SDXL clip (CLIP-G pooled)
        - time_ids: raw 6-value SDXL time ids (height, width, crop_h, crop_w, target_h, target_w)
        - y: ComfyUI SDXL ADM tensor (required by many SDXL ControlNet implementations)
        """
        out = {}

        # inpaint concat support (keep same as BaseModel)
        concat_cond = self.concat_cond(**kwargs)
        if concat_cond is not None:
            out["c_concat"] = comfy.conds.CONDNoiseShape(concat_cond)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out["c_crossattn"] = comfy.conds.CONDCrossAttn(cross_attn)

        pooled_output = kwargs.get("pooled_output", None)
        if pooled_output is None:
            raise ValueError(
                "SDXL requires pooled_output (pooled text embedding) but it was not provided. "
                "Use a proper SDXL CLIP (recommended: DualCLIPLoader type=sdxl, or a standard SDXL checkpoint CLIP) "
                "so conditioning includes pooled_output."
            )
        out["text_embeds"] = comfy.conds.CONDRegular(pooled_output)

        # Also provide ComfyUI-style SDXL ADM ("y") for ControlNet compatibility.
        # Some ControlNet SDXL models error out when y is missing.
        adm = self.encode_adm(**kwargs)
        if adm is not None:
            out["y"] = comfy.conds.CONDRegular(adm)

        # Raw SDXL time ids (6 values)
        width = kwargs.get("width", 768)
        height = kwargs.get("height", 768)
        crop_w = kwargs.get("crop_w", 0)
        crop_h = kwargs.get("crop_h", 0)
        target_width = kwargs.get("target_width", width)
        target_height = kwargs.get("target_height", height)

        device = kwargs.get("device", pooled_output.device)
        # diffusers expects float time ids (it applies add_time_proj internally)
        base = torch.tensor(
            [height, width, crop_h, crop_w, target_height, target_width],
            device=device,
            dtype=torch.float32,
        )
        time_ids = base.unsqueeze(0).repeat(pooled_output.shape[0], 1)
        out["time_ids"] = comfy.conds.CONDRegular(time_ids)

        if _dbg_enabled():
            cnt = getattr(self, "_nunchaku_sdxl_dbg_extra_count", 0) + 1
            self._nunchaku_sdxl_dbg_extra_count = cnt
            if cnt <= 3:
                _dbg_print(
                    "[NunchakuSDXL/extra_conds] "
                    f"pooled_output={_tinfo(pooled_output)} adm(y)={_tinfo(adm)} time_ids={_tinfo(time_ids)} cross_attn={_tinfo(cross_attn)} "
                    f"-> keys(out)={sorted(list(out.keys()))}"
                )

        return out

    def load_model_weights(self, sd: dict[str, torch.Tensor], unet_prefix: str = ""):
        """
        Load model weights into the diffusion model.

        Parameters
        ----------
        sd : dict of str to torch.Tensor
            State dictionary containing model weights.
        unet_prefix : str, optional
            Prefix for UNet weights (default is "").

        Raises
        ------
        ValueError
            If a required key is missing from the state dictionary.
        """
        diffusion_model = self.diffusion_model
        if isinstance(diffusion_model, NunchakuSDXLUNet2DConditionModel):
            # NunchakuSDXLUNet2DConditionModel handles its own loading
            diffusion_model.load_state_dict(sd, strict=False)
        else:
            state_dict = diffusion_model.state_dict()
            for k in state_dict.keys():
                if k not in sd:
                    raise ValueError(f"Key {k} not found in state_dict")
            diffusion_model.load_state_dict(sd, strict=True)

