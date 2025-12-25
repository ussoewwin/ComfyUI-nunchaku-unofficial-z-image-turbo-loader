"""
Nunchaku Z-Image-Turbo model base.

This module provides a wrapper for ComfyUI's Z-Image model base.
Z-Image-Turbo is based on Lumina2 architecture.
"""

import torch
import comfy.conds
import comfy.model_management
import comfy.utils
from comfy.model_base import ModelType, Lumina2

from nunchaku.models.transformers.transformer_zimage import NunchakuZImageTransformer2DModel


class NunchakuZImage(Lumina2):
    """
    Wrapper for the Nunchaku Z-Image-Turbo model.

    Z-Image-Turbo is based on Lumina2, so we inherit from Lumina2 and replace
    the transformer model with NunchakuZImageTransformer2DModel.

    Parameters
    ----------
    model_config : object
        Model configuration object.
    model_type : ModelType, optional
        Type of the model (default is ModelType.FLOW, same as Lumina2).
    device : torch.device or str, optional
        Device to load the model onto.
    """

    def __init__(self, model_config, model_type=ModelType.FLOW, device=None, **kwargs):
        """
        Initialize the NunchakuZImage model.

        Parameters
        ----------
        model_config : object
            Model configuration object.
        model_type : ModelType, optional
            Type of the model (default is ModelType.FLOW, same as Lumina2).
        device : torch.device or str, optional
            Device to load the model onto.
        **kwargs
            Additional keyword arguments.
        """
        # Z-Image-Turbo uses NunchakuZImageTransformer2DModel instead of NextDiT
        unet_model = kwargs.get("unet_model", NunchakuZImageTransformer2DModel)

        # Remove ComfyUI-specific keys from unet_config that ZImageTransformer2DModel doesn't accept
        # The transformer is already built in load_diffusion_model_state_dict, so we disable creation
        unet_config = model_config.unet_config.copy()
        unet_config["disable_unet_model_creation"] = True

        # Temporarily set unet_config to avoid passing invalid keys
        original_unet_config = model_config.unet_config
        model_config.unet_config = unet_config

        try:
            super(Lumina2, self).__init__(model_config, model_type, device=device, unet_model=unet_model)
        finally:
            # Restore original unet_config
            model_config.unet_config = original_unet_config

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        """
        Apply the Z-Image-Turbo model with the correct forward signature.

        Z-Image-Turbo expects:
        - x: List[torch.Tensor] (not a single tensor)
        - cap_feats: List[torch.Tensor] (not context)
        """
        sigma = t
        xc = self.model_sampling.calculate_input(sigma, x)

        if c_concat is not None:
            xc = torch.cat([xc] + [comfy.model_management.cast_to_device(c_concat, xc.device, xc.dtype)], dim=1)

        cap_feats = c_crossattn
        dtype = self.get_dtype()

        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype

        xc = xc.to(dtype)
        device = xc.device

        # ComfyUI (FLOW) passes `t` as sigma in range [1 -> 0] across steps.
        # Lumina2/NextDiT in ComfyUI internally uses: `t = 1.0 - timesteps` before embedding.
        # diffusers Z-Image pipeline also inverts time via: timestep = (1000 - t) / 1000.
        # So for Z-Image transformer we must feed an increasing normalized time in [0 -> 1].
        t_normalized = self.model_sampling.timestep(t).float()  # [1 -> 0]
        t_zimage = 1.0 - t_normalized  # [0 -> 1]

        # Z-Image-Turbo expects List[torch.Tensor] for x and cap_feats
        if cap_feats is not None:
            cap_feats = comfy.model_management.cast_to_device(cap_feats, device, dtype)
            # Convert to list format expected by Z-Image-Turbo
            if not isinstance(cap_feats, list):
                # Split batch into list of tensors
                cap_feats = [cap_feats[i] for i in range(cap_feats.shape[0])]
        else:
            cap_feats = []

        # Convert xc to list format
        # Z-Image-Turbo expects (C, F, H, W) for each image in the list
        # ComfyUI provides (B, C, H, W), so we need to add F dimension (F=1 for images)
        if not isinstance(xc, list):
            xc_list = []
            for i in range(xc.shape[0]):
                img = xc[i]  # (C, H, W)
                # Add frame dimension: (C, H, W) -> (C, 1, H, W)
                img = img.unsqueeze(1)  # Add F dimension
                xc_list.append(img)
        else:
            xc_list = xc
            # Ensure each element has F dimension
            xc_list = [img if img.dim() == 4 else img.unsqueeze(1) for img in xc_list]

        # Filter kwargs to only include arguments that Z-Image-Turbo forward accepts
        # Z-Image-Turbo forward signature: forward(x, t, cap_feats, patch_size=2, f_patch_size=1, return_dict=True)
        zimage_kwargs = {}
        if "patch_size" in kwargs:
            zimage_kwargs["patch_size"] = kwargs["patch_size"]
        if "f_patch_size" in kwargs:
            zimage_kwargs["f_patch_size"] = kwargs["f_patch_size"]
        # Explicitly set return_dict=False to get tuple (x,) instead of Transformer2DModelOutput
        zimage_kwargs["return_dict"] = False

        # Call Z-Image-Turbo forward with correct signature
        # return_dict=False returns tuple (x,) where x is List[torch.Tensor]
        # Each tensor in the list has shape (C, F, H, W) - velocity prediction
        # Pass normalized timestep (the model will scale by t_scale internally)
        model_output = self.diffusion_model(xc_list, t_zimage, cap_feats=cap_feats, **zimage_kwargs)

        # Extract list from tuple: (x,) -> x
        if isinstance(model_output, tuple):
            model_output = model_output[0]

        # model_output is now List[torch.Tensor] with shape (C, F, H, W) for each item
        # Convert to single tensor (B, C, F, H, W) then remove F dimension -> (B, C, H, W)
        if isinstance(model_output, list):
            # Stack list of tensors: each tensor is (C, F, H, W)
            model_output = torch.stack(model_output, dim=0)  # (B, C, F, H, W)
            # Remove frame dimension F=1: (B, C, F, H, W) -> (B, C, H, W)
            if model_output.shape[2] == 1:
                model_output = model_output.squeeze(2)  # Remove F dimension

        # According to diffusers Z-Image pipeline, the output needs to be negated: noise_pred = -noise_pred
        model_output = -model_output.float()

        # For ComfyUI ModelType.FLOW, model_sampling is CONST and calculate_denoised does:
        #   denoised = model_input - model_output * sigma
        return self.model_sampling.calculate_denoised(sigma, model_output, x)

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
        if isinstance(diffusion_model, NunchakuZImageTransformer2DModel):
            # NunchakuZImageTransformer2DModel handles its own loading
            diffusion_model.load_state_dict(sd, strict=False)
        else:
            state_dict = diffusion_model.state_dict()
            for k in state_dict.keys():
                if k not in sd:
                    raise ValueError(f"Key {k} not found in state_dict")
            diffusion_model.load_state_dict(sd, strict=True)

