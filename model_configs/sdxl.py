"""
This module provides a wrapper for ComfyUI's SDXL model configuration.
"""

import torch
from comfy.supported_models import SDXL

from .. import model_base


class NunchakuSDXL(SDXL):
    """
    Wrapper for the Nunchaku SDXL model configuration.
    """

    def get_model(
        self, state_dict: dict[str, torch.Tensor], prefix: str = "", device=None, **kwargs
    ) -> model_base.NunchakuSDXL:
        """
        Instantiate and return a NunchakuSDXL model.

        Parameters
        ----------
        state_dict : dict
            Model state dictionary.
        prefix : str, optional
            Prefix for parameter names (default is "").
        device : torch.device or str, optional
            Device to load the model onto.
        **kwargs
            Additional keyword arguments for model initialization.

        Returns
        -------
        model_base.NunchakuSDXL
            Instantiated NunchakuSDXL model.
        """
        out = model_base.NunchakuSDXL(self, device=device, **kwargs)
        return out

