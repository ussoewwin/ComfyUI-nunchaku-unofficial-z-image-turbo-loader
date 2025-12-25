"""
This module provides a wrapper for ComfyUI's Z-Image model configuration.
"""

import torch
from comfy.supported_models import ZImage

from .. import model_base


class NunchakuZImage(ZImage):
    """
    Wrapper for the Nunchaku Z-Image-Turbo model configuration.
    """

    def get_model(
        self, state_dict: dict[str, torch.Tensor], prefix: str = "", device=None, **kwargs
    ) -> model_base.NunchakuZImage:
        """
        Instantiate and return a NunchakuZImage model.

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
        model_base.NunchakuZImage
            Instantiated NunchakuZImage model.
        """
        out = model_base.NunchakuZImage(self, device=device, **kwargs)
        return out

