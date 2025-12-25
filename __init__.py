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

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
logger.info("=" * (80 + len(" ComfyUI-nunchaku Initialization ")))
