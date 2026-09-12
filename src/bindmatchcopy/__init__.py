"""BindMatchCopy: explicit checkpoint measurements and archived paper evidence.

See the README for separate analysis, checkpoint, demo and training workflows.
Optional Gated DeltaNet and Qwen dependencies are loaded only when required.
"""

__version__ = "0.3.0"

from .config import ModelConfig, build_model, hybrid_gdn_tf_config, pure_tf_config

__all__ = [
    "__version__",
    "ModelConfig",
    "build_model",
    "hybrid_gdn_tf_config",
    "pure_tf_config",
]
