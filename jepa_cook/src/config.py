from typing import Any

import yaml
from transformers import PretrainedConfig


def load_config(config_path: str) -> dict[str, Any]:
    """Loads a YAML configuration file.

    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        A dictionary containing configuration items.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


class RecipeJEPAConfig(PretrainedConfig):
    model_type = "recipe_jepa"

    def __init__(
        self,
        vocab_size: int = 30522,
        embed_dim: int = 384,
        latent_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        ema_momentum: float = 0.999,
        **kwargs,
    ) -> None:
        """Config class passing architectural settings into PreTrainedModel pipelines."""
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.ema_momentum = ema_momentum
