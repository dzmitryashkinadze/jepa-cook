import math
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from jepa_cook.src.config import RecipeJEPAConfig  # deptry: ignore
from jepa_cook.src.dataset import JEPACollateFn, PreTokenizedActionDataset  # deptry: ignore
from jepa_cook.src.models import RecipeJEPA  # deptry: ignore
from jepa_cook.src.trainer import JEPATrainer  # deptry: ignore


def handle_train(config: dict[str, Any], device: torch.device) -> None:
    """Configures structural pipelines to launch optimization workflows.

    Args:
        config: Central configuration values.
        device: Target hardware resource pointer.
    """

    train_cfg = config["train"]
    model_cfg = config["model"]

    collate_fn = JEPACollateFn(max_len=model_cfg["max_len"])

    train_dataset = PreTokenizedActionDataset(train_cfg["train_dataset"])
    val_dataset = PreTokenizedActionDataset(train_cfg["val_dataset"])

    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg["batch_size"], shuffle=True, drop_last=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg["batch_size"], shuffle=False, drop_last=True, collate_fn=collate_fn
    )

    # NEW: Instantiate the Hugging Face configuration utility object first
    hf_config = RecipeJEPAConfig(
        vocab_size=model_cfg["vocab_size"],
        embed_dim=model_cfg["embed_dim"],
        latent_dim=model_cfg["latent_dim"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
    )

    # Pass configuration object structure into model initializers
    model = RecipeJEPA(config=hf_config).to(device)

    optimizer = AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    num_warmup_steps = 3 * len(train_loader)
    total_steps = train_cfg["epochs"] * len(train_loader)

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, total_steps - num_warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    trainer = JEPATrainer(model, train_loader, val_loader, optimizer, scheduler, device, config)
    trainer.train(epochs=train_cfg["epochs"])
