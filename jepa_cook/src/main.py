import argparse
import ast
import json
import math
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from jepa_cook.src.config import load_config  # deptry: ignore
from jepa_cook.src.dataset import JEPACollateFn, PreTokenizedActionDataset, pad_nested_sequences  # deptry: ignore
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

    model = RecipeJEPA(
        vocab_size=model_cfg["vocab_size"],
        embed_dim=model_cfg["embed_dim"],
        latent_dim=model_cfg["latent_dim"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
    ).to(device)

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


def handle_inference(args: argparse.Namespace, config: dict[str, Any], device: torch.device) -> None:
    """Runs context vectors down distance spaces to score structural predictions.

    Args:
        args: Commandline properties mapping query assets.
        config: Central configuration values.
        device: Target hardware resource pointer.
    """
    model_cfg = config["model"]
    infer_cfg = config["inference"]

    try:
        targets_list: list[str] = ast.literal_eval(args.targets)
    except Exception:
        print("[!] Format error parsing target list strings.")
        return

    tokenizer = AutoTokenizer.from_pretrained(infer_cfg["tokenizer_name"])
    model = RecipeJEPA(
        vocab_size=model_cfg["vocab_size"],
        embed_dim=model_cfg["embed_dim"],
        latent_dim=model_cfg["latent_dim"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint if "state_dict" not in checkpoint else checkpoint["state_dict"])
    model.to(device).eval()

    with torch.no_grad():

        def tokenize_to_3d_input(text_input: str) -> torch.Tensor:
            try:
                lst = json.loads(text_input) if "[" in text_input else [text_input]
            except Exception:
                lst = [text_input]
            tokens_list = [torch.tensor(tokenizer(item, add_special_tokens=False)["input_ids"]) for item in lst]
            return pad_nested_sequences([tokens_list], max_len=model_cfg["max_len"]).to(device)

        x_tokens = tokenize_to_3d_input(args.ingredients)
        a_tokens = tokenize_to_3d_input(args.action)
        pred_embed, _, _, _ = model(x_tokens, a_tokens)
        pred_embed_norm = nn.functional.normalize(pred_embed, p=2, dim=-1)

        results = []
        for target_str in targets_list:
            target_tokens = tokenizer(target_str, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            target_embed = model.encode_target(target_tokens)
            target_embed_norm = nn.functional.normalize(target_embed, p=2, dim=-1)

            normalized_mse = torch.mean((pred_embed_norm - target_embed_norm) ** 2).item()
            results.append((target_str, normalized_mse))

    results.sort(key=lambda x: x[1])

    print("\n" + "=" * 60)
    print(" EVALUATION WITH L2 UNIT-NORMALIZATION")
    print("=" * 60)
    for target_str, score in results:
        print(f"{target_str:<30} | Normalized MSE: {score:.6f}")
    print("=" * 60)


def main() -> None:
    """Coordinates multi-command runtime configuration parsing profiles."""
    parser = argparse.ArgumentParser(description="Recipe JEPA Unified CLI Refactored Workflow")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Workflow Mode")

    # Train sub-command interface
    subparsers.add_parser("train", help="Run model training loop")

    # Inference sub-command interface
    infer_parser = subparsers.add_parser("inference", help="Run prediction inference")
    infer_parser.add_argument("--checkpoint", type=str, required=True)
    infer_parser.add_argument("--ingredients", type=str, required=True)
    infer_parser.add_argument("--action", type=str, required=True)
    infer_parser.add_argument("--targets", type=str, required=True)

    args = parser.parse_args()
    config = load_config(args.config)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using runtime device: {device}")

    if args.command == "train":
        handle_train(config, device)
    elif args.command == "inference":
        handle_inference(args, config, device)


if __name__ == "__main__":
    main()
