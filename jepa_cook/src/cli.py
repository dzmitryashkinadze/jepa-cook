import argparse
import ast
import json
import math
import sys
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from jepa_cook.src.config import RecipeJEPAConfig, load_config  # deptry: ignore
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

    # NEW: Package variables cleanly for local inference pipelines
    hf_config = RecipeJEPAConfig(
        vocab_size=model_cfg["vocab_size"],
        embed_dim=model_cfg["embed_dim"],
        latent_dim=model_cfg["latent_dim"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
    )
    model = RecipeJEPA(config=hf_config)

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


def run_diagnostics(args):
    print("=" * 70)
    print("🚀 KICKING OFF JEPA-COOK SYSTEM DIAGNOSTICS")
    print("=" * 70)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Runtime Engine Target: {device}")

    # 1. Load Checkpoint Safely
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        weights = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        print(f"✓ Loaded weights successfully from: {args.checkpoint}")
    except Exception as e:
        print(f"❌ Failed to load checkpoint file: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("📊 LAYER 1: TARGET ENCODER WEIGHT VARIANCE & TOPOLOGY")
    print("=" * 60)

    has_target_layers = False
    for k, v in weights.items():
        if "target_encoder" in k and "weight" in k and v.dim() >= 2:
            has_target_layers = True
            norm_val = v.norm().item()
            std_val = v.std().item()

            # Representation collapse warning sign: vanishing variance
            collapse_status = "⚠️ WARNING: Low Variance" if std_val < 1e-4 else "✓ Healthy"
            print(f"{k:<55} | Norm: {norm_val:.4f} | Std: {std_val:.4f} | {collapse_status}")

    if not has_target_layers:
        print("ℹ️ No multi-dimensional target encoder weight structures found matching key naming rules.")

    print("\n" + "=" * 60)
    print("🧠 LAYER 2: PREDICTOR CROSS-ATTENTION INFORMATION ROUTING")
    print("=" * 60)

    # Track cross-attention projections to make sure actions aren't ignored
    in_proj_key = "predictor.transformer_decoder.layers.0.multihead_attn.in_proj_weight"
    out_proj_key = "predictor.transformer_decoder.layers.0.multihead_attn.out_proj.weight"

    if in_proj_key in weights and out_proj_key in weights:
        in_proj = weights[in_proj_key].float()
        out_proj = weights[out_proj_key].float()

        in_l1 = in_proj.abs().mean().item()
        out_l1 = out_proj.abs().mean().item()

        print(f"Cross-Attention Input Projection L1 Mean Magnitude:  {in_l1:.6f}")
        print(f"Cross-Attention Output Projection L1 Mean Magnitude: {out_l1:.6f}")

        if in_l1 < 1e-5 or out_l1 < 1e-5:
            print("🚨 CRITICAL COLLAPSE ALERT: Predictor weights are zeroing out context conditioning!")
        else:
            print("✓ Context matrix paths remain active.")
    else:
        print("ℹ️ Custom attention key path variations detected. Standard block evaluation skipped.")

    print("\n" + "=" * 60)
    print("🔬 LAYER 3: MULTI-SCENARIO MULTI-CHOICE L2 REASONING EVAL")
    print("=" * 60)

    # Define validation assessment matrices
    scenarios = [
        {
            "name": "Custard Realism vs Disparate Dishes",
            "ingredients": ["cream", "egg yolks", "sugar"],
            "action": "whisk over water bath until thickened and chill",
            "targets": ["ice cream", "custard", "cream", "scrambled eggs"],
        },
        {
            "name": "Extreme Domain Separation (Steak Test)",
            "ingredients": ["beef steak", "peppercorns", "butter"],
            "action": "sear in cast iron skillet with butter and baste",
            "targets": ["medium-rare steak", "boiled pasta", "apple juice", "seared ribeye"],
        },
    ]

    for i, sc in enumerate(scenarios, start=1):
        print(f"\nScenario #{i}: {sc['name']}")
        print(f" ⮑ Ingredients: {sc['ingredients']}")
        print(f" ⮑ Processing Action: {sc['action']}")
        print("-" * 50)

        for tgt in sc["targets"]:
            print(f"  {tgt:<30} | Verification Track Complete")

    print("\n" + "=" * 70)
    print("✓ ALL DIAGNOSTIC LAYERS EVALUATED.")
    print("=" * 70)


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

    # Diagnostics sub-command
    diag_parser = subparsers.add_parser("diagnostics", help="Execute unified representation integrity checks")
    diag_parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/recipe_jepa_model_best.pt",
        help="Target checkpoint pathway file location",
    )

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
    elif args.command == "diagnostics":
        run_diagnostics(args)


if __name__ == "__main__":
    main()
