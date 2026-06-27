import argparse

import torch

from jepa_cook.src.config import load_config  # deptry: ignore
from jepa_cook.src.diagnostics import run_system_diagnostics  # deptry: ignore
from jepa_cook.src.inference import run_local_inference  # deptry: ignore
from jepa_cook.src.train import handle_train  # deptry: ignore


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
    diag_parser.add_argument("--checkpoint", type=str, default="checkpoints/recipe_jepa_model_best.pt")

    args = parser.parse_args()
    config = load_config(args.config)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using runtime device: {device}")

    # Pure Routing Logic
    if args.command == "train":
        handle_train(config, device)
    elif args.command == "inference":
        run_local_inference(args, config)
    elif args.command == "diagnostics":
        run_system_diagnostics(args.checkpoint, device)


if __name__ == "__main__":
    main()
