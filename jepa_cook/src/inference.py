import ast
import json

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from jepa_cook.src.config import RecipeJEPAConfig  # deptry: ignore
from jepa_cook.src.dataset import pad_nested_sequences  # deptry: ignore
from jepa_cook.src.models import RecipeJEPA  # deptry: ignore


def run_local_inference(args, config: dict) -> None:
    """Runs context vectors down distance spaces to score structural predictions locally."""
    model_cfg = config["model"]
    infer_cfg = config["inference"]

    try:
        targets_list: list[str] = ast.literal_eval(args.targets)
    except Exception:
        print("[!] Format error parsing target list strings.")
        return

    tokenizer = AutoTokenizer.from_pretrained(infer_cfg["tokenizer_name"])

    hf_config = RecipeJEPAConfig(
        vocab_size=model_cfg["vocab_size"],
        embed_dim=model_cfg["embed_dim"],
        latent_dim=model_cfg["latent_dim"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
    )
    model = RecipeJEPA(config=hf_config)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint if "state_dict" not in checkpoint else checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():

        def tokenize_to_3d_input(text_input: str) -> torch.Tensor:
            try:
                lst = json.loads(text_input) if "[" in text_input else [text_input]
            except Exception:
                lst = [text_input]
            tokens_list = [torch.tensor(tokenizer(item, add_special_tokens=False)["input_ids"]) for item in lst]
            return pad_nested_sequences([tokens_list], max_len=model_cfg["max_len"])

        x_tokens = tokenize_to_3d_input(args.ingredients)
        a_tokens = tokenize_to_3d_input(args.action)
        pred_embed, _, _, _ = model(x_tokens, a_tokens)
        pred_embed_norm = nn.functional.normalize(pred_embed, p=2, dim=-1)

        results = []
        for target_str in targets_list:
            target_tokens = tokenizer(target_str, add_special_tokens=False, return_tensors="pt")["input_ids"]
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
