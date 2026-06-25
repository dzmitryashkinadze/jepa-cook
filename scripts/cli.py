import argparse
import ast
import copy
import math
import os

import polars as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# =====================================================================
# 1. DATASETS & MODELS
# =====================================================================


class PreTokenizedActionDataset(Dataset):
    def __init__(self, dataset_path: str, max_len: int = 128):
        self.df = pl.read_parquet(dataset_path)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def _parse_and_pad(self, token_str: str) -> torch.Tensor:
        tokens = [int(t) for t in token_str.split()] if token_str else []
        if len(tokens) < self.max_len:
            tokens = tokens + [0] * (self.max_len - len(tokens))
        else:
            tokens = tokens[: self.max_len]
        return torch.tensor(tokens, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)
        return (
            self._parse_and_pad(row["x_tokens"]),
            self._parse_and_pad(row["a_tokens"]),
            self._parse_and_pad(row["y_tokens"]),
        )


class TransformerPredictor(nn.Module):
    def __init__(self, latent_dim: int = 256, nhead: int = 8, num_layers: int = 2):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=latent_dim * 4,
            dropout=0.1,
            activation=nn.functional.silu,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, z_t: torch.Tensor, u_seq: torch.Tensor, a_mask: torch.Tensor = None) -> torch.Tensor:
        tgt = z_t.unsqueeze(1)
        mem_mask = (a_mask == 0) if a_mask is not None else None
        out = self.transformer_decoder(tgt=tgt, memory=u_seq, memory_key_padding_mask=mem_mask)
        return out.squeeze(1)


class RecipeJEPA(nn.Module):
    def __init__(self, vocab_size: int = 30522, embed_dim: int = 384, latent_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.target_embedding = copy.deepcopy(self.embedding)
        for param in self.target_embedding.parameters():
            param.requires_grad = False

        self.state_encoder = nn.Sequential(
            nn.Linear(embed_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim)
        )

        self.target_encoder = copy.deepcopy(self.state_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.action_sequence_encoder = nn.Sequential(
            nn.Linear(embed_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU()
        )

        self.predictor = TransformerPredictor(latent_dim)
        self.delta_norm = nn.LayerNorm(latent_dim)
        self.action_gate = nn.Parameter(torch.tensor([0.1]))
        self.prediction_norm = nn.LayerNorm(latent_dim)

    def _pool_active_tokens(self, tokens: torch.Tensor, embedding_layer: nn.Embedding) -> torch.Tensor:
        mask = (tokens != 0).float()
        mask_counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        embeddings = embedding_layer(tokens)
        masked_embeddings = embeddings * mask.unsqueeze(-1)
        return masked_embeddings.sum(dim=1) / mask_counts

    def encode_state(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self._pool_active_tokens(tokens, self.embedding)
        return self.state_encoder(x)

    def encode_target(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self._pool_active_tokens(tokens, self.target_embedding)
            return self.target_encoder(x)

    def forward(self, x_tokens: torch.Tensor, a_tokens: torch.Tensor) -> torch.Tensor:
        z_t = self.encode_state(x_tokens)
        a_mask = (a_tokens != 0).float()
        a_embed = self.embedding(a_tokens)
        u_seq = self.action_sequence_encoder(a_embed)
        latent_delta = self.predictor(z_t, u_seq, a_mask)
        normalized_delta = self.delta_norm(latent_delta)
        z_next_pred = self.prediction_norm(z_t + self.action_gate * normalized_delta)
        return z_next_pred

    @torch.no_grad()
    def update_target_ema(self, momentum: float = 0.99):
        for target_param, online_param in zip(self.target_encoder.parameters(), self.state_encoder.parameters()):
            target_param.data.mul_(momentum).add_(online_param.data, alpha=1.0 - momentum)
        for target_param, online_param in zip(self.target_embedding.parameters(), self.embedding.parameters()):
            target_param.data.mul_(momentum).add_(online_param.data, alpha=1.0 - momentum)


# =====================================================================
# 2. LOSS FUNCTION
# =====================================================================


def vicreg_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    sim_coeff: float = 25.0,
    var_coeff: float = 25.0,
    cov_coeff: float = 5.0,
    gamma: float = 1.0,
    eps: float = 1e-4,
):
    batch_size, num_features = z_a.shape
    sim_loss = nn.functional.mse_loss(z_a, z_b)

    std_a = torch.sqrt(z_a.var(dim=0) + eps)
    std_b = torch.sqrt(z_b.var(dim=0) + eps)
    var_loss_a = torch.mean(nn.functional.relu(gamma - std_a))
    var_loss_b = torch.mean(nn.functional.relu(gamma - std_b))
    std_loss = var_loss_a + var_loss_b

    z_a_zero_mean = z_a - z_a.mean(dim=0)
    z_b_zero_mean = z_b - z_b.mean(dim=0)
    cov_a = (z_a_zero_mean.T @ z_a_zero_mean) / (batch_size - 1)
    cov_b = (z_b_zero_mean.T @ z_b_zero_mean) / (batch_size - 1)
    cov_loss_a = cov_a.pow(2).sum() - cov_a.diagonal().pow(2).sum()
    cov_loss_b = cov_b.pow(2).sum() - cov_b.diagonal().pow(2).sum()
    cov_loss = (cov_loss_a + cov_loss_b) / num_features

    total_loss = (sim_coeff * sim_loss) + (var_coeff * std_loss) + (cov_coeff * cov_loss)
    return total_loss, sim_loss, std_loss, cov_loss


# =====================================================================
# 3. SUBCOMMAND LOGIC (TRAIN & INFERENCE)
# =====================================================================


def handle_train(args, device):
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = PreTokenizedActionDataset(args.dataset_path, max_len=128)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = RecipeJEPA(vocab_size=30522, embed_dim=384, latent_dim=256).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    num_warmup_steps = 3 * len(dataloader)
    total_steps = args.epochs * len(dataloader)

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, total_steps - num_warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    print("Beginning Action-Conditioned JEPA Training Baseline...")
    model.train()

    for epoch in range(args.epochs):
        loss_monitor = [0.0, 0.0, 0.0]
        for batch_idx, (x_tokens, a_tokens, y_tokens) in enumerate(dataloader):
            optimizer.zero_grad()
            x_tokens, a_tokens, y_tokens = x_tokens.to(device), a_tokens.to(device), y_tokens.to(device)

            pred_embed = model(x_tokens, a_tokens)
            with torch.no_grad():
                true_embed = model.encode_target(y_tokens).detach()

            total_loss, sim_loss, std_loss, cov_loss = vicreg_loss(pred_embed, true_embed)
            sim_val, std_val, cov_val = sim_loss.item(), std_loss.item(), cov_loss.item()

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            model.update_target_ema(momentum=0.999)
            loss_monitor[0] += sim_val
            loss_monitor[1] += std_val
            loss_monitor[2] += cov_val

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(
                    f"Epoch {epoch + 1:02d} |",
                    f"Batch {batch_idx + 1}/{len(dataloader)} |",
                    f"SIM: {sim_val:.4f} |",
                    f"STD: {std_val:.4f} |",
                    f"COV: {cov_val:.4f}",
                )

            if device.type == "mps":
                torch.mps.empty_cache()

        epoch_sim = loss_monitor[0] / len(dataloader)
        epoch_std = loss_monitor[1] / len(dataloader)
        epoch_cov = loss_monitor[2] / len(dataloader)
        print(
            f"--- Epoch {epoch + 1} End |",
            f"Mean SIM: {epoch_sim:.4f} |",
            f"Mean STD: {epoch_std:.4f} |",
            f"Mean COV: {epoch_cov:.4f} ---",
        )
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"recipe_jepa_model_{epoch}.pt"))

    torch.save(model.state_dict(), os.path.join(args.output_dir, "recipe_jepa_model_final.pt"))
    print("Training run complete.")


def handle_inference(args, device):
    try:
        targets_list = ast.literal_eval(args.targets)
    except Exception:
        print("[!] Format error parsing target list strings.")
        return

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    model = RecipeJEPA(vocab_size=30522, embed_dim=384, latent_dim=256)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint if "state_dict" not in checkpoint else checkpoint["state_dict"])
    model.to(device).eval()

    with torch.no_grad():
        x_enc = tokenizer(args.ingredients, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        x_tokens = x_enc["input_ids"].to(device)

        a_enc = tokenizer(args.action, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        a_tokens = a_enc["input_ids"].to(device)

        pred_z_next = model(x_tokens, a_tokens)
        pred_z_next = nn.functional.normalize(pred_z_next, p=2, dim=-1)

        y_enc = tokenizer(targets_list, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        y_tokens_batch = y_enc["input_ids"].to(device)

        true_z_next_batch = model.encode_state(y_tokens_batch)
        true_z_next_batch = nn.functional.normalize(true_z_next_batch, p=2, dim=-1)

        rankings = []
        for idx, target_str in enumerate(targets_list):
            true_z_next = true_z_next_batch[idx].unsqueeze(0)
            distance = nn.functional.mse_loss(pred_z_next, true_z_next).item()
            rankings.append((target_str, distance))

        rankings.sort(key=lambda x: x[1])

    print("\n" + "=" * 60)
    print(f" INITIAL STATE (s_t): {args.ingredients}")
    print(f" ACTION (a_t):        {args.action}")
    print("=" * 60)
    for target, score in rankings:
        print(f"{target:<30} | MSE: {score:.6f}")
    print("=" * 60)


# =====================================================================
# 4. ENTRYPOINT MAIN
# =====================================================================


def main():
    parser = argparse.ArgumentParser(description="Recipe JEPA Unified CLI Workflow")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Workflow Mode")

    # Train Subcommand
    train_parser = subparsers.add_parser("train", help="Run model training loop")
    train_parser.add_argument("--dataset_path", type=str, default="data/recipe_sampled.parquet")
    train_parser.add_argument("--output_dir", type=str, default="checkpoints")
    train_parser.add_argument("--batch_size", type=int, default=64)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--lr", type=float, default=5e-4)

    # Inference Subcommand
    infer_parser = subparsers.add_parser("inference", help="Run evaluation/prediction inference")
    infer_parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    infer_parser.add_argument("--ingredients", type=str, required=True, help="Starting context string")
    infer_parser.add_argument("--action", type=str, required=True, help="Action execution string")
    infer_parser.add_argument(
        "--targets", type=str, required=True, help="String list representation of target outcomes"
    )

    args = parser.parse_args()

    # Hardware selector logic
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using runtime device: {device}")

    if args.command == "train":
        handle_train(args, device)
    elif args.command == "inference":
        handle_inference(args, device)


if __name__ == "__main__":
    main()
