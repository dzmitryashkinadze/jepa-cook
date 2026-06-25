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
from torch.utils.tensorboard import SummaryWriter
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

    def forward(self, z_t: torch.Tensor, u_seq: torch.Tensor, a_mask: torch.Tensor | None = None) -> torch.Tensor:
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

    def forward(self, x_tokens: torch.Tensor, a_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_t = self.encode_state(x_tokens)
        a_mask = (a_tokens != 0).float()
        a_embed = self.embedding(a_tokens)
        u_seq = self.action_sequence_encoder(a_embed)
        latent_delta = self.predictor(z_t, u_seq, a_mask)
        normalized_delta = self.delta_norm(latent_delta)
        z_next_pred = self.prediction_norm(z_t + self.action_gate * normalized_delta)
        return z_next_pred, z_t
        # return nn.functional.normalize(z_next_pred, p=2, dim=-1)

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
    z_pred: torch.Tensor,
    z_true: torch.Tensor,
    z_t: torch.Tensor,  # Raw input embedding: encode_state(x_tokens)
    sim_coeff: float = 25.0,
    var_coeff: float = 25.0,
    cov_coeff: float = 5.0,
    repel_coeff: float = 15.0,  # Strength of the shortcut penalty
    threshold: float = 0.25,  # Distance threshold (adjust based on latent space scaling)
    gamma: float = 1.0,
    eps: float = 1e-4,
):
    batch_size, num_features = z_pred.shape
    sim_loss = nn.functional.mse_loss(z_pred, z_true)

    # 2. List Repulsion Penalty (Hinge loss pushing prediction away from raw state)
    # Measures how far the predictor moved the representation from the ingredient base
    shortcut_distance = torch.mean((z_pred - z_t) ** 2, dim=-1)
    repel_loss = torch.mean(nn.functional.relu(shortcut_distance - threshold))

    std_a = torch.sqrt(z_pred.var(dim=0) + eps)
    std_b = torch.sqrt(z_true.var(dim=0) + eps)
    var_loss_a = torch.mean(nn.functional.relu(gamma - std_a))
    var_loss_b = torch.mean(nn.functional.relu(gamma - std_b))
    std_loss = var_loss_a + var_loss_b

    z_pred_zero_mean = z_pred - z_pred.mean(dim=0)
    z_true_zero_mean = z_true - z_true.mean(dim=0)
    cov_a = (z_pred_zero_mean.T @ z_pred_zero_mean) / (batch_size - 1)
    cov_b = (z_true_zero_mean.T @ z_true_zero_mean) / (batch_size - 1)
    cov_loss_a = cov_a.pow(2).sum() - cov_a.diagonal().pow(2).sum()
    cov_loss_b = cov_b.pow(2).sum() - cov_b.diagonal().pow(2).sum()
    cov_loss = (cov_loss_a + cov_loss_b) / num_features

    total_loss = (sim_coeff * sim_loss) + (var_coeff * std_loss) + (cov_coeff * cov_loss) + (repel_coeff * repel_loss)
    return total_loss, sim_loss, std_loss, cov_loss, repel_loss


# =====================================================================
# 3. TRAINER MODULE (WITH TRAIN/VAL + TENSORBOARD + EARLY STOPPING)
# =====================================================================


class JEPATrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        output_dir,
        log_dir,
        patience=5,
        min_delta=0.001,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir

        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

        # Early Stopping Tracking Parameters
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_sim = float("inf")
        self.patience_counter = 0

    def train(self, epochs: int):
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Beginning JEPA Training Loop (Early Stopping: patience={self.patience}, min_delta={self.min_delta})...")

        for epoch in range(epochs):
            # --- Training Pass ---
            self.model.train()
            train_loss_monitor = [0.0, 0.0, 0.0, 0.0]

            for batch_idx, (x_tokens, a_tokens, y_tokens) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                x_tokens, a_tokens, y_tokens = (
                    x_tokens.to(self.device),
                    a_tokens.to(self.device),
                    y_tokens.to(self.device),
                )

                pred_embed, z_t = self.model(x_tokens, a_tokens)
                with torch.no_grad():
                    true_embed = self.model.encode_target(y_tokens).detach()

                total_loss, sim_loss, std_loss, cov_loss, repel_loss = vicreg_loss(
                    z_pred=pred_embed, z_true=true_embed, z_t=z_t
                )
                sim_train, std_train, cov_train, rep_train = (
                    sim_loss.item(),
                    std_loss.item(),
                    cov_loss.item(),
                    repel_loss.item(),
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()

                self.model.update_target_ema(momentum=0.99)

                self.writer.add_scalar("Loss/Train_Total_Step", total_loss.item(), self.global_step)
                self.writer.add_scalar("Loss/Train_SIM_Step", sim_train, self.global_step)
                self.writer.add_scalar("Loss/Train_STD_Step", std_train, self.global_step)
                self.writer.add_scalar("Loss/Train_COV_Step", cov_train, self.global_step)
                self.writer.add_scalar("Loss/Train_Repel_Step", rep_train, self.global_step)
                self.writer.add_scalar(
                    "Hyperparameters/Learning_Rate", self.scheduler.get_last_lr()[0], self.global_step
                )

                train_loss_monitor[0] += sim_train
                train_loss_monitor[1] += std_train
                train_loss_monitor[2] += cov_train
                train_loss_monitor[3] += rep_train
                self.global_step += 1

                if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                    print(
                        f"Epoch {epoch + 1:02d} |",
                        f"Train Batch {batch_idx + 1}/{len(self.train_loader)} |",
                        f"SIM: {sim_train:.4f} |",
                        f"STD: {std_train:.4f} |",
                        f"COV: {cov_train:.4f} |",
                        f"REP: {rep_train:.4f}",
                    )

                if self.device.type == "mps":
                    torch.mps.empty_cache()

            # Train Epoch Metrics Evaluation
            epoch_train_sim = train_loss_monitor[0] / len(self.train_loader)
            epoch_train_std = train_loss_monitor[1] / len(self.train_loader)
            epoch_train_cov = train_loss_monitor[2] / len(self.train_loader)
            epoch_train_rep = train_loss_monitor[3] / len(self.train_loader)

            # --- Validation Pass ---
            epoch_val_sim, epoch_val_std, epoch_val_cov, epoch_val_rep = self.validate()

            # TensorBoard Logging
            self.writer.add_scalars("Epoch/Invariance_SIM", {"train": epoch_train_sim, "val": epoch_val_sim}, epoch)
            self.writer.add_scalars("Epoch/Variance_STD", {"train": epoch_train_std, "val": epoch_val_std}, epoch)
            self.writer.add_scalars("Epoch/Covariance_COV", {"train": epoch_train_cov, "val": epoch_val_cov}, epoch)
            self.writer.add_scalars("Epoch/Repulsion", {"train": epoch_train_rep, "val": epoch_val_rep}, epoch)

            print(f"--- Epoch {epoch + 1} Summary ---")
            print(
                f" [TRAIN] Mean SIM: {epoch_train_sim:.4f} |",
                f"Mean STD: {epoch_train_std:.4f} |",
                f"Mean COV: {epoch_train_cov:.4f} |",
                f"Mean REP: {epoch_train_rep:.4f}",
            )
            print(
                f" [VAL]   Mean SIM: {epoch_val_sim:.4f} |",
                f"Mean STD: {epoch_val_std:.4f} |",
                f"Mean COV: {epoch_val_cov:.4f} |",
                f"Mean REP: {epoch_val_rep:.4f}",
            )

            # --- Early Stopping Evaluation ---
            # Using SIM (Invariance / Prediction accuracy) as our core target metric
            if epoch_val_sim < (self.best_val_sim - self.min_delta):
                print(
                    f" [✓] Val SIM improved from {self.best_val_sim:.4f} to {epoch_val_sim:.4f}.",
                    "Model checkpoint recipe_jepa_model_best.pt written.",
                )
                self.best_val_sim = epoch_val_sim
                self.patience_counter = 0
                # Always preserve the ultimate generalized configuration state
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, "recipe_jepa_model_best.pt"))
            else:
                self.patience_counter += 1
                print(
                    " [!] No significant validation improvement.",
                    f"Early stopping counter: {self.patience_counter}/{self.patience}",
                )

            # Optional regular checkpoint generation
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"recipe_jepa_model_epoch_{epoch}.pt"))

            if self.patience_counter >= self.patience:
                print(f"\n[🛑] Early stopping triggered! Halting at epoch {epoch + 1} to prevent further overfitting.")
                break

        self.writer.close()
        print(f"Training exit complete. Best Validation SIM Score achieved: {self.best_val_sim:.4f}")

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        val_loss_monitor = [0.0, 0.0, 0.0, 0.0]

        for x_tokens, a_tokens, y_tokens in self.val_loader:
            x_tokens, a_tokens, y_tokens = x_tokens.to(self.device), a_tokens.to(self.device), y_tokens.to(self.device)
            pred_embed, z_t = self.model(x_tokens, a_tokens)
            true_embed = self.model.encode_target(y_tokens).detach()

            _, sim_loss, std_loss, cov_loss, rep_loss = vicreg_loss(pred_embed, true_embed, z_t)

            val_loss_monitor[0] += sim_loss.item()
            val_loss_monitor[1] += std_loss.item()
            val_loss_monitor[2] += cov_loss.item()
            val_loss_monitor[3] += rep_loss.item()

        val_sim = val_loss_monitor[0] / len(self.val_loader)
        val_std = val_loss_monitor[1] / len(self.val_loader)
        val_cov = val_loss_monitor[2] / len(self.val_loader)
        val_rep = val_loss_monitor[3] / len(self.val_loader)
        return val_sim, val_std, val_cov, val_rep


# =====================================================================
# 4. SUBCOMMAND INTERFACES
# =====================================================================


def handle_train(args, device):
    train_dataset = PreTokenizedActionDataset(args.train_dataset, max_len=128)
    val_dataset = PreTokenizedActionDataset(args.val_dataset, max_len=128)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = RecipeJEPA(vocab_size=30522, embed_dim=384, latent_dim=256).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    num_warmup_steps = 3 * len(train_loader)
    total_steps = args.epochs * len(train_loader)

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, total_steps - num_warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    trainer = JEPATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        patience=args.patience,
        min_delta=args.min_delta,
    )
    trainer.train(epochs=args.epochs)


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

        pred_z_next, _ = model(x_tokens, a_tokens)
        # pred_z_next = nn.functional.normalize(pred_z_next, p=2, dim=-1)

        y_enc = tokenizer(targets_list, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        y_tokens_batch = y_enc["input_ids"].to(device)

        true_z_next_batch = model.encode_state(y_tokens_batch)
        # true_z_next_batch = nn.functional.normalize(true_z_next_batch, p=2, dim=-1)

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
# 5. ENTRYPOINT MAIN
# =====================================================================


def main():
    parser = argparse.ArgumentParser(description="Recipe JEPA Unified CLI Workflow")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Workflow Mode")

    # Train Subcommand
    train_parser = subparsers.add_parser("train", help="Run model training loop with validation checks")
    train_parser.add_argument("--train_dataset", type=str, default="data/recipe_train.parquet")
    train_parser.add_argument("--val_dataset", type=str, default="data/recipe_val.parquet")
    train_parser.add_argument("--output_dir", type=str, default="checkpoints")
    train_parser.add_argument("--log_dir", type=str, default="runs/recipe_jepa_experiment")
    train_parser.add_argument("--batch_size", type=int, default=64)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    # New CLI Early Stopping Arguments
    train_parser.add_argument("--patience", type=int, default=3, help="Number of epochs to wait before stopping")
    train_parser.add_argument(
        "--min_delta", type=float, default=1e-4, help="Minimum improvement required to reset patience"
    )

    # Inference Subcommand
    infer_parser = subparsers.add_parser("inference", help="Run evaluation/prediction inference")
    infer_parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    infer_parser.add_argument("--ingredients", type=str, required=True, help="Starting context string")
    infer_parser.add_argument("--action", type=str, required=True, help="Action execution string")
    infer_parser.add_argument(
        "--targets", type=str, required=True, help="String list representation of target outcomes"
    )

    args = parser.parse_args()

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
